import os
import time

import argilla as rg
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    CombineColumns,
    LoadHubDataset,
    StepInput,
    StepOutput,
    step,
)
from distilabel.steps.tasks import TextGeneration, UltraFeedback
from dotenv import load_dotenv
from huggingface_hub import login

from custom_preference_to_argilla import CustomPreferenceToArgilla
from utils import is_empty_template, parse_markdown, try_load_text

load_dotenv()

NUM_EXAMPLES = None  # Number of examples to load from the dataset (None to load all)
MAX_NEW_TOKENS = 250  # Maximum number of new tokens to generate
INPUT_BATCH_SIZE = 2  # Input batch size for the model via the Inference Endpoints API
MIN_CARD_LENGTH = 2500  # Minimum length of the card to be used for generating summaries
# Argilla Configuration
ARGILLA_SPACE_URL = "https://dibt-demo-argilla-space.hf.space"  # Argilla Space URL
ARGILLA_DATASET_NAME = (
    "dataset-preferences-llm-course-full-dataset"  # Argilla dataset name
)
ARGILLA_WORKSPACE_NAME = "admin"  # Argilla workspace name
# Dataset Configuration
OUTPUT_DATASET_HUB_ID = f"davanstrien/{ARGILLA_DATASET_NAME}"  # Output dataset hub ID
SPLIT = "train"

HUGGINGFACE_TOKEN = os.getenv("HF_API_KEY")
assert (
    HUGGINGFACE_TOKEN is not None
), "Please set the HF_API_KEY environment variable or authenticate with the Hugging Face CLI using `huggingface-cli login`"
login(token=HUGGINGFACE_TOKEN)
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")

# Check if the API key is set
assert (
    ARGILLA_API_KEY is not None
), "Please set the ARGILLA_API_KEY environment variable or pass it as a parameter"

# Initialize the Argilla client
rg.init(
    api_url=ARGILLA_SPACE_URL, api_key=ARGILLA_API_KEY, workspace=ARGILLA_WORKSPACE_NAME
)


def remove_existing_dataset(ARGILLA_DATASET_NAME: str):
    """Remove an existing dataset from Argilla. This is useful when re-running the pipeline."""
    try:
        argilla_ds = rg.FeedbackDataset.from_argilla(ARGILLA_DATASET_NAME)
        argilla_ds.delete()
    except ValueError as e:
        print(e)


@step(
    inputs=["card", "datasetId"],
    outputs=["card", "datasetId"],
)
def filter_cards(inputs: StepInput) -> StepOutput:
    selected_rows = []
    for input in inputs:
        if "open-llm-leaderboard" in input["datasetId"]:
            continue
        if "not-for-all-audiences" in input["card"]:
            continue
        card = try_load_text(input["card"])
        if card is None:
            continue
        if is_empty_template(card):
            continue
        try:
            card = parse_markdown(card)
        except Exception as e:
            print(e)
            continue
        if card.startswith("Dataset Card for [Dataset Name]"):
            continue
        if len(card) < MIN_CARD_LENGTH:
            continue
        input["card"] = card
        selected_rows.append(input)
    yield selected_rows


@step(
    inputs=["ratings"],
    outputs=["ratings"],
)
def filter_ratings(inputs: StepInput) -> StepOutput:
    yield [input for input in inputs if None not in input["ratings"]]


SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant`. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
Your role is to write short tl;dr descriptions of datasets based on existing dataset cards"""


def format_prompt(card: str) -> str:
    return f"""
<instructions>
Write a tl;dr summary of a dataset based on the dataset card. Focus on the most critical aspects of the dataset.

The summary should aim to concisely describe the dataset.
</instructions>

<card>

{card[:6000]}

</card>

<instructions>
If the card provides the necessary information, say what the dataset can be used for.
You do not need to mention that the dataset is hosted or available on the Hugging Face Hub.
Do not mention the license of the dataset.
Do not mention the number of examples in the training or test split.
Only mention size if there is extensive discussion of the scale of the dataset in the dataset card.
Do not speculate on anything not explicitly mentioned in the dataset card.
In general avoid references to the quality of the dataset i.e. don't use phrases like 'a high-quality dataset' in the summary.
</instructions>

<One sentence summary>"""


@step(inputs=["card"], outputs=["instruction"])
def format_input_card(inputs: StepInput) -> StepOutput:
    for input in inputs:
        input["instruction"] = format_prompt(input["card"])
        input["system_prompt"] = SYSTEM_PROMPT
    yield inputs


with Pipeline(name="dataset-summary-preferences") as pipeline:
    llama3 = InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
        model_display_name="meta-llama/Meta-Llama-3-70B-Instruct",
        api_key=HUGGINGFACE_TOKEN,
    )
    mixtral = InferenceEndpointsLLM(
        endpoint_name="nous-hermes-2-mixtral-8x7b-d-yeg",
        # model_id="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        tokenizer_id="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        model_display_name="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        api_key=HUGGINGFACE_TOKEN,
    )
    # Load the dataset from the Hugging Face Hub
    load_hub_dataset = LoadHubDataset(
        name="load_dataset",
    )
    # Define the steps
    # filter the cards so we only use the ones that are not empty templates and have a minimum length
    card_filter = filter_cards(name="card_filter")
    # format the input prompt
    formatted_input = format_input_card(name="format_input_card")
    llama_summary = TextGeneration(
        name="llama_summary",
        llm=llama3,
        input_batch_size=INPUT_BATCH_SIZE,
        output_mappings={"model_name": "generation_model"},
        num_generations=1,
        use_system_prompt=True,
    )
    mixtral_summary = TextGeneration(
        name="mixtral_summary",
        llm=mixtral,
        input_batch_size=16,
        output_mappings={"model_name": "generation_model"},
        num_generations=1,
        use_system_prompt=True,
    )

    combine_columns = CombineColumns(
        name="combine_columns",
        columns=["generation", "generation_model"],
        output_columns=["generations", "generation_models"],
    )
    ultrafeedback = UltraFeedback(
        name="ultrafeedback", aspect="instruction-following", llm=llama3
    )
    remove_bad_ratings = filter_ratings(name="remove_bad_ratings")

    to_argilla = CustomPreferenceToArgilla(
        name="to_argilla",
        dataset_name=ARGILLA_DATASET_NAME,
        api_url=ARGILLA_SPACE_URL,
        api_key=ARGILLA_API_KEY,
        dataset_workspace=ARGILLA_WORKSPACE_NAME,
        num_generations=2,
    )
    # Define the pipeline
    load_hub_dataset >> card_filter
    card_filter >> formatted_input
    formatted_input >> llama_summary
    formatted_input >> mixtral_summary
    llama_summary >> combine_columns
    mixtral_summary >> combine_columns
    combine_columns >> ultrafeedback
    ultrafeedback >> remove_bad_ratings
    remove_bad_ratings >> to_argilla


if __name__ == "__main__":
    # time the pipeline
    start_time = time.time()
    if ARGILLA_DATASET_NAME:
        print(f"Removing existing dataset: {ARGILLA_DATASET_NAME}")
        remove_existing_dataset(ARGILLA_DATASET_NAME)
    # run the pipeline
    load_dataset_params = {
        "load_dataset": {
            "repo_id": "davanstrien/popular-cards",
            "split": SPLIT,
        },
    }
    if NUM_EXAMPLES is not None:
        load_dataset_params["load_dataset"]["num_examples"] = NUM_EXAMPLES
    dataset = pipeline.run(
        parameters={
            **load_dataset_params,
            "llama_summary": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "do_sample": True,
                        "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
                    }
                }
            },
            "mixtral_summary": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "do_sample": True,
                    },
                },
            },
            "ultrafeedback": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "do_sample": True,
                    }
                }
            },
            "to_argilla": {
                "dataset_name": ARGILLA_DATASET_NAME,
                "input_batch_size": 10,
            },
        }
    )
    dataset.push_to_hub(OUTPUT_DATASET_HUB_ID)
    # end time
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
