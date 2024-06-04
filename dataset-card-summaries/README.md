# Synthetic data pipeline for generating dataset card summaries

![Pipeline for synthetic data generation](pipeline.png)

- [choose-model.py](./choose-model.py): Script to generate generations from different LLMs. The goal is to identify which models we might want to use for generating synthetic data.
- [generate-full-dataset.py](dataset-card-summaries/generate-full-dataset.py)

## Utils

- [custom_preference_to_argilla.py](./custom_preference_to_argilla.py): a custom step for Distilabel that gives us the option of adding our own summary text when generating the data
- [utils.py](./utils.py): utility functions for the pipeline that we might reuse/adapt for other parts of the fine-tuning process
