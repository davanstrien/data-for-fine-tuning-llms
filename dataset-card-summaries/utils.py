import logging
import re

from huggingface_hub import ModelCard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MORE_INFO_PATTERN = re.compile(r"\[More Information Needed\]\(https?://\S+\)")


def try_load_text(card) -> str | None:
    try:
        return ModelCard(card).text
    except Exception as e:
        logger.info(e)
        return None


def parse_markdown(markdown_text) -> str:
    """Parse markdown text and remove sections with placeholder text."""
    lines = markdown_text.split("\n")
    parsed_lines = []
    skip_section = False
    empty_section = True
    table_of_contents = False
    more_info_pattern = MORE_INFO_PATTERN

    for line in lines:
        if "Table of Contents" in line:
            table_of_contents = True
            continue
        if table_of_contents:
            if line.startswith("#"):
                table_of_contents = False
            else:
                continue
        if line.startswith("#"):
            if skip_section or empty_section:
                continue
            empty_section = True

        if skip_section:
            if line.startswith("#"):
                skip_section = False
            else:
                continue
        if more_info_pattern.match(line.strip()):
            skip_section = True
            empty_section = True
            continue
        if line.strip():
            empty_section = False
            parsed_lines.append(line)
    if skip_section or empty_section:
        while parsed_lines and parsed_lines[-1].startswith("#"):
            parsed_lines.pop()
    return "\n".join(parsed_lines)


def is_empty_template(text) -> bool:
    """Check if a card is an empty template."""
    placeholders = [r"\[More Information Needed\]", r"\[optional\]"]
    for placeholder in placeholders:
        text = re.sub(placeholder, "", text)
    text = text.strip()
    return not text
