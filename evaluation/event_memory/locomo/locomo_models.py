"""LoCoMo dataset helpers."""

import json
from datetime import UTC, datetime


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    """Parse a LoCoMo session timestamp (e.g. '1:56 pm on 8 May, 2023')."""
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )


def load_locomo_dataset(file_path: str) -> list[dict]:
    """Load the LoCoMo10 dataset (a list of conversation/qa items)."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected top-level list in {file_path}, got {type(data)}")
    return data


def attachment_suffix(message: dict) -> str:
    """Return the formatted attachment suffix for an image, if any.

    Matches the reference Mem0/MemMachine LoCoMo loader: appends
    "[Attached <blip_caption>: <query>]" when both are present, or one of the
    single-field variants when only one is.
    """
    blip_caption = message.get("blip_caption")
    image_query = message.get("query")
    if blip_caption and image_query:
        return f" [Attached {blip_caption}: {image_query}]"
    if blip_caption:
        return f" [Attached {blip_caption}]"
    if image_query:
        return f" [Attached a photo: {image_query}]"
    return ""
