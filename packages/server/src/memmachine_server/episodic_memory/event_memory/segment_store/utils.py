"""Shared utilities for segment store implementations."""

import re

_PARTITION_KEY_RE = re.compile(r"^[a-z0-9_]+$")

PARTITION_KEY_MAX_BYTES = 32
"""Maximum partition key length in bytes."""


def validate_partition_key(partition_key: str) -> None:
    """Raise ValueError unless the key matches `[a-z0-9_]+` and is at most 32 bytes."""
    if not _PARTITION_KEY_RE.fullmatch(partition_key):
        raise ValueError(
            f"Partition key {partition_key!r} contains invalid characters. "
            "Only lowercase alphanumeric and underscores are allowed."
        )
    key_length_bytes = len(partition_key.encode())
    if key_length_bytes > PARTITION_KEY_MAX_BYTES:
        raise ValueError(
            f"Partition key {partition_key!r} is too long "
            f"({key_length_bytes} bytes). "
            f"Maximum is {PARTITION_KEY_MAX_BYTES}."
        )
