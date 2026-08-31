"""Shared utilities for segment store implementations."""

import re

_PARTITION_KEY_RE = re.compile(r"^[a-z0-9_]+$")
_PARTITION_KEY_MAX_BYTES = 32


def validate_partition_key(partition_key: str) -> None:
    """Raise ValueError unless the key is valid (a-z0-9_, max 32 bytes)."""
    if not _PARTITION_KEY_RE.match(partition_key):
        raise ValueError(
            f"Partition key {partition_key!r} contains invalid characters. "
            "Only lowercase alphanumeric and underscores are allowed."
        )
    if len(partition_key.encode()) > _PARTITION_KEY_MAX_BYTES:
        raise ValueError(
            f"Partition key {partition_key!r} is too long "
            f"({len(partition_key.encode())} bytes). "
            f"Maximum is {_PARTITION_KEY_MAX_BYTES}."
        )
