"""Shared utilities for segment store implementations."""

import re
from uuid import UUID, uuid4

_PARTITION_KEY_RE = re.compile(r"^[a-z0-9_]+$")
_PARTITION_KEY_MAX_BYTES = 32


def validate_partition_key(partition_key: str) -> None:
    """Raise ValueError unless the key matches `[a-z0-9_]+` and is at most 32 bytes."""
    if not _PARTITION_KEY_RE.match(partition_key):
        raise ValueError(
            f"Partition key {partition_key!r} contains invalid characters. "
            "Only lowercase alphanumeric and underscores are allowed."
        )
    key_length_bytes = len(partition_key.encode())
    if key_length_bytes > _PARTITION_KEY_MAX_BYTES:
        raise ValueError(
            f"Partition key {partition_key!r} is too long "
            f"({key_length_bytes} bytes). "
            f"Maximum is {_PARTITION_KEY_MAX_BYTES}."
        )


def new_incarnation() -> UUID:
    """A fresh incarnation identifier for a partition registry row.

    A random UUID so incarnations are globally unique across nodes
    without coordination; tenant moves between databases carry rows
    verbatim.
    """
    return uuid4()
