"""Shared utilities for segment store implementations."""

import re

_PARTITION_KEY_RE = re.compile(r"^[a-z0-9_]+$")
_PARTITION_KEY_MAX_BYTES = 32


def validate_partition_key(partition_key: str) -> bool:
    """Return True if the key matches `[a-z0-9_]+` and is at most 32 bytes."""
    return (
        bool(_PARTITION_KEY_RE.match(partition_key))
        and len(partition_key.encode()) <= _PARTITION_KEY_MAX_BYTES
    )
