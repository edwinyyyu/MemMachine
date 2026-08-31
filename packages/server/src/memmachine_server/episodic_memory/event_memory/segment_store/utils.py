"""Shared utilities for segment store implementations."""

import re
import secrets

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


def new_incarnation() -> str:
    """A fresh incarnation token for a partition registry row."""
    return secrets.token_hex(4)


def physical_partition_key(partition_key: str, incarnation: str) -> str:
    """The key data rows are stored under: `<logical_key>@<incarnation>`.

    `@` cannot appear in a logical key, so physical keys cannot collide
    with logical keys or with other incarnations of the same key.
    """
    return f"{partition_key}@{incarnation}"
