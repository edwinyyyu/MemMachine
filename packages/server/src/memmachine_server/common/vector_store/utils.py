"""Shared utilities for vector store implementations."""

import math
import re
from collections.abc import Mapping, Sequence

from memmachine_server.common.data_types import PropertyType, PropertyValue
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    filter_fields,
)

# A module, not its names: data_types imports this module, so its errors are
# not defined yet while this module loads.
from . import data_types

# Matched with fullmatch: `$` also matches before a trailing newline.
_IDENTIFIER_RE = re.compile(r"[a-z0-9_]+")
_IDENTIFIER_MAX_BYTES = 32
_PARTITION_KEY_MAX_BYTES = 255


def is_valid_identifier(value: str) -> bool:
    """Return True if value is a valid identifier (a-z0-9_, max 32 bytes)."""
    return (
        bool(_IDENTIFIER_RE.fullmatch(value))
        and len(value.encode()) <= _IDENTIFIER_MAX_BYTES
    )


def validate_partition_key(partition_key: str) -> None:
    """Raise ValueError unless the partition key is non-empty, at most `_PARTITION_KEY_MAX_BYTES` bytes, and free of NUL."""
    if (
        not partition_key
        or "\x00" in partition_key
        or len(partition_key.encode()) > _PARTITION_KEY_MAX_BYTES
    ):
        raise ValueError(
            f"Partition key {partition_key!r} must be non-empty, at most "
            f"{_PARTITION_KEY_MAX_BYTES} bytes, and free of NUL characters"
        )


def validate_declared_properties(
    properties: Mapping[str, PropertyValue],
    indexed_properties: Mapping[str, PropertyType],
) -> None:
    """
    Raise unless every property is declared and holds a value of its type.

    Called before anything is sent, so an undeclared key never reaches a
    backend, and a declared key never holds a value its column or index
    would have to coerce.
    """
    undeclared = properties.keys() - indexed_properties.keys()
    if undeclared:
        raise data_types.UndeclaredPropertyKeyError(
            undeclared, indexed_properties.keys()
        )
    for key, value in properties.items():
        # `bool` is an `int` at runtime and is its own property type here.
        if type(value) is not indexed_properties[key]:
            raise data_types.PropertyTypeMismatchError(
                key, indexed_properties[key], value
            )


def validate_query_vector(query_vector: Sequence[float], dimensions: int) -> None:
    """Raise ValueError unless a query vector has the store's dimensions and finite coordinates."""
    validate_dimensions(query_vector, dimensions)
    if not all(map(math.isfinite, query_vector)):
        position = next(
            position
            for position, coordinate in enumerate(query_vector)
            if not math.isfinite(coordinate)
        )
        raise ValueError(
            f"Query vector coordinate {position} is not finite: "
            f"{query_vector[position]}"
        )


def validate_score_threshold(score_threshold: float | None) -> None:
    """Raise ValueError if a score threshold is not finite; None means no threshold."""
    if score_threshold is not None and not math.isfinite(score_threshold):
        raise ValueError(f"Score threshold is not finite: {score_threshold}")


def validate_limit(limit: int) -> None:
    """Raise ValueError unless a query limit is positive."""
    if not limit > 0:
        raise ValueError(f"Limit is not positive: {limit}")


def validate_dimensions(vector: Sequence[float], dimensions: int) -> None:
    """Raise ValueError unless a vector has the store's dimensions."""
    if len(vector) != dimensions:
        raise ValueError(
            f"Vector has {len(vector)} dimensions; the store has {dimensions}"
        )


def validate_supported_filter(
    property_filter: FilterExpr,
    indexed_properties: Mapping[str, PropertyType],
) -> None:
    """
    Raise unless a filter names declared keys only.

    An undeclared key never exists in the store, so a filter naming one has
    nothing to scan for.
    """
    undeclared = filter_fields(property_filter) - indexed_properties.keys()
    if undeclared:
        raise data_types.UndeclaredPropertyKeyError(
            undeclared, indexed_properties.keys()
        )
