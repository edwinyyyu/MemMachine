"""Shared utilities for vector store implementations."""

import re

from memmachine_core.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
)

_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]+$")
_IDENTIFIER_MAX_BYTES = 32


def validate_identifier(value: str) -> bool:
    """Return True if value is a valid identifier (a-z0-9_, max 32 bytes)."""
    return (
        bool(_IDENTIFIER_RE.match(value))
        and len(value.encode()) <= _IDENTIFIER_MAX_BYTES
    )


def filter_fields(expr: FilterExpr) -> set[str]:
    """Return every field name addressed by a filter tree."""
    match expr:
        case (
            Equals(field)
            | NotEquals(field)
            | Ordering(field)
            | In(field)
            | IsMissing(field)
        ):
            return {field}
        case Not(operand):
            return filter_fields(operand)
        case And(operands) | Or(operands):
            return {field for o in operands for field in filter_fields(o)}


def validate_filter(expr: FilterExpr) -> bool:
    """Return whether all field names in the filter tree are valid identifiers."""
    return all(validate_identifier(field) for field in filter_fields(expr))
