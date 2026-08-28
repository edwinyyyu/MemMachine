"""
Filter expression trees.

A filter expression is built from the nodes below and compiled by each store
into its own query language. It is constructed, never parsed from text: a
caller states the tree it means, so there is no syntax to learn and no way to
write something that parses successfully into a different filter than intended.

Text belongs at a service boundary, where a filter genuinely arrives as a
string, and not in this library.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from memmachine_core.common import PropertyValue


@runtime_checkable
class FilterExpr(Protocol):
    """Marker protocol for filter expression nodes."""


ComparisonOp = Literal["=", "!=", ">", "<", ">=", "<="]


@dataclass(frozen=True)
class Comparison(FilterExpr):
    """Scalar comparison of a field against a value."""

    field: str
    op: ComparisonOp
    value: PropertyValue


@dataclass(frozen=True)
class In(FilterExpr):
    """Membership test of a field against a list of values."""

    field: str
    values: list[int] | list[str]


@dataclass(frozen=True)
class IsNull(FilterExpr):
    """Nullity check on a field (field IS NULL)."""

    field: str


@dataclass(frozen=True)
class And(FilterExpr):
    """Logical conjunction of two filter expressions."""

    left: FilterExpr
    right: FilterExpr


@dataclass(frozen=True)
class Or(FilterExpr):
    """Logical disjunction of two filter expressions."""

    left: FilterExpr
    right: FilterExpr


@dataclass(frozen=True)
class Not(FilterExpr):
    """Logical negation of a filter expression."""

    expr: FilterExpr


def map_filter_fields(
    expr: FilterExpr,
    transform: Callable[[str], str],
) -> FilterExpr:
    """Apply a field name transformation to all fields in a FilterExpr tree."""
    if isinstance(expr, Comparison):
        return Comparison(field=transform(expr.field), op=expr.op, value=expr.value)
    if isinstance(expr, In):
        return In(field=transform(expr.field), values=expr.values)
    if isinstance(expr, IsNull):
        return IsNull(field=transform(expr.field))
    if isinstance(expr, And):
        return And(
            left=map_filter_fields(expr.left, transform),
            right=map_filter_fields(expr.right, transform),
        )
    if isinstance(expr, Or):
        return Or(
            left=map_filter_fields(expr.left, transform),
            right=map_filter_fields(expr.right, transform),
        )
    if isinstance(expr, Not):
        return Not(expr=map_filter_fields(expr.expr, transform))
    raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")
