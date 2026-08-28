"""
Filter expression trees.

A filter expression is built from the nodes below and compiled by each store
into its own query language. It is constructed, never parsed from text: a
caller states the tree it means, so there is no syntax to learn and no way to
write something that parses successfully into a different filter than intended.

Text belongs at a service boundary, where a filter genuinely arrives as a
string, and not in this library.

`FilterExpr` is a closed union, so a compiler written as a `match` with no
default arm is checked for exhaustiveness: adding a node here fails type
checking in every store rather than raising on whichever query first reaches
the new node.

A predicate matches a record only when the field holds a value of the
compared type. A record that does not carry the field, or carries it with a
different type, is not a match -- which is what `NotEquals` and
`Not(Equals(...))` distinguish: the former keeps only records that hold a
comparable, differing value, the latter also keeps records that hold no
comparable value at all.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from memmachine_core.common import OrderedValue, PropertyValue

type FilterExpr = Equals | NotEquals | Ordering | In | IsMissing | And | Or | Not
"""Any node of a filter expression tree."""

OrderingOp = Literal[">", "<", ">=", "<="]


@dataclass(frozen=True)
class Equals:
    """Field holds a value equal to `value`."""

    field: str
    value: PropertyValue


@dataclass(frozen=True)
class NotEquals:
    """Field holds a comparable value that differs from `value`."""

    field: str
    value: PropertyValue


@dataclass(frozen=True)
class Ordering:
    """
    Field holds a value ordered against `value`.

    Only values with a total order are comparable, so `bool` and `str` are not
    accepted: ordering booleans is meaningless, and ordering strings is a
    lexicographic comparison whose result depends on how a store happens to
    encode the value. Equality on those types is expressed with `Equals`.
    """

    field: str
    op: OrderingOp
    value: OrderedValue


@dataclass(frozen=True)
class In:
    """
    Field holds a value among `values`.

    Empty `values` matches nothing, which is the identity of a disjunction of
    equalities and lets a caller pass a narrowed set without special-casing it.

    Values are homogeneous because a store indexes and compares a property by
    its type; a mixed list has no single type to compare against. Booleans are
    excluded rather than treated as integers.
    """

    field: str
    values: tuple[int, ...] | tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject value lists a store cannot compare as a single type."""
        if any(isinstance(value, bool) for value in self.values):
            raise TypeError(f"In({self.field!r}) values must be int or str, not bool")
        if len({type(value) for value in self.values}) > 1:
            raise TypeError(
                f"In({self.field!r}) values must all be int or all be str, got "
                f"{sorted({type(value).__name__ for value in self.values})}"
            )


@dataclass(frozen=True)
class IsMissing:
    """
    Field holds no comparable value.

    True for a record that does not carry the field at all. Property values
    are never null, so absence is the only way a field can hold nothing.
    """

    field: str


@dataclass(frozen=True)
class And:
    """
    All operands match.

    At least one operand is required: an empty conjunction would oblige every
    store to render an identity element, and callers already spell "no filter"
    as `None`.
    """

    operands: tuple[FilterExpr, ...]

    def __post_init__(self) -> None:
        """Reject an empty conjunction."""
        if not self.operands:
            raise ValueError("And requires at least one operand")


@dataclass(frozen=True)
class Or:
    """
    At least one operand matches.

    At least one operand is required, for the reason given on `And`.
    """

    operands: tuple[FilterExpr, ...]

    def __post_init__(self) -> None:
        """Reject an empty disjunction."""
        if not self.operands:
            raise ValueError("Or requires at least one operand")


@dataclass(frozen=True)
class Not:
    """The operand does not match."""

    operand: FilterExpr


def map_filter_fields(
    expr: FilterExpr,
    transform: Callable[[str], str],
) -> FilterExpr:
    """Apply a field name transformation to every field in a FilterExpr tree."""
    match expr:
        case Equals(field, value):
            return Equals(transform(field), value)
        case NotEquals(field, value):
            return NotEquals(transform(field), value)
        case Ordering(field, op, value):
            return Ordering(transform(field), op, value)
        case In(field, values):
            return In(transform(field), values)
        case IsMissing(field):
            return IsMissing(transform(field))
        case And(operands):
            return And(tuple(map_filter_fields(o, transform) for o in operands))
        case Or(operands):
            return Or(tuple(map_filter_fields(o, transform) for o in operands))
        case Not(operand):
            return Not(map_filter_fields(operand, transform))
