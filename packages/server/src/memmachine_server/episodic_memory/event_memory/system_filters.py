"""
System fields of EventMemory as filters.

EventMemory writes its system fields into every vector record under reserved
property keys, and a search bounds them with typed parameters (`since`,
`until`, `session_ids`, `source_ids`, `block_kinds`) beside the caller's
property filter. This module owns the translation between the two forms:
`system_predicates` builds the tree a vector store gets from the typed values,
and `split_system` recovers the typed values from a tree that names the
reserved keys. Nothing else in EventMemory knows the keys' spelling.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Final

from pydantic import BaseModel

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.property_keys import (
    is_reserved_property_key,
    reserved_property_key,
)

# Built rather than written out so the alphabet and length budget of the
# vector store's naming contract are checked at import time.
EVENT_TIMESTAMP_KEY: Final[str] = reserved_property_key("event", "timestamp")
EVENT_SESSION_KEY: Final[str] = reserved_property_key("event", "session")
EVENT_SOURCE_KEY: Final[str] = reserved_property_key("event", "source")
BLOCK_KIND_KEY: Final[str] = reserved_property_key("block", "kind")


class SystemFilters(BaseModel):
    """The typed system filters of a search or an expansion."""

    since: datetime | None = None
    """Inclusive lower bound on the event timestamp."""
    until: datetime | None = None
    """Exclusive upper bound on the event timestamp."""
    session_ids: list[str] | None = None
    source_ids: list[str] | None = None
    block_kinds: list[str] | None = None


def conjoin(clauses: Iterable[FilterExpr | None]) -> FilterExpr | None:
    """The conjunction of the given clauses; None when there are none."""
    combined: FilterExpr | None = None
    for clause in clauses:
        if clause is None:
            continue
        combined = clause if combined is None else And(left=combined, right=clause)
    return combined


def system_predicates(
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    session_ids: Iterable[str] | None = None,
    source_ids: Iterable[str] | None = None,
    block_kinds: Iterable[str] | None = None,
) -> FilterExpr | None:
    """The predicates on reserved keys that a vector store evaluates.

    `since` is inclusive and `until` exclusive, so ranges meet without
    overlap. An empty id or kind list admits nothing; None admits everything.
    """
    clauses: list[FilterExpr | None] = [
        Comparison(field=EVENT_TIMESTAMP_KEY, op=">=", value=since)
        if since is not None
        else None,
        Comparison(field=EVENT_TIMESTAMP_KEY, op="<", value=until)
        if until is not None
        else None,
        In(field=EVENT_SESSION_KEY, values=list(session_ids))
        if session_ids is not None
        else None,
        In(field=EVENT_SOURCE_KEY, values=list(source_ids))
        if source_ids is not None
        else None,
        In(field=BLOCK_KIND_KEY, values=list(block_kinds))
        if block_kinds is not None
        else None,
    ]
    return conjoin(clauses)


def split_system(expr: FilterExpr | None) -> tuple[SystemFilters, FilterExpr | None]:
    """Pull the conjuncts naming reserved keys out of a tree into typed values.

    The inverse of `system_predicates`, for an API boundary that lets a
    caller write a system field inside a filter: the returned tree names no
    reserved key and is what the caller's `property_filter` becomes. A
    reserved key is accepted only as a top-level conjunct in the shape
    `system_predicates` builds (a `>=` or `<` on the timestamp, an `=` or an
    `In` on the session, source or block kind); repeated conjuncts on one
    field intersect. Anywhere else, or on any other reserved key, it raises
    ValueError, since no typed value could carry it.
    """
    filters = SystemFilters()
    rest: list[FilterExpr] = []
    for conjunct in _conjuncts(expr):
        if not _names_reserved_key(conjunct):
            rest.append(conjunct)
            continue
        _absorb(filters, conjunct)
    return filters, conjoin(rest)


def _conjuncts(expr: FilterExpr | None) -> list[FilterExpr]:
    if expr is None:
        return []
    if isinstance(expr, And):
        return [*_conjuncts(expr.left), *_conjuncts(expr.right)]
    return [expr]


def _names_reserved_key(expr: FilterExpr) -> bool:
    if isinstance(expr, Comparison | In | IsNull):
        return is_reserved_property_key(expr.field)
    if isinstance(expr, And | Or):
        return _names_reserved_key(expr.left) or _names_reserved_key(expr.right)
    if isinstance(expr, Not):
        return _names_reserved_key(expr.expr)
    raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")


def _absorb(filters: SystemFilters, conjunct: FilterExpr) -> None:
    if isinstance(conjunct, Comparison) and conjunct.field == EVENT_TIMESTAMP_KEY:
        _absorb_timestamp(filters, conjunct)
    elif isinstance(conjunct, Comparison | In):
        _absorb_ids(filters, conjunct)
    else:
        raise ValueError(
            f"A reserved key is accepted only as a top-level = or In conjunct "
            f"over strings, or >= and < on {EVENT_TIMESTAMP_KEY}; got {conjunct!r}"
        )


def _absorb_timestamp(filters: SystemFilters, conjunct: Comparison) -> None:
    if not isinstance(conjunct.value, datetime):
        raise TypeError(f"{EVENT_TIMESTAMP_KEY} compares with a datetime")
    if conjunct.op == ">=":
        filters.since = _later(filters.since, conjunct.value)
    elif conjunct.op == "<":
        filters.until = _earlier(filters.until, conjunct.value)
    else:
        raise ValueError(
            f"{EVENT_TIMESTAMP_KEY} takes only >= (since) and < (until), "
            f"got {conjunct.op!r}"
        )


def _absorb_ids(filters: SystemFilters, conjunct: Comparison | In) -> None:
    ids = _string_ids(conjunct)
    if ids is None:
        raise ValueError(
            f"A reserved key is accepted only as a top-level = or In conjunct "
            f"over strings, or >= and < on {EVENT_TIMESTAMP_KEY}; got {conjunct!r}"
        )
    if conjunct.field == EVENT_SESSION_KEY:
        filters.session_ids = _intersect(filters.session_ids, ids)
    elif conjunct.field == EVENT_SOURCE_KEY:
        filters.source_ids = _intersect(filters.source_ids, ids)
    elif conjunct.field == BLOCK_KIND_KEY:
        filters.block_kinds = _intersect(filters.block_kinds, ids)
    else:
        raise ValueError(f"{conjunct.field!r} is not a filterable system field")


def _string_ids(conjunct: Comparison | In) -> list[str] | None:
    """The string values an `=` or `In` leaf names; None for any other leaf."""
    if isinstance(conjunct, Comparison):
        if conjunct.op == "=" and isinstance(conjunct.value, str):
            return [conjunct.value]
        return None
    if all(isinstance(value, str) for value in conjunct.values):
        return [str(value) for value in conjunct.values]
    return None


def _later(current: datetime | None, bound: datetime) -> datetime:
    return bound if current is None else max(current, bound)


def _earlier(current: datetime | None, bound: datetime) -> datetime:
    return bound if current is None else min(current, bound)


def _intersect(current: list[str] | None, ids: list[str]) -> list[str]:
    if current is None:
        return ids
    return [value for value in current if value in ids]
