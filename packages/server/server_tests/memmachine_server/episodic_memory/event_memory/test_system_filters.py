"""Tests for the translation between typed system filters and filter trees."""

from datetime import UTC, datetime

import pytest

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.property_keys import reserved_property_key
from memmachine_server.episodic_memory.event_memory.system_filters import (
    BLOCK_KIND_KEY,
    EVENT_SESSION_KEY,
    EVENT_SOURCE_KEY,
    EVENT_TIMESTAMP_KEY,
    SystemFilters,
    conjoin,
    split_system,
    system_predicates,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 1, 2, tzinfo=UTC)


def _conjuncts(expr):
    if isinstance(expr, And):
        return [*_conjuncts(expr.left), *_conjuncts(expr.right)]
    return [expr]


def test_no_system_filter_is_no_tree():
    assert system_predicates() is None


def test_predicates_name_the_reserved_keys():
    tree = system_predicates(
        since=_T0,
        until=_T1,
        session_ids=["s1"],
        source_ids=["alice", "bob"],
        block_kinds=["text"],
    )
    assert _conjuncts(tree) == [
        Comparison(field=EVENT_TIMESTAMP_KEY, op=">=", value=_T0),
        Comparison(field=EVENT_TIMESTAMP_KEY, op="<", value=_T1),
        In(field=EVENT_SESSION_KEY, values=["s1"]),
        In(field=EVENT_SOURCE_KEY, values=["alice", "bob"]),
        In(field=BLOCK_KIND_KEY, values=["text"]),
    ]


def test_empty_ids_admit_nothing_and_none_admits_everything():
    assert system_predicates(session_ids=None) is None
    assert system_predicates(session_ids=[]) == In(field=EVENT_SESSION_KEY, values=[])


def test_split_recovers_the_typed_values_and_leaves_the_rest():
    caller = Comparison(field="color", op="=", value="red")
    tree = conjoin(
        [
            system_predicates(since=_T0, until=_T1, session_ids=["s1"]),
            caller,
            Comparison(field=EVENT_SOURCE_KEY, op="=", value="alice"),
        ]
    )

    filters, rest = split_system(tree)

    assert filters == SystemFilters(
        since=_T0, until=_T1, session_ids=["s1"], source_ids=["alice"]
    )
    assert rest == caller


def test_split_of_nothing_is_nothing():
    assert split_system(None) == (SystemFilters(), None)


def test_split_intersects_repeated_conjuncts():
    tree = And(
        left=In(field=EVENT_SESSION_KEY, values=["a", "b"]),
        right=And(
            left=In(field=EVENT_SESSION_KEY, values=["b", "c"]),
            right=And(
                left=Comparison(field=EVENT_TIMESTAMP_KEY, op=">=", value=_T0),
                right=Comparison(field=EVENT_TIMESTAMP_KEY, op=">=", value=_T1),
            ),
        ),
    )
    filters, rest = split_system(tree)
    assert filters.session_ids == ["b"]
    assert filters.since == _T1
    assert rest is None


@pytest.mark.parametrize(
    "tree",
    [
        Or(
            left=Comparison(field=EVENT_SESSION_KEY, op="=", value="s1"),
            right=Comparison(field="color", op="=", value="red"),
        ),
        Not(expr=Comparison(field=EVENT_SESSION_KEY, op="=", value="s1")),
        IsNull(field=EVENT_SESSION_KEY),
        Comparison(field=EVENT_SESSION_KEY, op="!=", value="s1"),
        Comparison(field=EVENT_TIMESTAMP_KEY, op=">", value=_T0),
        Comparison(field=reserved_property_key("other", "field"), op="=", value="x"),
    ],
)
def test_split_rejects_a_reserved_key_no_typed_value_carries(tree):
    with pytest.raises(ValueError, match=r"reserved key|system field|takes only"):
        split_system(tree)
