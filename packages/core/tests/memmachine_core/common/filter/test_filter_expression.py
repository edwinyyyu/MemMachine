"""Tests for filter expression trees."""

from typing import Any, cast

import pytest

from memmachine_core.common.filter import (
    And,
    Equals,
    In,
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
    map_filter_fields,
)

# --- construction validity ---


def test_in_rejects_mixed_value_types() -> None:
    with pytest.raises(TypeError, match="must all be int or all be str"):
        In("x", cast(Any, (1, "two")))


def test_in_rejects_bool_values() -> None:
    with pytest.raises(TypeError, match="must be int or str, not bool"):
        In("flag", cast(Any, (True, False)))


def test_in_accepts_empty_and_homogeneous_values() -> None:
    assert In("x", ()).values == ()
    assert In("x", (1, 2)).values == (1, 2)
    assert In("tag", ("a", "b")).values == ("a", "b")


def test_and_or_reject_no_operands() -> None:
    with pytest.raises(ValueError, match="And requires at least one operand"):
        And(())
    with pytest.raises(ValueError, match="Or requires at least one operand"):
        Or(())


def test_and_or_accept_any_arity() -> None:
    operands = (Equals("a", 1), Equals("b", 2), Equals("c", 3))
    assert And(operands).operands == operands
    assert Or(operands[:1]).operands == operands[:1]


def test_conjunction_equality_is_independent_of_nesting_order() -> None:
    # Binary nodes made this depend on how the caller happened to fold them.
    clauses = (Equals("a", 1), Equals("b", 2), Equals("c", 3))
    assert And(clauses) == And(tuple(clauses))


# --- map_filter_fields ---


def test_map_filter_fields_equals() -> None:
    assert map_filter_fields(Equals("m.foo", "bar"), str.upper) == Equals(
        "M.FOO", "bar"
    )


def test_map_filter_fields_not_equals() -> None:
    assert map_filter_fields(NotEquals("m.foo", "bar"), str.upper) == NotEquals(
        "M.FOO", "bar"
    )


def test_map_filter_fields_ordering() -> None:
    assert map_filter_fields(Ordering("m.count", ">=", 10), str.upper) == Ordering(
        "M.COUNT", ">=", 10
    )


def test_map_filter_fields_in() -> None:
    assert map_filter_fields(In("m.tag", ("a", "b")), str.upper) == In(
        "M.TAG", ("a", "b")
    )


def test_map_filter_fields_is_missing() -> None:
    assert map_filter_fields(IsMissing("m.note"), str.upper) == IsMissing("M.NOTE")


def test_map_filter_fields_and() -> None:
    expr = And((Equals("a", 1), Equals("b", 2)))
    assert map_filter_fields(expr, str.upper) == And((Equals("A", 1), Equals("B", 2)))


def test_map_filter_fields_or() -> None:
    expr = Or((Equals("x", 1), Equals("y", 2)))
    assert map_filter_fields(expr, str.upper) == Or((Equals("X", 1), Equals("Y", 2)))


def test_map_filter_fields_not() -> None:
    expr = Not(Equals("status", "CLOSED"))
    assert map_filter_fields(expr, str.upper) == Not(Equals("STATUS", "CLOSED"))


def test_map_filter_fields_reaches_every_leaf_of_a_nested_tree() -> None:
    expr = Not(
        And(
            (
                Equals("a", 1),
                Or((Ordering("b", "<", 2), In("c", ("x",)), IsMissing("d"))),
            )
        )
    )
    assert map_filter_fields(expr, lambda field: f"prefix_{field}") == Not(
        And(
            (
                Equals("prefix_a", 1),
                Or(
                    (
                        Ordering("prefix_b", "<", 2),
                        In("prefix_c", ("x",)),
                        IsMissing("prefix_d"),
                    )
                ),
            )
        )
    )
