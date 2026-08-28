"""Tests for filter expression trees."""

from memmachine_core.common.filter import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
    map_filter_fields,
)

# --- map_filter_fields tests ---


def test_map_filter_fields_comparison() -> None:
    expr = Comparison(field="m.foo", op="=", value="bar")
    result = map_filter_fields(expr, lambda f: f.upper())
    assert result == Comparison(field="M.FOO", op="=", value="bar")


def test_map_filter_fields_in() -> None:
    expr = In(field="m.tag", values=["a", "b"])
    result = map_filter_fields(expr, lambda f: f.upper())
    assert result == In(field="M.TAG", values=["a", "b"])


def test_map_filter_fields_is_null() -> None:
    expr = IsNull(field="m.note")
    result = map_filter_fields(expr, lambda f: f.upper())
    assert result == IsNull(field="M.NOTE")


def test_map_filter_fields_and() -> None:
    expr = And(
        left=Comparison(field="a", op="=", value=1),
        right=Comparison(field="b", op="=", value=2),
    )
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, And)
    assert result.left == Comparison(field="A", op="=", value=1)
    assert result.right == Comparison(field="B", op="=", value=2)


def test_map_filter_fields_or() -> None:
    expr = Or(
        left=Comparison(field="x", op="=", value=1),
        right=Comparison(field="y", op="=", value=2),
    )
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, Or)
    assert result.left == Comparison(field="X", op="=", value=1)
    assert result.right == Comparison(field="Y", op="=", value=2)


def test_map_filter_fields_not() -> None:
    expr = Not(expr=Comparison(field="status", op="=", value="CLOSED"))
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, Not)
    assert result.expr == Comparison(field="STATUS", op="=", value="CLOSED")


def test_map_filter_fields_nested() -> None:
    # NOT (a = 1 AND b = 2)
    expr = Not(
        expr=And(
            left=Comparison(field="a", op="=", value=1),
            right=Comparison(field="b", op="=", value=2),
        )
    )
    result = map_filter_fields(expr, lambda f: f"prefix_{f}")
    assert isinstance(result, Not)
    assert isinstance(result.expr, And)
    assert result.expr.left == Comparison(field="prefix_a", op="=", value=1)
    assert result.expr.right == Comparison(field="prefix_b", op="=", value=2)


def test_map_filter_fields_applies_transform_to_every_leaf() -> None:
    expr = And(
        left=Comparison(field="foo", op="=", value="bar"),
        right=Comparison(field="producer_id", op="=", value="alice"),
    )
    result = map_filter_fields(expr, lambda f: f.upper())
    assert isinstance(result, And)
    assert result.left == Comparison(field="FOO", op="=", value="bar")
    assert result.right == Comparison(field="PRODUCER_ID", op="=", value="alice")
