"""Module for parsing filter strings into dictionaries."""

from collections.abc import Mapping
from decimal import Decimal
from typing import cast
from uuid import uuid4

import sqlglot
import sqlglot.expressions

from memmachine.common.data_types import FilterablePropertyValue


def parse_filter(filter_str: str) -> dict[str, FilterablePropertyValue | None]:
    """
    Parse a filter string into a dictionary.

    Args:
        filter_str (str): The filter string to parse.

    Returns:
        dict: A dictionary representation of the filter.

    """
    filter_dict: dict[str, FilterablePropertyValue | None] = {}
    if not filter_str:
        return filter_dict

    syntax_tree = sqlglot.parse_one(filter_str)
    return _extract_filter(syntax_tree)


def _extract_filter(
    expression: sqlglot.expressions.Expression,
) -> dict[str, FilterablePropertyValue | None]:
    match expression:
        case sqlglot.expressions.And():
            # May overwrite previous keys
            left_filter_dict = _extract_filter(expression.left)
            right_filter_dict = _extract_filter(expression.right)
            return _merge_filters(left_filter_dict, right_filter_dict)
        case sqlglot.expressions.EQ():
            filterable_property_key = _extract_filterable_property_key(expression.left)
            filterable_property_value = _extract_filterable_property_value(
                expression.right
            )
            return {filterable_property_key: filterable_property_value}
        case sqlglot.expressions.Is():
            filterable_property_key = _extract_filterable_property_key(expression.left)
            if expression.right.key != "null":
                raise ValueError(f"Expected NULL expression, got: {expression.right}")
            return {filterable_property_key: None}
        case _:
            raise ValueError(f"Unsupported filter expression: {expression}")


def _extract_filterable_property_key(expression: sqlglot.expressions.Expression) -> str:
    match expression:
        case sqlglot.expressions.Column():
            if expression.catalog:
                return f"{expression.catalog}.{expression.db}.{expression.table}.{expression.name}"
            if expression.db:
                return f"{expression.db}.{expression.table}.{expression.name}"
            if expression.table:
                return f"{expression.table}.{expression.name}"
            return expression.name
        case sqlglot.expressions.Dot():
            return f"{expression.this}.{expression.expression}"
        case _:
            raise ValueError(
                f"Unsupported filterable property key expression: {expression}"
            )


def _extract_filterable_property_value(
    expression: sqlglot.expressions.Expression,
) -> FilterablePropertyValue:
    value: FilterablePropertyValue | Decimal
    match expression:
        case sqlglot.expressions.Boolean():
            value = expression.to_py()
        case sqlglot.expressions.Literal():
            value = expression.to_py()
        case sqlglot.expressions.Neg():
            value = expression.to_py()
        case _:
            # Doesn't support datetime yet.
            raise ValueError(
                f"Unsupported filterable property value expression: {expression}"
            )

    if isinstance(value, Decimal):
        value = float(value)
    return cast(FilterablePropertyValue, value)


FILTER_CONTRADICTION_KEY = "filter_contradiction"


def _merge_filters(
    dict1: Mapping[str, FilterablePropertyValue | None],
    dict2: Mapping[str, FilterablePropertyValue | None],
) -> dict[str, FilterablePropertyValue | None]:
    if FILTER_CONTRADICTION_KEY in dict1 or FILTER_CONTRADICTION_KEY in dict2:
        return {FILTER_CONTRADICTION_KEY: str(uuid4())}

    common_keys = dict1.keys() & dict2.keys()
    for common_key in common_keys:
        if dict1[common_key] != dict2[common_key]:
            # Contradictory filters for the same key.
            # Return a filter that can never be satisfied.
            return {FILTER_CONTRADICTION_KEY: str(uuid4())}

    return {**dict1, **dict2}
