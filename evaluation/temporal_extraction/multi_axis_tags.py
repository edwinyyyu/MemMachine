"""Multi-axis tag generation.

Given a dict[axis -> AxisDistribution], emit a set of ``{axis}:{value}`` tags
for every axis-value with probability above ``TAG_PROB_THRESHOLD`` on axes
that are marked informative. Non-informative axes emit no tags.
"""

from __future__ import annotations

from typing import Any

from axis_distributions import (
    TAG_PROB_THRESHOLD,
    AxisDistribution,
    axes_for_expression,
)
from schema import TimeExpression

# Human-readable value formatters per axis.
_MONTH_NAME = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}
_WEEKDAY_NAME = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}


def _fmt(axis: str, value: Any) -> str:
    if axis == "year":
        return f"year:{int(value):04d}"
    if axis == "month":
        return f"month:{_MONTH_NAME.get(int(value), str(value))}"
    if axis == "day_of_month":
        return f"day_of_month:{int(value):02d}"
    if axis == "weekday":
        return f"weekday:{_WEEKDAY_NAME.get(int(value), str(value))}"
    if axis == "hour":
        return f"hour:{int(value):02d}"
    if axis == "quarter":
        return f"quarter:Q{int(value)}"
    if axis == "decade":
        return f"decade:{int(value)}s"
    if axis == "season":
        return f"season:{value}"
    if axis == "part_of_day":
        return f"part_of_day:{value}"
    if axis == "weekend":
        return f"weekend:{value}"
    return f"{axis}:{value}"


def tags_for_axes(
    axes: dict[str, AxisDistribution],
    threshold: float = TAG_PROB_THRESHOLD,
) -> set[str]:
    """Generate multi-axis tag set.

    Only considers axes marked informative. Emits a tag per axis-value with
    probability > threshold.
    """
    tags: set[str] = set()
    for axis, ad in axes.items():
        if not ad.informative or not ad.values:
            continue
        for val, p in ad.values.items():
            if p > threshold:
                tags.add(_fmt(axis, val))
    return tags


def tags_for_expression(te: TimeExpression) -> set[str]:
    return tags_for_axes(axes_for_expression(te))
