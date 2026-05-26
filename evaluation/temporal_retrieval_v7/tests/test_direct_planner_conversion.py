"""Unit tests for DirectQueryPlanner's JSON → TimeRange conversion.

Doesn't invoke the LLM — just tests the conversion logic against
synthetic JSON payloads matching the prompt's schema.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from temporal_retrieval_v7 import NEG_INF, POS_INF, intersect, is_empty
from temporal_retrieval_v7.planner_direct import (
    _iso_to_us,
    _json_to_refs,
    _us_to_iso,
)


def us(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1_000_000)


def test_iso_to_us_basic() -> None:
    assert _iso_to_us("2024-03-01") == us(2024, 3, 1)
    assert _iso_to_us("2024-01-01") == us(2024, 1, 1)


def test_iso_to_us_with_time() -> None:
    val = _iso_to_us("2024-03-01T12:34:56Z")
    expected = int(datetime(2024, 3, 1, 12, 34, 56, tzinfo=timezone.utc).timestamp() * 1_000_000)
    assert val == expected


def test_iso_to_us_invalid_raises() -> None:
    with pytest.raises(ValueError):
        _iso_to_us("not-a-date")


def test_us_to_iso_sentinels_to_none() -> None:
    assert _us_to_iso(NEG_INF) is None
    assert _us_to_iso(POS_INF) is None


def test_us_to_iso_real_date() -> None:
    assert _us_to_iso(us(2024, 3, 1)) == "2024-03-01"


# ---------------------------------------------------------------------------
# JSON → list[TimeRange] conversion
# ---------------------------------------------------------------------------


def test_json_to_refs_simple() -> None:
    """{lo: 2024-03-01, hi: 2024-04-01} → one bounded ref."""
    json_refs = [{"intervals": [{"lo": "2024-03-01", "hi": "2024-04-01"}]}]
    refs = _json_to_refs(json_refs)
    assert len(refs) == 1
    iv = refs[0].intervals[0]
    assert iv.earliest_us == us(2024, 3, 1)
    assert iv.latest_us == us(2024, 4, 1)


def test_json_to_refs_unbounded_via_null() -> None:
    """{lo: null, hi: 2024-01-01} → (-∞, 2024)."""
    json_refs = [{"intervals": [{"lo": None, "hi": "2024-01-01"}]}]
    refs = _json_to_refs(json_refs)
    iv = refs[0].intervals[0]
    assert iv.earliest_us <= NEG_INF + 1
    assert iv.latest_us == us(2024, 1, 1)


def test_json_to_refs_complement_two_pieces() -> None:
    """Complement of [2023] → one ref with two intervals."""
    json_refs = [{
        "intervals": [
            {"lo": None, "hi": "2023-01-01"},
            {"lo": "2024-01-01", "hi": None},
        ],
    }]
    refs = _json_to_refs(json_refs)
    intervals = refs[0].intervals
    assert len(intervals) == 2
    assert intervals[0].earliest_us <= NEG_INF + 1
    assert intervals[0].latest_us == us(2023, 1, 1)
    assert intervals[1].earliest_us == us(2024, 1, 1)
    assert intervals[1].latest_us >= POS_INF - 1


def test_json_to_refs_multi_period_three_refs() -> None:
    """3 separate periods (graded coverage): in 2017 and 2019 and 2021."""
    json_refs = [
        {"intervals": [{"lo": "2017-01-01", "hi": "2018-01-01"}]},
        {"intervals": [{"lo": "2019-01-01", "hi": "2020-01-01"}]},
        {"intervals": [{"lo": "2021-01-01", "hi": "2022-01-01"}]},
    ]
    refs = _json_to_refs(json_refs)
    assert len(refs) == 3


def test_json_to_refs_two_period_or() -> None:
    """Two disjoint periods (e.g., Q1 or Q4 2023) → two refs."""
    json_refs = [
        {"intervals": [{"lo": "2023-01-01", "hi": "2023-04-01"}]},
        {"intervals": [{"lo": "2023-10-01", "hi": "2024-01-01"}]},
    ]
    refs = _json_to_refs(json_refs)
    assert len(refs) == 2


def test_json_to_refs_drops_invalid_intervals() -> None:
    """lo >= hi intervals silently dropped."""
    json_refs = [{"intervals": [
        {"lo": "2024-03-01", "hi": "2024-04-01"},   # valid
        {"lo": "2024-05-01", "hi": "2024-04-01"},   # invalid (hi <= lo)
    ]}]
    refs = _json_to_refs(json_refs)
    assert len(refs[0].intervals) == 1


def test_json_to_refs_skips_empty() -> None:
    """If a ref ends up with no intervals (all invalid), drop the ref."""
    json_refs = [{"intervals": [
        {"lo": "2024-05-01", "hi": "2024-04-01"},   # invalid
    ]}]
    refs = _json_to_refs(json_refs)
    assert refs == []


def test_json_to_refs_complement_membership() -> None:
    """'in 2024 not in summer' — one ref, two intervals. Doc in March passes,
    doc in July excluded (intra-ref OR via multi-interval)."""
    from temporal_retrieval_v7 import TimeRange
    json_refs = [{"intervals": [
        {"lo": "2024-01-01", "hi": "2024-06-01"},
        {"lo": "2024-09-01", "hi": "2025-01-01"},
    ]}]
    refs = _json_to_refs(json_refs)
    qref = refs[0]
    march = TimeRange.closed(us(2024, 3, 1), us(2024, 4, 1))
    july = TimeRange.closed(us(2024, 7, 1), us(2024, 8, 1))
    assert not is_empty(intersect(qref, march))
    assert is_empty(intersect(qref, july))
