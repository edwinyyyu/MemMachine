"""Unit tests for time_range.py and scoring.py.

Covers all 18 §8 worked cases from SPEC.md plus edge cases on
canonicalization, set ops, and pair_overlap.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from temporal_retrieval_v7 import (
    NEG_INF,
    POS_INF,
    SENTINEL_THRESHOLD,
    Interval,
    TimeRange,
    complement,
    difference,
    final_score,
    intersect,
    intersect_all,
    measure,
    pair_overlap,
    symmetric_difference,
    temporal_pass,
    union,
    union_all,
)


# Calendar helpers — datetime → microseconds since epoch
def us(year: int, month: int = 1, day: int = 1) -> int:
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def month(year: int, month: int) -> TimeRange:
    """Single calendar month as a TimeRange."""
    if month == 12:
        return TimeRange.closed(us(year, 12, 1), us(year + 1, 1, 1))
    return TimeRange.closed(us(year, month, 1), us(year, month + 1, 1))


def year(y: int) -> TimeRange:
    return TimeRange.closed(us(y, 1, 1), us(y + 1, 1, 1))


def quarter(y: int, q: int) -> TimeRange:
    start_month = 1 + (q - 1) * 3
    end_month = start_month + 3
    if end_month > 12:
        return TimeRange.closed(us(y, start_month, 1), us(y + 1, 1, 1))
    return TimeRange.closed(us(y, start_month, 1), us(y, end_month, 1))


def summer(y: int) -> TimeRange:
    return TimeRange.closed(us(y, 6, 1), us(y, 9, 1))


# ===========================================================================
# Interval / canonicalization
# ===========================================================================


def test_interval_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Interval(100, 100)
    with pytest.raises(ValueError):
        Interval(200, 100)


def test_canonicalize_merges_overlapping() -> None:
    r = TimeRange.from_intervals([
        Interval(10, 30),
        Interval(20, 40),  # overlaps with [10, 30)
    ])
    assert len(r.intervals) == 1
    assert r.intervals[0] == Interval(10, 40)


def test_canonicalize_merges_adjacent() -> None:
    r = TimeRange.from_intervals([
        Interval(10, 20),
        Interval(20, 30),  # adjacent at 20
    ])
    assert len(r.intervals) == 1
    assert r.intervals[0] == Interval(10, 30)


def test_canonicalize_keeps_disjoint() -> None:
    r = TimeRange.from_intervals([
        Interval(40, 50),
        Interval(10, 20),
        Interval(30, 35),
    ])
    assert r.intervals == (
        Interval(10, 20),
        Interval(30, 35),
        Interval(40, 50),
    )


# ===========================================================================
# Set ops
# ===========================================================================


def test_intersect_basic() -> None:
    A = TimeRange.from_intervals([Interval(10, 30)])
    B = TimeRange.from_intervals([Interval(20, 40)])
    r = intersect(A, B)
    assert r.intervals == (Interval(20, 30),)


def test_intersect_half_open_no_boundary_overlap() -> None:
    # [10, 20) ∩ [20, 30) = ∅ (half-open semantics)
    A = TimeRange.from_intervals([Interval(10, 20)])
    B = TimeRange.from_intervals([Interval(20, 30)])
    assert intersect(A, B).intervals == ()


def test_intersect_multi_piece() -> None:
    A = TimeRange.from_intervals([Interval(0, 10), Interval(20, 30), Interval(40, 50)])
    B = TimeRange.from_intervals([Interval(5, 25), Interval(45, 60)])
    r = intersect(A, B)
    assert r.intervals == (Interval(5, 10), Interval(20, 25), Interval(45, 50))


def test_union_merges_overlaps() -> None:
    A = TimeRange.from_intervals([Interval(0, 10)])
    B = TimeRange.from_intervals([Interval(5, 15)])
    assert union(A, B).intervals == (Interval(0, 15),)


def test_complement_bounded() -> None:
    A = TimeRange.from_intervals([Interval(us(2024, 1, 1), us(2025, 1, 1))])
    r = complement(A)
    assert r.intervals == (
        Interval(NEG_INF, us(2024, 1, 1)),
        Interval(us(2025, 1, 1), POS_INF),
    )


def test_complement_of_universal_is_empty() -> None:
    U = TimeRange.universal()
    assert complement(U).intervals == ()


def test_complement_of_empty_is_universal() -> None:
    E = TimeRange.empty()
    c = complement(E)
    assert c.intervals == (Interval(NEG_INF, POS_INF),)


def test_complement_multi_piece() -> None:
    A = TimeRange.from_intervals([Interval(10, 20), Interval(30, 40)])
    r = complement(A)
    assert r.intervals == (
        Interval(NEG_INF, 10),
        Interval(20, 30),
        Interval(40, POS_INF),
    )


def test_difference() -> None:
    A = TimeRange.from_intervals([Interval(0, 100)])
    B = TimeRange.from_intervals([Interval(20, 50)])
    assert difference(A, B).intervals == (Interval(0, 20), Interval(50, 100))


def test_symmetric_difference() -> None:
    A = TimeRange.from_intervals([Interval(0, 20)])
    B = TimeRange.from_intervals([Interval(10, 30)])
    r = symmetric_difference(A, B)
    assert r.intervals == (Interval(0, 10), Interval(20, 30))


def test_intersect_all_empty_list_is_universal() -> None:
    assert intersect_all([]).intervals == (Interval(NEG_INF, POS_INF),)


def test_union_all_empty_list_is_empty() -> None:
    assert union_all([]).intervals == ()


# ===========================================================================
# Pair overlap — §8 worked cases
# ===========================================================================


def test_8_1_simple_intersect_identical() -> None:
    """§8.1: query 'in March 2024', doc 'March 2024' → 1.0"""
    q = month(2024, 3)
    d = month(2024, 3)
    assert pair_overlap(q, d) == 1.0


def test_8_2_doc_narrower_than_query() -> None:
    """§8.2: query 'in 2024', doc 'March 2024' → 1.0 (doc fully inside)"""
    q = year(2024)
    d = month(2024, 3)
    assert pair_overlap(q, d) == 1.0


def test_8_3a_open_ended_doc_crossing_boundary() -> None:
    """§8.3a: query 'before 2030', doc 'since March 2024' → 1.0
    (both have ±∞ endpoints, intersection non-empty)."""
    q = TimeRange.before(us(2030))  # (-∞, 2030)
    d = TimeRange.after(us(2024, 3))  # [Mar 2024, +∞)
    assert pair_overlap(q, d) == 1.0


def test_8_3b_open_ended_disjoint() -> None:
    """§8.3b: query 'before 2020', doc 'since March 2024' → 0 (gap)."""
    q = TimeRange.before(us(2020))
    d = TimeRange.after(us(2024, 3))
    assert pair_overlap(q, d) == 0.0


def test_8_4_compound_intersect_disjoint() -> None:
    """§8.4: query 'in 2024 not in summer' composed range × doc anchors."""
    summer_2024 = summer(2024)
    q = intersect(year(2024), complement(summer_2024))
    # Query range: [Jan-May 2024) ∪ [Sep-Dec 2024 + Jan 1 2025)
    assert pair_overlap(q, month(2024, 3)) == 1.0  # March is in
    assert pair_overlap(q, month(2024, 7)) == 0.0  # July is excluded
    # Jan 1 2025 is the half-open exclusive boundary → out
    jan_2025 = TimeRange.closed(us(2025, 1, 1), us(2025, 2, 1))
    assert pair_overlap(q, jan_2025) == 0.0


def test_8_5_and_of_multiple_disjoints() -> None:
    """§8.5: 'not in 2020 or 2022' → complement([2020] ∪ [2022])."""
    excluded = union(year(2020), year(2022))
    q = complement(excluded)
    # Doc in March 2020 → 0 (inside excluded)
    assert pair_overlap(q, month(2020, 3)) == 0.0
    # Doc in October 2023 → 1.0 (Oct 2023 ∈ [Jan 2023, +∞) part of q)
    assert pair_overlap(q, month(2023, 10)) == 1.0


def test_8_6_colloquial_and_as_or() -> None:
    """§8.6: 'in 2020 and 2024' → planner emits two refs (incompatible).
    Doc matching both → 1.0."""
    query_refs = [year(2020), year(2024)]
    # Doc with March 2020 + June 2024 → each qref hits one of the doc refs
    doc_refs = [month(2020, 3), month(2024, 6)]
    assert final_score(query_refs, doc_refs) == 1.0
    # Doc matching only one → 0.5
    assert final_score(query_refs, [month(2020, 3)]) == 0.5


def test_8_7_or_clauses() -> None:
    """§8.7: 'in Q1 or Q4 of 2023' → two refs."""
    query_refs = [quarter(2023, 1), quarter(2023, 4)]
    # Doc Feb 2023 → matches Q1, not Q4 → 0.5
    assert final_score(query_refs, [month(2023, 2)]) == 0.5
    # Doc Feb 2023 + Nov 2023 → matches both → 1.0
    assert final_score(query_refs, [month(2023, 2), month(2023, 11)]) == 1.0


def test_8_12_engagement_relevance() -> None:
    """§8.12: query 'outside summer 2024', doc has summer + October refs.
    Per-doc-ref max → October ref satisfies the query."""
    q = complement(summer(2024))
    doc_refs = [summer(2024), month(2024, 10)]
    assert final_score([q], doc_refs) == 1.0


def test_8_13_retrospective() -> None:
    """§8.13: query 'outside Q2 2024', doc content [Q3 2023], [Dec 2023].
    Doc ref_time (May 2024, inside Q2) is metadata; content drives score."""
    q = complement(quarter(2024, 2))
    doc_refs = [quarter(2023, 3), month(2023, 12)]
    assert final_score([q], doc_refs) == 1.0


def test_8_14_event_negation_temporal_layer_blind() -> None:
    """§8.14: query 'what did NOT happen on May 3' → planner emits [May 3].
    All May-3 docs score 1.0; polarity is semantic layer's job."""
    may_3 = TimeRange.closed(us(2024, 5, 3), us(2024, 5, 4))
    q = may_3
    # Doc "launch did NOT happen on May 3" → [May 3]
    assert pair_overlap(q, may_3) == 1.0


def test_8_15_universal_or_empty_doc_matches_all() -> None:
    """§8.15: timeless doc → 1.0 for all queries."""
    q = year(2024)
    assert final_score([q], []) == 1.0  # no doc refs
    # universal range is treated as infinite measure on both sides → both
    # intersect non-empty → 1.0
    assert pair_overlap(q, TimeRange.universal()) == 1.0


def test_8_16_both_end_unbounded_with_gap() -> None:
    """§8.16: A=(-∞,-1)∪[1,∞), B=[-1,1). Inter=∅, HARD GATE → 0.
    The both-infinite shortcut must NOT fire here."""
    A = TimeRange.from_intervals([Interval(NEG_INF, -1), Interval(1, POS_INF)])
    B = TimeRange.from_intervals([Interval(-1, 1)])
    assert pair_overlap(A, B) == 0.0


def test_8_17_opposite_end_unbounded_with_overlap() -> None:
    """§8.17a: A=(-∞,2030), B=(2020,+∞), inter=(2020,2030) non-empty.
    Both have infinite measure → shortcut fires → 1.0."""
    A = TimeRange.closed(NEG_INF, us(2030, 1, 1))
    B = TimeRange.closed(us(2020, 1, 1), POS_INF)
    assert pair_overlap(A, B) == 1.0


def test_8_17b_opposite_end_unbounded_with_gap() -> None:
    """§8.17b: A=(-∞,2020), B=(2024,+∞), inter=∅, HARD GATE → 0."""
    A = TimeRange.closed(NEG_INF, us(2020, 1, 1))
    B = TimeRange.closed(us(2024, 1, 1), POS_INF)
    assert pair_overlap(A, B) == 0.0


def test_8_18_multi_piece_infinite_with_bounded() -> None:
    """§8.18: A=(-∞,-1)∪[1,+∞), B=[-2,2). |inter|=2, |B|=4 → frac_min=0.5."""
    A = TimeRange.from_intervals([Interval(NEG_INF, -1), Interval(1, POS_INF)])
    B = TimeRange.from_intervals([Interval(-2, 2)])
    # inter = [-2, -1) ∪ [1, 2)
    assert pair_overlap(A, B) == pytest.approx(0.5)


# ===========================================================================
# Bounded recurrence (eager enumeration at construction time)
# ===========================================================================


def test_bounded_recurrence_every_march_2020_2024() -> None:
    """§8.8: query 'every March 2020-2024' enumerates to 5 explicit
    interval TimeRange. Doc 'March 2022' → frac_min over min(|Q|=5mo,
    |D|=1mo)=1mo → 1.0."""
    marches = [month(y, 3) for y in (2020, 2021, 2022, 2023, 2024)]
    q = union_all(marches)
    assert len(q.intervals) == 5
    # Doc March 2022
    assert pair_overlap(q, month(2022, 3)) == 1.0
    # Doc full year 2022 (12 months) → intersect = March (1 month), min(5,12)=5
    # → 1/5 = 0.2
    assert pair_overlap(q, year(2022)) == pytest.approx(0.2, abs=1e-3)


# ===========================================================================
# Filter (derived from score)
# ===========================================================================


def test_temporal_pass_derived() -> None:
    q = year(2024)
    assert temporal_pass([q], [month(2024, 3)]) is True
    assert temporal_pass([q], [month(2025, 3)]) is False
    assert temporal_pass([], [month(2024, 3)]) is True  # no constraint
    assert temporal_pass([q], []) is True  # timeless doc


def test_universal_query_matches_everything() -> None:
    U = TimeRange.universal()
    assert pair_overlap(U, year(2024)) == 1.0
    assert pair_overlap(U, U) == 1.0
