"""Tests for the planner / extractor adapters."""
from __future__ import annotations

from datetime import datetime, timezone

from temporal_retrieval_min.core import Interval as V1Interval
from temporal_retrieval_min.planner import Constraint, QueryPlan

from temporal_retrieval_v7 import POS_INF, intersect, is_empty
from temporal_retrieval_v7.adapters import (
    extractor_to_doc_refs,
    leaf_to_range,
    plan_to_query_refs,
)


def us(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1_000_000)


def _mar2024_v1() -> list[V1Interval]:
    return [V1Interval(us(2024, 3, 1), us(2024, 4, 1))]


def _summer2024_v1() -> list[V1Interval]:
    return [V1Interval(us(2024, 6, 1), us(2024, 9, 1))]


def _y2024_v1() -> list[V1Interval]:
    return [V1Interval(us(2024, 1, 1), us(2025, 1, 1))]


def test_extractor_to_doc_refs_basic() -> None:
    envs = [V1Interval(us(2024, 1, 1), us(2024, 2, 1)), V1Interval(us(2024, 5, 1), us(2024, 6, 1))]
    refs = extractor_to_doc_refs(envs)
    assert len(refs) == 2
    assert refs[0].intervals[0].earliest_us == us(2024, 1, 1)
    assert refs[1].intervals[0].earliest_us == us(2024, 5, 1)


def test_leaf_to_range_intersect() -> None:
    leaf = Constraint(phrase="March 2024", relation="intersect")
    resolver = lambda c: _mar2024_v1() if c.phrase == "March 2024" else []
    r = leaf_to_range(leaf, resolver)
    assert r.intervals[0].earliest_us == us(2024, 3, 1)
    assert r.intervals[0].latest_us == us(2024, 4, 1)


def test_leaf_to_range_disjoint() -> None:
    leaf = Constraint(phrase="summer 2024", relation="disjoint")
    resolver = lambda c: _summer2024_v1()
    r = leaf_to_range(leaf, resolver)
    # complement of summer = (-∞, Jun 1 2024) ∪ [Sep 1 2024, +∞)
    assert len(r.intervals) == 2


def test_leaf_to_range_after() -> None:
    leaf = Constraint(phrase="2020", relation="after")
    resolver = lambda c: [V1Interval(us(2020, 1, 1), us(2021, 1, 1))]
    r = leaf_to_range(leaf, resolver)
    # [end of 2020 range, +∞)
    assert r.intervals[0].earliest_us == us(2021, 1, 1)
    assert r.intervals[0].latest_us >= POS_INF - 1


def test_leaf_to_range_before() -> None:
    leaf = Constraint(phrase="2020", relation="before")
    resolver = lambda c: [V1Interval(us(2020, 1, 1), us(2021, 1, 1))]
    r = leaf_to_range(leaf, resolver)
    # (-∞, start of 2020 range)
    assert r.intervals[0].latest_us == us(2020, 1, 1)


def test_leaf_to_range_unresolved_phrase_is_universal() -> None:
    """Bare 'March' (no year) → extractor skips → empty resolver →
    leaf contributes universal range (the V1 trust-extractor behavior)."""
    leaf = Constraint(phrase="March", relation="intersect")
    resolver = lambda c: []  # extractor skipped
    r = leaf_to_range(leaf, resolver)
    assert r.intervals[0].earliest_us < 0
    assert r.intervals[0].latest_us >= POS_INF - 1


def test_plan_to_query_refs_compound_clause() -> None:
    """Query 'in 2024 not in summer' → one composed range."""
    plan = QueryPlan(
        expr=[[
            Constraint(phrase="2024", relation="intersect"),
            Constraint(phrase="summer 2024", relation="disjoint"),
        ]],
    )

    def resolver(c: Constraint) -> list[V1Interval]:
        if c.phrase == "2024":
            return _y2024_v1()
        if c.phrase == "summer 2024":
            return _summer2024_v1()
        return []

    refs = plan_to_query_refs(plan, resolver)
    assert len(refs) == 1
    # The composed range should NOT include summer 2024
    composed = refs[0]
    from temporal_retrieval_v7 import TimeRange
    summer_only = TimeRange.closed(us(2024, 6, 1), us(2024, 9, 1))
    inter = intersect(composed, summer_only)
    assert is_empty(inter)


def test_plan_to_query_refs_or_clauses() -> None:
    """Query 'in Q1 2023 or Q4 2023' → two refs (one per outer clause)."""
    plan = QueryPlan(
        expr=[
            [Constraint(phrase="Q1 2023", relation="intersect")],
            [Constraint(phrase="Q4 2023", relation="intersect")],
        ],
    )

    def resolver(c: Constraint) -> list[V1Interval]:
        if c.phrase == "Q1 2023":
            return [V1Interval(us(2023, 1, 1), us(2023, 4, 1))]
        if c.phrase == "Q4 2023":
            return [V1Interval(us(2023, 10, 1), us(2024, 1, 1))]
        return []

    refs = plan_to_query_refs(plan, resolver)
    assert len(refs) == 2


def test_plan_to_query_refs_incompatible_split() -> None:
    """Planner emits 'in 2020 and 2024' as a single AND clause → composer
    detects empty intersection → emits one ref PER non-empty leaf range
    (graded coverage via cross-ref mean)."""
    plan = QueryPlan(
        expr=[[
            Constraint(phrase="2020", relation="intersect"),
            Constraint(phrase="2024", relation="intersect"),
        ]],
    )

    def resolver(c: Constraint) -> list[V1Interval]:
        if c.phrase == "2020":
            return [V1Interval(us(2020, 1, 1), us(2021, 1, 1))]
        if c.phrase == "2024":
            return [V1Interval(us(2024, 1, 1), us(2025, 1, 1))]
        return []

    refs = plan_to_query_refs(plan, resolver)
    assert len(refs) == 2


def test_final_score_graded_coverage() -> None:
    """Flat-list scoring rewards graded coverage. Doc matching all refs
    → 1.0; doc matching half → 0.5; none → 0. Emergent from cross-ref
    mean — no separate 'and_incompat' machinery needed."""
    from temporal_retrieval_v7 import TimeRange, final_score

    y2020 = TimeRange.closed(us(2020, 1, 1), us(2021, 1, 1))
    y2024 = TimeRange.closed(us(2024, 1, 1), us(2025, 1, 1))
    refs = [y2020, y2024]

    mar2020 = TimeRange.closed(us(2020, 3, 1), us(2020, 4, 1))
    jun2024 = TimeRange.closed(us(2024, 6, 1), us(2024, 7, 1))

    assert final_score(refs, [mar2020, jun2024]) == 1.0
    assert final_score(refs, [mar2020]) == 0.5
    assert final_score(refs, [jun2024]) == 0.5
    assert final_score(refs, [TimeRange.closed(us(2022, 1, 1), us(2023, 1, 1))]) == 0.0


def test_plan_to_query_refs_empty_plan_returns_empty_refs() -> None:
    plan = QueryPlan(expr=[])
    refs = plan_to_query_refs(plan, lambda c: [])
    assert refs == []


def test_plan_to_query_refs_all_unresolved_drops_clause() -> None:
    """If every leaf in a clause is unresolved (extractor skipped), the
    clause adds no constraint → drop it (otherwise we'd add a universal
    ref that scores 1.0 on every doc and washes out other constraints)."""
    plan = QueryPlan(
        expr=[[Constraint(phrase="grad school", relation="intersect")]],
    )
    refs = plan_to_query_refs(plan, lambda c: [])
    assert refs == []
