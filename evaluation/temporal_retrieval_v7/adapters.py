"""Adapters connecting V1 production primitives to V7 TimeRange semantics.

V7 reuses the production planner LLM (DNF AST) and extractor LLM (single
half-open envelope per mention). The adapters convert their outputs to
`list[TimeRange]`:

- `extractor_to_doc_refs`: each envelope → single-interval TimeRange.
  Optionally bounds doc-side "since X" / "after Y" surfaces; for V1 the
  extractor doesn't carry direction so this is a no-op (the V1 extractor
  has always emitted closed intervals).

- `plan_to_query_refs`: walk the DNF AST, evaluating each clause via
  range composition (intersect leaf ranges). Empty (impossible)
  clauses get split into per-leaf refs so they still contribute under
  additive cross-ref scoring (matches the SPEC §6.2 incompat rule's
  safety net).
"""
from __future__ import annotations

from collections.abc import Callable

from temporal_retrieval_min.core import Interval as V1Interval
from temporal_retrieval_min.planner import Constraint, QueryPlan

from .time_range import (
    NEG_INF,
    POS_INF,
    Interval,
    TimeRange,
    complement,
    intersect_all,
    is_empty,
)


# ---------------------------------------------------------------------------
# Doc side
# ---------------------------------------------------------------------------


def extractor_to_doc_refs(ivs: list[V1Interval]) -> list[TimeRange]:
    """Convert extractor envelopes → one TimeRange per envelope.

    Each envelope is already a half-open `[earliest_us, latest_us)`.
    Wrap each as a single-interval TimeRange (preserves the per-mention
    granularity expected by per-doc-ref scoring).
    """
    out: list[TimeRange] = []
    for iv in ivs:
        if iv.latest_us > iv.earliest_us:
            out.append(TimeRange(intervals=(Interval(iv.earliest_us, iv.latest_us),)))
    return out


# ---------------------------------------------------------------------------
# Query side
# ---------------------------------------------------------------------------


# Type alias: leaf phrase resolver → list of envelope intervals
LeafResolver = Callable[[Constraint], list[V1Interval]]


def leaf_to_range(
    leaf: Constraint,
    resolver: LeafResolver,
) -> TimeRange:
    """Convert one DNF leaf to its TimeRange.

    Relations:
    - intersect → range of resolved anchor (union of all returned envelopes)
    - disjoint  → complement of resolved anchor union
    - after     → [resolved.end_max, +∞)
    - before    → (-∞, resolved.start_min)

    If the resolver returns no intervals (extractor skipped — bare period
    word, anaphoric mention with no calendar anchor), the leaf is a
    "trust the extractor" no-op and contributes the universal range
    (matches V1's behavior of multiplying by 1.0).
    """
    anchor_ivs = resolver(leaf)
    if not anchor_ivs:
        return TimeRange.universal()

    anchor = _v1_envelopes_to_range(anchor_ivs)

    rel = leaf.relation
    if rel == "intersect":
        return anchor
    if rel == "disjoint":
        return complement(anchor)
    if rel == "after":
        # [latest end, +∞)
        end_max = max(iv.latest_us for iv in anchor_ivs)
        return TimeRange.closed(end_max, POS_INF)
    if rel == "before":
        # (-∞, earliest start)
        start_min = min(iv.earliest_us for iv in anchor_ivs)
        return TimeRange.before(start_min)
    raise ValueError(f"Unknown relation: {rel}")


def _v1_envelopes_to_range(ivs: list[V1Interval]) -> TimeRange:
    """Union of V1 envelopes → one canonical TimeRange."""
    out_ivs = []
    for iv in ivs:
        if iv.latest_us > iv.earliest_us:
            out_ivs.append(Interval(iv.earliest_us, iv.latest_us))
    return TimeRange.from_intervals(out_ivs)


def plan_to_query_refs(
    plan: QueryPlan,
    resolver: LeafResolver,
) -> list[TimeRange]:
    """Walk the DNF AST → flat list of TimeRange refs.

    For each outer OR-clause:
      1. Convert each leaf to a TimeRange (via leaf_to_range).
      2. Intersect non-universal leaf ranges (range composition).
      3. If the composed range is non-empty → emit ONE ref.
      4. If empty (incompatible AND-conjuncts the planner emitted as a
         single clause, e.g., "in 2020 and 2024") → emit one ref PER
         non-empty leaf range (graded coverage via cross-ref mean).

    The flat-list scoring (mean of per-ref bests) absorbs both outer
    OR and incompatible-AND uniformly: more refs satisfied → higher
    score.
    """
    refs: list[TimeRange] = []
    if not plan.expr:
        return refs

    for ast_clause in plan.expr:
        leaf_ranges = [leaf_to_range(leaf, resolver) for leaf in ast_clause]
        if not leaf_ranges:
            continue

        non_universal = [r for r in leaf_ranges if not _is_universal(r)]
        if not non_universal:
            continue

        composed = intersect_all(non_universal)
        if not is_empty(composed):
            refs.append(composed)
        else:
            for r in non_universal:
                if not is_empty(r):
                    refs.append(r)
    return refs


def _is_universal(r: TimeRange) -> bool:
    if len(r.intervals) != 1:
        return False
    iv = r.intervals[0]
    return iv.earliest_us <= NEG_INF + 1 and iv.latest_us >= POS_INF - 1
