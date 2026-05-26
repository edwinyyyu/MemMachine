"""V6 probe: range-composition + existential-overlap semantics.

Tests the architectural correction: instead of per-leaf factor aggregation
(current `evaluate_dnf_match`), compose the AND-clause's leaf-ranges into
an intersected range, compose the OR-of-clauses into a union, then check
existential overlap with the doc's anchor set.

Comparisons (within this probe):
- V1: current strict (universal-anchor-outside-excluded for disjoint;
      kill on any anchor inside)
- V5: flat existential per leaf (∃ anchor satisfies each leaf
      independently)
- V6: range composition + existential overlap (∃ single anchor satisfies
      the COMPOSED range of the full clause)

Three scenarios:
1. Engagement (the originating case): "outside summer 2024" + doc with
   [summer 2024, October 2024] anchors. V1 kills (wrong), V5 passes
   (right because October exists), V6 passes (correct: composed range
   is complement([summer 2024]); October ∈ range).

2. Compound disjoint: "not in 2020 or 2022" (per user's example)
   = AND-clause [disjoint(2020), disjoint(2022)]. Doc with
   [March 2020, May 2022] should NOT match (neither anchor satisfies
   the conjunction). V5 surfaces it (false positive — different
   anchors satisfy different leaves). V6 correctly kills.

3. Compound intersect+disjoint: "in 2024 not in summer" + doc with
   [July 2024, Jan 2025]. Same false-positive risk for V5.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_range_composition
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from temporal_retrieval.core import (
    constraint_factor_for_doc,
    excluded_containment,
    flatten_intervals,
)
from temporal_retrieval.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval.planner import QueryPlanner
from temporal_retrieval.schema import parse_iso

from ._common import make_embed_fn, setup_env

setup_env()


# TimeRange = sorted list of disjoint (lo, hi) tuples in microseconds.
# -∞ and +∞ are represented by large sentinels.
NEG_INF = -(2**62)
POS_INF = (2**62)


def as_range(intervals) -> List[Tuple[int, int]]:
    """Canonicalize a list of intervals to disjoint sorted (lo, hi) tuples."""
    if not intervals:
        return []
    sorted_ivs = sorted([(i.earliest_us, i.latest_us) for i in intervals])
    merged = []
    for lo, hi in sorted_ivs:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def complement(rng: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Complement of a range over (-∞, +∞).

    Uses exclusive boundaries: complement of [lo, hi] is (-∞, lo-1) ∪
    (hi+1, +∞). This avoids the boundary-touch false-positive where a
    doc anchor starting exactly at `lo` would otherwise be considered
    inside both the original range and its complement.
    """
    if not rng:
        return [(NEG_INF, POS_INF)]
    out = []
    prev = NEG_INF
    for lo, hi in rng:
        if prev < lo:
            out.append((prev, lo - 1))
        prev = hi + 1
    if prev < POS_INF:
        out.append((prev, POS_INF))
    return out


def intersect_ranges(r1, r2) -> List[Tuple[int, int]]:
    """Intersection of two TimeRanges."""
    out = []
    i = j = 0
    while i < len(r1) and j < len(r2):
        a_lo, a_hi = r1[i]
        b_lo, b_hi = r2[j]
        lo = max(a_lo, b_lo)
        hi = min(a_hi, b_hi)
        if lo < hi:
            out.append((lo, hi))
        if a_hi < b_hi:
            i += 1
        else:
            j += 1
    return out


def union_ranges(r1, r2) -> List[Tuple[int, int]]:
    """Union of two TimeRanges."""
    return as_range_from_pairs(r1 + r2)


def as_range_from_pairs(pairs):
    if not pairs:
        return []
    sorted_p = sorted(pairs)
    merged = []
    for lo, hi in sorted_p:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def intersect_all(ranges):
    if not ranges:
        return [(NEG_INF, POS_INF)]
    out = ranges[0]
    for r in ranges[1:]:
        out = intersect_ranges(out, r)
        if not out:
            return []
    return out


def union_all(ranges):
    if not ranges:
        return []
    out = []
    for r in ranges:
        out = union_ranges(out, r)
    return out


def overlap_existential(doc_intervals, query_range) -> bool:
    """∃ anchor in doc_intervals overlapping any interval in query_range."""
    if not query_range:
        return True  # no constraint
    if not doc_intervals:
        return False
    for di in doc_intervals:
        d_lo, d_hi = di.earliest_us, di.latest_us
        for q_lo, q_hi in query_range:
            if d_lo <= q_hi and q_lo <= d_hi:
                return True
    return False


def overlap_size(doc_intervals, query_range) -> float:
    """Sum of overlap durations across all (anchor, query-interval) pairs."""
    if not query_range or not doc_intervals:
        return 0.0
    total = 0
    for di in doc_intervals:
        d_lo, d_hi = di.earliest_us, di.latest_us
        for q_lo, q_hi in query_range:
            lo = max(d_lo, q_lo)
            hi = min(d_hi, q_hi)
            if lo < hi:
                total += (hi - lo)
    return float(total)


def eval_v6_range(plan, resolver):
    """Compose the plan into a single TimeRange via range arithmetic.

    AND-clause = intersection of leaf ranges.
    OR-of-clauses = union of clause ranges.
    """
    if not plan.expr:
        return [(NEG_INF, POS_INF)]
    clause_ranges = []
    for ci, clause in enumerate(plan.expr):
        leaf_ranges = []
        for li, leaf in enumerate(clause):
            anchor_ivs = resolver(ci, li, leaf)
            if not anchor_ivs:
                leaf_ranges.append([(NEG_INF, POS_INF)])
                continue
            if leaf.relation == "intersect":
                leaf_ranges.append(as_range(anchor_ivs))
            elif leaf.relation == "disjoint":
                leaf_ranges.append(complement(as_range(anchor_ivs)))
            elif leaf.relation == "after":
                a_max = max(a.latest_us for a in anchor_ivs)
                leaf_ranges.append([(a_max + 1, POS_INF)])
            elif leaf.relation == "before":
                a_min = min(a.earliest_us for a in anchor_ivs)
                leaf_ranges.append([(NEG_INF, a_min - 1)])
            else:
                leaf_ranges.append([(NEG_INF, POS_INF)])
        clause_ranges.append(intersect_all(leaf_ranges))
    return union_all(clause_ranges)


# Compound-AND scenarios — direct test of same-anchor binding.
COMPOUND_SCENARIOS = [
    {
        "name": "AND of two disjoints — engagement",
        "query_text": "not in 2020 or 2022",  # surface; planner may emit differently
        "ref_time": "2025-01-01",
        # Manually-constructed plan to test the AND semantics directly,
        # bypassing planner LLM behavior:
        "manual_plan_leaves": [
            ("2020", "disjoint"),
            ("2022", "disjoint"),
        ],
        "docs": [
            ("d_mar2020_oct2023", "March 2020 trip; October 2023 reorg",
             "2024-01-01",
             # GOLD: has October 2023 outside both 2020 and 2022
             "G"),
            ("d_mar2020_may2022", "March 2020 trip; May 2022 conference",
             "2023-01-01",
             # NOT-GOLD: no anchor outside both
             "I"),
            ("d_oct2023_only", "October 2023 reorg",
             "2024-01-01",
             "G"),  # clean outside
        ],
    },
    {
        "name": "AND of intersect + disjoint — in X not in Y",
        "query_text": "in 2024 not in summer 2024",
        "ref_time": "2024-12-15",
        "manual_plan_leaves": [
            ("2024", "intersect"),
            ("summer 2024", "disjoint"),
        ],
        "docs": [
            ("d_jul_jan", "July 2024 vacation; January 2025 reorg",
             "2024-08-01",
             # NOT-GOLD: July is in summer, Jan 2025 is not in 2024
             # No single anchor in [2024] ∩ complement([summer 2024])
             "I"),
            ("d_oct2024", "October 2024 launch",
             "2024-10-15",
             # GOLD: October is in 2024 and outside summer
             "G"),
            ("d_jul2024_only", "Summer 2024 vacation in Greece",
             "2024-07-20",
             # NOT-GOLD: only in summer 2024
             "I"),
        ],
    },
]


async def main():
    print("Loading embed + extractor...", flush=True)
    embed_fn = await make_embed_fn()
    extractor = TemporalExtractorV3_3()

    print(f"\nProbing {len(COMPOUND_SCENARIOS)} compound scenarios "
          f"(V1 strict vs V5 flat-existential vs V6 range-composition)\n")

    for sc in COMPOUND_SCENARIOS:
        print("=" * 70)
        print(f"Scenario: {sc['name']}")
        print(f"  Query: {sc['query_text']}")
        print(f"  Plan-leaves (manual): {sc['manual_plan_leaves']}")

        # Resolve anchors for each manual leaf
        ref_dt = parse_iso(sc["ref_time"])
        leaf_anchors = []
        for phrase, _rel in sc["manual_plan_leaves"]:
            envs = await extractor.extract(phrase, ref_dt)
            leaf_anchors.append(flatten_intervals(envs))

        # Extract doc anchors
        doc_anchors = {}
        for did, text, ref, _role in sc["docs"]:
            envs = await extractor.extract(text, parse_iso(ref))
            doc_anchors[did] = flatten_intervals(envs)
        extractor.save_caches()

        # Build a fake plan object for V6 evaluator (using just the
        # leaf list as a single AND-clause; OR-of-multi-clauses tested
        # elsewhere)
        from dataclasses import dataclass as _dc

        @_dc
        class FakeLeaf:
            phrase: str
            relation: str

        @_dc
        class FakePlan:
            expr: list

        leaves = [FakeLeaf(phrase=p, relation=r) for p, r in sc["manual_plan_leaves"]]
        plan = FakePlan(expr=[leaves])  # single AND-clause

        def resolver(ci, li, _leaf):
            return leaf_anchors[li]

        # V6 range composition
        v6_range = eval_v6_range(plan, resolver)
        # Show what the composed range looks like (in microseconds — too
        # large to print, but show count + endpoints)
        print(f"  V6 composed range: {len(v6_range)} disjoint intervals")
        if len(v6_range) <= 4:
            for lo, hi in v6_range:
                from datetime import datetime, timezone
                lo_s = (datetime.fromtimestamp(lo / 1e6, tz=timezone.utc).isoformat()
                        if abs(lo) < 1e18 else f"{lo}")
                hi_s = (datetime.fromtimestamp(hi / 1e6, tz=timezone.utc).isoformat()
                        if abs(hi) < 1e18 else f"{hi}")
                print(f"    [{lo_s} .. {hi_s}]")

        print()
        print(f"  {'doc':35s} {'role':5s} {'V1':>5s} {'V5':>5s} {'V6':>5s}")
        for did, text, ref, role in sc["docs"]:
            d_ivs = doc_anchors[did]

            # V1 strict (current): filter kill on any anchor in disjoint;
            # AND-combine via leaf factor min
            v1_pass = True
            for (phrase, rel), anchor_ivs in zip(
                    sc["manual_plan_leaves"], leaf_anchors, strict=False):
                if rel == "disjoint":
                    cont = excluded_containment(d_ivs, anchor_ivs)
                    if cont >= 1.0:
                        v1_pass = False
                        break
                elif rel == "intersect":
                    if constraint_factor_for_doc(d_ivs, anchor_ivs, "intersect") < 1.0:
                        v1_pass = False
                        break

            # V5 flat existential per leaf
            v5_pass = True
            for (phrase, rel), anchor_ivs in zip(
                    sc["manual_plan_leaves"], leaf_anchors, strict=False):
                if rel == "disjoint":
                    # ∃ doc anchor outside this excluded anchor
                    has_outside = False
                    if not d_ivs:
                        has_outside = True
                    else:
                        for di in d_ivs:
                            inside_any = False
                            for ai in anchor_ivs:
                                if (di.earliest_us <= ai.latest_us
                                    and ai.earliest_us <= di.latest_us):
                                    inside_any = True
                                    break
                            if not inside_any:
                                has_outside = True
                                break
                    if not has_outside:
                        v5_pass = False
                        break
                elif rel == "intersect":
                    # ∃ doc anchor overlapping this anchor
                    if constraint_factor_for_doc(d_ivs, anchor_ivs, "intersect") < 1.0:
                        v5_pass = False
                        break

            # V6 range composition: ∃ doc anchor in v6_range
            v6_pass = overlap_existential(d_ivs, v6_range)

            def fmt(b): return "PASS" if b else "KILL"
            expected = "PASS" if role == "G" else "KILL"
            v1_mark = "✓" if (fmt(v1_pass) == expected) else "✗"
            v5_mark = "✓" if (fmt(v5_pass) == expected) else "✗"
            v6_mark = "✓" if (fmt(v6_pass) == expected) else "✗"
            print(f"  {did:35s} {role:5s} "
                  f"{fmt(v1_pass)}{v1_mark} {fmt(v5_pass)}{v5_mark} "
                  f"{fmt(v6_pass)}{v6_mark}")
            print(f"    extracted: {[(i.earliest_us, i.latest_us) for i in d_ivs[:3]]}{'...' if len(d_ivs) > 3 else ''}")

        print()


if __name__ == "__main__":
    asyncio.run(main())
