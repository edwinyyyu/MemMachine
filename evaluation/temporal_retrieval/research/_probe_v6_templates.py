"""V6 range-composition + list-of-templates probe on the 6 engagement
scenarios + 2 compound-AND scenarios.

Compares:
- V1 (production): per-leaf factor min/max, strict disjoint exclusion filter
- V6_templates (architectural recipe): list of templates, each evaluated
  to a TimeRange; per-template best-anchor overlap; sum across templates

V6 details:
- TimeRange = sorted list of half-open intervals [lo, hi)
- complement: half-open, no ±1us hacks
- overlap_existential: strict `<` comparisons (half-open)
- Per-template: max over doc.anchors of overlap_quality
- Per-doc: sum over templates of per-template score

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_v6_templates
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from temporal_retrieval.core import (
    build_pool,
    constraint_factor_for_doc,
    doc_passes_filter,
    excluded_containment,
    flatten_intervals,
    normalize_dict,
    normalize_rerank_full,
)
from temporal_retrieval.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval.planner import QueryPlanner
from temporal_retrieval.schema import parse_iso

from ._common import make_embed_fn, setup_env

setup_env()


NEG_INF = -(2**62)
POS_INF = (2**62)


def as_range(intervals) -> List[Tuple[int, int]]:
    """Canonicalize intervals to half-open [lo, hi) sorted disjoint list."""
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


def complement(rng) -> List[Tuple[int, int]]:
    """Half-open complement: complement of [lo, hi) is (-∞, lo) ∪ [hi, +∞).

    No ±1us hacks needed because the original `hi` is exclusive — `[hi, +∞)`
    starts at the same numeric value but `hi` itself is NOT in the original
    range. So under half-open semantics with strict `<` overlap check, the
    boundary is unambiguous.
    """
    if not rng:
        return [(NEG_INF, POS_INF)]
    out = []
    prev = NEG_INF
    for lo, hi in rng:
        if prev < lo:
            out.append((prev, lo))
        prev = hi
    if prev < POS_INF:
        out.append((prev, POS_INF))
    return out


def intersect_two(r1, r2) -> List[Tuple[int, int]]:
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


def union_two(r1, r2) -> List[Tuple[int, int]]:
    pairs = r1 + r2
    if not pairs:
        return []
    pairs = sorted(pairs)
    merged = []
    for lo, hi in pairs:
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
        out = intersect_two(out, r)
        if not out:
            return []
    return out


def eval_leaf_range(leaf, anchor_ivs):
    """Convert one leaf's (relation, anchor_ivs) to a TimeRange."""
    if not anchor_ivs:
        return [(NEG_INF, POS_INF)]  # unresolvable anchor = no constraint
    if leaf.relation == "intersect":
        return as_range(anchor_ivs)
    elif leaf.relation == "disjoint":
        return complement(as_range(anchor_ivs))
    elif leaf.relation == "after":
        a_max = max(a.latest_us for a in anchor_ivs)
        return [(a_max, POS_INF)]
    elif leaf.relation == "before":
        a_min = min(a.earliest_us for a in anchor_ivs)
        return [(NEG_INF, a_min)]
    return [(NEG_INF, POS_INF)]


def eval_template_range(clause, leaf_anchors_per_ci_li, ci):
    """Each clause is one template — compose its leaves via AND (intersect)."""
    leaf_ranges = []
    for li, leaf in enumerate(clause):
        leaf_ranges.append(eval_leaf_range(leaf, leaf_anchors_per_ci_li.get((ci, li), [])))
    return intersect_all(leaf_ranges)


def overlap_quality(doc_intervals, query_range) -> float:
    """Best-anchor overlap quality: max over anchors of frac_min-style ratio.

    For each (anchor, range-interval) pair: |anchor ∩ q| / min(|anchor|, |q|),
    clipped to [0, 1]. Use strict `<` for half-open overlap.
    """
    if not query_range or not doc_intervals:
        return 0.0
    best = 0.0
    for di in doc_intervals:
        d_lo, d_hi = di.earliest_us, di.latest_us
        d_w = d_hi - d_lo
        if d_w <= 0:
            d_w = 1
        for q_lo, q_hi in query_range:
            if d_lo < q_hi and q_lo < d_hi:  # half-open overlap
                inter = min(d_hi, q_hi) - max(d_lo, q_lo)
                q_w = q_hi - q_lo
                if q_w <= 0:
                    q_w = 1
                denom = min(d_w, q_w)
                f = min(1.0, max(0.0, inter / denom))
                if f > best:
                    best = f
    return best


def eval_v6_score(plan, leaf_anchors, doc_intervals) -> float:
    """List-of-templates aggregation: sum over templates of per-template
    best-anchor overlap quality."""
    if not plan.expr:
        return 1.0  # no temporal constraint
    total = 0.0
    for ci, clause in enumerate(plan.expr):
        template_range = eval_template_range(clause, leaf_anchors, ci)
        if not template_range:
            continue  # empty template = unsatisfiable (planner should never emit)
        total += overlap_quality(doc_intervals, template_range)
    return total


# Engagement + compound-AND probe scenarios.
SCENARIOS = [
    # Engagement scenarios (from original probe)
    {
        "query": "What did I do outside summer 2024?",
        "ref_time": "2024-12-15",
        "docs": [
            ("d_g", "Unlike summer 2024 when I was traveling, in October 2024 I focused on a quiet writing project at home.", "2024-10-15", "G"),
            ("d_i", "Summer 2024 vacation in Greece: visited Crete and Athens for two weeks in July.", "2024-07-20", "I"),
            ("d_ot", "Started outlining a writing project in November 2024.", "2024-11-10", "OT"),
            ("d_un", "Annual dentist visit in March 2024.", "2024-03-05", "UN"),
        ],
    },
    {
        "query": "What did I work on in 2024 not in Q1?",
        "ref_time": "2025-02-01",
        "docs": [
            ("d_g", "Q1 2024 was all migration work; the rest of 2024 I shifted to the new analytics dashboard, shipping the first version in May.", "2024-05-20", "G"),
            ("d_i", "Database migration sprint in February 2024.", "2024-02-15", "I"),
            ("d_ot", "Built a small internal tool in August 2024.", "2024-08-10", "OT"),
            ("d_un", "Reviewed insurance options in October 2024.", "2024-10-05", "UN"),
        ],
    },
    {
        "query": "Meetings excluding Q4 2023.",
        "ref_time": "2024-06-01",
        "docs": [
            ("d_g", "Aside from the Q4 2023 budget reviews which dominated that quarter, my main meeting in 2023 was the May leadership sync.", "2023-05-12", "G"),
            ("d_i", "Q4 2023 budget review meeting with finance leads.", "2023-10-20", "I"),
            ("d_ot", "1:1 with mentor in March 2024.", "2024-03-15", "OT"),
            ("d_un", "Birthday lunch in February 2024.", "2024-02-22", "UN"),
        ],
    },
]


async def main():
    print("Loading embed + extractor + planner...", flush=True)
    embed_fn = await make_embed_fn()
    extractor = TemporalExtractorV3_3()
    planner = QueryPlanner()

    print(f"\nProbing {len(SCENARIOS)} scenarios (V1 strict vs V6 templates)\n")

    for sc in SCENARIOS:
        print("=" * 70)
        q_text = sc["query"]
        q_ref = sc["ref_time"]
        docs = sc["docs"]
        roles = {d[0]: d[3] for d in docs}

        print(f"Q: {q_text}")
        plan = await planner.plan(q_text, q_ref)
        plan_repr = [[(l.phrase, l.relation) for l in c] for c in plan.expr]
        print(f"Plan: {plan_repr}")

        # Resolve doc intervals
        doc_ivs = {}
        for did, dtext, dref, _role in docs:
            envs = await extractor.extract(dtext, parse_iso(dref))
            doc_ivs[did] = flatten_intervals(envs)

        # Resolve leaf anchors
        leaves_flat = [(ci, li, leaf)
                       for ci, clause in enumerate(plan.expr)
                       for li, leaf in enumerate(clause)]
        leaf_anchors = {}
        if leaves_flat:
            ref_dt = parse_iso(q_ref)
            res = await asyncio.gather(
                *(extractor.extract(leaf.phrase, ref_dt)
                  for _, _, leaf in leaves_flat))
            for (ci, li, _l), envs in zip(leaves_flat, res, strict=False):
                leaf_anchors[(ci, li)] = flatten_intervals(envs)

        # V6 template ranges
        for ci in range(len(plan.expr)):
            t_range = eval_template_range(plan.expr[ci], leaf_anchors, ci)
            from datetime import datetime, timezone
            n = len(t_range)
            print(f"  Template {ci} range ({n} intervals):")
            for lo, hi in t_range[:4]:
                lo_s = (datetime.fromtimestamp(lo / 1e6, tz=timezone.utc).isoformat()
                        if abs(lo) < 1e18 else "-∞")
                hi_s = (datetime.fromtimestamp(hi / 1e6, tz=timezone.utc).isoformat()
                        if abs(hi) < 1e18 else "+∞")
                print(f"    [{lo_s} .. {hi_s})")

        # Semantic for ranking
        q_emb = np.asarray((await embed_fn([q_text]))[0], dtype=np.float32)
        d_embs = await embed_fn([d[1] for d in docs])
        sem = {}
        qn = float(np.linalg.norm(q_emb)) or 1e-9
        for did_doc, e in zip(docs, d_embs, strict=False):
            did = did_doc[0]
            de = np.asarray(e, dtype=np.float32)
            dn = float(np.linalg.norm(de)) or 1e-9
            sem[did] = float(np.dot(q_emb, de) / (qn * dn))

        # V1 strict: doc_passes_filter + strict disjoint + frac_min intersect
        valid_includes_v1 = []
        valid_excludes_v1 = []
        for ci, li, leaf in leaves_flat:
            ivs = leaf_anchors.get((ci, li), [])
            if not ivs:
                continue
            if leaf.relation == "disjoint":
                valid_excludes_v1.append(ivs)
            else:
                valid_includes_v1.append((leaf.relation, ivs))

        print(f"\n  {'doc':35s} {'role':4s} {'V1 score':>10s} {'V6 score':>10s} {'sem':>6s}")
        for did, dtext, dref, role in docs:
            d_ivs = doc_ivs[did]

            # V1 score (filter pass + per-leaf factor min)
            v1_pass = doc_passes_filter(d_ivs, valid_includes_v1, valid_excludes_v1)
            if v1_pass and plan.expr:
                # per-leaf factor with strict disjoint and binary intersect
                leaf_factors = []
                for ci, li, leaf in leaves_flat:
                    a_ivs = leaf_anchors.get((ci, li), [])
                    if not a_ivs:
                        leaf_factors.append(1.0)
                        continue
                    if leaf.relation == "disjoint":
                        cont = excluded_containment(d_ivs, a_ivs)
                        leaf_factors.append(max(0.0, 1.0 - cont))
                    elif leaf.relation == "intersect":
                        leaf_factors.append(
                            constraint_factor_for_doc(d_ivs, a_ivs, "intersect"))
                    else:
                        leaf_factors.append(
                            constraint_factor_for_doc(d_ivs, a_ivs, leaf.relation))
                v1_score = min(leaf_factors) if leaf_factors else 1.0
            elif v1_pass:
                v1_score = 1.0
            else:
                v1_score = 0.0

            v6_score = eval_v6_score(plan, leaf_anchors, d_ivs)

            print(f"  {did:35s} {role:4s} {v1_score:>10.3f} {v6_score:>10.3f} {sem[did]:>6.3f}")

        extractor.save_caches()
        print()


if __name__ == "__main__":
    asyncio.run(main())
