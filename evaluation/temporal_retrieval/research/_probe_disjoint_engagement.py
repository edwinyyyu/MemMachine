"""Qualitative probe: when a query says "outside / not in X" and a doc
ENGAGES with X in contrast (mentions both X and the answer period),
does the current pipeline surface it? Or does the strict exclusion
filter / scoring eliminate the engagement-relevant doc?

This is the case the existing notin-style benches DON'T cover —
their gold docs are always squeaky-clean outside docs with no mention
of the excluded window. So the existing R@1/R@5 numbers can't speak
to the engagement question at all.

Build 6 hand-crafted scenarios. For each query:
  - GOLD: engagement-relevant doc (mentions excluded period in contrast)
  - DISTRACTOR_INSIDE: doc entirely inside the excluded period
  - DISTRACTOR_OUTSIDE_TOPICAL: clean outside doc on a related but
    weaker topic (current logic ranks these high; engagement-relevant
    should rank higher)
  - DISTRACTOR_UNRELATED: doc outside the excluded period but
    semantically unrelated

Score with three variants:
  - V1 STRICT (current): filter on disjoint (max-over-pairs strict),
    leaf factor = 1 - containment
  - V2 SOFT (aggregate): notin_aggregate=True (no filter kill, fractional
    leaf factor = total inside fraction)
  - V3 POLARITY-BLIND: disjoint treated as intersect — find docs that
    mention the date in any polarity; let cosine handle direction

Print ranking for each query + variant. Look for cases where V1 ranks
the engagement-relevant doc behind the topical-but-clean doc.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_disjoint_engagement
"""
from __future__ import annotations

import asyncio

import numpy as np

from temporal_retrieval import Doc
from temporal_retrieval.core import (
    build_pool,
    constraint_factor_for_doc,
    doc_passes_filter,
    excluded_containment,
    excluded_containment_aggregate,
    flatten_intervals,
    normalize_dict,
    normalize_rerank_full,
)
from temporal_retrieval.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval.planner import QueryPlanner, evaluate_dnf_match
from temporal_retrieval.schema import parse_iso

from ._common import make_embed_fn, setup_env

setup_env()


# Scenarios. Each is (query, ref_time, [(doc_id, doc_text, doc_ref, role)])
# Role: G=gold (engagement-relevant), I=distractor inside, OT=outside topical,
# UN=unrelated outside
SCENARIOS = [
    {
        "query": "What did I do outside summer 2024?",
        "ref_time": "2024-12-15",
        "docs": [
            # G: engagement-relevant — explicitly contrasts summer with the answer period
            ("d_g", "Unlike summer 2024 when I was traveling, in October 2024 I focused on a quiet writing project at home.", "2024-10-15", "G"),
            # I: pure inside the excluded window
            ("d_i", "Summer 2024 vacation in Greece: visited Crete and Athens for two weeks in July.", "2024-07-20", "I"),
            # OT: clean outside, topical (also a writing project)
            ("d_ot", "Started outlining a writing project in November 2024.", "2024-11-10", "OT"),
            # UN: clean outside, unrelated topic
            ("d_un", "Annual dentist visit in March 2024.", "2024-03-05", "UN"),
        ],
    },
    {
        "query": "What expenses did I have outside the holiday season (November–December)?",
        "ref_time": "2025-04-01",
        "docs": [
            ("d_g", "Most of my big spending in 2024 hit during the November–December holiday season; my one notable outlier was a March 2024 dental bill of $800.", "2024-12-30", "G"),
            ("d_i", "Black Friday shopping in November 2024: $1,200 on electronics.", "2024-11-29", "I"),
            ("d_ot", "Filed quarterly tax estimate in April 2024.", "2024-04-15", "OT"),
            ("d_un", "Bought a coffee subscription in February 2024.", "2024-02-10", "UN"),
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
            # G: contrastive — explicitly says "outside Q4" and names a non-Q4 meeting
            ("d_g", "Aside from the Q4 2023 budget reviews which dominated that quarter, my main meeting in 2023 was the May leadership sync.", "2023-05-12", "G"),
            ("d_i", "Q4 2023 budget review meeting with finance leads.", "2023-10-20", "I"),
            ("d_ot", "1:1 with mentor in March 2024.", "2024-03-15", "OT"),
            ("d_un", "Birthday lunch in February 2024.", "2024-02-22", "UN"),
        ],
    },
    {
        "query": "What classes did I take in 2024 excluding the spring semester?",
        "ref_time": "2025-01-15",
        "docs": [
            ("d_g", "While the spring 2024 semester accounted for most of my coursework, I also picked up a fall 2024 elective on data ethics.", "2024-10-08", "G"),
            ("d_i", "Spring 2024 semester: enrolled in five classes including macroeconomics.", "2024-04-15", "I"),
            ("d_ot", "Joined a weekend pottery workshop in fall 2024.", "2024-10-22", "OT"),
            ("d_un", "Library card renewal in March 2024.", "2024-03-01", "UN"),
        ],
    },
    {
        # RETROSPECTIVE case: doc written INSIDE excluded period but
        # content-anchors are OUTSIDE. Tests V4 (ref_time, will fail)
        # vs V5 (content-anchors, will pass).
        "query": "What did I do outside Q2 2024?",
        "ref_time": "2024-12-15",
        "docs": [
            # G: written DURING Q2 2024 but reviews 2023 events
            ("d_g_retro", "Now in May 2024 I'm reflecting on what I did last year: my Q3 2023 trip to Italy was the highlight, plus the December 2023 family reunion.", "2024-05-15", "G"),
            # I: written during Q2 2024, content is purely Q2 2024
            ("d_i_retro", "April 2024 onboarding sprint: rebuilt the analytics dashboard from scratch.", "2024-04-22", "I"),
            # OT: clean outside doc
            ("d_ot_retro", "Started a writing project in October 2024.", "2024-10-15", "OT"),
            ("d_un_retro", "Birthday dinner in August 2024.", "2024-08-22", "UN"),
        ],
    },
    {
        "query": "What did I do not in 2023?",
        "ref_time": "2025-01-01",
        "docs": [
            ("d_g", "Looking back, 2023 was the bottleneck year — I didn't take any real vacation. The big one happened in April 2024: two weeks in Portugal.", "2024-04-20", "G"),
            ("d_i", "Steady work routine throughout 2023.", "2023-08-15", "I"),
            ("d_ot", "Renovated the kitchen in June 2024.", "2024-06-12", "OT"),
            ("d_un", "Routine car inspection in November 2024.", "2024-11-04", "UN"),
        ],
    },
]


# Variants:
# V1: current strict exclusion (filter + scoring)
# V2: aggregate (no filter kill, fractional leaf)
# V3: polarity-blind (disjoint treated as intersect everywhere)

VARIANTS = [
    ("V1_strict_exclude",        "strict",          False),
    ("V2_aggregate_soft",        "aggregate",       False),
    ("V2_5_floor_aggregate",     "floor_aggregate", False),
    ("V3_polarity_blind",        "blind",           True),
    ("V4_ref_time",              "ref_time",        False),
    ("V5_existential_outside",   "existential",     False),
]


def make_cosine_rerank_fn(embed_fn):
    async def cosine_rerank(query, doc_texts):
        if not doc_texts:
            return []
        qe = (await embed_fn([query]))[0]
        des = await embed_fn(doc_texts)
        qn = float(np.linalg.norm(qe)) or 1e-9
        out = []
        for de in des:
            den = float(np.linalg.norm(de)) or 1e-9
            out.append(float(np.dot(qe, de) / (qn * den)))
        return out
    return cosine_rerank


def evaluate_disjoint_variant(plan, doc_ivs, leaf_anchor_resolver,
                              variant: str,
                              doc_ref_us: int | None = None) -> tuple[float, bool]:
    """Returns (match_score, passes_filter_for_disjoint).

    variant: "strict" — current; max-over-pairs containment; filter on full overlap
             "aggregate" — fractional containment; no filter kill
             "blind" — treat disjoint as intersect (positive signal)
             "ref_time" — V4: ignore extracted intervals; use only the doc's
                          ref_time (the primary anchor metadata). Doc passes
                          and gets factor=1.0 if ref_time is OUTSIDE the
                          excluded window; killed otherwise. Engagement docs
                          (written outside but mentioning excluded period in
                          contrast) survive cleanly.
    """
    if not plan.expr:
        return 1.0, True
    # We need to compute both the temporal-match score AND the
    # filter-pass decision (because polarity-blind changes the filter).
    leaf_factors = []
    passes = True
    for ci, clause in enumerate(plan.expr):
        for li, leaf in enumerate(clause):
            anchor_ivs = leaf_anchor_resolver(ci, li, leaf)
            if not anchor_ivs:
                leaf_factors.append(1.0)
                continue
            if leaf.relation == "disjoint":
                if variant == "strict":
                    cont = excluded_containment(doc_ivs, anchor_ivs)
                    f = max(0.0, 1.0 - cont)
                    if cont >= 1.0:
                        passes = False
                elif variant == "aggregate":
                    cont = excluded_containment_aggregate(doc_ivs, anchor_ivs)
                    f = max(0.0, 1.0 - cont)
                elif variant == "floor_aggregate":
                    # Engagement-aware: kill ONLY when ALL doc intervals are
                    # inside the excluded window (aggregate containment ≥ 1).
                    # Engagement-relevant docs (some intervals outside, some
                    # inside) get at least 0.5 — they have a non-excluded
                    # reference, which is the user's actual answer space.
                    agg_cont = excluded_containment_aggregate(doc_ivs, anchor_ivs)
                    if agg_cont >= 1.0:
                        f = 0.0
                        passes = False
                    else:
                        f = max(0.5, 1.0 - agg_cont)
                elif variant == "blind":
                    f = constraint_factor_for_doc(doc_ivs, anchor_ivs, "intersect")
                elif variant == "ref_time":
                    # Check if doc's ref_time is inside any excluded interval.
                    ref_inside = any(
                        ai.earliest_us <= doc_ref_us <= ai.latest_us
                        for ai in anchor_ivs
                    ) if doc_ref_us is not None else False
                    if ref_inside:
                        f = 0.0
                        passes = False
                    else:
                        f = 1.0
                elif variant == "existential":
                    # V5: pass + factor=1.0 if doc has ANY content-anchor
                    # outside this excluded window. Engagement docs
                    # (mention excluded period in contrast) pass cleanly
                    # because they have other anchors outside.
                    has_outside = False
                    if not doc_ivs:
                        has_outside = True
                    else:
                        for di in doc_ivs:
                            inside_any = False
                            for ai in anchor_ivs:
                                if (di.earliest_us <= ai.latest_us
                                    and ai.earliest_us <= di.latest_us):
                                    inside_any = True
                                    break
                            if not inside_any:
                                has_outside = True
                                break
                    if has_outside:
                        f = 1.0
                    else:
                        f = 0.0
                        passes = False
                else:
                    raise ValueError(variant)
            elif leaf.relation == "intersect":
                f = _intersect_min_norm(doc_ivs, anchor_ivs)
            else:
                f = constraint_factor_for_doc(doc_ivs, anchor_ivs, leaf.relation)
            leaf_factors.append(f)
    score = min(leaf_factors) if leaf_factors else 1.0
    return score, passes


def _intersect_min_norm(d_ivs, a_ivs) -> float:
    best = 0.0
    for di in d_ivs:
        d_w = di.latest_us - di.earliest_us
        if d_w <= 0:
            d_w = 1
        for ai in a_ivs:
            a_w = ai.latest_us - ai.earliest_us
            if a_w <= 0:
                a_w = 1
            lo = max(di.earliest_us, ai.earliest_us)
            hi = min(di.latest_us, ai.latest_us)
            inter = max(0, hi - lo)
            denom = min(d_w, a_w)
            f = min(1.0, inter / denom)
            if f > best:
                best = f
    return best


async def main():
    print("Loading embed...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    extractor = TemporalExtractorV3_3()
    planner = QueryPlanner()

    print(f"\nProbing {len(SCENARIOS)} scenarios × {len(VARIANTS)} variants\n")

    for sc in SCENARIOS:
        q_text = sc["query"]
        q_ref = sc["ref_time"]
        docs = sc["docs"]
        roles = {d[0]: d[3] for d in docs}

        print("=" * 70)
        print(f"Q: {q_text}")
        print(f"ref: {q_ref}")
        plan = await planner.plan(q_text, q_ref)
        print(f"Plan.expr: {[[(l.phrase, l.relation) for l in c] for c in plan.expr]}")

        # Extract doc intervals
        ref_dt_docs = {d[0]: parse_iso(d[2]) for d in docs}
        doc_envs = {}
        for did, dtext, dref, _role in docs:
            try:
                envs = await extractor.extract(dtext, parse_iso(dref))
            except Exception:
                envs = []
            doc_envs[did] = envs
        doc_ivs = {did: flatten_intervals(envs) for did, envs in doc_envs.items()}

        # Resolve leaf anchors
        leaves_flat = [
            (ci, li, leaf)
            for ci, clause in enumerate(plan.expr)
            for li, leaf in enumerate(clause)
        ]
        anchors = {}
        if leaves_flat:
            ref_dt = parse_iso(q_ref)
            res = await asyncio.gather(
                *(extractor.extract(leaf.phrase, ref_dt)
                  for _, _, leaf in leaves_flat)
            )
            for (ci, li, _l), envs in zip(leaves_flat, res, strict=False):
                anchors[(ci, li)] = flatten_intervals(envs)

        def resolver(ci, li, _leaf):
            return anchors.get((ci, li), [])

        # Embed query + docs
        q_emb = np.asarray((await embed_fn([q_text]))[0], dtype=np.float32)
        d_texts = [d[1] for d in docs]
        d_embs = await embed_fn(d_texts)
        d_embs = {docs[i][0]: np.asarray(e, dtype=np.float32)
                  for i, e in enumerate(d_embs)}
        sem = {}
        qn = float(np.linalg.norm(q_emb)) or 1e-9
        for did, e in d_embs.items():
            dn = float(np.linalg.norm(e)) or 1e-9
            sem[did] = float(np.dot(q_emb, e) / (qn * dn))

        all_dids = [d[0] for d in docs]
        # Map of doc_id -> ref_us (for V4 ref_time variant)
        from temporal_retrieval.schema import to_us as _to_us
        doc_ref_us = {d[0]: _to_us(parse_iso(d[2])) for d in docs}
        # rerank within full pool
        rerank_raw = await rerank_fn(q_text, d_texts)
        rerank_partial = dict(zip(all_dids, rerank_raw, strict=False))

        # Score each variant separately
        for vname, variant_kind, blind_filter in VARIANTS:
            # Adapt filter to variant
            valid_includes = []
            valid_excludes = []
            for ci, li, leaf in leaves_flat:
                ivs = anchors.get((ci, li), [])
                if not ivs:
                    continue
                if leaf.relation == "disjoint":
                    if variant_kind == "blind":
                        # treat as include intersect
                        valid_includes.append(("intersect", ivs))
                    elif variant_kind == "strict":
                        valid_excludes.append(ivs)
                    elif variant_kind == "aggregate":
                        # No filter kill in aggregate — also include? skip filter
                        # Aggregate variant: don't enforce strict filter
                        valid_excludes.append(ivs)  # filter still uses max-over-pairs
                        # We'll override the filter-kill behavior via the scoring path.
                    elif variant_kind == "ref_time":
                        # V4: filter via ref_time check (see custom filter below)
                        valid_excludes.append(ivs)
                    elif variant_kind == "existential":
                        # V5: filter via existential-outside-anchor check
                        valid_excludes.append(ivs)
                else:
                    valid_includes.append((leaf.relation, ivs))

            if variant_kind == "aggregate":
                # In aggregate mode we soften the filter: skip the exclude filter
                eligible = [
                    did for did in all_dids
                    if doc_passes_filter(doc_ivs.get(did, []), valid_includes, [])
                ]
            elif variant_kind == "ref_time":
                # V4: ref_time-based filter — kill if doc's ref_us is inside
                # any excluded anchor interval. Engagement docs (written
                # outside the period but mentioning it in contrast) survive.
                eligible = []
                for did in all_dids:
                    ok = True
                    if valid_includes:
                        passed_inc = False
                        for relation, anchor_ivs in valid_includes:
                            if constraint_factor_for_doc(
                                doc_ivs.get(did, []), anchor_ivs, relation) >= 1.0:
                                passed_inc = True
                                break
                        if not passed_inc:
                            ok = False
                    if ok:
                        ref_us = doc_ref_us.get(did, 0)
                        for anchor_ivs in valid_excludes:
                            if any(ai.earliest_us <= ref_us <= ai.latest_us
                                   for ai in anchor_ivs):
                                ok = False
                                break
                    if ok:
                        eligible.append(did)
            elif variant_kind == "existential":
                # V5: pass if doc has ∃ content-anchor outside all excluded
                # anchors. Correctly surfaces engagement + retrospective
                # cases the ref_time heuristic (V4) misses.
                eligible = []
                for did in all_dids:
                    ok = True
                    if valid_includes:
                        passed_inc = False
                        for relation, anchor_ivs in valid_includes:
                            if constraint_factor_for_doc(
                                doc_ivs.get(did, []), anchor_ivs, relation) >= 1.0:
                                passed_inc = True
                                break
                        if not passed_inc:
                            ok = False
                    if ok and valid_excludes:
                        d_ivs = doc_ivs.get(did, [])
                        has_outside = False
                        if not d_ivs:
                            has_outside = True
                        else:
                            for di in d_ivs:
                                outside_all = True
                                for anchor_ivs in valid_excludes:
                                    for ai in anchor_ivs:
                                        if (di.earliest_us <= ai.latest_us
                                            and ai.earliest_us <= di.latest_us):
                                            outside_all = False
                                            break
                                    if not outside_all:
                                        break
                                if outside_all:
                                    has_outside = True
                                    break
                        if not has_outside:
                            ok = False
                    if ok:
                        eligible.append(did)
            else:
                eligible = [
                    did for did in all_dids
                    if doc_passes_filter(doc_ivs.get(did, []), valid_includes, valid_excludes)
                ]

            pool = build_pool(sem, all_dids, eligible, pool_size=10)
            pool_texts = [next(d[1] for d in docs if d[0] == did) for did in pool]
            rr = dict(zip(pool, [rerank_partial.get(d, 0.0) for d in pool], strict=False))
            rr_norm = normalize_rerank_full(rr, pool, tail_score=0.0)
            base_norm = normalize_dict(rr_norm)

            match_scores = {}
            for did in pool:
                m, _ = evaluate_disjoint_variant(
                    plan, doc_ivs.get(did, []), resolver, variant_kind,
                    doc_ref_us=doc_ref_us.get(did))
                match_scores[did] = m

            combined = {did: base_norm.get(did, 0.0) + match_scores.get(did, 0.0)
                        for did in pool}
            ranking = sorted(combined.keys(), key=lambda d: combined[d], reverse=True)

            print(f"\n  [{vname}]")
            for rank, did in enumerate(ranking, 1):
                role = roles[did]
                marker = "★" if role == "G" else " "
                print(f"    {marker} {rank}. {did} ({role})  match={match_scores.get(did,0):.3f}  "
                      f"sem={sem.get(did,0):.3f}  comb={combined.get(did,0):.3f}")
            # all_dids not in pool? Note them.
            kicked = [d for d in all_dids if d not in pool]
            for did in kicked:
                role = roles[did]
                marker = "★" if role == "G" else " "
                print(f"    {marker} -- KICKED-from-pool {did} ({role})")

    print("\n" + "=" * 70)
    print("KEY: G = gold engagement-relevant doc (should rank #1)")
    print("     I = distractor entirely inside excluded window")
    print("     OT = clean outside, topical (current logic favors)")
    print("     UN = clean outside, unrelated")


if __name__ == "__main__":
    asyncio.run(main())
