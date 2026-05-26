"""35-bench A/B: V1 strict vs V2 aggregate vs V2.7 binary-engagement+floor
on the `disjoint` relation.

Qualitative micro-probe (6 hand-crafted engagement scenarios) showed:
- V1 strict ranks gold @ #1 only 1/6 (kills engagement docs)
- V2 aggregate ranks gold @ #1 4/6 (fractional credit for engagement)
- V2.7 binary-engagement+floor ranks gold @ #1 5/6 (floors at 0.5)
- V3 polarity-blind ranks pure-inside #1 → fatal failure mode, rejected

Quantitative question: does V2 or V2.7 regress on existing strict-exclusion
benches (notin_multi_interval, negation_temporal, composition) by an amount
that outweighs the engagement-case wins?

The existing benches' gold docs are all clean-outside, so V1/V2/V2.7 should
give gold match=1.0 across the board. The difference is whether DISTRACTORS
that V1 kills (any interval inside excluded → filter rejects, scoring=0)
get partial credit and leak into top-K under V2 / V2.7.

All variants share frac_min (min_norm) intersect leaf, min/max aggregators,
and only differ in the disjoint leaf-factor computation.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_disjoint_variants
"""
from __future__ import annotations

import asyncio
import json

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
    recency_scores,
)
from temporal_retrieval.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval.planner import QueryPlanner
from temporal_retrieval.schema import parse_iso, to_us

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    setup_env,
)

setup_env()


BENCH_NAMES = [
    "adversarial", "allen", "ambiguous_year", "ambiguous_year_adv",
    "axis", "causal_relative", "composition", "cotemporal",
    "dense_cluster", "disc", "edge_conjunctive_temporal", "edge_era_refs",
    "edge_multi_te_doc", "edge_relative_time", "era", "goldilocks",
    "goldilocks_v2", "hard_bench", "hard_dense_cluster", "latest_recent",
    "lattice", "mixed_cue", "negation_temporal", "notin_multi_interval",
    "open_ended_date", "polarity", "precedents", "realq", "realq_deictic",
    "realq_v2", "sensitivity_curated", "speculative_anchors",
    "temporal_essential", "timeless_policies", "utterance",
]
BENCHES = {n: (f"{n}_docs.jsonl", f"{n}_queries.jsonl", f"{n}_gold.jsonl")
           for n in BENCH_NAMES}

# variant name
VARIANTS = ["V1_strict", "V2_aggregate", "V2_7_binary_engagement",
            "V4_ref_time", "V5_existential_outside"]


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


def _disjoint_factor(variant: str, doc_ivs, anchor_ivs,
                     doc_ref_us: int | None = None) -> float:
    """Pure leaf-factor computation; filter rules are kept separate."""
    if variant == "V1_strict":
        cont = excluded_containment(doc_ivs, anchor_ivs)
        return max(0.0, 1.0 - cont)
    if variant == "V2_aggregate":
        cont = excluded_containment_aggregate(doc_ivs, anchor_ivs)
        return max(0.0, 1.0 - cont)
    if variant == "V2_7_binary_engagement":
        agg = excluded_containment_aggregate(doc_ivs, anchor_ivs)
        if agg >= 1.0:
            return 0.0
        return max(0.5, 1.0 - agg)
    if variant == "V4_ref_time":
        ref_inside = any(
            ai.earliest_us <= doc_ref_us <= ai.latest_us
            for ai in anchor_ivs
        ) if doc_ref_us is not None else False
        return 0.0 if ref_inside else 1.0
    if variant == "V5_existential_outside":
        # Pass if ANY doc anchor falls outside ALL excluded anchors.
        # Semantically: doc has at least one content-anchor that lives in
        # the answer space (outside excluded). The retrospective doc
        # (ref_time inside but mentions outside event) survives correctly.
        if not doc_ivs:
            return 1.0  # no anchors = trivially "no anchor inside"
        for di in doc_ivs:
            # di "outside" if it doesn't overlap ANY excluded interval
            overlaps_excluded = False
            for ai in anchor_ivs:
                if di.earliest_us <= ai.latest_us and ai.earliest_us <= di.latest_us:
                    overlaps_excluded = True
                    break
            if not overlaps_excluded:
                return 1.0
        return 0.0
    raise ValueError(variant)


def _disjoint_filter(variant: str, doc_ivs, anchor_ivs) -> bool:
    """Does the doc pass the disjoint filter under this variant?

    V1: doc fails if any doc-interval is fully inside excluded (max_cont=1).
    V2: same as V1's filter — keeps the strict pool admission. Scoring is
        what changes (fractional credit for those that do pass).
        Actually for engagement docs the V1 filter ALREADY rejects them but
        build_pool tops up from raw semantic. So filter doesn't change
        outcomes on small pools.
    V2.7: doc fails only if aggregate containment >= 1 (all intervals
          inside excluded). Docs with any outside interval pass.
    """
    if variant == "V2_7_binary_engagement":
        agg = excluded_containment_aggregate(doc_ivs, anchor_ivs)
        return agg < 1.0
    # V1 and V2 both use V1's strict filter (max-over-pairs)
    cont = excluded_containment(doc_ivs, anchor_ivs)
    return cont < 1.0


def evaluate_match_for_variant(plan, doc_ivs, leaf_anchor_resolver,
                               variant: str,
                               doc_ref_us: int | None = None) -> float:
    if not plan.expr:
        return 1.0
    clause_scores = []
    for ci, clause in enumerate(plan.expr):
        leaf_factors = []
        for li, leaf in enumerate(clause):
            anchor_ivs = leaf_anchor_resolver(ci, li, leaf)
            if not anchor_ivs:
                f = 1.0
            elif leaf.relation == "disjoint":
                f = _disjoint_factor(variant, doc_ivs, anchor_ivs,
                                     doc_ref_us=doc_ref_us)
            elif leaf.relation == "intersect":
                f = _intersect_min_norm(doc_ivs, anchor_ivs)
            else:
                f = constraint_factor_for_doc(doc_ivs, anchor_ivs, leaf.relation)
            leaf_factors.append(f)
        clause_scores.append(min(leaf_factors) if leaf_factors else 1.0)
    return max(clause_scores) if clause_scores else 1.0


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
            out.append(float(np.dot(qe, de) / (qn * dn)) if (dn := float(np.linalg.norm(de))) else 0.0)
        return out
    async def cosine_rerank_simple(query, doc_texts):
        if not doc_texts:
            return []
        qe = (await embed_fn([query]))[0]
        des = await embed_fn(doc_texts)
        qn = float(np.linalg.norm(qe)) or 1e-9
        scores = []
        for de in des:
            den = float(np.linalg.norm(de)) or 1e-9
            scores.append(float(np.dot(qe, de) / (qn * den)))
        return scores
    return cosine_rerank_simple


async def evaluate_bench(bench, embed_fn, rerank_fn):
    docs_file, queries_file, gold_file = BENCHES[bench]
    try:
        docs_jsonl, queries, gold_rows = load_bench_jsonl(
            docs_file, queries_file, gold_file)
    except FileNotFoundError as e:
        return {"_error": str(e)}
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    extractor = TemporalExtractorV3_3()
    planner = QueryPlanner()

    async def _extract(d):
        try:
            envs = await extractor.extract(d.text, parse_iso(d.ref_time))
        except Exception:
            envs = []
        return d.id, envs
    results = await asyncio.gather(*(_extract(d) for d in docs))
    doc_envs = {did: envs for did, envs in results}
    doc_ivs = {did: flatten_intervals(envs) for did, envs in doc_envs.items()}
    extractor.save_caches()

    embs = await embed_fn([d.text for d in docs])
    doc_emb = {d.id: np.asarray(e, dtype=np.float32)
               for d, e in zip(docs, embs, strict=False)}
    doc_ref_us = {d.id: to_us(parse_iso(d.ref_time)) for d in docs}
    docs_by_id = {d.id: d for d in docs}
    all_dids = list(doc_ref_us.keys())

    stats = {v: {"n_eval": 0, "n_r1": 0, "n_r5": 0, "all_r5": []}
             for v in VARIANTS}

    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue

        plan = await planner.plan(q["text"], q["ref_time"])

        leaves_flat = [
            (ci, li, leaf)
            for ci, clause in enumerate(plan.expr)
            for li, leaf in enumerate(clause)
        ]

        anchors = {}
        if leaves_flat:
            ref_dt = parse_iso(q["ref_time"])
            res = await asyncio.gather(
                *(extractor.extract(leaf.phrase, ref_dt)
                  for _, _, leaf in leaves_flat)
            )
            for (ci, li, _l), envs in zip(leaves_flat, res, strict=False):
                anchors[(ci, li)] = flatten_intervals(envs)

        # Semantic
        q_emb = np.asarray((await embed_fn([q["text"]]))[0], dtype=np.float32)
        sem_scores = {}
        qn = float(np.linalg.norm(q_emb)) or 1e-9
        for did, demb in doc_emb.items():
            dn = float(np.linalg.norm(demb)) or 1e-9
            sem_scores[did] = float(np.dot(q_emb, demb) / (qn * dn))

        def resolver(ci, li, _leaf):
            return anchors.get((ci, li), [])

        for vname in VARIANTS:
            # Per-variant filter
            valid_includes = []
            valid_excludes = []
            for ci, li, leaf in leaves_flat:
                ivs = anchors.get((ci, li), [])
                if not ivs:
                    continue
                if leaf.relation == "disjoint":
                    valid_excludes.append(ivs)
                else:
                    valid_includes.append((leaf.relation, ivs))

            # Variant-specific exclude rule
            if vname == "V2_7_binary_engagement":
                # softer filter: only kill on agg_cont >= 1
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
                        for anchor_ivs in valid_excludes:
                            if excluded_containment_aggregate(
                                doc_ivs.get(did, []), anchor_ivs) >= 1.0:
                                ok = False
                                break
                    if ok:
                        eligible.append(did)
            elif vname == "V4_ref_time":
                # V4 filter: kill if doc's ref_us is inside any excluded interval
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
            elif vname == "V5_existential_outside":
                # V5 filter: pass if doc has ANY anchor outside ALL excluded
                # anchors (i.e., at least one anchor that doesn't overlap
                # any excluded interval). Handles retrospective + engagement
                # cases correctly.
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
                        # Need ∃ anchor outside ALL excluded anchors
                        has_outside = False
                        if not d_ivs:
                            has_outside = True  # no anchors = no anchor inside
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
                    if doc_passes_filter(doc_ivs.get(did, []),
                                         valid_includes, valid_excludes)
                ]

            pool = build_pool(sem_scores, all_dids, eligible, pool_size=10)
            pool_texts = [docs_by_id[did].text for did in pool]
            rerank_raw = await rerank_fn(q["text"], pool_texts)
            rerank_partial = dict(zip(pool, rerank_raw, strict=False))
            rerank_norm = normalize_rerank_full(rerank_partial, pool, tail_score=0.0)
            base_norm = normalize_dict(rerank_norm)

            recency_norm = {}
            if plan.latest_intent or plan.earliest_intent:
                direction = "latest" if plan.latest_intent else "earliest"
                bundles = {did: [{"intervals": doc_ivs.get(did, [])}]
                           for did in pool}
                refs_us = {did: doc_ref_us.get(did, 0) for did in pool}
                recency_norm = recency_scores(bundles, refs_us, direction=direction)

            match_scores = {}
            for did in pool:
                d_ivs = doc_ivs.get(did, [])
                if not d_ivs and plan.expr:
                    match_scores[did] = 1.0
                elif not plan.expr:
                    match_scores[did] = 1.0
                else:
                    match_scores[did] = evaluate_match_for_variant(
                        plan, d_ivs, resolver, vname,
                        doc_ref_us=doc_ref_us.get(did))

            combined = {}
            for did in pool:
                s = base_norm.get(did, 0.0) + match_scores.get(did, 0.0)
                if recency_norm:
                    s += recency_norm.get(did, 0.0)
                combined[did] = s
            ranking = sorted(combined.keys(),
                             key=lambda d: combined[d], reverse=True)

            stats[vname]["n_eval"] += 1
            first = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
            if first is not None:
                if first <= 1:
                    stats[vname]["n_r1"] += 1
                if first <= 5:
                    stats[vname]["n_r5"] += 1
            top5 = set(ranking[:5])
            stats[vname]["all_r5"].append(len(top5 & gold_set) / len(gold_set))

    return {
        v: {
            "R@1": s["n_r1"] / max(1, s["n_eval"]),
            "R@5": s["n_r5"] / max(1, s["n_eval"]),
            "all_R@5": sum(s["all_r5"]) / max(1, len(s["all_r5"])),
            "n_eval": s["n_eval"],
        }
        for v, s in stats.items()
    }


async def main():
    print(f"Loading embed_fn... ({len(BENCHES)} benches, {len(VARIANTS)} variants)",
          flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    all_results = {}
    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        try:
            res = await evaluate_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue
        if "_error" in res:
            print(f"  ERROR: {res['_error']}", flush=True)
            continue
        all_results[bench] = res
        b_r1 = res["V1_strict"]["R@1"]
        b_r5 = res["V1_strict"]["R@5"]
        for v in VARIANTS:
            m = res[v]
            d1 = m["R@1"] - b_r1
            d5 = m["R@5"] - b_r5
            print(f"  {v:24s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
                  f"all={m['all_R@5']:.3f}  Δr1={d1:+.3f}  Δr5={d5:+.3f}",
                  flush=True)

    print("\n" + "=" * 70, flush=True)
    print("MACRO (DNF planner, frac_min intersect):", flush=True)
    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    for bench in BENCHES:
        if bench not in all_results:
            continue
        r = all_results[bench]
        for k in ("R@1", "R@5", "all_R@5"):
            for v in macro:
                macro[v][k].append(r[v][k])
    b1 = sum(macro["V1_strict"]["R@1"]) / max(1, len(macro["V1_strict"]["R@1"]))
    b5 = sum(macro["V1_strict"]["R@5"]) / max(1, len(macro["V1_strict"]["R@5"]))
    print(f"{'variant':24s} {'R@1':>8s} {'R@5':>8s} {'all_R@5':>10s} "
          f"{'Δr1':>10s} {'Δr5':>10s}", flush=True)
    for v in VARIANTS:
        m1 = sum(macro[v]["R@1"]) / max(1, len(macro[v]["R@1"]))
        m5 = sum(macro[v]["R@5"]) / max(1, len(macro[v]["R@5"]))
        ma = sum(macro[v]["all_R@5"]) / max(1, len(macro[v]["all_R@5"]))
        d1 = m1 - b1
        d5 = m5 - b5
        print(f"{v:24s} {m1:>8.3f} {m5:>8.3f} {ma:>10.3f} "
              f"{d1:>+10.3f} {d5:>+10.3f}", flush=True)

    # Focus on disjoint-bearing benches
    print("\n" + "=" * 70, flush=True)
    print("Per disjoint-bearing bench:", flush=True)
    for bench in ["negation_temporal", "composition", "notin_multi_interval",
                  "polarity", "realq_v2"]:
        if bench not in all_results:
            continue
        r = all_results[bench]
        print(f"\n  {bench}:", flush=True)
        for v in VARIANTS:
            m = r[v]
            print(f"    {v:24s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
                  f"all={m['all_R@5']:.3f}", flush=True)

    out = ROOT / "disjoint_variants_validation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
