"""Investigate whether the AND/OR distinction in the planner is doing
any work, or whether it can be dropped entirely.

Two outputs in one pass over 35 benches on the PRODUCTION DNF planner:

A) STRUCTURE AUDIT
   For every query, log (n_clauses, n_leaves_per_clause, n_total_leaves)
   and report the distribution. Hypothesis from prior runs: ~95% of
   queries are single-clause-single-leaf, so AND/OR is a no-op on most.

B) AGGREGATOR ABLATION on the same plans + caches:
   - zadeh (baseline):       AND=min, OR=max, intersect=binary
   - frac_min (ship cand.):   AND=min, OR=max, intersect=min_norm
   - mean_both_fmin:          AND=mean, OR=mean, intersect=min_norm
   - flat_mean_fmin:          IGNORE structure; flat-mean of every leaf factor
                              in the plan (intersect=min_norm)
   - struct_flat_delta:       per-bench difference between mean_both_fmin
                              and flat_mean_fmin — tells us how much of
                              the structure-respecting aggregator's win
                              (if any) is actually attributable to
                              structure vs. just averaging more leaves.

If flat_mean_fmin ≈ frac_min macro AND per-bench breakdown shows no
benches where structure-respecting beats flat-mean by a meaningful
margin → the AND/OR distinction can be dropped on production DNF.
If flat_mean_fmin meaningfully underperforms on some benches → those
are the benches where AND/OR matters, and we look at WHY (compound
plan structure? planner could emit more?).

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._audit_and_or_distinction
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter

import numpy as np

from temporal_retrieval import Doc
from temporal_retrieval.core import (
    build_pool,
    constraint_factor_for_doc,
    doc_passes_filter,
    excluded_containment,
    flatten_intervals,
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from temporal_retrieval.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval.planner import QueryPlanner, evaluate_dnf_match
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


def evaluate_flat_mean(plan, doc_ivs, leaf_anchor_resolver,
                       intersect_leaf: str = "min_norm") -> float:
    """Flat-mean over every leaf factor in the plan, structure-ignoring.

    This is the 'drop AND/OR distinction' baseline: collapse the plan
    into a bag of leaves and average. If this ties the structured
    aggregator, the AND/OR distinction adds no value at the current
    plan-structure distribution.

    Empty plan → 1.0 (same convention as evaluate_dnf_match).
    """
    if not plan.expr:
        return 1.0
    leaf_factors = []
    for ci, clause in enumerate(plan.expr):
        for li, leaf in enumerate(clause):
            anchor_ivs = leaf_anchor_resolver(ci, li, leaf)
            if not anchor_ivs:
                f = 1.0
            elif leaf.relation == "disjoint":
                cont = excluded_containment(doc_ivs, anchor_ivs)
                f = max(0.0, 1.0 - cont)
            elif leaf.relation == "intersect" and intersect_leaf == "min_norm":
                f = _intersect_min_norm(doc_ivs, anchor_ivs)
            else:
                f = constraint_factor_for_doc(doc_ivs, anchor_ivs, leaf.relation)
            leaf_factors.append(f)
    if not leaf_factors:
        return 1.0
    return sum(leaf_factors) / len(leaf_factors)


# (name, and_aggregator, or_aggregator, intersect_leaf, mode)
VARIANTS = [
    ("zadeh",          "min",  "max",  "binary",   "dnf"),
    ("frac_min",       "min",  "max",  "min_norm", "dnf"),
    ("mean_both_fmin", "mean", "mean", "min_norm", "dnf"),
    ("flat_mean_fmin", None,   None,   "min_norm", "flat"),
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


async def evaluate_bench(bench, embed_fn, rerank_fn, plan_structure_counter):
    docs_file, queries_file, gold_file = BENCHES[bench]
    try:
        docs_jsonl, queries, gold_rows = load_bench_jsonl(
            docs_file, queries_file, gold_file)
    except FileNotFoundError as e:
        return {"_error": str(e)}, []
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    extractor = TemporalExtractorV3_3()
    planner = QueryPlanner()  # DNF (production)

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

    variant_stats = {
        v[0]: {"n_eval": 0, "n_r1": 0, "n_r5": 0, "all_r5": []}
        for v in VARIANTS
    }
    # Per-bench structure tallies for the audit table.
    bench_struct = Counter()

    # Per-query log: (n_clauses, n_leaves_total, max_leaves_per_clause)
    plans_log = []

    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue

        plan = await planner.plan(q["text"], q["ref_time"])

        n_clauses = len(plan.expr)
        n_leaves = sum(len(c) for c in plan.expr)
        max_leaves_per_clause = max((len(c) for c in plan.expr), default=0)

        # Structural bucket label for tally
        if n_clauses == 0:
            bucket = "empty"
        elif n_clauses == 1 and n_leaves == 1:
            bucket = "1clause_1leaf"
        elif n_clauses == 1 and n_leaves > 1:
            bucket = f"1clause_{n_leaves}leaves"
        elif n_clauses > 1 and max_leaves_per_clause == 1:
            bucket = f"{n_clauses}clauses_singleAND"
        else:
            bucket = f"{n_clauses}clauses_multiAND"
        plan_structure_counter[bucket] += 1
        bench_struct[bucket] += 1
        plans_log.append({
            "qid": qid, "n_clauses": n_clauses, "n_leaves": n_leaves,
            "max_leaves_per_clause": max_leaves_per_clause,
            "expr": [[{"phrase": l.phrase, "relation": l.relation} for l in c]
                      for c in plan.expr],
        })

        # Flatten DNF leaves: list of (ci, li, leaf)
        leaves_flat = [
            (ci, li, leaf)
            for ci, clause in enumerate(plan.expr)
            for li, leaf in enumerate(clause)
        ]

        anchors: dict[tuple[int, int], list] = {}
        if leaves_flat:
            ref_dt = parse_iso(q["ref_time"])
            res = await asyncio.gather(
                *(extractor.extract(leaf.phrase, ref_dt)
                  for _, _, leaf in leaves_flat)
            )
            for (ci, li, _l), envs in zip(leaves_flat, res, strict=False):
                anchors[(ci, li)] = flatten_intervals(envs)

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

        eligible = [
            did for did in all_dids
            if doc_passes_filter(doc_ivs.get(did, []), valid_includes, valid_excludes)
        ]
        q_emb = np.asarray((await embed_fn([q["text"]]))[0], dtype=np.float32)
        sem_scores = {}
        qn = float(np.linalg.norm(q_emb)) or 1e-9
        for did, demb in doc_emb.items():
            dn = float(np.linalg.norm(demb)) or 1e-9
            sem_scores[did] = float(np.dot(q_emb, demb) / (qn * dn))
        pool = build_pool(sem_scores, all_dids, eligible, pool_size=10)

        pool_texts = [docs_by_id[did].text for did in pool]
        rerank_raw = await rerank_fn(q["text"], pool_texts)
        rerank_partial = dict(zip(pool, rerank_raw, strict=False))
        rerank_norm = normalize_rerank_full(rerank_partial, pool, tail_score=0.0)
        base_norm = normalize_dict(rerank_norm)

        recency_norm = {}
        if plan.latest_intent or plan.earliest_intent:
            direction = "latest" if plan.latest_intent else "earliest"
            bundles = {did: [{"intervals": doc_ivs.get(did, [])}] for did in pool}
            refs_us = {did: doc_ref_us.get(did, 0) for did in pool}
            recency_norm = recency_scores(bundles, refs_us, direction=direction)

        def resolver(ci, li, _leaf):
            return anchors.get((ci, li), [])

        for vname, and_agg, or_agg, leaf_kind, mode in VARIANTS:
            match_scores = {}
            for did in pool:
                d_ivs = doc_ivs.get(did, [])
                if not d_ivs and plan.expr:
                    match_scores[did] = 1.0
                elif not plan.expr:
                    match_scores[did] = 1.0
                else:
                    if mode == "flat":
                        match_scores[did] = evaluate_flat_mean(
                            plan, d_ivs, resolver, intersect_leaf=leaf_kind)
                    else:
                        match_scores[did] = evaluate_dnf_match(
                            plan, d_ivs, resolver,
                            and_aggregator=and_agg,
                            or_aggregator=or_agg,
                            intersect_leaf=leaf_kind,
                        )

            combined = {}
            for did in pool:
                s = base_norm.get(did, 0.0) + match_scores.get(did, 0.0)
                if recency_norm:
                    s += recency_norm.get(did, 0.0)
                combined[did] = s
            ranking = sorted(combined.keys(), key=lambda d: combined[d], reverse=True)

            variant_stats[vname]["n_eval"] += 1
            first = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
            if first is not None:
                if first <= 1:
                    variant_stats[vname]["n_r1"] += 1
                if first <= 5:
                    variant_stats[vname]["n_r5"] += 1
            top5 = set(ranking[:5])
            variant_stats[vname]["all_r5"].append(len(top5 & gold_set) / len(gold_set))

    return {
        "variants": {
            v: {
                "R@1": s["n_r1"] / max(1, s["n_eval"]),
                "R@5": s["n_r5"] / max(1, s["n_eval"]),
                "all_R@5": sum(s["all_r5"]) / max(1, len(s["all_r5"])),
                "n_eval": s["n_eval"],
            }
            for v, s in variant_stats.items()
        },
        "structure": dict(bench_struct),
        "plans": plans_log,
    }, plans_log


async def main():
    print(f"Loading embed_fn... ({len(BENCHES)} benches, {len(VARIANTS)} variants)",
          flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    all_results = {}
    all_plans = {}
    plan_structure = Counter()

    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        try:
            res, plans = await evaluate_bench(bench, embed_fn, rerank_fn,
                                              plan_structure)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue
        if "_error" in res:
            print(f"  ERROR: {res['_error']}", flush=True)
            continue
        all_results[bench] = res
        all_plans[bench] = plans
        b_r1 = res["variants"]["zadeh"]["R@1"]
        for vname, _, _, _, _ in VARIANTS:
            m = res["variants"][vname]
            d = m["R@1"] - b_r1
            print(f"  {vname:16s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
                  f"all_R@5={m['all_R@5']:.3f}  Δr1={d:+.3f}", flush=True)
        struct = res["structure"]
        struct_str = ", ".join(f"{k}={v}" for k, v in sorted(struct.items()))
        print(f"  STRUCT: {struct_str}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("STRUCTURE AUDIT — total plans across all benches:", flush=True)
    total = sum(plan_structure.values())
    for bucket, n in sorted(plan_structure.items(),
                            key=lambda x: -x[1]):
        pct = 100 * n / max(1, total)
        print(f"  {bucket:30s}  {n:5d}  ({pct:5.1f}%)", flush=True)
    print(f"  TOTAL: {total} queries with gold", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("MACRO (DNF planner):", flush=True)
    macro = {v[0]: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    for bench in BENCHES:
        if bench not in all_results:
            continue
        r = all_results[bench]["variants"]
        for k in ("R@1", "R@5", "all_R@5"):
            for v in macro:
                macro[v][k].append(r[v][k])
    z_r1 = sum(macro["zadeh"]["R@1"]) / max(1, len(macro["zadeh"]["R@1"]))
    print(f"{'variant':16s} {'R@1':>8s} {'R@5':>8s} {'all_R@5':>10s} {'Δr1':>10s}",
          flush=True)
    for vname, _, _, _, _ in VARIANTS:
        m1 = sum(macro[vname]["R@1"]) / max(1, len(macro[vname]["R@1"]))
        m5 = sum(macro[vname]["R@5"]) / max(1, len(macro[vname]["R@5"]))
        ma = sum(macro[vname]["all_R@5"]) / max(1, len(macro[vname]["all_R@5"]))
        d = m1 - z_r1
        print(f"{vname:16s} {m1:>8.3f} {m5:>8.3f} {ma:>10.3f} {d:>+10.3f}",
              flush=True)

    # Per-bench: where does structure-respecting beat flat-mean?
    print("\n" + "=" * 70, flush=True)
    print("Per-bench: mean_both_fmin vs flat_mean_fmin", flush=True)
    print("(positive Δ = structure-respecting wins → AND/OR matters here)",
          flush=True)
    rows = []
    for bench in BENCHES:
        if bench not in all_results:
            continue
        r = all_results[bench]["variants"]
        delta = r["mean_both_fmin"]["R@1"] - r["flat_mean_fmin"]["R@1"]
        struct = all_results[bench]["structure"]
        n_compound = sum(n for k, n in struct.items()
                         if k != "1clause_1leaf" and k != "empty")
        rows.append((bench, delta, n_compound, struct.get("1clause_1leaf", 0)))
    rows.sort(key=lambda x: -abs(x[1]))
    print(f"  {'bench':30s} {'Δr1':>8s} {'compound':>10s} {'simple':>8s}",
          flush=True)
    for bench, delta, n_compound, simple in rows[:15]:
        print(f"  {bench:30s} {delta:>+8.3f} {n_compound:>10d} {simple:>8d}",
              flush=True)

    out = ROOT / "audit_and_or_distinction.json"
    with open(out, "w") as f:
        json.dump({
            "results": all_results,
            "structure_total": dict(plan_structure),
        }, f, indent=2, default=str)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
