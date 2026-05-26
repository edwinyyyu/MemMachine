"""Sweep AND × OR aggregator combinations on the AND/OR tree planner.

No re-extraction or re-planning needed — only the per-leaf-score → clause-score
→ doc-score aggregation changes. All caches reused.

Variants:
  A0 zadeh           : AND=min,        OR=max          baseline
  A1 mean            : AND=mean,       OR=max          partial-credit AND, selection OR
  A2 noisy_or_50     : AND=noisy_or(α=0.5), OR=max     user's algebra (canonical form)
  A3 noisy_or_70     : AND=noisy_or(α=0.7), OR=max     more aggressive evidence per leaf
  A4 product         : AND=product,    OR=max          probabilistic AND
  A5 sum_capped      : AND=sum_capped, OR=max          additive AND
  A6 mean_both       : AND=mean,       OR=mean         symmetric cumulative
  A7 noisy_or_full   : AND=noisy_or(α=0.5), OR=probsum pure noisy-OR pair
  A8 probsum_or      : AND=min,        OR=probsum      Zadeh AND + cumulative OR

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_aggregator_combos
"""
from __future__ import annotations

import asyncio
import json

import numpy as np

from temporal_retrieval import Doc
from temporal_retrieval.core import (
    build_pool,
    doc_passes_filter,
    flatten_intervals,
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from temporal_retrieval.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval.planner_tree import (
    And as AOAnd,
    Leaf as AOLeaf,
    Not as AONot,
    Or as AOOr,
    TreePlanner,
    evaluate_tree_match,
)
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

# (name, and_aggregator, or_aggregator, noisy_alpha)
VARIANTS = [
    ("zadeh",        "min",        "max",     0.5),
    ("mean",         "mean",       "max",     0.5),
    ("noisy_or_50",  "noisy_or",   "max",     0.5),
    ("noisy_or_70",  "noisy_or",   "max",     0.7),
    ("product",      "product",    "max",     0.5),
    ("sum_capped",   "sum_capped", "max",     0.5),
    ("mean_both",    "mean",       "mean",    0.5),
    ("noisy_or_full","noisy_or",   "probsum", 0.5),
    ("probsum_or",   "min",        "probsum", 0.5),
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


def _has_multileaf_and(node) -> bool:
    if node is None or isinstance(node, AOLeaf):
        return False
    if isinstance(node, AOAnd):
        if len(node.children) >= 2:
            return True
        return any(_has_multileaf_and(c) for c in node.children)
    if isinstance(node, AOOr):
        return any(_has_multileaf_and(c) for c in node.children)
    if isinstance(node, AONot):
        return _has_multileaf_and(node.child)
    return False


def _has_multiclause_or(node) -> bool:
    if node is None or isinstance(node, AOLeaf):
        return False
    if isinstance(node, AOOr):
        if len(node.children) >= 2:
            return True
        return any(_has_multiclause_or(c) for c in node.children)
    if isinstance(node, AOAnd):
        return any(_has_multiclause_or(c) for c in node.children)
    if isinstance(node, AONot):
        return _has_multiclause_or(node.child)
    return False


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
    planner = TreePlanner()

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
    n_mla = 0
    n_mor = 0

    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue

        plan = await planner.plan(q["text"], q["ref_time"])
        if _has_multileaf_and(plan.expr):
            n_mla += 1
        if _has_multiclause_or(plan.expr):
            n_mor += 1

        anchors = {}
        seen_ids = set()
        to_resolve = []
        for leaf in plan.iter_leaves():
            lid = id(leaf)
            if lid in seen_ids:
                continue
            seen_ids.add(lid)
            to_resolve.append((lid, leaf))
        if to_resolve:
            ref_dt = parse_iso(q["ref_time"])
            res = await asyncio.gather(
                *(extractor.extract(leaf.phrase, ref_dt) for _, leaf in to_resolve)
            )
            for (lid, _l), envs in zip(to_resolve, res, strict=False):
                anchors[lid] = flatten_intervals(envs)

        def _filter_andspine(node):
            inc, exc = [], []
            if node is None:
                return inc, exc
            if isinstance(node, AOLeaf):
                if node.relation == "disjoint":
                    exc.append(node)
                else:
                    inc.append(node)
                return inc, exc
            if isinstance(node, AOAnd):
                for c in node.children:
                    i2, e2 = _filter_andspine(c)
                    inc.extend(i2)
                    exc.extend(e2)
            return inc, exc

        includes_leaves, excludes_leaves = _filter_andspine(plan.expr)
        valid_includes, valid_excludes = [], []
        for leaf in includes_leaves:
            ivs = anchors.get(id(leaf), [])
            if ivs:
                valid_includes.append((leaf.relation, ivs))
        for leaf in excludes_leaves:
            ivs = anchors.get(id(leaf), [])
            if ivs:
                valid_excludes.append(ivs)

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

        def resolver(leaf):
            return anchors.get(id(leaf), [])

        for vname, and_agg, or_agg, alpha in VARIANTS:
            match_scores = {}
            for did in pool:
                d_ivs = doc_ivs.get(did, [])
                if not d_ivs and plan.expr is not None:
                    match_scores[did] = 1.0
                elif plan.expr is None:
                    match_scores[did] = 1.0
                else:
                    match_scores[did] = evaluate_tree_match(
                        plan, d_ivs, resolver,
                        and_aggregator=and_agg,
                        or_aggregator=or_agg,
                        noisy_alpha=alpha,
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
        v: {
            "R@1": s["n_r1"] / max(1, s["n_eval"]),
            "R@5": s["n_r5"] / max(1, s["n_eval"]),
            "all_R@5": sum(s["all_r5"]) / max(1, len(s["all_r5"])),
            "n_eval": s["n_eval"],
        }
        for v, s in variant_stats.items()
    } | {"_mla": n_mla, "_mor": n_mor}


async def main():
    print(f"Loading embed_fn... ({len(BENCHES)} benches, {len(VARIANTS)} variants)", flush=True)
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
        baseline_r1 = res["zadeh"]["R@1"]
        for vname, _, _, _ in VARIANTS:
            m = res[vname]
            d = m["R@1"] - baseline_r1
            mark = " " if abs(d) < 0.001 else ("+" if d > 0 else "-")
            print(f"  {vname:14s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
                  f"all_R@5={m['all_R@5']:.3f}  Δr1={d:+.3f}{mark}", flush=True)
        print(f"  multi-leaf-AND={res['_mla']}  multi-clause-OR={res['_mor']}", flush=True)

    # MACRO
    print("\n" + "=" * 140, flush=True)
    print(f"{'bench':28s} {'mla':>4s} {'mor':>4s}", end="", flush=True)
    for vname, _, _, _ in VARIANTS:
        print(f"  {vname[:10]:>10s}", end="")
    print(flush=True)
    print("-" * 140, flush=True)
    macro = {v[0]: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    tot_mla = tot_mor = 0
    for bench in BENCHES:
        if bench not in all_results:
            continue
        r = all_results[bench]
        tot_mla += r["_mla"]
        tot_mor += r["_mor"]
        for k in ("R@1", "R@5", "all_R@5"):
            for v in macro:
                macro[v][k].append(r[v][k])
        print(f"{bench:28s} {r['_mla']:>4d} {r['_mor']:>4d}", end="", flush=True)
        for vname, _, _, _ in VARIANTS:
            print(f"  {r[vname]['R@1']:>10.3f}", end="")
        print(flush=True)

    print(f"\nTotal multi-leaf-AND queries: {tot_mla}", flush=True)
    print(f"Total multi-clause-OR queries: {tot_mor}", flush=True)
    print("\nMACRO:", flush=True)
    print(f"{'variant':14s} {'R@1':>8s} {'R@5':>8s} {'all_R@5':>10s} {'Δr1 vs zadeh':>14s}", flush=True)
    z_r1 = sum(macro["zadeh"]["R@1"]) / max(1, len(macro["zadeh"]["R@1"]))
    for vname, _, _, _ in VARIANTS:
        m1 = sum(macro[vname]["R@1"]) / max(1, len(macro[vname]["R@1"]))
        m5 = sum(macro[vname]["R@5"]) / max(1, len(macro[vname]["R@5"]))
        ma = sum(macro[vname]["all_R@5"]) / max(1, len(macro[vname]["all_R@5"]))
        d = m1 - z_r1
        print(f"{vname:14s} {m1:>8.3f} {m5:>8.3f} {ma:>10.3f} {d:>+14.3f}", flush=True)

    out = ROOT / "aggregator_combos_validation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
