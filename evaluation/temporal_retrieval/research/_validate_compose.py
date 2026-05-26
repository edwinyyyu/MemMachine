"""Compose leaf-side (frac_min intersect) with aggregator (mean/mean).

Tests whether the two best-on-their-own findings stack:
  A0 zadeh            min AND, max OR, binary intersect (baseline)
  A1 frac_min         min AND, max OR, min-norm intersect
  A2 mean_both        mean AND, mean OR, binary intersect
  A3 mean_both_fmin   mean AND, mean OR, min-norm intersect ← compose

If A3 ≈ A1 + A2, layers are independent → ship the compose.

Reuses planner-tree + extractor-v3.3 caches; only scoring changes.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_compose
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

# (name, and_aggregator, or_aggregator, intersect_leaf, disjoint_skip)
VARIANTS = [
    ("zadeh",                "min",       "max",  "binary",   False),
    ("mean_both_fmin",       "mean",      "mean", "min_norm", False),
    ("mean_both_fmin_djskp", "mean",      "mean", "min_norm", True),   # pull disjoint out
    ("flat_mean_fmin",       "flat_mean", "max",  "min_norm", False),
    ("flat_mean_fmin_djskp", "flat_mean", "max",  "min_norm", True),
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

    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue

        plan = await planner.plan(q["text"], q["ref_time"])

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

        for vname, and_agg, or_agg, leaf_kind, dj_skip in VARIANTS:
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
                        intersect_leaf=leaf_kind,
                        disjoint_skip=dj_skip,
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
    }


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
        b_r1 = res["zadeh"]["R@1"]
        for vname, _, _, _, _ in VARIANTS:
            m = res[vname]
            d = m["R@1"] - b_r1
            print(f"  {vname:16s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
                  f"all_R@5={m['all_R@5']:.3f}  Δr1={d:+.3f}", flush=True)

    print("\n" + "=" * 110, flush=True)
    print(f"{'bench':28s}", end="", flush=True)
    for vname, _, _, _, _ in VARIANTS:
        print(f"  {vname[:14]:>14s}", end="")
    print(flush=True)
    print("-" * 110, flush=True)
    macro = {v[0]: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    for bench in BENCHES:
        if bench not in all_results:
            continue
        r = all_results[bench]
        for k in ("R@1", "R@5", "all_R@5"):
            for v in macro:
                macro[v][k].append(r[v][k])
        print(f"{bench:28s}", end="", flush=True)
        for vname, _, _, _, _ in VARIANTS:
            print(f"  {r[vname]['R@1']:>14.3f}", end="")
        print(flush=True)

    print("\nMACRO:", flush=True)
    print(f"{'variant':16s} {'R@1':>8s} {'R@5':>8s} {'all_R@5':>10s} {'Δr1':>10s}", flush=True)
    z_r1 = sum(macro["zadeh"]["R@1"]) / max(1, len(macro["zadeh"]["R@1"]))
    for vname, _, _, _, _ in VARIANTS:
        m1 = sum(macro[vname]["R@1"]) / max(1, len(macro[vname]["R@1"]))
        m5 = sum(macro[vname]["R@5"]) / max(1, len(macro[vname]["R@5"]))
        ma = sum(macro[vname]["all_R@5"]) / max(1, len(macro[vname]["all_R@5"]))
        d = m1 - z_r1
        print(f"{vname:16s} {m1:>8.3f} {m5:>8.3f} {ma:>10.3f} {d:>+10.3f}", flush=True)

    # Composability check
    print("\nComposability check:", flush=True)
    f_d = sum(macro["frac_min"]["R@1"]) / max(1, len(macro["frac_min"]["R@1"])) - z_r1
    m_d = sum(macro["mean_both"]["R@1"]) / max(1, len(macro["mean_both"]["R@1"])) - z_r1
    c_d = sum(macro["mean_both_fmin"]["R@1"]) / max(1, len(macro["mean_both_fmin"]["R@1"])) - z_r1
    print(f"  frac_min alone:        +{f_d:.3f}", flush=True)
    print(f"  mean_both alone:       +{m_d:.3f}", flush=True)
    print(f"  sum (if independent):  +{f_d + m_d:.3f}", flush=True)
    print(f"  compose actual:        +{c_d:.3f}", flush=True)
    print(f"  interaction:           {c_d - (f_d + m_d):+.3f}", flush=True)

    out = ROOT / "compose_validation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
