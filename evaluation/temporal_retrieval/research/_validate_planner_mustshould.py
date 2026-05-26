"""A/B: tree planner AND/OR vs tree planner must/should (renamed-only).

Same node structure, same JSON schema, same evaluator (min/max). The ONLY
difference between the two planners is the operator vocabulary in the
prompt and node-type enum. This isolates the LLM's "what tree should I
emit?" decision from any scoring-algebra change.

Outputs:
  - per-bench R@1 / R@5 / all_R@5 for each variant
  - plan-structure stats: how often does the LLM use must/should vs and/or
  - parse-failure rates per variant

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_planner_mustshould
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter

import numpy as np

from temporal_retrieval import Doc
from temporal_retrieval.core import (
    Interval,
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
from temporal_retrieval.planner_tree import (
    Leaf as AOLeaf,
    Not as AONot,
    And as AOAnd,
    Or as AOOr,
    TreePlanner as TreePlannerAndOr,
    evaluate_tree_match as eval_andor,
)
from temporal_retrieval.planner_tree_mustshould import (
    Leaf as MSLeaf,
    Must as MSMust,
    Should as MSShould,
    Not as MSNot,
    TreePlannerMustShould,
    evaluate_tree_match as eval_mustshould,
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


# -------------- Tree structure counters --------------

def _count_andor(node, c):
    if node is None:
        return
    if isinstance(node, AOLeaf):
        c["leaf"] += 1
        return
    if isinstance(node, AOAnd):
        c["and"] += 1
        for k in node.children:
            _count_andor(k, c)
        return
    if isinstance(node, AOOr):
        c["or"] += 1
        for k in node.children:
            _count_andor(k, c)
        return
    if isinstance(node, AONot):
        c["not"] += 1
        _count_andor(node.child, c)


def _count_mustshould(node, c):
    if node is None:
        return
    if isinstance(node, MSLeaf):
        c["leaf"] += 1
        return
    if isinstance(node, MSMust):
        c["must"] += 1
        for k in node.children:
            _count_mustshould(k, c)
        return
    if isinstance(node, MSShould):
        c["should"] += 1
        for k in node.children:
            _count_mustshould(k, c)
        return
    if isinstance(node, MSNot):
        c["not"] += 1
        _count_mustshould(node.child, c)


# -------------- Bench eval --------------

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
    planner_ao = TreePlannerAndOr()
    planner_ms = TreePlannerMustShould()

    # Index: extract doc envelopes
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

    counters_ao = Counter()
    counters_ms = Counter()
    n_parse_ao = 0
    n_parse_ms = 0

    variant_stats = {
        "andor":      {"n_eval": 0, "n_r1": 0, "n_r5": 0, "all_r5": []},
        "mustshould": {"n_eval": 0, "n_r1": 0, "n_r5": 0, "all_r5": []},
    }

    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue

        # Plan with both planners
        plan_ao, plan_ms = await asyncio.gather(
            planner_ao.plan(q["text"], q["ref_time"]),
            planner_ms.plan(q["text"], q["ref_time"]),
        )
        _count_andor(plan_ao.expr, counters_ao)
        _count_mustshould(plan_ms.expr, counters_ms)
        if plan_ao.parse_error:
            n_parse_ao += 1
        if plan_ms.parse_error:
            n_parse_ms += 1

        # Resolve anchors for each plan's unique leaves
        async def _resolve(plan, leaves_attr_iter):
            anchors = {}
            seen_ids = set()
            to_resolve = []
            for leaf in leaves_attr_iter:
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
            return anchors

        anchors_ao = await _resolve(plan_ao, plan_ao.iter_leaves())
        anchors_ms = await _resolve(plan_ms, plan_ms.iter_leaves())

        # Build filters (only top-level and-spine leaves -- mirror retriever)
        def _filter_andor(node):
            includes, excludes = [], []
            if node is None:
                return includes, excludes
            if isinstance(node, AOLeaf):
                if node.relation == "disjoint":
                    excludes.append(node)
                else:
                    includes.append(node)
                return includes, excludes
            if isinstance(node, AOAnd):
                for c in node.children:
                    inc, exc = _filter_andor(c)
                    includes.extend(inc)
                    excludes.extend(exc)
            return includes, excludes

        def _filter_mustshould(node):
            includes, excludes = [], []
            if node is None:
                return includes, excludes
            if isinstance(node, MSLeaf):
                if node.relation == "disjoint":
                    excludes.append(node)
                else:
                    includes.append(node)
                return includes, excludes
            if isinstance(node, MSMust):
                for c in node.children:
                    inc, exc = _filter_mustshould(c)
                    includes.extend(inc)
                    excludes.extend(exc)
            return includes, excludes

        for vname, plan, anchors, eval_fn, filter_fn in (
            ("andor", plan_ao, anchors_ao, eval_andor, _filter_andor),
            ("mustshould", plan_ms, anchors_ms, eval_mustshould, _filter_mustshould),
        ):
            includes, excludes = filter_fn(plan.expr)
            valid_includes = []
            valid_excludes = []
            for leaf in includes:
                ivs = anchors.get(id(leaf), [])
                if ivs:
                    valid_includes.append((leaf.relation, ivs))
            for leaf in excludes:
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

            def make_resolver(anchors):
                def r(leaf):
                    return anchors.get(id(leaf), [])
                return r

            resolver = make_resolver(anchors)

            match_scores = {}
            for did in pool:
                d_ivs = doc_ivs.get(did, [])
                if not d_ivs and plan.expr is not None:
                    match_scores[did] = 1.0  # empty doc match
                elif plan.expr is None:
                    match_scores[did] = 1.0
                else:
                    match_scores[did] = eval_fn(plan, d_ivs, resolver)

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
    } | {
        "_struct": {
            "andor": dict(counters_ao),
            "mustshould": dict(counters_ms),
            "parse_failures_andor": n_parse_ao,
            "parse_failures_mustshould": n_parse_ms,
        }
    }


async def main():
    print(f"Loading embed_fn... ({len(BENCHES)} benches)", flush=True)
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
        for v in ("andor", "mustshould"):
            m = res[v]
            print(f"  {v:10s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
                  f"all_R@5={m['all_R@5']:.3f}  n={m['n_eval']}", flush=True)
        st = res["_struct"]
        print(f"  structure  ao={st['andor']}  ms={st['mustshould']}  "
              f"parse_fail ao={st['parse_failures_andor']} "
              f"ms={st['parse_failures_mustshould']}", flush=True)

    # MACRO
    print("\n" + "=" * 110, flush=True)
    print(f"{'bench':28s} {'ao R@1':>9s} {'ms R@1':>9s} {'Δ':>7s}   "
          f"{'ao R@5':>9s} {'ms R@5':>9s} {'Δ':>7s}", flush=True)
    print("-" * 110, flush=True)
    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []}
             for v in ("andor", "mustshould")}
    tot_struct = {"andor": Counter(), "mustshould": Counter()}
    for bench in BENCHES:
        if bench not in all_results:
            continue
        r = all_results[bench]
        d1 = r["mustshould"]["R@1"] - r["andor"]["R@1"]
        d5 = r["mustshould"]["R@5"] - r["andor"]["R@5"]
        for k in ("R@1", "R@5", "all_R@5"):
            for v in macro:
                macro[v][k].append(r[v][k])
        tot_struct["andor"].update(r["_struct"]["andor"])
        tot_struct["mustshould"].update(r["_struct"]["mustshould"])
        print(f"{bench:28s} {r['andor']['R@1']:>9.3f} {r['mustshould']['R@1']:>9.3f} "
              f"{d1:>+7.3f}   "
              f"{r['andor']['R@5']:>9.3f} {r['mustshould']['R@5']:>9.3f} "
              f"{d5:>+7.3f}", flush=True)

    print("\nMACRO:", flush=True)
    for v in ("andor", "mustshould"):
        m1 = sum(macro[v]["R@1"]) / max(1, len(macro[v]["R@1"]))
        m5 = sum(macro[v]["R@5"]) / max(1, len(macro[v]["R@5"]))
        ma = sum(macro[v]["all_R@5"]) / max(1, len(macro[v]["all_R@5"]))
        print(f"  {v:12s} R@1={m1:.3f} R@5={m5:.3f} all_R@5={ma:.3f}", flush=True)
    print(f"\nTOTAL structure across all queries:", flush=True)
    print(f"  andor:      {dict(tot_struct['andor'])}", flush=True)
    print(f"  mustshould: {dict(tot_struct['mustshould'])}", flush=True)

    out = ROOT / "planner_mustshould_validation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
