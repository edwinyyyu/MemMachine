"""Iterate V7 scoring variants on key benches:
- baseline V7
- V7 + match_weight=2.0 (amplify temporal signal)
- V7 + strict_disjoint (V1-style strict containment as a discount)
- V7 + best_dref_dominant (avg per-clause instead of max; downweights partial)

Focus benches: engagement_disjoint (V7 should keep wins), notin_multi_interval
(V1 strict winning here), edge_conjunctive_temporal (V7 winning), and
3 macro benches (adversarial, hard_bench, realq_v2).

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._iterate_variants
"""
from __future__ import annotations

import asyncio
import json

import numpy as np

from temporal_retrieval_min import Doc as DocV1
from temporal_retrieval_min import TemporalRetriever
from temporal_retrieval_min.core import (
    build_pool,
    excluded_containment,
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from temporal_retrieval_min.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval_min.planner import Constraint, QueryPlan, QueryPlanner
from temporal_retrieval_min.schema import parse_iso, to_us

from temporal_retrieval_v7 import Doc as DocV7
from temporal_retrieval_v7 import TemporalRetrieverV7
from temporal_retrieval_v7.adapters import (
    extractor_to_doc_refs,
    plan_to_clauses,
)
from temporal_retrieval_v7.scoring import Clause, clause_score, final_score

from temporal_retrieval.research._common import (
    DATA_DIR,
    make_embed_fn,
    setup_env,
)
from temporal_retrieval_v7.research._full_ab import (
    _load_bench,
    _metrics,
    make_cosine_rerank_fn,
)

setup_env()

FOCUS_BENCHES = [
    "engagement_disjoint",
    "notin_multi_interval",
    "edge_conjunctive_temporal",
    "negation_temporal",
    "adversarial",
    "hard_bench",
    "realq_v2",
]


async def run_v7_variant(
    bench: str, embed_fn, rerank_fn,
    *,
    match_weight: float = 1.0,
    strict_disjoint: bool = False,
) -> dict:
    """V7 retriever with optional tweaks."""
    docs_jsonl, queries, gold = _load_bench(bench)
    if docs_jsonl is None:
        return None
    docs = [DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    extractor = TemporalExtractorV3_3()
    planner = QueryPlanner()

    # Index
    async def _extract(d):
        try:
            return d.id, await extractor.extract(d.text, parse_iso(d.ref_time))
        except Exception:
            return d.id, []

    extracted = dict(await asyncio.gather(*(_extract(d) for d in docs)))
    extractor.save_caches()
    doc_refs = {did: extractor_to_doc_refs(ivs)
                for did, ivs in extracted.items()}

    embs = await embed_fn([d.text for d in docs])
    doc_emb = {d.id: np.asarray(e, dtype=np.float32)
               for d, e in zip(docs, embs, strict=False)}
    doc_ref_us = {d.id: to_us(parse_iso(d.ref_time)) for d in docs}
    docs_by_id = {d.id: d for d in docs}
    all_dids = list(doc_ref_us.keys())

    rankings = {}
    for q in queries:
        plan = await planner.plan(q["text"], q["ref_time"])
        # resolve leaves
        all_leaves = [leaf for clause in plan.expr for leaf in clause]
        if all_leaves:
            phrases = list({l.phrase for l in all_leaves})
            rt = parse_iso(q["ref_time"])
            results = await asyncio.gather(*(
                extractor.extract(p, rt) for p in phrases
            ))
            by_phrase = dict(zip(phrases, results))
        else:
            by_phrase = {}

        def resolver(leaf: Constraint):
            return by_phrase.get(leaf.phrase, [])

        clauses = plan_to_clauses(plan, resolver)

        # Excluded anchors for strict_disjoint mode
        excluded_anchors = []
        if strict_disjoint:
            for ast_clause in plan.expr:
                for leaf in ast_clause:
                    if leaf.relation == "disjoint":
                        ivs = by_phrase.get(leaf.phrase, [])
                        if ivs:
                            excluded_anchors.append(ivs)

        # Semantic
        q_emb = np.asarray((await embed_fn([q["text"]]))[0], dtype=np.float32)
        qn = float(np.linalg.norm(q_emb)) or 1e-9
        sem_scores = {did: float(np.dot(q_emb, demb) / (qn * (np.linalg.norm(demb) or 1e-9)))
                      for did, demb in doc_emb.items()}

        # Match
        match = {}
        for did in all_dids:
            d_refs = doc_refs.get(did, [])
            if not d_refs and clauses:
                m = 1.0
            else:
                m = final_score(clauses, d_refs)

            # Optional strict-disjoint discount: V1's max-pair containment
            # applied per excluded anchor list.
            if strict_disjoint and excluded_anchors:
                d_ivs = extracted.get(did, [])
                worst_cont = 0.0
                for excl in excluded_anchors:
                    c = excluded_containment(d_ivs, excl)
                    if c > worst_cont:
                        worst_cont = c
                m *= max(0.0, 1.0 - worst_cont)

            match[did] = m

        eligible = [did for did in all_dids if match[did] > 0.0]

        pool = build_pool(sem_scores, all_dids, eligible, pool_size=10)
        if not pool:
            rankings[q["query_id"]] = []
            continue

        pool_texts = [docs_by_id[did].text for did in pool]
        rerank_raw = await rerank_fn(q["text"], pool_texts)
        rerank_partial = dict(zip(pool, rerank_raw, strict=False))
        r_full = normalize_rerank_full(rerank_partial, all_dids, 0.0)
        r_pool = {did: r_full.get(did, 0.0) for did in pool}
        base = normalize_dict(r_pool)

        rec = {}
        if plan.latest_intent or plan.earliest_intent:
            direction = "latest" if plan.latest_intent else "earliest"
            target = [d for d in pool if match.get(d, 0.0) > 0]
            target = target if len(target) >= 2 else pool
            if len(target) >= 2:
                bundles = {did: [{"intervals": extracted.get(did, [])}] if extracted.get(did) else [] for did in target}
                refs_us = {d: doc_ref_us[d] for d in target}
                rec = recency_scores(bundles, refs_us, direction=direction)

        combined = {}
        for did in pool:
            s = base.get(did, 0.0) + match_weight * match[did]
            if rec:
                s += rec.get(did, 0.0)
            combined[did] = s
        ranking = sorted(combined.keys(), key=lambda d: -combined[d])
        rankings[q["query_id"]] = ranking[:10]

    return _metrics(rankings, gold)


async def main():
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    variants = [
        ("V7_base", {}),
        ("V7_mw_1.5", {"match_weight": 1.5}),
        ("V7_mw_2.0", {"match_weight": 2.0}),
        ("V7_strict_disjoint", {"strict_disjoint": True}),
        ("V7_strict_mw1.5", {"match_weight": 1.5, "strict_disjoint": True}),
    ]

    print("Variant table (R@1 / R@5 per bench):\n")
    print(f"{'bench':28s}", end="")
    for v, _ in variants:
        print(f"  {v:>20s}", end="")
    print(f"  {'V1':>9s}")
    print("-" * (28 + 22 * len(variants) + 11))

    # V1 reference metrics (from full A/B json)
    try:
        ab = json.load(open(
            "temporal_retrieval_v7/ab_v1_vs_v7.json".replace("evaluation/", "")
        ))
    except FileNotFoundError:
        ab = {}

    for bench in FOCUS_BENCHES:
        print(f"{bench:28s}", end="", flush=True)
        for vname, kwargs in variants:
            m = await run_v7_variant(bench, embed_fn, rerank_fn, **kwargs)
            if m is None:
                s = "NA"
            else:
                s = f"{m['R@1']:.3f}/{m['R@5']:.3f}"
            print(f"  {s:>20s}", end="", flush=True)
        # V1 reference
        if bench in ab:
            v1 = ab[bench]["v1"]
            print(f"  {v1['R@1']:.3f}/{v1['R@5']:.3f}", flush=True)
        else:
            print("", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
