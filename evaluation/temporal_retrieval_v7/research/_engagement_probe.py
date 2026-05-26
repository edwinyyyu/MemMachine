"""V1 vs V7 qualitative A/B on the engagement_disjoint bench.

The new bench tests the architectural cases V7 is designed to fix:
- Engagement: doc mentions excluded period in contrast
- Retrospective: doc written inside excluded period, content outside
- Compound AND: 'in 2024 not in Q1'
- Multi-disjoint AND: 'not in 2020 or 2022'
- Colloquial-and-as-or: 'in 2020 and 2024' (incompatible conjuncts)
- After-relation: 'since the v3 launch' doc vs 'after launch' query

Reports per-query rank of each gold doc under V1 and V7, plus macro
R@1 / R@5 / all_recall@5 deltas.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._engagement_probe
"""
from __future__ import annotations

import asyncio
import json

from temporal_retrieval_min import (
    Doc as DocV1,
)
from temporal_retrieval_min import (
    TemporalRetriever,
)
from temporal_retrieval_v7 import Doc as DocV7
from temporal_retrieval_v7 import TemporalRetrieverV7

# Reuse the existing research helpers.
from temporal_retrieval.research._common import (
    DATA_DIR,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

BENCH = "engagement_disjoint"


def _load_bench():
    with open(DATA_DIR / f"{BENCH}_docs.jsonl") as f:
        docs_jsonl = [json.loads(line) for line in f]
    with open(DATA_DIR / f"{BENCH}_queries.jsonl") as f:
        queries = [json.loads(line) for line in f]
    with open(DATA_DIR / f"{BENCH}_gold.jsonl") as f:
        gold_rows = [json.loads(line) for line in f]
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    return docs_jsonl, queries, gold


async def _run_v1(docs_jsonl, queries, embed_fn, rerank_fn) -> dict:
    docs = [
        DocV1(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
        for d in docs_jsonl
    ]
    retriever = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await retriever.index(docs)
    rankings = {}
    for q in queries:
        res = await retriever.query(q["text"], q["ref_time"], k=10)
        rankings[q["query_id"]] = [r.doc_id for r in res]
    return rankings


async def _run_v7(docs_jsonl, queries, embed_fn, rerank_fn) -> dict:
    docs = [
        DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
        for d in docs_jsonl
    ]
    retriever = TemporalRetrieverV7(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await retriever.index(docs)
    rankings = {}
    for q in queries:
        res = await retriever.query(q["text"], q["ref_time"], k=10)
        rankings[q["query_id"]] = [r.doc_id for r in res]
    return rankings


def _metrics(rankings: dict, gold: dict, k_r5: int = 5) -> dict:
    n_r1 = n_r5 = n_eval = 0
    all_r5 = []
    for qid, ranking in rankings.items():
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        first = next(
            (i + 1 for i, d in enumerate(ranking) if d in gold_set), None
        )
        if first is not None:
            if first <= 1:
                n_r1 += 1
            if first <= k_r5:
                n_r5 += 1
        topk = set(ranking[:k_r5])
        all_r5.append(len(topk & gold_set) / len(gold_set))
    return {
        "n_eval": n_eval,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_R@5": sum(all_r5) / max(1, len(all_r5)),
    }


def _gold_ranks(rankings: dict, gold: dict) -> dict:
    """Per-query: dict of {gold_doc_id: rank or None}."""
    out = {}
    for qid, ranking in rankings.items():
        gset = gold.get(qid, set())
        ranks = {}
        for g in gset:
            try:
                ranks[g] = ranking.index(g) + 1
            except ValueError:
                ranks[g] = None
        out[qid] = ranks
    return out


async def main():
    docs_jsonl, queries, gold = _load_bench()
    print(f"=== {BENCH} bench: {len(docs_jsonl)} docs, {len(queries)} queries ===\n", flush=True)

    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    print("--- Running V1 (production) ---", flush=True)
    v1_rankings = await _run_v1(docs_jsonl, queries, embed_fn, rerank_fn)
    print("--- Running V7 ---", flush=True)
    v7_rankings = await _run_v7(docs_jsonl, queries, embed_fn, rerank_fn)

    v1_m = _metrics(v1_rankings, gold)
    v7_m = _metrics(v7_rankings, gold)
    v1_ranks = _gold_ranks(v1_rankings, gold)
    v7_ranks = _gold_ranks(v7_rankings, gold)

    print("\n--- Per-query gold ranks ---")
    print(f"{'query_id':28s}  {'V1 ranks':40s}  {'V7 ranks':40s}")
    print("-" * 115)
    for q in queries:
        qid = q["query_id"]
        v1_r = v1_ranks.get(qid, {})
        v7_r = v7_ranks.get(qid, {})
        v1_s = " ".join(
            f"{d.split('_')[-1]}:{rk if rk else 'X'}"
            for d, rk in sorted(v1_r.items(), key=lambda kv: (kv[1] or 99))
        )
        v7_s = " ".join(
            f"{d.split('_')[-1]}:{rk if rk else 'X'}"
            for d, rk in sorted(v7_r.items(), key=lambda kv: (kv[1] or 99))
        )
        marker = "*" if any(
            (v1_r.get(d) or 99) != (v7_r.get(d) or 99)
            for d in set(v1_r) | set(v7_r)
        ) else " "
        print(f"{marker} {qid:26s}  {v1_s:40s}  {v7_s:40s}")
        print(f"   query: {q['text']}")

    print("\n--- Macro metrics ---")
    print(f"{'variant':6s}  {'R@1':>6s}  {'R@5':>6s}  {'all_R@5':>9s}  n")
    print(f"{'V1':6s}  {v1_m['R@1']:>6.3f}  {v1_m['R@5']:>6.3f}  {v1_m['all_R@5']:>9.3f}  {v1_m['n_eval']}")
    print(f"{'V7':6s}  {v7_m['R@1']:>6.3f}  {v7_m['R@5']:>6.3f}  {v7_m['all_R@5']:>9.3f}  {v7_m['n_eval']}")
    print(f"{'Δ':6s}  {v7_m['R@1']-v1_m['R@1']:>+6.3f}  {v7_m['R@5']-v1_m['R@5']:>+6.3f}  {v7_m['all_R@5']-v1_m['all_R@5']:>+9.3f}")


if __name__ == "__main__":
    asyncio.run(main())
