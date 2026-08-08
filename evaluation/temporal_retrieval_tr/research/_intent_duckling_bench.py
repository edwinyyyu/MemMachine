"""Bench: duckling-extr + Duckling planner with intent regex layer.

Tests whether a small intent regex layer (~50 lines) closes the gap
between rule-based and LLM-based query planning.

Baselines:
  duckling-both (no intent):       R@1=0.6527  R@5=0.8978
  duckling-extr + LLM-plan:        R@1=0.8013  R@5=0.9533
  LLM-extr + LLM-plan (ship):      R@1=0.8044  R@5=0.9618
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval_tr.research._duckling_extractor import DucklingHTTPExtractor
from temporal_retrieval_tr.research._intent_duckling_planner import IntentDucklingPlanner
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()


async def run_bench(bench, extractor, planner, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor, planner=planner,
        ranking_method="copeland_pairwise",
    )
    await vd.index(docs)
    rk = {q["query_id"]: [x.doc_id for x in
          await vd.query(q["text"], q["ref_time"], k=10)]
          for q in queries}
    m = metrics(rk, gold)
    del vd, docs
    gc.collect()
    return m


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    extractor = DucklingHTTPExtractor()
    planner = IntentDucklingPlanner()

    print("\n=== duckling-extr + intent-duckling-plan ===\n", flush=True)
    rows = []
    for bench in BENCH_NAMES:
        try:
            m = await run_bench(bench, extractor, planner, embed_fn, rerank_fn)
        except Exception as e:
            print(f"  ERROR {bench}: {e}", flush=True)
            continue
        if m is None:
            continue
        rows.append((bench, m))
        print(
            f"  {bench:30s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  n={m['n']}",
            flush=True,
        )
        extractor.save_caches()
        planner.save_caches()
    n = len(rows)
    r1 = sum(m["R@1"] for _, m in rows) / n
    r5 = sum(m["R@5"] for _, m in rows) / n
    print(f"\nMACRO duckling-extr + intent-duck-plan:  R@1={r1:.4f}  R@5={r5:.4f}  n={n}",
          flush=True)
    print("MACRO duckling-both (no intent):          R@1=0.6527  R@5=0.8978",
          flush=True)
    print("MACRO duckling-extr + LLM-plan:           R@1=0.8013  R@5=0.9533",
          flush=True)
    print("MACRO LLM-extr + LLM-plan (ship):         R@1=0.8044  R@5=0.9618",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
