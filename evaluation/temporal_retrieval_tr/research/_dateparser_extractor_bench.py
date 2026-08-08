"""A/B: dateparser extractor vs LLM extractor (TemporalExtractor).

Replaces the doc-side extractor with the rule-based DateparserExtractor.
Keeps the LLM planner — this isolates the extractor cost/quality trade.

Baselines (from previous runs, all extremum-Copeland ON):
  LLM extractor + LLM planner + copeland_pairwise:  0.8044 / 0.9618
  LLM extractor + LLM planner + additive+0.8:       0.7989 / 0.9611
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval_tr.planner import QueryPlanner
from temporal_retrieval_tr.research._dateparser_extractor import DateparserExtractor
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()


async def run_bench(bench, extractor, planner, embed_fn, rerank_fn,
                    ranking_method):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor, planner=planner,
        ranking_method=ranking_method,
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
    extractor = DateparserExtractor()
    planner = QueryPlanner()  # keep LLM planner for now

    print("\n=== Dateparser EXTRACTOR + LLM planner ===\n", flush=True)
    rows = []
    for bench in BENCH_NAMES:
        try:
            m = await run_bench(bench, extractor, planner, embed_fn, rerank_fn,
                                ranking_method="copeland_pairwise")
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
    n = len(rows)
    r1 = sum(m["R@1"] for _, m in rows) / n
    r5 = sum(m["R@5"] for _, m in rows) / n
    print(f"\nMACRO dateparser-extr + LLM-plan:  R@1={r1:.4f}  R@5={r5:.4f}  n={n}", flush=True)
    print("MACRO LLM-extr + LLM-plan (ship):  R@1=0.8044  R@5=0.9618", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
