"""A/B: ref-time-only doc extraction vs full extraction.

Tests whether doc-content extraction is doing anything beyond ref_time.

Configs:
  reftime-only + LLM-plan       : tests if content extraction matters with LLM plans
  reftime-only + duckling-plan  : tests if content extraction matters with rule plans
  reftime-only + dateparser-plan: same but with dateparser
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval_tr.planner import QueryPlanner
from temporal_retrieval_tr.research._reftime_only_extractor import RefTimeOnlyExtractor
from temporal_retrieval_tr.research._rule_planners import (
    DateparserPlanner, DucklingPlanner,
)
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()


CONFIGS = [
    ("reftime-only + LLM-plan",        RefTimeOnlyExtractor, QueryPlanner),
    ("reftime-only + duckling-plan",   RefTimeOnlyExtractor, DucklingPlanner),
    ("reftime-only + dateparser-plan", RefTimeOnlyExtractor, DateparserPlanner),
]


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

    for label, ExtractorCls, PlannerCls in CONFIGS:
        print(f"\n=== {label} ===\n", flush=True)
        extractor = ExtractorCls()
        planner = PlannerCls()
        rows = []
        for bench in BENCH_NAMES:
            try:
                m = await run_bench(bench, extractor, planner, bench, embed_fn, rerank_fn) \
                    if False else await run_bench(bench, extractor, planner, embed_fn, rerank_fn)
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
            if hasattr(planner, "save_caches"):
                planner.save_caches()
        n = len(rows)
        r1 = sum(m["R@1"] for _, m in rows) / n
        r5 = sum(m["R@5"] for _, m in rows) / n
        print(f"\nMACRO {label}:  R@1={r1:.4f}  R@5={r5:.4f}  n={n}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
