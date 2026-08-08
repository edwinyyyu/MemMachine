"""Single-arm bench for Copeland-pairwise ranking method.

Compares timed/timeless pairs by base only; timed/timed pairs by
base + match. Avoids the fixed timeless_match constant.

Compare against current ship: timeless_match=0.8 fixed → macro 0.7989.
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalExtractor, TemporalRetriever
from temporal_retrieval_tr.planner import QueryPlanner
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()

BASELINE_R1 = 0.7989  # ship: timeless_match=0.8, additive
BASELINE_R5 = 0.9611


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
    extractor = TemporalExtractor()
    planner = QueryPlanner()
    print(f"\n=== Copeland-pairwise ranking, vs ship additive+0.8 "
          f"(R@1={BASELINE_R1}, R@5={BASELINE_R5}) ===\n", flush=True)
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
        print(f"  {bench:30s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  n={m['n']}",
              flush=True)
    n = len(rows)
    r1 = sum(m["R@1"] for _, m in rows) / n
    r5 = sum(m["R@5"] for _, m in rows) / n
    print(f"\nMACRO  copeland-pairwise: R@1={r1:.4f}  R@5={r5:.4f}  n={n}",
          flush=True)
    print(f"MACRO  additive+0.8 ship:  R@1={BASELINE_R1:.4f}  R@5={BASELINE_R5:.4f}",
          flush=True)
    print(f"Δ                          R@1={r1-BASELINE_R1:+.4f}  R@5={r5-BASELINE_R5:+.4f}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
