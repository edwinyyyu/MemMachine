"""Sweep timeless_match_in_scope ∈ {0.0, 0.3, 0.5, 0.7, 0.8, 1.0}.

For timeless docs (no extracted anchors) when the query has bounded
scope, what's the right rank credit?
  1.0: vacuous match (current default; timeless trivially matches)
  0.0: no evidence (timeless cannot satisfy bounded scope)
  0.5/0.7/0.8: middle ground

The extractor + planner LLM calls are shared across values (same
caches), so this is mostly a rescoring sweep — only the FIRST run
costs LLM calls; subsequent values hit cache.
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


VALUES = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0, "base"]


async def run_value_on_bench(value, bench, extractor, planner, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor, planner=planner,
        timeless_match_in_scope=value,
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
    # Shared extractor + planner instances so cache is shared across values
    extractor = TemporalExtractor()
    planner = QueryPlanner()

    print(f"\n=== Sweep timeless_match_in_scope over {len(BENCH_NAMES)} "
          f"benches × {len(VALUES)} values ===\n", flush=True)
    # value → list of (bench, metrics)
    rows = {v: [] for v in VALUES}
    for bench in BENCH_NAMES:
        for value in VALUES:
            try:
                m = await run_value_on_bench(value, bench, extractor, planner,
                                             embed_fn, rerank_fn)
            except Exception as e:
                print(f"  ERROR ({bench}, value={value}): {e}", flush=True)
                continue
            if m is None:
                continue
            rows[value].append((bench, m))
        # Print per-bench across values inline for visibility
        line = f"  {bench:30s}"
        for v in VALUES:
            r = next((m for b, m in rows[v] if b == bench), None)
            if r:
                line += f"  v={v}: R@1={r['R@1']:.3f}"
        print(line, flush=True)

    print("\n=== Macro per value ===", flush=True)
    for v in VALUES:
        if not rows[v]:
            continue
        n = len(rows[v])
        r1 = sum(m["R@1"] for _, m in rows[v]) / n
        r5 = sum(m["R@5"] for _, m in rows[v]) / n
        print(f"  timeless_match={v}:  macro R@1={r1:.4f}  macro R@5={r5:.4f}  n={n}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
