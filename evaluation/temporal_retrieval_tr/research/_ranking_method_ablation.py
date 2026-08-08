"""Ablation: compare ranking methods on full bench.

- base_only: ignore temporal entirely (floor)
- additive+0.8: shipped pre-Copeland (timeless_match=0.8 constant)
- copeland_pairwise: current ship (heterogeneous pair rule)

If base_only is close to copeland_pairwise, temporal scoring isn't
doing much (and the Copeland advantage might be artifacts of
disabling temporal in mixed pairs).
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


CONFIGS = [
    ("base_only", "additive", 0.8),  # ranking_method, ignored, ignored — uses base_only
    ("additive+0.8", "additive", 0.8),
    ("copeland_pairwise", "copeland_pairwise", 0.8),
]


async def run_bench(label, ranking, tm, bench, extractor, planner,
                    embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor, planner=planner,
        ranking_method=ranking, timeless_match_in_scope=tm,
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

    # Per-config macros
    by_config = {label: [] for label, _, _ in CONFIGS}
    for bench in BENCH_NAMES:
        for label, ranking, tm in CONFIGS:
            try:
                m = await run_bench(label, ranking, tm, bench,
                                    extractor, planner, embed_fn, rerank_fn)
            except Exception as e:
                print(f"  ERROR ({bench}, {label}): {e}", flush=True)
                continue
            if m is None:
                continue
            by_config[label].append((bench, m))

    print("\n=== Macro per ranking method ===")
    for label, _, _ in CONFIGS:
        rows = by_config[label]
        if not rows:
            continue
        n = len(rows)
        r1 = sum(m["R@1"] for _, m in rows) / n
        r5 = sum(m["R@5"] for _, m in rows) / n
        print(f"  {label:25s}  R@1={r1:.4f}  R@5={r5:.4f}  n={n}")


if __name__ == "__main__":
    asyncio.run(main())
