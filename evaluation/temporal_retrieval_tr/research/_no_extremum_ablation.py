"""Isolate additive vs Copeland-pairwise WITHOUT extremum recency tournament.

The extremum tournament `_copeland_rerank` bundles temporal-overlap match
AND recency bonus, so the usual additive-vs-copeland_pairwise A/B is
confounded — both methods get short-circuited on extremum queries.

Here we disable extremum dispatch (extremum_copeland=False) so all
queries (extremum and non-extremum) go through the configured method.
This isolates the policy choice on the entire bench.

CONFIGS:
  base_only_strict:                 floor (no temporal scoring at all)
  additive  (no extremum):          base + match for all queries
  copeland_pairwise (no extremum):  Copeland pairs for all queries

Compare to baselines with extremum ON:
  base_only (extremum on):          0.7989
  additive+0.8 (extremum on):       0.7989
  copeland_pairwise (extremum on):  0.8044
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
    ("additive (no extr)",          "additive",          0.8, False),
    ("copeland_pairwise (no extr)", "copeland_pairwise", 0.8, False),
]


async def run_bench(label, ranking, tm, extr_on, bench,
                    extractor, planner, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor, planner=planner,
        ranking_method=ranking,
        timeless_match_in_scope=tm,
        extremum_copeland=extr_on,
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

    by_config = {label: [] for label, _, _, _ in CONFIGS}
    for bench in BENCH_NAMES:
        for label, ranking, tm, extr_on in CONFIGS:
            try:
                m = await run_bench(label, ranking, tm, extr_on, bench,
                                    extractor, planner, embed_fn, rerank_fn)
            except Exception as e:
                print(f"  ERROR ({bench}, {label}): {e}", flush=True)
                continue
            if m is None:
                continue
            by_config[label].append((bench, m))

    print("\n=== Macro per config (extremum recency Copeland OFF) ===")
    for label, _, _, _ in CONFIGS:
        rows = by_config[label]
        if not rows:
            continue
        n = len(rows)
        r1 = sum(m["R@1"] for _, m in rows) / n
        r5 = sum(m["R@5"] for _, m in rows) / n
        print(f"  {label:35s}  R@1={r1:.4f}  R@5={r5:.4f}  n={n}")

    print("\n=== Reference baselines (extremum ON) ===")
    print(f"  {'base_only_strict (no temporal)':35s}  R@1=0.6283  R@5=0.9037")
    print(f"  {'pure-semantic':35s}  R@1=0.6241  R@5=0.9064")
    print(f"  {'base_only (extremum on)':35s}  R@1=0.7989  R@5=0.9611")
    print(f"  {'additive+0.8 (extremum on)':35s}  R@1=0.7989  R@5=0.9611")
    print(f"  {'copeland_pairwise (extremum on)':35s}  R@1=0.8044  R@5=0.9618")


if __name__ == "__main__":
    asyncio.run(main())
