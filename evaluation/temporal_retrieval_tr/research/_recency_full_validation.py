"""Full 38-bench validation: recency_weight 1.0 vs 1.5.

The 9-bench sweep (_recency_weight_ab.py) found w=1.5 lifts composition
R@1 by +0.08 with zero detected regression on 8 other benches. This
runs the full bench list at w=1.0 vs w=1.5 to catch any extremum
queries on benches I didn't include in the targeted sweep.

The retriever's recency_weight only fires when plan.latest_intent or
plan.earliest_intent is set, so non-extremum benches are provably
unchanged. The validation's job: confirm no extremum-query regression
on the un-tested benches.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._recency_full_validation
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES,
    load_bench,
    make_cosine_rerank_fn,
    metrics,
)

setup_env()

WEIGHTS = [1.0, 1.5]


async def run_bench(bench: str, embed_fn, rerank_fn) -> dict | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    out = {}
    for w in WEIGHTS:
        vd.recency_weight = w
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[w] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== recency_weight 1.0 vs 1.5 over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    hdr = f"{'bench':28s}  {'w=1.0':>7s} {'w=1.5':>7s} {'ΔR@1':>7s}  " \
          f"{'R@5 w1':>7s} {'R@5 w1.5':>9s}  {'n':>4s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    for bench in BENCH_NAMES:
        try:
            res = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:28s}  SKIPPED", flush=True)
            continue
        rows[bench] = res
        m1, m15 = res[1.0], res[1.5]
        d = m15["R@1"] - m1["R@1"]
        mark = "+" if d > 0.02 else "*" if d < -0.02 else " "
        print(f"{mark} {bench:26s}  {m1['R@1']:>7.3f} {m15['R@1']:>7.3f} "
              f"{d:>+7.3f}  {m1['R@5']:>7.3f} {m15['R@5']:>9.3f}  "
              f"{m1['n']:>4d}", flush=True)
    if rows:
        n = len(rows)
        a1 = sum(r[1.0]["R@1"] for r in rows.values()) / n
        a15 = sum(r[1.5]["R@1"] for r in rows.values()) / n
        b1 = sum(r[1.0]["R@5"] for r in rows.values()) / n
        b15 = sum(r[1.5]["R@5"] for r in rows.values()) / n
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO':26s}  {a1:>7.3f} {a15:>7.3f} {a15-a1:>+7.3f}  "
              f"{b1:>7.3f} {b15:>9.3f}  n={n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
