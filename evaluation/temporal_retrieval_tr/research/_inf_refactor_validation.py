"""Validate the int-sentinel → _Inf refactor: full 42-bench run at
production defaults (Copeland 0.40 + raw cosine + pool-scaled match).

Must match the pre-refactor numbers byte-for-byte (we changed only the
internal representation of ±∞; scoring math is identical).

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._inf_refactor_validation
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn,
    metrics,
)

setup_env()


async def run_bench(bench: str, embed_fn, rerank_fn) -> dict | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    rk = {}
    for q in queries:
        r = await vd.query(q["text"], q["ref_time"], k=10)
        rk[q["query_id"]] = [x.doc_id for x in r]
    m = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return m


async def main() -> None:
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== _Inf refactor validation over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    print(f"{'bench':30s}  {'R@1':>7s} {'R@5':>7s} {'R@10':>7s} {'n':>4s}", flush=True)
    print("-" * 60, flush=True)
    rows = {}
    for bench in BENCH_NAMES:
        try:
            m = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:30s}  ERROR: {e}", flush=True)
            continue
        if m is None:
            print(f"{bench:30s}  SKIPPED", flush=True)
            continue
        rows[bench] = m
        print(f"{bench:30s}  {m['R@1']:>7.3f} {m['R@5']:>7.3f} "
              f"{m['R@10']:>7.3f} {m['n']:>4d}", flush=True)
    if rows:
        n = len(rows)
        macro_r1 = sum(m["R@1"] for m in rows.values()) / n
        macro_r5 = sum(m["R@5"] for m in rows.values()) / n
        macro_r10 = sum(m["R@10"] for m in rows.values()) / n
        print("-" * 60, flush=True)
        print(f"{'MACRO':30s}  {macro_r1:>7.4f} {macro_r5:>7.4f} "
              f"{macro_r10:>7.4f}  n={n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
