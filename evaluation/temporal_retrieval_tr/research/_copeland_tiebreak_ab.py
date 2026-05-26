"""Copeland tertiary tiebreak: sim (base+match) vs base (rerank-only).

The tiebreak only matters in cycle regions (where multiple docs have
equal wins AND equal margin-sums). Cycles are predicted rare with
bonus=0.15 — most queries should give identical rankings between the
two. This A/B confirms whether tiebreak affects R@1 at all on the
production bench suite.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._copeland_tiebreak_ab
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

# (label, copeland_tiebreak)
ARMS: list[tuple[str, str]] = [
    ("tie_sim",  "sim"),
    ("tie_base", "base"),
]


async def run_bench(bench: str, embed_fn, rerank_fn) -> dict | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        copeland_bonus=0.15,
    )
    await vd.index(docs)
    out = {}
    for label, tiebreak in ARMS:
        vd.copeland_tiebreak = tiebreak
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[label] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== Copeland tiebreak: sim vs base over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    hdr = (f"{'bench':28s}  {'tie_sim':>8s} {'tie_base':>9s} {'ΔR@1':>7s}  "
           f"{'R5_sim':>7s} {'R5_base':>8s}  {'n':>4s}")
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
        s, b = res["tie_sim"], res["tie_base"]
        d = b["R@1"] - s["R@1"]
        mark = "+" if d > 0.02 else "*" if d < -0.02 else " "
        print(f"{mark} {bench:26s}  {s['R@1']:>8.3f} {b['R@1']:>9.3f} "
              f"{d:>+7.3f}  {s['R@5']:>7.3f} {b['R@5']:>8.3f}  "
              f"{s['n']:>4d}", flush=True)
    if rows:
        n = len(rows)
        ms = sum(r["tie_sim"]["R@1"] for r in rows.values()) / n
        mb = sum(r["tie_base"]["R@1"] for r in rows.values()) / n
        ms5 = sum(r["tie_sim"]["R@5"] for r in rows.values()) / n
        mb5 = sum(r["tie_base"]["R@5"] for r in rows.values()) / n
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO':26s}  {ms:>8.3f} {mb:>9.3f} {mb-ms:>+7.3f}  "
              f"{ms5:>7.3f} {mb5:>8.3f}  n={n}", flush=True)
        # Per-query identical-ranking check
        # (count how many benches show ANY R@1 difference)
        diffs = sum(1 for r in rows.values()
                    if r["tie_sim"]["R@1"] != r["tie_base"]["R@1"])
        print(f"\nBenches where tiebreak materially changes R@1: {diffs}/{n}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
