"""Full 36-bench A/B: V1 vs V7-Direct (no V7-Legacy).

Validates the SPEC §13 criteria on the direct-planner V7.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._direct_full_ab
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from temporal_retrieval_min import Doc as DocV1
from temporal_retrieval_min import TemporalRetriever
from temporal_retrieval_v7 import Doc as DocV7
from temporal_retrieval_v7 import TemporalRetrieverV7Direct

from temporal_retrieval.research._common import (
    make_embed_fn,
    setup_env,
)
from temporal_retrieval_v7.research._full_ab import (
    BENCH_NAMES,
    _load_bench,
    _metrics,
    make_cosine_rerank_fn,
)

setup_env()


async def run_one_bench(bench: str, embed_fn, rerank_fn) -> dict | None:
    docs_jsonl, queries, gold = _load_bench(bench)
    if docs_jsonl is None:
        return None
    docs_v1 = [DocV1(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]
    docs_v7 = [DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
               for d in docs_jsonl]
    v1 = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    vd = TemporalRetrieverV7Direct(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await v1.index(docs_v1)
    await vd.index(docs_v7)
    v1_rk, vd_rk = {}, {}
    for q in queries:
        r1 = await v1.query(q["text"], q["ref_time"], k=10)
        rD = await vd.query(q["text"], q["ref_time"], k=10)
        v1_rk[q["query_id"]] = [r.doc_id for r in r1]
        vd_rk[q["query_id"]] = [r.doc_id for r in rD]
    return {"v1": _metrics(v1_rk, gold), "v7_direct": _metrics(vd_rk, gold)}


async def main():
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== V1 vs V7-Direct over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    header = (
        f"{'bench':28s}  {'V1 R@1':>8s} {'D R@1':>8s} {'ΔR@1':>8s}  "
        f"{'V1 R@5':>8s} {'D R@5':>8s} {'ΔR@5':>8s}  {'n':>4s}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    rows = {}
    for bench in BENCH_NAMES:
        try:
            res = await run_one_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:28s}  SKIPPED", flush=True)
            continue
        rows[bench] = res
        v1m, vDm = res["v1"], res["v7_direct"]
        d1 = vDm["R@1"] - v1m["R@1"]
        d5 = vDm["R@5"] - v1m["R@5"]
        marker = "+" if d1 > 0.05 else "*" if d1 < -0.05 else " "
        print(
            f"{marker} {bench:26s}  {v1m['R@1']:>8.3f} {vDm['R@1']:>8.3f} {d1:>+8.3f}  "
            f"{v1m['R@5']:>8.3f} {vDm['R@5']:>8.3f} {d5:>+8.3f}  {v1m['n']:>4d}",
            flush=True,
        )
    if rows:
        n = len(rows)
        v1_r1 = sum(r["v1"]["R@1"] for r in rows.values()) / n
        vD_r1 = sum(r["v7_direct"]["R@1"] for r in rows.values()) / n
        v1_r5 = sum(r["v1"]["R@5"] for r in rows.values()) / n
        vD_r5 = sum(r["v7_direct"]["R@5"] for r in rows.values()) / n
        v1_a5 = sum(r["v1"]["all_R@5"] for r in rows.values()) / n
        vD_a5 = sum(r["v7_direct"]["all_R@5"] for r in rows.values()) / n
        print("-" * len(header), flush=True)
        print(
            f"  {'MACRO':26s}  {v1_r1:>8.3f} {vD_r1:>8.3f} {vD_r1-v1_r1:>+8.3f}  "
            f"{v1_r5:>8.3f} {vD_r5:>8.3f} {vD_r5-v1_r5:>+8.3f}  n_benches={n}",
            flush=True,
        )
        print(
            f"  {'(all_R@5)':26s}  {' ':8s} {' ':8s} {' ':8s}  "
            f"{v1_a5:>8.3f} {vD_a5:>8.3f} {vD_a5-v1_a5:>+8.3f}",
            flush=True,
        )
    out = Path(__file__).resolve().parent.parent / "ab_v1_vs_v7_direct.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
