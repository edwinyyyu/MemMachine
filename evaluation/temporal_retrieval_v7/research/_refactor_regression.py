"""Regression check after the Clause→flat-refs refactor.

Runs V7-Direct on the 9 discriminating benches (the noisy-OR A/B set)
and prints R@1 / R@5. Compare against the pre-refactor V7-Direct macro
R@1 0.793 on this set. The refactor is mathematically equivalent on
all non-mixed-plan query shapes, so any delta here is from the new
`direct_v3` prompt, not the scoring change.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._refactor_regression
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_min import Doc as DocV1
from temporal_retrieval_min import TemporalRetriever
from temporal_retrieval_v7 import Doc as DocV7
from temporal_retrieval_v7 import TemporalRetrieverV7Direct

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_v7.research._full_ab import (
    _load_bench,
    _metrics,
    make_cosine_rerank_fn,
)

setup_env()

BENCHES = [
    "engagement_disjoint", "edge_conjunctive_temporal", "polarity",
    "sensitivity_curated", "hard_bench", "composition", "axis",
    "v7_compound_hard", "v7_doc_directional",
]


async def run_one(bench: str, embed_fn, rerank_fn) -> dict | None:
    loaded = _load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
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
        rd = await vd.query(q["text"], q["ref_time"], k=10)
        v1_rk[q["query_id"]] = [r.doc_id for r in r1]
        vd_rk[q["query_id"]] = [r.doc_id for r in rd]
    res = {"v1": _metrics(v1_rk, gold), "v7_direct": _metrics(vd_rk, gold)}
    del v1, vd, docs_v1, docs_v7, docs_jsonl
    gc.collect()
    return res


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print("=== Refactor regression: V7-Direct on 9 discriminating benches ===\n",
          flush=True)
    print(f"{'bench':28s}  {'V1 R@1':>8s} {'D R@1':>8s} {'D R@5':>8s}  {'n':>4s}",
          flush=True)
    print("-" * 60, flush=True)
    rows = {}
    for bench in BENCHES:
        try:
            res = await run_one(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:28s}  SKIPPED", flush=True)
            continue
        rows[bench] = res
        v1m, vdm = res["v1"], res["v7_direct"]
        print(f"{bench:28s}  {v1m['R@1']:>8.3f} {vdm['R@1']:>8.3f} "
              f"{vdm['R@5']:>8.3f}  {v1m['n']:>4d}", flush=True)
    if rows:
        n = len(rows)
        vd_r1 = sum(r["v7_direct"]["R@1"] for r in rows.values()) / n
        vd_r5 = sum(r["v7_direct"]["R@5"] for r in rows.values()) / n
        print("-" * 60, flush=True)
        print(f"{'MACRO V7-Direct':28s}  {'':>8s} {vd_r1:>8.3f} {vd_r5:>8.3f}",
              flush=True)
        print("\n(pre-refactor V7-Direct macro R@1 on this set: 0.793)",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
