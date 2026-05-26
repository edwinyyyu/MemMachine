"""V1 vs V7 on the new V7-targeted benches (v7_compound_hard and any
others added). Inherits the cosine reranker and the harness wiring
from _full_ab.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._new_benches_ab
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from temporal_retrieval.research._common import (
    make_embed_fn,
    setup_env,
)
from temporal_retrieval_v7.research._full_ab import (
    make_cosine_rerank_fn,
    run_one_bench,
)

setup_env()

NEW_BENCHES = [
    "v7_compound_hard",
    "v7_doc_directional",
]


async def main():
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== V1 vs V7 — new benches ({len(NEW_BENCHES)}) ===\n")
    print(f"{'bench':28s}  {'V1 R@1':>8s} {'V7 R@1':>8s} {'ΔR@1':>8s}  "
          f"{'V1 R@5':>8s} {'V7 R@5':>8s} {'ΔR@5':>8s}  "
          f"{'V1 a@5':>8s} {'V7 a@5':>8s}  n", flush=True)
    print("-" * 115)
    rows = {}
    for bench in NEW_BENCHES:
        try:
            res = await run_one_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:28s}  SKIPPED", flush=True)
            continue
        v1, v7 = res
        rows[bench] = {"v1": v1, "v7": v7}
        d1 = v7["R@1"] - v1["R@1"]
        d5 = v7["R@5"] - v1["R@5"]
        print(
            f"  {bench:26s}  {v1['R@1']:>8.3f} {v7['R@1']:>8.3f} {d1:>+8.3f}  "
            f"{v1['R@5']:>8.3f} {v7['R@5']:>8.3f} {d5:>+8.3f}  "
            f"{v1['all_R@5']:>8.3f} {v7['all_R@5']:>8.3f}  {v1['n']}",
            flush=True,
        )
    out = Path(__file__).resolve().parent.parent / "ab_new_benches.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
