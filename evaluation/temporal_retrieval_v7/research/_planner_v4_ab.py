"""A/B: planner direct_v3 vs v4 (absolute+anaphoric fix).

v4 extends the anaphora rule: drop only the anaphoric part, still emit
every absolute/deictic scope the query also contains. See
_planner_v4_probe.py for the qualitative viability check.

Runs composition (target) + anaphora/deictic-heavy benches
(speculative_anchors, realq_deictic, realq) for regression. Reports
R@1 / R@5 per bench.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._planner_v4_ab
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_v7 import Doc as DocV7
from temporal_retrieval_v7 import TemporalRetrieverV7Direct
from temporal_retrieval_v7.planner_direct import DirectQueryPlanner

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_v7.research._full_ab import (
    _load_bench,
    _metrics,
    make_cosine_rerank_fn,
)
from temporal_retrieval_v7.research._planner_v4_probe import PROMPT_V4

setup_env()

BENCHES = ["composition", "speculative_anchors", "realq_deictic", "realq"]


async def run_variant(bench: str, prompt_template, cache_subdir,
                      embed_fn, rerank_fn) -> dict:
    docs_jsonl, queries, gold = _load_bench(bench)
    docs = [DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    planner = DirectQueryPlanner(prompt_template=prompt_template,
                                 cache_subdir=cache_subdir)
    vd = TemporalRetrieverV7Direct(embed_fn=embed_fn, rerank_fn=rerank_fn,
                                   planner=planner)
    await vd.index(docs)
    rk = {}
    for q in queries:
        r = await vd.query(q["text"], q["ref_time"], k=10)
        rk[q["query_id"]] = [x.doc_id for x in r]
    m = _metrics(rk, gold)
    del vd, planner, docs, docs_jsonl
    gc.collect()
    return m


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print("=== planner v3 vs v4 (absolute+anaphoric fix) ===\n", flush=True)
    print(f"{'bench':22s}  {'v3 R@1':>8s} {'v4 R@1':>8s} {'ΔR@1':>8s}  "
          f"{'v3 R@5':>8s} {'v4 R@5':>8s}  {'n':>4s}", flush=True)
    print("-" * 70, flush=True)
    rows = {}
    for bench in BENCHES:
        try:
            m3 = await run_variant(bench, None, f"abv3_{bench}",
                                   embed_fn, rerank_fn)
            m4 = await run_variant(bench, PROMPT_V4, f"abv4_{bench}",
                                   embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:22s}  ERROR: {e}", flush=True)
            continue
        rows[bench] = (m3, m4)
        d1 = m4["R@1"] - m3["R@1"]
        print(f"{bench:22s}  {m3['R@1']:>8.3f} {m4['R@1']:>8.3f} {d1:>+8.3f}  "
              f"{m3['R@5']:>8.3f} {m4['R@5']:>8.3f}  {m3['n']:>4d}", flush=True)
    if rows:
        n = len(rows)
        v3 = sum(m3["R@1"] for m3, _ in rows.values()) / n
        v4 = sum(m4["R@1"] for _, m4 in rows.values()) / n
        print("-" * 70, flush=True)
        print(f"{'MACRO':22s}  {v3:>8.3f} {v4:>8.3f} {v4-v3:>+8.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
