"""Smoke test for the Q1 retrieval ablation: composition bench only."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

for _k in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
):
    os.environ.pop(_k, None)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

from _v3_q1_retrieval_ablation import (
    MODES,
    aggregate_overall,
    run_bench_ablation,
)
from query_planner_v2 import QueryPlanner


async def main():
    print("Loading cross-encoder...", flush=True)
    from memmachine_server.common.reranker.cross_encoder_reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerParams,
    )
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    reranker = CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=ce,
            max_input_length=512,
        )
    )
    planner = QueryPlanner()

    results = await run_bench_ablation(
        "composition",
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
        "edge-composition",
        reranker,
        planner,
    )
    overall = aggregate_overall(results, MODES)
    print(f"\nComposition ({overall['n']} queries):")
    for m in MODES:
        d = overall[m]
        print(f"  {m:8s}: R@1 = {d['R@1']:.3f}  ({d['r1_count']}/{overall['n']})")

    print("\nPer-query (mode hits):")
    print(
        f"  {'qid':18s} {'type':6s} {'extremum':10s} "
        f"{'n_filt':>7s}  " + "  ".join(f"{m:>6s}" for m in MODES)
    )
    for r in results:
        print(
            f"  {r['qid'][:18]:18s} {r['type']:6s} "
            f"{r['plan_extremum']!s:10s} {r['n_eligible_filt']:>7d}  "
            + "  ".join(
                f"{('-' if r[f'hit_{m}'] is None else r[f'hit_{m}']):>6}" for m in MODES
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
