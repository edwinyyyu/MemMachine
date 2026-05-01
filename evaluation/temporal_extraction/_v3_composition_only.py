"""Composition-bench-only smoke run for composition_eval_v3 to verify
the prompt fix (open_lower/open_upper semantics + 'from' as closed).

Reuses run_bench from composition_eval_v3.
"""

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

from composition_eval_v3 import aggregate_per_type, run_bench
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

    results = await run_bench(
        "composition",
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
        "edge-composition",
        reranker,
        planner,
    )
    variants = ["rerank_only", "regex_stack", "planner_v2_stack"]
    per_type = aggregate_per_type(results, variants)
    print("\nComposition by type (R@1):")
    print(f"{'type':6s}  {'rerank':>8s}  {'regex':>8s}  {'planner_v2':>10s}")
    for t in ["A", "B", "C", "D", "E", "ALL"]:
        if t not in per_type:
            continue
        a = per_type[t]
        n = a["n"]
        ro = a["rerank_only"]["R@1"]
        re_ = a["regex_stack"]["R@1"]
        pl = a["planner_v2_stack"]["R@1"]
        print(
            f"{t:6s}  {ro:.3f} ({a['rerank_only']['r1_count']:>2}/{n})  "
            f"{re_:.3f} ({a['regex_stack']['r1_count']:>2}/{n})  "
            f"{pl:.3f} ({a['planner_v2_stack']['r1_count']:>2}/{n})"
        )
    print(f"\nplanner stats: {planner.stats()}")

    # Per-query failure dump (planner_v2_stack misses)
    print("\nPlanner_v2 misses:")
    for r in results:
        if r["planner_v2_stack"] != 1:
            print(
                f"  {r['qid']} (type {r['type']}, rank {r['planner_v2_stack']}): {r['qtext']}"
            )
            print(f"    plan: {r['plan']}")
            print(f"    p2 top5: {r['top5_planner_v2']}")
            print(f"    gold: {r['gold']}")


if __name__ == "__main__":
    asyncio.run(main())
