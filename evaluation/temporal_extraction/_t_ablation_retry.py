"""Retry composition + hard_bench, merge into existing T_ablation.json/md."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from query_planner import QueryPlanner
from t_ablation_eval import aggregate, run_bench, write_md


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

    benches_def = [
        (
            "composition",
            "composition_docs.jsonl",
            "composition_queries.jsonl",
            "composition_gold.jsonl",
            "edge-composition",
        ),
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
            "v7l-hard_bench",
        ),
    ]
    variants = ["A_filter_only", "B_filter_T", "C_no_T_retrieval", "D_T_wider"]

    out_path = ROOT / "results" / "T_ablation.json"
    existing = json.load(open(out_path))

    for nm, dp, qp, gp, cl in benches_def:
        try:
            results = await run_bench(nm, dp, qp, gp, cl, reranker, planner)
            overall = aggregate(results, variants)
            existing["benches"][nm] = {
                "n": overall["n"],
                "overall": overall,
                "per_q": results,
            }
            for v in variants:
                d = existing["benches"][nm]["overall"][v]
                print(
                    f"  {v:20s} R@1={d['R@1']:.3f} ({d['r1_count']}/{existing['benches'][nm]['n']})",
                    flush=True,
                )
        except Exception:
            import traceback

            traceback.print_exc()

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"Wrote {out_path}", flush=True)

    md_path = ROOT / "results" / "T_ablation.md"
    md_path.write_text(write_md(existing, md_path))
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
