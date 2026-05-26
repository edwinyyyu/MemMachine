"""Validate principle-based PASS1 (current production) vs legacy v5.1 PASS1
on bench data we never optimized against.

The 7-bench validation in `_validate_best_prompt.py` covered the benches we
DID optimize on (composition, adversarial, mixed_cue, realq_v2, era_refs,
conjunctive_temporal, multi_te_doc). Those benches were used to design or
validate prompts during development; we are partially overfit to them.

This script validates on three held-out benches that were never fed to the
optimizer or used as a baseline reference during prompt-design iteration:

- temporal_essential (150 docs / 25 queries): "essentially temporal" Qs
  with deictic + offset patterns ("early April 2024", "Q4 2022",
  "team retrospective in early May 2024").
- negation_temporal (75 docs / 15 queries): negation-bearing queries
  ("What did I do not in 2023?", "outside the holiday season").
- hard_bench (size unknown): the historic hard-bench from the research
  era; designed to break early planner versions.

If principle ≥ legacy on held-out benches, that's evidence the principle
generalizes beyond the specific patterns we tuned against.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_held_out
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval import extractor as ext_mod
from temporal_retrieval.legacy_prompts import LEGACY_PASS1_SYSTEM

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

HELD_OUT_BENCHES = {
    "temporal_essential": (
        "temporal_essential_docs.jsonl",
        "temporal_essential_queries.jsonl",
        "temporal_essential_gold.jsonl",
    ),
    "negation_temporal": (
        "negation_temporal_docs.jsonl",
        "negation_temporal_queries.jsonl",
        "negation_temporal_gold.jsonl",
    ),
    "hard_bench": (
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
    ),
}

INPUT_COST_PER_M = 0.25
OUTPUT_COST_PER_M = 2.00


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * INPUT_COST_PER_M + (
        output_tokens / 1e6
    ) * OUTPUT_COST_PER_M


async def evaluate_bench(
    bench: str, prompt: str, embed_fn, rerank_fn, label: str
) -> dict:
    docs_file, queries_file, gold_file = HELD_OUT_BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(
        docs_file, queries_file, gold_file
    )
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    extractor = ext_mod.TemporalExtractor(
        cache_subdir=f"heldout_{label}_{bench}", pass1_system=prompt
    )
    retriever = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn, extractor=extractor
    )
    await retriever.index(docs)

    K = 5
    n_eval = 0
    n_r5 = 0
    n_r1 = 0
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        first_gold = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
        if first_gold is not None:
            if first_gold <= 1:
                n_r1 += 1
            if first_gold <= K:
                n_r5 += 1

    stats = retriever.stats()
    in_t = stats["extractor_usage"]["input"]
    out_t = stats["extractor_usage"]["output"]
    n_docs = len(docs)
    intervals = retriever.doc_intervals()
    total_intervals = sum(len(v) for v in intervals.values())
    return {
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "n_eval": n_eval,
        "input_tokens": in_t,
        "output_tokens": out_t,
        "cost_usd": cost_usd(in_t, out_t),
        "intervals_per_doc": total_intervals / max(1, n_docs),
    }


async def main() -> None:
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    bench_order = list(HELD_OUT_BENCHES.keys())
    rows: list[dict] = []
    for bench in bench_order:
        print(f"\n=== bench: {bench} ===", flush=True)
        for label, prompt in (
            ("legacy", LEGACY_PASS1_SYSTEM),
            ("principle", ext_mod.PASS1_SYSTEM),
        ):
            print(f"  {label}...", flush=True)
            r = await evaluate_bench(bench, prompt, embed_fn, rerank_fn, label)
            print(
                f"    R@5={r['R@5']:.3f} R@1={r['R@1']:.3f} "
                f"cost=${r['cost_usd']:.4f} "
                f"in={r['input_tokens']} out={r['output_tokens']} "
                f"iv/doc={r['intervals_per_doc']:.2f} "
                f"n_eval={r['n_eval']}",
                flush=True,
            )
            rows.append({"bench": bench, "label": label, **r})

    print("\n" + "=" * 100, flush=True)
    print(
        f"{'bench':22s} {'label':10s} {'R@5':>6s} {'R@1':>6s} "
        f"{'cost_usd':>10s} {'iv/doc':>7s} {'verdict':>14s}",
        flush=True,
    )
    print("-" * 100, flush=True)
    grand_legacy = 0.0
    grand_principle = 0.0
    for bench in bench_order:
        legacy = next(r for r in rows if r["bench"] == bench and r["label"] == "legacy")
        principle = next(
            r for r in rows if r["bench"] == bench and r["label"] == "principle"
        )
        grand_legacy += legacy["cost_usd"]
        grand_principle += principle["cost_usd"]
        for label, r in (("legacy", legacy), ("principle", principle)):
            verdict = ""
            if label == "principle":
                d = r["R@5"] - legacy["R@5"]
                if d <= -0.005:
                    verdict = f"REGRESS({d:+.3f})"
                elif d >= 0.005:
                    verdict = f"BEAT({d:+.3f})"
                else:
                    verdict = "match"
            print(
                f"{bench:22s} {label:10s} {r['R@5']:>6.3f} {r['R@1']:>6.3f} "
                f"${r['cost_usd']:>9.4f} {r['intervals_per_doc']:>7.2f} {verdict:>14s}",
                flush=True,
            )

    print("-" * 100, flush=True)
    delta = grand_legacy - grand_principle
    pct = delta / grand_legacy * 100 if grand_legacy else 0
    print(
        f"GRAND TOTAL  legacy ${grand_legacy:.4f} | "
        f"principle ${grand_principle:.4f} | "
        f"DELTA ${delta:+.4f} ({pct:+.1f}%)",
        flush=True,
    )

    out_path = ROOT / "held_out_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
