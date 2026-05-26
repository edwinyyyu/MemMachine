"""Validate the optimizer's best feasible prompt against the FULL hard-bench
suite to make sure cost-savings on the curated bench don't trade R@5 on the
saturated benches.

Reads ``best_prompt.txt``, runs it on all 4 hard benches plus era_refs +
conjunctive_temporal + multi_te_doc (the 3 saturated standard benches), and
compares to the recorded ablation_proper / ablation_hard baseline.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_best_prompt
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval import extractor as ext_mod

from ._ablation_hard import BENCHES as HARD_BENCHES
from ._ablation_proper import BENCHES as STD_BENCHES
from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

# Combined bench set: standard saturated + hard signal-bearing.
ALL_BENCHES = {**STD_BENCHES, **HARD_BENCHES}

INPUT_COST_PER_M = 0.25
OUTPUT_COST_PER_M = 2.00


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * INPUT_COST_PER_M + (
        output_tokens / 1e6
    ) * OUTPUT_COST_PER_M


def _load_prompt(path: Path) -> str:
    """Read best_prompt.txt, strip leading comment lines starting with #."""
    with open(path) as f:
        lines = f.readlines()
    body_start = 0
    for i, line in enumerate(lines):
        if not line.startswith("#") and line.strip():
            body_start = i
            break
    return "".join(lines[body_start:])


async def evaluate_bench(
    bench: str, prompt: str, embed_fn, rerank_fn, label: str
) -> dict:
    docs_jsonl, queries, gold_rows = load_bench_jsonl(*ALL_BENCHES[bench])
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    extractor = ext_mod.TemporalExtractor(
        cache_subdir=f"validate_{label}_{bench}", pass1_system=prompt
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

    best_prompt = _load_prompt(ROOT / "best_prompt.txt")
    baseline_prompt = ext_mod.PASS1_SYSTEM

    bench_order = list(ALL_BENCHES.keys())
    rows: list[dict] = []
    for bench in bench_order:
        print(f"\n=== bench: {bench} ===", flush=True)
        for label, prompt in (("baseline", baseline_prompt), ("r3v2", best_prompt)):
            print(f"  {label}...", flush=True)
            r = await evaluate_bench(bench, prompt, embed_fn, rerank_fn, label)
            print(
                f"    R@5={r['R@5']:.3f} R@1={r['R@1']:.3f} "
                f"cost=${r['cost_usd']:.4f} "
                f"in={r['input_tokens']} out={r['output_tokens']} "
                f"iv/doc={r['intervals_per_doc']:.2f}",
                flush=True,
            )
            rows.append({"bench": bench, "label": label, **r})

    print("\n" + "=" * 95, flush=True)
    print(
        f"{'bench':22s} {'label':10s} {'R@5':>6s} {'R@1':>6s} "
        f"{'cost_usd':>10s} {'iv/doc':>7s} {'verdict':>10s}",
        flush=True,
    )
    print("-" * 95, flush=True)
    grand_baseline = 0.0
    grand_r3v2 = 0.0
    for bench in bench_order:
        b = next(r for r in rows if r["bench"] == bench and r["label"] == "baseline")
        v = next(r for r in rows if r["bench"] == bench and r["label"] == "r3v2")
        grand_baseline += b["cost_usd"]
        grand_r3v2 += v["cost_usd"]
        for label, r in (("baseline", b), ("r3v2", v)):
            verdict = ""
            if label == "r3v2":
                d = r["R@5"] - b["R@5"]
                if d <= -0.005:
                    verdict = f"REGRESS({d:+.3f})"
                elif d >= 0.005:
                    verdict = f"BEAT({d:+.3f})"
                else:
                    verdict = "match"
            print(
                f"{bench:22s} {label:10s} {r['R@5']:>6.3f} {r['R@1']:>6.3f} "
                f"${r['cost_usd']:>9.4f} {r['intervals_per_doc']:>7.2f} {verdict:>10s}",
                flush=True,
            )

    print("-" * 95, flush=True)
    print(
        f"{'GRAND TOTAL':22s} {'baseline':10s} {'':>6s} {'':>6s} "
        f"${grand_baseline:>9.4f}",
        flush=True,
    )
    print(
        f"{'GRAND TOTAL':22s} {'r3v2':10s} {'':>6s} {'':>6s} "
        f"${grand_r3v2:>9.4f}  "
        f"SAVE=${grand_baseline - grand_r3v2:.4f} "
        f"({(grand_baseline - grand_r3v2) / grand_baseline * 100:+.1f}%)",
        flush=True,
    )

    out_path = ROOT / "best_prompt_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
