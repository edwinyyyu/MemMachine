"""Sweep gpt-5-nano + gpt-5.4-nano (Responses API) at minimal/low/medium/high.

Asymmetric metric: false-reject rate (gold=KEEP, pred=REJECT) is the
primary signal. We report it across N replicates per cell to measure
variance.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from classifier import classify_many
from dotenv import load_dotenv
from test_set import all_items
from test_set_hard import all_hard

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


CELLS = [
    # (model, api, reasoning_effort)
    ("gpt-5-mini", "chat", "low"),  # baseline reference
    ("gpt-5-nano", "responses", "minimal"),
    ("gpt-5-nano", "responses", "low"),
    ("gpt-5-nano", "responses", "medium"),
    ("gpt-5.4-nano", "responses", "minimal"),
    ("gpt-5.4-nano", "responses", "low"),
    ("gpt-5.4-nano", "responses", "medium"),
]


async def run_cell(
    items: list[tuple[str, str]],
    *,
    model: str,
    api: str,
    reasoning: str,
    n_replicates: int,
    concurrency: int,
) -> dict:
    texts = [t for _, t in items]
    labels = [lab for lab, _ in items]
    n_keep = sum(1 for x in labels if x == "KEEP")
    n_rej = sum(1 for x in labels if x == "REJECT")

    fr_rates: list[float] = []
    fk_rates: list[float] = []
    fr_examples_union: set[str] = set()
    elapsed_total = 0.0

    for _ in range(n_replicates):
        t0 = time.monotonic()
        results = await classify_many(
            texts,
            model=model,
            prompt="v1",
            reasoning_effort=reasoning,
            concurrency=concurrency,
            api=api,
        )
        elapsed_total += time.monotonic() - t0

        fr = sum(
            1
            for (lab, _), r in zip(items, results, strict=False)
            if lab == "KEEP" and r.label == "REJECT"
        )
        fk = sum(
            1
            for (lab, _), r in zip(items, results, strict=False)
            if lab == "REJECT" and r.label == "KEEP"
        )
        for (lab, t), r in zip(items, results, strict=False):
            if lab == "KEEP" and r.label == "REJECT":
                fr_examples_union.add(t)
        fr_rates.append(fr / max(n_keep, 1))
        fk_rates.append(fk / max(n_rej, 1))

    return {
        "model": model,
        "api": api,
        "reasoning": reasoning,
        "fr_rates": fr_rates,
        "fk_rates": fk_rates,
        "fr_examples_union": sorted(fr_examples_union),
        "elapsed_per_run_s": elapsed_total / n_replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()

    items = all_items() + all_hard()
    n_keep = sum(1 for lab, _ in items if lab == "KEEP")
    n_rej = sum(1 for lab, _ in items if lab == "REJECT")
    print(f"Test set: {n_keep} KEEP / {n_rej} REJECT (n={len(items)})")
    print(f"Replicates per cell: {args.replicates}")
    print()

    rows = []
    for model, api, reasoning in CELLS:
        try:
            row = asyncio.run(
                run_cell(
                    items,
                    model=model,
                    api=api,
                    reasoning=reasoning,
                    n_replicates=args.replicates,
                    concurrency=args.concurrency,
                )
            )
        except Exception as e:
            row = {
                "model": model,
                "api": api,
                "reasoning": reasoning,
                "error": f"{type(e).__name__}: {e}",
            }
        rows.append(row)

    print(
        f"{'model':16s} {'api':10s} {'reasoning':9s} "
        f"{'FR mean':>8s} {'FR worst':>9s} {'FK mean':>8s} {'sec/run':>8s}"
    )
    print("-" * 80)
    for r in rows:
        if "error" in r:
            print(
                f"{r['model']:16s} {r['api']:10s} {r['reasoning']:9s}  ERROR: {r['error']}"
            )
            continue
        fr = r["fr_rates"]
        fk = r["fk_rates"]
        print(
            f"{r['model']:16s} {r['api']:10s} {r['reasoning']:9s} "
            f"{sum(fr) / len(fr):>7.1%} {max(fr):>8.1%} "
            f"{sum(fk) / len(fk):>7.1%} {r['elapsed_per_run_s']:>7.1f}s"
        )

    print()
    for r in rows:
        if "error" in r:
            continue
        if not r["fr_examples_union"]:
            continue
        print(
            f"=== {r['model']} {r['api']} {r['reasoning']} false-reject union ({len(r['fr_examples_union'])}):"
        )
        for t in r["fr_examples_union"]:
            print(f"  - {t!r}")


if __name__ == "__main__":
    main()
