"""Tabulate memory-index experiment results.

Prints two tables:
  1. Overall variant × dataset matrix (r@20 / r@50 + W/T/L).
  2. Per-category breakdown for the interesting categories on each dataset.

Reads results/memindex_*.json produced by memory_index.py.

Usage:
    uv run python analyze_memindex.py
"""

from __future__ import annotations

import json
from pathlib import Path


R = Path(__file__).resolve().parent / "results"

VARIANTS = [
    "v15_with_index",
    "v2f_v2_with_index",
    "index_only",
    "v2f_without_index",
]
DATASETS = ["locomo_30q", "synthetic_19q", "puzzle_16q", "advanced_23q"]

# Per-dataset categories to break down. (all categories shown if empty list)
INTERESTING = {
    "locomo_30q": ["locomo_temporal", "locomo_single_hop", "locomo_multi_hop"],
    "synthetic_19q": [
        "proactive", "completeness", "conjunction", "inference", "procedural",
        "control",
    ],
    "puzzle_16q": [
        "logic_constraint", "absence_inference", "state_change", "contradiction",
        "open_exploration", "sequential_chain",
    ],
    "advanced_23q": [
        "evolving_terminology", "proactive", "perspective_separation",
        "unfinished_business", "negation", "frequency_detection",
        "constraint_propagation", "consistency_checking",
        "quantitative_aggregation",
    ],
}


def load(variant: str, ds: str) -> dict | None:
    p = R / f"memindex_{variant}_{ds}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def print_overall() -> None:
    print("=" * 110)
    print("OVERALL (variant × dataset) — fair-backfill, K=20 and K=50")
    print("=" * 110)
    header = (
        f"{'Variant':<22s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} {'W/T/L@20':>10s} "
        f"{'base@50':>8s} {'arch@50':>8s} {'d@50':>7s} {'W/T/L@50':>10s}"
    )
    print(header)
    print("-" * len(header))
    for v in VARIANTS:
        for ds in DATASETS:
            data = load(v, ds)
            if not data:
                continue
            s = data["summary"]
            print(
                f"{v:<22s} {ds:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} {s['W/T/L_r@20']:>10s} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['delta_r@50']:>+7.3f} {s['W/T/L_r@50']:>10s}"
            )
        print()


def print_per_category() -> None:
    print("=" * 110)
    print("PER-CATEGORY (selected)")
    print("=" * 110)
    header = (
        f"{'category':<22s} {'variant':<22s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} {'d@50':>7s} "
        f"{'W/T/L@20':>10s}"
    )
    for ds in DATASETS:
        print(f"\n# {ds}")
        print(header)
        print("-" * len(header))
        cats = INTERESTING.get(ds, [])
        for cat in cats:
            for v in VARIANTS:
                data = load(v, ds)
                if not data:
                    continue
                cb = data.get("category_breakdown", {})
                c = cb.get(cat)
                if not c:
                    continue
                print(
                    f"{cat:<22s} {v:<22s} "
                    f"{c['baseline_r@20']:>8.3f} {c['arch_r@20']:>8.3f} "
                    f"{c['delta_r@20']:>+7.3f} {c['delta_r@50']:>+7.3f} "
                    f"{c['W/T/L_r@20']:>10s}"
                )
            print()


def print_head_to_head() -> None:
    """For each dataset, head-to-head comparison: best variant at r@20, r@50."""
    print("=" * 110)
    print("HEAD-TO-HEAD: best variant per dataset")
    print("=" * 110)
    for ds in DATASETS:
        print(f"\n{ds}")
        rows = []
        for v in VARIANTS:
            data = load(v, ds)
            if not data:
                continue
            s = data["summary"]
            rows.append((v, s["delta_r@20"], s["delta_r@50"], s["W/T/L_r@20"], s["W/T/L_r@50"]))
        rows.sort(key=lambda r: r[2], reverse=True)  # sort by d@50
        print(f"  {'variant':<24s} {'d@20':>7s} {'W/T/L@20':>10s} {'d@50':>7s} {'W/T/L@50':>10s}")
        for name, d20, d50, w20, w50 in rows:
            print(f"  {name:<24s} {d20:>+7.3f} {w20:>10s} {d50:>+7.3f} {w50:>10s}")


def main() -> None:
    print_overall()
    print_per_category()
    print_head_to_head()


if __name__ == "__main__":
    main()
