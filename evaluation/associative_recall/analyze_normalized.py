"""Analyze normalized evaluation results and produce summary tables.

Reads results from normalized evaluations and produces:
1. Properly normalized baseline vs associative comparison
2. Per-category breakdowns
3. Failure analysis
4. Budget-curve analysis (recall vs segment budget)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50, 100, 150]


def load_results(label: str) -> list[dict] | None:
    path = RESULTS_DIR / f"normalized_{label}.json"
    if not path.exists():
        print(f"Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def analyze_single(results: list[dict], label: str) -> dict:
    """Analyze a single configuration's results."""
    print(f"\n{'='*100}")
    print(f"ANALYSIS: {label} ({len(results)} questions)")
    print(f"{'='*100}")

    budget_labels = [f"r@{b}" for b in BUDGETS] + ["r@actual"]

    # Overall
    print(f"\n  OVERALL:")
    for lbl in budget_labels:
        b_vals = [r["baseline_recalls"].get(lbl, 0) for r in results]
        a_vals = [r["assoc_recalls"].get(lbl, 0) for r in results]
        b_mean = sum(b_vals) / len(b_vals)
        a_mean = sum(a_vals) / len(a_vals)
        delta = a_mean - b_mean
        # Count wins/ties/losses
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = len(b_vals) - wins - losses
        print(f"    {lbl:>10s}: baseline={b_mean:.3f}  assoc={a_mean:.3f}  "
              f"delta={delta:+.3f}  W/T/L={wins}/{ties}/{losses}")

    avg_frac = sum(r["retrieval_fraction"] for r in results) / len(results)
    avg_total = sum(r["total_assoc_retrieved"] for r in results) / len(results)
    print(f"\n    Avg retrieval fraction: {avg_frac:.2%}")
    print(f"    Avg total assoc segments: {avg_total:.0f}")

    # Per category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print(f"\n  PER CATEGORY (delta at each budget):")
    print(f"    {'category':30s} {'n':>3s}", end="")
    for lbl in budget_labels:
        print(f"  {'D-'+lbl:>10s}", end="")
    print()
    print("    " + "-" * 90)

    for cat in sorted(by_cat):
        rows = by_cat[cat]
        print(f"    {cat:30s} {len(rows):>3d}", end="")
        for lbl in budget_labels:
            b = sum(r["baseline_recalls"].get(lbl, 0) for r in rows) / len(rows)
            a = sum(r["assoc_recalls"].get(lbl, 0) for r in rows) / len(rows)
            d = a - b
            print(f"  {d:>+10.3f}", end="")
        print()

    # Questions where assoc does WORSE than baseline
    print(f"\n  QUESTIONS WHERE ASSOCIATIVE IS WORSE (at r@50):")
    for r in results:
        b50 = r["baseline_recalls"].get("r@50", 0)
        a50 = r["assoc_recalls"].get("r@50", 0)
        if b50 > a50 + 0.001:
            print(f"    Q: {r['question'][:80]}")
            print(f"      cat={r['category']} conv={r['conversation_id']} "
                  f"src={r['source_chat_ids']}")
            print(f"      baseline r@50={b50:.3f}  assoc r@50={a50:.3f}  "
                  f"delta={a50-b50:+.3f}")
            print(f"      total_assoc={r['total_assoc_retrieved']} "
                  f"({r['retrieval_fraction']:.0%})")

    # Questions where assoc HELPS at r@20
    print(f"\n  QUESTIONS WHERE ASSOCIATIVE HELPS (at r@20):")
    for r in results:
        b20 = r["baseline_recalls"].get("r@20", 0)
        a20 = r["assoc_recalls"].get("r@20", 0)
        if a20 > b20 + 0.001:
            print(f"    Q: {r['question'][:80]}")
            print(f"      cat={r['category']} "
                  f"baseline r@20={b20:.3f}  assoc r@20={a20:.3f}  "
                  f"delta={a20-b20:+.3f}")

    return {
        "label": label,
        "num_questions": len(results),
        "overall": {
            lbl: {
                "baseline": sum(r["baseline_recalls"].get(lbl, 0) for r in results) / len(results),
                "assoc": sum(r["assoc_recalls"].get(lbl, 0) for r in results) / len(results),
            }
            for lbl in budget_labels
        },
    }


def compare_configs(configs: list[tuple[str, list[dict]]]) -> None:
    """Compare multiple configurations side by side."""
    print(f"\n{'='*100}")
    print("COMPARISON ACROSS CONFIGURATIONS")
    print(f"{'='*100}")

    budget_labels = [f"r@{b}" for b in BUDGETS]

    print(f"\n  {'Config':30s}", end="")
    for lbl in budget_labels:
        print(f"  {'B-'+lbl:>7s} {'A-'+lbl:>7s} {'D':>6s}", end="")
    print(f"  {'AvgSegs':>7s}")
    print("  " + "-" * 100)

    for label, results in configs:
        avg_segs = sum(r["total_assoc_retrieved"] for r in results) / len(results)
        print(f"  {label:30s}", end="")
        for lbl in budget_labels:
            b = sum(r["baseline_recalls"].get(lbl, 0) for r in results) / len(results)
            a = sum(r["assoc_recalls"].get(lbl, 0) for r in results) / len(results)
            d = a - b
            print(f"  {b:>7.3f} {a:>7.3f} {d:>+6.3f}", end="")
        print(f"  {avg_segs:>7.0f}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="*", default=[])
    args = parser.parse_args()

    if not args.labels:
        # Find all normalized result files
        files = sorted(RESULTS_DIR.glob("normalized_*.json"))
        labels = [f.stem.replace("normalized_", "") for f in files]
    else:
        labels = args.labels

    configs = []
    for label in labels:
        results = load_results(label)
        if results:
            configs.append((label, results))

    if not configs:
        print("No results found")
        return

    for label, results in configs:
        analyze_single(results, label)

    if len(configs) > 1:
        compare_configs(configs)


if __name__ == "__main__":
    main()
