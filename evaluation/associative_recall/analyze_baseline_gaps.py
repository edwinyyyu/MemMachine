"""Analyze where baseline cosine retrieval fails at r@20.

Identifies questions where the baseline misses source turns,
grouped by category. These are the questions where associative
recall has the most opportunity to help.
"""

import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def analyze_baseline_gaps(results_file: str) -> None:
    path = RESULTS_DIR / results_file
    with open(path) as f:
        results = json.load(f)

    print(f"Analyzing: {results_file} ({len(results)} questions)")
    print(f"{'='*80}")

    # Categorize by baseline recall at r@20
    perfect = []  # baseline r@20 = 1.0
    partial = []  # 0 < baseline r@20 < 1.0
    zero = []     # baseline r@20 = 0.0

    for r in results:
        b20 = r["baseline_recalls"]["r@20"]
        if b20 >= 0.999:
            perfect.append(r)
        elif b20 > 0.001:
            partial.append(r)
        else:
            zero.append(r)

    print(f"\nBaseline r@20 breakdown:")
    print(f"  Perfect (r@20 = 1.0): {len(perfect)} ({len(perfect)/len(results):.0%})")
    print(f"  Partial (0 < r@20 < 1): {len(partial)} ({len(partial)/len(results):.0%})")
    print(f"  Zero (r@20 = 0.0): {len(zero)} ({len(zero)/len(results):.0%})")

    # By category
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print(f"\nPer-category baseline r@20:")
    print(f"  {'category':35s} {'n':>3s} {'mean':>6s} {'perf':>5s} {'zero':>5s} {'#src':>5s}")
    print("  " + "-" * 65)
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        mean_b20 = sum(r["baseline_recalls"]["r@20"] for r in rows) / len(rows)
        n_perfect = sum(1 for r in rows if r["baseline_recalls"]["r@20"] >= 0.999)
        n_zero = sum(1 for r in rows if r["baseline_recalls"]["r@20"] < 0.001)
        avg_src = sum(r["num_source_turns"] for r in rows) / len(rows)
        print(f"  {cat:35s} {len(rows):>3d} {mean_b20:>6.3f} {n_perfect:>5d} {n_zero:>5d} {avg_src:>5.1f}")

    # Show the zero-recall questions
    if zero:
        print(f"\n  Questions with baseline r@20 = 0:")
        for r in zero:
            a20 = r["assoc_recalls"]["r@20"]
            print(f"    [{r['category']:30s}] assoc r@20={a20:.3f} src={r['source_chat_ids']}")
            print(f"      {r['question'][:80]}")

    # Show partial recall questions
    if partial:
        print(f"\n  Questions with 0 < baseline r@20 < 1:")
        for r in sorted(partial, key=lambda x: x["baseline_recalls"]["r@20"]):
            b20 = r["baseline_recalls"]["r@20"]
            a20 = r["assoc_recalls"]["r@20"]
            print(f"    [{r['category']:30s}] b={b20:.3f} a={a20:.3f} d={a20-b20:+.3f} src={r['source_chat_ids']}")
            print(f"      {r['question'][:80]}")


if __name__ == "__main__":
    import sys
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "normalized_v8_nr1_h1_beam_ext.json",
        "normalized_v8_nr1_h1_locomo_ext.json",
    ]
    for f in files:
        analyze_baseline_gaps(f)
        print()
