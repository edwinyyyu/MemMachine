"""Compare associative recall results across prompt versions and models."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

RECALL_DEPTHS = ["r@5", "r@10", "r@20", "r@50", "r@all"]


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def summarize(results: list[dict], label: str) -> dict:
    from collections import defaultdict

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    summary = {"label": label, "categories": {}, "overall": {}}
    all_b = {d: [] for d in RECALL_DEPTHS}
    all_a = {d: [] for d in RECALL_DEPTHS}

    for cat in sorted(by_cat):
        rows = by_cat[cat]
        cat_b = {}
        cat_a = {}
        for d in RECALL_DEPTHS:
            bvals = [r["baseline_recalls"][d] for r in rows]
            avals = [r["assoc_recalls"][d] for r in rows]
            cat_b[d] = sum(bvals) / len(bvals)
            cat_a[d] = sum(avals) / len(avals)
            all_b[d].extend(bvals)
            all_a[d].extend(avals)
        summary["categories"][cat] = {
            "count": len(rows),
            "baseline": cat_b,
            "associative": cat_a,
            "avg_incremental": sum(r["num_incremental_hits"] for r in rows) / len(rows),
        }

    for d in RECALL_DEPTHS:
        summary["overall"]["baseline_" + d] = sum(all_b[d]) / len(all_b[d])
        summary["overall"]["assoc_" + d] = sum(all_a[d]) / len(all_a[d])
        summary["overall"]["delta_" + d] = (
            sum(all_a[d]) / len(all_a[d]) - sum(all_b[d]) / len(all_b[d])
        )

    return summary


def main() -> None:
    result_files = sorted(RESULTS_DIR.glob("results_*.json"))
    if not result_files:
        print("No result files found.")
        return

    summaries = []
    for path in result_files:
        label = path.stem.replace("results_", "")
        results = load_results(path)
        summary = summarize(results, label)
        summaries.append(summary)

    print("=" * 90)
    print("COMPARISON OF ASSOCIATIVE RECALL CONFIGURATIONS")
    print("=" * 90)
    print()

    header = f"{'Config':25s}  {'B-r@all':>7s}  {'A-r@all':>7s}  {'Delta':>7s}  {'B-r@20':>7s}  {'A-r@20':>7s}  {'Delta':>7s}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        o = s["overall"]
        b_all = o["baseline_r@all"]
        a_all = o["assoc_r@all"]
        d_all = o["delta_r@all"]
        b_20 = o["baseline_r@20"]
        a_20 = o["assoc_r@20"]
        d_20 = o["delta_r@20"]
        print(
            f"{s['label']:25s}  {b_all:>7.3f}  {a_all:>7.3f}  {d_all:>+7.3f}"
            f"  {b_20:>7.3f}  {a_20:>7.3f}  {d_20:>+7.3f}"
        )

    print()
    print("PER-CATEGORY BREAKDOWN (r@all):")
    print()

    all_cats = set()
    for s in summaries:
        all_cats.update(s["categories"].keys())

    for cat in sorted(all_cats):
        print(f"  {cat}:")
        for s in summaries:
            if cat in s["categories"]:
                c = s["categories"][cat]
                b = c["baseline"]["r@all"]
                a = c["associative"]["r@all"]
                d = a - b
                inc = c["avg_incremental"]
                print(
                    f"    {s['label']:25s}  B={b:.3f}  A={a:.3f}  "
                    f"delta={d:+.3f}  avg_incr={inc:.1f}"
                )
        print()


if __name__ == "__main__":
    main()
