"""Analyze fulleval results and print tables.

Reads all fulleval_*.json result files and produces per-category tables
and cross-architecture summaries.

Usage:
    uv run python fulleval_analyze.py
"""

import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DATASETS = {
    "synthetic": "Synthetic 19q",
    "puzzle": "Puzzle 16q",
    "advanced": "Advanced 23q",
}

ARCH_SHORT_NAMES = {
    "v15_control": "v15",
    "meta_v2f": "v2f",
    "frontier_v2_iterative": "frontier",
    "retrieve_then_decompose": "ret_dec",
    "gen_check_v2": "gen_chk",
    "decompose_retrieve": "dec_ret",
}

ARCH_ORDER = [
    "v15_control",
    "meta_v2f",
    "frontier_v2_iterative",
    "retrieve_then_decompose",
    "gen_check_v2",
    "decompose_retrieve",
]


def load_results() -> dict[str, dict[str, list[dict]]]:
    """Load all available fulleval result files."""
    all_data: dict[str, dict[str, list[dict]]] = {}
    for ds in DATASETS:
        all_data[ds] = {}
        for arch in ARCH_ORDER:
            path = RESULTS_DIR / f"fulleval_{ds}_{arch}.json"
            if path.exists():
                with open(path) as f:
                    all_data[ds][arch] = json.load(f)
    return all_data


def print_table(
    dataset_name: str, ds_label: str, results: dict[str, list[dict]], budget: int = 20
):
    """Print per-category table for one dataset."""
    if not results:
        print(f"\n  No results for {ds_label}")
        return

    # Get categories from any available architecture
    cat_counts: dict[str, int] = defaultdict(int)
    first_arch = next(iter(results.values()))
    for r in first_arch:
        cat_counts[r["category"]] += 1

    available_archs = [a for a in ARCH_ORDER if a in results]

    print(f"\n{'=' * 120}")
    print(f"DATASET: {ds_label} | r@{budget}")
    print(f"{'=' * 120}")

    # Header
    header = f"{'Category':<28} | {'Baseline':>8}"
    for an in available_archs:
        header += f" | {ARCH_SHORT_NAMES.get(an, an[:7]):>8}"
    header += " | Best"
    print(header)
    print("-" * len(header))

    overall_arch_sums: dict[str, float] = defaultdict(float)
    overall_baseline_sum = 0.0
    total_q = 0

    for cat in sorted(cat_counts.keys()):
        n = cat_counts[cat]

        # Baseline from first available arch
        first_results = results[available_archs[0]]
        cat_results = [r for r in first_results if r["category"] == cat]
        bl_vals = [r["baseline_recalls"][f"r@{budget}"] for r in cat_results]
        bl_mean = sum(bl_vals) / len(bl_vals) if bl_vals else 0
        overall_baseline_sum += bl_mean * n
        total_q += n

        row = f"{cat} ({n}q)"
        row = f"{row:<28} | {bl_mean:>8.2f}"

        best_val = -1.0
        best_arch = ""
        for an in available_archs:
            cr = [r for r in results[an] if r["category"] == cat]
            if cr:
                vals = [r["arch_recalls"][f"r@{budget}"] for r in cr]
                v = sum(vals) / len(vals)
            else:
                v = 0.0
            overall_arch_sums[an] += v * n
            row += f" | {v:>8.2f}"
            if v > best_val:
                best_val = v
                best_arch = an

        row += f" | {ARCH_SHORT_NAMES.get(best_arch, best_arch[:7])}"
        print(row)

    # Overall
    print("-" * len(header))
    bl_avg = overall_baseline_sum / total_q if total_q else 0
    row = f"{'OVERALL':<28} | {bl_avg:>8.2f}"
    best_overall_val = -1.0
    best_overall_arch = ""
    for an in available_archs:
        a_avg = overall_arch_sums[an] / total_q if total_q else 0
        row += f" | {a_avg:>8.2f}"
        if a_avg > best_overall_val:
            best_overall_val = a_avg
            best_overall_arch = an
    row += f" | {ARCH_SHORT_NAMES.get(best_overall_arch, best_overall_arch[:7])}"
    print(row)
    print()


def print_cross_arch_summary(
    all_data: dict[str, dict[str, list[dict]]], budget: int = 20
):
    """Print which architecture is best for each category across all datasets."""
    print(f"\n{'=' * 120}")
    print(f"CROSS-ARCHITECTURE SUMMARY: Best architecture per category (r@{budget})")
    print(f"{'=' * 120}")

    print(
        f"\n{'Dataset':<12} {'Category':<28} {'Best Arch':<12} {'Score':>6} {'2nd Best':<12} {'Score':>6} {'Baseline':>8} {'Delta':>8}"
    )
    print("-" * 110)

    arch_win_counts: dict[str, int] = defaultdict(int)

    for ds_name in ["synthetic", "puzzle", "advanced"]:
        ds_results = all_data.get(ds_name, {})
        if not ds_results:
            continue

        available_archs = [a for a in ARCH_ORDER if a in ds_results]
        if not available_archs:
            continue

        cat_counts: dict[str, int] = defaultdict(int)
        first_arch = available_archs[0]
        for r in ds_results[first_arch]:
            cat_counts[r["category"]] += 1

        for cat in sorted(cat_counts.keys()):
            n = cat_counts[cat]

            # Baseline
            cat_results = [r for r in ds_results[first_arch] if r["category"] == cat]
            bl_vals = [r["baseline_recalls"][f"r@{budget}"] for r in cat_results]
            bl_mean = sum(bl_vals) / len(bl_vals) if bl_vals else 0

            # Per-arch
            arch_scores: list[tuple[str, float]] = []
            for an in available_archs:
                cr = [r for r in ds_results[an] if r["category"] == cat]
                if cr:
                    vals = [r["arch_recalls"][f"r@{budget}"] for r in cr]
                    arch_scores.append((an, sum(vals) / len(vals)))
                else:
                    arch_scores.append((an, 0.0))

            arch_scores.sort(key=lambda x: x[1], reverse=True)
            best_arch, best_score = arch_scores[0]
            second_arch, second_score = (
                arch_scores[1] if len(arch_scores) > 1 else ("", 0)
            )
            arch_win_counts[best_arch] += 1

            delta = best_score - bl_mean

            print(
                f"{ds_name:<12} {f'{cat} ({n}q)':<28} "
                f"{ARCH_SHORT_NAMES.get(best_arch, best_arch[:10]):<12} {best_score:>6.2f} "
                f"{ARCH_SHORT_NAMES.get(second_arch, second_arch[:10]):<12} {second_score:>6.2f} "
                f"{bl_mean:>8.2f} {delta:>+8.2f}"
            )

    print("\n--- Win counts (categories where each arch is best) ---")
    for an in sorted(arch_win_counts, key=arch_win_counts.get, reverse=True):
        print(f"  {ARCH_SHORT_NAMES.get(an, an)}: {arch_win_counts[an]}")


def main():
    all_data = load_results()

    # Count available data
    total_files = 0
    for ds in DATASETS:
        for arch in ARCH_ORDER:
            if arch in all_data.get(ds, {}):
                total_files += 1
    total_expected = len(DATASETS) * len(ARCH_ORDER)
    print(f"Loaded {total_files}/{total_expected} result files")

    # Per-dataset tables
    for ds_name, ds_label in DATASETS.items():
        ds_results = all_data.get(ds_name, {})
        if ds_results:
            print_table(ds_name, ds_label, ds_results, budget=20)
            print_table(ds_name, ds_label, ds_results, budget=50)

    # Cross-architecture summary
    print_cross_arch_summary(all_data, budget=20)
    print_cross_arch_summary(all_data, budget=50)


if __name__ == "__main__":
    main()
