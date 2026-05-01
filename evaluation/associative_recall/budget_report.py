"""Aggregates budget_aware_eval result files into a comparison table.

Usage:
    uv run python budget_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


BUDGET_K20 = [
    "baseline_20",
    "v15_tight_20",
    "v2f_tight_20",
    "pure_cue_20",
    "single_cue_20",
]
BUDGET_K50 = [
    "baseline_50",
    "v15_tight_50",
    "v2f_tight_50",
    "wide_cue_50",
    "gencheck_50",
]
BUDGET_K100 = ["baseline_100", "v2f_100"]
ALL_ARCHES = BUDGET_K20 + BUDGET_K50 + BUDGET_K100

DATASETS = ["locomo_30q", "synthetic_19q", "puzzle_16q", "advanced_23q"]
SHORT_DS = {
    "locomo_30q": "LoCoMo",
    "synthetic_19q": "Synth",
    "puzzle_16q": "Puzzle",
    "advanced_23q": "Advanced",
}


def load() -> dict:
    out: dict[tuple[str, str], dict] = {}
    for f in RESULTS_DIR.glob("budget_*.json"):
        name = f.stem  # budget_<arch>_<dataset>
        if name == "budget_all_summaries":
            continue
        # Split off trailing dataset token
        for ds in DATASETS:
            suffix = f"_{ds}"
            if name.endswith(suffix):
                arch = name[len("budget_") : -len(suffix)]
                data = json.load(open(f))
                out[(arch, ds)] = data["summary"]
                break
    return out


def print_abs_table(summaries: dict) -> None:
    print("\n" + "=" * 110)
    print("ABSOLUTE RECALL @ K (exact budget)")
    print("=" * 110)
    for group_name, arch_list in [
        ("K=20", BUDGET_K20),
        ("K=50", BUDGET_K50),
        ("K=100", BUDGET_K100),
    ]:
        print(f"\n--- Budget {group_name} ---")
        header = f"{'Architecture':<22s}" + "".join(
            f"{SHORT_DS[ds]:>12s}" for ds in DATASETS
        )
        print(header)
        print("-" * len(header))
        for arch in arch_list:
            row = f"{arch:<22s}"
            for ds in DATASETS:
                s = summaries.get((arch, ds))
                if s is None:
                    row += f"{'—':>12s}"
                else:
                    rec = s.get("mean_recall", 0.0)
                    under = s.get("under_budget", 0)
                    mark = "*" if under > 0 else " "
                    row += f"{rec:>11.3f}{mark}"
            print(row)


def print_delta_table(summaries: dict) -> None:
    print("\n" + "=" * 110)
    print("DELTA vs cosine baseline at same K (positive = arch beats baseline)")
    print("=" * 110)
    baselines = {20: "baseline_20", 50: "baseline_50", 100: "baseline_100"}
    for group_name, arch_list in [
        ("K=20", BUDGET_K20),
        ("K=50", BUDGET_K50),
        ("K=100", BUDGET_K100),
    ]:
        budget = int(group_name.split("=")[1])
        base = baselines[budget]
        print(f"\n--- Budget {group_name} (vs {base}) ---")
        header = f"{'Architecture':<22s}" + "".join(
            f"{SHORT_DS[ds]:>12s}" for ds in DATASETS
        )
        print(header)
        print("-" * len(header))
        for arch in arch_list:
            row = f"{arch:<22s}"
            for ds in DATASETS:
                s = summaries.get((arch, ds))
                b = summaries.get((base, ds))
                if s is None or b is None:
                    row += f"{'—':>12s}"
                else:
                    d = s["mean_recall"] - b["mean_recall"]
                    row += f"{d:>+12.3f}"
            print(row)


def print_winner_per_budget(summaries: dict) -> None:
    print("\n" + "=" * 110)
    print("WINNER per budget (best arch averaged across datasets)")
    print("=" * 110)
    for group_name, arch_list in [
        ("K=20", BUDGET_K20),
        ("K=50", BUDGET_K50),
        ("K=100", BUDGET_K100),
    ]:
        budget = int(group_name.split("=")[1])
        print(f"\n--- Budget {group_name} ---")
        means: list[tuple[str, float, int]] = []
        for arch in arch_list:
            vals = []
            for ds in DATASETS:
                s = summaries.get((arch, ds))
                if s is not None:
                    vals.append(s["mean_recall"])
            if vals:
                means.append((arch, sum(vals) / len(vals), len(vals)))
        means.sort(key=lambda t: -t[1])
        for arch, m, n in means:
            print(f"  {arch:<25s} mean recall={m:.3f}  (n_datasets={n})")


def print_per_category(summaries: dict) -> None:
    print("\n" + "=" * 110)
    print("PER-CATEGORY recall at K=20 per dataset")
    print("=" * 110)
    for ds in DATASETS:
        print(f"\n--- {ds} ---")
        # gather all categories across all arches' summaries for this dataset
        cats: set[str] = set()
        for arch in BUDGET_K20:
            s = summaries.get((arch, ds))
            if s and "per_category" in s:
                cats.update(s["per_category"].keys())
        cats_sorted = sorted(cats)
        if not cats_sorted:
            print("  (no results)")
            continue
        header = f"{'category':<30s}"
        for arch in BUDGET_K20:
            header += f"{arch[:10]:>12s}"
        print(header)
        print("-" * len(header))
        for cat in cats_sorted:
            row = f"{cat[:30]:<30s}"
            for arch in BUDGET_K20:
                s = summaries.get((arch, ds))
                pc = s.get("per_category", {}) if s else {}
                if cat in pc:
                    row += f"{pc[cat]['mean_recall']:>12.3f}"
                else:
                    row += f"{'—':>12s}"
            print(row)


def main() -> None:
    summaries = load()
    print_abs_table(summaries)
    print_delta_table(summaries)
    print_winner_per_budget(summaries)
    print_per_category(summaries)


if __name__ == "__main__":
    main()
