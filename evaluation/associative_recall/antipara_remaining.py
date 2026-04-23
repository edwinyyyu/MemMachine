"""Targeted resume: run meta_v2f + v2f_anti_paraphrase on puzzle_16q and
advanced_23q only. Verbatim variants already demonstrated degenerate behavior
on locomo_30q (r@20 collapsed from 0.756 to 0.361), so we skip them on the
remaining datasets per decision rule (abandon once variant loses on LoCoMo).

Also emits the final `antipara_cue_study.md` / `.json` using whatever
results files are present.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from antipara_cue_eval import (
    ARCH_CLASSES,
    evaluate_question,
    render_markdown,
    run_one,
)
from associative_recall import Segment, SegmentStore
from fair_backfill_eval import (
    BUDGETS,
    DATA_DIR,
    DATASETS,
    RESULTS_DIR,
    load_dataset,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# (arch_name, dataset) pairs we still need to run
TO_RUN: list[tuple[str, str]] = [
    ("meta_v2f", "puzzle_16q"),
    ("v2f_anti_paraphrase", "puzzle_16q"),
    ("meta_v2f", "advanced_23q"),
    ("v2f_anti_paraphrase", "advanced_23q"),
]


def load_existing(arch_name: str, ds_name: str) -> dict | None:
    path = RESULTS_DIR / f"antipara_{arch_name}_{ds_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main() -> None:
    # Group by dataset so we reuse the store
    by_ds: dict[str, list[str]] = {}
    for arch, ds in TO_RUN:
        if load_existing(arch, ds) is not None:
            print(f"skip {arch}/{ds}: exists")
            continue
        by_ds.setdefault(ds, []).append(arch)

    for ds_name, arch_names in by_ds.items():
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )
        for arch_name in arch_names:
            cls = ARCH_CLASSES[arch_name]
            arch = cls(store)
            results, summary, by_cat = run_one(
                arch_name, arch, ds_name, questions
            )
            out_path = RESULTS_DIR / f"antipara_{arch_name}_{ds_name}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "arch": arch_name,
                        "dataset": ds_name,
                        "summary": summary,
                        "category_breakdown": by_cat,
                        "results": results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            print(f"  Saved: {out_path}")

    # Consolidate: read every antipara_<arch>_<ds>.json we have
    all_summaries: dict = {}
    all_results: dict = {}
    arch_names_present: list[str] = []
    for arch_name in (
        "meta_v2f",
        "v2f_anti_paraphrase",
        "v2f_verbatim_quote",
        "v2f_anti_paraphrase_verbatim",
    ):
        for ds_name in DATASETS:
            data = load_existing(arch_name, ds_name)
            if data is None:
                continue
            if arch_name not in arch_names_present:
                arch_names_present.append(arch_name)
            all_summaries.setdefault(arch_name, {})[ds_name] = {
                "summary": data["summary"],
                "category_breakdown": data["category_breakdown"],
            }
            all_results.setdefault(arch_name, {})[ds_name] = data["results"]

    summary_path = RESULTS_DIR / "antipara_cue_study.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved summary JSON: {summary_path}")

    md = render_markdown(all_summaries, all_results, arch_names_present)
    md_path = RESULTS_DIR / "antipara_cue_study.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved summary MD: {md_path}")

    # Console table
    print("\n" + "=" * 100)
    print("ANTI-PARAPHRASE CUE STUDY SUMMARY")
    print("=" * 100)
    header = (
        f"{'Arch':<32s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} {'W/T/L@20':>10s} "
        f"{'base@50':>8s} {'arch@50':>8s} {'d@50':>7s} {'W/T/L@50':>10s}"
    )
    print(header)
    print("-" * len(header))
    for arch_name in arch_names_present:
        for ds_name in DATASETS:
            s = all_summaries.get(arch_name, {}).get(ds_name, {}).get("summary")
            if not s:
                continue
            print(
                f"{arch_name:<32s} {ds_name:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} {s['W/T/L_r@20']:>10s} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['delta_r@50']:>+7.3f} {s['W/T/L_r@50']:>10s}"
            )


if __name__ == "__main__":
    main()
