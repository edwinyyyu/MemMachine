"""Complete v2f_register_inferred evaluation on missing datasets.

Runs register_inferred and v2f (for parity) on synthetic_19q, puzzle_16q,
advanced_23q at K=20 and K=50 using the existing domain_agnostic caches.

Outputs per-(variant, dataset) JSON files at:
    results/domain_agnostic_{variant}_{dataset}.json
(overwriting existing K=20-only files with new K=20+K=50 data).
"""

import json

from domain_agnostic import (
    RESULTS_DIR,
    build_variant,
    load_dataset,
    run_variant_parallel,
    summarize_by_category,
    summarize_results,
)


def run(variant_name: str, ds_name: str, budgets: list[int]) -> dict:
    store, questions = load_dataset(ds_name)
    print(
        f"\n=== {variant_name} on {ds_name} "
        f"(n={len(questions)}, segs={len(store.segments)}) @ K={budgets} ==="
    )
    arch = build_variant(store, variant_name)
    results = run_variant_parallel(arch, questions, budgets, workers=8)
    arch.save_caches()

    summary = summarize_results(results, variant_name, ds_name, budgets)
    by_cat = summarize_by_category(results, budgets)
    record = {
        "summary": summary,
        "category_breakdown": by_cat,
        "results": results,
    }
    out_path = RESULTS_DIR / f"domain_agnostic_{variant_name}_{ds_name}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"Saved: {out_path}")
    for K in budgets:
        print(
            f"  r@{K}: base={summary[f'baseline_r@{K}']:.3f}  "
            f"arch={summary[f'arch_r@{K}']:.3f}  "
            f"delta={summary[f'delta_r@{K}']:+.3f}  "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    return record


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BUDGETS = [20, 50]

    # What the task calls out explicitly: v2f_register_inferred on the 3 non-LoCoMo
    # datasets. Also re-run v2f on the same datasets with the same budgets so
    # baseline comparison is apples-to-apples at both K=20 and K=50.
    plan = [
        ("v2f_register_inferred", "synthetic_19q"),
        ("v2f_register_inferred", "puzzle_16q"),
        ("v2f_register_inferred", "advanced_23q"),
        ("v2f", "synthetic_19q"),
        ("v2f", "puzzle_16q"),
        ("v2f", "advanced_23q"),
    ]

    records = {}
    for variant_name, ds_name in plan:
        records[(variant_name, ds_name)] = run(variant_name, ds_name, BUDGETS)

    # Cross-dataset summary for the two variants of interest
    print("\n" + "=" * 90)
    print("REGISTER_INFERRED vs V2f SUMMARY (fair K-budget, cosine backfill)")
    print("=" * 90)
    for K in BUDGETS:
        print(f"\n--- K={K} ---")
        header = f"{'dataset':<20s} {'v2f':>12s} {'register':>12s} {'delta':>10s}"
        print(header)
        print("-" * len(header))
        for ds_name in ("locomo_30q", "synthetic_19q", "puzzle_16q", "advanced_23q"):
            # locomo comes from existing result file
            if ds_name == "locomo_30q":
                with open(RESULTS_DIR / "domain_agnostic_v2f_locomo_30q.json") as f:
                    v2f = json.load(f)["summary"]
                with open(
                    RESULTS_DIR
                    / "domain_agnostic_v2f_register_inferred_locomo_30q.json"
                ) as f:
                    reg = json.load(f)["summary"]
            else:
                v2f = records[("v2f", ds_name)]["summary"]
                reg = records[("v2f_register_inferred", ds_name)]["summary"]
            key = f"arch_r@{K}"
            if key not in v2f or key not in reg:
                continue
            v = v2f[key]
            r = reg[key]
            print(f"{ds_name:<20s} {v:>12.3f} {r:>12.3f} {r - v:>+10.3f}")


if __name__ == "__main__":
    main()
