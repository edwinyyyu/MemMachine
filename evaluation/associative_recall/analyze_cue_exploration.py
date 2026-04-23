"""Analyze cue format exploration results.

Compares all prompt versions tested in the cue format exploration,
showing per-version, per-category, and per-question results.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50, 100]


def load_results(label: str) -> list[dict] | None:
    path = RESULTS_DIR / f"normalized_{label}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_metrics(results: list[dict], budget_label: str = "r@20") -> dict:
    """Compute aggregate metrics for a set of results."""
    b_vals = [r["baseline_recalls"].get(budget_label, 0) for r in results]
    a_vals = [r["assoc_recalls"].get(budget_label, 0) for r in results]

    b_mean = sum(b_vals) / len(b_vals) if b_vals else 0
    a_mean = sum(a_vals) / len(a_vals) if a_vals else 0
    delta = a_mean - b_mean

    wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
    losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
    ties = len(b_vals) - wins - losses

    return {
        "baseline": b_mean,
        "assoc": a_mean,
        "delta": delta,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "n": len(results),
    }


def main() -> None:
    # Define configs to compare
    configs = [
        ("v8", "v8_nr1_h1_{bench}_ext_full", "v8 keyword-dense (baseline)"),
        ("v8-30q", "v8_nr1_h1_{bench}_ext", "v8 (30q subset, ref only)"),
        ("v10", "v10_nr1_h1_{bench}_ext", "v10 HyDE hypothetical answer"),
        ("v11", "v11_nr1_h1_{bench}_ext", "v11 Utterance-style"),
        ("v12", "v12_nr1_h1_{bench}_ext", "v12 Narrative paragraphs"),
        ("v13", "v13_nr1_h1_{bench}_ext", "v13 Freeform (no format)"),
        ("v14", "v14_nr1_h1_{bench}_ext", "v14 Contrastive"),
        ("v15", "v15_nr1_h1_{bench}_ext", "v15 Self-monitoring"),
        ("v16", "v16_nr1_h1_{bench}_ext", "v16 Scratchpad"),
    ]

    for benchmark in ["beam", "locomo"]:
        print(f"\n{'='*100}")
        print(f"  {benchmark.upper()} BENCHMARK")
        print(f"{'='*100}")

        print(f"\n  {'Config':35s} {'n':>3s}", end="")
        for b in BUDGETS:
            print(f"  {'delta':>7s} {'W/T/L':>9s}", end="")
        print(f"  {'AvgSeg':>6s}")
        print("  " + "-" * 95)

        loaded_configs = []
        for short_name, label_template, desc in configs:
            label = label_template.format(bench=benchmark)
            results = load_results(label)
            if results is None:
                continue

            loaded_configs.append((short_name, desc, results))

            avg_segs = sum(r["total_assoc_retrieved"] for r in results) / len(results)
            print(f"  {desc:35s} {len(results):>3d}", end="")
            for b in BUDGETS:
                m = compute_metrics(results, f"r@{b}")
                print(f"  {m['delta']:>+7.3f} {m['wins']}/{m['ties']}/{m['losses']:>5s}", end="")
            print(f"  {avg_segs:>6.0f}")

        # Per-category breakdown for loaded configs
        if loaded_configs:
            print(f"\n  Per-category deltas at r@20:")
            # Get all categories
            all_cats = sorted(set(
                r["category"]
                for _, _, results in loaded_configs
                for r in results
            ))

            print(f"    {'category':35s}", end="")
            for short_name, _, _ in loaded_configs:
                print(f"  {short_name:>8s}", end="")
            print()
            print("    " + "-" * (35 + 10 * len(loaded_configs)))

            for cat in all_cats:
                print(f"    {cat:35s}", end="")
                for short_name, _, results in loaded_configs:
                    cat_results = [r for r in results if r["category"] == cat]
                    if cat_results:
                        m = compute_metrics(cat_results, "r@20")
                        print(f"  {m['delta']:>+8.3f}", end="")
                    else:
                        print(f"  {'N/A':>8s}", end="")
                print()

    # Example cues comparison
    print(f"\n{'='*100}")
    print("  EXAMPLE CUES BY PROMPT VERSION")
    print(f"{'='*100}")

    # Pick a few questions and show cues from each version
    example_questions = []
    for short_name, label_template, desc in configs:
        label = label_template.format(bench="beam")
        results = load_results(label)
        if results is None:
            continue
        for r in results[:10]:
            q_key = (r["question"][:60], r["conversation_id"])
            if q_key not in [eq[0] for eq in example_questions]:
                example_questions.append((q_key, []))
            for eq_key, eq_versions in example_questions:
                if eq_key == q_key:
                    hop1_cues = []
                    for h in r.get("hop_details", []):
                        if h["hop"] == 1:
                            hop1_cues = h["cues"]
                    eq_versions.append((short_name, hop1_cues, r))
                    break

    for (q_text, conv_id), versions in example_questions[:5]:
        print(f"\n  Q: {q_text}...")
        for short_name, cues, r in versions:
            b20 = r["baseline_recalls"]["r@20"]
            a20 = r["assoc_recalls"]["r@20"]
            print(f"    [{short_name:>5s}] delta={a20-b20:+.3f} | {', '.join(c[:60] for c in cues[:2])}")


if __name__ == "__main__":
    main()
