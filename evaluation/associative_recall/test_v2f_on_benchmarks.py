"""Test V2f prompt variant vs V15 control on synthetic, puzzle, and advanced benchmarks.

Evaluates whether V2f's two additions to V15:
  1. "If the question implies MULTIPLE items or asks 'all/every', keep searching..."
  2. "Do NOT write questions ('Did you mention X?'). Write text that would actually appear..."
provide advantages on harder retrieval scenarios beyond LoCoMo.

Usage:
    uv run python test_v2f_on_benchmarks.py [--force] [--verbose]
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EmbeddingCache,
    LLMCache,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI
from prompt_optimization import (
    META_V2F_PROMPT,
    V15_CONTROL_PROMPT,
    MetaV2Variant,
    run_variant,
    summarize,
    summarize_by_category,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Cache that reads from all existing caches, writes to synth_test_*
# ---------------------------------------------------------------------------
class SynthTestEmbeddingCache(EmbeddingCache):
    """Reads all existing caches, writes to synth_test-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "arch_embedding_cache.json",
            "agent_embedding_cache.json",
            "frontier_embedding_cache.json",
            "meta_embedding_cache.json",
            "optim_embedding_cache.json",
            "synth_test_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "synth_test_embedding_cache.json"
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class SynthTestLLMCache(LLMCache):
    """Reads all existing caches, writes to synth_test-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "arch_llm_cache.json",
            "agent_llm_cache.json",
            "tree_llm_cache.json",
            "frontier_llm_cache.json",
            "meta_llm_cache.json",
            "optim_llm_cache.json",
            "synth_test_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "synth_test_llm_cache.json"
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class SynthTestMetaV2Variant(MetaV2Variant):
    """MetaV2Variant that uses synth_test caches."""

    def __init__(
        self, store: SegmentStore, prompt_template: str, client: OpenAI | None = None
    ):
        super().__init__(store, prompt_template, client)
        # Override caches with synth_test versions
        self.embedding_cache = SynthTestEmbeddingCache()
        self.llm_cache = SynthTestLLMCache()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "synth",
        "npz": "segments_synthetic.npz",
        "questions_file": "questions_synthetic.json",
        "label": "Synthetic (19q)",
    },
    {
        "name": "puzzle",
        "npz": "segments_puzzle.npz",
        "questions_file": "questions_puzzle.json",
        "label": "Puzzle (16q)",
    },
    {
        "name": "advanced",
        "npz": "segments_advanced.npz",
        "questions_file": "questions_advanced.json",
        "label": "Advanced (23q)",
    },
]

VARIANTS = [
    ("v15_control", V15_CONTROL_PROMPT),
    ("v2f", META_V2F_PROMPT),
]


def compare_variants(
    v15_results: list[dict],
    v2f_results: list[dict],
    dataset_label: str,
) -> dict:
    """Head-to-head comparison of v15_control vs v2f at r@20."""
    n = len(v15_results)
    assert n == len(v2f_results), "Results must have same length"

    # Overall comparison at r@20
    v15_vals = [r["arch_recalls"]["r@20"] for r in v15_results]
    v2f_vals = [r["arch_recalls"]["r@20"] for r in v2f_results]

    wins = sum(1 for a, b in zip(v2f_vals, v15_vals) if a > b + 0.001)
    losses = sum(1 for a, b in zip(v2f_vals, v15_vals) if b > a + 0.001)
    ties = n - wins - losses

    v15_mean = sum(v15_vals) / n
    v2f_mean = sum(v2f_vals) / n

    comparison = {
        "dataset": dataset_label,
        "n": n,
        "v15_r@20": round(v15_mean, 4),
        "v2f_r@20": round(v2f_mean, 4),
        "delta_v2f_vs_v15": round(v2f_mean - v15_mean, 4),
        "W/T/L": f"{wins}/{ties}/{losses}",
    }

    # Per-category comparison
    by_cat: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for v15_r, v2f_r in zip(v15_results, v2f_results):
        by_cat[v15_r["category"]].append((v15_r, v2f_r))

    cat_comparisons = {}
    for cat, pairs in sorted(by_cat.items()):
        cn = len(pairs)
        v15_cat = [p[0]["arch_recalls"]["r@20"] for p in pairs]
        v2f_cat = [p[1]["arch_recalls"]["r@20"] for p in pairs]
        base_cat = [p[0]["baseline_recalls"]["r@20"] for p in pairs]
        cw = sum(1 for a, b in zip(v2f_cat, v15_cat) if a > b + 0.001)
        cl = sum(1 for a, b in zip(v2f_cat, v15_cat) if b > a + 0.001)
        cat_comparisons[cat] = {
            "n": cn,
            "baseline_r@20": round(sum(base_cat) / cn, 4),
            "v15_r@20": round(sum(v15_cat) / cn, 4),
            "v2f_r@20": round(sum(v2f_cat) / cn, 4),
            "delta_v2f_vs_v15": round(sum(v2f_cat) / cn - sum(v15_cat) / cn, 4),
            "delta_v15_vs_base": round(sum(v15_cat) / cn - sum(base_cat) / cn, 4),
            "delta_v2f_vs_base": round(sum(v2f_cat) / cn - sum(base_cat) / cn, 4),
            "W/T/L": f"{cw}/{cn - cw - cl}/{cl}",
        }

    comparison["per_category"] = cat_comparisons
    return comparison


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test V2f vs V15 control on synthetic/puzzle/advanced benchmarks"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_comparisons = []

    for dataset in DATASETS:
        ds_name = dataset["name"]
        ds_label = dataset["label"]

        print(f"\n{'#' * 80}")
        print(f"# DATASET: {ds_label}")
        print(f"{'#' * 80}")

        # Load data
        store = SegmentStore(data_dir=DATA_DIR, npz_name=dataset["npz"])
        with open(DATA_DIR / dataset["questions_file"]) as f:
            questions = json.load(f)

        # Ensure all questions have question_index
        for i, q in enumerate(questions):
            if "question_index" not in q:
                q["question_index"] = i

        print(f"Loaded {len(store.segments)} segments, {len(questions)} questions")

        variant_results = {}

        for variant_name, prompt_template in VARIANTS:
            results_file = RESULTS_DIR / f"{ds_name}_{variant_name}.json"

            if results_file.exists() and not args.force:
                print(f"\nLoading existing results for {variant_name} on {ds_name}")
                with open(results_file) as f:
                    results = json.load(f)
                summary = summarize(results, variant_name, ds_label)
                print(
                    f"  r@20: baseline={summary.get('baseline_r@20', 0):.3f} "
                    f"arch={summary.get('arch_r@20', 0):.3f} "
                    f"delta={summary.get('delta_r@20', 0):+.3f} "
                    f"W/T/L={summary.get('W/T/L_r@20', '?')}"
                )
                cat_summaries = summarize_by_category(results)
                print("  Per-category (r@20):")
                for cat, cs in cat_summaries.items():
                    print(
                        f"    {cat}: baseline={cs['baseline_r@20']:.3f} "
                        f"arch={cs['arch_r@20']:.3f} "
                        f"delta={cs['delta_r@20']:+.3f} "
                        f"W/T/L={cs['W/T/L']} (n={cs['n']})"
                    )
                variant_results[variant_name] = results
                continue

            arch = SynthTestMetaV2Variant(store, prompt_template)
            results, summary = run_variant(
                variant_name, arch, questions, ds_label, verbose=args.verbose
            )
            variant_results[variant_name] = results

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Saved to {results_file}")

        # Head-to-head comparison
        if "v15_control" in variant_results and "v2f" in variant_results:
            comparison = compare_variants(
                variant_results["v15_control"],
                variant_results["v2f"],
                ds_label,
            )
            all_comparisons.append(comparison)

    # ===========================================================================
    # Grand summary
    # ===========================================================================
    print(f"\n\n{'=' * 100}")
    print("V2f vs V15 CONTROL — Grand Summary")
    print(f"{'=' * 100}")

    for comp in all_comparisons:
        print(f"\n--- {comp['dataset']} ({comp['n']} questions) ---")
        print(
            f"  Overall r@20: v15={comp['v15_r@20']:.3f}  v2f={comp['v2f_r@20']:.3f}  "
            f"delta={comp['delta_v2f_vs_v15']:+.4f}  W/T/L={comp['W/T/L']}"
        )
        print("  Per-category:")
        print(
            f"    {'Category':<28s} {'n':>3s} {'Base':>7s} {'V15':>7s} "
            f"{'V2f':>7s} {'V2f-V15':>8s} {'V15-B':>7s} {'V2f-B':>7s} {'W/T/L':>7s}"
        )
        print(f"    {'-' * 88}")
        for cat, cs in comp["per_category"].items():
            print(
                f"    {cat:<28s} {cs['n']:>3d} {cs['baseline_r@20']:>7.3f} "
                f"{cs['v15_r@20']:>7.3f} {cs['v2f_r@20']:>7.3f} "
                f"{cs['delta_v2f_vs_v15']:>+8.4f} "
                f"{cs['delta_v15_vs_base']:>+7.3f} "
                f"{cs['delta_v2f_vs_base']:>+7.3f} "
                f"{cs['W/T/L']:>7s}"
            )

    # Key questions analysis
    print(f"\n\n{'=' * 100}")
    print("KEY QUESTIONS ANALYSIS")
    print(f"{'=' * 100}")

    for comp in all_comparisons:
        cats = comp["per_category"]

        if "completeness" in cats:
            cs = cats["completeness"]
            print(
                f"\n1. V2f completeness instruction on 'completeness' category "
                f"({comp['dataset']}):"
            )
            print(
                f"   Baseline r@20={cs['baseline_r@20']:.3f}, "
                f"V15={cs['v15_r@20']:.3f} ({cs['delta_v15_vs_base']:+.3f}), "
                f"V2f={cs['v2f_r@20']:.3f} ({cs['delta_v2f_vs_base']:+.3f})"
            )
            print(f"   V2f vs V15: {cs['delta_v2f_vs_v15']:+.4f}, W/T/L={cs['W/T/L']}")

        for cat_name in ("proactive", "procedural"):
            if cat_name in cats:
                cs = cats[cat_name]
                print(
                    f"\n2. V2f anti-question instruction on '{cat_name}' category "
                    f"({comp['dataset']}):"
                )
                print(
                    f"   Baseline r@20={cs['baseline_r@20']:.3f}, "
                    f"V15={cs['v15_r@20']:.3f} ({cs['delta_v15_vs_base']:+.3f}), "
                    f"V2f={cs['v2f_r@20']:.3f} ({cs['delta_v2f_vs_base']:+.3f})"
                )
                print(
                    f"   V2f vs V15: {cs['delta_v2f_vs_v15']:+.4f}, W/T/L={cs['W/T/L']}"
                )

        for cat_name in ("sequential_chain", "logic_constraint"):
            if cat_name in cats:
                cs = cats[cat_name]
                print(f"\n4. Hard puzzle category '{cat_name}' ({comp['dataset']}):")
                print(
                    f"   Baseline r@20={cs['baseline_r@20']:.3f}, "
                    f"V15={cs['v15_r@20']:.3f} ({cs['delta_v15_vs_base']:+.3f}), "
                    f"V2f={cs['v2f_r@20']:.3f} ({cs['delta_v2f_vs_base']:+.3f})"
                )
                print(
                    f"   V2f vs V15: {cs['delta_v2f_vs_v15']:+.4f}, W/T/L={cs['W/T/L']}"
                )

    # Check for regressions (any category where V2f loses to V15)
    print("\n\n3. Regression check (categories where V2f loses to V15):")
    any_regression = False
    for comp in all_comparisons:
        for cat, cs in comp["per_category"].items():
            if cs["delta_v2f_vs_v15"] < -0.01:
                any_regression = True
                print(
                    f"   REGRESSION: {cat} ({comp['dataset']}): "
                    f"V2f-V15={cs['delta_v2f_vs_v15']:+.4f}, W/T/L={cs['W/T/L']}"
                )
    if not any_regression:
        print("   No regressions found (no category with V2f delta < -0.01)")

    # Save full comparison
    comp_file = RESULTS_DIR / "v2f_benchmark_comparison.json"
    with open(comp_file, "w") as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"\nSaved comparison to {comp_file}")


if __name__ == "__main__":
    main()
