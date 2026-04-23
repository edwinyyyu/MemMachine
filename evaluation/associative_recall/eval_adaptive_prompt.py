"""Test domain-agnostic V2f prompt variants that infer content register from examples.

V2f_adaptive: removes conversation-specific language, adds "vocabulary and style
that matches the retrieved content above" -- the model infers register from examples.

V2f_minimal: same removals but WITHOUT the register-matching instruction -- control
to test whether just removing domain language is enough.

V2f_original: reference V2f with conversation-specific language.

Tests on three datasets:
  1. LoCoMo 30q (conversations -- where V2f was +37.2pp)
  2. Synthetic 19q (mixed domains -- where V2f regressed)
  3. Advanced 23q (harder questions -- where V2f was slightly better)

Usage:
    uv run python eval_adaptive_prompt.py [--force] [--verbose]
"""

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from prompt_optimization import (
    META_V2F_PROMPT,
    BUDGETS,
    MetaV2Variant,
    OptimResult,
    _format_segments,
    _parse_cues,
    compute_recall,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# V2f_adaptive: domain-agnostic + register-matching from examples
V2F_ADAPTIVE_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Each cue should use \
vocabulary and style that matches the retrieved content above.

Do NOT write questions ("Did you mention X?") or search commands ("Search \
for..."). Write text that looks like it could be an excerpt from the content \
being searched.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# V2f_minimal: domain-agnostic but NO register-matching instruction -- control
V2F_MINIMAL_PROMPT = """\
You are generating search text for semantic retrieval. Your cues will be \
embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment.

Do NOT write questions ("Did you mention X?") or search commands ("Search \
for..."). Write text that looks like it could be an excerpt from the content \
being searched.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# ---------------------------------------------------------------------------
# Cache classes -- writes to adaptive_llm_cache.json / adaptive_embedding_cache.json
# ---------------------------------------------------------------------------
class AdaptiveEmbeddingCache(EmbeddingCache):
    """Reads all existing caches, writes to adaptive-specific file."""

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
            "general_embedding_cache.json",
            "adaptive_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "adaptive_embedding_cache.json"
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


class AdaptiveLLMCache(LLMCache):
    """Reads all existing caches, writes to adaptive_llm_cache.json."""

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
            "general_llm_cache.json",
            "adaptive_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "adaptive_llm_cache.json"
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


# ---------------------------------------------------------------------------
# Variant class with adaptive caches
# ---------------------------------------------------------------------------
class AdaptiveMetaV2Variant(MetaV2Variant):
    """MetaV2Variant that uses adaptive caches."""

    def __init__(self, store: SegmentStore, prompt_template: str,
                 client: OpenAI | None = None):
        super().__init__(store, prompt_template, client)
        self.embedding_cache = AdaptiveEmbeddingCache()
        self.llm_cache = AdaptiveLLMCache()


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------
def evaluate_one(
    arch: MetaV2Variant,
    question: dict,
    verbose: bool = False,
) -> dict:
    """Evaluate a single variant on a single question."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Deduplicate preserving order
    seen: set[int] = set()
    deduped: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            deduped.append(seg)
            seen.add(seg.index)
    arch_segments = deduped
    total_retrieved = len(arch_segments)

    # Baseline: cosine top-N at same budget
    query_emb = arch.embed_text(q_text)
    max_budget = max(BUDGETS + [total_retrieved])
    baseline_result = arch.store.search(
        query_emb, top_k=max_budget, conversation_id=conv_id
    )

    baseline_recalls: dict[str, float] = {}
    arch_recalls: dict[str, float] = {}
    for budget in BUDGETS:
        baseline_ids = {s.turn_id for s in baseline_result.segments[:budget]}
        baseline_recalls[f"r@{budget}"] = compute_recall(baseline_ids, source_ids)

        arch_ids = {s.turn_id for s in arch_segments[:budget]}
        arch_recalls[f"r@{budget}"] = compute_recall(arch_ids, source_ids)

    # Also at actual retrieval size
    baseline_ids_actual = {
        s.turn_id for s in baseline_result.segments[:total_retrieved]
    }
    arch_ids_actual = {s.turn_id for s in arch_segments}
    baseline_recalls["r@actual"] = compute_recall(baseline_ids_actual, source_ids)
    arch_recalls["r@actual"] = compute_recall(arch_ids_actual, source_ids)

    row = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question["question_index"],
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "baseline_recalls": baseline_recalls,
        "arch_recalls": arch_recalls,
        "total_retrieved": total_retrieved,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "metadata": result.metadata,
    }

    if verbose:
        print(f"  Source: {sorted(source_ids)} ({len(source_ids)} turns)")
        print(
            f"  Retrieved: {total_retrieved}, Embed: {arch.embed_calls}, "
            f"LLM: {arch.llm_calls}, Time: {elapsed:.1f}s"
        )
        for budget in BUDGETS:
            b = baseline_recalls[f"r@{budget}"]
            a = arch_recalls[f"r@{budget}"]
            delta = a - b
            marker = "W" if delta > 0.001 else ("L" if delta < -0.001 else "T")
            print(
                f"  @{budget:3d}: baseline={b:.3f} arch={a:.3f} "
                f"delta={delta:+.3f} [{marker}]"
            )
        cues = result.metadata.get("cues", [])
        for cue in cues[:4]:
            print(f"    Cue: {cue[:120]}")

    return row


def summarize(results: list[dict], variant_name: str, benchmark: str) -> dict:
    """Compute summary statistics."""
    n = len(results)
    if n == 0:
        return {}

    summary: dict = {"variant": variant_name, "benchmark": benchmark, "n": n}

    for label in [f"r@{b}" for b in BUDGETS] + ["r@actual"]:
        b_vals = [r["baseline_recalls"][label] for r in results]
        a_vals = [r["arch_recalls"][label] for r in results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n

        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses

        summary[f"baseline_{label}"] = round(b_mean, 4)
        summary[f"arch_{label}"] = round(a_mean, 4)
        summary[f"delta_{label}"] = round(a_mean - b_mean, 4)
        summary[f"W/T/L_{label}"] = f"{wins}/{ties}/{losses}"

    summary["avg_total_retrieved"] = round(
        sum(r["total_retrieved"] for r in results) / n, 1
    )
    summary["avg_embed_calls"] = round(
        sum(r["embed_calls"] for r in results) / n, 1
    )
    summary["avg_llm_calls"] = round(
        sum(r["llm_calls"] for r in results) / n, 1
    )
    summary["avg_time_s"] = round(sum(r["time_s"] for r in results) / n, 2)

    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    """Per-category breakdown at r@20."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    cat_summaries = {}
    for cat, cat_results in sorted(by_cat.items()):
        n = len(cat_results)
        b_vals = [r["baseline_recalls"]["r@20"] for r in cat_results]
        a_vals = [r["arch_recalls"]["r@20"] for r in cat_results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        cat_summaries[cat] = {
            "n": n,
            "baseline_r@20": round(b_mean, 4),
            "arch_r@20": round(a_mean, 4),
            "delta_r@20": round(a_mean - b_mean, 4),
            "W/T/L": f"{wins}/{n - wins - losses}/{losses}",
        }
    return cat_summaries


def run_variant(
    variant_name: str,
    arch: MetaV2Variant,
    questions: list[dict],
    benchmark_label: str,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Run one variant, return (results, summary)."""
    print(f"\n{'='*70}")
    print(
        f"VARIANT: {variant_name} | BENCHMARK: {benchmark_label} | "
        f"{len(questions)} questions"
    )
    print(f"{'='*70}")

    results = []
    for i, question in enumerate(questions):
        q_short = question["question"][:55]
        print(
            f"  [{i+1}/{len(questions)}] {question['category']}: "
            f"{q_short}...",
            flush=True,
        )
        try:
            result = evaluate_one(arch, question, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, variant_name, benchmark_label)

    # Print compact summary
    print(f"\n--- {variant_name} on {benchmark_label} ---")
    for budget in BUDGETS:
        lbl = f"r@{budget}"
        print(
            f"  {lbl}: baseline={summary.get(f'baseline_{lbl}', 0):.3f} "
            f"arch={summary.get(f'arch_{lbl}', 0):.3f} "
            f"delta={summary.get(f'delta_{lbl}', 0):+.3f} "
            f"W/T/L={summary.get(f'W/T/L_{lbl}', '?')}"
        )
    print(
        f"  Avg retrieved: {summary.get('avg_total_retrieved', 0):.0f}, "
        f"Embed: {summary.get('avg_embed_calls', 0):.1f}, "
        f"LLM: {summary.get('avg_llm_calls', 0):.1f}, "
        f"Time: {summary.get('avg_time_s', 0):.1f}s"
    )

    cat_summaries = summarize_by_category(results)
    print(f"\n  Per-category (r@20):")
    for cat, cs in cat_summaries.items():
        print(
            f"    {cat}: baseline={cs['baseline_r@20']:.3f} "
            f"arch={cs['arch_r@20']:.3f} "
            f"delta={cs['delta_r@20']:+.3f} "
            f"W/T/L={cs['W/T/L']} (n={cs['n']})"
        )

    return results, summary


# ---------------------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------------------
def compare_variants(
    ref_results: list[dict],
    test_results: list[dict],
    ref_name: str,
    test_name: str,
    dataset_label: str,
) -> dict:
    """Head-to-head comparison at r@20."""
    n = len(ref_results)
    assert n == len(test_results), "Results must have same length"

    ref_vals = [r["arch_recalls"]["r@20"] for r in ref_results]
    test_vals = [r["arch_recalls"]["r@20"] for r in test_results]

    wins = sum(1 for a, b in zip(test_vals, ref_vals) if a > b + 0.001)
    losses = sum(1 for a, b in zip(test_vals, ref_vals) if b > a + 0.001)
    ties = n - wins - losses

    ref_mean = sum(ref_vals) / n
    test_mean = sum(test_vals) / n

    comparison = {
        "dataset": dataset_label,
        "n": n,
        "ref_name": ref_name,
        "test_name": test_name,
        f"{ref_name}_r@20": round(ref_mean, 4),
        f"{test_name}_r@20": round(test_mean, 4),
        "delta_r@20": round(test_mean - ref_mean, 4),
        "W/T/L": f"{wins}/{ties}/{losses}",
    }

    # Per-category
    by_cat: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for ref_r, test_r in zip(ref_results, test_results):
        by_cat[ref_r["category"]].append((ref_r, test_r))

    cat_comparisons = {}
    for cat, pairs in sorted(by_cat.items()):
        cn = len(pairs)
        ref_cat = [p[0]["arch_recalls"]["r@20"] for p in pairs]
        test_cat = [p[1]["arch_recalls"]["r@20"] for p in pairs]
        base_cat = [p[0]["baseline_recalls"]["r@20"] for p in pairs]
        cw = sum(1 for a, b in zip(test_cat, ref_cat) if a > b + 0.001)
        cl = sum(1 for a, b in zip(test_cat, ref_cat) if b > a + 0.001)
        cat_comparisons[cat] = {
            "n": cn,
            "baseline_r@20": round(sum(base_cat) / cn, 4),
            f"{ref_name}_r@20": round(sum(ref_cat) / cn, 4),
            f"{test_name}_r@20": round(sum(test_cat) / cn, 4),
            "delta": round(sum(test_cat) / cn - sum(ref_cat) / cn, 4),
            "W/T/L": f"{cw}/{cn - cw - cl}/{cl}",
        }

    comparison["per_category"] = cat_comparisons
    return comparison


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "locomo",
        "npz": "segments_extended.npz",
        "questions_file": "questions_extended.json",
        "filter_benchmark": "locomo",
        "max_questions": 30,
        "label": "LoCoMo (30q)",
    },
    {
        "name": "synth",
        "npz": "segments_synthetic.npz",
        "questions_file": "questions_synthetic.json",
        "filter_benchmark": None,
        "max_questions": None,
        "label": "Synthetic (19q)",
    },
    {
        "name": "advanced",
        "npz": "segments_advanced.npz",
        "questions_file": "questions_advanced.json",
        "filter_benchmark": None,
        "max_questions": None,
        "label": "Advanced (23q)",
    },
]

VARIANTS = [
    ("v2f_original", META_V2F_PROMPT),
    ("v2f_adaptive", V2F_ADAPTIVE_PROMPT),
    ("v2f_minimal", V2F_MINIMAL_PROMPT),
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test adaptive V2f prompt variants across datasets"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_comparisons = []
    all_summaries = []

    for dataset in DATASETS:
        ds_name = dataset["name"]
        ds_label = dataset["label"]

        print(f"\n{'#'*80}")
        print(f"# DATASET: {ds_label}")
        print(f"{'#'*80}")

        # Load data
        store = SegmentStore(data_dir=DATA_DIR, npz_name=dataset["npz"])
        with open(DATA_DIR / dataset["questions_file"]) as f:
            questions = json.load(f)

        # Filter by benchmark if needed
        if dataset["filter_benchmark"]:
            questions = [
                q for q in questions
                if q.get("benchmark") == dataset["filter_benchmark"]
            ]

        # Limit questions if needed
        if dataset["max_questions"]:
            questions = questions[:dataset["max_questions"]]

        # Ensure all questions have question_index
        for i, q in enumerate(questions):
            if "question_index" not in q:
                q["question_index"] = i

        print(f"Loaded {len(store.segments)} segments, {len(questions)} questions")

        variant_results = {}

        for variant_name, prompt_template in VARIANTS:
            results_file = RESULTS_DIR / f"adaptive_{ds_name}_{variant_name}.json"

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
                print(f"  Per-category (r@20):")
                for cat, cs in cat_summaries.items():
                    print(
                        f"    {cat}: baseline={cs['baseline_r@20']:.3f} "
                        f"arch={cs['arch_r@20']:.3f} "
                        f"delta={cs['delta_r@20']:+.3f} "
                        f"W/T/L={cs['W/T/L']} (n={cs['n']})"
                    )
                variant_results[variant_name] = results
                all_summaries.append(summary)
                continue

            arch = AdaptiveMetaV2Variant(store, prompt_template)
            results, summary = run_variant(
                variant_name, arch, questions, ds_label, verbose=args.verbose
            )
            variant_results[variant_name] = results
            all_summaries.append(summary)

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Saved to {results_file}")

        # Head-to-head: adaptive vs original
        if "v2f_original" in variant_results and "v2f_adaptive" in variant_results:
            comp = compare_variants(
                variant_results["v2f_original"],
                variant_results["v2f_adaptive"],
                "v2f_original",
                "v2f_adaptive",
                ds_label,
            )
            all_comparisons.append(("adaptive_vs_original", comp))

        # Head-to-head: minimal vs original
        if "v2f_original" in variant_results and "v2f_minimal" in variant_results:
            comp = compare_variants(
                variant_results["v2f_original"],
                variant_results["v2f_minimal"],
                "v2f_original",
                "v2f_minimal",
                ds_label,
            )
            all_comparisons.append(("minimal_vs_original", comp))

    # ===========================================================================
    # Grand summary
    # ===========================================================================
    print(f"\n\n{'='*100}")
    print("ADAPTIVE V2f — Grand Summary")
    print(f"{'='*100}")

    # Summary table: all variants x all datasets
    print(f"\n{'Variant':<20s} {'Dataset':<20s} {'B-r@20':>8s} {'A-r@20':>8s} "
          f"{'Delta':>8s} {'W/T/L':>10s}")
    print("-" * 76)
    for s in all_summaries:
        if not s:
            continue
        print(
            f"{s['variant']:<20s} "
            f"{s['benchmark']:<20s} "
            f"{s.get('baseline_r@20', 0):>8.3f} "
            f"{s.get('arch_r@20', 0):>8.3f} "
            f"{s.get('delta_r@20', 0):>+8.3f} "
            f"{s.get('W/T/L_r@20', '?'):>10s}"
        )

    # Head-to-head comparisons
    print(f"\n\n{'='*100}")
    print("HEAD-TO-HEAD COMPARISONS (r@20)")
    print(f"{'='*100}")

    for comp_label, comp in all_comparisons:
        ref_name = comp["ref_name"]
        test_name = comp["test_name"]
        print(f"\n--- {test_name} vs {ref_name} on {comp['dataset']} "
              f"({comp['n']} questions) ---")
        print(
            f"  Overall: {ref_name}={comp[f'{ref_name}_r@20']:.3f}  "
            f"{test_name}={comp[f'{test_name}_r@20']:.3f}  "
            f"delta={comp['delta_r@20']:+.4f}  W/T/L={comp['W/T/L']}"
        )
        print(f"  Per-category:")
        print(
            f"    {'Category':<28s} {'n':>3s} {'Base':>7s} "
            f"{'Ref':>7s} {'Test':>7s} {'Delta':>8s} {'W/T/L':>7s}"
        )
        print(f"    {'-'*70}")
        for cat, cs in comp["per_category"].items():
            print(
                f"    {cat:<28s} {cs['n']:>3d} {cs['baseline_r@20']:>7.3f} "
                f"{cs[f'{ref_name}_r@20']:>7.3f} "
                f"{cs[f'{test_name}_r@20']:>7.3f} "
                f"{cs['delta']:>+8.4f} "
                f"{cs['W/T/L']:>7s}"
            )

    # Key question: does adaptive match original on LoCoMo?
    print(f"\n\n{'='*100}")
    print("KEY QUESTIONS")
    print(f"{'='*100}")

    locomo_adaptive = None
    synth_adaptive = None
    advanced_adaptive = None
    locomo_minimal = None
    synth_minimal = None
    advanced_minimal = None

    for comp_label, comp in all_comparisons:
        if comp_label == "adaptive_vs_original":
            if "LoCoMo" in comp["dataset"]:
                locomo_adaptive = comp
            elif "Synthetic" in comp["dataset"]:
                synth_adaptive = comp
            elif "Advanced" in comp["dataset"]:
                advanced_adaptive = comp
        elif comp_label == "minimal_vs_original":
            if "LoCoMo" in comp["dataset"]:
                locomo_minimal = comp
            elif "Synthetic" in comp["dataset"]:
                synth_minimal = comp
            elif "Advanced" in comp["dataset"]:
                advanced_minimal = comp

    print("\n1. Does V2f_adaptive match V2f on LoCoMo? (register inference works)")
    if locomo_adaptive:
        delta = locomo_adaptive["delta_r@20"]
        verdict = "YES" if abs(delta) < 0.02 else ("BETTER" if delta > 0.02 else "NO - regression")
        print(f"   Delta: {delta:+.4f}, W/T/L: {locomo_adaptive['W/T/L']} => {verdict}")

    print("\n2. Does V2f_adaptive avoid regression on Synthetic?")
    if synth_adaptive:
        delta = synth_adaptive["delta_r@20"]
        verdict = "YES - improved" if delta > 0.01 else ("YES - matched" if delta > -0.01 else "NO - still regresses")
        print(f"   Delta: {delta:+.4f}, W/T/L: {synth_adaptive['W/T/L']} => {verdict}")

    print("\n3. Does V2f_adaptive avoid regression on Advanced?")
    if advanced_adaptive:
        delta = advanced_adaptive["delta_r@20"]
        verdict = "YES - improved" if delta > 0.01 else ("YES - matched" if delta > -0.01 else "NO - still regresses")
        print(f"   Delta: {delta:+.4f}, W/T/L: {advanced_adaptive['W/T/L']} => {verdict}")

    print("\n4. Is register-matching instruction load-bearing? (adaptive vs minimal)")
    if locomo_adaptive and locomo_minimal:
        adaptive_delta = locomo_adaptive["delta_r@20"]
        minimal_delta = locomo_minimal["delta_r@20"]
        diff = adaptive_delta - minimal_delta
        print(f"   LoCoMo: adaptive delta={adaptive_delta:+.4f}, "
              f"minimal delta={minimal_delta:+.4f}, "
              f"difference={diff:+.4f}")
    if synth_adaptive and synth_minimal:
        adaptive_delta = synth_adaptive["delta_r@20"]
        minimal_delta = synth_minimal["delta_r@20"]
        diff = adaptive_delta - minimal_delta
        print(f"   Synthetic: adaptive delta={adaptive_delta:+.4f}, "
              f"minimal delta={minimal_delta:+.4f}, "
              f"difference={diff:+.4f}")
    if advanced_adaptive and advanced_minimal:
        adaptive_delta = advanced_adaptive["delta_r@20"]
        minimal_delta = advanced_minimal["delta_r@20"]
        diff = adaptive_delta - minimal_delta
        print(f"   Advanced: adaptive delta={adaptive_delta:+.4f}, "
              f"minimal delta={minimal_delta:+.4f}, "
              f"difference={diff:+.4f}")

    # Save all comparisons
    comp_file = RESULTS_DIR / "adaptive_comparison.json"
    with open(comp_file, "w") as f:
        json.dump(
            [{"label": label, "data": data} for label, data in all_comparisons],
            f, indent=2,
        )
    print(f"\nSaved comparison to {comp_file}")


if __name__ == "__main__":
    main()
