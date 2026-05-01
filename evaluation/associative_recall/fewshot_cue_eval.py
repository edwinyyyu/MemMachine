"""Fair-backfill eval of few-shot v2f vs v2f baseline.

Architectures:
  meta_v2f              — v2f baseline (re-eval for apples-to-apples)
  fewshot_v2f_k2        — v2f + 2 exemplars
  fewshot_v2f_k3        — v2f + 3 exemplars
  fewshot_v2f_category_k2 — v2f + 2 category-matched exemplars (optional)

Runs all 4 datasets, K=20 and K=50.

Usage:
    uv run python fewshot_cue_eval.py
    uv run python fewshot_cue_eval.py --category    # add category variant
"""

import argparse
import json
import sys
import time
from pathlib import Path

from associative_recall import Segment
from best_shot import MetaV2f
from dotenv import load_dotenv
from fair_backfill_eval import (
    BUDGETS,
    DATASETS,
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from fewshot_cue_gen import (
    FewshotV2fCategoryK2,
    FewshotV2fK2,
    FewshotV2fK3,
    load_exemplar_bank,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def evaluate_question(
    arch,
    question: dict,
    set_category: bool = False,
) -> dict:
    """Run arch on a single question, produce fair-backfill metrics + cues.

    If set_category is True and arch supports .set_category(), pass the
    question's category for category-matched exemplars.
    """
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])
    category = question.get("category", "unknown")

    if set_category and hasattr(arch, "set_category"):
        arch.set_category(category)

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedupe arch segments preserving order
    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    # Cosine top-K
    query_emb = arch.embed_text(q_text)
    max_K = max(BUDGETS)
    cosine_result = arch.store.search(query_emb, top_k=max_K, conversation_id=conv_id)
    cosine_segments = list(cosine_result.segments)

    row = {
        "conversation_id": conv_id,
        "category": category,
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_arch_retrieved": len(arch_segments),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
        "cues": result.metadata.get("cues", []),
        "exemplars_used": result.metadata.get("exemplars_used", []),
    }

    for K in BUDGETS:
        b_rec, a_rec, _ = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)

    return row


def run_one(
    arch_name: str,
    arch,
    dataset: str,
    questions: list[dict],
    set_category: bool = False,
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 70}")
    print(f"{arch_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}")

    results = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q, set_category=set_category)
            results.append(row)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, arch_name, dataset)
    by_cat = summarize_by_category(results)

    print(f"\n--- {arch_name} on {dataset} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )

    return results, summary, by_cat


ARCH_CLASSES = {
    "meta_v2f": MetaV2f,
    "fewshot_v2f_k2": FewshotV2fK2,
    "fewshot_v2f_k3": FewshotV2fK3,
    "fewshot_v2f_category_k2": FewshotV2fCategoryK2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        action="store_true",
        help="Also run fewshot_v2f_category_k2 variant",
    )
    parser.add_argument(
        "--archs",
        type=str,
        default=None,
        help="Comma-separated archs to run (default: baseline + k2 + k3)",
    )
    args = parser.parse_args()

    if args.archs:
        arch_names = [a.strip() for a in args.archs.split(",")]
    else:
        arch_names = ["meta_v2f", "fewshot_v2f_k2", "fewshot_v2f_k3"]
        if args.category:
            arch_names.append("fewshot_v2f_category_k2")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load exemplar bank once (shared across arch instances)
    exemplars = load_exemplar_bank()
    print(f"Loaded {len(exemplars)} exemplars from bank.")

    all_summaries: dict = {}
    all_results: dict = {}

    for ds_name in DATASETS:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        for arch_name in arch_names:
            cls = ARCH_CLASSES[arch_name]
            if arch_name == "meta_v2f":
                arch = cls(store)
            else:
                arch = cls(store, exemplars=exemplars)

            set_category = arch_name == "fewshot_v2f_category_k2"

            results, summary, by_cat = run_one(
                arch_name,
                arch,
                ds_name,
                questions,
                set_category=set_category,
            )

            out_path = RESULTS_DIR / f"fewshot_{arch_name}_{ds_name}.json"
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

            all_summaries.setdefault(arch_name, {})[ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
            }
            all_results.setdefault(arch_name, {})[ds_name] = results

    # Consolidated summary
    summary_path = RESULTS_DIR / "fewshot_cue_study.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    # Final table
    print("\n" + "=" * 100)
    print("FEW-SHOT CUE STUDY SUMMARY")
    print("=" * 100)
    header = (
        f"{'Arch':<26s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'d@20':>7s} {'W/T/L@20':>10s} "
        f"{'base@50':>8s} {'arch@50':>8s} {'d@50':>7s} {'W/T/L@50':>10s}"
    )
    print(header)
    print("-" * len(header))
    for arch_name in arch_names:
        for ds_name in DATASETS:
            if ds_name not in all_summaries.get(arch_name, {}):
                continue
            s = all_summaries[arch_name][ds_name]["summary"]
            print(
                f"{arch_name:<26s} {ds_name:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['delta_r@20']:>+7.3f} {s['W/T/L_r@20']:>10s} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['delta_r@50']:>+7.3f} {s['W/T/L_r@50']:>10s}"
            )


if __name__ == "__main__":
    main()
