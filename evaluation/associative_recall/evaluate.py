"""Evaluate associative recall vs single-shot retrieval on BEAM probing questions.

Measures recall@k at various depths, comparing:
1. Single-shot baseline (embed question -> top-k)
2. Iterative associative recall (multi-hop with LLM cue generation)

Reports per-question results and summary statistics, with breakdowns by
category and model.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from associative_recall import AssociativeRecallEngine, SegmentStore

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

RECALL_DEPTHS = [5, 10, 20, 50]


def compute_recall(
    retrieved_turn_ids: set[int],
    source_turn_ids: set[int],
    max_k: int | None = None,
) -> float:
    if not source_turn_ids:
        return 1.0
    hits = retrieved_turn_ids & source_turn_ids
    return len(hits) / len(source_turn_ids)


def evaluate_question(
    engine: AssociativeRecallEngine,
    question: dict,
    verbose: bool = False,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    t0 = time.time()
    baseline_result = engine.single_shot_retrieve(q_text, conv_id, top_k=50)
    baseline_time = time.time() - t0

    baseline_turn_ids_by_depth: dict[int, set[int]] = {}
    for k in RECALL_DEPTHS:
        ids = {s.turn_id for s in baseline_result.segments[:k]}
        baseline_turn_ids_by_depth[k] = ids

    t0 = time.time()
    assoc_result = engine.associative_retrieve(q_text, conv_id, top_k_initial=10)
    assoc_time = time.time() - t0

    baseline_recalls = {}
    assoc_recalls = {}

    baseline_all_ids = {s.turn_id for s in baseline_result.segments}
    assoc_extra_ids = assoc_result.all_retrieved_turn_ids

    for k in RECALL_DEPTHS:
        baseline_recalls[f"r@{k}"] = compute_recall(
            baseline_turn_ids_by_depth[k], source_ids
        )
        combined_ids = baseline_turn_ids_by_depth[k] | assoc_extra_ids
        assoc_recalls[f"r@{k}"] = compute_recall(combined_ids, source_ids)

    baseline_recalls["r@all"] = compute_recall(baseline_all_ids, source_ids)
    assoc_recalls["r@all"] = compute_recall(
        baseline_all_ids | assoc_extra_ids, source_ids
    )

    hop0_ids = set()
    if assoc_result.hops:
        hop0_ids = {s.turn_id for s in assoc_result.hops[0].retrieved.segments}
    incremental_ids = assoc_result.all_retrieved_turn_ids - hop0_ids
    incremental_hits = incremental_ids & source_ids

    hop_details = []
    for hop in assoc_result.hops:
        hop_details.append({
            "hop": hop.hop_number,
            "cues": hop.cues,
            "num_retrieved": len(hop.retrieved.segments),
            "new_turn_ids_count": len(hop.new_turn_ids),
            "new_source_hits": list(hop.new_turn_ids & source_ids),
        })

    result = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question["question_index"],
        "question": q_text,
        "source_chat_ids": list(source_ids),
        "num_source_turns": len(source_ids),
        "baseline_recalls": baseline_recalls,
        "assoc_recalls": assoc_recalls,
        "incremental_source_hits": list(incremental_hits),
        "num_incremental_hits": len(incremental_hits),
        "num_hops": len(assoc_result.hops),
        "total_retrieved": len(assoc_result.all_retrieved_segments),
        "baseline_time_s": round(baseline_time, 2),
        "assoc_time_s": round(assoc_time, 2),
        "hop_details": hop_details,
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"Q: {q_text[:100]}...")
        print(f"Category: {question['category']}")
        print(f"Source IDs: {source_ids}")
        print(f"Baseline r@10={baseline_recalls['r@10']:.2f} "
              f"r@20={baseline_recalls['r@20']:.2f} "
              f"r@all={baseline_recalls['r@all']:.2f}")
        print(f"Assoc    r@10={assoc_recalls['r@10']:.2f} "
              f"r@20={assoc_recalls['r@20']:.2f} "
              f"r@all={assoc_recalls['r@all']:.2f}")
        print(f"Incremental hits: {incremental_hits}")
        for hop in hop_details:
            print(f"  Hop {hop['hop']}: {hop['num_retrieved']} retrieved, "
                  f"{hop['new_turn_ids_count']} new, "
                  f"source hits={hop['new_source_hits']}")
            for cue in hop["cues"]:
                print(f"    Cue: {cue[:100]}")

    return result


def print_summary(results: list[dict], label: str) -> dict:
    print(f"\n{'='*80}")
    print(f"SUMMARY: {label}")
    print(f"{'='*80}")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    depth_labels = [f"r@{k}" for k in RECALL_DEPTHS] + ["r@all"]
    header = (
        f"{'category':30s} {'n':>3s}  "
        + "  ".join(f"{'B-'+lbl:>7s}" for lbl in depth_labels)
        + "  |  "
        + "  ".join(f"{'A-'+lbl:>7s}" for lbl in depth_labels)
        + "  {'incr':>5s}"
    )
    print(header)
    print("-" * len(header))

    all_results = []
    for cat in sorted(by_category):
        rows = by_category[cat]
        all_results.extend(rows)
        b_means = {}
        a_means = {}
        for lbl in depth_labels:
            b_means[lbl] = sum(r["baseline_recalls"][lbl] for r in rows) / len(rows)
            a_means[lbl] = sum(r["assoc_recalls"][lbl] for r in rows) / len(rows)
        avg_incr = sum(r["num_incremental_hits"] for r in rows) / len(rows)

        row_str = (
            f"{cat:30s} {len(rows):>3d}  "
            + "  ".join(f"{b_means[lbl]:>7.3f}" for lbl in depth_labels)
            + "  |  "
            + "  ".join(f"{a_means[lbl]:>7.3f}" for lbl in depth_labels)
            + f"  {avg_incr:>5.1f}"
        )
        print(row_str)

    if all_results:
        b_overall = {}
        a_overall = {}
        for lbl in depth_labels:
            b_overall[lbl] = sum(
                r["baseline_recalls"][lbl] for r in all_results
            ) / len(all_results)
            a_overall[lbl] = sum(
                r["assoc_recalls"][lbl] for r in all_results
            ) / len(all_results)
        avg_incr = sum(
            r["num_incremental_hits"] for r in all_results
        ) / len(all_results)

        print("-" * len(header))
        row_str = (
            f"{'OVERALL':30s} {len(all_results):>3d}  "
            + "  ".join(f"{b_overall[lbl]:>7.3f}" for lbl in depth_labels)
            + "  |  "
            + "  ".join(f"{a_overall[lbl]:>7.3f}" for lbl in depth_labels)
            + f"  {avg_incr:>5.1f}"
        )
        print(row_str)

    summary = {
        "label": label,
        "num_questions": len(all_results),
        "by_category": {
            cat: {
                "count": len(rows),
                "baseline": {
                    lbl: sum(r["baseline_recalls"][lbl] for r in rows) / len(rows)
                    for lbl in depth_labels
                },
                "associative": {
                    lbl: sum(r["assoc_recalls"][lbl] for r in rows) / len(rows)
                    for lbl in depth_labels
                },
                "avg_incremental_hits": sum(
                    r["num_incremental_hits"] for r in rows
                ) / len(rows),
            }
            for cat, rows in by_category.items()
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini",
                        help="LLM model for cue generation")
    parser.add_argument("--prompt-version", default="v1",
                        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"])
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--num-cues", type=int, default=2)
    parser.add_argument("--top-k-per-hop", type=int, default=10)
    parser.add_argument("--neighbor-radius", type=int, default=0,
                        help="Include N neighboring turns around each found segment")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare-models", action="store_true",
                        help="Run both gpt-5-mini and gpt-5-nano")
    parser.add_argument("--compare-prompts", action="store_true",
                        help="Run all prompt versions")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Filter to specific categories")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit total number of questions")
    args = parser.parse_args()

    questions_path = DATA_DIR / "questions.json"
    with open(questions_path) as f:
        questions = json.load(f)

    if args.categories:
        cat_set = set(args.categories)
        questions = [q for q in questions if q["category"] in cat_set]
    if args.max_questions:
        questions = questions[:args.max_questions]

    print(f"Loaded {len(questions)} questions")

    store = SegmentStore()
    print(f"Loaded {len(store.segments)} segments")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = []
    if args.compare_models:
        for model in ["gpt-5-mini", "gpt-5-nano"]:
            configs.append({
                "model": model,
                "prompt_version": args.prompt_version,
                "label": f"{model}/{args.prompt_version}",
            })
    elif args.compare_prompts:
        for pv in ["v1", "v2", "v3", "v4"]:
            configs.append({
                "model": args.model,
                "prompt_version": pv,
                "label": f"{args.model}/{pv}",
            })
    else:
        configs.append({
            "model": args.model,
            "prompt_version": args.prompt_version,
            "label": f"{args.model}/{args.prompt_version}",
        })

    all_summaries = []
    for config in configs:
        print(f"\n{'#'*80}")
        print(f"Running: {config['label']}")
        print(f"{'#'*80}")

        engine = AssociativeRecallEngine(
            store=store,
            cue_model=config["model"],
            prompt_version=config["prompt_version"],
            max_hops=args.max_hops,
            top_k_per_hop=args.top_k_per_hop,
            num_cues=args.num_cues,
            neighbor_radius=args.neighbor_radius,
        )

        results = []
        for i, question in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {question['category']}: "
                  f"{question['question'][:60]}...", flush=True)
            result = evaluate_question(engine, question, verbose=args.verbose)
            results.append(result)
            sys.stdout.flush()
            # Save caches periodically to avoid losing progress
            if (i + 1) % 5 == 0:
                engine.save_caches()

        engine.save_caches()

        summary = print_summary(results, config["label"])
        all_summaries.append(summary)

        results_file = (
            RESULTS_DIR
            / f"results_{config['model']}_{config['prompt_version']}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved detailed results to {results_file}")

    if len(all_summaries) > 1:
        print(f"\n\n{'='*80}")
        print("COMPARISON ACROSS CONFIGURATIONS")
        print(f"{'='*80}")
        depth_labels = [f"r@{k}" for k in RECALL_DEPTHS] + ["r@all"]
        for s in all_summaries:
            if "OVERALL" not in str(s):
                all_cats = s["by_category"]
                n = s["num_questions"]
                overall_b = {
                    lbl: sum(
                        cat["baseline"][lbl] * cat["count"]
                        for cat in all_cats.values()
                    ) / n
                    for lbl in depth_labels
                }
                overall_a = {
                    lbl: sum(
                        cat["associative"][lbl] * cat["count"]
                        for cat in all_cats.values()
                    ) / n
                    for lbl in depth_labels
                }
                print(f"\n{s['label']} (n={n}):")
                print(f"  Baseline:    " + "  ".join(
                    f"{lbl}={overall_b[lbl]:.3f}" for lbl in depth_labels
                ))
                print(f"  Associative: " + "  ".join(
                    f"{lbl}={overall_a[lbl]:.3f}" for lbl in depth_labels
                ))
                delta = {
                    lbl: overall_a[lbl] - overall_b[lbl] for lbl in depth_labels
                }
                print(f"  Delta:       " + "  ".join(
                    f"{lbl}={delta[lbl]:+.3f}" for lbl in depth_labels
                ))

    summaries_file = RESULTS_DIR / "summaries.json"
    with open(summaries_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summaries to {summaries_file}")


if __name__ == "__main__":
    main()
