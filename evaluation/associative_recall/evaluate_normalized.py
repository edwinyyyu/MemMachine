"""Properly normalized evaluation of associative recall vs baseline.

The key insight: a fair comparison must use the SAME segment budget.
If associative retrieval uses 100 segments total, baseline should also
get 100 segments (top-100 by embedding similarity).

Reports recall at fixed budgets: 20, 50, 100, 150 segments.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from associative_recall import AssociativeRecallEngine, SegmentStore

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Fixed segment budgets for fair comparison
BUDGETS = [20, 50, 100, 150]


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    hits = retrieved_turn_ids & source_turn_ids
    return len(hits) / len(source_turn_ids)


def evaluate_question_normalized(
    engine: AssociativeRecallEngine,
    question: dict,
    verbose: bool = False,
    rerank: bool = False,
    fusion: bool = False,
    backfill: bool = False,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    # Get conversation size
    conv_length, max_turn_id = engine._get_conversation_metadata(conv_id)

    # --- Associative retrieval ---
    t0 = time.time()
    assoc_result = engine.associative_retrieve(q_text, conv_id, top_k_initial=10)
    assoc_time = time.time() - t0

    # Collect ALL associative segments in retrieval order (hop0 first, then hop1, etc.)
    # This gives us a ranked list where earlier hops = higher priority
    assoc_segments_ordered = []
    seen_indices = set()
    for hop in assoc_result.hops:
        for seg in hop.retrieved.segments:
            if seg.index not in seen_indices:
                assoc_segments_ordered.append(seg)
                seen_indices.add(seg.index)

    total_assoc_retrieved = len(assoc_segments_ordered)

    # --- Re-rank associative pool by cosine sim to original question ---
    if rerank and assoc_segments_ordered:
        query_embedding = engine.embed_text(q_text)
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
        # Score each segment in the pool by cosine similarity to original question
        rerank_scores = []
        for seg in assoc_segments_ordered:
            seg_emb = engine.store.normalized_embeddings[seg.index]
            score = float(np.dot(seg_emb, query_norm))
            rerank_scores.append(score)
        # Sort by score descending
        sorted_pairs = sorted(
            zip(rerank_scores, assoc_segments_ordered),
            key=lambda x: x[0],
            reverse=True,
        )
        assoc_segments_ordered = [seg for _, seg in sorted_pairs]

    # --- Baseline retrieval at various budgets ---
    # For fair comparison, get baseline top-k at each budget level
    max_budget = max(BUDGETS + [total_assoc_retrieved])
    t0 = time.time()
    baseline_result = engine.single_shot_retrieve(q_text, conv_id, top_k=max_budget)
    baseline_time = time.time() - t0

    # --- Fusion: merge assoc pool + baseline pool, rank by cosine ---
    if fusion and assoc_segments_ordered:
        query_embedding = engine.embed_text(q_text)
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
        # Merge both pools (union by index)
        fused_indices = set()
        fused_segments = []
        for seg in assoc_segments_ordered:
            if seg.index not in fused_indices:
                fused_segments.append(seg)
                fused_indices.add(seg.index)
        for seg in baseline_result.segments:
            if seg.index not in fused_indices:
                fused_segments.append(seg)
                fused_indices.add(seg.index)
        # Score all by cosine to original question
        fused_scores = []
        for seg in fused_segments:
            seg_emb = engine.store.normalized_embeddings[seg.index]
            score = float(np.dot(seg_emb, query_norm))
            fused_scores.append(score)
        # Sort by score descending
        sorted_pairs = sorted(
            zip(fused_scores, fused_segments),
            key=lambda x: x[0],
            reverse=True,
        )
        assoc_segments_ordered = [seg for _, seg in sorted_pairs]
        total_assoc_retrieved = len(assoc_segments_ordered)

    # --- Backfill: extend assoc pool with baseline results ---
    if backfill:
        assoc_indices = {seg.index for seg in assoc_segments_ordered}
        backfill_segments = [
            seg for seg in baseline_result.segments
            if seg.index not in assoc_indices
        ]
        assoc_segments_with_backfill = list(assoc_segments_ordered) + backfill_segments
    else:
        assoc_segments_with_backfill = assoc_segments_ordered

    # Compute recall at each fixed budget
    baseline_recalls = {}
    assoc_recalls = {}
    for budget in BUDGETS:
        # Baseline: top-`budget` segments by embedding similarity
        baseline_ids_at_budget = {s.turn_id for s in baseline_result.segments[:budget]}
        baseline_recalls[f"r@{budget}"] = compute_recall(baseline_ids_at_budget, source_ids)

        # Associative: first `budget` segments in retrieval order (with backfill if enabled)
        assoc_ids_at_budget = {s.turn_id for s in assoc_segments_with_backfill[:budget]}
        assoc_recalls[f"r@{budget}"] = compute_recall(assoc_ids_at_budget, source_ids)

    # Also compute at the actual retrieval size (how many assoc actually retrieved)
    baseline_ids_at_actual = {s.turn_id for s in baseline_result.segments[:total_assoc_retrieved]}
    assoc_ids_at_actual = {s.turn_id for s in assoc_segments_ordered}

    baseline_recalls["r@actual"] = compute_recall(baseline_ids_at_actual, source_ids)
    assoc_recalls["r@actual"] = compute_recall(assoc_ids_at_actual, source_ids)

    # Hop-by-hop details for analysis
    hop_details = []
    cumulative_ids = set()
    for hop in assoc_result.hops:
        hop_turn_ids = {s.turn_id for s in hop.retrieved.segments}
        new_ids = hop_turn_ids - cumulative_ids
        cumulative_ids |= hop_turn_ids
        hop_detail = {
            "hop": hop.hop_number,
            "cues": hop.cues,
            "num_retrieved": len(hop.retrieved.segments),
            "new_turn_ids_count": len(new_ids),
            "new_source_hits": sorted(new_ids & source_ids),
            "cumulative_recall": compute_recall(cumulative_ids, source_ids),
        }
        if hop.expand_targets is not None:
            hop_detail["expand_targets"] = sorted(hop.expand_targets)
            hop_detail["num_expanded"] = hop.num_expanded
        hop_details.append(hop_detail)

    result = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question["question_index"],
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "conv_length": conv_length,
        "baseline_recalls": baseline_recalls,
        "assoc_recalls": assoc_recalls,
        "total_assoc_retrieved": total_assoc_retrieved,
        "retrieval_fraction": round(total_assoc_retrieved / conv_length, 3),
        "num_hops": len(assoc_result.hops),
        "baseline_time_s": round(baseline_time, 2),
        "assoc_time_s": round(assoc_time, 2),
        "hop_details": hop_details,
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"Q: {q_text[:100]}...")
        print(f"Category: {question['category']}")
        print(f"Source IDs: {sorted(source_ids)} ({len(source_ids)} turns)")
        print(f"Conv size: {conv_length}, Assoc retrieved: {total_assoc_retrieved} "
              f"({total_assoc_retrieved/conv_length:.0%})")
        for budget in BUDGETS:
            b = baseline_recalls[f"r@{budget}"]
            a = assoc_recalls[f"r@{budget}"]
            delta = a - b
            print(f"  @{budget:3d}: baseline={b:.3f} assoc={a:.3f} delta={delta:+.3f}")
        b = baseline_recalls["r@actual"]
        a = assoc_recalls["r@actual"]
        print(f"  @{total_assoc_retrieved:3d}: baseline={b:.3f} assoc={a:.3f} "
              f"delta={a-b:+.3f} (actual)")
        for hop in hop_details:
            print(f"  Hop {hop['hop']}: +{hop['num_retrieved']} segs, "
                  f"+{hop['new_turn_ids_count']} new, "
                  f"source_hits={hop['new_source_hits']} "
                  f"cum_recall={hop['cumulative_recall']:.3f}")
            for cue in hop["cues"][:3]:
                print(f"    Cue: {cue[:100]}")

    return result


def print_summary(results: list[dict], label: str) -> dict:
    print(f"\n{'='*80}")
    print(f"NORMALIZED EVALUATION: {label}")
    print(f"{'='*80}")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    budget_labels = [f"r@{b}" for b in BUDGETS] + ["r@actual"]
    header = (
        f"{'category':30s} {'n':>3s}  "
        + "  ".join(f"{'B-'+lbl:>10s}" for lbl in budget_labels)
        + "  |  "
        + "  ".join(f"{'A-'+lbl:>10s}" for lbl in budget_labels)
    )
    print(header)
    print("-" * len(header))

    all_results = []
    for cat in sorted(by_category):
        rows = by_category[cat]
        all_results.extend(rows)
        b_means = {}
        a_means = {}
        for lbl in budget_labels:
            b_means[lbl] = sum(r["baseline_recalls"][lbl] for r in rows) / len(rows)
            a_means[lbl] = sum(r["assoc_recalls"][lbl] for r in rows) / len(rows)

        row_str = (
            f"{cat:30s} {len(rows):>3d}  "
            + "  ".join(f"{b_means[lbl]:>10.3f}" for lbl in budget_labels)
            + "  |  "
            + "  ".join(f"{a_means[lbl]:>10.3f}" for lbl in budget_labels)
        )
        print(row_str)

    if all_results:
        b_overall = {}
        a_overall = {}
        for lbl in budget_labels:
            b_overall[lbl] = sum(
                r["baseline_recalls"][lbl] for r in all_results
            ) / len(all_results)
            a_overall[lbl] = sum(
                r["assoc_recalls"][lbl] for r in all_results
            ) / len(all_results)
        avg_frac = sum(r["retrieval_fraction"] for r in all_results) / len(all_results)

        print("-" * len(header))
        row_str = (
            f"{'OVERALL':30s} {len(all_results):>3d}  "
            + "  ".join(f"{b_overall[lbl]:>10.3f}" for lbl in budget_labels)
            + "  |  "
            + "  ".join(f"{a_overall[lbl]:>10.3f}" for lbl in budget_labels)
        )
        print(row_str)

        print(f"\nDELTAS (assoc - baseline):")
        delta_str = (
            f"{'':30s} {'':>3s}  "
            + "  ".join(
                f"{a_overall[lbl] - b_overall[lbl]:>+10.3f}" for lbl in budget_labels
            )
        )
        print(delta_str)
        print(f"\nAvg retrieval fraction: {avg_frac:.2%}")
        print(f"Avg total assoc segments: "
              f"{sum(r['total_assoc_retrieved'] for r in all_results) / len(all_results):.0f}")

    summary = {
        "label": label,
        "num_questions": len(all_results),
        "overall": {
            f"baseline_{lbl}": b_overall[lbl] for lbl in budget_labels
        } | {
            f"assoc_{lbl}": a_overall[lbl] for lbl in budget_labels
        } | {
            f"delta_{lbl}": a_overall[lbl] - b_overall[lbl] for lbl in budget_labels
        },
        "avg_retrieval_fraction": avg_frac,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--prompt-version", default="v8",
                        choices=list(f"v{i}" for i in range(1, 30)))
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--num-cues", type=int, default=2)
    parser.add_argument("--top-k-per-hop", type=int, default=10)
    parser.add_argument("--neighbor-radius", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--label", default=None,
                        help="Label for this run")
    parser.add_argument("--data-suffix", default="",
                        help="Suffix for data files (e.g. '_extended')")
    parser.add_argument("--benchmark-filter", default=None,
                        help="Filter to specific benchmark (beam or locomo)")
    parser.add_argument("--max-conv-questions", type=int, default=None,
                        help="Max questions per conversation")
    parser.add_argument("--rerank", action="store_true",
                        help="Re-rank associative pool by cosine sim to original question")
    parser.add_argument("--fusion", action="store_true",
                        help="Fuse assoc pool + baseline pool, rank union by cosine sim")
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill assoc pool with baseline results beyond assoc pool size")
    args = parser.parse_args()

    questions_path = DATA_DIR / f"questions{args.data_suffix}.json"
    with open(questions_path) as f:
        questions = json.load(f)
    if args.benchmark_filter:
        questions = [q for q in questions if q.get("benchmark") == args.benchmark_filter]
    if args.max_questions:
        questions = questions[:args.max_questions]

    print(f"Loaded {len(questions)} questions")

    segments_path = DATA_DIR / f"segments{args.data_suffix}.npz"
    store = SegmentStore(data_dir=DATA_DIR, npz_name=f"segments{args.data_suffix}.npz")
    print(f"Loaded {len(store.segments)} segments")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    label = args.label or f"{args.model}_{args.prompt_version}_nr{args.neighbor_radius}"

    engine = AssociativeRecallEngine(
        store=store,
        cue_model=args.model,
        prompt_version=args.prompt_version,
        max_hops=args.max_hops,
        top_k_per_hop=args.top_k_per_hop,
        num_cues=args.num_cues,
        neighbor_radius=args.neighbor_radius,
    )

    results = []
    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {question['category']}: "
              f"{question['question'][:60]}...", flush=True)
        result = evaluate_question_normalized(
            engine, question, verbose=args.verbose,
            rerank=args.rerank, fusion=args.fusion,
            backfill=args.backfill,
        )
        results.append(result)
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            engine.save_caches()

    engine.save_caches()

    summary = print_summary(results, label)

    results_file = RESULTS_DIR / f"normalized_{label}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved detailed results to {results_file}")


if __name__ == "__main__":
    main()
