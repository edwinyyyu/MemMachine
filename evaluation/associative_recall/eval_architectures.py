"""Evaluate alternative retrieval architectures with proper normalization.

For each architecture:
1. Retrieve segments
2. Count total retrieved
3. Compare recall at r@20, r@50, r@100 against baseline with SAME budget
4. Report Win/Tie/Loss at r@20
"""

import json
import sys
import time
from pathlib import Path

from architectures import (
    AgentWorkingSet,
    BaseArchitecture,
    CentroidWalk,
    ClusterDiversify,
    HybridGapFill,
    MMRDiversified,
    MultiQueryFusion,
    NegativeSpace,
    RetrieveSummarizeRetrieve,
    SegmentAsQuery,
)
from associative_recall import SegmentStore

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50, 100]


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    hits = retrieved_turn_ids & source_turn_ids
    return len(hits) / len(source_turn_ids)


def evaluate_architecture(
    arch: BaseArchitecture,
    question: dict,
    verbose: bool = False,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    arch_result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Get architecture's segments in order
    arch_segments = arch_result.segments
    seen_indices = set()
    deduped: list = []
    for seg in arch_segments:
        if seg.index not in seen_indices:
            deduped.append(seg)
            seen_indices.add(seg.index)
    arch_segments = deduped
    total_retrieved = len(arch_segments)

    # Baseline: top-N by cosine (using same embedding call)
    query_emb = arch.embed_text(q_text)
    max_budget = max(BUDGETS + [total_retrieved])
    baseline_result = arch.store.search(
        query_emb, top_k=max_budget, conversation_id=conv_id
    )

    # Compute recalls at fixed budgets
    baseline_recalls = {}
    arch_recalls = {}
    for budget in BUDGETS:
        baseline_ids = {s.turn_id for s in baseline_result.segments[:budget]}
        baseline_recalls[f"r@{budget}"] = compute_recall(baseline_ids, source_ids)

        arch_ids = {s.turn_id for s in arch_segments[:budget]}
        arch_recalls[f"r@{budget}"] = compute_recall(arch_ids, source_ids)

    # At actual retrieval size (normalized comparison)
    baseline_ids_actual = {
        s.turn_id for s in baseline_result.segments[:total_retrieved]
    }
    arch_ids_actual = {s.turn_id for s in arch_segments}
    baseline_recalls["r@actual"] = compute_recall(baseline_ids_actual, source_ids)
    arch_recalls["r@actual"] = compute_recall(arch_ids_actual, source_ids)

    result = {
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
        "metadata": arch_result.metadata,
    }

    if verbose:
        print(f"  Source IDs: {sorted(source_ids)} ({len(source_ids)} turns)")
        print(
            f"  Retrieved: {total_retrieved}, Embed: {arch.embed_calls}, LLM: {arch.llm_calls}"
        )
        for budget in BUDGETS:
            b = baseline_recalls[f"r@{budget}"]
            a = arch_recalls[f"r@{budget}"]
            print(f"  @{budget:3d}: baseline={b:.3f} arch={a:.3f} delta={a - b:+.3f}")

    return result


def print_summary(results: list[dict], arch_name: str, benchmark: str) -> dict:
    n = len(results)
    if n == 0:
        return {}

    summary = {"arch": arch_name, "benchmark": benchmark, "n": n}

    for budget_label in [f"r@{b}" for b in BUDGETS] + ["r@actual"]:
        b_mean = sum(r["baseline_recalls"][budget_label] for r in results) / n
        a_mean = sum(r["arch_recalls"][budget_label] for r in results) / n
        delta = a_mean - b_mean

        wins = sum(
            1
            for r in results
            if r["arch_recalls"][budget_label]
            > r["baseline_recalls"][budget_label] + 0.001
        )
        losses = sum(
            1
            for r in results
            if r["baseline_recalls"][budget_label]
            > r["arch_recalls"][budget_label] + 0.001
        )
        ties = n - wins - losses

        summary[f"baseline_{budget_label}"] = round(b_mean, 4)
        summary[f"arch_{budget_label}"] = round(a_mean, 4)
        summary[f"delta_{budget_label}"] = round(delta, 4)
        summary[f"W/T/L_{budget_label}"] = f"{wins}/{ties}/{losses}"

    summary["avg_total_retrieved"] = round(
        sum(r["total_retrieved"] for r in results) / n, 1
    )
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    summary["avg_time_s"] = round(sum(r["time_s"] for r in results) / n, 2)

    return summary


def run_evaluation(
    arch_name: str,
    arch: BaseArchitecture,
    questions: list[dict],
    benchmark_label: str,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    print(f"\n{'=' * 70}")
    print(
        f"ARCH: {arch_name} | BENCHMARK: {benchmark_label} | {len(questions)} questions"
    )
    print(f"{'=' * 70}")

    results = []
    for i, question in enumerate(questions):
        print(
            f"  [{i + 1}/{len(questions)}] {question['category']}: "
            f"{question['question'][:50]}...",
            flush=True,
        )
        try:
            result = evaluate_architecture(arch, question, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
        sys.stdout.flush()
        if (i + 1) % 10 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = print_summary(results, arch_name, benchmark_label)

    # Print compact summary
    print(f"\n--- {arch_name} on {benchmark_label} ---")
    for budget in BUDGETS:
        lbl = f"r@{budget}"
        print(
            f"  {lbl}: baseline={summary[f'baseline_{lbl}']:.3f} "
            f"arch={summary[f'arch_{lbl}']:.3f} "
            f"delta={summary[f'delta_{lbl}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_{lbl}']}"
        )
    print(
        f"  Avg retrieved: {summary['avg_total_retrieved']:.0f}, "
        f"Embed calls: {summary['avg_embed_calls']:.1f}, "
        f"LLM calls: {summary['avg_llm_calls']:.1f}"
    )

    return results, summary


def main() -> None:
    # Load data
    with open(DATA_DIR / "questions_extended.json") as f:
        all_questions = json.load(f)

    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    print(f"Loaded {len(store.segments)} segments")

    # Split by benchmark
    beam_qs = [q for q in all_questions if q.get("benchmark") == "beam"][:30]
    locomo_qs = [q for q in all_questions if q.get("benchmark") == "locomo"][:30]
    print(f"BEAM: {len(beam_qs)} questions, LoCoMo: {len(locomo_qs)} questions")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Define architectures to test
    arch_configs = {
        "segment_as_query": lambda: SegmentAsQuery(
            store, initial_top_k=5, walk_hops=4, walk_top_k=5
        ),
        "cluster_diversify": lambda: ClusterDiversify(
            store, initial_top_k=100, n_clusters=8
        ),
        "multi_query_fusion": lambda: MultiQueryFusion(
            store, num_queries=5, per_query_k=20
        ),
        "retrieve_summarize_retrieve": lambda: RetrieveSummarizeRetrieve(
            store, initial_top_k=10, summary_hops=2, per_hop_k=15
        ),
        "agent_working_set": lambda: AgentWorkingSet(
            store, max_tool_calls=5, per_search_k=10
        ),
        "hybrid_gap_fill": lambda: HybridGapFill(store, baseline_k=20, gap_fill_k=10),
        "centroid_walk": lambda: CentroidWalk(
            store, initial_top_k=10, hops=3, per_hop_k=10, drift_alpha=0.3
        ),
        "negative_space": lambda: NegativeSpace(
            store, initial_top_k=15, hops=2, per_hop_k=15, push_alpha=0.3
        ),
        "mmr_diversified": lambda: MMRDiversified(
            store, total_k=60, lambda_param=0.7, candidate_pool=150
        ),
    }

    all_summaries = []

    for arch_name, arch_factory in arch_configs.items():
        for benchmark_label, questions in [
            ("beam_30q", beam_qs),
            ("locomo_30q", locomo_qs),
        ]:
            results_file = RESULTS_DIR / f"arch_{arch_name}_{benchmark_label}.json"

            # Skip if already completed
            if results_file.exists():
                print(f"\nSkipping {arch_name}/{benchmark_label} (already exists)")
                existing = json.load(open(results_file))
                summary = print_summary(existing, arch_name, benchmark_label)
                all_summaries.append(summary)
                continue

            arch = arch_factory()
            results, summary = run_evaluation(
                arch_name, arch, questions, benchmark_label, verbose=False
            )
            all_summaries.append(summary)

            # Save detailed results
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Saved to {results_file}")

    # Save all summaries
    summary_file = RESULTS_DIR / "arch_all_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summaries to {summary_file}")

    # Print grand summary table
    print(f"\n{'=' * 100}")
    print("GRAND SUMMARY TABLE")
    print(f"{'=' * 100}")
    print(
        f"{'Architecture':<30s} {'Bench':>10s} {'B-r@20':>8s} {'A-r@20':>8s} "
        f"{'Delta':>8s} {'W/T/L':>10s} {'#Ret':>6s} {'Emb':>5s} {'LLM':>5s}"
    )
    print("-" * 100)
    for s in all_summaries:
        print(
            f"{s['arch']:<30s} {s['benchmark']:>10s} "
            f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
            f"{s['delta_r@20']:>+8.3f} {s['W/T/L_r@20']:>10s} "
            f"{s['avg_total_retrieved']:>6.0f} "
            f"{s['avg_embed_calls']:>5.1f} {s['avg_llm_calls']:>5.0f}"
        )

    # Reference: v15 baseline
    print("-" * 100)
    print("Reference v15: LoCoMo +0.339 W/T/L=13/17/0 | BEAM +0.066 W/T/L=6/22/2")


if __name__ == "__main__":
    main()
