"""Evaluate agent-centric retrieval architectures.

Normalized comparison: each architecture is compared against a cosine
baseline given the SAME retrieval budget (total segments retrieved).

Reports:
- r@20, r@50 recall for both architecture and baseline
- Win/Tie/Loss at r@20
- Average segments retrieved, LLM calls, embed calls
- Per-category breakdown for hard categories
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from agent_architectures import (
    AGENT_ARCHITECTURES,
    AgentBase,
)
from associative_recall import SegmentStore

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50]


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: AgentBase,
    question: dict,
    verbose: bool = False,
) -> dict:
    """Evaluate a single architecture on a single question."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    arch_result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Deduplicate segments preserving order
    seen = set()
    deduped: list = []
    for seg in arch_result.segments:
        if seg.index not in seen:
            deduped.append(seg)
            seen.add(seg.index)
    arch_segments = deduped
    total_retrieved = len(arch_segments)

    # Baseline: cosine top-N with same budget
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

    # Normalized: baseline gets same total as arch
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

    return result


def summarize(results: list[dict], arch_name: str, benchmark: str) -> dict:
    """Compute summary statistics from a list of per-question results."""
    n = len(results)
    if n == 0:
        return {}

    summary = {"arch": arch_name, "benchmark": benchmark, "n": n}

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
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    summary["avg_time_s"] = round(sum(r["time_s"] for r in results) / n, 2)

    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    """Group results by category and compute per-category summaries."""
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


def run_architecture(
    arch_name: str,
    arch: AgentBase,
    questions: list[dict],
    benchmark_label: str,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Run evaluation for one architecture on one benchmark."""
    print(f"\n{'=' * 70}")
    print(
        f"ARCH: {arch_name} | BENCHMARK: {benchmark_label} | {len(questions)} questions"
    )
    print(f"{'=' * 70}")

    results = []
    for i, question in enumerate(questions):
        q_short = question["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {question['category']}: {q_short}...",
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
    summary = summarize(results, arch_name, benchmark_label)

    # Print compact summary
    print(f"\n--- {arch_name} on {benchmark_label} ---")
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

    # Per-category breakdown
    cat_summaries = summarize_by_category(results)
    print("\n  Per-category (r@20):")
    for cat, cs in cat_summaries.items():
        print(
            f"    {cat}: delta={cs['delta_r@20']:+.3f} "
            f"W/T/L={cs['W/T/L']} (n={cs['n']})"
        )

    return results, summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Run specific architecture (default: all)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=["beam", "locomo", "both"],
        help="Run specific benchmark (default: both)",
    )
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing results"
    )
    args = parser.parse_args()

    # Load data
    with open(DATA_DIR / "questions_extended.json") as f:
        all_questions = json.load(f)

    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    print(f"Loaded {len(store.segments)} segments")

    beam_qs = [q for q in all_questions if q.get("benchmark") == "beam"][
        : args.num_questions
    ]
    locomo_qs = [q for q in all_questions if q.get("benchmark") == "locomo"][
        : args.num_questions
    ]
    print(f"BEAM: {len(beam_qs)} questions, LoCoMo: {len(locomo_qs)} questions")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which architectures to run
    if args.arch:
        arch_names = [args.arch]
    else:
        arch_names = list(AGENT_ARCHITECTURES.keys())

    # Determine which benchmarks to run
    benchmarks = []
    if args.benchmark in (None, "both", "locomo"):
        benchmarks.append(("locomo_30q", locomo_qs))
    if args.benchmark in (None, "both", "beam"):
        benchmarks.append(("beam_30q", beam_qs))

    all_summaries = []

    for arch_name in arch_names:
        if arch_name not in AGENT_ARCHITECTURES:
            print(f"Unknown architecture: {arch_name}")
            continue

        for benchmark_label, questions in benchmarks:
            results_file = RESULTS_DIR / f"agent_{arch_name}_{benchmark_label}.json"

            if results_file.exists() and not args.force:
                print(
                    f"\nSkipping {arch_name}/{benchmark_label} (exists, "
                    f"use --force to overwrite)"
                )
                existing = json.load(open(results_file))
                summary = summarize(existing, arch_name, benchmark_label)
                all_summaries.append(summary)

                # Still print summary
                print(
                    f"  r@20: delta={summary.get('delta_r@20', 0):+.3f} "
                    f"W/T/L={summary.get('W/T/L_r@20', '?')}"
                )
                continue

            arch_cls = AGENT_ARCHITECTURES[arch_name]
            arch = arch_cls(store)
            results, summary = run_architecture(
                arch_name,
                arch,
                questions,
                benchmark_label,
                verbose=args.verbose,
            )
            all_summaries.append(summary)

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Saved to {results_file}")

    # Save all summaries
    summary_file = RESULTS_DIR / "agent_all_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Grand summary table
    print(f"\n{'=' * 110}")
    print("GRAND SUMMARY TABLE — Agent Architectures")
    print(f"{'=' * 110}")
    print(
        f"{'Architecture':<25s} {'Bench':>10s} {'B-r@20':>8s} {'A-r@20':>8s} "
        f"{'Delta':>8s} {'W/T/L':>10s} {'#Ret':>6s} {'Emb':>5s} "
        f"{'LLM':>5s} {'Time':>6s}"
    )
    print("-" * 110)
    for s in all_summaries:
        if not s:
            continue
        print(
            f"{s['arch']:<25s} {s['benchmark']:>10s} "
            f"{s.get('baseline_r@20', 0):>8.3f} "
            f"{s.get('arch_r@20', 0):>8.3f} "
            f"{s.get('delta_r@20', 0):>+8.3f} "
            f"{s.get('W/T/L_r@20', '?'):>10s} "
            f"{s.get('avg_total_retrieved', 0):>6.0f} "
            f"{s.get('avg_embed_calls', 0):>5.1f} "
            f"{s.get('avg_llm_calls', 0):>5.0f} "
            f"{s.get('avg_time_s', 0):>6.1f}"
        )
    print("-" * 110)
    print(
        "Reference v15 (1 hop, 2 cues): LoCoMo +0.339 (13W/17T/0L) | "
        "BEAM +0.066 (6W/22T/2L)"
    )
    print(
        "Reference agent_working_set:   LoCoMo +0.192 (9W/18T/3L) | "
        "BEAM -0.146 (1W/21T/8L)"
    )


if __name__ == "__main__":
    main()
