"""Re-analyze existing associative recall results with proper baseline normalization.

Instead of re-running API calls, this recomputes baseline recall at the same
segment budget as the associative retrieval used, using the stored embeddings.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from associative_recall import SegmentStore

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BUDGETS = [20, 50, 100, 150]


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    hits = retrieved_turn_ids & source_turn_ids
    return len(hits) / len(source_turn_ids)


def reanalyze(results_file: Path, store: SegmentStore) -> list[dict]:
    """Reanalyze a results file with proper normalization."""
    with open(results_file) as f:
        results = json.load(f)

    # We need question embeddings. We can re-embed via cached embeddings.
    from associative_recall import EmbeddingCache
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    client = OpenAI(timeout=60.0)
    embedding_cache = EmbeddingCache()

    reanalyzed = []
    for r in results:
        q_text = r["question"]
        conv_id = r["conversation_id"]
        source_ids = set(r["source_chat_ids"])
        total_assoc = r["total_retrieved"]

        # Get conversation size
        mask = store.conversation_ids == conv_id
        conv_length = int(mask.sum())

        # Get question embedding (should be cached)
        import hashlib

        key = hashlib.sha256(q_text.encode()).hexdigest()
        cached = embedding_cache.get(q_text)
        if cached is None:
            resp = client.embeddings.create(
                model="text-embedding-3-small", input=[q_text]
            )
            cached = np.array(resp.data[0].embedding, dtype=np.float32)
            embedding_cache.put(q_text, cached)

        # Compute baseline at various budgets
        max_budget = max(BUDGETS + [total_assoc])
        baseline_result = store.search(
            cached, top_k=max_budget, conversation_id=conv_id
        )

        baseline_recalls = {}
        for budget in BUDGETS:
            ids = {s.turn_id for s in baseline_result.segments[:budget]}
            baseline_recalls[f"r@{budget}"] = compute_recall(ids, source_ids)
        # Baseline at actual assoc budget
        ids_at_actual = {s.turn_id for s in baseline_result.segments[:total_assoc]}
        baseline_recalls["r@actual"] = compute_recall(ids_at_actual, source_ids)

        # Compute assoc recall at various budgets
        # We need to reconstruct the ordered retrieval from hop_details
        # Since we don't have the full segments, we use the original assoc_recalls
        # and also compute from the hop data
        assoc_all_ids = set()
        assoc_ordered_hits = []
        for hop in r["hop_details"]:
            for tid in hop.get("new_source_hits", []):
                if tid not in assoc_all_ids:
                    assoc_ordered_hits.append(tid)
            assoc_all_ids.update(hop.get("new_source_hits", []))

        # For the budget-based recall, we need to know which turn IDs were
        # retrieved at each point. We only have counts, not full turn IDs.
        # Use the original r@all assoc recall as "r@actual"
        assoc_recalls = {}
        assoc_recalls["r@actual"] = r["assoc_recalls"].get("r@all", 0.0)

        # For budget-based assoc recalls, we can't perfectly reconstruct
        # without the full segment list. Use the hop cumulative counts as proxy.
        cumulative_segments = 0
        cumulative_recall = 0.0
        budget_recalls = {}
        for hop in r["hop_details"]:
            cumulative_segments += hop["num_retrieved"]
            cum_hits = set()
            for h2 in r["hop_details"][: r["hop_details"].index(hop) + 1]:
                cum_hits.update(h2.get("new_source_hits", []))
            this_recall = compute_recall(cum_hits, source_ids)
            budget_recalls[cumulative_segments] = this_recall

        # Interpolate to standard budgets
        sorted_budgets = sorted(budget_recalls.keys())
        for budget in BUDGETS:
            if budget >= total_assoc:
                assoc_recalls[f"r@{budget}"] = assoc_recalls["r@actual"]
            elif budget <= sorted_budgets[0]:
                assoc_recalls[f"r@{budget}"] = budget_recalls[sorted_budgets[0]]
            else:
                # Find the closest budget we have data for
                best_key = min(sorted_budgets, key=lambda x: abs(x - budget))
                if best_key <= budget:
                    assoc_recalls[f"r@{budget}"] = budget_recalls[best_key]
                else:
                    # Conservative: use the one below
                    below = [k for k in sorted_budgets if k <= budget]
                    if below:
                        assoc_recalls[f"r@{budget}"] = budget_recalls[below[-1]]
                    else:
                        assoc_recalls[f"r@{budget}"] = budget_recalls[sorted_budgets[0]]

        reanalyzed.append(
            {
                "question": q_text,
                "conversation_id": conv_id,
                "category": r["category"],
                "source_chat_ids": sorted(source_ids),
                "num_source_turns": len(source_ids),
                "conv_length": conv_length,
                "total_assoc_retrieved": total_assoc,
                "retrieval_fraction": round(total_assoc / conv_length, 3),
                "baseline_recalls": baseline_recalls,
                "assoc_recalls": assoc_recalls,
            }
        )

    embedding_cache.save()
    return reanalyzed


def print_summary(results: list[dict], label: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"NORMALIZED REANALYSIS: {label}")
    print(f"{'=' * 80}")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    budget_labels = [f"r@{b}" for b in BUDGETS] + ["r@actual"]

    print(f"\n{'Budget':>8s}", end="")
    for lbl in budget_labels:
        print(f"  {'B-' + lbl:>10s}  {'A-' + lbl:>10s}  {'Delta':>7s}", end="")
    print()
    print("-" * 120)

    all_results = list(results)
    for lbl in budget_labels:
        b_mean = sum(r["baseline_recalls"].get(lbl, 0) for r in all_results) / len(
            all_results
        )
        a_mean = sum(r["assoc_recalls"].get(lbl, 0) for r in all_results) / len(
            all_results
        )
        print(
            f"  {lbl:>8s}: baseline={b_mean:.3f}  assoc={a_mean:.3f}  delta={a_mean - b_mean:+.3f}"
        )

    avg_frac = sum(r["retrieval_fraction"] for r in all_results) / len(all_results)
    avg_total = sum(r["total_assoc_retrieved"] for r in all_results) / len(all_results)
    print(f"\nAvg retrieval fraction: {avg_frac:.2%}")
    print(f"Avg total assoc segments: {avg_total:.0f}")

    # Per question detail
    print("\nPER-QUESTION DETAIL:")
    print(f"{'Q':>3s} {'Category':>30s} {'Conv':>4s} {'Retr':>4s} {'Frac':>5s}", end="")
    print(f"  {'B@50':>5s} {'A@50':>5s} {'D@50':>5s}", end="")
    print(f"  {'B@act':>5s} {'A@act':>5s} {'D@act':>5s}")
    for i, r in enumerate(all_results):
        b50 = r["baseline_recalls"].get("r@50", 0)
        a50 = r["assoc_recalls"].get("r@50", 0)
        bact = r["baseline_recalls"].get("r@actual", 0)
        aact = r["assoc_recalls"].get("r@actual", 0)
        print(
            f"{i:>3d} {r['category']:>30s} {r['conversation_id']:>4s} "
            f"{r['total_assoc_retrieved']:>4d} {r['retrieval_fraction']:>5.2f}"
            f"  {b50:>5.3f} {a50:>5.3f} {a50 - b50:>+5.3f}"
            f"  {bact:>5.3f} {aact:>5.3f} {aact - bact:>+5.3f}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", nargs="?", default=None)
    args = parser.parse_args()

    store = SegmentStore()
    print(f"Loaded {len(store.segments)} segments")

    if args.results_file:
        files = [Path(args.results_file)]
    else:
        files = sorted(RESULTS_DIR.glob("results_*.json"))

    for f in files:
        label = f.stem
        print(f"\nReanalyzing: {label}")
        reanalyzed = reanalyze(f, store)
        print_summary(reanalyzed, label)


if __name__ == "__main__":
    main()
