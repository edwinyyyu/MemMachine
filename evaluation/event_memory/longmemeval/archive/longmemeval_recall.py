import argparse
import json
from collections import defaultdict
from datetime import datetime
from uuid import UUID

from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    QueryResult,
    ScoredSegmentContext,
    Segment,
    Text,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from pydantic import TypeAdapter

_ContextAdapter = TypeAdapter(Context | None)


def _turn_key(props: dict) -> str:
    return f"{props['longmemeval_session_id']}:{props['turn_id']}"


def _reconstruct_segment(seg_data: dict) -> Segment:
    return Segment(
        uuid=UUID(seg_data["uuid"]),
        event_uuid=UUID(seg_data["event_uuid"]),
        index=seg_data["index"],
        offset=seg_data["offset"],
        timestamp=datetime.fromisoformat(seg_data["timestamp"]),
        context=_ContextAdapter.validate_python(seg_data.get("context")),
        block=Text(text=seg_data.get("text") or ""),
        properties=seg_data.get("properties", {}),
    )


def compute_recall(items: list[dict], max_k: int) -> dict:
    """Recall@k by ranked context position (fast).

    A turn split into multiple segments is counted once at the earliest rank
    any of its segments appears (optimistic upper bound).
    """
    total_answer_turns = 0
    cumulative_hits_at_k = [0] * (max_k + 1)
    per_question = []

    for item in items:
        answer_turns = set(item["answer_turn_indices"])
        if not answer_turns:
            per_question.append(
                {
                    "question_id": item["question_id"],
                    "num_answer_turns": 0,
                    "first_hit_rank": None,
                }
            )
            continue

        total_answer_turns += len(answer_turns)

        found_at_rank: dict[str, int] = {}
        for sc in item["segment_contexts"]:
            rank = sc["rank"]
            for seg in sc["segments"]:
                tk = _turn_key(seg["properties"])
                if tk in answer_turns and tk not in found_at_rank:
                    found_at_rank[tk] = rank

        rank_hits: dict[int, int] = defaultdict(int)
        for rank in found_at_rank.values():
            rank_hits[rank] += 1

        cumulative = 0
        for k in range(max_k + 1):
            cumulative += rank_hits.get(k, 0)
            cumulative_hits_at_k[k] += cumulative

        first_hit_rank = min(found_at_rank.values()) if found_at_rank else None
        per_question.append(
            {
                "question_id": item["question_id"],
                "num_answer_turns": len(answer_turns),
                "recalled": len(found_at_rank),
                "first_hit_rank": first_hit_rank,
            }
        )

    recall_at_k = [
        hits / total_answer_turns if total_answer_turns > 0 else 0.0
        for hits in cumulative_hits_at_k
    ]
    return {
        "count": len(items),
        "total_answer_turns": total_answer_turns,
        "recall_at_k": recall_at_k,
        "per_question": per_question,
    }


def compute_recall_unified(items: list[dict], max_k: int) -> dict:
    """Recall@k with unification (slow).

    For each segment budget k, runs _unify_anchored_segment_contexts and
    checks which answer turns appear in the unified result.
    """
    total_answer_turns = 0
    cumulative_hits_at_k = [0] * (max_k + 1)
    per_question = []

    for item in items:
        answer_turns = set(item["answer_turn_indices"])
        if not answer_turns:
            per_question.append(
                {
                    "question_id": item["question_id"],
                    "num_answer_turns": 0,
                    "first_hit_k": None,
                }
            )
            continue

        total_answer_turns += len(answer_turns)

        scored_segment_contexts = []
        for rank, sc in enumerate(item["segment_contexts"]):
            segments = [_reconstruct_segment(s) for s in sc["segments"]]
            seed_uuid = UUID(sc["seed_segment_uuid"])
            scored_segment_contexts.append(
                ScoredSegmentContext(
                    score=float(len(item["segment_contexts"]) - rank),
                    seed_segment_uuid=seed_uuid,
                    segments=segments,
                )
            )
        query_result = QueryResult(scored_segment_contexts=scored_segment_contexts)

        first_hit_k = None
        for k in range(max_k + 1):
            unified = EventMemory.build_query_result_context(
                query_result, max_num_segments=k
            )
            found = {_turn_key(seg.properties) for seg in unified} & answer_turns
            cumulative_hits_at_k[k] += len(found)
            if found and first_hit_k is None:
                first_hit_k = k

        recalled_at_max = len(
            {
                _turn_key(seg.properties)
                for seg in EventMemory.build_query_result_context(
                    query_result, max_num_segments=max_k
                )
            }
            & answer_turns
        )
        per_question.append(
            {
                "question_id": item["question_id"],
                "num_answer_turns": len(answer_turns),
                "recalled": recalled_at_max,
                "first_hit_k": first_hit_k,
            }
        )

    recall_at_k = [
        hits / total_answer_turns if total_answer_turns > 0 else 0.0
        for hits in cumulative_hits_at_k
    ]
    return {
        "count": len(items),
        "total_answer_turns": total_answer_turns,
        "recall_at_k": recall_at_k,
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-path", required=True, help="Path to longmemeval_search_raw output"
    )
    parser.add_argument("--target-path", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--max-k", type=int, default=200, help="Maximum k for recall@k (default: 200)"
    )
    parser.add_argument(
        "--unify",
        action="store_true",
        help="Use unification strategy for recall@k (much slower)",
    )
    args = parser.parse_args()

    with open(args.search_path) as f:
        search_results = json.load(f)

    max_k = args.max_k
    recall_fn = compute_recall_unified if args.unify else compute_recall

    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in search_results:
        by_category[item["question_type"]].append(item)

    results = {"mode": "unified" if args.unify else "ranked"}
    for category, items in sorted(by_category.items()):
        results[category] = recall_fn(items, max_k)
    results["overall"] = recall_fn(search_results, max_k)

    with open(args.target_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
