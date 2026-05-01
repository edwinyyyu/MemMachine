"""Measure retrieval recall against BEAM `source_chat_ids`.

Pure post-hoc analysis on a search output file produced by `beam_search.py`
(official or vectorize). Joins each result back to its probing question in the
raw BEAM dataset, extracts the ground-truth `source_chat_ids`, and compares
against the `beam_turn_id` values carried on every retrieved segment's
properties.

This is a memory-only metric — no LLM calls, no model answers. It answers:
"When our pipeline reports a given score, is the ceiling the retriever or the
answerer?" High recall + low BEAM score → answerer-bound. Low recall → memory-
bound.

Reports recall at several cumulative rank depths (top-1, top-5, top-10,
top-25, top-50, all). Each "rank" is one scored segment context in reranker
order; a context may contain multiple segments due to `--expand-context`, so
all its segments count toward recall at that rank.

Excludes `abstention` questions (no ground-truth source turns — the answer is
supposed to be "I don't know").
"""

import argparse
import ast
import json
from collections import defaultdict

# Category → list of dict subfield names to flatten when source_chat_ids is a
# dict. Any subfield not listed here is still picked up by the dict-flatten
# fallback; this mapping is only documentation of the expected schema.
_EXPECTED_DICT_SUBFIELDS: dict[str, tuple[str, ...]] = {
    "contradiction_resolution": ("first_statement", "second_statement"),
    "knowledge_update": ("original_info", "updated_info"),
    "temporal_reasoning": ("first_event", "second_event"),
}

_RANK_DEPTHS = (1, 5, 10, 25, 50, None)  # None = all ranks


def _parse_probing_questions(item: dict) -> dict[str, list[dict]]:
    pq = item.get("probing_questions", {})
    if isinstance(pq, str):
        try:
            pq = json.loads(pq)
        except Exception:
            try:
                pq = ast.literal_eval(pq)
            except Exception:
                pq = {}
    return pq if isinstance(pq, dict) else {}


def _flatten_source_chat_ids(raw: object) -> set[int]:
    """Return all turn IDs mentioned in `source_chat_ids`, flattened.

    Handles both shapes seen in BEAM:
    - list[int] — single flat list of turn IDs
    - dict[str, list[int]] — e.g. {"first_statement": [58], ...}
    """
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {
            int(x)
            for x in raw
            if isinstance(x, (int, float, str)) and str(x).lstrip("-").isdigit()
        }
    if isinstance(raw, dict):
        out: set[int] = set()
        for v in raw.values():
            out |= _flatten_source_chat_ids(v)
        return out
    return set()


def build_source_index(
    data_path: str,
) -> dict[tuple[str, str, int], set[int]]:
    """Index (conversation_id, category, index_within_category) → source turn IDs.

    Mirrors `beam_models._build_questions`: iterates categories in
    `ALL_CATEGORIES` order and assigns each question a 0-based index within
    its category. The search script writes `question_index` from the same
    source, so the join is direct.
    """
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)

    index: dict[tuple[str, str, int], set[int]] = {}
    for conv_idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        conv_id = str(item.get("conversation_id", f"conv_{conv_idx}"))
        pq = _parse_probing_questions(item)
        for cat, items in pq.items():
            if not isinstance(items, list):
                continue
            for i, q in enumerate(items):
                if not isinstance(q, dict):
                    continue
                sci = q.get("source_chat_ids")
                index[(conv_id, cat, i)] = _flatten_source_chat_ids(sci)
    return index


def _segment_turn_id(segment: dict) -> int | None:
    """Pull `beam_turn_id` out of a segment's type-tagged properties dict."""
    props = segment.get("properties") or {}
    tid_wrapped = props.get("beam_turn_id")
    if isinstance(tid_wrapped, dict):
        v = tid_wrapped.get("v")
    else:
        v = tid_wrapped
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _retrieved_turn_ids_by_rank(query_result: dict) -> list[set[int]]:
    """Return retrieved turn IDs grouped by reranker rank.

    Each scored segment context occupies one rank. The returned list[i] is the
    set of turn IDs contributed by rank i (which may be more than one segment
    when `--expand-context > 0` was used at search time).
    """
    contexts = (query_result or {}).get("scored_segment_contexts", [])
    out: list[set[int]] = []
    for ctx in contexts:
        ids: set[int] = set()
        for seg in ctx.get("segments", []):
            tid = _segment_turn_id(seg)
            if tid is not None:
                ids.add(tid)
        out.append(ids)
    return out


def _recall_at_depth(
    ranked_ids: list[set[int]], source: set[int], depth: int | None
) -> float:
    """Recall of ground-truth `source` turns within the top-`depth` ranks.

    `depth=None` uses all ranks. Returns 1.0 if `source` is empty (degenerate).
    """
    if not source:
        return 1.0
    k = len(ranked_ids) if depth is None else min(depth, len(ranked_ids))
    seen: set[int] = set()
    for i in range(k):
        seen |= ranked_ids[i]
    return len(seen & source) / len(source)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--search-path", required=True, help="Path to beam_search.py output JSON"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to raw BEAM dataset JSON (for source_chat_ids)",
    )
    parser.add_argument(
        "--target-path",
        default=None,
        help="Optional: write per-question recall details to this JSON file",
    )
    args = parser.parse_args()

    source_index = build_source_index(args.data_path)

    with open(args.search_path) as f:
        search_results = json.load(f)

    per_category: dict[str, list[dict]] = defaultdict(list)
    skipped_no_source = 0
    skipped_unjoined = 0

    for category, items in search_results.items():
        for item in items:
            conv_id = str(item.get("conversation_id", ""))
            q_idx = item.get("question_index")
            if q_idx is None:
                skipped_unjoined += 1
                continue
            key = (conv_id, category, int(q_idx))
            if key not in source_index:
                skipped_unjoined += 1
                continue
            source = source_index[key]
            if category == "abstention" or not source:
                skipped_no_source += 1
                continue

            ranked_ids = _retrieved_turn_ids_by_rank(item.get("query_result", {}))
            recalls = {
                (f"r@{d}" if d is not None else "r@all"): _recall_at_depth(
                    ranked_ids, source, d
                )
                for d in _RANK_DEPTHS
            }
            per_category[category].append(
                {
                    "conversation_id": conv_id,
                    "question_index": q_idx,
                    "source_turn_count": len(source),
                    "retrieved_rank_count": len(ranked_ids),
                    "recalls": recalls,
                }
            )

    # Aggregate: per-category and overall mean recall at each depth.
    summary: dict[str, dict] = {}
    all_questions: list[dict] = []
    for cat in sorted(per_category):
        rows = per_category[cat]
        all_questions.extend(rows)
        means = {}
        for d in _RANK_DEPTHS:
            key = f"r@{d}" if d is not None else "r@all"
            means[key] = sum(r["recalls"][key] for r in rows) / len(rows)
        summary[cat] = {"count": len(rows), **means}

    if all_questions:
        overall_means = {}
        for d in _RANK_DEPTHS:
            key = f"r@{d}" if d is not None else "r@all"
            overall_means[key] = sum(r["recalls"][key] for r in all_questions) / len(
                all_questions
            )
        summary["overall"] = {"count": len(all_questions), **overall_means}

    if args.target_path:
        output = {
            "summary": summary,
            "skipped_no_source": skipped_no_source,
            "skipped_unjoined": skipped_unjoined,
            "per_question": dict(per_category),
        }
        with open(args.target_path, "w") as f:
            json.dump(output, f, indent=2)

    # Console report.
    depth_labels = [f"r@{d}" if d is not None else "r@all" for d in _RANK_DEPTHS]
    header = f"{'category':30s} {'n':>4s}  " + "  ".join(
        f"{lbl:>7s}" for lbl in depth_labels
    )
    print("\n=== BEAM Retrieval Recall ===")
    print(
        f"(skipped: {skipped_no_source} with no source, {skipped_unjoined} unjoinable)"
    )
    print(header)
    print("-" * len(header))
    for cat in sorted(summary):
        m = summary[cat]
        row = f"{cat:30s} {m['count']:>4d}  " + "  ".join(
            f"{m[lbl]:>7.3f}" for lbl in depth_labels
        )
        print(row)


if __name__ == "__main__":
    main()
