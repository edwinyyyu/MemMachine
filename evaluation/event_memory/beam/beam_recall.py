"""Measure retrieval recall against BEAM `source_chat_ids`."""

import argparse
import ast
import json
from collections import defaultdict

_RANK_DEPTHS = (1, 5, 10, 25, 50, None)


def _parse_probing_questions(item: dict) -> dict[str, list[dict]]:
    pq = item.get("probing_questions", {})
    if isinstance(pq, str):
        try:
            pq = json.loads(pq)
        except (json.JSONDecodeError, ValueError):
            try:
                pq = ast.literal_eval(pq)
            except (ValueError, SyntaxError):
                pq = {}
    return pq if isinstance(pq, dict) else {}


def _flatten_source_chat_ids(raw: object) -> set[int]:
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
    props = segment.get("properties") or {}
    tid_wrapped = props.get("beam_turn_id")
    v = tid_wrapped.get("v") if isinstance(tid_wrapped, dict) else tid_wrapped
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _retrieved_turn_ids_by_rank(query_result: dict) -> list[set[int]]:
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
    if not source:
        return 1.0
    k = len(ranked_ids) if depth is None else min(depth, len(ranked_ids))
    seen: set[int] = set()
    for i in range(k):
        seen |= ranked_ids[i]
    return len(seen & source) / len(source)


UNJOINED = "unjoined"
NO_SOURCE = "no_source"


def _score_item(
    category: str,
    item: dict,
    source_index: dict[tuple[str, str, int], set[int]],
) -> tuple[str, dict | None]:
    """Return (status, row). status ∈ {"ok", UNJOINED, NO_SOURCE}; row is None for non-ok."""
    conv_id = str(item.get("conversation_id", ""))
    q_idx = item.get("question_index")
    if q_idx is None:
        return UNJOINED, None
    key = (conv_id, category, int(q_idx))
    if key not in source_index:
        return UNJOINED, None
    source = source_index[key]
    if category == "abstention" or not source:
        return NO_SOURCE, None

    ranked_ids = _retrieved_turn_ids_by_rank(item.get("query_result", {}))
    recalls = {
        (f"r@{d}" if d is not None else "r@all"): _recall_at_depth(
            ranked_ids, source, d
        )
        for d in _RANK_DEPTHS
    }
    return "ok", {
        "conversation_id": conv_id,
        "question_index": q_idx,
        "source_turn_count": len(source),
        "retrieved_rank_count": len(ranked_ids),
        "recalls": recalls,
    }


def _means_over_rows(rows: list[dict]) -> dict[str, float]:
    means: dict[str, float] = {}
    for d in _RANK_DEPTHS:
        key = f"r@{d}" if d is not None else "r@all"
        means[key] = sum(r["recalls"][key] for r in rows) / len(rows)
    return means


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
            status, row = _score_item(category, item, source_index)
            if status == UNJOINED:
                skipped_unjoined += 1
            elif status == NO_SOURCE:
                skipped_no_source += 1
            elif row is not None:
                per_category[category].append(row)

    summary: dict[str, dict] = {}
    all_questions: list[dict] = []
    for cat in sorted(per_category):
        rows = per_category[cat]
        all_questions.extend(rows)
        summary[cat] = {"count": len(rows), **_means_over_rows(rows)}

    if all_questions:
        summary["overall"] = {
            "count": len(all_questions),
            **_means_over_rows(all_questions),
        }

    if args.target_path:
        output = {
            "summary": summary,
            "skipped_no_source": skipped_no_source,
            "skipped_unjoined": skipped_unjoined,
            "per_question": dict(per_category),
        }
        with open(args.target_path, "w") as f:
            json.dump(output, f, indent=2)

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
