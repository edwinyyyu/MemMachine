"""Vectorize-style BEAM evaluation."""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent.parent))
from beam_models import QuestionCategory
from llm_provider import (
    DEFAULT_JUDGE_MODEL,
    PROVIDER_OPENAI,
    PROVIDERS,
    ChatClient,
    make_chat_client,
)

RUBRIC_ITEM_PROMPT = """You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## QUESTION:
{query}

## LLM RESPONSE:
{answer}

## RUBRIC CRITERION:
{rubric_item}

## SCORING:
- **1.0** = Fully satisfied: The response clearly and completely addresses this rubric criterion.
- **0.5** = Partially satisfied: The response addresses this criterion but is incomplete, vague, or only partially correct.
- **0.0** = Not satisfied: The response does not address this criterion at all, or is incorrect.

Evaluate the response against ONLY this specific rubric criterion. Provide your score and a brief reason."""

LLM_EQUIVALENCE_PROMPT = """Do these two items refer to the same event, topic, or concept? Answer only YES or NO.

Item A: {a}
Item B: {b}

Answer (YES or NO):"""

RUBRIC_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Score: 0.0, 0.5, or 1.0"},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}

EQUIVALENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "YES or NO"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


async def _call_judge(
    client: ChatClient,
    model: str,
    prompt: str,
    schema: dict,
) -> dict:
    result = await client.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": True,
            },
        },
    )
    return json.loads(result["content"] or "{}")


def _clamp_three_way(raw: float) -> float:
    if raw >= 0.75:
        return 1.0
    if raw >= 0.25:
        return 0.5
    return 0.0


async def score_rubric_item(
    client: ChatClient,
    model: str,
    query: str,
    answer: str,
    rubric_item: str,
) -> tuple[float, str]:
    prompt = RUBRIC_ITEM_PROMPT.format(
        query=query, answer=answer, rubric_item=rubric_item
    )
    try:
        parsed = await _call_judge(client, model, prompt, RUBRIC_SCORE_SCHEMA)
        score = _clamp_three_way(float(parsed.get("score", 0)))
        reason = str(parsed.get("reason", ""))
    except Exception as e:
        score = 0.0
        reason = f"judge_error: {e}"
    return score, reason


async def llm_equivalence(client: ChatClient, model: str, a: str, b: str) -> bool:
    try:
        parsed = await _call_judge(
            client, model, LLM_EQUIVALENCE_PROMPT.format(a=a, b=b), EQUIVALENCE_SCHEMA
        )
        return str(parsed.get("answer", "")).strip().upper().startswith("YES")
    except Exception:
        return False


def _extract_ordered_items(text: str) -> list[str]:
    out: list[str] = []
    for line in text.strip().splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", line).strip()
        if cleaned:
            out.append(cleaned)
    return out


async def align_with_llm(
    client: ChatClient,
    model: str,
    reference: list[str],
    system: list[str],
) -> tuple[list[str], list[str]]:
    used: set[int] = set()
    system_out: list[str] = []
    for s in system:
        matched_idx = None
        for idx, r in enumerate(reference):
            if idx in used:
                continue
            if await llm_equivalence(client, model, r, s):
                matched_idx = idx
                break
        if matched_idx is not None:
            system_out.append(reference[matched_idx])
            used.add(matched_idx)
        else:
            system_out.append(s)
    return reference, system_out


async def event_ordering_score(
    client: ChatClient,
    model: str,
    reference: list[str],
    system: list[str],
) -> float:
    if not reference or not system:
        return 0.0
    _, system_canon = await align_with_llm(client, model, reference, system)

    union = list(dict.fromkeys(reference + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq: list[str]) -> list[int]:
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b, _ = kendalltau(to_rank(reference), to_rank(system_canon), variant="b")
    if tau_b is None or str(tau_b) == "nan":
        return 0.0
    return (tau_b + 1) / 2


async def score_result_for_sample(
    client: ChatClient,
    model: str,
    item: dict,
) -> dict:
    category = str(item.get("category", ""))
    question = str(item.get("question", ""))
    answer = str(item.get("model_answer", item.get("response", "")))

    result = dict(item)
    result["judge_model_responses"] = []

    if category == QuestionCategory.EVENT_ORDERING.value:
        reference = list(item.get("ordering_tested") or [])
        if not reference and item.get("gold_answer"):
            reference = _extract_ordered_items(str(item["gold_answer"]))
        system = _extract_ordered_items(answer)
        score = await event_ordering_score(client, model, reference, system)
        result["primary_score"] = score
        result["tau_norm"] = score
        return result

    rubric = list(item.get("rubric", []))
    if not rubric and item.get("gold_answer"):
        rubric = [f"LLM response should contain: {item['gold_answer']}"]

    scores: list[float] = []
    for rubric_item in rubric:
        s, reason = await score_rubric_item(
            client, model, question, answer, rubric_item
        )
        scores.append(s)
        result["judge_model_responses"].append(
            {"rubric_item": rubric_item, "score": s, "reason": reason}
        )

    rubric_avg = sum(scores) / len(scores) if scores else 0.0
    result["primary_score"] = rubric_avg
    result["llm_judge_score"] = rubric_avg
    return result


async def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--search-path", required=True, help="Path to vectorize beam_search output"
    )
    parser.add_argument("--target-path", required=True, help="Path to output JSON")
    parser.add_argument(
        "--judge-provider",
        default=PROVIDER_OPENAI,
        choices=list(PROVIDERS),
        help="Judge LLM provider",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge LLM model (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    judge_model = args.judge_model

    with open(args.search_path) as f:
        search_results = json.load(f)

    client = make_chat_client(args.judge_provider)

    flat: list[tuple[str, int, dict]] = []
    for category, items in search_results.items():
        for idx, item in enumerate(items):
            flat.append((category, idx, item))

    total = len(flat)
    remaining = total
    sem = asyncio.Semaphore(args.concurrency)

    async def _run(category: str, item: dict):
        nonlocal remaining
        async with sem:
            scored = await score_result_for_sample(client, judge_model, item)
        remaining -= 1
        print(
            f"[{category}] conv={item.get('conversation_id')} "
            f"idx={item.get('question_index')} "
            f"score={scored['primary_score']:.3f} "
            f"({total - remaining}/{total}, {remaining} left)"
        )
        return scored

    tasks = [_run(category, item) for category, _, item in flat]
    scored_flat = await asyncio.gather(*tasks)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for (category, _, _), scored in zip(flat, scored_flat, strict=True):
        by_category[category].append(scored)

    summary: dict[str, dict] = {}
    all_scores: list[float] = []
    for cat in sorted(by_category):
        scores = [s["primary_score"] for s in by_category[cat]]
        if not scores:
            continue
        summary[cat] = {"count": len(scores), "mean_score": sum(scores) / len(scores)}
        all_scores.extend(scores)
    if all_scores:
        summary["overall"] = {
            "count": len(all_scores),
            "mean_score": sum(all_scores) / len(all_scores),
        }

    output = {
        "summary": summary,
        "mode": "vectorize_score_result",
        "results": dict(by_category),
    }
    with open(args.target_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== BEAM Evaluation Summary (Vectorize score_result) ===")
    for cat, m in sorted(summary.items()):
        print(f"  {cat:30s} {m['mean_score']:.4f}  ({m['count']} samples)")

    await client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
