"""Official BEAM rubric evaluation, adapted to read EventMemory search output."""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
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

# Verbatim from
# https://github.com/mohammadtavakoli78/BEAM/blob/main/src/prompts.py
# (unified_llm_judge_base_prompt)
UNIFIED_JUDGE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): {question}
- RUBRIC CRITERION (what to check): {rubric_item}
- RESPONSE TO EVALUATE: {llm_response}

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}}

NOTE: ONLY output the json object, without any explanation before or after that
"""

LLM_EQUIVALENCE_PROMPT = """Do these two items refer to the same event, topic, or concept?

Item A: {a}
Item B: {b}

Return JSON: {{"answer": "YES"}} or {{"answer": "NO"}}"""

OFFICIAL_EQUIVALENCE_SYSTEM = """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """

OFFICIAL_EQUIVALENCE_USER_TEMPLATE = """First snippet: {a} \n
                       Second snippet: {b}
                    """


@dataclass(frozen=True)
class MatchOfficial:
    int_bug: bool = False
    extraction: bool = False
    equivalence: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MatchOfficial":
        if args.match_official_methodology:
            return cls(int_bug=True, extraction=True, equivalence=True)
        return cls(
            int_bug=args.match_official_int_bug,
            extraction=args.match_official_extraction,
            equivalence=args.match_official_equivalence,
        )

    def any(self) -> bool:
        return self.int_bug or self.extraction or self.equivalence


def _strip_code_fence(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def _parse_json_response(text: str) -> dict:
    text = _strip_code_fence(text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No valid JSON in response: {text!r}")


def _coerce_rubric_score(raw: object, *, match_int_bug: bool) -> float:
    value = float(raw) if isinstance(raw, (bool, int, float, str)) else 0.0
    if match_int_bug:
        return float(int(value))
    return value


def _extract_ordered_items(text: str, *, match_official: bool = False) -> list[str]:
    if match_official:
        return text.split("\n")
    out: list[str] = []
    for line in text.strip().splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", line).strip()
        if cleaned:
            out.append(cleaned)
    return out


async def _call_judge(client: ChatClient, model: str, prompt: str) -> dict:
    result = await client.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json_response(result["content"])


async def score_rubric_item(
    client: ChatClient,
    model: str,
    question: str,
    rubric_item: str,
    llm_response: str,
    *,
    match_int_bug: bool,
) -> tuple[float, str]:
    prompt = UNIFIED_JUDGE_PROMPT.format(
        question=question,
        rubric_item=rubric_item,
        llm_response=llm_response,
    )
    try:
        parsed = await _call_judge(client, model, prompt)
        score = _coerce_rubric_score(
            parsed.get("score", 0), match_int_bug=match_int_bug
        )
        reason = str(parsed.get("reason", ""))
    except Exception as e:
        score = 0.0
        reason = f"judge_error: {e}"
    return score, reason


async def llm_equivalence(
    client: ChatClient,
    model: str,
    a: str,
    b: str,
    *,
    match_official: bool = False,
) -> bool:
    if match_official:
        try:
            result = await client.create(
                model=model,
                messages=[
                    {"role": "system", "content": OFFICIAL_EQUIVALENCE_SYSTEM},
                    {
                        "role": "user",
                        "content": OFFICIAL_EQUIVALENCE_USER_TEMPLATE.format(a=a, b=b),
                    },
                ],
            )
            return "yes" in result["content"].lower()
        except Exception:
            return False

    prompt = LLM_EQUIVALENCE_PROMPT.format(a=a, b=b)
    try:
        parsed = await _call_judge(client, model, prompt)
        return str(parsed.get("answer", "")).strip().upper().startswith("YES")
    except Exception:
        return False


async def align_system_to_reference(
    client: ChatClient,
    model: str,
    reference: list[str],
    system: list[str],
    *,
    match: MatchOfficial,
) -> list[str]:
    used: set[int] = set()
    aligned: list[str] = []
    for sys_item in system:
        matched = None
        for idx, ref_item in enumerate(reference):
            if idx in used:
                continue
            if await llm_equivalence(
                client, model, ref_item, sys_item, match_official=match.equivalence
            ):
                matched = idx
                break
        if matched is not None:
            aligned.append(reference[matched])
            used.add(matched)
        else:
            aligned.append(sys_item)
    return aligned


async def score_event_ordering(
    client: ChatClient,
    model: str,
    reference: list[str],
    llm_response: str,
    *,
    match: MatchOfficial,
) -> dict:
    if not reference:
        return {
            "tau_norm": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "final_score": 0.0,
        }

    system = (
        _extract_ordered_items(llm_response, match_official=match.extraction)
        or llm_response.strip().splitlines()
    )
    if not system:
        return {
            "tau_norm": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "final_score": 0.0,
        }

    system_canon = await align_system_to_reference(
        client, model, reference, system, match=match
    )

    tp = len(set(reference) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference])
    fn = len([x for x in reference if x not in system_canon])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    union = list(dict.fromkeys(reference + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq: list[str]) -> list[int]:
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b, _ = kendalltau(to_rank(reference), to_rank(system_canon), variant="b")
    tau_b_norm = (tau_b + 1) / 2 if tau_b is not None and str(tau_b) != "nan" else 0.0
    final_score = tau_b_norm * f1
    return {
        "tau_norm": tau_b_norm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "final_score": final_score,
    }


async def process_sample(
    client: ChatClient,
    model: str,
    item: dict,
    *,
    match: MatchOfficial,
) -> dict:
    question = str(item.get("question", ""))
    llm_response = str(item.get("model_answer", item.get("response", "")))
    rubric = list(item.get("rubric", []))
    category = str(item.get("category", ""))

    result: dict = dict(item)
    result["judge_model_responses"] = []

    rubric_scores: list[float] = []
    for rubric_item in rubric:
        score, reason = await score_rubric_item(
            client,
            model,
            question,
            rubric_item,
            llm_response,
            match_int_bug=match.int_bug,
        )
        rubric_scores.append(score)
        result["judge_model_responses"].append(
            {"rubric_item": rubric_item, "score": score, "reason": reason}
        )

    rubric_avg = sum(rubric_scores) / len(rubric_scores) if rubric_scores else 0.0
    result["llm_judge_score"] = rubric_avg

    if category == QuestionCategory.EVENT_ORDERING.value:
        reference = rubric
        ordering = await score_event_ordering(
            client, model, reference, llm_response, match=match
        )
        result.update(ordering)
        result["primary_score"] = ordering["tau_norm"]
    else:
        result["primary_score"] = rubric_avg

    return result


async def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--search-path", required=True, help="Path to beam_search.py output"
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
        help=f"Judge LLM model (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Max concurrent samples"
    )
    parser.add_argument(
        "--match-official-int-bug",
        action="store_true",
        help="Mimic official compute_metrics.py: cast judge score to int.",
    )
    parser.add_argument(
        "--match-official-extraction",
        action="store_true",
        help="Mimic official evaluate_event_ordering: split on newlines without prefix stripping.",
    )
    parser.add_argument(
        "--match-official-equivalence",
        action="store_true",
        help="Mimic official llm_equivalence: two-message system+user prompt.",
    )
    parser.add_argument(
        "--match-official-methodology",
        action="store_true",
        help="Enable ALL official-reproduction flags at once.",
    )
    args = parser.parse_args()

    match = MatchOfficial.from_args(args)

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
            scored = await process_sample(
                client,
                judge_model,
                item,
                match=match,
            )
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
        summary[cat] = {
            "count": len(scores),
            "mean_score": sum(scores) / len(scores),
        }
        all_scores.extend(scores)
    if all_scores:
        summary["overall"] = {
            "count": len(all_scores),
            "mean_score": sum(all_scores) / len(all_scores),
        }

    output = {
        "summary": summary,
        "match_official": {
            "int_bug": match.int_bug,
            "extraction": match.extraction,
            "equivalence": match.equivalence,
        },
        "results": dict(by_category),
    }
    with open(args.target_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== BEAM Evaluation Summary ===")
    for cat, m in sorted(summary.items()):
        print(f"  {cat:30s} {m['mean_score']:.4f}  ({m['count']} samples)")

    await client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
