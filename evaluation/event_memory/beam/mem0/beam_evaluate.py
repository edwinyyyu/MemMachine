"""mem0-variant BEAM evaluation, adapted to read EventMemory search output.

Implements the rubric-judging and event-ordering algorithms from
mem0ai/memory-benchmarks, whose prompts live at
https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/beam/prompts.py
and whose evaluation pipeline lives in
https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/beam/run.py.

Differences from the official BEAM methodology (and from this repo's
`official/beam_evaluate.py`) — all driven by mem0's deliberate choices:

- **Rubric judge prompt**: mem0's `JUDGE_PROMPT` / `BEAM_JUDGE_SYSTEM_PROMPT`,
  a rewrite of BEAM's `unified_llm_judge_base_prompt`. Same inputs (question,
  rubric, response), same 0/0.5/1 JSON output, same per-item independent
  evaluation — only the prompt wording differs. The raw LLM score is passed
  through mem0's `_clamp_nugget_score` (>=0.75 → 1.0, >=0.25 → 0.5, else 0.0)
  to tolerate off-scale hallucinations.

- **Event-ordering extraction**: mem0 replaces BEAM's regex-based
  numbered/bulleted parsing with an LLM call
  (`get_beam_fact_extraction_prompt`) that asks for a JSON array of events.
  Because the mem0 pipeline forces `response_format={"type": "json_object"}`,
  the model returns either `{"events"|"facts"|"result": [...]}` or a bare
  list — both are accepted.

- **Event-ordering alignment**: mem0 replaces BEAM's pairwise greedy
  YES/NO equivalence with an LLM call
  (`get_beam_event_alignment_prompt`) that returns the best-matching
  reference index (or -1). Each extracted event gets one alignment call;
  there is no "used set" restricting reuse.

- **Kendall tau-b**: mem0 uses its own implementation (from
  `benchmarks/common/metrics.py`), not scipy's `variant="b"`. The tie
  accounting differs slightly from scipy. `primary_score` is still
  `(tau_b + 1) / 2` (a.k.a. `tau_norm`). Precision/recall/F1 are **not**
  reported — mem0's event-ordering result is just
  `{tau_b, predicted_order, reference_order}`.

Non-differences: sample flattening, per-category mean aggregation, CLI
surface (search-path / target-path / judge-provider / judge-model /
concurrency), and JSON output layout all mirror `official/beam_evaluate.py`
so the two scripts are drop-in interchangeable on the same search output.
"""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

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
# https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/beam/prompts.py
BEAM_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing whether an AI assistant's response satisfies "
    "specific rubric criteria. You must be objective, fair, and consistent. "
    "Return ONLY valid JSON with the exact format requested."
)


def _build_judge_user_prompt(question: str, nugget: str, llm_response: str) -> str:
    """Body of `get_beam_nugget_judge_prompt` from mem0's prompts.py."""
    return f"""Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
{question}

LLM RESPONSE:
{llm_response}

RUBRIC CRITERION:
{nugget}

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

**For POSITIVE requirements** (response should contain, mention, or demonstrate something):
- **1.0 (Complete Compliance)**: The required element is present, accurate, and complete. The response fully and clearly satisfies the rubric criterion.
- **0.5 (Partial Compliance)**: The required element is partially present, has minor inaccuracies, or is incomplete. The core intent is present but not fully realized.
- **0.0 (No Compliance)**: The required element is missing, incorrect, or the response is entirely off-topic / non-responsive.

**For NEGATIVE constraints** (response should NOT contain or should avoid something):
- **1.0 (Complete Compliance)**: The response is responsive to the question AND the prohibited element is absent.
- **0.5 (Partial Compliance)**: The response is responsive but contains a borderline or ambiguous reference to the prohibited element.
- **0.0 (No Compliance)**: The prohibited element is present in the response, OR the response is non-responsive (off-topic, refusal, empty).

**Compound statement handling**: If the rubric criterion contains "and" or commas connecting multiple required elements:
- All elements present and correct = 1.0
- Some (but not all) elements present and correct = 0.5
- No elements present or correct = 0.0

EVALUATION RULES:
1. **Semantic tolerance**: Paraphrases and synonyms are acceptable. The response does not need to use the exact same words as the rubric.
2. **Numeric and date equivalence**: Treat equivalent representations as identical. "$68,000" = "68k" = "sixty-eight thousand dollars". "2 years" = "24 months". Prefer normalized comparison for numbers, currencies, dates, and durations.
3. **Case / punctuation / whitespace tolerance**: Differences in capitalization, punctuation, and whitespace must be ignored when comparing content.
4. **Hedging tolerance**: Do not penalize hedging language ("I think", "probably", "it seems"), passive voice, or verbosity if the substantive content satisfies the rubric criterion.
5. **Style neutrality**: Do not penalize for tone, formatting, or length unless the rubric criterion specifically requires a particular format.
6. **Responsiveness**: If the LLM response is completely off-topic or refuses to answer, score 0.0 for all criteria.
7. **Independence**: Evaluate this criterion in isolation — do not consider other rubric items.
8. **Specificity matters**: Vague or generic answers that could apply to any question score lower than specific, detailed answers.

STEP-BY-STEP EVALUATION:
Follow these steps in order:
1. **Understand the Requirement**: Read the rubric criterion and classify it as a positive requirement or a negative constraint.
2. **Parse Compound Statements**: If the criterion contains multiple sub-requirements joined by "and" or commas, identify each element separately.
3. **Check Compliance**: Compare the LLM response against each element, applying the tolerance rules above (semantic, numeric, case, hedging).
4. **Assign Score**: Use the appropriate scoring table (positive or negative) and compound-statement rule to determine the score.
5. **Provide Reasoning**: Write a concise explanation referencing which elements were or were not satisfied.

Return your evaluation as a JSON object with exactly two fields:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence explaining your score>"}}"""


def _build_fact_extraction_prompt(response: str) -> str:
    """Body of `get_beam_fact_extraction_prompt` from mem0's prompts.py."""
    return f"""Extract all distinct events or facts mentioned in the following response,
in the exact order they are presented. Return ONLY a JSON array of short event descriptions.

RESPONSE:
{response}

Return format: ["event 1 description", "event 2 description", ...]"""


def _build_event_alignment_prompt(extracted_event: str, rubric_events: list[str]) -> str:
    """Body of `get_beam_event_alignment_prompt` from mem0's prompts.py."""
    events_list = "\n".join(f"{i}. {e}" for i, e in enumerate(rubric_events))
    return f"""Given the following extracted event from an LLM response, determine which
reference event it best corresponds to. Return ONLY a JSON object.

EXTRACTED EVENT:
{extracted_event}

REFERENCE EVENTS:
{events_list}

If the extracted event matches one of the reference events (even approximately or paraphrased),
return the 0-based index. If it doesn't match any, return -1.

Return format: {{"index": <integer>, "reason": "<brief explanation>"}}"""


_JSON_OBJECT_FORMAT = {"type": "json_object"}

# mem0's generate_structured defaults, verbatim:
# https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/common/llm_client.py
_MAX_TOKENS = 4096
_MAX_RETRIES = 3


def _strip_code_fence(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def _parse_json_loose(text: str) -> object:
    """Best-effort JSON parse tolerant of code fences and prose wrappers."""
    cleaned = _strip_code_fence(text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No valid JSON in response: {text!r}")


async def _call_json(
    client: ChatClient,
    model: str,
    system: str,
    user: str,
) -> object:
    """Mirror of mem0's `_generate_structured_openai`:

    - `response_format={"type": "json_object"}`
    - `max_completion_tokens`/`max_tokens=4096` (routed by model family in
      `llm_provider._openai_max_tokens_kwargs`)
    - 3-attempt retry on JSON parse / timeout / empty-response / other
      exceptions, with `2 * (attempt + 1)`-second backoff
    - "final"-wrapper unwrap for models that wrap output in `{"final": ...}`
    - Returns `{}` on total failure (matches mem0's `response_format=None` path)
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    for attempt in range(_MAX_RETRIES):
        try:
            result = await client.create(
                model=model,
                messages=messages,
                response_format=_JSON_OBJECT_FORMAT,
                max_tokens=_MAX_TOKENS,
            )
            raw = result.get("content", "") or ""
            if not raw:
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                return {}
            parsed = _parse_json_loose(raw)
            if isinstance(parsed, dict) and len(parsed) == 1 and "final" in parsed:
                inner = parsed["final"]
                if isinstance(inner, str):
                    parsed = json.loads(inner)
                elif isinstance(inner, (dict, list)):
                    parsed = inner
            return parsed
        except (json.JSONDecodeError, ValueError, asyncio.TimeoutError):
            pass
        except Exception:
            pass
        if attempt < _MAX_RETRIES - 1:
            await asyncio.sleep(2 * (attempt + 1))
    return {}


def _clamp_nugget_score(raw: float) -> float:
    """Verbatim from mem0's run.py `_clamp_nugget_score`."""
    if raw >= 0.75:
        return 1.0
    if raw >= 0.25:
        return 0.5
    return 0.0


def _compute_kendall_tau_b(
    predicted_order: list[int], reference_order: list[int]
) -> float:
    """Verbatim from
    https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/common/metrics.py
    `compute_kendall_tau_b`. Kept as-is (rather than delegating to scipy) to
    reproduce mem0's exact tie accounting.
    """
    if len(predicted_order) < 2 or len(reference_order) < 2:
        return 0.0

    pred_rank = {v: i for i, v in enumerate(predicted_order)}
    ref_rank = {v: i for i, v in enumerate(reference_order)}

    common = set(predicted_order) & set(reference_order)
    items = sorted(common)
    if len(items) < 2:
        return 0.0

    concordant = 0
    discordant = 0
    tied_pred = 0
    tied_ref = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            pred_diff = pred_rank[a] - pred_rank[b]
            ref_diff = ref_rank[a] - ref_rank[b]
            if pred_diff == 0 and ref_diff == 0:
                tied_pred += 1
                tied_ref += 1
            elif pred_diff == 0:
                tied_pred += 1
            elif ref_diff == 0:
                tied_ref += 1
            elif (pred_diff > 0 and ref_diff > 0) or (pred_diff < 0 and ref_diff < 0):
                concordant += 1
            else:
                discordant += 1

    n1 = concordant + discordant + tied_pred
    n2 = concordant + discordant + tied_ref
    if n1 == 0 or n2 == 0:
        return 0.0
    return (concordant - discordant) / ((n1 * n2) ** 0.5)


async def score_rubric_item(
    client: ChatClient,
    model: str,
    question: str,
    rubric_item: str,
    llm_response: str,
) -> tuple[float, str]:
    """Mirror of mem0's `judge_single_nugget` including its non-dict fallback."""
    user_prompt = _build_judge_user_prompt(question, rubric_item, llm_response)
    parsed = await _call_json(client, model, BEAM_JUDGE_SYSTEM_PROMPT, user_prompt)
    if isinstance(parsed, dict):
        try:
            score = _clamp_nugget_score(float(parsed.get("score", 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        return score, str(parsed.get("reason", ""))

    raw_str = str(parsed)
    if "1.0" in raw_str:
        return 1.0, raw_str[:200]
    if "0.5" in raw_str:
        return 0.5, raw_str[:200]
    return 0.0, f"Parse error: {raw_str[:200]}"


async def _extract_events(
    client: ChatClient, model: str, llm_response: str
) -> list[str]:
    user_prompt = _build_fact_extraction_prompt(llm_response)
    try:
        parsed = await _call_json(
            client,
            model,
            "Extract events as a JSON array of strings.",
            user_prompt,
        )
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, dict):
        for key in ("events", "facts", "result"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [str(x) for x in value]
    return []


async def _align_event(
    client: ChatClient,
    model: str,
    event: str,
    rubric_events: list[str],
) -> int:
    user_prompt = _build_event_alignment_prompt(event, rubric_events)
    try:
        parsed = await _call_json(
            client,
            model,
            "Align the event to a reference event index. Return JSON.",
            user_prompt,
        )
    except Exception:
        return -1
    if not isinstance(parsed, dict):
        return -1
    try:
        return int(parsed.get("index", -1))
    except (TypeError, ValueError):
        return -1


async def score_event_ordering(
    client: ChatClient,
    model: str,
    rubric_nuggets: list[str],
    llm_response: str,
) -> dict:
    """Mirror of mem0's `compute_event_ordering_score` in run.py."""
    extracted = await _extract_events(client, model, llm_response)
    if not extracted or not rubric_nuggets:
        return {"tau_b": 0.0, "predicted_order": [], "reference_order": []}

    predicted_indices: list[int] = []
    for event in extracted:
        idx = await _align_event(client, model, event, rubric_nuggets)
        if 0 <= idx < len(rubric_nuggets):
            predicted_indices.append(idx)

    reference_order = list(range(len(rubric_nuggets)))
    tau_b = _compute_kendall_tau_b(predicted_indices, reference_order)
    return {
        "tau_b": round(tau_b, 4),
        "predicted_order": predicted_indices,
        "reference_order": reference_order,
    }


async def process_sample(client: ChatClient, model: str, item: dict) -> dict:
    question = str(item.get("question", ""))
    llm_response = str(item.get("model_answer", item.get("response", "")))
    rubric = list(item.get("rubric", []))
    category = str(item.get("category", ""))

    result: dict = dict(item)
    result["judge_model_responses"] = []

    rubric_scores: list[float] = []
    for rubric_item in rubric:
        score, reason = await score_rubric_item(
            client, model, question, rubric_item, llm_response
        )
        rubric_scores.append(score)
        result["judge_model_responses"].append(
            {"rubric_item": rubric_item, "score": score, "reason": reason}
        )

    rubric_avg = sum(rubric_scores) / len(rubric_scores) if rubric_scores else 0.0
    result["llm_judge_score"] = rubric_avg

    if category == QuestionCategory.EVENT_ORDERING.value:
        ordering = await score_event_ordering(client, model, rubric, llm_response)
        result.update(ordering)
        # mem0's primary metric for event_ordering is tau normalized to [0,1]
        # (`tau_normalized = (tau_result["tau_b"] + 1.0) / 2.0` in run.py).
        result["tau_norm"] = (ordering["tau_b"] + 1.0) / 2.0
        result["primary_score"] = result["tau_norm"]
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
        help="Judge LLM provider (default: openai).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge LLM model (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Max concurrent samples"
    )
    args = parser.parse_args()

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
            scored = await process_sample(client, args.judge_model, item)
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
        "variant": "mem0",
        "results": dict(by_category),
    }
    with open(args.target_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== BEAM Evaluation Summary (mem0 variant) ===")
    for cat, m in sorted(summary.items()):
        print(f"  {cat:30s} {m['mean_score']:.4f}  ({m['count']} samples)")

    await client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
