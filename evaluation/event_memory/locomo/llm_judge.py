"""LLM judges for LoCoMo answers.

Two prompt variants are exposed:

- ``mem0-classic`` — single-message ``CORRECT``/``WRONG`` rubric byte-for-byte
  identical to the original Mem0 LoCoMo evaluator
  (https://github.com/mem0ai/mem0/blob/main/evaluation/metrics/llm_judge.py).
- ``mem0-bench`` — system + user message variant from Mem0's newer
  ``memory-benchmarks`` repo
  (https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/locomo/prompts.py).
  Includes the LoCoMo category mapping, ``preprocess_answer`` semicolon trim
  for category 3, and JSON output with ``reasoning`` + ``label``.
"""

import json
from typing import Any, Literal

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

JudgeVariant = Literal["mem0-classic", "mem0-bench"]
JUDGE_VARIANTS: tuple[JudgeVariant, ...] = ("mem0-classic", "mem0-bench")

# Category mapping from memory-benchmarks (used by --skip-category-5 etc.).
CATEGORY_NAMES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
CATEGORIES_TO_EVALUATE: list[int] = [1, 2, 3, 4]


# ===============================================================================
# mem0-classic — Mem0's original LoCoMo judge prompt
# ===============================================================================

CLASSIC_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a ’gold’ (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


# ===============================================================================
# mem0-bench — memory-benchmarks repo judge prompt
# ===============================================================================

BENCH_SYSTEM_PROMPT = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."

# Verbatim from
# https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/locomo/prompts.py
# `_JUDGE_TEMPLATE` with `evidence_section`, `evidence_rule`, and
# `evidence_wrong_clause` substituted for the no-evidence path. The double
# braces around `{{question}}`/`{{answer}}`/`{{response}}` in the template
# are pre-resolved here to single braces.
BENCH_PROMPT = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


def preprocess_answer(category: int, answer: str) -> str:
    """Preprocess ground truth answer; for category 3 (open-domain), keep only
    the first part before a semicolon. Matches the Mem0 memory-benchmarks
    behaviour.
    """
    if category == 3 and ";" in answer:
        return answer.split(";", 1)[0].strip()
    return answer


# ===============================================================================
# Async judge entry point
# ===============================================================================


async def evaluate_llm_judge(
    client: AsyncOpenAI,
    question: str,
    gold_answer: str,
    generated_answer: str,
    *,
    model: str = "gpt-4o-mini",
    variant: JudgeVariant = "mem0-classic",
    category: int | None = None,
) -> int:
    """Score one answer against the gold answer; returns 1 (CORRECT) or 0 (WRONG).

    Args:
        client: Async OpenAI client.
        question: The user question.
        gold_answer: The ground truth answer.
        generated_answer: The model's generated answer.
        model: OpenAI model name.
        variant: Which judge prompt to use.
        category: LoCoMo category id (1-5). Only used by ``mem0-bench`` for
            answer preprocessing on category 3.
    """
    messages: list[ChatCompletionMessageParam]
    if variant == "mem0-classic":
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content=CLASSIC_PROMPT.format(
                    question=question,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer,
                ),
            )
        ]
    else:
        processed_gold = (
            preprocess_answer(category, gold_answer)
            if category is not None
            else gold_answer
        )
        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=BENCH_SYSTEM_PROMPT
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=BENCH_PROMPT.format(
                    question=question,
                    answer=processed_gold,
                    response=generated_answer,
                ),
            ),
        ]

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    # Mem0's memory-benchmarks omits `temperature` for gpt-5 / o-series since
    # those models only accept the default; everything else uses temperature=0.
    if not model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
        create_kwargs["temperature"] = 0

    response = await client.chat.completions.create(**create_kwargs)
    content = response.choices[0].message.content or "{}"
    raw = json.loads(content)
    label = str(raw.get("label", "")).upper()
    return 1 if label == "CORRECT" else 0
