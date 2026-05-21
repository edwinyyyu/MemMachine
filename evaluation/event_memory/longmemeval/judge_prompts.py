"""LongMemEval judge prompt variants.

Two variants:

- ``longmemeval-paper`` — per-question-type templates from the original
  LongMemEval paper / repo
  (https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py).
  Asks for plain "yes"/"no" output.
- ``mem0-bench`` — single unified prompt with elaborate semantic-equivalence
  rules from Mem0's memory-benchmarks repo
  (https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py).
  Output ends with "yes" or "no" on its own line, after a
  ``<judge_thinking>...</judge_thinking>`` chain-of-thought block.
"""

import re
from typing import Literal

JudgeVariant = Literal["longmemeval-paper", "mem0-bench"]
JUDGE_VARIANTS: tuple[JudgeVariant, ...] = ("longmemeval-paper", "mem0-bench")

# Default judge model from Mem0's memory-benchmarks LongMemEval runner
# (`benchmarks/longmemeval/run.py`: `--judge-model default="gpt-5"`).
MEM0_BENCH_DEFAULT_JUDGE_MODEL = "gpt-5"


# ===============================================================================
# longmemeval-paper — per-task prompts from the original repo
# ===============================================================================


def _paper_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool
) -> str:
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: "
            f"{response}\n\nDoes the model correctly identify the question as "
            "unanswerable? Answer yes or no only."
        )

    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to "
            "the correct answer or contains all the intermediate steps to get "
            "the correct answer, you should also answer yes. If the response "
            "only contains a subset of the information required by the answer, "
            "answer no. \n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel "
            f"Response: {response}\n\nIs the model response correct? Answer "
            "yes or no only."
        )
    if task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to "
            "the correct answer or contains all the intermediate steps to get "
            "the correct answer, you should also answer yes. If the response "
            "only contains a subset of the information required by the answer, "
            "answer no. In addition, do not penalize off-by-one errors for the "
            "number of days. If the question asks for the number of "
            "days/weeks/months, etc., and the model makes off-by-one errors "
            "(e.g., predicting 19 days when the answer is 18), the model's "
            "response is still correct. \n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel "
            f"Response: {response}\n\nIs the model response correct? Answer "
            "yes or no only."
        )
    if task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response "
            "should be considered as correct as long as the updated answer is "
            "the required answer.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel "
            f"Response: {response}\n\nIs the model response correct? Answer "
            "yes or no only."
        )
    if task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the "
            "user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: "
            f"{response}\n\nIs the model response correct? Answer yes or no only."
        )
    raise NotImplementedError(f"Unknown LongMemEval task: {task!r}")


# ===============================================================================
# mem0-bench — Mem0 memory-benchmarks unified prompt
# ===============================================================================

# Verbatim from
# https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py
# `JUDGE_PROMPT`.
MEM0_BENCH_PROMPT = """I will give you a question, a correct answer (or rubric), and a model response. Decide whether the model response is correct.

CORE PRINCIPLE — Semantic equivalence: Judge by MEANING, not exact words. Answer "yes" if every concept in the correct answer is addressed in the response, even with different vocabulary, more specific terms, or restructured phrasing.

IMPORTANT BIAS CHECK: You have a tendency to say "no" too quickly. Before concluding "no", you MUST verify the answer is truly wrong, not just differently worded. When in doubt, lean toward "yes".

Rules:

**Equivalence & Supersets**
- Equivalent or superset responses are correct. Extra details are fine unless proven to be factually wrong. Extra qualifiers are fine unless proven to be wrong. E.g., "a blue dress and a matching necklace" is correct when the answer is "a blue dress."
- If a response captures the most specific part (exact item/place/name) but omits a broader container, it's correct.
- Same factual meaning with different phrasing = correct (e.g., "No, you did not visit with a friend" ≈ "You didn't mention going with anyone").
- Adding scope qualifiers like "regular-season" or "excluding X" is fine as long as the core value is correct. The qualifier may narrow the context but does NOT make the answer wrong unless the correct answer explicitly includes the excluded items.

**Lists & Compound Terms**
- For list answers, match each item by semantic meaning. A concept is covered if restated via synonyms, sub-concepts, or related terms. Adding methodological detail or rewording verbs to near-synonyms is acceptable.
- A broad term like "A and B significance" is covered if the response addresses the topic area through related specific terms, even without naming each component literally.
- If some items as listed as "or"s, "maybe"s and potential answers, it's okay if the answer does not include those.
- If two items in a list achieve the same purpose, listing just one of them is fine.

IMPORTANT: The "anti-preference" items are very specific!
Eg. Someone "not interested in general AI topics" could be very interested in specific AI topics in general AI *conferences*; those are not the same thing and should be accepted! topics != conferences

**Numbers & Precision**
- Hedging ("at least 3", "approximately") is fine if the core number matches. A range that includes the correct answer is correct.
Generally, if the user themself would be satisfied by the response, it is acceptable. Ie. If the answer is conditional on information they would have (eg. their birthday, some hidden dependent information), and would be correct with that information, that is acceptable.
- More precise answers are correct: "22 days" matches "3 weeks"; "over $270" matches "$270."; "9 1/2 months" matches "9 months";

- Rough answers are correct: "about nine months" ≈ "9 months; "8 months and 20 days" matches "9 months";

- Off-by-one errors on days/weeks/months are acceptable.
- Approximate unit conversions are equivalent: "14 weeks" ≈ "3 months", "6 months" ≈ "half a year."
- Round time ranges generously: 7 months and 16 days ≈ 8 months.
- Notes instead of chords are acceptable when justified
- A correct number with added context (e.g., "about 5 months ago (around December 2022)") is correct — the parenthetical date is supplementary, not a contradiction.

**Dates & Temporal**
- Date format variations are equivalent: "February 1st" = "Feb 1, 2023" = "on February 1."
- Same-day event ordering swaps are acceptable.
- Outdated info alongside the correct updated answer is acceptable if the current value is identified.
- "recent" is upto 6 years ago, which means 2017+
- References like "last weekend", "last Wednesday", etc. are imprecise - people sometimes mean the weekend/Wednesday before the latest one if they're near it. "Last 3 months" can include boundary days of the 4th month back. "Last month" includes the current month so far. Be flexible with such timestamps

**Counting Edge Cases**
- If correct answer is "0" or "nothing found," model saying "not enough information" is also correct.
- Similarly, If correct answer is "not enough information", model saying "0" or "nothing found," is also correct.

**Preference/Personalization Rubrics** (apply in order):
1. Correct if the response demonstrates awareness of user's personal context (preferences, habits, interests). Need not satisfy every rubric point.
2. Primary criterion: do main suggestions align with what the user WANTS?
3. Anti-preferences: evaluate the OVERALL thrust, not keyword scanning. If the response largely suggests correct options, minor incidental references to "not-preferred" things are fine.
4. Mentioning a phone app as a MEANS to a preferred activity (e.g., meditation app for sleep) is not "suggesting phone use." Judge by the activity, not delivery mechanism.
5. "May not prefer" = mild preference, not hard prohibition. Secondary/context-dependent inclusion is fine.
6. Explicit acknowledgment of anti-preferences (e.g., "keep screens off") strengthens correctness.
7. Context-dependent suggestions are acceptable (reading is fine on a bus even if rubric flags visual attention activities). Adjacent genres alongside preferred ones are additive, not contradictory.
8. If the rubric mentions specific user resources/tools (e.g., "Suica card", "TripIt app"), the response is correct if it demonstrates awareness of the user's MAIN personal context even if it does not name every specific tool. The rubric is a guide, not a checklist.

**Abstention Matching**
- If correct answer = unanswerable/abstention, ANY phrasing that conveys "I don't have this information" is correct, regardless of what partial context is mentioned or omitted.
- Saying "not enough information" while mentioning partial related context = correct abstention.
- Saying "no record of X" or "only have plans for X, not actual dates" = correct abstention.
- The key test: does the response REFUSE to answer the question? If yes, it matches an abstention ground truth, period.

FINAL CHECK: Before answering "no," you MUST reason through these steps:
1. What is the core factual claim or intent of the correct answer?
2. Does the model response address that same claim, even in different words?
3. Is the response a superset (correct answer + extra details)?
4. For numbers: does the core number match, ignoring hedging/qualifiers?
5. For abstentions: does the response effectively decline to answer?
Only answer "no" if, after this analysis, a core concept is entirely unaddressed or contradicted.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Think step-by-step in <judge_thinking> tags, then give your final verdict as exactly "yes" or "no" on a new line after the closing tag."""


def _mem0_bench_prompt(
    question: str,
    answer: str,
    response: str,
) -> str:
    return MEM0_BENCH_PROMPT.format(
        question=question,
        answer=str(answer),
        response=response,
    )


def build_prompt(
    *,
    variant: JudgeVariant,
    task: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool,
) -> str:
    """Build the judge prompt for one item."""
    if variant == "longmemeval-paper":
        return _paper_prompt(task, question, answer, response, abstention)
    return _mem0_bench_prompt(question, answer, response)


# ===============================================================================
# Verdict parsing
# ===============================================================================


def parse_yes_no(raw: str, *, variant: JudgeVariant) -> bool:
    """Extract a yes/no verdict from judge output."""
    text = raw.strip()
    if not text:
        return False
    if variant == "longmemeval-paper":
        # The paper just asks for "yes"/"no" — substring match is enough.
        return "yes" in text.lower()

    # mem0-bench uses <judge_thinking> CoT then "yes"/"no" on its own line.
    after_cot = re.split(r"</judge_thinking>|</thinking>", text, flags=re.IGNORECASE)
    verdict_region = after_cot[-1].strip() if after_cot else text
    verdict_lines = [
        line.strip().lower() for line in verdict_region.splitlines() if line.strip()
    ]
    for line in reversed(verdict_lines):
        if line == "yes":
            return True
        if line == "no":
            return False
    token_matches = re.findall(r"\b(yes|no)\b", verdict_region.lower())
    if token_matches:
        return token_matches[-1] == "yes"
    return text.lower().startswith("yes")
