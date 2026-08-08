"""LLM judges for hard_bench scoring (gpt-5-mini)."""

from __future__ import annotations

from .system import UnifiedSystem, _extract_json

# -----------------------------------------------------------------------------
# Judge: did the agent's transcript address the gold subdecision?
# -----------------------------------------------------------------------------

SUBDECISION_JUDGE_PROMPT = """The user gave an agent a task. The agent produced the transcript below. We want to know if the agent ADDRESSED a specific gold subdecision (a thing the agent should have surfaced/handled given memory context).

Task: {task_prompt}

Gold subdecision the agent should address:
  {subdecision}

Representative gold memory item that informs this subdecision:
  {gold_text}

Agent's full transcript:
{transcript}

Judge: did the agent address this subdecision in its plan or per-step outputs?

Return STRICT JSON:
{{
  "addressed": true|false,
  "step_label": "<the step number/label that addressed it, or null>",
  "evidence_quote": "<short quote from transcript showing the address, or empty>"
}}

Be strict — generic gestures don't count; the agent must actually engage with the specific subdecision."""


async def judge_subdecision(
    system: UnifiedSystem,
    task_prompt: str,
    subdecision_description: str,
    gold_text: str,
    transcript: str,
) -> dict:
    prompt = SUBDECISION_JUDGE_PROMPT.format(
        task_prompt=task_prompt,
        subdecision=subdecision_description,
        gold_text=gold_text,
        transcript=transcript,
    )
    raw = await system.llm(prompt, reasoning_effort="low")
    obj = _extract_json(raw) or {}
    return {
        "addressed": bool(obj.get("addressed", False)),
        "step_label": obj.get("step_label"),
        "evidence_quote": obj.get("evidence_quote", ""),
    }


# -----------------------------------------------------------------------------
# Judge: did the agent surface and obey a guideline?
# -----------------------------------------------------------------------------

GUIDELINE_JUDGE_PROMPT = """The user asked the agent to do a task. Memory contained a TEAM-SPECIFIC GUIDELINE that the proposed action would violate. The agent should have surfaced the guideline AND proposed an alternative.

Task: {task_prompt}

Guideline (from memory): {guideline_text}

Why the task violates this guideline: {violation_explanation}

Recommended alternative respecting the guideline: {recommended_alt}

Agent's full transcript:
{transcript}

Judge:
1. SURFACED: did the agent reference the guideline (or a substantively-equivalent rule) anywhere in its transcript? (yes/no)
2. RECOMMENDED_ALTERNATIVE: did the agent propose the recommended alternative (or a substantively-equivalent fix)? (yes/no)
3. WARNED: did the agent explicitly flag a concern about the user's planned approach? (yes/no)

Return STRICT JSON:
{{
  "surfaced": true|false,
  "recommended_alternative": true|false,
  "warned": true|false,
  "rationale": "<one sentence>"
}}

Be strict — generic best-practice mentions don't count; agent must engage with the specific team-rule from memory."""


async def judge_guideline(
    system: UnifiedSystem,
    task_prompt: str,
    guideline_text: str,
    violation_explanation: str,
    recommended_alt: str,
    transcript: str,
) -> dict:
    prompt = GUIDELINE_JUDGE_PROMPT.format(
        task_prompt=task_prompt,
        guideline_text=guideline_text,
        violation_explanation=violation_explanation,
        recommended_alt=recommended_alt,
        transcript=transcript,
    )
    raw = await system.llm(prompt, reasoning_effort="medium")
    obj = _extract_json(raw) or {}
    return {
        "surfaced": bool(obj.get("surfaced", False)),
        "recommended_alternative": bool(obj.get("recommended_alternative", False)),
        "warned": bool(obj.get("warned", False)),
        "rationale": obj.get("rationale", ""),
    }


# -----------------------------------------------------------------------------
# Plant retrieval check (no LLM): for an addressed subdecision, was the
# gold plant retrieved within Phase1 + Phase3 hits?
# -----------------------------------------------------------------------------


def plant_retrieved(gold_plant_ids: list[str], hit_properties: list[dict]) -> bool:
    """True if any gold plant_id appears in any hit's properties."""
    gold_set = set(gold_plant_ids)
    for props in hit_properties:
        pid = props.get("plant_id")
        if pid and pid in gold_set:
            return True
    return False


# -----------------------------------------------------------------------------
# Judge: QA-style (gold_answer match)
# -----------------------------------------------------------------------------

QA_JUDGE_PROMPT = """The user asked a question. We have the agent's answer and the gold answer. Decide whether the agent's answer is substantively correct.

Question: {question}

Gold answer: {gold_answer}

Agent's full transcript:
{transcript}

Judge:
1. CORRECT: does the agent's answer convey the same factual content as the gold answer? (yes/no — minor paraphrasing OK; missing key facts or contradicting facts NOT OK)
2. EVIDENCE_CITED: does the agent reference any specific memory item ([date, time] format or quote) supporting the answer? (yes/no)

Return STRICT JSON: {{"correct": true|false, "evidence_cited": true|false, "rationale": "<one sentence>"}}"""


async def judge_qa(
    system,  # UnifiedSystem
    question: str,
    gold_answer: str,
    transcript: str,
) -> dict:
    prompt = QA_JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        transcript=transcript,
    )
    raw = await system.llm(prompt, reasoning_effort="low")
    obj = _extract_json(raw) or {}
    return {
        "correct": bool(obj.get("correct", False)),
        "evidence_cited": bool(obj.get("evidence_cited", False)),
        "rationale": obj.get("rationale", ""),
    }


# -----------------------------------------------------------------------------
# Judge: Temporal-anchored QA (gold answer + out-of-window check)
# -----------------------------------------------------------------------------

TEMPORAL_JUDGE_PROMPT = """The user asked a temporally-anchored question. We have the agent's answer, the gold answer, and the anchor resolution.

Question: {question}

Anchor resolution (what time window the question refers to): {anchor_resolution}

Gold answer: {gold_answer}

Agent's full transcript:
{transcript}

Judge:
1. CORRECT: does the agent's answer convey the same factual content as the gold? (yes/no)
2. RESPECTED_ANCHOR: does the agent's answer engage ONLY with events from the anchor's time window, not from other periods? (yes/no — if the agent's answer mixes content from out-of-window events, mark NO)
3. EVIDENCE_CITED: does the agent cite specific memory items? (yes/no)

Return STRICT JSON: {{"correct": true|false, "respected_anchor": true|false, "evidence_cited": true|false, "rationale": "<one sentence>"}}"""


async def judge_temporal(
    system,
    question: str,
    anchor_resolution: str,
    gold_answer: str,
    transcript: str,
) -> dict:
    prompt = TEMPORAL_JUDGE_PROMPT.format(
        question=question,
        anchor_resolution=anchor_resolution,
        gold_answer=gold_answer,
        transcript=transcript,
    )
    raw = await system.llm(prompt, reasoning_effort="low")
    obj = _extract_json(raw) or {}
    return {
        "correct": bool(obj.get("correct", False)),
        "respected_anchor": bool(obj.get("respected_anchor", False)),
        "evidence_cited": bool(obj.get("evidence_cited", False)),
        "rationale": obj.get("rationale", ""),
    }
