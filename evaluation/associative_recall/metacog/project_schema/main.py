"""Project-schema operator: bounded WM (<=10k tokens) + retrieval-on-demand.

ARCHITECTURE
------------
- Bounded working memory (WM) per step. WM hard ceiling: 10,000 tokens.
- External memory (EM): list[MemoryChunk] per scenario, totaling 30-100k
  tokens. The full EM is NEVER passed in any prompt.
- Retrieval-on-demand: at each step the agent issues a probe; we score
  chunks by token-overlap with the probe + tag overlap and return top-K
  snippets.
- Compaction/eviction between steps: WM is trimmed/compacted before the
  next retrieval.

OPERATOR UNDER TEST: PROJECT SCHEMA ACCUMULATION
A persistent, slot-filled mental model of (USER / TASK / CONTEXT) that
gets updated AFTER each retrieval. Slots are GENERAL — not domain-specific.
Each retrieval extract-and-merges new information; raw retrieved snippets
are SHED after their contribution is merged. WM = (schema + minimal recent
buffer).

BASELINE: same retrieval, same K, but WM holds raw snippets and FIFO-evicts
when over budget. Final answer uses raw WM.

Usage:
    uv run python evaluation/associative_recall/metacog/project_schema/main.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from scenarios import SCENARIOS, MemoryChunk, TestCase  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL = "gpt-5-mini"
WM_BUDGET_TOKENS = 10_000  # hard cap (POC). 50k is the production ceiling.
SCHEMA_TARGET_TOKENS = 1_000  # schema target size (operator only)
RECENT_BUFFER_BUDGET = 1_500  # operator: tokens of "minimal recent buffer"
RETRIEVAL_K = 4  # top-K chunks per probe
MAX_STEPS_BEYOND_PLAN = 0  # we just iterate scenario.plan_steps once

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"
TRACES_DIR = OUT_DIR / "token_traces"
TRACES_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Token approximation
# ----------------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ----------------------------------------------------------------------------
# External-memory retrieval (lightweight lexical index)
# ----------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")


def _toks(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def retrieve(memory: list[MemoryChunk], probe: str, k: int) -> list[MemoryChunk]:
    """Score by lexical overlap (probe tokens vs chunk text + tags).

    Tags weighted 3x. Tie-break by chunk_id for determinism.
    """
    probe_tokens = _toks(probe)
    if not probe_tokens:
        return []
    scored: list[tuple[float, str, MemoryChunk]] = []
    for chunk in memory:
        body_tokens = _toks(chunk.title + " " + chunk.text)
        tag_tokens = {t.lower() for t in chunk.tags}
        score = len(probe_tokens & body_tokens) + 3 * len(probe_tokens & tag_tokens)
        if score > 0:
            scored.append((score, chunk.chunk_id, chunk))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:k]]


def render_snippet(chunk: MemoryChunk) -> str:
    return f"[{chunk.chunk_id}] {chunk.title}\n{chunk.text}"


# ----------------------------------------------------------------------------
# OpenAI helpers
# ----------------------------------------------------------------------------

CLIENT: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global CLIENT
    if CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; check evaluation/associative_recall/.env"
            )
        CLIENT = AsyncOpenAI(api_key=api_key)
    return CLIENT


async def llm(system: str, user: str, *, reasoning: str = "low") -> str:
    resp = await client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        reasoning_effort=reasoning,
    )
    return resp.choices[0].message.content or ""


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_obj(text: str) -> dict:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"_raw": cleaned}


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

PROBE_SYSTEM = """You are an agent executing a multi-step task with a SMALL working memory and a LARGE external memory you can probe.

You are given:
- the task brief
- the CURRENT STEP (what you must address right now)
- your current working-memory contents

Output ONE retrieval probe — a short query (3-15 words) that targets the
specific information you need to address the current step. Output ONLY the
probe text. No JSON, no prose around it."""


PROBE_USER_TEMPLATE = """TASK BRIEF:
{task_brief}

CURRENT STEP ({step_idx}/{step_total}): {step}

WORKING MEMORY (current contents):
{wm}

Probe (one short query):"""


SCHEMA_INIT = json.dumps(
    {
        "USER": {"preferences": [], "constraints": [], "expertise": [], "people": []},
        "TASK": {
            "goals": [],
            "deadlines": [],
            "stakeholders": [],
            "open_questions": [],
        },
        "CONTEXT": {
            "location": [],
            "budget": [],
            "knowns": [],
            "unknowns": [],
            "tools": [],
            "outputs_so_far": [],
        },
        "CONFLICTS": [],
    },
    indent=2,
)


SCHEMA_UPDATE_SYSTEM = f"""You maintain a PROJECT SCHEMA — a persistent, slot-filled mental model of the user, task, and context. It IS your bounded working-memory artifact (target: under {SCHEMA_TARGET_TOKENS} tokens; hard ceiling on total WM: {WM_BUDGET_TOKENS} tokens).

WHY a schema (vs raw retrieved snippets):
- COMPACTION: the external memory dwarfs your working memory. Raw retrievals will exceed budget after a few steps. The schema is the compressed handle that survives.
- PERSISTENT INFERRED MODEL: inferences ("ER-level shellfish allergy → never any shellfish anywhere", "HIPAA in-scope → encryption at rest+transit") are derived ONCE and recorded, not re-derived each retrieval.
- VARIANCE REDUCTION: re-deriving inferences from raw chunks each step yields inconsistent answers. A persistent schema fixes the inferred state.

SLOT TYPES (general, not domain-specific):
- USER: who the user is — preferences, hard constraints, expertise level, important people.
- TASK: what they're trying to do — concrete goals, deadlines, stakeholders, open questions.
- CONTEXT: situational facts — location, budget, knowns, unknowns, tools, outputs already produced.
- CONFLICTS: detected contradictions. Format: "<earlier> vs <later> — resolution: prefer most recent (default)".

UPDATE RULES (extract-and-merge):
1. Read the PRIOR SCHEMA and the NEW RETRIEVED SNIPPETS.
2. EXTRACT facts — both verbatim and inferred. Carry inferred content, not just verbatim quotes.
3. MERGE into the appropriate slot. Deduplicate. Keep entries terse (one short phrase each).
4. CONFLICT DETECTION: if a snippet contradicts an existing slot value, log it in CONFLICTS and replace the slot value with the more recent one (prefer most recent on conflict). Do NOT silently overwrite.
5. SIZE: keep total schema under ~{SCHEMA_TARGET_TOKENS} tokens. Drop low-signal phatic content.

OUTPUT: a single JSON object with the same top-level keys as the prior schema. No prose, no code fences. Just JSON."""


SCHEMA_UPDATE_USER_TEMPLATE = """PRIOR SCHEMA:
{schema}

NEW RETRIEVED SNIPPETS (for current step "{step}"):
{snippets}

Updated schema (JSON only):"""


# Final-answer prompts — one for each variant
BASELINE_FINAL_SYSTEM = f"""You are the user's assistant executing a multi-step task. Your bounded working memory holds raw retrieved snippets gathered across the plan steps.

Your final answer MUST honor every load-bearing constraint, preference, and goal that appeared in the snippets — including ones that surfaced many steps ago.

(WM hard ceiling: {WM_BUDGET_TOKENS} tokens. The full external memory is far larger and is NOT in your context.)

Be concrete and direct."""


OPERATOR_FINAL_SYSTEM = f"""You are the user's assistant executing a multi-step task. Your bounded working memory has been compacted into a PROJECT SCHEMA (USER / TASK / CONTEXT slots) that was updated after every retrieval. Raw retrievals were shed after their contribution merged into the schema.

Treat the schema as authoritative. Treat CONFLICTS entries as "use the most recent value". Your final answer MUST honor every constraint, preference, and goal recorded in the schema.

(WM hard ceiling: {WM_BUDGET_TOKENS} tokens; schema target: under {SCHEMA_TARGET_TOKENS} tokens.)

Be concrete and direct."""


JUDGE_SYSTEM = """You are a strict rubric judge. You receive:
- the final user request,
- the assistant's final answer,
- a list of rubric items (load-bearing constraints), each with a description and signal phrases.

For each rubric item, decide:
- HONORED: the answer clearly respects the constraint.
- VIOLATED: the answer ignores or contradicts the constraint, OR uses something the rubric forbids without an explicit avoidance phrase.
- UNCLEAR: not enough information in the answer to tell.

Output JSON ONLY in this exact form:
{
  "items": [
    {"id": "<rubric_id>", "verdict": "HONORED"|"VIOLATED"|"UNCLEAR", "reason": "<one short sentence>"}
  ]
}"""


# ----------------------------------------------------------------------------
# Working-memory containers
# ----------------------------------------------------------------------------


@dataclass
class WMState:
    """Shared accounting for both variants."""

    variant: str
    contents: str = ""  # operator: schema JSON; baseline: concatenated snippets
    recent_buffer: str = ""  # operator: most recent snippet block (small)
    tokens: int = 0
    history: list[dict] = None  # token trace per step

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def total_tokens(self) -> int:
        return approx_tokens(self.contents) + approx_tokens(self.recent_buffer)


def baseline_evict(snippets: list[str], budget: int) -> list[str]:
    """FIFO evict oldest snippets until total tokens under budget."""
    while snippets and sum(approx_tokens(s) for s in snippets) > budget:
        snippets.pop(0)
    return snippets


# ----------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------


async def run_baseline(case: TestCase) -> dict:
    """WM holds raw retrieved snippets; FIFO drop when over budget.

    No schema. Final answer reads the raw WM.
    """
    snippets: list[str] = []
    trace: list[dict] = []

    for i, step in enumerate(case.plan_steps):
        wm_render = "\n\n".join(snippets) if snippets else "(empty)"
        wm_render_clipped = wm_render[: WM_BUDGET_TOKENS * 4]  # sanity clip in chars
        probe = await llm(
            PROBE_SYSTEM,
            PROBE_USER_TEMPLATE.format(
                task_brief=case.task_brief,
                step_idx=i + 1,
                step_total=len(case.plan_steps),
                step=step,
                wm=wm_render_clipped,
            ),
            reasoning="low",
        )
        probe = probe.strip().splitlines()[0][:300] if probe.strip() else step
        retrieved = retrieve(case.memory, probe, RETRIEVAL_K)
        new_block = "\n\n".join(render_snippet(c) for c in retrieved)
        snippets.append(f"[step {i + 1} retrieval — probe: {probe!r}]\n{new_block}")
        # Enforce 10k WM ceiling
        snippets = baseline_evict(snippets, WM_BUDGET_TOKENS)

        wm_tokens_after = approx_tokens("\n\n".join(snippets))
        trace.append(
            {
                "step": i + 1,
                "step_text": step,
                "probe": probe,
                "n_retrieved": len(retrieved),
                "retrieved_ids": [c.chunk_id for c in retrieved],
                "wm_tokens_after": wm_tokens_after,
                "evicted": wm_tokens_after
                < approx_tokens("\n\n".join(snippets) + new_block),
            }
        )

    final_wm = "\n\n".join(snippets) if snippets else "(empty)"
    final_user = (
        f"WORKING MEMORY (raw retrieved snippets, FIFO bounded):\n\n{final_wm}\n\n"
        f"---\n\nTASK BRIEF:\n{case.task_brief}\n\n"
        f"---\n\nFINAL REQUEST:\n{case.final_question}"
    )
    answer = await llm(BASELINE_FINAL_SYSTEM, final_user, reasoning="low")
    return {
        "answer": answer,
        "trace": trace,
        "final_wm_tokens": approx_tokens(final_wm),
        "final_prompt_tokens": approx_tokens(final_user),
    }


async def run_operator(case: TestCase) -> dict:
    """WM holds (schema + minimal recent buffer). Schema updated after each retrieval."""
    schema_text = SCHEMA_INIT
    recent_buffer = ""
    trace: list[dict] = []

    for i, step in enumerate(case.plan_steps):
        # The agent sees the schema (compact WM) when generating the probe.
        wm_render = (
            f"PROJECT SCHEMA:\n{schema_text}\n\n"
            f"RECENT BUFFER (most recent snippet, dropped after merge):\n"
            f"{recent_buffer or '(empty)'}"
        )
        wm_render_clipped = wm_render[: WM_BUDGET_TOKENS * 4]

        probe = await llm(
            PROBE_SYSTEM,
            PROBE_USER_TEMPLATE.format(
                task_brief=case.task_brief,
                step_idx=i + 1,
                step_total=len(case.plan_steps),
                step=step,
                wm=wm_render_clipped,
            ),
            reasoning="low",
        )
        probe = probe.strip().splitlines()[0][:300] if probe.strip() else step

        retrieved = retrieve(case.memory, probe, RETRIEVAL_K)
        snippets_block = "\n\n".join(render_snippet(c) for c in retrieved)

        # Update schema by extracting-and-merging the new snippets.
        new_schema_raw = await llm(
            SCHEMA_UPDATE_SYSTEM,
            SCHEMA_UPDATE_USER_TEMPLATE.format(
                schema=schema_text,
                step=step,
                snippets=snippets_block or "(no relevant snippets returned)",
            ),
            reasoning="low",
        )
        new_schema_obj = parse_json_obj(new_schema_raw)
        if "_raw" in new_schema_obj and len(new_schema_obj) == 1:
            # Parsing failed; keep prior schema.
            schema_text_next = schema_text
        else:
            schema_text_next = json.dumps(new_schema_obj, indent=2)

        # SHED raw retrievals after merge: keep only a tiny recent buffer.
        recent_buffer_candidate = snippets_block
        if approx_tokens(recent_buffer_candidate) > RECENT_BUFFER_BUDGET:
            # Truncate to budget.
            cap_chars = RECENT_BUFFER_BUDGET * 4
            recent_buffer_candidate = (
                recent_buffer_candidate[:cap_chars] + " ...[truncated]"
            )

        # Enforce overall WM ceiling: schema + buffer must fit in WM_BUDGET.
        schema_tokens = approx_tokens(schema_text_next)
        budget_for_buffer = max(0, WM_BUDGET_TOKENS - schema_tokens)
        if approx_tokens(recent_buffer_candidate) > budget_for_buffer:
            cap_chars = budget_for_buffer * 4
            recent_buffer_candidate = (
                recent_buffer_candidate[:cap_chars] + " ...[trimmed_for_wm_budget]"
            )

        schema_text = schema_text_next
        recent_buffer = recent_buffer_candidate

        wm_total = approx_tokens(schema_text) + approx_tokens(recent_buffer)
        trace.append(
            {
                "step": i + 1,
                "step_text": step,
                "probe": probe,
                "n_retrieved": len(retrieved),
                "retrieved_ids": [c.chunk_id for c in retrieved],
                "schema_tokens": approx_tokens(schema_text),
                "recent_buffer_tokens": approx_tokens(recent_buffer),
                "wm_tokens_total": wm_total,
                "wm_under_budget": wm_total <= WM_BUDGET_TOKENS,
            }
        )

    # Final answer: schema is authoritative. Keep recent buffer too (still inside budget).
    final_user = (
        f"PROJECT SCHEMA (authoritative; raw retrievals shed after merge):\n\n"
        f"{schema_text}\n\n"
        f"RECENT BUFFER (last retrieved snippet, kept for fresh detail):\n\n"
        f"{recent_buffer or '(empty)'}\n\n"
        f"---\n\nTASK BRIEF:\n{case.task_brief}\n\n"
        f"---\n\nFINAL REQUEST:\n{case.final_question}"
    )
    answer = await llm(OPERATOR_FINAL_SYSTEM, final_user, reasoning="low")
    return {
        "answer": answer,
        "trace": trace,
        "final_schema": schema_text,
        "final_schema_tokens": approx_tokens(schema_text),
        "final_recent_buffer_tokens": approx_tokens(recent_buffer),
        "final_prompt_tokens": approx_tokens(final_user),
    }


# ----------------------------------------------------------------------------
# Judge
# ----------------------------------------------------------------------------


async def judge_answer(case: TestCase, answer: str) -> dict:
    rubric_block = "\n".join(
        f"- {item['id']}: {item['description']}"
        f" | signal phrases: {', '.join(item.get('needles_any', []))}"
        + (
            f" | forbidden: {', '.join(item.get('forbidden_any', []))}"
            if item.get("forbidden_any")
            else ""
        )
        for item in case.rubric
    )
    user_msg = (
        f"FINAL USER REQUEST:\n{case.final_question}\n\n"
        f"ASSISTANT ANSWER:\n{answer}\n\n"
        f"RUBRIC ITEMS:\n{rubric_block}\n\n"
        f"Output JSON only."
    )
    raw = await llm(JUDGE_SYSTEM, user_msg, reasoning="low")
    parsed = parse_json_obj(raw)
    items = parsed.get("items", []) if isinstance(parsed, dict) else []
    honored = sum(1 for it in items if it.get("verdict") == "HONORED")
    violated = sum(1 for it in items if it.get("verdict") == "VIOLATED")
    unclear = sum(1 for it in items if it.get("verdict") == "UNCLEAR")
    total = len(case.rubric)
    return {
        "items": items,
        "honored": honored,
        "violated": violated,
        "unclear": unclear,
        "total": total,
        "honored_rate": honored / total if total else 0.0,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


async def run_one(case: TestCase) -> dict:
    em_total = sum(approx_tokens(c.text) for c in case.memory)
    print(
        f"\n=== {case.name} ({case.domain}) | EM≈{em_total} tokens, {len(case.memory)} chunks ==="
    )

    baseline = await run_baseline(case)
    operator = await run_operator(case)

    baseline_judge = await judge_answer(case, baseline["answer"])
    operator_judge = await judge_answer(case, operator["answer"])

    print(
        f"  baseline: honored {baseline_judge['honored']}/{baseline_judge['total']} | "
        f"violated {baseline_judge['violated']} | "
        f"final_wm≈{baseline['final_wm_tokens']}t"
    )
    print(
        f"  operator: honored {operator_judge['honored']}/{operator_judge['total']} | "
        f"violated {operator_judge['violated']} | "
        f"schema≈{operator['final_schema_tokens']}t buf≈{operator['final_recent_buffer_tokens']}t"
    )

    # Token traces — per-case JSON.
    trace_path = TRACES_DIR / f"{case.name}.trace.json"
    trace_payload = {
        "case": case.name,
        "domain": case.domain,
        "external_memory_tokens": em_total,
        "external_memory_chunks": len(case.memory),
        "wm_budget_tokens": WM_BUDGET_TOKENS,
        "schema_target_tokens": SCHEMA_TARGET_TOKENS,
        "baseline_trace": baseline["trace"],
        "operator_trace": operator["trace"],
        "operator_final_schema": operator["final_schema"],
    }
    trace_path.write_text(json.dumps(trace_payload, indent=2))

    return {
        "name": case.name,
        "domain": case.domain,
        "external_memory_tokens": em_total,
        "external_memory_chunks": len(case.memory),
        "rubric_total": len(case.rubric),
        "baseline": {
            "answer": baseline["answer"],
            "judge": baseline_judge,
            "final_wm_tokens": baseline["final_wm_tokens"],
            "final_prompt_tokens": baseline["final_prompt_tokens"],
        },
        "operator": {
            "answer": operator["answer"],
            "judge": operator_judge,
            "final_schema_tokens": operator["final_schema_tokens"],
            "final_recent_buffer_tokens": operator["final_recent_buffer_tokens"],
            "final_prompt_tokens": operator["final_prompt_tokens"],
        },
    }


async def main() -> None:
    results: list[dict] = []
    for case in SCENARIOS:
        results.append(await run_one(case))

    baseline_rates = [r["baseline"]["judge"]["honored_rate"] for r in results]
    operator_rates = [r["operator"]["judge"]["honored_rate"] for r in results]
    baseline_violations = [r["baseline"]["judge"]["violated"] for r in results]
    operator_violations = [r["operator"]["judge"]["violated"] for r in results]
    operator_schema_sizes = [r["operator"]["final_schema_tokens"] for r in results]
    baseline_wm_sizes = [r["baseline"]["final_wm_tokens"] for r in results]

    summary = {
        "n_scenarios": len(results),
        "wm_budget_tokens": WM_BUDGET_TOKENS,
        "schema_target_tokens": SCHEMA_TARGET_TOKENS,
        "retrieval_k": RETRIEVAL_K,
        "baseline_mean_honored_rate": sum(baseline_rates) / len(baseline_rates)
        if baseline_rates
        else 0.0,
        "operator_mean_honored_rate": sum(operator_rates) / len(operator_rates)
        if operator_rates
        else 0.0,
        "baseline_total_violations": sum(baseline_violations),
        "operator_total_violations": sum(operator_violations),
        "operator_mean_schema_tokens": (
            sum(operator_schema_sizes) / len(operator_schema_sizes)
        )
        if operator_schema_sizes
        else 0,
        "operator_max_schema_tokens": max(operator_schema_sizes)
        if operator_schema_sizes
        else 0,
        "baseline_mean_wm_tokens": (sum(baseline_wm_sizes) / len(baseline_wm_sizes))
        if baseline_wm_sizes
        else 0,
        "external_memory_tokens_per_case": [
            r["external_memory_tokens"] for r in results
        ],
    }

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    payload = {"summary": summary, "scenarios": results}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote results: {RESULTS_PATH}")
    print(f"Wrote per-case token traces under: {TRACES_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
