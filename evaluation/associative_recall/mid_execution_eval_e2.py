"""E2 — freelance executor agent: tests COVERAGE + retrieval together.

The previous E1 handed the executor a hand-written sub-decision script. That
gave the agent a free pass on the harder competence — thinking of "what
colors?" or "what about allergies?" in the first place. E2 fixes that:

  1. Agent gets only the task_prompt.
  2. Agent produces its OWN plan + executes step by step.
  3. The hand-written `subdecision_script` becomes the GOLD COVERAGE CHECKLIST
     — for each gold sub-decision, an LLM judge checks whether the agent's
     transcript addresses it.
  4. For covered sub-decisions, the agent's self-identified step (cue and/or
     content) is used to query EM and score retrieval against the gold plant.

End-to-end metric per scenario:

    triggered_recall_full@K = mean over gold sub-decisions of
        (1[covered AND gold plant in top-K] / 1)

Coverage failures (agent forgot the sub-decision exists) score 0.

Two executor modes are compared:

  - "freelance_natural"   : agent plans + executes; no CUE: lines.
                            Cue per step = the agent's content for that step.
  - "freelance_cue_aware" : agent plans + executes + emits `CUE: <query>`
                            per step. Cue per step = the CUE: text.

Usage:

    uv run python evaluation/associative_recall/mid_execution_eval_e2.py
    uv run python evaluation/associative_recall/mid_execution_eval_e2.py --scenario banquet-01 --mode cue_aware
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import openai
from dotenv import load_dotenv
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Content,
    Event,
    MessageContext,
    Text,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from mid_execution_eval import (  # type: ignore
    COLLECTION_PREFIX,
    NAMESPACE,
    RESULTS_DIR,
    _scenario_collection,
    ingest_scenario,
    load_locomo_segments,
    load_scenarios,
    load_speakers,
    probe,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv(Path(__file__).resolve().parent / ".env")

CACHE_DIR = Path(__file__).resolve().parent / "cache"
EXECUTOR_CACHE_FILE = CACHE_DIR / "mid_exec_e2_executor_cache.json"
JUDGE_CACHE_FILE = CACHE_DIR / "mid_exec_e2_judge_cache.json"

EXECUTOR_MODEL = "gpt-5-mini"
JUDGE_MODEL = "gpt-5-mini"

# Executor backend — set EXECUTOR_BACKEND=claude to route _llm through the
# `claude --print` CLI (uses your subscription OAuth). Default: OpenAI.
EXECUTOR_BACKEND = os.environ.get("EXECUTOR_BACKEND", "openai").lower()
# Reasoning mode for SA-full Phase 1:
#   "explicit" (default): prompted THINKING: lines via Chat Completions API.
#   "native": OpenAI Responses API + previous_response_id (server-side reasoning).
REASONING_MODE = os.environ.get("REASONING_MODE", "explicit").lower()
CLAUDE_CLI = "claude"
# Limit Claude concurrency to avoid hammering the subscription rate limits.
_CLAUDE_SEM = asyncio.Semaphore(int(os.environ.get("CLAUDE_CONCURRENCY", "3")))


# --------------------------------------------------------------------------
# Cache (mirror E1)
# --------------------------------------------------------------------------


class _SimpleCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, str] = {}
        if path.exists():
            try:
                self._cache = json.loads(path.read_text())
            except Exception:
                self._cache = {}
        self._dirty = False

    @staticmethod
    def _key(tag: str, payload: str) -> str:
        return hashlib.sha256(f"{tag}:{payload}".encode()).hexdigest()

    def get(self, tag: str, payload: str) -> str | None:
        return self._cache.get(self._key(tag, payload))

    def put(self, tag: str, payload: str, value: str) -> None:
        self._cache[self._key(tag, payload)] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._cache))
        tmp.replace(self._path)
        self._dirty = False


# --------------------------------------------------------------------------
# Executor prompts — freelance (no script)
# --------------------------------------------------------------------------


NATURAL_FREELANCE_SYSTEM = """\
You are an executor agent given a task by a coworker. The coworker will not \
walk you through the steps — you must plan AND execute the work yourself.

OVERALL TASK:
{task_prompt}

OUTPUT FORMAT (follow exactly):

Section 1 — PLAN. Number the sub-steps you'll do. Write each like:
    [STEP 1] <one short line describing the sub-step>
    [STEP 2] <one short line>
    ...
Cover all sub-steps a competent person would do — don't skip "obvious" \
sub-decisions. If the task involves a presentation, "decide on visual style" \
is a real sub-step. If it involves a meal, "choose the menu" is a real \
sub-step. Etc. Aim for 4-8 plan items total.

Section 2 — EXECUTE. For each step in order:
    --- STEP N ---
    <1-3 short concrete sentences delivering the actual content / decision \
for that step. Use real names, real numbers, real choices. Do NOT write \
meta-commentary like "I would consult brand guidelines" — make a concrete \
choice and move on. If you don't know a relevant fact, use a plausible \
placeholder.>

Section 3 — END. Emit `--- DONE ---` on its own line.

Do not write commentary outside this format.
"""


CUE_AWARE_FREELANCE_SYSTEM = """\
You are an executor agent given a task by a coworker. The coworker will not \
walk you through the steps — you must plan AND execute the work yourself. \
You also have access to a memory of past chat history that may contain \
useful context (brand guidelines, allergies, deadlines, preferences, \
interpersonal dynamics, etc.) — but you must ASK for what you need before \
each step.

OVERALL TASK:
{task_prompt}

OUTPUT FORMAT (follow exactly):

Section 1 — PLAN. Number the sub-steps you'll do:
    [STEP 1] <one short line describing the sub-step>
    [STEP 2] <one short line>
    ...
Cover all sub-steps a competent person would do — don't skip "obvious" \
sub-decisions like visual style for a deck or menu choice for a meal. Aim \
for 4-8 plan items total.

Section 2 — EXECUTE. For each step in order:
    --- STEP N ---
    CUE: <one short retrieval query (5-15 words) about what fact you'd \
want to recall from past chat history for THIS step. Use `CUE: none` if \
nothing in past context applies. Make the query specific enough that it \
won't just retrieve generic chat about the overall task.>
    <1-3 short concrete sentences delivering the actual content / decision \
for that step. Use real names, real numbers, real choices.>

Section 3 — END. Emit `--- DONE ---` on its own line.

Do not write commentary outside this format.
"""


# Primed variant: explicitly enumerate the common sub-decision categories an
# experienced worker would think about. Tests whether the agent's competence
# gap is "doesn't know what to think about" vs "knows but skips."
PRIMED_CUE_AWARE_SYSTEM = """\
You are an experienced executor agent given a task by a coworker. The \
coworker will not walk you through the steps — you must plan AND execute \
the work yourself. You also have access to a memory of past chat history \
that may contain useful context — but you must ASK for what you need before \
each step.

OVERALL TASK:
{task_prompt}

PLANNING DISCIPLINE — before drafting your plan, mentally check whether \
ANY of these sub-decision categories apply to THIS task. They are the \
categories experienced workers routinely address; do not skip ones that \
genuinely apply:

For deliverables to a specific recipient (presentation, document, deck, email):
- recipient-specific audience constraints (accessibility, format preferences, reading device)
- brand identity / visual style / company guidelines from this recipient
- past feedback or complaints from this recipient
- legal / compliance / disclaimer requirements
- length / page-count / time-budget caps
- closing / signoff conventions specific to this recipient

For tasks involving people coming together (meal, party, retreat, meeting):
- dietary / medical constraints (allergies, religious, vegan/gluten, etc.)
- interpersonal dynamics (existing conflicts to avoid in seating)
- accessibility / mobility constraints
- per-head or total budget caps
- weather / dress code / venue-specific logistics
- past failures or incidents at similar gatherings

For travel / scheduling tasks:
- document validity (passport, visa, license, expiry dates)
- language constraints / interpreter needs
- vendor / supplier bans or preferences (airlines, hotels, caterers)
- mobility / accommodation preferences
- expense / receipt / reimbursement quirks
- past incidents with specific vendors

For multi-step / project / engineering tasks:
- collaborator availability and single-bus-factor bottlenecks
- past blockers from similar projects
- external commitments / contractual deadlines / penalty clauses
- QA / review / approval lead times
- team unavailability windows (offsites, holidays, leave)
- vendor / platform-team intake policies

For communication / writing tasks:
- name forms / nicknames / how key people prefer to be addressed
- tradition / signature lines / closing conventions
- subject-line conventions
- distribution-list / forwarding / cc conventions

This is not an exhaustive list — use it as a checklist, then add anything \
specific to your task that isn't here.

OUTPUT FORMAT (follow exactly):

Section 1 — PLAN. Number the sub-steps:
    [STEP 1] <one short line describing the sub-step>
    [STEP 2] <one short line>
    ...
Cover every category from the discipline list above that applies to your \
specific task. Aim for 6-12 plan items.

Section 2 — EXECUTE. For each step in order:
    --- STEP N ---
    CUE: <one short retrieval query (5-15 words) about what fact you'd \
want to recall from past chat history for THIS step. Use `CUE: none` if \
nothing in past context applies.>
    <1-3 short concrete sentences delivering the actual content / decision \
for that step. Use real names, real numbers, real choices.>

Section 3 — END. Emit `--- DONE ---` on its own line.

Do not write commentary outside this format.
"""


# Spreading activation: iterative probe-then-see loop. Each iteration the
# agent inspects what's been retrieved and decides what to ask next. Tests
# whether iterating beats single-pass retrieve_revise — the cognitive-science
# analog of spreading activation in human associative memory.

SPREADING_PROBE_SYSTEM = """\
You are exploring a memory of past chat history to gather context for a \
task. You can run multiple rounds of memory queries — each round you see \
what the prior probes returned, then decide what to ask next.

Memory retrieves by semantic similarity to your probe text. So a probe \
that uses words appearing in a stored fact will surface that fact; a \
probe whose words don't match won't, even if the fact is logically \
related.

OVERALL TASK:
{task_prompt}

PROBES YOU'VE ALREADY RUN (do not repeat — semantically equivalent or \
verbatim):
{prior_probes}

WHAT YOU'VE LEARNED FROM MEMORY SO FAR (deduped snippets):
---
{accumulated_context}
---

Before generating probes, take a moment to actually think:"""


# Multi-turn variant of the iterative probe loop. The system prompt is static;
# each round the user message carries only the DELTA (new snippets surfaced
# by prior probes, prior-probes list to avoid repeats). Rounds run as one
# growing conversation so the agent's THINKING blocks from prior rounds
# remain visible — closer to human "where I left off" continuity than the
# single-turn rebuild.

SPREADING_PROBE_SYSTEM_MT = """\
You are exploring a memory of past chat history to gather context for a \
task. You'll run multiple rounds of memory queries. Each round you see \
what the prior probes returned, then decide what to ask next. Your \
THINKING from prior rounds is visible to you — use it. Don't restart \
your reasoning each round; build on what you already concluded.

Memory retrieves by semantic similarity to your probe text. So a probe \
that uses words appearing in a stored fact will surface that fact; a \
probe whose words don't match won't, even if the fact is logically \
related.

OVERALL TASK:
{task_prompt}

PROBE-GENERATION REASONING DISCIPLINES (apply each round):

- **Implications and chains**: when you find a fact, ask "if this is \
true, what other fact must / probably exists alongside it?" — and probe \
for that next. Many sub-decisions need 2-5 facts combined; surface them \
all, not just the most obvious one.
- **Close reading**: important facts can be stated by negation, buried \
in narrative, or mentioned in passing. Don't restrict yourself to direct \
declarations.

Each round, output:

THINKING: <2-6 sentences naming what you learned this round, what \
threads are still open, what you're chasing next. Reference your prior \
rounds' THINKING when relevant — you can see it.>
PROBE: <a specific retrieval query that builds on what you've learned>
PROBE: <another query, different angle>
... up to 4 probes ...

Or output exactly: STOP

if you've genuinely saturated."""


SPREADING_PROBE_SYSTEM_MT_NATIVE = """\
You are exploring a memory of past chat history to gather context for a \
task. You'll run multiple rounds of memory queries. Each round you see \
what the prior probes returned, then decide what to ask next. Your \
reasoning state from prior rounds is carried forward internally — build \
on it; don't restart your reasoning each round.

Memory retrieves by semantic similarity to your probe text. So a probe \
that uses words appearing in a stored fact will surface that fact; a \
probe whose words don't match won't, even if the fact is logically \
related.

OVERALL TASK:
{task_prompt}

PROBE-GENERATION REASONING DISCIPLINES (apply each round):

- **Implications and chains**: when you find a fact, ask "if this is \
true, what other fact must / probably exists alongside it?" — and probe \
for that next. Many sub-decisions need 2-5 facts combined; surface them \
all, not just the most obvious one.
- **Close reading**: important facts can be stated by negation, buried \
in narrative, or mentioned in passing. Don't restrict yourself to direct \
declarations.

Output format (one probe per line, no other text):
PROBE: <text>
PROBE: <text>
PROBE: <text>

Or output exactly: STOP

if you've genuinely saturated."""


SPREADING_PROBE_USER_INITIAL_MT = """\
Round 1. Here are the seed snippets returned by probing memory with the \
task description itself:
---
{seed_snippets}
---

Probes used so far: ["task description"]

Generate next probes (or STOP)."""


SPREADING_PROBE_USER_FOLLOWUP_MT = """\
Round {round_n}. New snippets surfaced by the probes you just emitted:
---
{new_snippets}
---

Total probes used so far: {n_probes_total}.

Generate next probes (or STOP)."""


PROBE_LINE_RE = re.compile(r"^\s*PROBE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


SPREADING_PLAN_EXECUTE_SYSTEM = """\
You are an executor agent. You explored a memory of past chat history \
across multiple rounds and accumulated relevant context. Now plan and \
execute the task using everything you learned.

OVERALL TASK:
{task_prompt}

ACCUMULATED MEMORY CONTEXT (the snippets you surfaced):
---
{full_context}
---

OUTPUT FORMAT (follow exactly):

Section 1 — PLAN. Number sub-steps incorporating everything you learned. \
Add sub-decisions that the memory revealed; modify existing ones; remove \
anything memory contradicts. Aim for 6-14 items.
    [STEP 1] <one short line>
    [STEP 2] <one short line>
    ...

Section 2 — EXECUTE. For each step:
    --- STEP N ---
    CUE: <one short retrieval query for THIS step, or `CUE: none`>
    <1-3 short concrete sentences delivering the actual content / decision \
for the step. Use real names and real values from the memory above where \
applicable.>

Section 3 — END. Emit `--- DONE ---` on its own line.
"""


# === SPREADING ACTIVATION AT BOTH PLANNING AND EXECUTION ===
# After planning-time spread + plan generation, each step independently spreads
# again — the agent may discover decision points or facts that the plan missed.
# This is closer to real agent cognition: plans don't capture everything; new
# threads emerge from doing the work.

SPREADING_PLAN_ONLY_SYSTEM = """\
You are an executor agent. You've done initial memory exploration and have \
the context below. Now PLAN the task — list the sub-steps you intend to \
execute. You'll execute each step separately and may probe memory again as \
each step comes up.

OVERALL TASK:
{task_prompt}

CONTEXT GATHERED FROM PLANNING-TIME MEMORY EXPLORATION:
---
{full_context}
---

PLANNING REASONING DISCIPLINES (apply when drafting your plan):

- **External / world facts**: do external facts (calendar conventions, \
holidays, time zones, geography, regulations, language conventions, \
common cultural expectations) apply to the entities, dates, places, or \
people in the context above? They aren't in memory but are in your \
training. Bring them to bear. Add a sub-step if any external fact \
materially changes what the plan should do.
- **Recency / supersession**: if multiple snippets in the context \
contradict each other, look at the `[date, time]` timestamp prefix on \
each. Treat the most recent statement of a rule as the current one; \
older statements are stale unless explicitly grandfathered. If you're \
not sure which is current, plan a sub-step to verify.
- **Don't skip sub-decisions** you noticed from the context above, even \
if they feel "obvious" or "implicit." If a competent worker would \
address them, list them.

OUTPUT FORMAT (only the plan, nothing else):
    [STEP 1] <one short line>
    [STEP 2] <one short line>
    ...
Aim for 6-14 plan items.
"""


SPREADING_EXEC_STEP_SYSTEM = """\
You are executing a step of a multi-step task. Before writing the step's \
content, decide whether you need to probe memory again — the planning \
context might not cover what THIS specific step needs.

OVERALL TASK:
{task_prompt}

THE FULL PLAN (you are at step {step_id}):
{plan}

CONTEXT YOU GATHERED EARLIER FROM MEMORY (planning-time exploration):
---
{plan_context}
---

YOUR PRIOR REASONING (excerpts from your own planning-round THINKING, \
retrieved by similarity to this step — useful for chain-of-thought \
continuity, but does NOT contain external facts):
---
{cognition_context}
---

NEW CONTEXT GATHERED FROM PRIOR EXECUTION STEPS (may include additional \
memory snippets surfaced for those steps):
---
{exec_context}
---

WHAT YOU'VE WRITTEN SO FAR:
{prior_outputs}

CURRENT STEP TO EXECUTE:
[STEP {step_id}] {step_label}

Think briefly: looking at this specific step now, are there facts you'd \
want that the planning context might not have surfaced? Did doing the \
prior steps reveal new threads worth probing? If a name or label appeared \
in earlier context but you don't yet have its specifics for THIS step's \
purpose, probe for them now. (Memory matches on surface text — different \
forms of a name retrieve different turns.)

If genuinely nothing new to probe — context is sufficient — emit \
exactly:
CUE: none

Otherwise emit 1-3 CUE lines targeting what you need for this step:
THINKING: <2-4 sentences naming what you need>
CUE: <short retrieval query that references a specific entity, label, or \
concept>
CUE: <text>
CUE: <text>
"""


SPREADING_EXEC_WRITE_SYSTEM = """\
Execute step {step_id} now. Use any relevant facts from the context.

OVERALL TASK:
{task_prompt}

THE FULL PLAN:
{plan}

ALL MEMORY CONTEXT YOU'VE GATHERED (planning + prior steps + this step's \
fresh probes):
---
{full_context}
---

YOUR PRIOR REASONING (excerpts from your own planning-round THINKING, \
retrieved by similarity to this step — useful for chain-of-thought \
continuity, but does NOT contain external facts):
---
{cognition_context}
---

WHAT YOU'VE WRITTEN SO FAR:
{prior_outputs}

CURRENT STEP:
[STEP {step_id}] {step_label}

DECISION REASONING DISCIPLINES (apply when writing this step):

- **External / world facts**: do external facts (calendar conventions, \
holidays, time zones, geography, regulations) apply to this specific \
step's decision? They aren't in memory but are in your training. Apply \
them.
- **Recency / supersession**: if context snippets contradict each other, \
the timestamps in the `[date, time]` prefix tell you which is most \
recent. Apply the most-recent rule unless an older one is explicitly \
grandfathered.

Write 1-3 short concrete sentences delivering the step's actual content / \
decision. Use real names, values, and rules from the context above where \
applicable. No meta-commentary, no plans — just the deliverable.
"""


# Plan-retrieve-revise: draft plan, use each plan item as a memory probe,
# inject retrieved snippets into context, agent revises and executes. Tests
# whether "agent can't think of the sub-decision" is fixed by giving the
# agent the relevant facts at planning time (RAG-on-plan-items).
RETRIEVE_REVISE_AND_EXECUTE_SYSTEM = """\
You are an executor agent. You drafted an initial plan, then a memory \
search was run for each of your plan items against past chat history. The \
memory may contain context (brand guidelines, allergies, deadlines, \
preferences, interpersonal dynamics, etc.) that should reshape your plan \
and inform what you write.

OVERALL TASK:
{task_prompt}

YOUR INITIAL DRAFT PLAN:
{draft_plan}

MEMORY CONTEXT — past chat turns surfaced by querying memory with each of \
your plan items. Read this carefully — it should change your plan if it \
reveals constraints, preferences, or facts you didn't know:
---
{memory_context}
---

OUTPUT FORMAT (follow exactly):

Section 1 — REVISED PLAN. Number sub-steps incorporating what you learned \
from memory. Add new sub-decisions the memory revealed; modify existing \
ones; remove anything the memory contradicts. Aim for 6-14 items.
    [STEP 1] <one short line>
    [STEP 2] <one short line>
    ...

Section 2 — EXECUTE. For each step in order:
    --- STEP N ---
    CUE: <one short retrieval query for THIS step, or `CUE: none`>
    <1-3 short concrete sentences delivering the actual content / decision \
for the step. Use real names and real values from the memory above where \
applicable.>

Section 3 — END. Emit `--- DONE ---` on its own line.

Do not write commentary outside this format.
"""


# Critic-pass: agent drafts a plan, separate critic call lists what's missing,
# agent revises and executes. Two-pass planning, plus per-step CUEs.
CRITIC_DRAFT_SYSTEM = """\
You are an executor agent. Draft a numbered plan for this task — list the \
sub-steps you intend to execute. Don't execute yet, just plan.

OVERALL TASK:
{task_prompt}

OUTPUT FORMAT — only the plan, nothing else:
    [STEP 1] <one short line>
    [STEP 2] <one short line>
    ...
Aim for 4-8 plan items.
"""


CRITIC_REVIEW_PROMPT = """\
You are a senior reviewer auditing another agent's plan for completeness. \
The agent's job is to execute a task end-to-end. Your job is to identify \
sub-decisions the agent's plan FORGOT to address.

OVERALL TASK:
{task_prompt}

AGENT'S DRAFT PLAN:
{draft_plan}

Identify between 2 and 6 sub-decisions the plan likely needs but does NOT \
explicitly cover. Consider categories like: recipient-specific constraints \
(brand, accessibility, past feedback, legal disclaimers); people-related \
constraints (dietary, interpersonal, mobility, budget); document/credential \
validity; vendor preferences/bans; collaborator availability; external \
deadlines; QA windows; communication conventions (name forms, signature \
traditions). Skip categories that genuinely don't apply.

OUTPUT FORMAT — one missing sub-decision per line, nothing else:
    MISSING: <short description of the sub-decision the plan should add>
    MISSING: <...>

If the plan looks complete to you, output exactly the single line:
    MISSING: none
"""


CRITIC_REVISE_AND_EXECUTE_SYSTEM = """\
You are an executor agent. A senior reviewer has read your draft plan and \
identified sub-decisions you forgot. Incorporate ALL the reviewer's missing \
sub-decisions into your revised plan, then execute the revised plan.

OVERALL TASK:
{task_prompt}

YOUR DRAFT PLAN:
{draft_plan}

REVIEWER FOUND THESE MISSING SUB-DECISIONS — you MUST add each as a step:
{critic_findings}

You also have access to past chat history via memory — ASK for what you \
need before each step.

OUTPUT FORMAT (follow exactly):

Section 1 — REVISED PLAN. Number the sub-steps (your original plan + \
reviewer's additions). Aim for 6-14 items:
    [STEP 1] <one short line>
    [STEP 2] <one short line>
    ...

Section 2 — EXECUTE. For each step in order:
    --- STEP N ---
    CUE: <one short retrieval query about what fact you'd want for THIS \
step, or `CUE: none`>
    <1-3 short concrete sentences delivering the actual content / decision.>

Section 3 — END. Emit `--- DONE ---` on its own line.

Do not write commentary outside this format.
"""


STEP_LABEL_RE = re.compile(r"^\s*\[STEP\s+(\d+)\]\s*(.+?)\s*$", re.MULTILINE)
STEP_DELIM_RE = re.compile(r"^\s*---\s*STEP\s+(\d+)\s*---\s*$", re.MULTILINE)
DONE_RE = re.compile(r"^\s*---\s*DONE\s*---\s*$", re.MULTILINE)
CUE_LINE_RE = re.compile(r"^\s*CUE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def parse_executor_response(response: str) -> dict:
    """Parse the executor agent's response into structured steps.

    Returns: {
        plan: [{step_id, label}],
        steps: [{step_id, content, cue (or "")}],
        raw: original text,
    }
    """
    out = {"plan": [], "steps": [], "raw": response or ""}

    # ---- Plan section ----
    for m in STEP_LABEL_RE.finditer(response or ""):
        out["plan"].append(
            {
                "step_id": int(m.group(1)),
                "label": m.group(2).strip(),
            }
        )

    # ---- Execute section ----
    delim_matches = list(STEP_DELIM_RE.finditer(response or ""))
    done_match = DONE_RE.search(response or "")
    end_pos = done_match.start() if done_match else len(response or "")

    for i, m in enumerate(delim_matches):
        step_id = int(m.group(1))
        body_start = m.end()
        body_end = (
            delim_matches[i + 1].start() if i + 1 < len(delim_matches) else end_pos
        )
        body = (response or "")[body_start:body_end].strip()
        cue = ""
        cue_m = CUE_LINE_RE.search(body)
        if cue_m:
            cue_text = cue_m.group(1).strip()
            if cue_text.lower() not in ("none", "(none)", "n/a"):
                cue = cue_text
            body = (body[: cue_m.start()] + body[cue_m.end() :]).strip()
        out["steps"].append(
            {
                "step_id": step_id,
                "content": body,
                "cue": cue,
            }
        )

    return out


# --------------------------------------------------------------------------
# Executor agent
# --------------------------------------------------------------------------


async def _claude_print(prompt: str) -> str:
    """Invoke `claude --print` with prompt on stdin; return stdout text."""
    async with _CLAUDE_SEM:
        proc = await asyncio.create_subprocess_exec(
            CLAUDE_CLI,
            "--print",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(prompt.encode())
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI returned {proc.returncode}: {stderr.decode()[:500]}"
            )
        return stdout.decode()


async def _llm(
    openai_client, system: str, user: str, *, cache: _SimpleCache, cache_tag: str
) -> str:
    # Backend selector: "openai" (default) or "claude" (subprocess to claude CLI)
    backend = EXECUTOR_BACKEND
    full_payload = json.dumps({"backend": backend, "system": system, "user": user})
    effective_tag = f"{backend}:{cache_tag}"
    cached = cache.get(effective_tag, full_payload)
    if cached is not None:
        return cached

    if backend == "claude":
        # claude --print takes a single prompt; merge system + user.
        merged = f"{system}\n\n---\n\n{user}" if system else user
        out = await _claude_print(merged)
    else:
        resp = await openai_client.chat.completions.create(
            model=EXECUTOR_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            reasoning_effort="low",
        )
        out = resp.choices[0].message.content or ""

    cache.put(effective_tag, full_payload, out)
    return out


async def _llm_multiturn(
    openai_client,
    messages: list[dict],
    *,
    cache: _SimpleCache,
    cache_tag: str,
) -> str:
    """Multi-turn chat. Persists the agent's prior assistant outputs across
    rounds so its thinking thread (THINKING: blocks emitted by the prompted
    format) carries forward — closer to human "where I left off" continuity.

    Caller maintains the messages list and appends the assistant response
    after each call.
    """
    backend = EXECUTOR_BACKEND
    payload = json.dumps({"backend": backend, "messages": messages})
    effective_tag = f"{backend}:{cache_tag}"
    cached = cache.get(effective_tag, payload)
    if cached is not None:
        return cached

    if backend == "claude":
        # claude --print takes a single string. Concatenate the conversation.
        # System prompt + alternating user/assistant turns.
        parts: list[str] = []
        for m in messages:
            role = m["role"].upper()
            parts.append(f"--- {role} ---\n{m['content']}")
        parts.append("--- ASSISTANT ---")  # cue claude to continue
        merged = "\n\n".join(parts)
        out = await _claude_print(merged)
    else:
        resp = await openai_client.chat.completions.create(
            model=EXECUTOR_MODEL,
            messages=messages,
            reasoning_effort="low",
        )
        out = resp.choices[0].message.content or ""

    cache.put(effective_tag, payload, out)
    return out


async def _llm_responses(
    openai_client,
    user_text: str,
    system_text: str | None,
    previous_response_id: str | None,
    *,
    cache: _SimpleCache,
    cache_tag: str,
) -> tuple[str, str]:
    """OpenAI Responses API call with native reasoning persistence.

    Reasoning state is carried server-side via previous_response_id; the
    caller does not maintain a messages[] list. Returns (text, new_response_id)
    so the caller can chain the next round.

    Cached by sha256 of {user_text, system_text, previous_response_id, "responses"}
    so that re-runs are deterministic. Both text AND new_response_id are
    cached (as a JSON-encoded pair) so chaining stays consistent across runs.
    """
    payload = json.dumps(
        {
            "user_text": user_text,
            "system_text": system_text,
            "previous_response_id": previous_response_id,
            "kind": "responses",
        }
    )
    effective_tag = f"openai-responses:{cache_tag}"
    cached = cache.get(effective_tag, payload)
    if cached is not None:
        try:
            obj = json.loads(cached)
            return obj["text"], obj["id"]
        except Exception:
            # Corrupt cache entry — fall through and recompute.
            pass

    kwargs: dict = {
        "model": EXECUTOR_MODEL,
        "input": user_text,
        "reasoning": {"effort": "low"},
    }
    if system_text is not None:
        kwargs["instructions"] = system_text
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id

    resp = await openai_client.responses.create(**kwargs)

    # Prefer output_text convenience accessor; fall back to iterating output items.
    text = getattr(resp, "output_text", None) or ""
    if not text:
        parts: list[str] = []
        for item in getattr(resp, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
                    else:
                        # Some SDK versions wrap text in a Text object.
                        inner = getattr(t, "value", None)
                        if isinstance(inner, str):
                            parts.append(inner)
        text = "".join(parts)

    new_id = getattr(resp, "id", "") or ""
    cache.put(effective_tag, payload, json.dumps({"text": text, "id": new_id}))
    return text, new_id


# --------------------------------------------------------------------------
# LLM precision rerank
# --------------------------------------------------------------------------

# When EM returns top-K by embedding cosine, the top of the list is sometimes
# dominated by surface-token-similar decoys. A small LLM rerank pass reads
# the query and the top-K candidates and picks the K most precisely matching.
# Embedding cheap, rerank cheap; combined this fixes adversarial outranking
# and partial multi-hop without changing the agent's planning loop.

RERANK_PROMPT = """\
You are reranking retrieval results from a memory of past chat history.

QUERY (what the agent is looking for):
{query}

CANDIDATES (numbered, with timestamp + speaker):
{numbered_candidates}

Rank candidates by how directly they answer the query. Be precise: if \
multiple candidates discuss the same entity, prefer the one that is most \
recent / active over stale ones; prefer specific facts over generic chat; \
prefer the candidate whose content actually answers the query over ones \
that just mention adjacent topics.

Output ONE line, indices only, in rank order, top {k_keep} only:
RANKED: <i1>, <i2>, <i3>, ...
"""

_RANKED_RE = re.compile(r"^\s*RANKED\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


async def probe_with_rerank(
    memory,
    query: str,
    *,
    K_initial: int,
    K_final: int,
    openai_client,
    cache: _SimpleCache,
):
    """Embedding probe top-K_initial, LLM rerank to top-K_final."""
    if not query.strip():
        return []
    initial_hits = await probe(memory, query, K_initial)
    if len(initial_hits) <= K_final:
        return initial_hits

    numbered = "\n".join(
        f"[{i + 1}] {(h.formatted_text or h.text or '')[:240]}"
        for i, h in enumerate(initial_hits)
    )
    prompt = RERANK_PROMPT.format(
        query=query[:500],
        numbered_candidates=numbered,
        k_keep=K_final,
    )
    raw = await _llm(
        openai_client,
        "You are a precise reranker that returns indices in rank order.",
        prompt,
        cache=cache,
        cache_tag=f"{EXECUTOR_MODEL}:rerank",
    )
    m = _RANKED_RE.search(raw)
    if not m:
        return initial_hits[:K_final]
    parts = re.findall(r"\d+", m.group(1))
    out = []
    seen: set[int] = set()
    for p in parts:
        idx = int(p) - 1
        if 0 <= idx < len(initial_hits) and idx not in seen:
            out.append(initial_hits[idx])
            seen.add(idx)
        if len(out) >= K_final:
            break
    if len(out) < K_final:
        for i, h in enumerate(initial_hits):
            if i in seen:
                continue
            out.append(h)
            if len(out) >= K_final:
                break
    return out


# Switch via env: SA_RERANK=1 routes SA-full probes through the rerank wrapper.
SA_RERANK = os.environ.get("SA_RERANK", "0") == "1"

# Switch via env: EM_INGEST_THINKING=1 ingests the executor agent's own
# THINKING blocks and PROBE lists back into EM after each Phase 1 round of
# spreading_activation_full mode, so subsequent probes can surface the
# agent's prior reflections alongside the chat plants.
EM_INGEST_THINKING = os.environ.get("EM_INGEST_THINKING", "0") == "1"

# Companion toggle: when 1, retrieval-time hit lists drop event_type ∈
# {agent_thinking, agent_probes} so the agent's own ingested reflections do
# not displace real plants/decoys in the top-K. The events remain in EM (so
# the agent could still surface them for cue enrichment via a separate
# retrieval call), but they are invisible to the SA-full probe path. Set to
# 1 to test whether the substrate-style ingest helps or hurts ONCE the
# displacement effect is removed.
EM_RETRIEVAL_FILTER_AGENT = os.environ.get("EM_RETRIEVAL_FILTER_AGENT", "0") == "1"
_AGENT_EVENT_TYPES = {"agent_thinking", "agent_probes"}

# Stratified cognition channel: when 1, agent THINKING/PROBE events are written
# to a SEPARATE EM partition+collection ("<base>_cog") instead of the main chat
# partition. The main chat probes never see them (so no displacement). At
# Phase 3 per-step exec, the cognition channel is probed independently with its
# own K-budget and surfaced as a labeled "YOUR PRIOR REASONING" prompt section.
# Tests the user's "different collections per modality" architectural fix.
# Implies EM_INGEST_THINKING=1.
EM_COGNITION_CHANNEL = os.environ.get("EM_COGNITION_CHANNEL", "0") == "1"
COGNITION_PROBE_K = int(os.environ.get("COGNITION_PROBE_K", "2"))


# Base timestamp for ingested agent events. Mirrors the `base_ts` and
# `_turn_ts` formula used in mid_execution_eval.py (timestamp = base + 60s
# * turn_id) so the agent events are placed monotonically after any plant
# or distractor turn (which use turn_ids well below 100000).
_AGENT_INGEST_BASE_TS = datetime(2023, 1, 1, tzinfo=timezone.utc)


def _agent_turn_ts(turn_id: int) -> datetime:
    return _AGENT_INGEST_BASE_TS + timedelta(seconds=60 * turn_id)


async def _ingest_agent_round(
    memory,
    *,
    sid: str,
    raw_text: str,
    probes: list[str],
    round_n: int,
    turn_id_counter: int,
) -> None:
    """Encode the agent's THINKING block and emitted PROBE list back into EM.

    Two events are written:
      1. event_type="agent_thinking", body = raw_text (full assistant output;
         the THINKING block is naturally embedded in it, and for native
         reasoning mode the assistant output IS the thinking surface).
      2. event_type="agent_probes",   body = the probe lines joined.

    Both are written with speaker="self" and `turn_id` strictly later than
    any plant or distractor turn, so they appear "after" the chat history.
    """
    thinking_text = (raw_text or "").strip() or "(empty)"
    probes_text = "\n".join(probes) if probes else "(none)"

    thinking_event = Event(
        uuid=uuid4(),
        timestamp=_agent_turn_ts(turn_id_counter),
        body=Content(
            context=MessageContext(source="self"),
            items=[Text(text=thinking_text)],
        ),
        properties={
            "scenario_id": sid,
            "turn_id": turn_id_counter,
            "speaker": "self",
            "event_type": "agent_thinking",
            "round": round_n,
        },
    )
    probes_event = Event(
        uuid=uuid4(),
        timestamp=_agent_turn_ts(turn_id_counter + 1),
        body=Content(
            context=MessageContext(source="self"),
            items=[Text(text=probes_text)],
        ),
        properties={
            "scenario_id": sid,
            "turn_id": turn_id_counter + 1,
            "speaker": "self",
            "event_type": "agent_probes",
            "round": round_n,
        },
    )
    await memory.encode_events([thinking_event, probes_event])


# Helper: create a SECOND EM instance for the cognition channel of a scenario.
# Mirrors the collection/partition setup in mid_execution_eval.ingest_scenario
# but uses a "_cog" suffix on the collection/partition name. The cognition EM
# starts empty; agent THINKING/PROBE events are written here via
# _ingest_agent_round when EM_COGNITION_CHANNEL=1.
async def _create_cognition_memory(
    scenario: dict,
    *,
    vector_store,
    segment_store,
    embedder,
    overwrite: bool = True,
) -> EventMemory:
    sid = scenario["scenario_id"]
    base_name = _scenario_collection(sid)
    cog_name = f"{base_name}_cog"
    if len(cog_name) > 32:
        # Defensive: shouldn't happen given _scenario_collection's hash form.
        import hashlib as _h

        digest = _h.sha256((sid + "_cog").encode()).hexdigest()[:8]
        cog_name = f"{COLLECTION_PREFIX}_{digest}_c"
    if overwrite:
        await vector_store.delete_collection(namespace=NAMESPACE, name=cog_name)
        await segment_store.delete_partition(cog_name)
    cog_collection = await vector_store.open_or_create_collection(
        namespace=NAMESPACE,
        name=cog_name,
        config=VectorStoreCollectionConfig(
            vector_dimensions=embedder.dimensions,
            similarity_metric=embedder.similarity_metric,
            properties_schema=EventMemory.expected_vector_store_collection_schema(),
        ),
    )
    cog_partition = await segment_store.open_or_create_partition(cog_name)
    return EventMemory(
        EventMemoryParams(
            vector_store_collection=cog_collection,
            segment_store_partition=cog_partition,
            embedder=embedder,
            reranker=None,
            derive_sentences=False,
            max_text_chunk_length=500,
        )
    )


async def run_freelance_executor(
    scenario: dict,
    *,
    mode: str,
    openai_client,
    cache: _SimpleCache,
    memory=None,  # required only for "retrieve_revise_cue_aware"
    cognition_memory=None,  # used only when EM_COGNITION_CHANNEL=1
) -> dict:
    task_prompt = scenario["task_prompt"]

    if mode == "natural":
        system = NATURAL_FREELANCE_SYSTEM.format(task_prompt=task_prompt)
        user = "Begin. Write the PLAN section, then the EXECUTE section, then the END line."
        raw = await _llm(
            openai_client,
            system,
            user,
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:{mode}",
        )
        return parse_executor_response(raw)

    if mode == "cue_aware":
        system = CUE_AWARE_FREELANCE_SYSTEM.format(task_prompt=task_prompt)
        user = "Begin. Write the PLAN section, then the EXECUTE section, then the END line."
        raw = await _llm(
            openai_client,
            system,
            user,
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:{mode}",
        )
        return parse_executor_response(raw)

    if mode == "primed_cue_aware":
        system = PRIMED_CUE_AWARE_SYSTEM.format(task_prompt=task_prompt)
        user = "Begin. Write the PLAN section, then the EXECUTE section, then the END line."
        raw = await _llm(
            openai_client,
            system,
            user,
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:{mode}",
        )
        return parse_executor_response(raw)

    if mode == "spreading_activation_full":
        # === Spreading activation at BOTH planning AND execution ===
        if memory is None:
            raise ValueError("spreading_activation_full requires memory= kwarg")

        # ---- Phase 1: planning-time spreading (same as basic SA) ----
        MAX_ITERS = 8  # bumped from 6 to give deeper chains room to assemble
        PER_STEP_PROBE_ROUNDS = 2  # mid-step iterative probing
        K_PER_PROBE = 3
        accumulated: dict[int, str] = {}  # turn_id -> snippet
        prior_probes: list[str] = []
        # Counter for ingesting the agent's own outputs (THINKING + PROBE
        # list) back into EM. Starts at a safe offset above any plant or
        # distractor turn_id so monotonicity is preserved without a lookup.
        turn_id_counter = 100000

        # When SA_RERANK is on, route every probe through embedding+LLM rerank.
        # When EM_RETRIEVAL_FILTER_AGENT is on, request 2x the K and drop hits
        # whose event_type is in _AGENT_EVENT_TYPES, then truncate back to K.
        async def _saf_probe(query: str, K: int):
            if SA_RERANK:
                hits = await probe_with_rerank(
                    memory,
                    query,
                    K_initial=K * 2,
                    K_final=K,
                    openai_client=openai_client,
                    cache=cache,
                )
            elif EM_RETRIEVAL_FILTER_AGENT and EM_INGEST_THINKING:
                # Over-fetch then drop agent_thinking/agent_probes hits.
                raw_hits = await probe(memory, query, K * 3)
                hits = [h for h in raw_hits if h.event_type not in _AGENT_EVENT_TYPES][
                    :K
                ]
            else:
                hits = await probe(memory, query, K)
            return hits

        # Seed: probe with task_prompt itself.
        seed_hits = await _saf_probe(task_prompt, K_PER_PROBE * 2)
        seed_snippets_list: list[str] = []
        for h in seed_hits[:K_PER_PROBE]:
            if h.turn_id in accumulated:
                continue
            snippet = (h.formatted_text or h.text or "").replace("\n", " ").strip()
            snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
            accumulated[h.turn_id] = snippet
            seed_snippets_list.append(snippet)
        prior_probes.append(task_prompt)

        # Iterative spreading: either explicit-text-thinking via Chat
        # Completions multi-turn, or native server-side reasoning via the
        # Responses API + previous_response_id chain.
        if REASONING_MODE == "native":
            # Reasoning state is carried server-side; we only track the
            # previous_response_id and the new user delta each round.
            prev_response_id: str | None = None

            for it in range(MAX_ITERS):
                if it == 0:
                    user_msg = SPREADING_PROBE_USER_INITIAL_MT.format(
                        seed_snippets="\n".join(f"- {s}" for s in seed_snippets_list)
                        or "(none)",
                    )
                    system_msg: str | None = SPREADING_PROBE_SYSTEM_MT_NATIVE.format(
                        task_prompt=task_prompt,
                    )
                else:
                    # new_snippets_this_round / prior_probes set in prior iter.
                    user_msg = SPREADING_PROBE_USER_FOLLOWUP_MT.format(
                        round_n=it + 1,
                        new_snippets="\n".join(
                            f"- {s}" for s in new_snippets_this_round
                        ),
                        n_probes_total=len(prior_probes),
                    )
                    system_msg = None

                raw, prev_response_id = await _llm_responses(
                    openai_client,
                    user_msg,
                    system_msg,
                    prev_response_id,
                    cache=cache,
                    cache_tag=f"{EXECUTOR_MODEL}:saf_plan_native_round{it}",
                )
                if raw.strip().upper().startswith("STOP"):
                    break
                new_probes = [m.group(1).strip() for m in PROBE_LINE_RE.finditer(raw)]
                new_probes = [
                    p
                    for p in new_probes
                    if p and p.lower() not in {x.lower() for x in prior_probes}
                ]
                if not new_probes:
                    break
                hits_lists = await asyncio.gather(
                    *[_saf_probe(p, K_PER_PROBE * 2) for p in new_probes]
                )
                new_snippets_this_round = []
                for hits in hits_lists:
                    for h in hits[:K_PER_PROBE]:
                        if h.turn_id in accumulated:
                            continue
                        snip = (
                            (h.formatted_text or h.text or "")
                            .replace("\n", " ")
                            .strip()
                        )
                        snip = snip[:200] + ("..." if len(snip) > 200 else "")
                        accumulated[h.turn_id] = snip
                        new_snippets_this_round.append(snip)
                prior_probes.extend(new_probes)
                if EM_INGEST_THINKING:
                    target_em = (
                        cognition_memory
                        if (EM_COGNITION_CHANNEL and cognition_memory is not None)
                        else memory
                    )
                    await _ingest_agent_round(
                        target_em,
                        sid=scenario["scenario_id"],
                        raw_text=raw,
                        probes=new_probes,
                        round_n=it,
                        turn_id_counter=turn_id_counter,
                    )
                    turn_id_counter += 2
                if not new_snippets_this_round:
                    break  # saturation: no new turns surfaced
        else:
            # Multi-turn iterative spreading: a single growing conversation
            # across rounds. Agent's THINKING from prior rounds remains
            # visible in the message history — closer to human "where I left
            # off" continuity.
            mt_messages: list[dict] = [
                {
                    "role": "system",
                    "content": SPREADING_PROBE_SYSTEM_MT.format(
                        task_prompt=task_prompt
                    ),
                },
                {
                    "role": "user",
                    "content": SPREADING_PROBE_USER_INITIAL_MT.format(
                        seed_snippets="\n".join(f"- {s}" for s in seed_snippets_list)
                        or "(none)",
                    ),
                },
            ]

            for it in range(MAX_ITERS):
                raw = await _llm_multiturn(
                    openai_client,
                    mt_messages,
                    cache=cache,
                    cache_tag=f"{EXECUTOR_MODEL}:saf_plan_mt_round{it}",
                )
                mt_messages.append({"role": "assistant", "content": raw})
                if raw.strip().upper().startswith("STOP"):
                    break
                new_probes = [m.group(1).strip() for m in PROBE_LINE_RE.finditer(raw)]
                new_probes = [
                    p
                    for p in new_probes
                    if p and p.lower() not in {x.lower() for x in prior_probes}
                ]
                if not new_probes:
                    break
                hits_lists = await asyncio.gather(
                    *[_saf_probe(p, K_PER_PROBE * 2) for p in new_probes]
                )
                new_snippets_this_round: list[str] = []
                for hits in hits_lists:
                    for h in hits[:K_PER_PROBE]:
                        if h.turn_id in accumulated:
                            continue
                        snip = (
                            (h.formatted_text or h.text or "")
                            .replace("\n", " ")
                            .strip()
                        )
                        snip = snip[:200] + ("..." if len(snip) > 200 else "")
                        accumulated[h.turn_id] = snip
                        new_snippets_this_round.append(snip)
                prior_probes.extend(new_probes)
                if EM_INGEST_THINKING:
                    target_em = (
                        cognition_memory
                        if (EM_COGNITION_CHANNEL and cognition_memory is not None)
                        else memory
                    )
                    await _ingest_agent_round(
                        target_em,
                        sid=scenario["scenario_id"],
                        raw_text=raw,
                        probes=new_probes,
                        round_n=it,
                        turn_id_counter=turn_id_counter,
                    )
                    turn_id_counter += 2
                if not new_snippets_this_round:
                    break  # saturation: no new turns surfaced
                mt_messages.append(
                    {
                        "role": "user",
                        "content": SPREADING_PROBE_USER_FOLLOWUP_MT.format(
                            round_n=it + 2,
                            new_snippets="\n".join(
                                f"- {s}" for s in new_snippets_this_round
                            ),
                            n_probes_total=len(prior_probes),
                        ),
                    }
                )

        plan_context_block = (
            "\n".join(f"- {s}" for s in accumulated.values())
            or "(memory returned no relevant snippets)"
        )

        # ---- Phase 2: generate plan only (no execution yet) ----
        plan_sys = SPREADING_PLAN_ONLY_SYSTEM.format(
            task_prompt=task_prompt,
            full_context=plan_context_block,
        )
        plan_raw = await _llm(
            openai_client,
            plan_sys,
            "Write the plan now.",
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:saf_plan_only",
        )
        plan_items = [
            (int(m.group(1)), m.group(2).strip())
            for m in STEP_LABEL_RE.finditer(plan_raw)
        ]

        # ---- Phase 3: per-step execution-time spreading ----
        exec_accumulated: dict[int, str] = {}  # NEW snippets surfaced at exec
        exec_steps: list[dict] = []
        prior_outputs_text = ""

        for step_id, step_label in plan_items:
            # Mid-step iterative probing: do up to PER_STEP_PROBE_ROUNDS
            # cue-gen-then-probe rounds before writing the step content.
            # Round 2+ sees the snippets from round 1 in exec_context, so the
            # agent can probe for what's still missing (chain assembly).
            step_step_snippets: list[str] = []
            all_step_cues: list[str] = []
            primary_cue = ""

            # Cognition channel: probe Phase-1 reasoning by similarity to the
            # current step's label. Surfaced as a SEPARATELY LABELED prompt
            # section, never mixed with chat hits in retrieval ranking. Only
            # populated when the cognition channel is on AND has events.
            cognition_context_block = "(no prior reasoning surfaced)"
            if EM_COGNITION_CHANNEL and cognition_memory is not None:
                try:
                    cog_hits = await probe(
                        cognition_memory, step_label, COGNITION_PROBE_K
                    )
                except Exception:
                    cog_hits = []
                if cog_hits:
                    cog_lines: list[str] = []
                    for h in cog_hits[:COGNITION_PROBE_K]:
                        txt = (
                            (h.formatted_text or h.text or "")
                            .replace("\n", " ")
                            .strip()
                        )
                        txt = txt[:300] + ("..." if len(txt) > 300 else "")
                        cog_lines.append(f"- {txt}")
                    if cog_lines:
                        cognition_context_block = "\n".join(cog_lines)

            for probe_round in range(PER_STEP_PROBE_ROUNDS):
                cue_sys = SPREADING_EXEC_STEP_SYSTEM.format(
                    task_prompt=task_prompt,
                    step_id=step_id,
                    step_label=step_label,
                    plan="\n".join(f"[STEP {sid}] {lbl}" for sid, lbl in plan_items),
                    plan_context=plan_context_block,
                    cognition_context=cognition_context_block,
                    exec_context="\n".join(f"- {s}" for s in exec_accumulated.values())
                    or "(none yet)",
                    prior_outputs=prior_outputs_text or "(none yet)",
                )
                cue_raw = await _llm(
                    openai_client,
                    cue_sys,
                    "Decide whether to probe; emit CUE lines.",
                    cache=cache,
                    cache_tag=f"{EXECUTOR_MODEL}:saf_exec_cue_step{step_id}_r{probe_round}",
                )
                round_cues_raw = [
                    m.group(1).strip() for m in CUE_LINE_RE.finditer(cue_raw)
                ]
                round_cues = [
                    c
                    for c in round_cues_raw
                    if c
                    and c.lower() not in ("none", "(none)", "n/a")
                    and c.lower() not in {x.lower() for x in all_step_cues}
                ]
                if not round_cues:
                    break  # agent has nothing more to probe for this step
                all_step_cues.extend(round_cues)
                if probe_round == 0 and round_cues:
                    primary_cue = round_cues[0]

                step_hits_lists = await asyncio.gather(
                    *[_saf_probe(c, K_PER_PROBE * 2) for c in round_cues]
                )
                added_this_round = 0
                for hits in step_hits_lists:
                    for h in hits[:K_PER_PROBE]:
                        if h.turn_id in accumulated or h.turn_id in exec_accumulated:
                            continue
                        snip = (
                            (h.formatted_text or h.text or "")
                            .replace("\n", " ")
                            .strip()
                        )
                        snip = snip[:200] + ("..." if len(snip) > 200 else "")
                        exec_accumulated[h.turn_id] = snip
                        step_step_snippets.append(snip)
                        added_this_round += 1
                if added_this_round == 0:
                    break  # saturated for this step

            step_cues = all_step_cues

            # Build full context for the writer call.
            full_ctx_lines: list[str] = []
            for s in accumulated.values():
                full_ctx_lines.append(f"- {s}")
            for s in exec_accumulated.values():
                full_ctx_lines.append(f"- {s}")
            full_ctx = "\n".join(full_ctx_lines) or "(no context)"

            # Write step content.
            write_sys = SPREADING_EXEC_WRITE_SYSTEM.format(
                task_prompt=task_prompt,
                step_id=step_id,
                step_label=step_label,
                plan="\n".join(f"[STEP {sid}] {lbl}" for sid, lbl in plan_items),
                full_context=full_ctx,
                cognition_context=cognition_context_block,
                prior_outputs=prior_outputs_text or "(none yet)",
            )
            content = await _llm(
                openai_client,
                write_sys,
                f"Write step {step_id}'s content now.",
                cache=cache,
                cache_tag=f"{EXECUTOR_MODEL}:saf_exec_write_step{step_id}",
            )
            content = content.strip()

            exec_steps.append(
                {
                    "step_id": step_id,
                    "content": content,
                    "cue": primary_cue,
                    "all_cues": step_cues,
                    "n_new_snippets_this_step": len(step_step_snippets),
                }
            )
            prior_outputs_text += f"\n--- STEP {step_id} ---\n{content}\n"

        # Format response in shape parse_executor_response would have produced.
        # Preserves all_cues so scoring can pick alternative cue-text strategies.
        return {
            "plan": [{"step_id": sid, "label": lbl} for sid, lbl in plan_items],
            "steps": [
                {
                    "step_id": s["step_id"],
                    "content": s["content"],
                    "cue": s["cue"],
                    "all_cues": s.get("all_cues", []),
                }
                for s in exec_steps
            ],
            "raw": plan_raw + "\n\n" + prior_outputs_text,
            "spreading_full_meta": {
                "n_planning_iterations": len(prior_probes) - 1,
                "n_planning_snippets": len(accumulated),
                "n_exec_new_snippets": len(exec_accumulated),
                "n_steps": len(exec_steps),
                "step_cue_counts": [s["all_cues"] for s in exec_steps],
            },
        }

    if mode == "spreading_activation_cue_aware":
        if memory is None:
            raise ValueError("spreading_activation_cue_aware requires memory= kwarg")
        # Iterative loop: probe → think → next probes → ... → plan + execute.
        MAX_ITERS = 6
        K_PER_PROBE = 3
        accumulated_turns: dict[int, str] = {}  # turn_id -> snippet
        prior_probes: list[str] = []

        # Iteration 1 seeded with the task prompt itself.
        seed_hits = await probe(memory, task_prompt, K_PER_PROBE * 2)
        for h in seed_hits[:K_PER_PROBE]:
            if h.turn_id in accumulated_turns:
                continue
            snippet = (h.formatted_text or h.text or "").replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            accumulated_turns[h.turn_id] = snippet
        prior_probes.append(task_prompt)

        for it in range(MAX_ITERS):
            ctx_lines = [f"- {s}" for s in accumulated_turns.values()]
            accumulated_context = "\n".join(ctx_lines) if ctx_lines else "(nothing yet)"
            prior_block = "\n".join(f"- {p[:120]}" for p in prior_probes)
            sys = SPREADING_PROBE_SYSTEM.format(
                task_prompt=task_prompt,
                accumulated_context=accumulated_context,
                prior_probes=prior_block,
            )
            user = "Generate next probes (or STOP)."
            raw = await _llm(
                openai_client,
                sys,
                user,
                cache=cache,
                cache_tag=f"{EXECUTOR_MODEL}:sa_probe_iter{it}",
            )
            if raw.strip().upper().startswith("STOP"):
                break
            new_probes = [m.group(1).strip() for m in PROBE_LINE_RE.finditer(raw)]
            new_probes = [
                p
                for p in new_probes
                if p and p.lower() not in {x.lower() for x in prior_probes}
            ]
            if not new_probes:
                break
            # Run all new probes in parallel.
            hits_lists = await asyncio.gather(
                *[probe(memory, p, K_PER_PROBE * 2) for p in new_probes]
            )
            n_new_turns = 0
            for hits in hits_lists:
                for h in hits[:K_PER_PROBE]:
                    if h.turn_id in accumulated_turns:
                        continue
                    snippet = (
                        (h.formatted_text or h.text or "").replace("\n", " ").strip()
                    )
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    accumulated_turns[h.turn_id] = snippet
                    n_new_turns += 1
            prior_probes.extend(new_probes)
            if n_new_turns == 0:
                break  # saturation: no new turns surfaced

        # Final pass: plan + execute with accumulated context.
        full_ctx = "\n".join(f"- {s}" for s in accumulated_turns.values())
        final_system = SPREADING_PLAN_EXECUTE_SYSTEM.format(
            task_prompt=task_prompt,
            full_context=full_ctx or "(memory returned no relevant snippets)",
        )
        final_user = "Begin. PLAN, then EXECUTE, then END."
        raw_final = await _llm(
            openai_client,
            final_system,
            final_user,
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:sa_plan_execute",
        )
        parsed = parse_executor_response(raw_final)
        parsed["spreading_meta"] = {
            "n_iterations": len(prior_probes) - 1,  # excluding seed
            "n_total_probes": len(prior_probes),
            "n_unique_snippets": len(accumulated_turns),
            "all_probes": [p[:120] for p in prior_probes],
        }
        return parsed

    if mode == "retrieve_revise_cue_aware":
        if memory is None:
            raise ValueError("retrieve_revise_cue_aware requires memory= kwarg")
        # Pass 1: draft plan from task_prompt only.
        draft_system = CRITIC_DRAFT_SYSTEM.format(task_prompt=task_prompt)
        draft_user = "Draft the plan now."
        draft_plan = await _llm(
            openai_client,
            draft_system,
            draft_user,
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:rr_draft",
        )

        # Probe each plan item against EM, collect top-K turns, dedupe.
        plan_items = [m.group(2).strip() for m in STEP_LABEL_RE.finditer(draft_plan)]
        if not plan_items:
            plan_items = [task_prompt]
        K_per_item = 3
        all_lists = await asyncio.gather(
            *[probe(memory, item, K_per_item * 2) for item in plan_items]
        )
        seen_tids: set[int] = set()
        ctx_lines: list[str] = []
        for hits in all_lists:
            for h in hits[:K_per_item]:
                if h.turn_id in seen_tids:
                    continue
                seen_tids.add(h.turn_id)
                snippet = (h.formatted_text or h.text or "").replace("\n", " ").strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                ctx_lines.append(f"- {snippet}")
        memory_context = (
            "\n".join(ctx_lines)
            if ctx_lines
            else "(memory returned no relevant snippets)"
        )

        # Pass 2: revise plan + execute with memory context in scope.
        revise_system = RETRIEVE_REVISE_AND_EXECUTE_SYSTEM.format(
            task_prompt=task_prompt,
            draft_plan=draft_plan,
            memory_context=memory_context,
        )
        revise_user = "Begin. Write the REVISED PLAN section, then EXECUTE, then END."
        raw = await _llm(
            openai_client,
            revise_system,
            revise_user,
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:rr_revise_execute",
        )
        parsed = parse_executor_response(raw)
        parsed["retrieve_revise_meta"] = {
            "draft_plan": draft_plan[:600],
            "n_plan_items_probed": len(plan_items),
            "n_unique_snippets": len(ctx_lines),
            "memory_context_preview": "\n".join(ctx_lines[:8])[:600],
        }
        return parsed

    if mode == "critic_cue_aware":
        # Pass 1: draft plan only.
        draft_system = CRITIC_DRAFT_SYSTEM.format(task_prompt=task_prompt)
        draft_user = "Draft the plan now."
        draft_plan = await _llm(
            openai_client,
            draft_system,
            draft_user,
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:critic_draft",
        )

        # Pass 2: critic identifies missing sub-decisions.
        critic_prompt = CRITIC_REVIEW_PROMPT.format(
            task_prompt=task_prompt,
            draft_plan=draft_plan,
        )
        critic_findings = await _llm(
            openai_client,
            "You are a senior reviewer auditing plans for completeness.",
            critic_prompt,
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:critic_review",
        )

        # Pass 3: revise + execute.
        revise_system = CRITIC_REVISE_AND_EXECUTE_SYSTEM.format(
            task_prompt=task_prompt,
            draft_plan=draft_plan,
            critic_findings=critic_findings,
        )
        revise_user = "Begin. Write the REVISED PLAN section, then EXECUTE, then END."
        raw = await _llm(
            openai_client,
            revise_system,
            revise_user,
            cache=cache,
            cache_tag=f"{EXECUTOR_MODEL}:critic_revise_execute",
        )
        parsed = parse_executor_response(raw)
        parsed["critic_meta"] = {
            "draft_plan": draft_plan[:600],
            "critic_findings": critic_findings[:600],
        }
        return parsed

    raise ValueError(f"Unknown mode: {mode}")


# --------------------------------------------------------------------------
# Coverage judge
# --------------------------------------------------------------------------


COVERAGE_JUDGE_PROMPT = """\
You are evaluating whether an executor agent's transcript addresses a \
specific sub-decision that a competent worker would have made for this task.

GOLD SUB-DECISION (a sub-step a competent worker should have addressed):
"{decision_text}"

GOLD FACT (a past-context fact that should have informed this sub-decision; \
you don't need to check whether the agent USED the fact, only whether the \
agent's transcript made a concrete decision in this area):
"{plant_text}"

AGENT'S FULL TRANSCRIPT (PLAN + EXECUTE sections):
---
{transcript}
---

Question 1: Did the agent address the gold sub-decision area in their \
transcript? "Addressed" means: somewhere in PLAN or EXECUTE the agent made \
a concrete choice or wrote content directly relevant to this sub-decision \
area. Whether they used the gold fact is NOT what's being judged here — \
only whether the decision area was covered at all.

Question 2: If yes, which `[STEP N]` label does it primarily correspond to? \
Use the integer step number from the agent's plan / execute sections. If \
the agent's transcript split the decision across multiple steps, pick the \
one that addresses it most directly. If the agent addressed it but didn't \
attach a clear STEP label, return "no_step_label".

Output ONLY a JSON object, no prose:

{{"addressed": true|false, "step_label": <integer> | "no_step_label" | null, \
"evidence_quote": "<short quote (<=140 chars) from the transcript that \
addresses the decision, or empty string>"}}
"""


def parse_judge_response(text: str) -> dict | None:
    """Extract JSON object from judge response, tolerating fences."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(t[s : e + 1])
    except Exception:
        return None


async def judge_coverage(
    *,
    transcript: str,
    decision_text: str,
    plant_text: str,
    openai_client,
    cache: _SimpleCache,
) -> dict:
    """Returns {addressed: bool, step_label: int|str|None, evidence_quote: str}.

    On parse failure returns {addressed: False, step_label: None, evidence_quote: ""}.
    """
    prompt = COVERAGE_JUDGE_PROMPT.format(
        transcript=transcript,
        decision_text=decision_text,
        plant_text=plant_text,
    )
    cached = cache.get(JUDGE_MODEL, prompt)
    if cached is None:
        resp = await openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low",
        )
        cached = resp.choices[0].message.content or ""
        cache.put(JUDGE_MODEL, prompt, cached)
    parsed = parse_judge_response(cached) or {}
    return {
        "addressed": bool(parsed.get("addressed", False)),
        "step_label": parsed.get("step_label"),
        "evidence_quote": str(parsed.get("evidence_quote", ""))[:200],
        "raw": cached,
    }


# --------------------------------------------------------------------------
# Per-scenario orchestration
# --------------------------------------------------------------------------


async def run_scenario_e2(
    scenario: dict,
    locomo_segments: dict,
    speakers_map: dict,
    *,
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
    openai_client,
    executor_cache: _SimpleCache,
    judge_cache: _SimpleCache,
    K_list: list[int],
    modes: list[str],  # subset of ["natural", "cue_aware"]
    overwrite: bool = True,
) -> dict:
    sid = scenario["scenario_id"]
    base_conv = scenario["base_conversation"]
    locomo_turns = locomo_segments[base_conv]
    speakers = speakers_map.get(base_conv) or {}

    extra_distractor_runs = []
    for extra_conv in scenario.get("extra_base_conversations") or []:
        extra_distractor_runs.append(
            (
                locomo_segments[extra_conv],
                speakers_map.get(extra_conv) or {},
            )
        )

    # Ingest EM (reuse E0).
    t0 = time.monotonic()
    memory, ingest_info = await ingest_scenario(
        scenario,
        locomo_turns,
        speakers,
        vector_store=vector_store,
        segment_store=segment_store,
        embedder=embedder,
        overwrite=overwrite,
        extra_distractor_runs=extra_distractor_runs or None,
    )
    ingest_time = time.monotonic() - t0

    # Optional second EM partition for the agent's cognition channel
    # (THINKING/PROBE events). Lives in its own collection + partition so chat
    # retrieval and cognition retrieval are independently scored.
    cognition_memory = None
    if EM_COGNITION_CHANNEL:
        cognition_memory = await _create_cognition_memory(
            scenario,
            vector_store=vector_store,
            segment_store=segment_store,
            embedder=embedder,
            overwrite=overwrite,
        )

    plants_by_id = {
        p["plant_id"]: p for p in scenario["preamble_turns"] if p.get("plant_id")
    }

    # ---- Run executor in each requested mode + judge each gold sub-decision ----
    K_max = max(K_list)
    per_mode: dict[str, dict] = {}

    for mode in modes:
        executor_out = await run_freelance_executor(
            scenario,
            mode=mode,
            openai_client=openai_client,
            cache=executor_cache,
            memory=memory,
            cognition_memory=cognition_memory,
        )
        executor_cache.save()
        steps_by_id = {s["step_id"]: s for s in executor_out["steps"]}

        # Pre-fetch judges in parallel (only over non-no-op gold steps —
        # no-op steps are coverage-irrelevant and have no plant).
        gold_steps = [
            s for s in scenario["subdecision_script"] if s.get("gold_plant_ids")
        ]

        async def _judge_one(gold_step):
            # Build representative plant text from the FIRST gold plant only —
            # passing all of them inflates the prompt and conflates judges.
            first_plant = plants_by_id.get(gold_step["gold_plant_ids"][0])
            plant_text = first_plant["text"] if first_plant else ""
            judgement = await judge_coverage(
                transcript=executor_out["raw"],
                decision_text=gold_step["decision_text"],
                plant_text=plant_text,
                openai_client=openai_client,
                cache=judge_cache,
            )
            return gold_step, judgement

        judgements = await asyncio.gather(*[_judge_one(g) for g in gold_steps])
        judge_cache.save()

        # ---- Score each gold sub-decision ----
        per_gold: list[dict] = []
        for gold_step, judgement in judgements:
            entry: dict = {
                "gold_step_id": gold_step["step_id"],
                "gold_decision_text": gold_step["decision_text"],
                "gold_plant_ids": gold_step["gold_plant_ids"],
                "addressed": judgement["addressed"],
                "judge_step_label": judgement["step_label"],
                "judge_evidence_quote": judgement["evidence_quote"],
            }
            for K in K_list:
                # Coverage-failure: counts as 0 retrieval (the agent never
                # surfaced the sub-decision so retrieval can't have helped).
                entry[f"triggered_recall_full@{K}"] = 0.0
                entry[f"recall_given_covered@{K}"] = None

            if judgement["addressed"]:
                # Try to identify which agent step covered it. If the judge
                # returned an integer step label that exists in our parsed
                # steps, use that step's cue/content. If not, fall back to
                # the entire transcript (worst case).
                label = judgement["step_label"]
                step_rec = steps_by_id.get(label) if isinstance(label, int) else None

                if step_rec:
                    if mode == "cue_aware" and step_rec["cue"]:
                        cue_text = step_rec["cue"]
                    else:
                        cue_text = step_rec["content"] or ""
                else:
                    # Agent addressed it but we couldn't pinpoint a step —
                    # use the evidence quote from the judge as the cue.
                    cue_text = judgement["evidence_quote"]

                entry["cue_used"] = cue_text[:200]
                entry["cue_source"] = (
                    "step_cue"
                    if (mode == "cue_aware" and step_rec and step_rec["cue"])
                    else ("step_content" if step_rec else "evidence_quote")
                )

                if cue_text.strip():
                    if EM_RETRIEVAL_FILTER_AGENT and EM_INGEST_THINKING:
                        raw_hits = await probe(memory, cue_text, K_max * 3)
                        hits = [
                            h
                            for h in raw_hits
                            if h.event_type not in _AGENT_EVENT_TYPES
                        ][:K_max]
                    else:
                        hits = await probe(memory, cue_text, K_max)
                    entry["top_hits"] = [
                        {
                            "rank": i + 1,
                            "turn_id": h.turn_id,
                            "plant_id": h.plant_id,
                            "score": round(h.score, 4),
                        }
                        for i, h in enumerate(hits[:K_max])
                    ]
                    found = {h.plant_id for h in hits if h.plant_id}
                    for K in K_list:
                        topK_found = {h.plant_id for h in hits[:K] if h.plant_id}
                        rec = sum(
                            1 for g in gold_step["gold_plant_ids"] if g in topK_found
                        ) / len(gold_step["gold_plant_ids"])
                        entry[f"recall_given_covered@{K}"] = rec
                        entry[f"triggered_recall_full@{K}"] = rec  # since covered=True
                else:
                    entry["top_hits"] = []

            per_gold.append(entry)

        # ---- Aggregate per-mode ----
        n_gold = len(per_gold)
        n_addressed = sum(1 for e in per_gold if e["addressed"])
        coverage_rate = n_addressed / n_gold if n_gold else 0.0

        agg: dict = {"coverage_rate": round(coverage_rate, 4)}
        for K in K_list:
            full = [e[f"triggered_recall_full@{K}"] for e in per_gold]
            cond = [
                e[f"recall_given_covered@{K}"]
                for e in per_gold
                if e[f"recall_given_covered@{K}"] is not None
            ]
            agg[f"triggered_recall_full@{K}"] = (
                round(sum(full) / len(full), 4) if full else 0.0
            )
            agg[f"recall_given_covered@{K}"] = (
                round(sum(cond) / len(cond), 4) if cond else None
            )

        per_mode[mode] = {
            "executor_plan": executor_out["plan"],
            "executor_n_steps": len(executor_out["steps"]),
            "per_gold": per_gold,
            "aggregates": agg,
        }

    return {
        "scenario_id": sid,
        "category": scenario.get("category", ""),
        "base_conversation": base_conv,
        "ingest_time_s": round(ingest_time, 2),
        "ingest_info": ingest_info,
        "K_list": K_list,
        "per_mode": per_mode,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--K", default="1,3,5,10")
    parser.add_argument("--modes", default="natural,cue_aware")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    K_list = sorted({int(x) for x in args.K.split(",") if x.strip()})
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {
        "natural",
        "cue_aware",
        "primed_cue_aware",
        "critic_cue_aware",
        "retrieve_revise_cue_aware",
        "spreading_activation_cue_aware",
        "spreading_activation_full",
    }
    for m in modes:
        if m not in valid_modes:
            raise SystemExit(f"Unknown mode {m!r}; valid: {sorted(valid_modes)}")

    scenarios = load_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s["scenario_id"] == args.scenario]
        if not scenarios:
            raise SystemExit(f"No scenario matched: {args.scenario}")

    locomo_segments = load_locomo_segments()
    speakers_map = load_speakers()

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sqlite_path = RESULTS_DIR / "eventmemory_mid_exec_e2.sqlite3"
    sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
    engine = create_async_engine(sql_url)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    executor_cache = _SimpleCache(EXECUTOR_CACHE_FILE)
    judge_cache = _SimpleCache(JUDGE_CACHE_FILE)

    results: list[dict] = []
    try:
        for scenario in scenarios:
            print(f"[run] {scenario['scenario_id']} (modes={modes})", flush=True)
            r = await run_scenario_e2(
                scenario,
                locomo_segments,
                speakers_map,
                vector_store=vector_store,
                segment_store=segment_store,
                embedder=embedder,
                openai_client=openai_client,
                executor_cache=executor_cache,
                judge_cache=judge_cache,
                K_list=K_list,
                modes=modes,
                overwrite=not args.no_overwrite,
            )
            results.append(r)
            for mode in modes:
                agg = r["per_mode"][mode]["aggregates"]
                cov = agg["coverage_rate"]
                full = agg.get("triggered_recall_full@5", "n/a")
                cond = agg.get("recall_given_covered@5", "n/a")
                print(f"  {mode}: coverage={cov} | full_R@5={full} | cond_R@5={cond}")
    finally:
        executor_cache.save()
        judge_cache.save()
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    out_path = (
        Path(args.out)
        if args.out
        else (RESULTS_DIR / f"mid_execution_eval_e2_{int(time.time())}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "K_list": K_list,
                "modes": modes,
                "n_scenarios": len(results),
                "scenarios": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")

    # ---- Cross-scenario summary ----
    print("\n=== Cross-scenario means ===")
    print(
        f"  {'mode':<22s} | {'coverage':>10s} | "
        + " | ".join(f"{('full_R@' + str(K)):>10s}" for K in K_list)
        + " | "
        + " | ".join(f"{('cond_R@' + str(K)):>10s}" for K in K_list)
    )
    for mode in modes:
        cov_vals = [r["per_mode"][mode]["aggregates"]["coverage_rate"] for r in results]
        cov = sum(cov_vals) / len(cov_vals) if cov_vals else 0.0
        line = f"  {mode:<22s} | {cov:>10.3f}"
        for K in K_list:
            vals = [
                r["per_mode"][mode]["aggregates"].get(f"triggered_recall_full@{K}")
                for r in results
                if r["per_mode"][mode]["aggregates"].get(f"triggered_recall_full@{K}")
                is not None
            ]
            line += f" | {sum(vals) / len(vals):>10.3f}" if vals else " |        n/a"
        for K in K_list:
            vals = [
                r["per_mode"][mode]["aggregates"].get(f"recall_given_covered@{K}")
                for r in results
                if r["per_mode"][mode]["aggregates"].get(f"recall_given_covered@{K}")
                is not None
            ]
            line += f" | {sum(vals) / len(vals):>10.3f}" if vals else " |        n/a"
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
