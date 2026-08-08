"""Shared harness v4_mtmsg_revisit_initprime — revisit + turn-1 initprime directive.

Forked from shared_harness_v4_mtmsg_revisit (iter11/15).

Iter16 hypothesis: combine the two validated mechanisms operating at
different decision-point timings.

  - Subdec_split (validated for bounded single-task benchmarks) primes
    the agent at TURN 1 via an implicit-constraint enumeration directive
    in USER_INITIAL. The agent enumerates distinct sub-decisions and
    probes for the implicit-constraint facts behind each one BEFORE
    crafting its first round of probes.
  - Revisit (validated for perpetual streams) acts LATE: a deterministic
    harness scheduler picks the OLDEST open commitment that has aged out
    of focus and injects an action-oriented revisit prompt.

This harness PREFIXES the existing USER_INITIAL_STREAM content with a
subdec_split-style turn-1 directive (adapted for stream context — the
agent doesn't yet know all sub-decisions but is asked to anticipate the
ones likely to arise across the stream), while keeping ALL revisit
machinery intact: open_sub_decisions tracking, revisit triggers,
action-oriented injections. SH_DISABLE_MOTIVATION=1 remains the
recommended config.

Mechanism summary:
  - Compaction still on (consolidates evicted span into facts + sub-decisions
    and updates the harness-level ``open_sub_decisions`` tracker).
  - Motivation OFF by default — recommended config is SH_DISABLE_MOTIVATION=1.
    Motivation generator may be re-enabled with SH_DISABLE_MOTIVATION=0 but
    is decoupled from the revisit mechanism.
  - Revisit scheduler fires every SH_REVISIT_PERIOD turns (default 4) where:
      (a) no stream message was delivered THIS turn, AND
      (b) ``open_sub_decisions`` contains at least one active commitment
          older than SH_REVISIT_MIN_AGE (default 3 turns).
    When triggered, the harness picks the OLDEST active commitment by
    ``opened_at_compaction_turn`` and APPENDS a synthesis turn-end addendum
    to the next user followup:
       ``[REVISIT TRIGGER: Commitment "<label>" opened turn <N>
         (currently <age> turns old). Either produce a STEP_OUTPUT
         addressing this commitment in your next turn, or explicitly mark
         it deferred/no-longer-needed and the harness will close it.]``
  - Parsing for revisit response (on the agent's NEXT turn):
      * STEP_OUTPUT for that commitment counts as a revisit-driven close.
      * ``DEFER: <label>`` or ``CLOSE: <label>`` markers explicitly close
        the commitment in the harness tracker.
      * Otherwise: no penalty, just record `no_action`.

Set SH_DISABLE_REVISIT=1 to ablate the scheduler.
Set SH_DISABLE_COMPACTION=1 to fall back to hard-truncate.
Set SH_DISABLE_MOTIVATION=1 (RECOMMENDED) to silence the motivation generator.

Usage:

    # Recommended smoke config (compaction on, motivation off, revisit on,
    # initprime baked into USER_INITIAL_STREAM)
    SH_SCENARIO_FILE=evaluation/associative_recall/data/perpetual_scenarios.json \\
        SH_RESULTS_SUBDIR=results_smoke_initprime_s0 \\
        SH_DB_SUFFIX=_initprime_s0 \\
        SH_DISABLE_MOTIVATION=1 \\
        uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_revisit_initprime/main.py \\
            --max-turns 120

Outputs per-scenario JSON files in this dir's results subdir (configurable via
``SH_RESULTS_SUBDIR``) plus a SUMMARY.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import openai
import tiktoken
from dotenv import load_dotenv

_AR_DIR = Path(__file__).resolve().parents[2]
if str(_AR_DIR) not in sys.path:
    sys.path.insert(0, str(_AR_DIR))

# Ensure metacog dir is importable so that `motivation.motivation` resolves
# without requiring the package install.
_METACOG_DIR = Path(__file__).resolve().parents[1]
if str(_METACOG_DIR) not in sys.path:
    sys.path.insert(0, str(_METACOG_DIR))

from memmachine_server.common.embedder.openai_embedder import (  # noqa: E402
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.vector_store.data_types import (  # noqa: E402
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (  # noqa: E402
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (  # noqa: E402
    Content,
    Event,
    MessageContext,
    Text,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (  # noqa: E402
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (  # noqa: E402
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from mid_execution_eval import (  # type: ignore  # noqa: E402
    RESULTS_DIR,
    probe,
)
from motivation.motivation import (  # noqa: E402
    MotivationState,
    initial_motivation_state,
    update_motivation,
)
from qdrant_client import AsyncQdrantClient  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent

# Results subdir is configurable so parallel ablation runs don't collide.
SH_RESULTS_SUBDIR = os.environ.get("SH_RESULTS_SUBDIR", "results")
RESULTS_OUT_DIR = THIS_DIR / SH_RESULTS_SUBDIR
RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = _AR_DIR / "data"

# Scenario file is configurable via env var (default: streaming v2 corpus).
SH_SCENARIO_FILE = os.environ.get("SH_SCENARIO_FILE", "")
if SH_SCENARIO_FILE:
    _scen_path = Path(SH_SCENARIO_FILE)
    if not _scen_path.is_absolute():
        # Try interpreting as relative to repo cwd first, fall back to data dir.
        if _scen_path.exists():
            DEFAULT_SCENARIOS_FILE = _scen_path.resolve()
        else:
            DEFAULT_SCENARIOS_FILE = (DATA_DIR / _scen_path.name).resolve()
    else:
        DEFAULT_SCENARIOS_FILE = _scen_path
else:
    DEFAULT_SCENARIOS_FILE = DATA_DIR / "streaming_scenarios.json"

ENV_PATH = _AR_DIR / ".env"
load_dotenv(ENV_PATH)

MODEL = "gpt-5-mini"
JUDGE_MODEL = "gpt-5-mini"
COMPACTION_MODEL = "gpt-5-mini"

MT_HARD_CAP = 10_000

MAX_TURNS = 28
RETRIEVE_K = 5
MAX_COMPLETION_TOKENS = 4500

STREAM_INTERVAL = 2

# Distinct namespaces from parent harnesses so parallel runs don't clobber
# each other's EM state.
NAMESPACE = "arc_em_revisit_initprime"
COLLECTION_PREFIX = "arc_rvip"

# Cap on premature-DONE nudges per run; if the agent insists past this many
# nudges, we honor the DONE to avoid infinite loops.
MAX_PREMATURE_DONE_NUDGES = 3

# ---------- Compaction config ----------

# Trigger compaction when at-turn-start tokens exceed this fraction of the cap.
SH_COMPACTION_THRESHOLD = float(os.environ.get("SH_COMPACTION_THRESHOLD", "0.85"))

# How many user/assistant pairs to keep intact at the END of the thread.
SH_COMPACTION_KEEP_RECENT = int(os.environ.get("SH_COMPACTION_KEEP_RECENT", "3"))

# When set to 1/true, compaction is disabled — fall back to hard-truncate.
SH_DISABLE_COMPACTION = os.environ.get("SH_DISABLE_COMPACTION", "0").lower() in (
    "1",
    "true",
    "yes",
)

COMPACTION_MAX_COMPLETION_TOKENS = 2500

# ---------- Motivation config ----------

# Periodic motivation update cadence (in agent turns).
SH_MOTIVATION_PERIOD = int(os.environ.get("SH_MOTIVATION_PERIOD", "8"))

# Cap on facts emitted by the compactor per event.
SH_COMPACTION_MAX_FACTS = int(os.environ.get("SH_COMPACTION_MAX_FACTS", "8"))

# When 1/true, motivation generator is not called and no directive is injected.
# Default is OFF: iter10 found motivation injection added noise without lift.
# This harness's primary mechanism is the deterministic revisit scheduler.
SH_DISABLE_MOTIVATION = os.environ.get("SH_DISABLE_MOTIVATION", "1").lower() in (
    "1",
    "true",
    "yes",
)

# ---------- Revisit scheduler config ----------

# Fire revisit every N turns (subject to eligibility).
SH_REVISIT_PERIOD = int(os.environ.get("SH_REVISIT_PERIOD", "4"))

# A commitment must be at least this old (in turns since opening) to be
# eligible for a revisit injection.
SH_REVISIT_MIN_AGE = int(os.environ.get("SH_REVISIT_MIN_AGE", "3"))

# When 1/true, the revisit scheduler is disabled (ablation).
SH_DISABLE_REVISIT = os.environ.get("SH_DISABLE_REVISIT", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Optional sqlite db filename suffix (lets parallel jobs avoid contention).
SH_DB_SUFFIX = os.environ.get("SH_DB_SUFFIX", "")

try:
    ENC = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(n_tokens(m.get("content", "") or "") for m in messages)


# ---------- Line parsers (extended from compaction harness) ----------

PROBE_LINE_RE = re.compile(r"^\s*PROBE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
DONE_LINE_RE = re.compile(r"^\s*DONE\s*$", re.MULTILINE | re.IGNORECASE)
STEP_OUTPUT_HEAD_RE = re.compile(
    r"^\s*STEP_OUTPUT\s*:\s*([^\n:\-]+?)(?:\s*[:\-]\s*(.*))?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
DIRECTIVE_LINE_RE = re.compile(
    r"^\s*(THINKING|PROBE|STEP_OUTPUT|DONE|DEFER|CLOSE)\b",
    re.MULTILINE | re.IGNORECASE,
)
THINKING_LINE_RE = re.compile(
    r"^\s*THINKING\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Revisit closure markers — agent can explicitly close a commitment without
# producing a fresh STEP_OUTPUT.
DEFER_LINE_RE = re.compile(
    r"^\s*DEFER\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)
CLOSE_LINE_RE = re.compile(
    r"^\s*CLOSE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)


def parse_step_outputs(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    heads = list(STEP_OUTPUT_HEAD_RE.finditer(raw))
    for m in heads:
        raw_label = (m.group(1) or "").strip()
        first_line_content = (
            (m.group(2) or "").strip() if m.lastindex and m.lastindex >= 2 else ""
        )
        body_start = m.end()
        next_directive = None
        for d in DIRECTIVE_LINE_RE.finditer(raw, pos=body_start):
            next_directive = d
            break
        body_end = next_directive.start() if next_directive else len(raw)
        body = raw[body_start:body_end].strip()
        full_content = first_line_content
        if body:
            full_content = (
                first_line_content + ("\n" if first_line_content else "") + body
            ).strip()
        out.append({"raw_label": raw_label, "content": full_content})
    return out


def parse_probes(raw: str) -> list[str]:
    return [m.group(1).strip() for m in PROBE_LINE_RE.finditer(raw or "")]


def has_done(raw: str) -> bool:
    return bool(DONE_LINE_RE.search(raw or ""))


def parse_thinking_blocks(raw: str) -> list[str]:
    """Extract THINKING line bodies (best-effort, single-line; multiline
    THINKING is rare enough we don't bother with full body extraction).
    """
    return [m.group(1).strip() for m in THINKING_LINE_RE.finditer(raw or "")]


def parse_defer_close(raw: str) -> tuple[list[str], list[str]]:
    """Return (defer_labels, close_labels) extracted from a raw turn output.

    Labels are the text after `DEFER:` / `CLOSE:` directives. Each label is
    stripped; empty entries are dropped. These markers let the agent
    explicitly close a commitment in the harness tracker without emitting
    a fresh STEP_OUTPUT.
    """
    defers = [m.group(1).strip() for m in DEFER_LINE_RE.finditer(raw or "")]
    closes = [m.group(1).strip() for m in CLOSE_LINE_RE.finditer(raw or "")]
    return ([d for d in defers if d], [c for c in closes if c])


# ---------- LLM ----------


async def llm_chat_messages(
    openai_client,
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
) -> str:
    resp = await openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=max(MAX_COMPLETION_TOKENS, max_completion_tokens),
    )
    return (resp.choices[0].message.content or "").strip()


# ---------- System prompt — extended from streaming v2 with revisit-aware
# DEFER/CLOSE markers documentation. ----------

SYSTEM_PROMPT = """\
You are a memory-augmented agent in an endless working loop with bounded \
working memory. The user is in an ongoing conversation with you and will \
keep delivering new context and new sub-tasks over many messages spread \
across this thread. There is no single up-front task statement. The work \
is open-ended.

You are running inside a continuous conversation thread. The thread is \
your context. When the thread crosses a token cap, the OLDEST user/assistant \
turns get hard-dropped (the system prompt stays). The harness also runs \
periodic COMPACTION events that consolidate evicted spans into a small set \
of long-term memory facts (probe-able via the memory tool).

This means: load-bearing facts you surface from memory survive ONLY as long \
as they remain in your recent reasoning. If old user context falls off the \
front of the thread, your only way back to it is via a memory PROBE.

You have a memory tool over PAST chat history. Crucially: in this streaming \
setup the memory contains EVERY past user message verbatim, including ones \
that have already scrolled out of your immediate context.

USER MESSAGES will arrive one at a time, embedded in the harness's followup \
messages tagged with `--- INCOMING USER STREAM MESSAGE (turn N) ---`. \
Treat each new stream message as: (a) potentially adding new context (a \
fact, a constraint, an offhand mention) you'll want to remember and probe \
on later, AND/OR (b) potentially asking you to make a sub-decision NOW \
based on the cumulative context so far.

When a stream message is just chatty/contextual (not asking for a \
deliverable), THINKING is appropriate — note what the user shared, do not \
emit a STEP_OUTPUT prematurely. When a stream message asks you to draft / \
plan / decide / write something, that IS a sub-decision request — emit \
STEP_OUTPUT for it after probing for relevant past context.

PROBE-GENERATION DISCIPLINES:
- **Implicit constraints**: when the user asks you to make a decision, \
ASK YOURSELF: "what relevant facts has the user shared in earlier messages \
of this same thread that the current ask doesn't repeat?" Probe for those \
specifically — by topic, by entity name, by likely keyword.
- **Optimistic cues**: when an abstract probe ("user's preferences") fails, \
also probe specific plausible values.
- **Implications and chains**: when a fact surfaces, ask what other fact \
must / probably exists alongside it, and probe for that too.

REVISIT TRIGGERS: the harness may periodically inject a `[REVISIT TRIGGER: \
Commitment "<label>" opened turn N (currently X turns old)...]` addendum at \
the end of a followup. When you see one, treat the named commitment as the \
TOP-PRIORITY action for the next turn. You have three valid responses:
  1. Produce a STEP_OUTPUT addressing the commitment (preferred when you \
have enough context).
  2. Emit `CLOSE: <label>` on its own line to mark the commitment finished \
or no-longer-needed if that's the right call.
  3. Emit `DEFER: <label>` on its own line to mark the commitment as \
deliberately deferred — the harness will close it and stop nagging.
Pick exactly one. Don't ignore the trigger.

OUTPUT FORMAT — each turn, emit free-text using these line patterns:

THINKING: <free text>
  Purpose: private reasoning, working notes, restated facts.
  Consequence: nothing system-side. The harness does not parse it.

PROBE: <retrieval query>
  Purpose: surface stored facts (the user's own past messages) that \
materially change the deliverable.
  Consequence: the harness runs each probe (top 5) and appends results in \
the next user followup. At most 4 PROBE lines per turn.

STEP_OUTPUT
  Format: `STEP_OUTPUT: <id>: <deliverable text>`. <id> can be a small \
integer or a short label (e.g., `STEP_OUTPUT: 3: ...` or `STEP_OUTPUT: \
menu_draft: ...`). A given id is a single sub-decision; re-emitting that \
id REVISES the previous version (latest replaces earlier).
  Purpose: a deliverable for ONE sub-decision triggered by a stream \
message OR a revisit trigger. Emit STEP_OUTPUT only when the user has \
actually asked for something concrete — not for context they've just shared.

DEFER: <label>
  Purpose: in response to a REVISIT TRIGGER, mark the named commitment as \
deliberately deferred — the harness will close it in its tracker.

CLOSE: <label>
  Purpose: in response to a REVISIT TRIGGER, mark the named commitment as \
finished or no-longer-needed — the harness will close it in its tracker.

DONE
  Purpose: signal you have nothing more to do AND no more stream messages \
are pending. Use sparingly — emit DONE only after the harness explicitly \
signals the stream is closed.

The harness will not deliver DONE prematurely; if you DONE before the \
stream is exhausted, you'll have shipped incomplete work.
"""


USER_INITIAL_STREAM = """\
This is the start of work. You have no probe results yet. Before \
crafting probes for this first user message, in your THINKING list the \
distinct sub-decisions you'll need to make as the work unfolds (you'll \
see them arise across the stream). For each, ask: what implicit \
constraints — facts the user has shared in this message or might share \
later — could materially shape the answer? Your probes should target \
both the obvious facts in THIS message AND retrievals of \
implicit-constraint facts you'd need to address each sub-decision.

This is the start of an ongoing conversation. New messages will arrive \
from the user over many subsequent turns. You will not see all the work \
up front.

--- INCOMING USER STREAM MESSAGE (turn {stream_turn}) ---
{stream_text}
---

For this first turn: in your THINKING, briefly note what kind of work this \
appears to be and what categories of constraint-facts you might want to \
listen for as the user shares more. Don't enumerate all sub-decisions yet — \
you don't have enough context. Don't emit a STEP_OUTPUT unless the user \
has explicitly asked for a concrete deliverable in the message above. \
PROBES on this first turn are optional — memory may already contain prior \
session context but for a brand-new thread it's likely empty for the \
relevant facts.
"""


USER_FOLLOWUP_PROBE_ONLY = """\
Turn {turn}. Probe results from memory:
---
{snippets}
---

Probes used this run: {n_probes_total}. New useful facts surfaced this \
turn: {new_facts_this_turn}. Consecutive turns with zero new useful \
facts: {consecutive_dry_turns}.

(No new stream message this turn. The user is letting you continue \
working with the context you have so far.)
"""


USER_FOLLOWUP_WITH_STREAM = """\
Turn {turn}. Probe results from memory:
---
{snippets}
---

Probes used this run: {n_probes_total}. New useful facts surfaced this \
turn: {new_facts_this_turn}. Consecutive turns with zero new useful \
facts: {consecutive_dry_turns}.

--- INCOMING USER STREAM MESSAGE (turn {stream_turn}) ---
{stream_text}
---

Process this new message: is it adding context, asking for a deliverable, \
or both? Respond accordingly.
"""


USER_FOLLOWUP_STREAM_CLOSED = """\
Turn {turn}. Probe results from memory:
---
{snippets}
---

Probes used this run: {n_probes_total}. New useful facts surfaced this \
turn: {new_facts_this_turn}. Consecutive turns with zero new useful \
facts: {consecutive_dry_turns}.

--- STREAM CLOSED ---
The user has no more incoming messages. Finish any pending sub-decisions, \
revise STEP_OUTPUTs as needed, then emit DONE.
"""


USER_FOLLOWUP_PREMATURE_DONE = """\
Turn {turn}. Probe results from memory:
---
{snippets}
---

Probes used this run: {n_probes_total}. New useful facts surfaced this \
turn: {new_facts_this_turn}. Consecutive turns with zero new useful \
facts: {consecutive_dry_turns}.

[a new user message is still expected — keep working on what you've heard \
so far and watch for the next message]

Do NOT emit DONE until the harness explicitly tells you the stream is \
closed. If you have nothing concrete to add right now, that's fine — emit \
THINKING about what you'd want to listen for, optionally PROBE for facts \
you may have missed, and wait.
"""


# ---------- Hard-truncate fallback ----------


def truncate_thread(messages: list[dict[str, str]], cap: int) -> dict[str, Any]:
    stats = {
        "dropped_pairs": 0,
        "dropped_msgs": 0,
        "tokens_before": messages_tokens(messages),
        "tokens_after": messages_tokens(messages),
    }
    while messages_tokens(messages) > cap and len(messages) >= 3:
        dropped = 0
        if (
            len(messages) >= 3
            and messages[1]["role"] == "user"
            and messages[2]["role"] == "assistant"
        ):
            del messages[1:3]
            dropped = 2
            stats["dropped_pairs"] += 1
        elif len(messages) >= 2 and messages[1]["role"] in ("user", "assistant"):
            del messages[1]
            dropped = 1
        else:
            break
        stats["dropped_msgs"] += dropped
        if dropped == 0:
            break
    stats["tokens_after"] = messages_tokens(messages)
    return stats


# ---------- Compaction ----------


COMPACTION_SCHEMA: dict[str, Any] = {
    "name": "compaction_extract",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "narrative": {
                "type": "string",
                "description": (
                    "A 3-5 sentence narrative summary of what happened in the "
                    "evicted span: what the user shared, what the agent "
                    "decided/drafted, what loose ends remain."
                ),
            },
            "facts": {
                "type": "array",
                "maxItems": 8,
                "description": (
                    "Concrete facts mentioned in the span. Each fact is a "
                    "single self-contained statement; include source turn "
                    "index when known. Extract ONLY NEW facts that materially "
                    "constrain a future decision; skip paraphrases / "
                    "background coloring. Aim for 3-7 items, hard cap 8."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source_turn": {"type": ["integer", "null"]},
                    },
                    "required": ["text"],
                },
            },
            "sub_decisions": {
                "type": "array",
                "description": (
                    "Sub-decisions opened, closed, or still active in the "
                    "evicted span."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "state": {
                            "type": "string",
                            "description": "One of: opened | closed | active",
                        },
                        "summary": {"type": "string"},
                    },
                    "required": ["label", "state"],
                },
            },
            "retrieval_cues": {
                "type": "array",
                "description": (
                    "Short retrieval cue strings (entity names, topic "
                    "keywords, dates) the agent should be ready to PROBE for "
                    "later in the run."
                ),
                "items": {"type": "string"},
            },
        },
        "required": ["narrative", "facts", "sub_decisions", "retrieval_cues"],
    },
}


COMPACTION_PROMPT = """\
You are consolidating an agent's working memory. The following transcript \
span is about to be evicted from the agent's context window. Your job is \
to extract a structured artifact that lets the agent recover load-bearing \
facts later via memory probes.

Extract:
  (a) a 3-5 sentence narrative summary of what happened in this span
  (b) a list of concrete facts mentioned (each with the original turn \
index when known)
  (c) sub-decisions that were opened/closed/still-active in this span
  (d) retrieval cues the agent should be ready to look up later

Guidelines:
- Extract ONLY facts that are NEW (not already in EventMemory or recent \
context) AND that materially constrain a future decision. Skip facts that \
are mere paraphrases or background coloring.
- Aim for 3-7 facts per compaction event, not exhaustive coverage. The \
schema enforces a hard cap of 8.
- A "fact" is one self-contained statement. Do not pack multiple unrelated \
claims into one fact.
- For sub_decisions.state, use "opened" (raised but not addressed), \
"closed" (delivered), or "active" (in progress).
- retrieval_cues should be short — entity names, topic keywords, dates — \
not full sentences.
- Do NOT include facts that were merely speculative or rejected by the \
user.

The span covers turns {turn_start} through {turn_end} of the agent's \
conversation thread.

TRANSCRIPT SPAN (each block is one role/content message; assistant turns \
are the agent's full output for that turn):

{span_text}

Output JSON matching the schema. Do not include any prose before or after \
the JSON object.
"""


def _parse_compaction_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(t[s : e + 1])
    except Exception:
        return None


def _format_span_for_compaction(span_messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for i, m in enumerate(span_messages):
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        chunks.append(f"--- [{i}] role={role} ---\n{content}")
    return "\n".join(chunks)


async def call_compactor(
    openai_client,
    span_messages: list[dict[str, str]],
    *,
    turn_start: int,
    turn_end: int,
) -> dict[str, Any]:
    span_text = _format_span_for_compaction(span_messages)
    prompt = COMPACTION_PROMPT.format(
        turn_start=turn_start, turn_end=turn_end, span_text=span_text
    )

    kwargs: dict[str, Any] = {
        "model": COMPACTION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": COMPACTION_MAX_COMPLETION_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": COMPACTION_SCHEMA,
        },
        "reasoning_effort": "low",
    }

    try:
        resp = await openai_client.chat.completions.create(**kwargs)
    except Exception as e:
        if "reasoning_effort" in str(e).lower() or "unsupported" in str(e).lower():
            kwargs.pop("reasoning_effort", None)
            try:
                resp = await openai_client.chat.completions.create(**kwargs)
            except Exception as e2:
                return {
                    "narrative": "",
                    "facts": [],
                    "sub_decisions": [],
                    "retrieval_cues": [],
                    "raw": "",
                    "error": f"compactor_call_failed: {e2!r}",
                }
        else:
            return {
                "narrative": "",
                "facts": [],
                "sub_decisions": [],
                "retrieval_cues": [],
                "raw": "",
                "error": f"compactor_call_failed: {e!r}",
            }

    raw = (resp.choices[0].message.content or "").strip()
    parsed = _parse_compaction_json(raw) or {}
    return {
        "narrative": str(parsed.get("narrative", "") or ""),
        "facts": list(parsed.get("facts") or []),
        "sub_decisions": list(parsed.get("sub_decisions") or []),
        "retrieval_cues": list(parsed.get("retrieval_cues") or []),
        "raw": raw,
        "error": None,
    }


def _identify_compaction_span(
    mt_messages: list[dict[str, str]],
    keep_recent_pairs: int,
) -> tuple[int, int]:
    if len(mt_messages) <= 1:
        return (1, 1)

    body_indices = list(range(1, len(mt_messages)))
    pairs: list[list[int]] = []
    i = 0
    while i < len(body_indices):
        idx_u = body_indices[i]
        if (
            i + 1 < len(body_indices)
            and mt_messages[idx_u]["role"] == "user"
            and mt_messages[body_indices[i + 1]]["role"] == "assistant"
        ):
            pairs.append([idx_u, body_indices[i + 1]])
            i += 2
        else:
            pairs.append([idx_u])
            i += 1

    if len(pairs) <= keep_recent_pairs:
        return (1, 1)

    eligible_pairs = pairs[:-keep_recent_pairs] if keep_recent_pairs > 0 else pairs[:]
    if not eligible_pairs:
        return (1, 1)

    n_to_compact = max(1, (len(eligible_pairs) + 1) // 2)
    span_pairs = eligible_pairs[:n_to_compact]

    flat = [idx for pair in span_pairs for idx in pair]
    start_idx = min(flat)
    end_idx = max(flat) + 1
    return (start_idx, end_idx)


def _stream_turns_in_span(span_messages: list[dict[str, str]]) -> list[int]:
    tag_re = re.compile(
        r"INCOMING USER STREAM MESSAGE \(turn (\d+)\)", re.IGNORECASE
    )
    turn_re = re.compile(r"^\s*Turn\s+(\d+)\.", re.IGNORECASE | re.MULTILINE)
    seen: set[int] = set()
    for m in span_messages:
        c = m.get("content", "") or ""
        for mm in tag_re.finditer(c):
            try:
                seen.add(int(mm.group(1)))
            except Exception:
                pass
        for mm in turn_re.finditer(c):
            try:
                seen.add(int(mm.group(1)))
            except Exception:
                pass
    return sorted(seen)


# ---------- Hit rendering ----------


def hit_to_id_text(hit, fallback_idx: int) -> tuple[str, str]:
    tid = getattr(hit, "turn_id", None)
    if tid is None or tid < 0:
        tid = 9000 + fallback_idx
    chat_id = f"chat-{tid}"
    content = (hit.formatted_text or hit.text or "").strip()
    return chat_id, content


# ---------- Online stream ingestion ----------


def _scenario_collection(scenario_id: str) -> str:
    safe = scenario_id.replace("-", "_")
    name = f"{COLLECTION_PREFIX}_{safe}"
    if len(name) <= 32:
        return name
    import hashlib as _h

    digest = _h.sha256(scenario_id.encode()).hexdigest()[:8]
    return f"{COLLECTION_PREFIX}_{digest}"


def _turn_ts(base: datetime, turn_id: int) -> datetime:
    return base + timedelta(seconds=60 * turn_id)


async def open_empty_memory(
    scenario: dict,
    *,
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
    overwrite: bool = True,
) -> tuple[EventMemory, dict]:
    sid = scenario["id"]
    collection_name = _scenario_collection(sid)
    partition_key = collection_name

    if overwrite:
        await vector_store.delete_collection(namespace=NAMESPACE, name=collection_name)
        await segment_store.delete_partition(partition_key)

    collection = await vector_store.open_or_create_collection(
        namespace=NAMESPACE,
        name=collection_name,
        config=VectorStoreCollectionConfig(
            vector_dimensions=embedder.dimensions,
            similarity_metric=embedder.similarity_metric,
            properties_schema=EventMemory.expected_vector_store_collection_schema(),
        ),
    )
    partition = await segment_store.open_or_create_partition(partition_key)

    memory = EventMemory(
        EventMemoryParams(
            vector_store_collection=collection,
            segment_store_partition=partition,
            embedder=embedder,
            reranker=None,
            derive_sentences=False,
            max_text_chunk_length=500,
        )
    )

    info = {
        "scenario_id": sid,
        "collection_name": collection_name,
        "n_messages": 0,
        "ingest_mode": "online_with_revisit",
    }
    return memory, info


class OnlineStreamIngester:
    def __init__(self, scenario: dict, memory: EventMemory) -> None:
        self.scenario = scenario
        self.memory = memory
        self.scenario_id = scenario["id"]
        self.base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
        self._messages_by_turn: dict[int, dict] = {
            int(m["turn"]): m for m in scenario["messages"]
        }
        self.ingested_turns: list[int] = []

    async def ingest(self, stream_turn: int) -> dict[str, Any]:
        msg = self._messages_by_turn.get(int(stream_turn))
        if msg is None:
            return {
                "stream_turn": stream_turn,
                "ingested": False,
                "reason": "no such turn",
            }
        if int(stream_turn) in self.ingested_turns:
            return {
                "stream_turn": stream_turn,
                "ingested": False,
                "reason": "already ingested",
            }

        ev = Event(
            uuid=uuid4(),
            timestamp=_turn_ts(self.base_ts, int(stream_turn)),
            body=Content(
                context=MessageContext(source="User"),
                items=[Text(text=msg["text"].strip())],
            ),
            properties={
                "scenario_id": self.scenario_id,
                "turn_id": int(stream_turn),
                "speaker": "User",
                "event_type": "stream_message",
                "plant_id": f"stream_turn_{stream_turn}",
                "from_turn": int(stream_turn),
            },
        )
        await self.memory.encode_events([ev])
        self.ingested_turns.append(int(stream_turn))
        return {
            "stream_turn": stream_turn,
            "ingested": True,
            "n_ingested_total": len(self.ingested_turns),
        }


def online_stream_ingester(
    scenario: dict, em_session: EventMemory
) -> OnlineStreamIngester:
    return OnlineStreamIngester(scenario, em_session)


async def write_compacted_fact(
    *,
    memory: EventMemory,
    scenario_id: str,
    fact_text: str,
    fact_source_turn: int | None,
    compaction_at_turn: int,
    turn_range_start: int,
    turn_range_end: int,
    base_ts: datetime,
) -> str:
    fact_id = f"compacted_fact_{uuid4().hex[:12]}"
    ev = Event(
        uuid=uuid4(),
        timestamp=_turn_ts(base_ts, int(compaction_at_turn)),
        body=Content(
            context=MessageContext(source="System"),
            items=[Text(text=fact_text.strip())],
        ),
        properties={
            "scenario_id": scenario_id,
            "turn_id": int(compaction_at_turn),
            "speaker": "System",
            "event_type": "compacted_fact",
            "plant_id": fact_id,
            "from_turn": (
                int(fact_source_turn)
                if fact_source_turn is not None
                else int(compaction_at_turn)
            ),
            "compaction_at_turn": int(compaction_at_turn),
            "compacted_from_turn_range": (
                f"{turn_range_start}-{turn_range_end}"
            ),
        },
    )
    await memory.encode_events([ev])
    return fact_id


# ---------- LLM-judge DP coverage ----------


DP_JUDGE_SCHEMA: dict[str, Any] = {
    "name": "dp_coverage_judgement",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "covered": {
                "type": "boolean",
                "description": "True iff the agent's STEP_OUTPUT text references / uses the gold required-fact content while addressing this sub-decision area.",
            },
            "evidence": {
                "type": "string",
                "description": "Short quote (<=160 chars) from the STEP_OUTPUT that demonstrates fact use, or an empty string if not covered.",
            },
        },
        "required": ["covered", "evidence"],
    },
}


DP_JUDGE_PROMPT = """\
You are judging whether a streaming-agent's deliverable correctly used a \
specific past-context fact when addressing a sub-decision.

GOLD SUB-DECISION (the area the agent should have addressed):
"{sub_decision}"

REQUIRED FACTS (each fact below is one the agent SHOULD have surfaced from \
memory and reflected in their deliverable):
{required_facts_block}

AGENT'S STEP_OUTPUT (the deliverable text under review):
---
{step_output_text}
---

Question: Does the STEP_OUTPUT above concretely reflect / honor the \
required fact(s)? "Reflect" means: the deliverable contains a choice or \
phrasing that is consistent with the fact AND would not have been chosen \
by an agent ignorant of the fact. Mere generic content that neither \
contradicts nor uses the fact does NOT count as covered.

If multiple facts are listed, ALL must be reflected for `covered: true`. \
If the agent reflected some but not all, mark `covered: false` and quote \
the strongest single piece of fact-aware evidence in `evidence`.

Output ONLY a JSON object matching this shape:
{{"covered": true|false, "evidence": "<short quote (<=160 chars) or empty>"}}
"""


def _format_required_facts_block(
    required_facts: list[str], gold_text_for_facts: dict[str, str]
) -> str:
    if not required_facts:
        return "(none — DP has no required facts)"
    lines: list[str] = []
    for fid in required_facts:
        gtext = gold_text_for_facts.get(fid, "").strip() or "(no gold text registered)"
        lines.append(f"- [{fid}] {gtext}")
    return "\n".join(lines)


def _parse_judge_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(t[s : e + 1])
    except Exception:
        return None


async def judge_dp_coverage(
    step_output_text: str,
    required_facts: list[str],
    gold_text_for_facts: dict[str, str],
    *,
    openai_client,
    sub_decision: str = "",
) -> dict[str, Any]:
    if not (step_output_text or "").strip():
        return {
            "covered": False,
            "evidence": "",
            "raw": "",
            "error": "empty_step_output",
        }

    prompt = DP_JUDGE_PROMPT.format(
        sub_decision=sub_decision or "(unspecified)",
        required_facts_block=_format_required_facts_block(
            required_facts, gold_text_for_facts
        ),
        step_output_text=step_output_text.strip(),
    )

    kwargs: dict[str, Any] = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 1500,
        "response_format": {"type": "json_schema", "json_schema": DP_JUDGE_SCHEMA},
        "reasoning_effort": "low",
    }

    try:
        resp = await openai_client.chat.completions.create(**kwargs)
    except Exception as e:
        if "reasoning_effort" in str(e).lower() or "unsupported" in str(e).lower():
            kwargs.pop("reasoning_effort", None)
            try:
                resp = await openai_client.chat.completions.create(**kwargs)
            except Exception as e2:
                return {
                    "covered": False,
                    "evidence": "",
                    "raw": "",
                    "error": f"judge_call_failed: {e2!r}",
                }
        else:
            return {
                "covered": False,
                "evidence": "",
                "raw": "",
                "error": f"judge_call_failed: {e!r}",
            }

    raw = (resp.choices[0].message.content or "").strip()
    parsed = _parse_judge_json(raw) or {}
    return {
        "covered": bool(parsed.get("covered", False)),
        "evidence": str(parsed.get("evidence", ""))[:240],
        "raw": raw,
    }


# ---------- Agent loop ----------


@dataclass
class TurnLog:
    turn: int
    raw_excerpt: str
    raw_full: str
    stream_msg_delivered_this_turn: int | None = None
    stream_msg_ingested_this_turn: int | None = None
    probes: list[str] = field(default_factory=list)
    n_hits: int = 0
    new_hit_ids: list[str] = field(default_factory=list)
    n_step_outs: int = 0
    step_outs_emitted: list[dict[str, Any]] = field(default_factory=list)
    done_emitted: bool = False
    premature_done: bool = False
    thread_tokens_before: int = 0
    thread_tokens_after_response: int = 0
    thread_tokens_after_truncate: int = 0
    truncate_dropped_pairs: int = 0
    truncate_dropped_msgs: int = 0
    new_facts_this_turn: int = 0
    compaction_fired_this_turn: bool = False
    motivation_updated_this_turn: bool = False
    motivation_state_after: str = ""
    motivation_intensity_after: float = 0.0
    n_active_sub_decisions: int = 0
    # Revisit-specific fields
    revisit_injected_after_this_turn: bool = False
    revisit_target_label: str = ""
    revisit_target_age: int = 0
    revisit_response_kind: str = ""  # one of: "", step_output, defer, close, no_action
    defers_emitted: list[str] = field(default_factory=list)
    closes_emitted: list[str] = field(default_factory=list)


def td_to_dict(t: TurnLog) -> dict[str, Any]:
    return {
        "turn": t.turn,
        "raw_excerpt": t.raw_excerpt,
        "raw_full": t.raw_full,
        "stream_msg_delivered_this_turn": t.stream_msg_delivered_this_turn,
        "stream_msg_ingested_this_turn": t.stream_msg_ingested_this_turn,
        "probes": t.probes,
        "n_hits": t.n_hits,
        "new_hit_ids": t.new_hit_ids,
        "n_step_outs": t.n_step_outs,
        "step_outs_emitted": t.step_outs_emitted,
        "done_emitted": t.done_emitted,
        "premature_done": t.premature_done,
        "thread_tokens_before": t.thread_tokens_before,
        "thread_tokens_after_response": t.thread_tokens_after_response,
        "thread_tokens_after_truncate": t.thread_tokens_after_truncate,
        "truncate_dropped_pairs": t.truncate_dropped_pairs,
        "truncate_dropped_msgs": t.truncate_dropped_msgs,
        "new_facts_this_turn": t.new_facts_this_turn,
        "compaction_fired_this_turn": t.compaction_fired_this_turn,
        "motivation_updated_this_turn": t.motivation_updated_this_turn,
        "motivation_state_after": t.motivation_state_after,
        "motivation_intensity_after": t.motivation_intensity_after,
        "n_active_sub_decisions": t.n_active_sub_decisions,
        "revisit_injected_after_this_turn": t.revisit_injected_after_this_turn,
        "revisit_target_label": t.revisit_target_label,
        "revisit_target_age": t.revisit_target_age,
        "revisit_response_kind": t.revisit_response_kind,
        "defers_emitted": t.defers_emitted,
        "closes_emitted": t.closes_emitted,
    }


def _build_gold_text_for_facts(scenario: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in scenario.get("ground_truth_facts") or []:
        fid = f.get("id")
        if fid:
            out[fid] = (f.get("text") or "").strip()
    return out


def _refresh_open_sub_decisions(
    open_sub_decisions: dict[str, dict[str, Any]],
    sub_decisions_from_compaction: list[dict[str, Any]],
    *,
    at_turn: int,
) -> dict[str, int]:
    """Apply this compaction's sub_decisions[*] to the harness-level open
    sub-decisions tracker.

    Mutates ``open_sub_decisions`` in place. Returns counters for tracing.
    """
    counters = {"opened": 0, "closed": 0, "active_kept": 0, "ignored": 0}
    for sd in sub_decisions_from_compaction or []:
        label = (sd.get("label") or "").strip()
        state = (sd.get("state") or "").strip().lower()
        summary = (sd.get("summary") or "").strip()
        if not label:
            counters["ignored"] += 1
            continue
        existing = open_sub_decisions.get(label)
        if state == "closed":
            if existing is not None:
                existing["state"] = "closed"
                existing["closed_at_turn"] = at_turn
            else:
                open_sub_decisions[label] = {
                    "state": "closed",
                    "summary": summary,
                    "opened_at_compaction_turn": at_turn,
                    "closed_at_turn": at_turn,
                }
            counters["closed"] += 1
        elif state == "opened":
            if existing is None:
                open_sub_decisions[label] = {
                    "state": "opened",
                    "summary": summary,
                    "opened_at_compaction_turn": at_turn,
                }
                counters["opened"] += 1
            else:
                if summary and not existing.get("summary"):
                    existing["summary"] = summary
                counters["active_kept"] += 1
        elif state == "active":
            if existing is None:
                open_sub_decisions[label] = {
                    "state": "active",
                    "summary": summary,
                    "opened_at_compaction_turn": at_turn,
                }
                counters["opened"] += 1
            else:
                if summary and not existing.get("summary"):
                    existing["summary"] = summary
                if existing.get("state") not in ("closed",):
                    existing["state"] = "active"
                counters["active_kept"] += 1
        else:
            counters["ignored"] += 1
    return counters


def _format_open_commitments_block(
    open_sub_decisions: dict[str, dict[str, Any]],
    *,
    at_turn: int,
) -> str:
    """Render the 'OPEN COMMITMENTS' line for the compaction injection."""
    parts: list[str] = []
    for label, info in open_sub_decisions.items():
        if info.get("state") == "closed":
            continue
        opened = info.get("opened_at_compaction_turn", at_turn)
        parts.append(f"{label} (opened turn {opened})")
    if not parts:
        return ""
    return "[OPEN COMMITMENTS: " + ", ".join(parts) + "]"


# ---------- Revisit scheduler ----------


def _select_revisit_target(
    open_sub_decisions: dict[str, dict[str, Any]],
    *,
    at_turn: int,
    min_age: int,
) -> tuple[str, int, dict[str, Any]] | None:
    """Pick the OLDEST active commitment older than ``min_age`` turns.

    Returns (label, age_in_turns, info) or None if nothing eligible.
    """
    eligible: list[tuple[int, str, dict[str, Any]]] = []
    for label, info in open_sub_decisions.items():
        if info.get("state") == "closed":
            continue
        opened_at = int(info.get("opened_at_compaction_turn", at_turn))
        age = max(0, at_turn - opened_at)
        if age < min_age:
            continue
        eligible.append((opened_at, label, info))
    if not eligible:
        return None
    eligible.sort(key=lambda t: (t[0], t[1]))
    opened_at, label, info = eligible[0]
    return label, max(0, at_turn - opened_at), info


def _format_revisit_addendum(
    label: str,
    age: int,
    info: dict[str, Any],
) -> str:
    """Build the deterministic revisit trigger string injected into the
    next followup. Principle-level — references the abstract commitment
    label only, no scenario specifics."""
    opened_at = info.get("opened_at_compaction_turn", "?")
    summary = (info.get("summary") or "").strip()
    summary_clause = f' (summary: "{summary}")' if summary else ""
    return (
        f'[REVISIT TRIGGER: Commitment "{label}" opened turn {opened_at} '
        f'(currently {age} turns old){summary_clause}. Either produce a '
        f"STEP_OUTPUT addressing this commitment in your next turn, or "
        f"emit `DEFER: {label}` / `CLOSE: {label}` on its own line — the "
        f"harness will close it.]"
    )


def _close_commitment(
    open_sub_decisions: dict[str, dict[str, Any]],
    label: str,
    *,
    at_turn: int,
    reason: str,
) -> bool:
    """Close a commitment in the tracker. Tolerates label variants:
    exact match, then case-insensitive, then substring."""
    if not label:
        return False
    if label in open_sub_decisions:
        rec = open_sub_decisions[label]
        if rec.get("state") == "closed":
            return False
        rec["state"] = "closed"
        rec["closed_at_turn"] = at_turn
        rec["closed_reason"] = reason
        return True
    # Case-insensitive match
    lkey = label.lower().strip()
    for k, rec in open_sub_decisions.items():
        if k.lower().strip() == lkey and rec.get("state") != "closed":
            rec["state"] = "closed"
            rec["closed_at_turn"] = at_turn
            rec["closed_reason"] = reason
            return True
    # Substring fallback (agent may quote a fragment of the label)
    for k, rec in open_sub_decisions.items():
        if rec.get("state") == "closed":
            continue
        if lkey and lkey in k.lower():
            rec["state"] = "closed"
            rec["closed_at_turn"] = at_turn
            rec["closed_reason"] = reason
            return True
    return False


async def maybe_compact(
    *,
    mt_messages: list[dict[str, str]],
    memory: EventMemory,
    openai_client,
    scenario_id: str,
    turn: int,
    base_ts: datetime,
    open_sub_decisions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if SH_DISABLE_COMPACTION:
        return None

    tokens_before = messages_tokens(mt_messages)
    threshold_tokens = SH_COMPACTION_THRESHOLD * MT_HARD_CAP
    if tokens_before <= threshold_tokens:
        return None

    start_idx, end_idx = _identify_compaction_span(
        mt_messages, SH_COMPACTION_KEEP_RECENT
    )
    if end_idx <= start_idx:
        return None

    span_messages = mt_messages[start_idx:end_idx]
    span_tokens = messages_tokens(span_messages)
    if not span_messages:
        return None

    stream_turns_in_span = _stream_turns_in_span(span_messages)
    turn_start = stream_turns_in_span[0] if stream_turns_in_span else 1
    turn_end = stream_turns_in_span[-1] if stream_turns_in_span else turn
    n_pair_msgs = len(span_messages)
    evicted_pair_count = sum(
        1 for m in span_messages if m.get("role") == "assistant"
    )

    compaction_result = await call_compactor(
        openai_client,
        span_messages,
        turn_start=turn_start,
        turn_end=turn_end,
    )

    facts = compaction_result.get("facts") or []
    narrative = compaction_result.get("narrative") or ""
    sub_decisions = compaction_result.get("sub_decisions") or []
    retrieval_cues = compaction_result.get("retrieval_cues") or []

    n_facts_dropped_by_cap = 0
    if len(facts) > SH_COMPACTION_MAX_FACTS:
        n_facts_dropped_by_cap = len(facts) - SH_COMPACTION_MAX_FACTS
        facts = facts[:SH_COMPACTION_MAX_FACTS]

    written_facts: list[dict[str, Any]] = []
    for f in facts:
        ftext = (f.get("text") or "").strip()
        if not ftext:
            continue
        src = f.get("source_turn")
        try:
            src_int = int(src) if src is not None else None
        except Exception:
            src_int = None
        try:
            fact_id = await write_compacted_fact(
                memory=memory,
                scenario_id=scenario_id,
                fact_text=ftext,
                fact_source_turn=src_int,
                compaction_at_turn=turn,
                turn_range_start=turn_start,
                turn_range_end=turn_end,
                base_ts=base_ts,
            )
        except Exception as e:
            written_facts.append(
                {
                    "fact_text": ftext,
                    "fact_from_turn": src_int,
                    "compaction_at_turn": turn,
                    "fact_id": None,
                    "error": f"em_write_failed: {e!r}",
                }
            )
            continue
        written_facts.append(
            {
                "fact_text": ftext,
                "fact_from_turn": src_int,
                "compaction_at_turn": turn,
                "fact_id": fact_id,
            }
        )

    sub_dec_counters: dict[str, int] = {
        "opened": 0,
        "closed": 0,
        "active_kept": 0,
        "ignored": 0,
    }
    if open_sub_decisions is not None:
        sub_dec_counters = _refresh_open_sub_decisions(
            open_sub_decisions, sub_decisions, at_turn=turn
        )

    sub_dec_lines = []
    for sd in sub_decisions:
        label = sd.get("label", "?")
        state = sd.get("state", "?")
        summ = sd.get("summary", "") or ""
        line = f"{label} [{state}]"
        if summ:
            line += f": {summ}"
        sub_dec_lines.append(line)
    sub_dec_block = "; ".join(sub_dec_lines) if sub_dec_lines else "(none)"
    cues_block = ", ".join(retrieval_cues) if retrieval_cues else "(none)"

    open_commitments_block = (
        _format_open_commitments_block(
            open_sub_decisions, at_turn=turn
        )
        if open_sub_decisions is not None
        else ""
    )

    summary_msg = (
        f"[COMPACTED MEMORY (covering turns {turn_start}-{turn_end}): "
        f"{narrative.strip()} "
        f"Key facts written to long-term memory ({len(written_facts)} fact(s)); "
        f"you can probe for them as needed. "
        f"Active sub-decisions (this span): {sub_dec_block}. "
        f"Open retrieval cues: {cues_block}.]"
    )
    if open_commitments_block:
        summary_msg = summary_msg + "\n" + open_commitments_block

    replacement = {"role": "user", "content": summary_msg}
    del mt_messages[start_idx:end_idx]
    mt_messages.insert(start_idx, replacement)

    tokens_after = messages_tokens(mt_messages)

    return {
        "at_turn": turn,
        "evicted_pair_count": evicted_pair_count,
        "evicted_msg_count": n_pair_msgs,
        "evicted_token_count": span_tokens,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "n_facts_written": len([w for w in written_facts if w.get("fact_id")]),
        "n_facts_failed": len([w for w in written_facts if w.get("error")]),
        "n_facts_dropped_by_cap": n_facts_dropped_by_cap,
        "narrative": narrative,
        "narrative_length": len(narrative),
        "sub_decisions": sub_decisions,
        "sub_decision_counters": sub_dec_counters,
        "retrieval_cues": retrieval_cues,
        "facts_written": written_facts,
        "stream_turns_in_span": stream_turns_in_span,
        "turn_start": turn_start,
        "turn_end": turn_end,
        "summary_msg": summary_msg,
        "open_commitments_block": open_commitments_block,
        "compactor_error": compaction_result.get("error"),
    }


# ---------- Motivation injection helpers (vestigial; on only if not disabled) ----------


def _format_motivation_prefix(state: MotivationState) -> str:
    intensity_str = f"{state.intensity:.2f}"
    directive = (state.drive_directive or "").strip()
    if not directive:
        directive = (
            f"Stay {state.state} about the most actionable open thread "
            f"and take one concrete next step now."
        )
    return (
        f"[CURRENT MOTIVATION: {state.state}, intensity {intensity_str}. "
        f"{directive}]"
    )


def _build_recent_activity_summary(
    *,
    last_compaction: dict[str, Any] | None,
    compaction_age_turns: int,
    recent_thinkings: list[str],
) -> str:
    if (
        last_compaction is not None
        and compaction_age_turns <= SH_MOTIVATION_PERIOD
        and last_compaction.get("narrative")
    ):
        return (
            f"(from compaction at turn {last_compaction.get('at_turn')}) "
            + str(last_compaction["narrative"]).strip()
        )

    if recent_thinkings:
        joined = " | ".join(t for t in recent_thinkings if t)
        return f"(recent THINKING blocks) {joined}" if joined else "(empty)"
    return "(empty)"


async def run_agent_loop(
    *,
    scenario: dict,
    memory,
    openai_client,
    stream_interval: int = STREAM_INTERVAL,
    max_turns: int = MAX_TURNS,
) -> dict[str, Any]:
    sid = scenario["id"]
    stream_messages: list[dict[str, Any]] = sorted(
        scenario["messages"], key=lambda m: m["turn"]
    )
    decision_points: list[dict[str, Any]] = scenario.get("decision_points") or []
    gold_text_for_facts = _build_gold_text_for_facts(scenario)

    ingester = online_stream_ingester(scenario, memory)
    base_ts = ingester.base_ts

    first_msg = stream_messages[0]
    await ingester.ingest(first_msg["turn"])

    stream_idx = 1
    stream_turns_delivered = first_msg["turn"]

    mt_messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_INITIAL_STREAM.format(
                stream_turn=first_msg["turn"], stream_text=first_msg["text"]
            ),
        },
    ]

    trace: list[TurnLog] = []
    step_outputs_by_id: dict[int, dict[str, Any]] = {}
    label_to_int: dict[str, int] = {}
    next_label_int = 1
    step_outputs_log: list[dict[str, Any]] = []
    n_probes_total = 0
    seen_chat_ids: set[str] = set()
    consecutive_dry_turns = 0
    stream_closed_announced = False
    premature_done_count = 0

    dp_coverage_log: list[dict[str, Any]] = []

    compaction_events: list[dict[str, Any]] = []
    compacted_facts: list[dict[str, Any]] = []

    open_sub_decisions: dict[str, dict[str, Any]] = {}

    # ---- Motivation tracking (vestigial unless re-enabled) ----
    current_motivation: MotivationState = initial_motivation_state(turn=0)
    motivation_events: list[dict[str, Any]] = []
    last_motivation_update_turn = 0
    last_compaction_turn: int | None = None
    pending_motivation_prefix: str | None = None
    recent_thinkings: list[str] = []

    turns_since_last_user_input = 0
    turns_since_last_completion = 0

    # ---- Revisit scheduler state ----
    # We track the LAST turn on which we INJECTED a revisit so we can:
    #   (a) space them out by SH_REVISIT_PERIOD
    #   (b) classify the NEXT turn's response (STEP_OUTPUT / DEFER / CLOSE / no_action)
    revisit_events: list[dict[str, Any]] = []
    # If a revisit was injected at end of turn T, the next agent turn (T+1)
    # is the one whose response should be classified. We carry the event dict
    # forward through this pointer so we can update response fields in-place
    # after the next turn's parse.
    pending_revisit_event: dict[str, Any] | None = None
    pending_revisit_target: str = ""
    last_revisit_turn: int = -10**9  # turn on which revisit was last injected
    n_commitments_closed_via_revisit = 0
    n_commitments_deferred_via_revisit = 0

    for turn in range(1, max_turns + 1):
        # ---- Compaction check at TURN START ----
        compaction_record = await maybe_compact(
            mt_messages=mt_messages,
            memory=memory,
            openai_client=openai_client,
            scenario_id=sid,
            turn=turn,
            base_ts=base_ts,
            open_sub_decisions=open_sub_decisions,
        )
        compacted_this_turn = compaction_record is not None
        if compacted_this_turn:
            compaction_events.append(compaction_record)
            last_compaction_turn = turn
            for fw in compaction_record.get("facts_written", []):
                compacted_facts.append(
                    {
                        "fact_text": fw.get("fact_text"),
                        "fact_from_turn": fw.get("fact_from_turn"),
                        "compaction_at_turn": fw.get("compaction_at_turn"),
                        "fact_id": fw.get("fact_id"),
                        "error": fw.get("error"),
                    }
                )

        thread_tokens_before = messages_tokens(mt_messages)

        try:
            raw = await llm_chat_messages(openai_client, mt_messages)
        except Exception as exc:
            log = TurnLog(
                turn=turn,
                raw_excerpt=f"LLM ERROR: {exc!r}"[:300],
                raw_full="",
                thread_tokens_before=thread_tokens_before,
                thread_tokens_after_response=thread_tokens_before,
                thread_tokens_after_truncate=thread_tokens_before,
                compaction_fired_this_turn=compacted_this_turn,
                motivation_state_after=current_motivation.state,
                motivation_intensity_after=current_motivation.intensity,
                n_active_sub_decisions=sum(
                    1
                    for info in open_sub_decisions.values()
                    if info.get("state") != "closed"
                ),
            )
            trace.append(log)
            break

        mt_messages.append({"role": "assistant", "content": raw})
        thread_tokens_after_response = messages_tokens(mt_messages)

        log = TurnLog(
            turn=turn,
            raw_excerpt=raw[:400],
            raw_full=raw,
            thread_tokens_before=thread_tokens_before,
            thread_tokens_after_response=thread_tokens_after_response,
            thread_tokens_after_truncate=thread_tokens_after_response,
            compaction_fired_this_turn=compacted_this_turn,
            motivation_state_after=current_motivation.state,
            motivation_intensity_after=current_motivation.intensity,
            n_active_sub_decisions=sum(
                1
                for info in open_sub_decisions.values()
                if info.get("state") != "closed"
            ),
        )
        if turn == 1:
            log.stream_msg_delivered_this_turn = first_msg["turn"]
            log.stream_msg_ingested_this_turn = first_msg["turn"]

        thinkings_this_turn = parse_thinking_blocks(raw)
        if thinkings_this_turn:
            recent_thinkings.extend(thinkings_this_turn)
            recent_thinkings = recent_thinkings[-6:]

        # --- Parse step_outputs and run LLM-judge per active DP ---
        step_outs = parse_step_outputs(raw)
        log.n_step_outs = len(step_outs)
        emitted_step_output_this_turn = bool(step_outs)

        for so in step_outs:
            raw_label = so["raw_label"]
            content = so["content"]
            if raw_label.isdigit():
                sid_int = int(raw_label)
            else:
                if raw_label not in label_to_int:
                    label_to_int[raw_label] = next_label_int
                    next_label_int += 1
                sid_int = label_to_int[raw_label]

            eligible_dps = [
                d for d in decision_points if d["after_turn"] <= stream_turns_delivered
            ]

            judge_tasks = [
                judge_dp_coverage(
                    content,
                    list(d.get("required_facts") or []),
                    gold_text_for_facts,
                    openai_client=openai_client,
                    sub_decision=d.get("sub_decision", ""),
                )
                for d in eligible_dps
            ]
            judge_results: list[dict[str, Any]] = (
                await asyncio.gather(*judge_tasks, return_exceptions=False)
                if judge_tasks
                else []
            )

            per_dp_judgements: list[dict[str, Any]] = []
            for d, jr in zip(eligible_dps, judge_results):
                per_dp_judgements.append(
                    {
                        "sub_decision": d["sub_decision"],
                        "after_turn": d["after_turn"],
                        "required_facts": list(d.get("required_facts") or []),
                        "covered": bool(jr.get("covered", False)),
                        "evidence": jr.get("evidence", ""),
                        "judge_error": jr.get("error"),
                    }
                )
                dp_coverage_log.append(
                    {
                        "step_id": sid_int,
                        "raw_label": raw_label,
                        "agent_turn": turn,
                        "stream_turns_delivered_at_emit": stream_turns_delivered,
                        "sub_decision": d["sub_decision"],
                        "after_turn": d["after_turn"],
                        "required_facts": list(d.get("required_facts") or []),
                        "covered": bool(jr.get("covered", False)),
                        "evidence": jr.get("evidence", ""),
                        "judge_error": jr.get("error"),
                    }
                )

            step_outputs_by_id[sid_int] = {
                "step_id": sid_int,
                "raw_label": raw_label,
                "label": raw_label[:200],
                "content": content,
                "turn": turn,
                "stream_turns_delivered_at_emit": stream_turns_delivered,
                "per_dp_judgements": per_dp_judgements,
            }
            step_outputs_log.append(
                {
                    "step_id": sid_int,
                    "raw_label": raw_label,
                    "content": content,
                    "turn": turn,
                    "stream_turns_delivered_at_emit": stream_turns_delivered,
                    "per_dp_judgements": per_dp_judgements,
                }
            )
            log.step_outs_emitted.append(
                {
                    "step_id": sid_int,
                    "raw_label": raw_label,
                    "len": len(content),
                    "n_dps_judged": len(per_dp_judgements),
                    "n_dps_covered": sum(
                        1 for j in per_dp_judgements if j["covered"]
                    ),
                }
            )

        # --- Parse DEFER / CLOSE markers ---
        defers, closes = parse_defer_close(raw)
        log.defers_emitted = list(defers)
        log.closes_emitted = list(closes)
        # Apply harness-side closures for explicit markers.
        for d_label in defers:
            if _close_commitment(
                open_sub_decisions, d_label, at_turn=turn, reason="deferred_by_agent"
            ):
                n_commitments_deferred_via_revisit += 1
        for c_label in closes:
            if _close_commitment(
                open_sub_decisions, c_label, at_turn=turn, reason="closed_by_agent"
            ):
                n_commitments_closed_via_revisit += 1

        # --- Classify response to a pending revisit (if injected last turn) ---
        if pending_revisit_event is not None:
            # The previous-turn injection targeted pending_revisit_target.
            # Determine how this turn responded.
            resp_kind = "no_action"
            target_lkey = (pending_revisit_target or "").lower().strip()
            # Heuristic: any STEP_OUTPUT mentioning the label (or simply
            # any STEP_OUTPUT emitted this turn while the commitment was
            # being prompted) counts as step_output-driven closure. Be
            # generous; the agent often paraphrases labels.
            so_addressed_target = False
            for so in step_outs:
                if so.get("raw_label", "").lower().strip() == target_lkey:
                    so_addressed_target = True
                    break
                # Substring match in either direction (label may be longer than id).
                if target_lkey and target_lkey in (so.get("raw_label", "") or "").lower():
                    so_addressed_target = True
                    break
                # Also accept if content references the label.
                if target_lkey and target_lkey in (so.get("content", "") or "").lower():
                    so_addressed_target = True
                    break
            target_deferred = any(
                (d.lower().strip() == target_lkey) or
                (target_lkey and target_lkey in d.lower())
                for d in defers
            )
            target_closed = any(
                (c.lower().strip() == target_lkey) or
                (target_lkey and target_lkey in c.lower())
                for c in closes
            )
            if so_addressed_target:
                resp_kind = "step_output"
                # Close the commitment to clear it from future revisit eligibility.
                if _close_commitment(
                    open_sub_decisions,
                    pending_revisit_target,
                    at_turn=turn,
                    reason="closed_via_revisit_step_output",
                ):
                    n_commitments_closed_via_revisit += 1
            elif target_deferred:
                resp_kind = "defer"
            elif target_closed:
                resp_kind = "close"
            else:
                resp_kind = "no_action"

            pending_revisit_event["agent_responded"] = resp_kind != "no_action"
            pending_revisit_event["response_kind"] = resp_kind
            pending_revisit_event["responded_at_turn"] = turn
            log.revisit_response_kind = resp_kind
            pending_revisit_event = None
            pending_revisit_target = ""

        done_this_turn = has_done(raw)
        if done_this_turn:
            log.done_emitted = True

        # --- Probes ---
        probes = parse_probes(raw)
        probes = probes[:4]
        log.probes = probes

        new_snippets: list[str] = []
        new_facts = 0
        if probes:
            try:
                hits_lists = await asyncio.gather(
                    *[probe(memory, p, RETRIEVE_K) for p in probes]
                )
            except Exception as exc:
                hits_lists = [[] for _ in probes]
                log.raw_excerpt = (log.raw_excerpt or "") + f" | RET ERR: {exc!r}"[:120]

            n_probes_total += len(probes)
            new_hit_ids: list[str] = []
            for hits in hits_lists:
                for idx, h in enumerate(hits):
                    chat_id, content = hit_to_id_text(h, idx)
                    if not content:
                        continue
                    if chat_id in seen_chat_ids:
                        continue
                    seen_chat_ids.add(chat_id)
                    new_hit_ids.append(chat_id)
                    if h.plant_id:
                        new_facts += 1
                    snip = content.replace("\n", " ").strip()
                    if len(snip) > 240:
                        snip = snip[:237] + "..."
                    new_snippets.append(f"[{chat_id}] {snip}")
            log.n_hits = len(new_snippets)
            log.new_hit_ids = new_hit_ids
            log.new_facts_this_turn = new_facts

            if new_facts == 0:
                consecutive_dry_turns += 1
            else:
                consecutive_dry_turns = 0
        else:
            consecutive_dry_turns += 1

        snippets_block = (
            "\n".join(f"- {s}" for s in new_snippets)
            if new_snippets
            else "(no new snippets surfaced)"
        )

        # ---- Motivation update decision (only if SH_DISABLE_MOTIVATION=0) ----
        turns_since_motivation_update = turn - last_motivation_update_turn
        motivation_due = False
        motivation_trigger = None
        if not SH_DISABLE_MOTIVATION:
            if compacted_this_turn:
                motivation_due = True
                motivation_trigger = "post_compaction"
            elif turns_since_motivation_update >= SH_MOTIVATION_PERIOD:
                motivation_due = True
                motivation_trigger = "periodic"

        if motivation_due:
            compaction_age_turns = (
                turn - (last_compaction_turn or -10**9)
                if last_compaction_turn is not None
                else 10**9
            )
            recent_summary = _build_recent_activity_summary(
                last_compaction=(
                    compaction_events[-1] if compaction_events else None
                ),
                compaction_age_turns=compaction_age_turns,
                recent_thinkings=list(recent_thinkings[-3:]),
            )
            unresolved_goals_input: list[str] = []
            for label, info in open_sub_decisions.items():
                if info.get("state") == "closed":
                    continue
                opened_at = int(info.get("opened_at_compaction_turn", turn))
                age = max(0, turn - opened_at)
                summary_text = (info.get("summary") or "").strip()
                if summary_text:
                    unresolved_goals_input.append(
                        f"{label}: {summary_text} (opened {age} turns ago)"
                    )
                else:
                    unresolved_goals_input.append(
                        f"{label} (opened {age} turns ago)"
                    )
            try:
                new_motivation = await update_motivation(
                    openai_client,
                    current_state=current_motivation,
                    recent_activity_summary=recent_summary,
                    unresolved_goals=unresolved_goals_input,
                    turns_since_last_user_input=turns_since_last_user_input,
                    turns_since_last_completion=turns_since_last_completion,
                    turns_since_motivation_update=turns_since_motivation_update,
                    current_turn=turn,
                )
            except Exception as exc:
                motivation_events.append(
                    {
                        "at_turn": turn,
                        "trigger": motivation_trigger,
                        "error": f"update_motivation_failed: {exc!r}",
                        "state": current_motivation.state,
                        "intensity": current_motivation.intensity,
                        "rationale": current_motivation.rationale,
                        "drive_directive": current_motivation.drive_directive,
                        "unresolved_goals": unresolved_goals_input,
                        "n_unresolved_goals": len(unresolved_goals_input),
                    }
                )
                new_motivation = current_motivation
            else:
                motivation_events.append(
                    {
                        "at_turn": turn,
                        "trigger": motivation_trigger,
                        "error": None,
                        "state": new_motivation.state,
                        "intensity": new_motivation.intensity,
                        "rationale": new_motivation.rationale,
                        "drive_directive": new_motivation.drive_directive,
                        "unresolved_goals": unresolved_goals_input,
                        "n_unresolved_goals": len(unresolved_goals_input),
                    }
                )

            current_motivation = new_motivation
            last_motivation_update_turn = turn
            log.motivation_updated_this_turn = True
            log.motivation_state_after = current_motivation.state
            log.motivation_intensity_after = current_motivation.intensity
            pending_motivation_prefix = _format_motivation_prefix(current_motivation)
        else:
            if not SH_DISABLE_MOTIVATION:
                pending_motivation_prefix = _format_motivation_prefix(current_motivation)
            else:
                pending_motivation_prefix = None

        # --- Decide next user message ---
        delivered_new_stream_this_turn = False
        next_body: str
        if done_this_turn and not stream_closed_announced:
            log.premature_done = True
            premature_done_count += 1
            next_body = USER_FOLLOWUP_PREMATURE_DONE.format(
                turn=turn + 1,
                snippets=snippets_block,
                n_probes_total=n_probes_total,
                new_facts_this_turn=new_facts,
                consecutive_dry_turns=consecutive_dry_turns,
            )
        else:
            deliver_new_stream = (
                stream_idx < len(stream_messages)
                and turn % stream_interval == 0
            )

            if deliver_new_stream:
                next_msg = stream_messages[stream_idx]
                stream_idx += 1
                stream_turns_delivered = next_msg["turn"]
                log.stream_msg_delivered_this_turn = next_msg["turn"]
                ing_info = await ingester.ingest(next_msg["turn"])
                if ing_info.get("ingested"):
                    log.stream_msg_ingested_this_turn = next_msg["turn"]
                delivered_new_stream_this_turn = True
                next_body = USER_FOLLOWUP_WITH_STREAM.format(
                    turn=turn + 1,
                    snippets=snippets_block,
                    n_probes_total=n_probes_total,
                    new_facts_this_turn=new_facts,
                    consecutive_dry_turns=consecutive_dry_turns,
                    stream_turn=next_msg["turn"],
                    stream_text=next_msg["text"],
                )
            elif (
                stream_idx >= len(stream_messages) and not stream_closed_announced
            ):
                stream_closed_announced = True
                next_body = USER_FOLLOWUP_STREAM_CLOSED.format(
                    turn=turn + 1,
                    snippets=snippets_block,
                    n_probes_total=n_probes_total,
                    new_facts_this_turn=new_facts,
                    consecutive_dry_turns=consecutive_dry_turns,
                )
            else:
                next_body = USER_FOLLOWUP_PROBE_ONLY.format(
                    turn=turn + 1,
                    snippets=snippets_block,
                    n_probes_total=n_probes_total,
                    new_facts_this_turn=new_facts,
                    consecutive_dry_turns=consecutive_dry_turns,
                )

        # ---- Decide if a revisit trigger should be appended ----
        # Eligibility:
        #   - scheduler enabled
        #   - this turn did NOT deliver a new stream message
        #   - turns_since_last_revisit >= SH_REVISIT_PERIOD
        #   - at least one open commitment older than SH_REVISIT_MIN_AGE
        #   - not piggybacked on STREAM_CLOSED or PREMATURE_DONE (don't pile on)
        is_premature_done = log.premature_done
        revisit_eligible = (
            not SH_DISABLE_REVISIT
            and not delivered_new_stream_this_turn
            and not is_premature_done
            and (turn - last_revisit_turn) >= SH_REVISIT_PERIOD
        )
        revisit_appendix = ""
        if revisit_eligible:
            picked = _select_revisit_target(
                open_sub_decisions, at_turn=turn, min_age=SH_REVISIT_MIN_AGE
            )
            if picked is not None:
                label, age, info = picked
                revisit_appendix = _format_revisit_addendum(label, age, info)
                log.revisit_injected_after_this_turn = True
                log.revisit_target_label = label
                log.revisit_target_age = age
                event_dict: dict[str, Any] = {
                    "at_turn": turn,
                    "target_label": label,
                    "age_at_revisit": age,
                    "target_opened_at_compaction_turn": info.get(
                        "opened_at_compaction_turn"
                    ),
                    "target_summary": (info.get("summary") or "")[:240],
                    "agent_responded": False,
                    "response_kind": None,
                    "responded_at_turn": None,
                }
                revisit_events.append(event_dict)
                pending_revisit_event = event_dict
                pending_revisit_target = label
                last_revisit_turn = turn

        # Compose the followup message: optional motivation prefix + body
        # + optional revisit addendum.
        followup_parts: list[str] = []
        if pending_motivation_prefix:
            followup_parts.append(pending_motivation_prefix)
        followup_parts.append(next_body)
        if revisit_appendix:
            followup_parts.append(revisit_appendix)
        mt_messages.append(
            {
                "role": "user",
                "content": "\n\n".join(followup_parts),
            }
        )

        # Update user-input / completion counters.
        if delivered_new_stream_this_turn or turn == 1:
            turns_since_last_user_input = 0
        else:
            turns_since_last_user_input += 1

        if emitted_step_output_this_turn:
            turns_since_last_completion = 0
        else:
            turns_since_last_completion += 1

        # Hard-truncate fallback.
        trunc = truncate_thread(mt_messages, MT_HARD_CAP)
        log.truncate_dropped_pairs = trunc["dropped_pairs"]
        log.truncate_dropped_msgs = trunc["dropped_msgs"]
        log.thread_tokens_after_truncate = messages_tokens(mt_messages)

        trace.append(log)

        if done_this_turn and stream_closed_announced:
            break
        if (
            done_this_turn
            and not stream_closed_announced
            and premature_done_count >= MAX_PREMATURE_DONE_NUDGES
        ):
            break

    final_step_outputs = sorted(step_outputs_by_id.values(), key=lambda s: s["step_id"])

    dp_coverage: list[dict[str, Any]] = []
    for dp in decision_points:
        sub = dp["sub_decision"]
        cumulative_stream = stream_messages[0]["turn"]
        earliest_eligible_agent_turn = None
        for t in trace:
            if t.stream_msg_delivered_this_turn is not None:
                cumulative_stream = max(
                    cumulative_stream, t.stream_msg_delivered_this_turn
                )
            if cumulative_stream >= dp["after_turn"]:
                earliest_eligible_agent_turn = t.turn
                break

        covered_emits = [
            row
            for row in dp_coverage_log
            if row["sub_decision"] == sub and row.get("covered")
        ]
        first_covered = (
            min(covered_emits, key=lambda r: r["agent_turn"]) if covered_emits else None
        )
        latest_covered = (
            max(covered_emits, key=lambda r: r["agent_turn"]) if covered_emits else None
        )

        all_emits = [row for row in dp_coverage_log if row["sub_decision"] == sub]

        n_required = len(dp.get("required_facts") or [])
        dp_coverage.append(
            {
                "sub_decision": sub,
                "after_turn": dp["after_turn"],
                "required_facts": dp.get("required_facts", []),
                "n_required_facts": n_required,
                "earliest_eligible_agent_turn": earliest_eligible_agent_turn,
                "n_emits_judged_eligible": len(all_emits),
                "n_emits_covered": len(covered_emits),
                "covered": bool(covered_emits),
                "first_covered_agent_turn": (
                    first_covered["agent_turn"] if first_covered else None
                ),
                "first_covered_step_id": (
                    first_covered["step_id"] if first_covered else None
                ),
                "latest_covered_step_id": (
                    latest_covered["step_id"] if latest_covered else None
                ),
                "latest_covered_evidence": (
                    latest_covered["evidence"] if latest_covered else ""
                ),
            }
        )

    # Aggregate revisit metrics.
    n_revisits_total = len(revisit_events)
    n_revisits_responded = sum(
        1 for ev in revisit_events if ev.get("agent_responded")
    )

    return {
        "scenario_id": sid,
        "variant": "mtmsg_revisit",
        "config": {
            "compaction_threshold": SH_COMPACTION_THRESHOLD,
            "compaction_keep_recent": SH_COMPACTION_KEEP_RECENT,
            "compaction_disabled": SH_DISABLE_COMPACTION,
            "compaction_max_facts": SH_COMPACTION_MAX_FACTS,
            "motivation_disabled": SH_DISABLE_MOTIVATION,
            "motivation_period": SH_MOTIVATION_PERIOD,
            "revisit_disabled": SH_DISABLE_REVISIT,
            "revisit_period": SH_REVISIT_PERIOD,
            "revisit_min_age": SH_REVISIT_MIN_AGE,
            "mt_hard_cap": MT_HARD_CAP,
        },
        "trace": [td_to_dict(t) for t in trace],
        "step_outputs": final_step_outputs,
        "step_outputs_log": step_outputs_log,
        "dp_coverage_log": dp_coverage_log,
        "decision_point_coverage": dp_coverage,
        "n_turns": len(trace),
        "n_probes_total": n_probes_total,
        "n_stream_messages_delivered": stream_idx,
        "n_stream_messages_ingested": len(ingester.ingested_turns),
        "stream_closed_announced": stream_closed_announced,
        "max_thread_tokens": max(
            (t.thread_tokens_after_response for t in trace), default=0
        ),
        "max_thread_tokens_after_truncate": max(
            (t.thread_tokens_after_truncate for t in trace), default=0
        ),
        "total_truncate_dropped_pairs": sum(t.truncate_dropped_pairs for t in trace),
        "total_truncate_dropped_msgs": sum(t.truncate_dropped_msgs for t in trace),
        "n_premature_done_nudges": premature_done_count,
        "done_emitted": any(t.done_emitted for t in trace),
        "done_at_close": bool(
            stream_closed_announced and trace and trace[-1].done_emitted
        ),
        # Compaction-specific
        "n_compactions": len(compaction_events),
        "compaction_events": compaction_events,
        "compacted_facts": compacted_facts,
        "n_compacted_facts_written": sum(
            1 for f in compacted_facts if f.get("fact_id")
        ),
        # Motivation-specific (mostly vestigial here)
        "motivation_events": motivation_events,
        "n_motivation_updates": len(motivation_events),
        "final_motivation": current_motivation.as_dict(),
        # Open-sub-decisions tracker
        "open_sub_decisions_final": open_sub_decisions,
        "n_open_sub_decisions_final": sum(
            1
            for info in open_sub_decisions.values()
            if info.get("state") != "closed"
        ),
        # Revisit-specific
        "revisit_events": revisit_events,
        "n_revisits_total": n_revisits_total,
        "n_revisits_responded": n_revisits_responded,
        "n_commitments_closed_via_revisit": n_commitments_closed_via_revisit,
        "n_commitments_deferred_via_revisit": n_commitments_deferred_via_revisit,
    }


# ---------- Driver ----------


def load_streaming_scenarios(scenarios_file: Path) -> list[dict]:
    return json.loads(scenarios_file.read_text())


async def run_one_scenario(
    *,
    scenario: dict,
    vector_store,
    segment_store,
    embedder,
    openai_client,
    stream_interval: int = STREAM_INTERVAL,
    max_turns: int = MAX_TURNS,
    overwrite: bool = True,
) -> dict[str, Any]:
    sid = scenario["id"]
    t0 = time.monotonic()
    memory, ingest_info = await open_empty_memory(
        scenario,
        vector_store=vector_store,
        segment_store=segment_store,
        embedder=embedder,
        overwrite=overwrite,
    )
    setup_time = time.monotonic() - t0

    print(f"  [{sid}] starting variant=mtmsg_revisit", flush=True)
    agent_result = await run_agent_loop(
        scenario=scenario,
        memory=memory,
        openai_client=openai_client,
        stream_interval=stream_interval,
        max_turns=max_turns,
    )

    out = {
        "scenario_id": sid,
        "kind": scenario.get("kind", "stream"),
        "category": scenario.get("category", ""),
        "ingest_setup_time_s": round(setup_time, 2),
        "ingest_info": ingest_info,
        "stream_interval": stream_interval,
        "max_turns": max_turns,
        "scenario_messages": scenario["messages"],
        "scenario_decision_points": scenario.get("decision_points") or [],
        "scenario_ground_truth_facts": scenario.get("ground_truth_facts") or [],
        "scenario_distractor_facts": scenario.get("distractor_facts") or [],
        "agent_result": agent_result,
    }

    fp = RESULTS_OUT_DIR / f"{sid}.json"
    fp.write_text(json.dumps(out, indent=2, default=str))

    n_step_emits = sum(t.get("n_step_outs", 0) for t in agent_result["trace"])
    n_dp_covered = sum(
        1 for dpc in agent_result["decision_point_coverage"] if dpc["covered"]
    )
    n_dp_total = len(agent_result["decision_point_coverage"])
    print(
        f"  [{sid}] mtmsg_revisit: turns={agent_result['n_turns']} | "
        f"stream_msgs_delivered={agent_result['n_stream_messages_delivered']}/{len(scenario['messages'])} | "
        f"ingested={agent_result['n_stream_messages_ingested']} | "
        f"compactions={agent_result['n_compactions']} | "
        f"compacted_facts={agent_result['n_compacted_facts_written']} | "
        f"revisits={agent_result['n_revisits_total']} (responded={agent_result['n_revisits_responded']}) | "
        f"rv_closes={agent_result['n_commitments_closed_via_revisit']} rv_defers={agent_result['n_commitments_deferred_via_revisit']} | "
        f"step_emits={n_step_emits} | "
        f"final_steps={len(agent_result['step_outputs'])} | "
        f"DP_judged_cov={n_dp_covered}/{n_dp_total} | "
        f"probes_total={agent_result['n_probes_total']} | "
        f"premature_done={agent_result['n_premature_done_nudges']} | "
        f"done={agent_result['done_emitted']} | stream_closed={agent_result['stream_closed_announced']}",
        flush=True,
    )

    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default=None,
        help="Single scenario id (default: all). E.g., extended-project-stream-01.",
    )
    parser.add_argument(
        "--scenarios-file",
        default=None,
        help=(
            "Override scenarios JSON filename (relative to evaluation/associative_recall/data/) "
            "or absolute path. Defaults to value of SH_SCENARIO_FILE env var, "
            "else streaming_scenarios.json."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reserved.",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=STREAM_INTERVAL,
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Reuse existing collections if present.",
    )
    parser.add_argument(
        "--sqlite-suffix",
        default=None,
        help=(
            "Optional suffix for the EM sqlite db filename, useful when "
            "running parallel ablation jobs. Defaults to SH_DB_SUFFIX env var."
        ),
    )
    args = parser.parse_args()

    if args.scenarios_file:
        sf = Path(args.scenarios_file)
        if not sf.is_absolute():
            sf = DATA_DIR / sf
    else:
        sf = DEFAULT_SCENARIOS_FILE
    if not sf.exists():
        raise SystemExit(f"Scenarios file not found: {sf}")

    scenarios = load_streaming_scenarios(sf)
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            raise SystemExit(f"No scenario matched: {args.scenario}")

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    suffix = args.sqlite_suffix if args.sqlite_suffix is not None else SH_DB_SUFFIX
    sqlite_path = (
        RESULTS_DIR
        / f"eventmemory_shared_harness_v4_mtmsg_revisit_initprime{suffix}.sqlite3"
    )
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
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

    results: list[dict[str, Any]] = []
    try:
        for scenario in scenarios:
            sid = scenario["id"]
            print(
                f"[run] {sid} (msgs={len(scenario['messages'])}, "
                f"dps={len(scenario.get('decision_points') or [])})",
                flush=True,
            )
            try:
                r = await run_one_scenario(
                    scenario=scenario,
                    vector_store=vector_store,
                    segment_store=segment_store,
                    embedder=embedder,
                    openai_client=openai_client,
                    stream_interval=args.stream_interval,
                    max_turns=args.max_turns,
                    overwrite=not args.no_overwrite,
                )
                results.append(r)
            except Exception as exc:
                print(f"  ERROR {sid}: {exc!r}", flush=True)
                results.append({"scenario_id": sid, "error": repr(exc)})
    finally:
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    summary = {
        "n_scenarios": len(results),
        "scenarios_file": str(sf),
        "stream_interval": args.stream_interval,
        "max_turns": args.max_turns,
        "compaction_threshold": SH_COMPACTION_THRESHOLD,
        "compaction_keep_recent": SH_COMPACTION_KEEP_RECENT,
        "compaction_disabled": SH_DISABLE_COMPACTION,
        "compaction_max_facts": SH_COMPACTION_MAX_FACTS,
        "motivation_disabled": SH_DISABLE_MOTIVATION,
        "motivation_period": SH_MOTIVATION_PERIOD,
        "revisit_disabled": SH_DISABLE_REVISIT,
        "revisit_period": SH_REVISIT_PERIOD,
        "revisit_min_age": SH_REVISIT_MIN_AGE,
        "results_subdir": SH_RESULTS_SUBDIR,
        "scenarios": [
            {
                "scenario_id": r.get("scenario_id"),
                "error": r.get("error"),
                "n_dp_total": len(r.get("scenario_decision_points") or []),
                "n_dp_covered": (
                    sum(
                        1
                        for dpc in (r.get("agent_result") or {}).get(
                            "decision_point_coverage", []
                        )
                        if dpc.get("covered")
                    )
                    if r.get("agent_result")
                    else None
                ),
                "n_turns": (r.get("agent_result") or {}).get("n_turns"),
                "stream_msgs_delivered": (r.get("agent_result") or {}).get(
                    "n_stream_messages_delivered"
                ),
                "stream_msgs_ingested": (r.get("agent_result") or {}).get(
                    "n_stream_messages_ingested"
                ),
                "n_compactions": (r.get("agent_result") or {}).get("n_compactions"),
                "n_compacted_facts_written": (r.get("agent_result") or {}).get(
                    "n_compacted_facts_written"
                ),
                "n_revisits_total": (r.get("agent_result") or {}).get(
                    "n_revisits_total"
                ),
                "n_revisits_responded": (r.get("agent_result") or {}).get(
                    "n_revisits_responded"
                ),
                "n_commitments_closed_via_revisit": (r.get("agent_result") or {}).get(
                    "n_commitments_closed_via_revisit"
                ),
                "n_commitments_deferred_via_revisit": (r.get("agent_result") or {}).get(
                    "n_commitments_deferred_via_revisit"
                ),
                "n_open_sub_decisions_final": (r.get("agent_result") or {}).get(
                    "n_open_sub_decisions_final"
                ),
                "stream_closed_announced": (r.get("agent_result") or {}).get(
                    "stream_closed_announced"
                ),
                "done_emitted": (r.get("agent_result") or {}).get("done_emitted"),
                "n_premature_done_nudges": (r.get("agent_result") or {}).get(
                    "n_premature_done_nudges"
                ),
            }
            for r in results
        ],
    }
    summary_path = THIS_DIR / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {summary_path}")
    print(f"Per-scenario files in {RESULTS_OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
