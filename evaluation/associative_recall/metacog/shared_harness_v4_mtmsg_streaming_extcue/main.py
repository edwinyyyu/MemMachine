"""Shared harness v4_mtmsg_streaming_extcue — extcue + streaming_v2 layered.

Base: shared_harness_v4_mtmsg_streaming_v2 (online ingest, LLM-judge DP coverage,
premature-DONE handling, streaming-aware system prompt).

Layered mechanism: external cue generator from shared_harness_v4_mtmsg_extcue_easy.
A separate gpt-5-mini call (low reasoning) runs once per agent turn AFTER the
agent emits its turn output and BEFORE retrieval runs. It proposes 1-2
retrieval queries for IMPLICIT user-context facts based on the agent's current
focus, in a SEPARATE prompt scope (not the main conversation thread).

Replacement vs addition policy: REPLACEMENT. The total per-turn probe cap
remains 4. If the agent emits N agent-probes, external cues fill up to
(4 - N) remaining slots, capped at `EXT_CUE_BUDGET` (default 2). When the
agent saturates the budget (N >= 4), external cues are dropped this turn.
This isolates the "decoupled cognition" effect from a probe-budget expansion
confound. Each external cue is rendered to the agent (in the NEXT user
turn, alongside agent-probe hits) prefixed with `[EXT]` so the agent can
attribute hits.

Adaptation for streaming context (vs easy-10):
  - "Current focus" is more dynamic: agent might be reacting to a stream
    message that just arrived. The cue generator's `recent_agent_output`
    is the agent's just-emitted raw turn output (THINKING / PROBE /
    STEP_OUTPUT / DONE), and `recent_history` includes the most recent
    user-side stream message AND the agent's previous emit so the
    generator can see the user's incoming context, not just the agent's
    reaction to it.
  - The base streaming-aware SYSTEM_PROMPT is preserved byte-equal.
  - All streaming-v2 mechanisms are preserved: online ingest, LLM-judge
    DP coverage, premature-DONE re-prompting.

Hypothesis: in an open-ended streaming setting, decoupled cognition can
surface implicit-constraint cues that a busy agent (reacting to a fresh
stream message and emitting a STEP_OUTPUT) might miss. The 4-probe cap and
2-cue cap mirror the easy-10 design so we can compare lift across
benchmark settings.

Env hooks:
  SH_EXT_CUE_BUDGET     - max external cues per turn (default 2)

Usage:
    uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming_extcue/main.py
    uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming_extcue/main.py --scenario family-dinner-stream-01
    uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming_extcue/main.py --scenarios-file streaming_scenarios_v2.json --scenario home-renovation-stream-01

Outputs per-scenario JSON files in this dir's `results/` plus a SUMMARY.json.
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
from qdrant_client import AsyncQdrantClient  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
RESULTS_OUT_DIR = THIS_DIR / "results"
RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = _AR_DIR / "data"
DEFAULT_SCENARIOS_FILE = DATA_DIR / "streaming_scenarios.json"

ENV_PATH = _AR_DIR / ".env"
load_dotenv(ENV_PATH)

MODEL = "gpt-5-mini"
JUDGE_MODEL = "gpt-5-mini"

MT_HARD_CAP = 10_000

MAX_TURNS = 28
RETRIEVE_K = 5
MAX_COMPLETION_TOKENS = 4500

STREAM_INTERVAL = 2

NAMESPACE = "arc_em_streaming_extcue"
COLLECTION_PREFIX = "arc_sec"

# Cap on premature-DONE nudges per run; if the agent insists past this many
# nudges, we honor the DONE to avoid infinite loops.
MAX_PREMATURE_DONE_NUDGES = 3

# Per-turn cap on external cues emitted by the external generator. Total
# per-turn probe cap (agent + external) is held at 4 (TOTAL_PROBE_BUDGET_PER_TURN).
# External cues fill remaining slots after agent probes, up to EXT_CUE_BUDGET.
EXT_CUE_BUDGET = int(os.environ.get("SH_EXT_CUE_BUDGET", "2"))
TOTAL_PROBE_BUDGET_PER_TURN = 4

try:
    ENC = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(n_tokens(m.get("content", "") or "") for m in messages)


# ---------- Line parsers (verbatim from streaming_v2) ----------

PROBE_LINE_RE = re.compile(r"^\s*PROBE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
DONE_LINE_RE = re.compile(r"^\s*DONE\s*$", re.MULTILINE | re.IGNORECASE)
STEP_OUTPUT_HEAD_RE = re.compile(
    r"^\s*STEP_OUTPUT\s*:\s*([^\n:\-]+?)(?:\s*[:\-]\s*(.*))?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
DIRECTIVE_LINE_RE = re.compile(
    r"^\s*(THINKING|PROBE|STEP_OUTPUT|DONE)\b",
    re.MULTILINE | re.IGNORECASE,
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


# ---------- System prompt — IDENTICAL to streaming_v2 ----------
#
# Per task instructions: do NOT redesign the prompt. Keep it byte-equal.

SYSTEM_PROMPT = """\
You are a memory-augmented agent in an endless working loop with bounded \
working memory. The user is in an ongoing conversation with you and will \
keep delivering new context and new sub-tasks over many messages spread \
across this thread. There is no single up-front task statement. The work \
is open-ended.

You are running inside a continuous conversation thread. The thread is \
your context. When the thread crosses a token cap, the OLDEST user/assistant \
turns get hard-dropped (the system prompt stays). There is NO compression, \
NO LRU, NO citations — just hard truncation from the front.

This means: load-bearing facts you surface from memory survive ONLY as long \
as they remain in your recent reasoning. If old user context falls off the \
front of the thread, your only way back to it is via a memory PROBE.

You have a memory tool over PAST chat history. Crucially: in this streaming \
setup the memory contains EVERY past user message verbatim, including ones \
that have already scrolled out of your immediate context. So when the user \
says something at message 3 and asks you to make a decision at message 9, \
the message-3 fact is still in memory and can be PROBE'd back.

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
message. Emit STEP_OUTPUT only when the user has actually asked for \
something concrete — not for context they've just shared.
  Consequence: at task end, the latest STEP_OUTPUT per id ships verbatim \
to the user as your final deliverable for that sub-decision.

DONE
  Purpose: signal you have nothing more to do AND no more stream messages \
are pending. Use sparingly — emit DONE only after the harness explicitly \
signals the stream is closed.

The harness will not deliver DONE prematurely; if you DONE before the \
stream is exhausted, you'll have shipped incomplete work.
"""


USER_INITIAL_STREAM = """\
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


# ---------- Thread truncation (verbatim from streaming_v2) ----------


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


# ---------- Hit rendering ----------


def hit_to_id_text(hit, fallback_idx: int) -> tuple[str, str]:
    tid = getattr(hit, "turn_id", None)
    if tid is None or tid < 0:
        tid = 9000 + fallback_idx
    chat_id = f"chat-{tid}"
    content = (hit.formatted_text or hit.text or "").strip()
    return chat_id, content


# ---------- Online stream ingestion (verbatim from streaming_v2) ----------


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
    """Create a fresh EM collection for online ingestion. Memory is EMPTY at
    return — no stream messages have been encoded yet. The caller drives
    ingestion turn-by-turn via OnlineStreamIngester.
    """
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
        "n_messages": 0,  # nothing ingested yet
        "ingest_mode": "online",
    }
    return memory, info


class OnlineStreamIngester:
    """Encodes scenario stream messages into EM one at a time.

    Indexed by stream `turn` (the message's `turn` field in the scenario).
    Each call to `ingest(stream_turn)` looks up the message and encodes it as
    an Event with the same property layout streaming_v2 uses, so probe
    results render identically.
    """

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
            return {"stream_turn": stream_turn, "ingested": False, "reason": "no such turn"}
        if int(stream_turn) in self.ingested_turns:
            return {"stream_turn": stream_turn, "ingested": False, "reason": "already ingested"}

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
    """Factory matching the function-style signature requested by the task."""
    return OnlineStreamIngester(scenario, em_session)


# ---------- LLM-judge DP coverage (verbatim from streaming_v2) ----------


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
    """Ask gpt-5-mini whether `step_output_text` reflects `required_facts`.

    Returns {"covered": bool, "evidence": str, "raw": str}. On parse failure
    or LLM error, returns covered=False with an error tag in evidence so
    callers can distinguish judge failures from honest non-coverage.
    """
    if not (step_output_text or "").strip():
        return {"covered": False, "evidence": "", "raw": "", "error": "empty_step_output"}

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
        # Older deployments may reject reasoning_effort. Retry without it.
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


# ---------- External cue generator (ported from extcue_easy) ----------
#
# Adaptation for streaming context: the cue generator receives
#   (a) the agent's just-emitted raw turn output (`recent_agent_output`),
#   (b) a brief recent_history of mt_messages tail (so it can see the most
#       recent INCOMING USER STREAM MESSAGE blocks AND prior assistant emits),
#   (c) the agent's probes already issued this turn (no duplication).
#
# In the streaming setting there is no fixed up-front task_prompt the way
# easy-10 has. We pass an empty/placeholder task field so the prompt
# template still renders, but the real "task framing" comes from the
# history+focus pair.

EXT_CUE_SYSTEM_PROMPT = """\
You generate retrieval queries for a memory-augmented agent. The agent is \
working on a multi-step task and consults a memory store of past chat \
turns where the user has shared specific facts (constraints, preferences, \
allergies, dates, numbers, identities) that materially shape correct \
deliverables.

Your job: given the agent's current focus (recent thinking + most recent \
deliverable), propose 1-2 retrieval queries that target IMPLICIT \
USER-CONTEXT FACTS — facts the user has shared previously that the task \
description itself does not name, but which would change what a correct \
answer looks like for the agent's current focus.

Discipline:
- Target facts the agent's current working memory does NOT already show. \
If the agent has already surfaced and is reasoning about a relevant fact, \
do NOT re-probe for it.
- Be concrete and specific. Prefer queries that name a plausible value \
(e.g., "user prefers Thursday afternoons") over abstract ones \
("user's scheduling preferences"). Memory retrieves by semantic similarity, \
so words in the query should resemble words a stored fact would use.
- Stay close to the agent's current focus. Do not propose queries for \
sub-decisions the agent hasn't started yet.
- Each query is a short retrieval probe, not a question to the user. \
Phrase it as a fragment a stored fact might literally contain.

Output ONLY a JSON object matching:
{"cues": ["<probe 1>", "<probe 2>"]}
Maximum {budget} cues. Fewer is fine if you don't have a confident \
candidate. Empty list is acceptable when the agent's current focus does \
not invite implicit user-context facts.
"""


EXT_CUE_USER_TEMPLATE = """\
ORIGINAL TASK:
{task_prompt}

AGENT'S CURRENT FOCUS (most recent turn output):
---
{recent_agent_output}
---

AGENT'S RECENT TURN HISTORY (older→newer, abbreviated):
---
{recent_history}
---

AGENT'S PROBES THIS TURN (already issued, do not duplicate):
{agent_probes_block}

Propose up to {budget} retrieval queries for implicit user-context facts \
that might apply to the agent's current focus. Output JSON only.
"""


EXT_CUE_SCHEMA = {
    "name": "ext_cue_set",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "cues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Retrieval queries for implicit user-context facts. 0-N items.",
            },
        },
        "required": ["cues"],
        "additionalProperties": False,
    },
}


def _parse_ext_cues_json(text: str) -> list[str]:
    if not text:
        return []
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return []
    try:
        obj = json.loads(t[s : e + 1])
    except Exception:
        return []
    cues = obj.get("cues") if isinstance(obj, dict) else None
    if not isinstance(cues, list):
        return []
    out: list[str] = []
    for c in cues:
        if isinstance(c, str):
            cs = c.strip()
            if cs:
                out.append(cs)
    return out


async def external_cue_generator(
    *,
    task_prompt: str,
    recent_agent_output: str,
    recent_history: str,
    agent_probes: list[str],
    budget: int = EXT_CUE_BUDGET,
    openai_client,
) -> list[str]:
    """Generate up to `budget` retrieval cues from a separate gpt-5-mini call.

    Decoupled from the main agent's conversation thread: a fresh prompt
    scope each turn focused on the agent's current focus + recent history.
    Returns 0..budget cues. Errors return [] (no-op fallback).
    """
    if budget <= 0:
        return []

    agent_probes_block = (
        "\n".join(f"- {p}" for p in agent_probes) if agent_probes else "(none)"
    )

    system_msg = EXT_CUE_SYSTEM_PROMPT.replace("{budget}", str(budget))
    user_msg = EXT_CUE_USER_TEMPLATE.format(
        task_prompt=(task_prompt or "(open-ended streaming task — no fixed up-front prompt)").strip(),
        recent_agent_output=(recent_agent_output or "").strip()[:2000],
        recent_history=(recent_history or "(none)").strip()[:2000],
        agent_probes_block=agent_probes_block,
        budget=budget,
    )

    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_completion_tokens": 800,
        "response_format": {"type": "json_schema", "json_schema": EXT_CUE_SCHEMA},
        "reasoning_effort": "low",
    }

    try:
        resp = await openai_client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Older deployments may reject reasoning_effort; retry without.
        if "reasoning_effort" in str(exc).lower() or "unsupported" in str(exc).lower():
            kwargs.pop("reasoning_effort", None)
            try:
                resp = await openai_client.chat.completions.create(**kwargs)
            except Exception:
                return []
        else:
            return []

    raw = (resp.choices[0].message.content or "").strip()
    cues = _parse_ext_cues_json(raw)
    return cues[:budget]


def _format_recent_history(messages: list[dict[str, str]], k: int = 4) -> str:
    """Render the last k user/assistant messages (excluding system) as a
    short concatenation for the external cue generator's context.

    In streaming context, the user-side messages contain the
    `--- INCOMING USER STREAM MESSAGE (turn N) ---` blocks AND the harness's
    bookkeeping (probe results, dry-turn counters). The cue generator sees
    this verbatim, which is the right shape for it to anchor on the most
    recent stream message the agent just heard about.
    """
    if not messages:
        return ""
    # Skip system at index 0; take up to last k content snippets.
    body = messages[1:]
    tail = body[-k:] if len(body) > k else body
    parts: list[str] = []
    for m in tail:
        role = m.get("role", "?")
        content = (m.get("content", "") or "").strip()
        if not content:
            continue
        snip = content if len(content) <= 600 else (content[:597] + "...")
        parts.append(f"[{role}] {snip}")
    return "\n".join(parts)


# ---------- Agent loop ----------


@dataclass
class TurnLog:
    turn: int
    raw_excerpt: str
    raw_full: str
    stream_msg_delivered_this_turn: int | None = None
    stream_msg_ingested_this_turn: int | None = None
    probes: list[str] = field(default_factory=list)
    ext_probes: list[str] = field(default_factory=list)
    n_agent_probes: int = 0
    n_ext_probes: int = 0
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


def td_to_dict(t: TurnLog) -> dict[str, Any]:
    return {
        "turn": t.turn,
        "raw_excerpt": t.raw_excerpt,
        "raw_full": t.raw_full,
        "stream_msg_delivered_this_turn": t.stream_msg_delivered_this_turn,
        "stream_msg_ingested_this_turn": t.stream_msg_ingested_this_turn,
        "probes": t.probes,
        "ext_probes": t.ext_probes,
        "n_agent_probes": t.n_agent_probes,
        "n_ext_probes": t.n_ext_probes,
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
    }


def _build_gold_text_for_facts(scenario: dict) -> dict[str, str]:
    """Map fact_id -> gold-text from `ground_truth_facts`. Missing IDs return empty."""
    out: dict[str, str] = {}
    for f in scenario.get("ground_truth_facts") or []:
        fid = f.get("id")
        if fid:
            out[fid] = (f.get("text") or "").strip()
    return out


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

    # In streaming, there's no fixed up-front task_prompt; some scenarios may
    # carry one in metadata. We pass it through if present.
    task_prompt_for_extcue = (
        scenario.get("task_prompt")
        or scenario.get("description")
        or "(open-ended streaming task — no fixed up-front prompt)"
    )

    ingester = online_stream_ingester(scenario, memory)

    # First stream message is delivered AND ingested at turn 1.
    first_msg = stream_messages[0]
    await ingester.ingest(first_msg["turn"])

    stream_idx = 1  # next index into stream_messages
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

    # dp_coverage_log: per (step_id, dp_index) judge call, accumulated.
    dp_coverage_log: list[dict[str, Any]] = []

    for turn in range(1, max_turns + 1):
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
        )
        # turn 1 delivers AND ingests the first stream message.
        if turn == 1:
            log.stream_msg_delivered_this_turn = first_msg["turn"]
            log.stream_msg_ingested_this_turn = first_msg["turn"]

        # --- Parse step_outputs and run LLM-judge per active DP ---
        step_outs = parse_step_outputs(raw)
        log.n_step_outs = len(step_outs)

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

            # Only judge against DPs that are eligible at emit time
            # (after_turn <= stream_turns_delivered). Future-DP judging
            # would be unfair (the agent couldn't know).
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

        done_this_turn = has_done(raw)
        if done_this_turn:
            log.done_emitted = True

        # --- Probes (agent + ext, replacement policy under TOTAL_PROBE_BUDGET_PER_TURN) ---
        agent_probes = parse_probes(raw)[:TOTAL_PROBE_BUDGET_PER_TURN]
        log.n_agent_probes = len(agent_probes)

        # Compute remaining slots for external cues.
        remaining_slots = max(
            0, TOTAL_PROBE_BUDGET_PER_TURN - len(agent_probes)
        )
        ext_budget_this_turn = min(EXT_CUE_BUDGET, remaining_slots)

        ext_cues: list[str] = []
        # Skip ext-cue generation when:
        #   - no remaining budget (agent saturated)
        #   - the agent emitted DONE this turn (we'll either honor it or
        #     re-prompt, and ext cues won't help shape the next-turn state)
        if ext_budget_this_turn > 0 and not done_this_turn:
            try:
                ext_cues = await external_cue_generator(
                    task_prompt=task_prompt_for_extcue,
                    recent_agent_output=raw,
                    recent_history=_format_recent_history(mt_messages, k=4),
                    agent_probes=agent_probes,
                    budget=ext_budget_this_turn,
                    openai_client=openai_client,
                )
            except Exception as exc:
                ext_cues = []
                log.raw_excerpt = (
                    (log.raw_excerpt or "") + f" | EXT ERR: {exc!r}"[:120]
                )
        ext_cues = ext_cues[:ext_budget_this_turn]
        log.ext_probes = list(ext_cues)
        log.n_ext_probes = len(ext_cues)

        # Combined probe set: agent first, then external. Source-tag so we
        # can attribute hits in trace.
        probes_with_source: list[tuple[str, str]] = (
            [(p, "agent") for p in agent_probes]
            + [(c, "ext") for c in ext_cues]
        )
        # Plain `probes` kept for downstream legacy-trace compatibility.
        probes = [p for p, _ in probes_with_source]
        log.probes = probes

        new_snippets: list[str] = []
        new_facts = 0
        if probes_with_source:
            try:
                hits_lists = await asyncio.gather(
                    *[probe(memory, p, RETRIEVE_K) for p, _ in probes_with_source]
                )
            except Exception as exc:
                hits_lists = [[] for _ in probes_with_source]
                log.raw_excerpt = (log.raw_excerpt or "") + f" | RET ERR: {exc!r}"[:120]

            n_probes_total += len(probes_with_source)
            new_hit_ids: list[str] = []
            for (probe_text, source), hits in zip(probes_with_source, hits_lists):
                tag = "[EXT]" if source == "ext" else ""
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
                    prefix = f"{tag} " if tag else ""
                    new_snippets.append(f"{prefix}[{chat_id}] {snip}")
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

        # --- Decide next user message ---
        # Premature DONE has highest priority: if the agent emitted DONE but
        # the stream is not yet closed, inject a nudge regardless of schedule
        # and do NOT advance the stream this turn.
        if done_this_turn and not stream_closed_announced:
            log.premature_done = True
            premature_done_count += 1
            mt_messages.append(
                {
                    "role": "user",
                    "content": USER_FOLLOWUP_PREMATURE_DONE.format(
                        turn=turn + 1,
                        snippets=snippets_block,
                        n_probes_total=n_probes_total,
                        new_facts_this_turn=new_facts,
                        consecutive_dry_turns=consecutive_dry_turns,
                    ),
                }
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
                # Online ingest happens RIGHT BEFORE we hand the message to
                # the agent — so by the time the agent sees the new message,
                # all stream messages up to and including this one are in EM.
                ing_info = await ingester.ingest(next_msg["turn"])
                if ing_info.get("ingested"):
                    log.stream_msg_ingested_this_turn = next_msg["turn"]
                mt_messages.append(
                    {
                        "role": "user",
                        "content": USER_FOLLOWUP_WITH_STREAM.format(
                            turn=turn + 1,
                            snippets=snippets_block,
                            n_probes_total=n_probes_total,
                            new_facts_this_turn=new_facts,
                            consecutive_dry_turns=consecutive_dry_turns,
                            stream_turn=next_msg["turn"],
                            stream_text=next_msg["text"],
                        ),
                    }
                )
            elif (
                stream_idx >= len(stream_messages) and not stream_closed_announced
            ):
                stream_closed_announced = True
                mt_messages.append(
                    {
                        "role": "user",
                        "content": USER_FOLLOWUP_STREAM_CLOSED.format(
                            turn=turn + 1,
                            snippets=snippets_block,
                            n_probes_total=n_probes_total,
                            new_facts_this_turn=new_facts,
                            consecutive_dry_turns=consecutive_dry_turns,
                        ),
                    }
                )
            else:
                mt_messages.append(
                    {
                        "role": "user",
                        "content": USER_FOLLOWUP_PROBE_ONLY.format(
                            turn=turn + 1,
                            snippets=snippets_block,
                            n_probes_total=n_probes_total,
                            new_facts_this_turn=new_facts,
                            consecutive_dry_turns=consecutive_dry_turns,
                        ),
                    }
                )

        trunc = truncate_thread(mt_messages, MT_HARD_CAP)
        log.truncate_dropped_pairs = trunc["dropped_pairs"]
        log.truncate_dropped_msgs = trunc["dropped_msgs"]
        log.thread_tokens_after_truncate = messages_tokens(mt_messages)

        trace.append(log)

        # Break only when DONE is legitimate (stream closed) OR we've nudged
        # too many times and the agent insists.
        if done_this_turn and stream_closed_announced:
            break
        if (
            done_this_turn
            and not stream_closed_announced
            and premature_done_count >= MAX_PREMATURE_DONE_NUDGES
        ):
            break

    final_step_outputs = sorted(step_outputs_by_id.values(), key=lambda s: s["step_id"])

    # --- Decision-point coverage summary (LLM-judged) ---
    # For each DP, find the latest STEP_OUTPUT whose judge said covered=true.
    dp_coverage: list[dict[str, Any]] = []
    for dp in decision_points:
        sub = dp["sub_decision"]
        # Walk through trace and find earliest agent turn at which this DP was eligible.
        cumulative_stream = stream_messages[0]["turn"]
        earliest_eligible_agent_turn = None
        for t in trace:
            if t.stream_msg_delivered_this_turn is not None:
                cumulative_stream = max(cumulative_stream, t.stream_msg_delivered_this_turn)
            if cumulative_stream >= dp["after_turn"]:
                earliest_eligible_agent_turn = t.turn
                break

        # Latest covered judgement for this DP (revision wins).
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

        # Total emits where this DP was judged at all (eligible at emit time).
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

    return {
        "scenario_id": sid,
        "variant": "mtmsg_streaming_extcue",
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
        "n_agent_probes_total": sum(t.n_agent_probes for t in trace),
        "n_ext_probes_total": sum(t.n_ext_probes for t in trace),
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

    print(f"  [{sid}] starting variant=mtmsg_streaming_extcue", flush=True)
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
        f"  [{sid}] mtmsg_streaming_extcue: turns={agent_result['n_turns']} | "
        f"stream_msgs_delivered={agent_result['n_stream_messages_delivered']}/{len(scenario['messages'])} | "
        f"ingested={agent_result['n_stream_messages_ingested']} | "
        f"step_emits={n_step_emits} | "
        f"final_steps={len(agent_result['step_outputs'])} | "
        f"DP_judged_cov={n_dp_covered}/{n_dp_total} | "
        f"probes_total={agent_result['n_probes_total']} | "
        f"agent_probes={agent_result['n_agent_probes_total']} | "
        f"ext_probes={agent_result['n_ext_probes_total']} | "
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
        help="Single scenario id (default: all). E.g., family-dinner-stream-01.",
    )
    parser.add_argument(
        "--scenarios-file",
        default=None,
        help=(
            "Override scenarios JSON filename (relative to evaluation/associative_recall/data/) "
            "or absolute path. Default: streaming_scenarios.json"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reserved for future use; chat completions API doesn't accept seed for gpt-5 family.",
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

    sqlite_path = (
        RESULTS_DIR / "eventmemory_shared_harness_v4_mtmsg_streaming_extcue.sqlite3"
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
        "ext_cue_budget": EXT_CUE_BUDGET,
        "total_probe_budget_per_turn": TOTAL_PROBE_BUDGET_PER_TURN,
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
                "stream_closed_announced": (r.get("agent_result") or {}).get(
                    "stream_closed_announced"
                ),
                "done_emitted": (r.get("agent_result") or {}).get("done_emitted"),
                "n_premature_done_nudges": (r.get("agent_result") or {}).get(
                    "n_premature_done_nudges"
                ),
                "n_agent_probes_total": (r.get("agent_result") or {}).get(
                    "n_agent_probes_total"
                ),
                "n_ext_probes_total": (r.get("agent_result") or {}).get(
                    "n_ext_probes_total"
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
