"""Shared harness v4_mtmsg_streaming — adapter for streaming scenarios.

Adapted from shared_harness_v4_mtmsg_subdec_split_easy. Changes vs that
harness:

  1. Consumes streaming_scenarios.json (extension of mid_execution_scenarios
     schema): each scenario has a list of `messages` arriving over many
     turns plus `decision_points` with `after_turn` markers and
     `required_facts` references.

  2. The first user message becomes USER_INITIAL_STREAM (note: explicitly
     labeled "this task is going to unfold over multiple incoming user
     messages — do NOT enumerate all sub-decisions yet").

  3. Subsequent stream user messages are delivered as USER_FOLLOWUP_STREAM
     (mixed with probe results). Schedule: one new stream message every
     STREAM_INTERVAL agent turns. While stream messages remain undelivered,
     we override the probe-only followup with a combined followup that
     includes both probe results AND the next stream message. Once all
     stream messages are exhausted, we revert to the standard probe-only
     USER_FOLLOWUP and let the agent run to DONE.

  4. Memory: pre-ingest ALL stream messages at staggered timestamps before
     the agent loop. Each message has plant_id = its stream-turn id, so
     downstream we can score by-message later. This is iteration-1 simple;
     iteration-2 should switch to ONLINE ingestion (inject each message as
     it arrives — more realistic streaming).

  5. STEP_OUTPUT annotation: when the model emits STEP_OUTPUT, we check
     against decision_points and annotate which sub_decision the emit
     nominally matched (by sub_decision label-string fuzzy contains, or
     by being emitted after the matching after_turn). This is QUALITATIVE
     only — actual scoring against required_facts is iteration-2 work.

  6. NO scoring/judge call. Just write the raw trace + DP-coverage
     annotations.

Usage:
    uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming/main.py
    uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming/main.py --scenario family-dinner-stream-01 --seed 0
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

# Make the parent dir importable like v3 does.
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
SCENARIOS_FILE = DATA_DIR / "streaming_scenarios.json"

ENV_PATH = _AR_DIR / ".env"
load_dotenv(ENV_PATH)

MODEL = "gpt-5-mini"

# WM cap on the mt_messages thread.
MT_HARD_CAP = 10_000

# Total agent turns budget. Streaming scenarios can be long; bump default a bit.
MAX_TURNS = 24
RETRIEVE_K = 5
MAX_COMPLETION_TOKENS = 4500

# Schedule: one new stream user message every STREAM_INTERVAL agent turns.
STREAM_INTERVAL = 2

NAMESPACE = "arc_em_streaming"
COLLECTION_PREFIX = "arc_str"

try:
    ENC = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(n_tokens(m.get("content", "") or "") for m in messages)


# ---------- Line parsers (verbatim from subdec_split_easy) ----------

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


# ---------- System prompt — STREAMING-AWARE adaptation ----------
#
# Critical differences from subdec_split_easy SYSTEM_PROMPT:
#   - Tells the agent it is in an ENDLESS WORK loop, not a one-shot task.
#   - Sub-decisions are NOT all knowable at turn 1 — they arise as the user
#     stream brings new context.
#   - User messages will continue arriving over time; the harness will mix
#     them into the followup messages (clearly tagged as USER STREAM MESSAGE).
#   - Probe-generation discipline is the same.

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


# ---------- Thread truncation (verbatim from subdec_split_easy) ----------


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


# ---------- Streaming-scenario ingestion ----------


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


async def ingest_streaming_scenario(
    scenario: dict,
    *,
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
    overwrite: bool = True,
) -> tuple[EventMemory, dict]:
    """Pre-ingest ALL stream messages into a fresh EM collection.

    Iteration-1 simplification: every message is in memory before agent
    loop starts. Iteration-2 should switch to online ingestion (insert each
    message as it arrives so memory state reflects real time).

    Each message becomes an Event with plant_id = `stream_turn_<N>` and
    properties.from_turn = N for downstream coverage scoring.
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

    base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    events = []
    for msg in scenario["messages"]:
        stream_turn = msg["turn"]
        ev = Event(
            uuid=uuid4(),
            timestamp=_turn_ts(base_ts, stream_turn),
            body=Content(
                context=MessageContext(source="User"),
                items=[Text(text=msg["text"].strip())],
            ),
            properties={
                "scenario_id": sid,
                "turn_id": stream_turn,
                "speaker": "User",
                "event_type": "stream_message",
                "plant_id": f"stream_turn_{stream_turn}",
                "from_turn": stream_turn,
            },
        )
        events.append(ev)
    await memory.encode_events(events)

    info = {
        "scenario_id": sid,
        "collection_name": collection_name,
        "n_messages": len(events),
    }
    return memory, info


# ---------- Decision-point coverage annotation ----------


def annotate_step_output(
    so: dict[str, Any],
    *,
    decision_points: list[dict[str, Any]],
    stream_turns_delivered: int,
) -> dict[str, Any]:
    """Best-effort match of an emitted STEP_OUTPUT to a decision_point.

    Heuristics (qualitative, NOT for scoring):
      1. label-string contains sub_decision keywords (token-overlap >= 1)
      2. only consider DPs whose `after_turn` <= stream_turns_delivered
      3. if multiple DPs match by token, pick the one with the LATEST
         after_turn that is <= stream_turns_delivered (most recent ask)

    Returns a dict with: matched_dp_id (sub_decision label or None),
    match_reason (string), eligible_dp_ids (DPs that were active at emit
    time), unmatched_eligible (DPs that were active but didn't match this
    emit — candidate misses for iteration-2 scoring).
    """
    raw_label = (so.get("raw_label") or "").lower()
    content = (so.get("content") or "").lower()
    haystack = raw_label + " " + content

    eligible = [d for d in decision_points if d["after_turn"] <= stream_turns_delivered]
    eligible_ids = [d["sub_decision"] for d in eligible]

    candidates = []
    for d in eligible:
        sub = d["sub_decision"].lower()
        sub_tokens = [t for t in re.split(r"[^a-z0-9]+", sub) if len(t) >= 3]
        if not sub_tokens:
            continue
        hits = sum(1 for t in sub_tokens if t in haystack)
        if hits >= 1:
            candidates.append((hits, d["after_turn"], d))

    if not candidates:
        return {
            "matched_dp_id": None,
            "match_reason": "no token overlap with any active DP",
            "eligible_dp_ids": eligible_ids,
        }

    # Pick highest hits, then latest after_turn.
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    matched = candidates[0][2]
    return {
        "matched_dp_id": matched["sub_decision"],
        "match_reason": f"label/content token-overlap >= 1 with sub_decision tokens",
        "eligible_dp_ids": eligible_ids,
    }


# ---------- Agent loop ----------


@dataclass
class TurnLog:
    turn: int
    raw_excerpt: str
    raw_full: str
    stream_msg_delivered_this_turn: int | None = None  # stream turn delivered, if any
    probes: list[str] = field(default_factory=list)
    n_hits: int = 0
    new_hit_ids: list[str] = field(default_factory=list)
    n_step_outs: int = 0
    step_outs_emitted: list[dict[str, Any]] = field(default_factory=list)
    done_emitted: bool = False
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
        "probes": t.probes,
        "n_hits": t.n_hits,
        "new_hit_ids": t.new_hit_ids,
        "n_step_outs": t.n_step_outs,
        "step_outs_emitted": t.step_outs_emitted,
        "done_emitted": t.done_emitted,
        "thread_tokens_before": t.thread_tokens_before,
        "thread_tokens_after_response": t.thread_tokens_after_response,
        "thread_tokens_after_truncate": t.thread_tokens_after_truncate,
        "truncate_dropped_pairs": t.truncate_dropped_pairs,
        "truncate_dropped_msgs": t.truncate_dropped_msgs,
        "new_facts_this_turn": t.new_facts_this_turn,
    }


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

    # Initial user message uses the FIRST stream message.
    first_msg = stream_messages[0]
    stream_idx = 1  # index of the next stream message to deliver
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
    done_emitted = False
    consecutive_dry_turns = 0
    stream_closed_announced = False

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

        # --- Parse step_outputs and annotate against decision_points ---
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

            annotation = annotate_step_output(
                {"raw_label": raw_label, "content": content},
                decision_points=decision_points,
                stream_turns_delivered=stream_turns_delivered,
            )
            step_outputs_by_id[sid_int] = {
                "step_id": sid_int,
                "raw_label": raw_label,
                "label": raw_label[:200],
                "content": content,
                "turn": turn,
                "stream_turns_delivered_at_emit": stream_turns_delivered,
                "annotation": annotation,
            }
            step_outputs_log.append(
                {
                    "step_id": sid_int,
                    "raw_label": raw_label,
                    "content": content,
                    "turn": turn,
                    "stream_turns_delivered_at_emit": stream_turns_delivered,
                    "annotation": annotation,
                }
            )
            log.step_outs_emitted.append(
                {
                    "step_id": sid_int,
                    "raw_label": raw_label,
                    "len": len(content),
                    "matched_dp_id": annotation.get("matched_dp_id"),
                }
            )

        if has_done(raw):
            done_emitted = True
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

        # --- Decide if this turn delivers a new stream message ---
        # Schedule: every `stream_interval` agent turns, deliver the next
        # stream message (if any remain). Turn 1 already delivered the first
        # message via USER_INITIAL_STREAM, so first re-delivery is at turn
        # 1 + stream_interval.
        deliver_new_stream = (
            stream_idx < len(stream_messages)
            and turn % stream_interval == 0
        )

        if deliver_new_stream:
            next_msg = stream_messages[stream_idx]
            stream_idx += 1
            stream_turns_delivered = next_msg["turn"]
            log.stream_msg_delivered_this_turn = next_msg["turn"]
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
        elif stream_idx >= len(stream_messages) and not stream_closed_announced:
            # All stream messages exhausted — announce once, then standard probe-only thereafter.
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

        # Truncate.
        trunc = truncate_thread(mt_messages, MT_HARD_CAP)
        log.truncate_dropped_pairs = trunc["dropped_pairs"]
        log.truncate_dropped_msgs = trunc["dropped_msgs"]
        log.thread_tokens_after_truncate = messages_tokens(mt_messages)

        trace.append(log)

        # Only allow DONE once stream is closed; otherwise treat as
        # premature DONE and continue.
        if done_emitted and stream_closed_announced:
            break
        if done_emitted and not stream_closed_announced:
            # Premature DONE: log it but keep going. The system prompt warns
            # about this; if the model insists, we still break to respect
            # the signal — but mark it as premature in the result.
            break

    final_step_outputs = sorted(step_outputs_by_id.values(), key=lambda s: s["step_id"])

    # --- Decision-point coverage summary (qualitative annotations only) ---
    dp_coverage: list[dict[str, Any]] = []
    for dp in decision_points:
        # Was there any STEP_OUTPUT emitted whose annotation matched this DP?
        matching_emits = [
            so for so in step_outputs_log
            if (so.get("annotation") or {}).get("matched_dp_id") == dp["sub_decision"]
        ]
        # Find earliest agent-turn at which this DP became eligible (after_turn delivered).
        earliest_eligible_agent_turn = None
        # Walk through trace and find first agent turn where stream_turns_delivered_at_emit >= dp.after_turn
        cumulative_stream = stream_messages[0]["turn"]
        for t in trace:
            if t.stream_msg_delivered_this_turn is not None:
                cumulative_stream = max(cumulative_stream, t.stream_msg_delivered_this_turn)
            if cumulative_stream >= dp["after_turn"]:
                earliest_eligible_agent_turn = t.turn
                break
        dp_coverage.append(
            {
                "sub_decision": dp["sub_decision"],
                "after_turn": dp["after_turn"],
                "required_facts": dp.get("required_facts", []),
                "earliest_eligible_agent_turn": earliest_eligible_agent_turn,
                "n_matching_emits": len(matching_emits),
                "matching_emit_step_ids": [m["step_id"] for m in matching_emits],
                "first_match_agent_turn": (
                    matching_emits[0]["turn"] if matching_emits else None
                ),
            }
        )

    return {
        "scenario_id": sid,
        "variant": "mtmsg_streaming",
        "trace": [td_to_dict(t) for t in trace],
        "step_outputs": final_step_outputs,
        "step_outputs_log": step_outputs_log,
        "decision_point_coverage": dp_coverage,
        "n_turns": len(trace),
        "n_probes_total": n_probes_total,
        "n_stream_messages_delivered": stream_idx,  # at end of run
        "stream_closed_announced": stream_closed_announced,
        "max_thread_tokens": max(
            (t.thread_tokens_after_response for t in trace), default=0
        ),
        "max_thread_tokens_after_truncate": max(
            (t.thread_tokens_after_truncate for t in trace), default=0
        ),
        "total_truncate_dropped_pairs": sum(t.truncate_dropped_pairs for t in trace),
        "total_truncate_dropped_msgs": sum(t.truncate_dropped_msgs for t in trace),
        "done_emitted": done_emitted,
    }


# ---------- Driver ----------


def load_streaming_scenarios() -> list[dict]:
    return json.loads(SCENARIOS_FILE.read_text())


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
    memory, ingest_info = await ingest_streaming_scenario(
        scenario,
        vector_store=vector_store,
        segment_store=segment_store,
        embedder=embedder,
        overwrite=overwrite,
    )
    ingest_time = time.monotonic() - t0

    print(f"  [{sid}] starting variant=mtmsg_streaming", flush=True)
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
        "ingest_time_s": round(ingest_time, 2),
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
        1 for dpc in agent_result["decision_point_coverage"] if dpc["n_matching_emits"] > 0
    )
    n_dp_total = len(agent_result["decision_point_coverage"])
    print(
        f"  [{sid}] mtmsg_streaming: turns={agent_result['n_turns']} | "
        f"stream_msgs_delivered={agent_result['n_stream_messages_delivered']}/{len(scenario['messages'])} | "
        f"step_emits={n_step_emits} | "
        f"final_steps={len(agent_result['step_outputs'])} | "
        f"DP_nominal_cov={n_dp_covered}/{n_dp_total} | "
        f"probes_total={agent_result['n_probes_total']} | "
        f"max_thread={agent_result['max_thread_tokens']} | "
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
        "--seed",
        type=int,
        default=0,
        help="Reserved for future use; chat completions API doesn't accept seed for gpt-5 family but kept for parity.",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=STREAM_INTERVAL,
        help="Inject one new stream user message every N agent turns.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Reuse existing collections if present (faster reruns).",
    )
    args = parser.parse_args()

    scenarios = load_streaming_scenarios()
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

    sqlite_path = RESULTS_DIR / "eventmemory_shared_harness_v4_mtmsg_streaming.sqlite3"
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
        "stream_interval": args.stream_interval,
        "max_turns": args.max_turns,
        "scenarios": [
            {
                "scenario_id": r.get("scenario_id"),
                "error": r.get("error"),
                "n_dp_total": len(r.get("scenario_decision_points") or []),
                "n_dp_nominal_covered": (
                    sum(
                        1
                        for dpc in (r.get("agent_result") or {}).get(
                            "decision_point_coverage", []
                        )
                        if dpc["n_matching_emits"] > 0
                    )
                    if r.get("agent_result")
                    else None
                ),
                "n_turns": (r.get("agent_result") or {}).get("n_turns"),
                "stream_msgs_delivered": (r.get("agent_result") or {}).get(
                    "n_stream_messages_delivered"
                ),
                "stream_closed_announced": (r.get("agent_result") or {}).get(
                    "stream_closed_announced"
                ),
                "done_emitted": (r.get("agent_result") or {}).get("done_emitted"),
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
