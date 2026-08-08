"""Shared harness v4_mtmsg_streaming_negbaseline — negative baseline.

This is a CONTROL variant of shared_harness_v4_mtmsg_streaming. The harness
logic (scheduler, ingestion, annotation, truncation, decision-point coverage
summary) is IDENTICAL to the streaming variant.

The ONLY difference: the SYSTEM_PROMPT, USER_INITIAL, and USER_FOLLOWUP
message texts are taken verbatim from `shared_harness_v4_mtmsg_subdec_split_easy`
— the original turn-1-enumerator prompts. These prompts assume a single
up-front task statement at turn 1 and instruct the model to enumerate all
distinct sub-decisions at turn 1.

Goal: isolate whether the streaming-aware system prompt rewrite is
load-bearing, vs. whether the harness scheduling mechanism alone gives the
adaptation lift.

Adaptation rules (kept minimal so the port is faithful):
  - USER_INITIAL receives the FIRST stream message's text in place of
    {task_prompt} — subdec_split's prompt expects a task statement at turn 1.
  - Subsequent stream messages get appended into USER_FOLLOWUP via the same
    `--- INCOMING USER STREAM MESSAGE (turn N) ---` envelope the streaming
    variant uses, so the message-arrival mechanism is held identical.
  - Subdec_split's USER_FOLLOWUP wording is preserved verbatim above the
    stream-message envelope when a new stream message arrives this turn.
  - When the stream is closed we just send subdec_split's plain USER_FOLLOWUP
    without any "stream closed" cue (the turn-1 prompt has no concept of a
    stream).

Usage:
    uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming_negbaseline/main.py --scenario family-dinner-stream-01 --seed 0
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

MT_HARD_CAP = 10_000

MAX_TURNS = 24
RETRIEVE_K = 5
MAX_COMPLETION_TOKENS = 4500

STREAM_INTERVAL = 2

NAMESPACE = "arc_em_streaming_negbaseline"
COLLECTION_PREFIX = "arc_strnb"

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


# ---------- System + user prompts — VERBATIM from subdec_split_easy ----------

SYSTEM_PROMPT = """\
You are a memory-augmented agent solving a multi-step task with bounded \
working memory. You are running inside a continuous conversation thread. \
The thread is your context. When the thread crosses a token cap, the OLDEST \
user/assistant turns get hard-dropped (the system prompt stays). There is \
NO compression, NO LRU, NO citations — just hard truncation from the front.

This means: load-bearing facts you surface from memory survive ONLY as long \
as they remain in your recent reasoning. If the next probe results push old \
content past the cap, that old content is GONE. So when you find an \
important fact, restate it inline in your next THINKING — your own words \
will outlive the original retrieval text.

You have a memory tool over PAST chat history that contains specific facts \
(constraints, preferences, allergies, dates, numbers, identities) the user \
has shared previously. THESE FACTS MATERIALLY CHANGE THE TASK OUTPUT — \
without retrieving them you will write generic placeholder answers that \
miss real binding constraints. The task description ALONE does not contain \
those facts.

Memory retrieves by semantic similarity to your probe text. A probe whose \
words appear in a stored fact will surface that fact; a probe whose words \
don't match won't, even if logically related.

PROBE-GENERATION DISCIPLINES:
- **Implications and chains**: when you find a fact, ask "if this is true, \
what other fact must / probably exists alongside it?" — and probe for that \
next. Many sub-decisions need 2-5 facts combined; surface them all.
- **Optimistic cues**: when an abstract probe ("X's preference") fails, also \
probe specific plausible values ("X prefers Thursday", "X prefers 30 min"). \
If memory contains the fact in any framing, one will surface.
- **Close reading**: facts can be stated by negation, buried in narrative, \
or mentioned in passing.

OUTPUT FORMAT — each turn, emit free-text using these line patterns:

OUTPUT TYPES — purpose and consequence of each:

THINKING: <free text>
  Purpose: your private reasoning, working notes, and restated facts.
  Consequence: nothing system-side. The harness does not parse it. The \
only effect is that your future self may read it after truncation.

PROBE: <retrieval query>
  Purpose: surface stored facts that materially change the deliverables.
  Consequence: the harness runs each probe (top 5 hits by semantic \
similarity) immediately after parsing your turn, then appends the hits \
as the next user message. At most 4 PROBE lines per turn; extras are \
dropped. No across-run cap — keep probing while memory has more to give.

STEP_OUTPUT
  Format: each line starts with `STEP_OUTPUT:` (note the colon attached \
to STEP_OUTPUT), followed by a small integer id (1, 2, 3, ...), a \
colon, then the deliverable text. Example: `STEP_OUTPUT: 3: Catering \
plan — buffet for 60, two veg mains, gluten-free station, $42/head.`
  Purpose: a deliverable for one sub-decision. The id names which \
sub-decision; different sub-decisions get different ids. Re-emitting \
the same id is a REVISION (latest version replaces earlier).
  Consequence: at task end, the latest STEP_OUTPUT per id is shipped \
verbatim to the user as the final deliverable for that sub-decision. \
The system does not edit or fill it in.

DONE
  Purpose: signal that you have nothing more worth retrieving and your \
STEP_OUTPUTs reflect the work as well as you can do it.
  Consequence: emitting `DONE` ends the run; the recorded STEP_OUTPUTs \
are what the user sees.

You may emit any combination of THINKING / PROBE / STEP_OUTPUT lines per \
turn, optionally followed by DONE. After your turn the harness parses \
PROBE lines, runs retrieval, and either appends the next user message \
(with hits) or terminates if you emitted DONE.
"""


USER_INITIAL = """\
TASK:
{task_prompt}

This is turn 1. You have no probe results yet. Before crafting probes, \
in your THINKING list the distinct sub-decisions you'll need to make as \
the work unfolds. Each sub-decision will become its own STEP_OUTPUT \
(different ids), so list them at the right granularity — split a topic \
into multiple sub-decisions when each will need its own deliverable, \
keep them together when they're a single unit. Then, for each \
sub-decision, ask: what implicit constraints — facts the user has \
shared previously that the task description doesn't mention but that \
materially shape the answer — apply? (For example, planning a banquet \
implicitly requires guest allergies; a presentation implicitly requires \
brand guidelines.) Your probes should target both the obvious facts and \
the implicit-constraint facts behind each listed sub-decision.
"""


USER_FOLLOWUP = """\
Turn {turn}. New probe results from memory:
---
{snippets}
---

Probes used this run: {n_probes_total}. New useful facts surfaced this \
turn: {new_facts_this_turn}. Consecutive turns with zero new useful \
facts: {consecutive_dry_turns}.
"""


# Tail block appended to USER_FOLLOWUP when a new stream message arrives this
# turn. We KEEP the same `--- INCOMING USER STREAM MESSAGE ---` envelope as
# the streaming variant so the message-arrival mechanism is held identical;
# the negative baseline is purely about the prompt-text framing, not the
# scheduler. (subdec_split's USER_FOLLOWUP says nothing about new messages;
# we just paste the next message after the standard followup body.)
STREAM_MSG_TAIL = """\

--- INCOMING USER STREAM MESSAGE (turn {stream_turn}) ---
{stream_text}
---
"""


# ---------- Thread truncation (verbatim) ----------


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


# ---------- Streaming-scenario ingestion (verbatim from streaming) ----------


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


# ---------- Decision-point coverage annotation (verbatim) ----------


def annotate_step_output(
    so: dict[str, Any],
    *,
    decision_points: list[dict[str, Any]],
    stream_turns_delivered: int,
) -> dict[str, Any]:
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
    stream_msg_delivered_this_turn: int | None = None
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

    # First stream message becomes the {task_prompt} in subdec_split's
    # USER_INITIAL — that prompt expects a single up-front task statement.
    first_msg = stream_messages[0]
    stream_idx = 1
    stream_turns_delivered = first_msg["turn"]

    mt_messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_INITIAL.format(task_prompt=first_msg["text"]),
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

        # Schedule (verbatim from streaming): every `stream_interval` agent
        # turns, deliver next stream message.
        deliver_new_stream = (
            stream_idx < len(stream_messages)
            and turn % stream_interval == 0
        )

        # Build the followup using subdec_split's USER_FOLLOWUP body. If a new
        # stream message is delivered this turn, append the stream-message
        # envelope (same envelope as streaming variant) AS-IS — no extra
        # framing instructions, since subdec_split's prompt has no concept
        # of one.
        followup_body = USER_FOLLOWUP.format(
            turn=turn + 1,
            snippets=snippets_block,
            n_probes_total=n_probes_total,
            new_facts_this_turn=new_facts,
            consecutive_dry_turns=consecutive_dry_turns,
        )

        if deliver_new_stream:
            next_msg = stream_messages[stream_idx]
            stream_idx += 1
            stream_turns_delivered = next_msg["turn"]
            log.stream_msg_delivered_this_turn = next_msg["turn"]
            content = followup_body + STREAM_MSG_TAIL.format(
                stream_turn=next_msg["turn"], stream_text=next_msg["text"]
            )
            mt_messages.append({"role": "user", "content": content})
        elif stream_idx >= len(stream_messages) and not stream_closed_announced:
            # All stream messages exhausted. Subdec_split has no
            # "stream closed" cue, so we just send the standard followup
            # going forward. Mark it so the loop knows DONE is permitted.
            stream_closed_announced = True
            mt_messages.append({"role": "user", "content": followup_body})
        else:
            mt_messages.append({"role": "user", "content": followup_body})

        trunc = truncate_thread(mt_messages, MT_HARD_CAP)
        log.truncate_dropped_pairs = trunc["dropped_pairs"]
        log.truncate_dropped_msgs = trunc["dropped_msgs"]
        log.thread_tokens_after_truncate = messages_tokens(mt_messages)

        trace.append(log)

        # Honor DONE the same way as streaming: only allow termination once
        # stream is exhausted. If model emits DONE early, break and mark.
        if done_emitted and stream_closed_announced:
            break
        if done_emitted and not stream_closed_announced:
            break

    final_step_outputs = sorted(step_outputs_by_id.values(), key=lambda s: s["step_id"])

    dp_coverage: list[dict[str, Any]] = []
    for dp in decision_points:
        matching_emits = [
            so for so in step_outputs_log
            if (so.get("annotation") or {}).get("matched_dp_id") == dp["sub_decision"]
        ]
        earliest_eligible_agent_turn = None
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
        "variant": "mtmsg_streaming_negbaseline",
        "trace": [td_to_dict(t) for t in trace],
        "step_outputs": final_step_outputs,
        "step_outputs_log": step_outputs_log,
        "decision_point_coverage": dp_coverage,
        "n_turns": len(trace),
        "n_probes_total": n_probes_total,
        "n_stream_messages_delivered": stream_idx,
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

    print(f"  [{sid}] starting variant=mtmsg_streaming_negbaseline", flush=True)
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
        f"  [{sid}] mtmsg_streaming_negbaseline: turns={agent_result['n_turns']} | "
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
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
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

    sqlite_path = (
        RESULTS_DIR / "eventmemory_shared_harness_v4_mtmsg_streaming_negbaseline.sqlite3"
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
