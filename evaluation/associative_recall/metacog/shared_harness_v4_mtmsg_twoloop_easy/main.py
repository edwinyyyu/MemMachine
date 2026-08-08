"""Shared harness v4_mtmsg_twoloop_easy — two-loop architecture (Option B).

Builds on shared_harness_v4_mtmsg_subdec_split_easy (turn-1 sub-decision
enumeration + implicit-constraint reflection). Adds DECISION_SPAWN /
DECISION_CLOSE markers that bracket "inner sessions" inline within the same
mt_messages thread.

ARCHITECTURE (Option B — lightweight, in-thread):
  Outer loop = the existing subdec_split coordinator: turn-1 enumerates the
  initial sub-decisions and probes implicit-constraint facts.
  Inner loop = a new sub-decision session, spawned mid-thread when the model
  emits `DECISION_SPAWN: <id>: <one-line summary>`. The harness injects, into
  the NEXT user followup, a FRESH turn-1-style directive (the full
  USER_INITIAL enumeration text) but scoped to the sub-decision summary
  ("STARTING SUB-TASK: <summary>. Within this sub-task, ..."). The model is
  expected to engage with the directive AS A NEW TURN-1 for that sub-task —
  enumerate its own further sub-decisions, ask the implicit-constraint
  question, then probe.
  Closing the sub-task with `DECISION_CLOSE: <id>` triggers a brief
  "[sub-task <id> closed; resume coordinating overall task]" injection.

KEY DIFFERENCE FROM boundaries_easy:
  Boundaries injected a SHORT directive ("[on opening: identify implicit
  constraints]"). Two-loop injects the FULL turn-1 enumeration directive,
  scoped to the sub-decision. The hypothesis: each new sub-decision deserves
  its OWN fresh turn-1 priming, uncontaminated by retrieval context that has
  accumulated. Pergate (per-emission gating) and cogonly (per-turn cog) both
  regressed — they nudged at every turn. Two-loop only nudges at sub-decision
  BOUNDARIES, but with a much LARGER directive than boundaries_easy.

GUARDRAILS — what this is NOT:
- NOT per-emission gating (no IMPLICIT_CONSTRAINTS_PROBE output type).
- NOT per-turn cog reflection (USER_FOLLOWUP unchanged from subdec_split base
  except for the spawn/close directive injection slot).
- ONLY a marker mechanism + harness-side directive injection at boundaries.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from memmachine_server.common.vector_store.qdrant_vector_store import (  # noqa: E402
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (  # noqa: E402
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from mid_execution_eval import (  # type: ignore  # noqa: E402
    RESULTS_DIR,
    ingest_scenario,
    load_locomo_segments,
    load_scenarios,
    load_speakers,
    probe,
)
from qdrant_client import AsyncQdrantClient  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
# Allow caller to override results subdir (e.g., results_run2 for the n=2 seed).
_RESULTS_SUBDIR = os.environ.get("SH_RESULTS_SUBDIR", "results")
RESULTS_OUT_DIR = THIS_DIR / _RESULTS_SUBDIR
RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = _AR_DIR / ".env"
load_dotenv(ENV_PATH)

MODEL = "gpt-5-mini"

# WM cap on the mt_messages thread. When exceeded, oldest user/assistant pairs
# get dropped (system stays).
MT_HARD_CAP = 10_000

MAX_TURNS = 14
RETRIEVE_K = 5
MAX_COMPLETION_TOKENS = 4500
SCORE_K_LIST = [1, 3, 5, 10]

try:
    ENC = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(n_tokens(m.get("content", "") or "") for m in messages)


# ---------- Line parsers ----------

PROBE_LINE_RE = re.compile(r"^\s*PROBE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
DONE_LINE_RE = re.compile(r"^\s*DONE\s*$", re.MULTILINE | re.IGNORECASE)
# STEP_OUTPUT header: "STEP_OUTPUT: <id_or_label>: <content...>" or
# "STEP_OUTPUT: <id_or_label>\n<body>".
STEP_OUTPUT_HEAD_RE = re.compile(
    r"^\s*STEP_OUTPUT\s*:\s*([^\n:\-]+?)(?:\s*[:\-]\s*(.*))?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
# DECISION_SPAWN: <id>: <one-line summary>
DECISION_SPAWN_RE = re.compile(
    r"^\s*DECISION_SPAWN\s*:\s*([^\n:\-]+?)\s*[:\-]\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
# DECISION_CLOSE: <id>
DECISION_CLOSE_RE = re.compile(
    r"^\s*DECISION_CLOSE\s*:\s*([^\n:\-]+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
DIRECTIVE_LINE_RE = re.compile(
    r"^\s*(THINKING|PROBE|STEP_OUTPUT|DECISION_SPAWN|DECISION_CLOSE|DONE)\b",
    re.MULTILINE | re.IGNORECASE,
)


def parse_step_outputs(raw: str) -> list[dict[str, Any]]:
    """Parse STEP_OUTPUT: <id_or_label>[: <content>] blocks."""
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


def parse_decision_spawns(raw: str) -> list[dict[str, str]]:
    """Parse DECISION_SPAWN: <id>: <summary> lines."""
    out: list[dict[str, str]] = []
    for m in DECISION_SPAWN_RE.finditer(raw or ""):
        did = (m.group(1) or "").strip()
        summary = (m.group(2) or "").strip()
        if did:
            out.append({"id": did, "summary": summary})
    return out


def parse_decision_closes(raw: str) -> list[str]:
    """Parse DECISION_CLOSE: <id> lines."""
    return [m.group(1).strip() for m in DECISION_CLOSE_RE.finditer(raw or "")]


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


# ---------- System prompt ----------

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

DECISION_SPAWN: <id>: <one-line summary>
  Purpose: formally START work on a NEW sub-decision. Use this when you \
recognize the work has shifted to a new sub-decision area that will \
eventually have its own deliverable, AND you want a fresh turn-1-style \
priming for it. The id is a short stable handle (a small integer like \
`1`, `2`, `3`, OR a short string label like `catering`) that you will \
reuse on the matching STEP_OUTPUT and DECISION_CLOSE. The summary is a \
brief description of what this sub-decision must produce.
  Consequence: the harness records the spawn. The FIRST time it sees a \
given id spawned, it will inject — into the next user message — a \
FRESH turn-1-style enumeration directive scoped to the sub-decision \
summary. Engage with that directive AS A NEW TURN-1 for the sub-task: \
list its further sub-decisions in your THINKING, ask the \
implicit-constraint question for each, then probe both the obvious \
facts and the implicit-constraint facts behind that sub-task.

DECISION_CLOSE: <id>
  Purpose: mark that the sub-decision identified by <id> is complete \
(its STEP_OUTPUT reflects the work as well as you can do for now). The \
id must match a previously-spawned DECISION_SPAWN.
  Consequence: the harness injects a brief "sub-task closed; resume \
coordinating overall task" line into the next user message. Closing is \
not strictly required (closing all decisions is implied by emitting \
DONE), but explicit closes help keep your bookkeeping clean.

STEP_OUTPUT
  Format: each line starts with `STEP_OUTPUT:` (note the colon attached \
to STEP_OUTPUT), followed by an id (matching a DECISION_SPAWN id you \
previously emitted, or a small integer if you skipped spawning), a \
colon, then the deliverable text. Example: `STEP_OUTPUT: 3: Catering \
plan — buffet for 60, two veg mains, gluten-free station, $42/head.`
  Purpose: a deliverable for one sub-decision. The id names which \
sub-decision; different sub-decisions get different ids. Re-emitting \
the same id is a REVISION (latest version replaces earlier).
  Consequence: at task end, the latest STEP_OUTPUT per id is shipped \
verbatim to the user as the final deliverable for that sub-decision.

DONE
  Purpose: signal that you have nothing more worth retrieving and your \
STEP_OUTPUTs reflect the work as well as you can do it.
  Consequence: emitting `DONE` ends the run; the recorded STEP_OUTPUTs \
are what the user sees.

TWO-LOOP DISCIPLINE:
The OUTER loop coordinates the overall task — that's what you start in \
on turn 1. INNER loops are per-sub-decision sessions you formally open \
with `DECISION_SPAWN`. When you spawn, the harness gives you a fresh \
turn-1-style enumeration directive scoped to that sub-decision; treat \
that injected directive as a new "turn 1" — re-enumerate further \
sub-decisions WITHIN the spawned sub-task, ask the implicit-constraint \
question for each, and craft probes targeting both obvious and \
implicit-constraint facts of THIS sub-task. Close the inner loop with \
`DECISION_CLOSE: <id>` when its STEP_OUTPUT is complete; the harness \
will then prompt you to resume coordinating the overall task.

You may emit any combination of THINKING / PROBE / DECISION_SPAWN / \
DECISION_CLOSE / STEP_OUTPUT lines per turn, optionally followed by \
DONE. After your turn the harness parses PROBE lines, runs retrieval, \
and either appends the next user message (with hits + any spawn/close \
directives) or terminates if you emitted DONE.
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

If during work you discover NEW sub-decisions not in your initial \
enumeration, formally start them with `DECISION_SPAWN: <id>: <summary>` \
— the harness will inject a fresh turn-1-style directive scoped to that \
sub-task, which lets you re-enumerate further sub-decisions and \
implicit-constraint facts WITHIN the spawned sub-task before probing it.
"""


# Inner-loop turn-1 directive, injected by the harness at DECISION_SPAWN
# boundaries. Mirrors USER_INITIAL's structure (full enumeration directive)
# but is scoped to the spawned sub-decision rather than the overall task.
SPAWN_DIRECTIVE_TEMPLATE = """\

[STARTING SUB-TASK {decision_id}: "{summary}". Within this sub-task, \
treat this as a fresh turn-1 for the inner loop. Before crafting \
further probes for this sub-task, in your THINKING list the distinct \
sub-decisions you'll need to make WITHIN this sub-task as its work \
unfolds. Split a sub-task topic into multiple sub-decisions when each \
will need its own deliverable, keep them together when they're a \
single unit. Then, for each, ask: what implicit constraints — facts \
the user has shared previously that the task description doesn't \
mention but that materially shape the answer — apply to THIS \
sub-task? (For example, planning a banquet implicitly requires guest \
allergies; a presentation implicitly requires brand guidelines.) Your \
probes for this sub-task should target both the obvious facts and the \
implicit-constraint facts behind it.]"""


# Brief close-injection. Returns the model to outer-loop coordination mode.
CLOSE_DIRECTIVE_TEMPLATE = (
    "\n\n[sub-task {decision_id} closed; resume coordinating overall task]"
)


USER_FOLLOWUP = """\
Turn {turn}. New probe results from memory:
---
{snippets}
---

Probes used this run: {n_probes_total}. New useful facts surfaced this \
turn: {new_facts_this_turn}. Consecutive turns with zero new useful \
facts: {consecutive_dry_turns}.{spawn_close_directives}
"""


# ---------- Thread truncation ----------


def truncate_thread(messages: list[dict[str, str]], cap: int) -> dict[str, Any]:
    """Drop oldest user/assistant pairs from the front until messages fit cap."""
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


# ---------- Agent loop ----------


@dataclass
class TurnLog:
    turn: int
    raw_excerpt: str
    raw_full: str
    probes: list[str] = field(default_factory=list)
    n_hits: int = 0
    new_hit_ids: list[str] = field(default_factory=list)
    n_step_outs: int = 0
    step_outs_emitted: list[dict[str, Any]] = field(default_factory=list)
    decision_spawns: list[dict[str, str]] = field(default_factory=list)
    new_decision_spawns: list[dict[str, str]] = field(default_factory=list)
    decision_closes: list[str] = field(default_factory=list)
    spawn_directives_injected: list[str] = field(default_factory=list)
    close_directives_injected: list[str] = field(default_factory=list)
    currently_active_decision_id_at_turn_start: str | None = None
    done_emitted: bool = False
    thread_tokens_before: int = 0
    thread_tokens_after_response: int = 0
    thread_tokens_after_truncate: int = 0
    truncate_dropped_pairs: int = 0
    truncate_dropped_msgs: int = 0
    new_facts_this_turn: int = 0


async def run_agent_loop(
    *,
    scenario: dict,
    memory,
    openai_client,
) -> dict[str, Any]:
    task_prompt = scenario["task_prompt"]
    sid = scenario["scenario_id"]

    mt_messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_INITIAL.format(task_prompt=task_prompt)},
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

    # Two-loop tracking.
    # spawned_decisions: ids ever spawned (so we know what's "new")
    # currently_open_decisions: ids spawned-but-not-closed (LIFO; we treat
    #   the most recently-spawned-still-open as currently_active_decision_id
    #   for trace visibility)
    # decision_summaries: id -> first-emitted summary (for directive text)
    spawned_decisions: set[str] = set()
    currently_open_decisions: list[str] = []  # stack-ordered
    decision_summaries: dict[str, str] = {}
    n_decision_spawns_total = 0
    n_decision_closes_total = 0
    n_spawn_directives_total = 0
    n_close_directives_total = 0

    for turn in range(1, MAX_TURNS + 1):
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
                currently_active_decision_id_at_turn_start=(
                    currently_open_decisions[-1] if currently_open_decisions else None
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
            currently_active_decision_id_at_turn_start=(
                currently_open_decisions[-1] if currently_open_decisions else None
            ),
        )

        # Parse decision spawn/close markers FIRST so we can plan the
        # directive injection for the NEXT user followup.
        decision_spawns = parse_decision_spawns(raw)
        decision_closes = parse_decision_closes(raw)
        log.decision_spawns = decision_spawns
        log.decision_closes = decision_closes
        n_decision_spawns_total += len(decision_spawns)
        n_decision_closes_total += len(decision_closes)

        new_spawns_this_turn: list[dict[str, str]] = []
        for dsp in decision_spawns:
            did = dsp["id"]
            summary = dsp["summary"]
            if did not in spawned_decisions:
                spawned_decisions.add(did)
                decision_summaries[did] = summary
                new_spawns_this_turn.append({"id": did, "summary": summary})
            if did not in currently_open_decisions:
                currently_open_decisions.append(did)
        log.new_decision_spawns = new_spawns_this_turn

        closed_this_turn: list[str] = []
        for did in decision_closes:
            if did in currently_open_decisions:
                currently_open_decisions.remove(did)
                closed_this_turn.append(did)

        # Parse step_outputs.
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
            step_outputs_by_id[sid_int] = {
                "step_id": sid_int,
                "raw_label": raw_label,
                "label": raw_label[:200],
                "content": content,
                "turn": turn,
            }
            step_outputs_log.append(
                {
                    "step_id": sid_int,
                    "raw_label": raw_label,
                    "content": content,
                    "turn": turn,
                }
            )
            log.step_outs_emitted.append(
                {"step_id": sid_int, "raw_label": raw_label, "len": len(content)}
            )

        # Parse done.
        if has_done(raw):
            done_emitted = True
            log.done_emitted = True

        # Parse probes and run retrieval.
        probes = parse_probes(raw)
        probes = probes[:4]
        log.probes = probes

        # Build the spawn/close directive block to APPEND to the next
        # followup message. Spawns inject the FULL turn-1-style directive
        # scoped to the sub-decision; closes inject a brief resume note.
        directive_parts: list[str] = []
        if new_spawns_this_turn:
            for nsp in new_spawns_this_turn:
                directive_parts.append(
                    SPAWN_DIRECTIVE_TEMPLATE.format(
                        decision_id=nsp["id"],
                        summary=nsp["summary"],
                    )
                )
            log.spawn_directives_injected = [nsp["id"] for nsp in new_spawns_this_turn]
            n_spawn_directives_total += len(new_spawns_this_turn)
        if closed_this_turn:
            for did in closed_this_turn:
                directive_parts.append(
                    CLOSE_DIRECTIVE_TEMPLATE.format(decision_id=did)
                )
            log.close_directives_injected = list(closed_this_turn)
            n_close_directives_total += len(closed_this_turn)
        spawn_close_block = "".join(directive_parts)

        if probes:
            try:
                hits_lists = await asyncio.gather(
                    *[probe(memory, p, RETRIEVE_K) for p in probes]
                )
            except Exception as exc:
                hits_lists = [[] for _ in probes]
                log.raw_excerpt = (log.raw_excerpt or "") + f" | RET ERR: {exc!r}"[:120]

            n_probes_total += len(probes)
            new_snippets: list[str] = []
            new_hit_ids: list[str] = []
            new_facts = 0
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

            if not new_snippets:
                snippets_block = (
                    "(no new snippets surfaced — memory may be silent on these probes)"
                )
            else:
                snippets_block = "\n".join(f"- {s}" for s in new_snippets)

            if new_facts == 0:
                consecutive_dry_turns += 1
            else:
                consecutive_dry_turns = 0

            mt_messages.append(
                {
                    "role": "user",
                    "content": USER_FOLLOWUP.format(
                        turn=turn + 1,
                        snippets=snippets_block,
                        n_probes_total=n_probes_total,
                        new_facts_this_turn=new_facts,
                        consecutive_dry_turns=consecutive_dry_turns,
                        spawn_close_directives=spawn_close_block,
                    ),
                }
            )
        else:
            consecutive_dry_turns += 1
            if not done_emitted:
                mt_messages.append(
                    {
                        "role": "user",
                        "content": USER_FOLLOWUP.format(
                            turn=turn + 1,
                            snippets="(no probes emitted last turn)",
                            n_probes_total=n_probes_total,
                            new_facts_this_turn=0,
                            consecutive_dry_turns=consecutive_dry_turns,
                            spawn_close_directives=spawn_close_block,
                        ),
                    }
                )

        # Hard-truncate if over cap.
        trunc = truncate_thread(mt_messages, MT_HARD_CAP)
        log.truncate_dropped_pairs = trunc["dropped_pairs"]
        log.truncate_dropped_msgs = trunc["dropped_msgs"]
        log.thread_tokens_after_truncate = messages_tokens(mt_messages)

        trace.append(log)

        if done_emitted:
            break

    final_step_outputs = sorted(step_outputs_by_id.values(), key=lambda s: s["step_id"])

    # Compliance metrics: count STEP_OUTPUTs whose raw_label was never a
    # spawned decision id (orphans).
    orphan_step_labels: list[str] = []
    for so in final_step_outputs:
        if so["raw_label"] not in spawned_decisions:
            orphan_step_labels.append(so["raw_label"])
    n_step_outs_final = len(final_step_outputs)
    n_step_outs_orphan = len(orphan_step_labels)

    # Engagement signal: did each spawn directive actually generate a
    # follow-up turn whose THINKING/probes engaged with the sub-task? Hard
    # to score perfectly without an LLM judge; surface a coarse proxy:
    # for each spawn-id, did probes appear in the same turn as the spawn
    # OR in the next turn? (We log spawn turns and probe-turn-after; the
    # offline analyzer can assemble final engagement metrics.)
    spawn_engagement_turns: list[dict[str, Any]] = []
    for tlog in trace:
        for nsp in tlog.new_decision_spawns:
            spawn_engagement_turns.append(
                {
                    "decision_id": nsp["id"],
                    "summary": nsp["summary"],
                    "spawn_turn": tlog.turn,
                }
            )

    return {
        "scenario_id": sid,
        "variant": "mtmsg_capped_twoloop",
        "trace": [td_to_dict(t) for t in trace],
        "step_outputs": final_step_outputs,
        "step_outputs_log": step_outputs_log,
        "n_turns": len(trace),
        "n_probes_total": n_probes_total,
        "n_decision_spawns_total": n_decision_spawns_total,
        "n_decision_closes_total": n_decision_closes_total,
        "n_unique_decisions_spawned": len(spawned_decisions),
        "n_spawn_directives_injected": n_spawn_directives_total,
        "n_close_directives_injected": n_close_directives_total,
        "spawn_engagement_turns": spawn_engagement_turns,
        "n_step_outs_final": n_step_outs_final,
        "n_step_outs_orphan": n_step_outs_orphan,
        "orphan_step_labels": orphan_step_labels,
        "currently_open_at_end": list(currently_open_decisions),
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


def td_to_dict(t: TurnLog) -> dict[str, Any]:
    return {
        "turn": t.turn,
        "raw_excerpt": t.raw_excerpt,
        "raw_full": t.raw_full,
        "probes": t.probes,
        "n_hits": t.n_hits,
        "new_hit_ids": t.new_hit_ids,
        "n_step_outs": t.n_step_outs,
        "step_outs_emitted": t.step_outs_emitted,
        "decision_spawns": t.decision_spawns,
        "new_decision_spawns": t.new_decision_spawns,
        "decision_closes": t.decision_closes,
        "spawn_directives_injected": t.spawn_directives_injected,
        "close_directives_injected": t.close_directives_injected,
        "currently_active_decision_id_at_turn_start": (
            t.currently_active_decision_id_at_turn_start
        ),
        "done_emitted": t.done_emitted,
        "thread_tokens_before": t.thread_tokens_before,
        "thread_tokens_after_response": t.thread_tokens_after_response,
        "thread_tokens_after_truncate": t.thread_tokens_after_truncate,
        "truncate_dropped_pairs": t.truncate_dropped_pairs,
        "truncate_dropped_msgs": t.truncate_dropped_msgs,
        "new_facts_this_turn": t.new_facts_this_turn,
    }


# ---------- Coverage judge (verbatim from v3) ----------

COVERAGE_JUDGE_PROMPT = """\
You are evaluating whether an executor agent's transcript addresses a \
specific sub-decision that a competent worker would have made for this task.

GOLD SUB-DECISION (a sub-step a competent worker should have addressed):
"{decision_text}"

GOLD FACT (a past-context fact that should have informed this sub-decision; \
you don't need to check whether the agent USED the fact, only whether the \
agent's transcript made a concrete decision in this area):
"{plant_text}"

AGENT'S STEP OUTPUTS (PLAN + EXECUTE deliverables):
---
{transcript}
---

Question 1: Did the agent address the gold sub-decision area in their \
transcript? "Addressed" means: somewhere in the step outputs the agent made \
a concrete choice or wrote content directly relevant to this sub-decision \
area. Whether they used the gold fact is NOT what's being judged here — \
only whether the decision area was covered at all.

Question 2: If yes, which `step_id` does it primarily correspond to? Use the \
integer step_id from the agent's step_outputs section. If the agent split it \
across multiple steps, pick the one that addresses it most directly. If no \
clear step, return "no_step_label".

Output ONLY a JSON object, no prose:
{{"addressed": true|false, "step_label": <integer> | "no_step_label" | null, "evidence_quote": "<short quote (<=140 chars) from transcript that addresses the decision, or empty string>"}}
"""


def parse_judge_response(text: str) -> dict[str, Any] | None:
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


async def judge_coverage(
    *,
    transcript: str,
    decision_text: str,
    plant_text: str,
    openai_client,
) -> dict[str, Any]:
    prompt = COVERAGE_JUDGE_PROMPT.format(
        transcript=transcript,
        decision_text=decision_text,
        plant_text=plant_text,
    )
    resp = await openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4500,
    )
    raw = resp.choices[0].message.content or ""
    parsed = parse_judge_response(raw) or {}
    return {
        "addressed": bool(parsed.get("addressed", False)),
        "step_label": parsed.get("step_label"),
        "evidence_quote": str(parsed.get("evidence_quote", ""))[:200],
        "raw": raw,
    }


# ---------- Score ----------


async def score_variant(
    *,
    scenario: dict,
    agent_result: dict[str, Any],
    memory,
    openai_client,
    K_list: list[int],
) -> dict[str, Any]:
    plants_by_id = {
        p["plant_id"]: p for p in scenario["preamble_turns"] if p.get("plant_id")
    }

    step_outputs = agent_result["step_outputs"]
    steps_by_id: dict[int, dict[str, Any]] = {
        int(so["step_id"]): so for so in step_outputs
    }

    transcript_parts = []
    transcript_parts.append("PLAN:")
    for so in step_outputs:
        transcript_parts.append(f"  [STEP {so['step_id']}] {so['label']}")
    transcript_parts.append("\nEXECUTE:")
    for so in step_outputs:
        transcript_parts.append(f"--- STEP {so['step_id']} ---")
        transcript_parts.append(so["content"])
    transcript = "\n".join(transcript_parts)

    gold_steps = [s for s in scenario["subdecision_script"] if s.get("gold_plant_ids")]

    async def _judge_one(gold_step):
        first_plant = plants_by_id.get(gold_step["gold_plant_ids"][0])
        plant_text = first_plant["text"] if first_plant else ""
        j = await judge_coverage(
            transcript=transcript,
            decision_text=gold_step["decision_text"],
            plant_text=plant_text,
            openai_client=openai_client,
        )
        return gold_step, j

    judgements = await asyncio.gather(*[_judge_one(g) for g in gold_steps])

    K_max = max(K_list)
    per_gold: list[dict[str, Any]] = []
    for gold_step, judgement in judgements:
        entry: dict[str, Any] = {
            "gold_step_id": gold_step["step_id"],
            "gold_decision_text": gold_step["decision_text"],
            "gold_plant_ids": gold_step["gold_plant_ids"],
            "addressed": judgement["addressed"],
            "judge_step_label": judgement["step_label"],
            "judge_evidence_quote": judgement["evidence_quote"],
        }
        for K in K_list:
            entry[f"triggered_recall_full@{K}"] = 0.0
            entry[f"recall_given_covered@{K}"] = None

        if judgement["addressed"]:
            label = judgement["step_label"]
            step_rec = steps_by_id.get(label) if isinstance(label, int) else None
            if step_rec:
                cue_text = step_rec.get("content") or step_rec.get("label") or ""
            else:
                cue_text = judgement["evidence_quote"] or ""
            entry["cue_used"] = cue_text[:200]
            entry["cue_source"] = "step_content" if step_rec else "evidence_quote"

            if cue_text.strip():
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
                for K in K_list:
                    topK_found = {h.plant_id for h in hits[:K] if h.plant_id}
                    rec = sum(
                        1 for g in gold_step["gold_plant_ids"] if g in topK_found
                    ) / len(gold_step["gold_plant_ids"])
                    entry[f"recall_given_covered@{K}"] = rec
                    entry[f"triggered_recall_full@{K}"] = rec
            else:
                entry["top_hits"] = []
        per_gold.append(entry)

    n_gold = len(per_gold)
    n_addressed = sum(1 for e in per_gold if e["addressed"])
    coverage_rate = n_addressed / n_gold if n_gold else 0.0
    agg: dict[str, Any] = {"coverage_rate": round(coverage_rate, 4)}
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
    return {
        "per_gold": per_gold,
        "aggregates": agg,
        "transcript_excerpt": transcript[:1500],
    }


# ---------- Driver ----------


async def run_one_scenario(
    *,
    scenario: dict,
    locomo_segments,
    speakers_map,
    vector_store,
    segment_store,
    embedder,
    openai_client,
    K_list: list[int],
    overwrite: bool = True,
) -> dict[str, Any]:
    sid = scenario["scenario_id"]
    base_conv = scenario["base_conversation"]
    locomo_turns = locomo_segments[base_conv]
    speakers = speakers_map.get(base_conv) or {}

    extra_distractor_runs = []
    for extra_conv in scenario.get("extra_base_conversations") or []:
        extra_distractor_runs.append(
            (locomo_segments[extra_conv], speakers_map.get(extra_conv) or {})
        )

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

    print(f"  [{sid}] starting variant=mtmsg_capped_twoloop", flush=True)
    agent_result = await run_agent_loop(
        scenario=scenario,
        memory=memory,
        openai_client=openai_client,
    )
    score = await score_variant(
        scenario=scenario,
        agent_result=agent_result,
        memory=memory,
        openai_client=openai_client,
        K_list=K_list,
    )
    per = {**agent_result, **score}
    fp = RESULTS_OUT_DIR / f"{sid}.json"
    fp.write_text(json.dumps(per, indent=2, default=str))

    agg = score["aggregates"]
    cov = agg["coverage_rate"]
    full = agg.get("triggered_recall_full@5", "n/a")
    cond = agg.get("recall_given_covered@5", "n/a")
    n_step_emits = sum(t.get("n_step_outs", 0) for t in agent_result["trace"])
    print(
        f"  [{sid}] twoloop: cov={cov} | full_R@5={full} | cond_R@5={cond} | "
        f"turns={agent_result['n_turns']} | step_emits={n_step_emits} | "
        f"final_steps={len(agent_result['step_outputs'])} | "
        f"probes_total={agent_result['n_probes_total']} | "
        f"spawns={agent_result['n_decision_spawns_total']} | "
        f"unique_decisions={agent_result['n_unique_decisions_spawned']} | "
        f"closes={agent_result['n_decision_closes_total']} | "
        f"spawn_dirs={agent_result['n_spawn_directives_injected']} | "
        f"close_dirs={agent_result['n_close_directives_injected']} | "
        f"orphans={agent_result['n_step_outs_orphan']}/{agent_result['n_step_outs_final']} | "
        f"max_thread={agent_result['max_thread_tokens']} | "
        f"trunc_pairs={agent_result['total_truncate_dropped_pairs']} | "
        f"done={agent_result['done_emitted']}",
        flush=True,
    )

    return {
        "scenario_id": sid,
        "category": scenario.get("category", ""),
        "ingest_time_s": round(ingest_time, 2),
        "ingest_info": ingest_info,
        "K_list": K_list,
        "per_variant": {"mtmsg_capped": per},
    }


async def main() -> None:
    # 10 easier scenarios — indices 0-9 of mid_execution_scenarios.json.
    # Allow env override SH_SCENARIO_INDICES (comma-separated ints) for
    # subset/smoke runs.
    _idx_override = os.environ.get("SH_SCENARIO_INDICES", "").strip()
    if _idx_override:
        SCENARIO_INDICES = [int(x) for x in _idx_override.split(",") if x.strip()]
    else:
        SCENARIO_INDICES = list(range(10))
    K_list = SCORE_K_LIST

    scenarios_all = load_scenarios()
    scenarios = [scenarios_all[i] for i in SCENARIO_INDICES]
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

    # Per-seed DB isolation: SH_DB_SUFFIX appended to the SQLite filename so
    # multiple seeds can run safely without colliding on the same file.
    _DB_SUFFIX = os.environ.get("SH_DB_SUFFIX", "")
    sqlite_path = (
        RESULTS_DIR
        / f"eventmemory_shared_harness_v4_mtmsg_twoloop{_DB_SUFFIX}.sqlite3"
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
    SCEN_SEM = asyncio.Semaphore(int(os.getenv("SH_SCEN_CONCURRENCY", "1")))

    async def _run_scen(scenario: dict) -> dict[str, Any]:
        async with SCEN_SEM:
            sid = scenario["scenario_id"]
            print(f"[run] {sid} (variant=mtmsg_capped_twoloop)", flush=True)
            try:
                return await run_one_scenario(
                    scenario=scenario,
                    locomo_segments=locomo_segments,
                    speakers_map=speakers_map,
                    vector_store=vector_store,
                    segment_store=segment_store,
                    embedder=embedder,
                    openai_client=openai_client,
                    K_list=K_list,
                )
            except Exception as exc:
                print(f"  ERROR {sid}: {exc!r}", flush=True)
                return {"scenario_id": sid, "error": repr(exc)}

    try:
        results = await asyncio.gather(*[_run_scen(s) for s in scenarios])
    finally:
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    summary: dict[str, Any] = {
        "n_scenarios": len(results),
        "scenarios": [],
        "per_variant_means": {},
    }
    for r in results:
        if "error" in r:
            summary["scenarios"].append(
                {"scenario_id": r["scenario_id"], "error": r["error"]}
            )
            continue
        agent_result = r["per_variant"]["mtmsg_capped"]
        agg = agent_result["aggregates"]
        n_step_emits = sum(t.get("n_step_outs", 0) for t in agent_result["trace"])
        n_retrieves_total = sum(len(t.get("probes", [])) for t in agent_result["trace"])
        row = {
            "scenario_id": r["scenario_id"],
            "category": r.get("category", ""),
            "mtmsg_capped": {
                "coverage_rate": agg["coverage_rate"],
                "full_R@5": agg.get("triggered_recall_full@5"),
                "cond_R@5": agg.get("recall_given_covered@5"),
                "full_R@10": agg.get("triggered_recall_full@10"),
                "n_turns": agent_result["n_turns"],
                "max_thread_tokens": agent_result["max_thread_tokens"],
                "max_thread_tokens_after_truncate": agent_result[
                    "max_thread_tokens_after_truncate"
                ],
                "total_truncate_dropped_pairs": agent_result[
                    "total_truncate_dropped_pairs"
                ],
                "n_step_emits": n_step_emits,
                "n_step_finals": len(agent_result["step_outputs"]),
                "n_retrieves_total": n_retrieves_total,
                "n_probes_total": agent_result["n_probes_total"],
                "n_decision_spawns_total": agent_result.get(
                    "n_decision_spawns_total", 0
                ),
                "n_decision_closes_total": agent_result.get(
                    "n_decision_closes_total", 0
                ),
                "n_unique_decisions_spawned": agent_result.get(
                    "n_unique_decisions_spawned", 0
                ),
                "n_spawn_directives_injected": agent_result.get(
                    "n_spawn_directives_injected", 0
                ),
                "n_close_directives_injected": agent_result.get(
                    "n_close_directives_injected", 0
                ),
                "n_step_outs_orphan": agent_result.get("n_step_outs_orphan", 0),
                "orphan_step_labels": agent_result.get("orphan_step_labels", []),
                "currently_open_at_end": agent_result.get(
                    "currently_open_at_end", []
                ),
                "done_emitted": agent_result["done_emitted"],
            },
        }
        summary["scenarios"].append(row)

    cov_vals = []
    full5_vals = []
    full10_vals = []
    cond5_vals = []
    thread_vals = []
    trunc_vals = []
    n_step_emits_total = 0
    n_retrieves_grand = 0
    n_turns_total = 0
    n_decision_spawns_grand = 0
    n_decision_closes_grand = 0
    n_unique_decisions_grand = 0
    n_spawn_directives_grand = 0
    n_close_directives_grand = 0
    n_orphans_grand = 0
    n_step_finals_grand = 0
    for r in results:
        if "error" in r:
            continue
        agent_result = r["per_variant"]["mtmsg_capped"]
        agg = agent_result["aggregates"]
        cov_vals.append(agg["coverage_rate"])
        v = agg.get("triggered_recall_full@5")
        if v is not None:
            full5_vals.append(v)
        v = agg.get("triggered_recall_full@10")
        if v is not None:
            full10_vals.append(v)
        v = agg.get("recall_given_covered@5")
        if v is not None:
            cond5_vals.append(v)
        thread_vals.append(agent_result["max_thread_tokens"])
        trunc_vals.append(agent_result["total_truncate_dropped_pairs"])
        n_turns_total += agent_result["n_turns"]
        n_step_emits_total += sum(
            t.get("n_step_outs", 0) for t in agent_result["trace"]
        )
        n_retrieves_grand += sum(
            len(t.get("probes", [])) for t in agent_result["trace"]
        )
        n_decision_spawns_grand += agent_result.get("n_decision_spawns_total", 0)
        n_decision_closes_grand += agent_result.get("n_decision_closes_total", 0)
        n_unique_decisions_grand += agent_result.get("n_unique_decisions_spawned", 0)
        n_spawn_directives_grand += agent_result.get(
            "n_spawn_directives_injected", 0
        )
        n_close_directives_grand += agent_result.get(
            "n_close_directives_injected", 0
        )
        n_orphans_grand += agent_result.get("n_step_outs_orphan", 0)
        n_step_finals_grand += len(agent_result.get("step_outputs", []))
    summary["per_variant_means"]["mtmsg_capped"] = {
        "coverage_mean": round(sum(cov_vals) / len(cov_vals), 4) if cov_vals else None,
        "full_R@5_mean": round(sum(full5_vals) / len(full5_vals), 4)
        if full5_vals
        else None,
        "full_R@10_mean": round(sum(full10_vals) / len(full10_vals), 4)
        if full10_vals
        else None,
        "cond_R@5_mean": round(sum(cond5_vals) / len(cond5_vals), 4)
        if cond5_vals
        else None,
        "max_thread_tokens_mean": round(sum(thread_vals) / len(thread_vals), 1)
        if thread_vals
        else None,
        "max_thread_tokens_max": max(thread_vals) if thread_vals else None,
        "truncate_pairs_total": sum(trunc_vals) if trunc_vals else 0,
        "n_turns_total": n_turns_total,
        "n_step_emits_total": n_step_emits_total,
        "n_retrieves_total": n_retrieves_grand,
        "n_decision_spawns_total": n_decision_spawns_grand,
        "n_decision_closes_total": n_decision_closes_grand,
        "n_unique_decisions_total": n_unique_decisions_grand,
        "n_spawn_directives_injected_total": n_spawn_directives_grand,
        "n_close_directives_injected_total": n_close_directives_grand,
        "n_step_outs_orphan_total": n_orphans_grand,
        "n_step_finals_total": n_step_finals_grand,
        "orphan_rate": round(n_orphans_grand / n_step_finals_grand, 4)
        if n_step_finals_grand
        else None,
        "step_emits_per_turn": round(n_step_emits_total / n_turns_total, 4)
        if n_turns_total
        else None,
        "retrieves_per_turn": round(n_retrieves_grand / n_turns_total, 4)
        if n_turns_total
        else None,
    }

    summary_name = (
        "SUMMARY.json"
        if _RESULTS_SUBDIR == "results"
        else f"SUMMARY_{_RESULTS_SUBDIR}.json"
    )
    summary_path = THIS_DIR / summary_name
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(
        f"\n=== Cross-scenario means (subdec_twoloop_easy, n={len(results)}, "
        f"out={_RESULTS_SUBDIR}) ==="
    )
    m = summary["per_variant_means"]["mtmsg_capped"]
    print(
        f"  twoloop: cov={m['coverage_mean']} | full_R@5={m['full_R@5_mean']} | "
        f"full_R@10={m['full_R@10_mean']} | cond_R@5={m['cond_R@5_mean']} | "
        f"spawns={m.get('n_decision_spawns_total')} | "
        f"closes={m.get('n_decision_closes_total')} | "
        f"unique={m.get('n_unique_decisions_total')} | "
        f"spawn_dirs={m.get('n_spawn_directives_injected_total')} | "
        f"close_dirs={m.get('n_close_directives_injected_total')} | "
        f"orphan_rate={m.get('orphan_rate')} | "
        f"step_emits/turn={m['step_emits_per_turn']} | "
        f"retrieves/turn={m['retrieves_per_turn']} | "
        f"max_thread_mean={m['max_thread_tokens_mean']} | "
        f"trunc_pairs_total={m['truncate_pairs_total']}"
    )
    print("\n=== Baseline subdec_split_easy reference ===")
    print("  run1 (n=1 excl pres):  cov=0.917 | full_R@5=0.799")
    print("  run2 (n=1):            cov=0.95  | full_R@5=0.7528")
    print("  n=2 mean:              cov=0.934 | full_R@5=0.776")
    print(f"\nWrote {summary_path}")
    print(f"Per-scenario files in {RESULTS_OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
