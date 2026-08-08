"""Shared harness v4_mtmsg_full — validation pass on ALL 10 hard scenarios.

Same architecture as shared_harness_v4_mtmsg (the 2-scenario qualitative run):
bounded WM via continuous mt_messages thread, hard-truncated from oldest at
10k tokens. NO three-tier LRU, NO compression, NO citations, NO structured
JSON output, NO behavioral history blocks. Just a plain multi-turn
conversation thread (system + alternating user/assistant) where the model
emits THINKING:, PROBE:, STEP_OUTPUT:, and DONE lines verbatim. When the
thread crosses 10k tokens, the system hard-drops the oldest user/assistant
pair from the start of the thread (system prompt is preserved).

Validation hypothesis: in 2-scenario qualitative testing, mtmsg-capped was
the standout (world-knowledge R@5 0.370 vs v3 op's 0.044). Does it hold up
on the full 10-scenario hard set?

Reference numbers (10 hard scenarios, indices 10-19):
  v3 baseline_fifo:    cov 0.489 / R@5 0.282
  v3 operator_lru:     cov 0.718 / R@5 0.302
  v1 operator_lru:     cov 0.581 / R@5 0.434
  SA-full champion:    cov 0.876 / R@5 0.646

Usage:
    uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_full/main.py
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
RESULTS_OUT_DIR = THIS_DIR / "results"
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
# "STEP_OUTPUT: <id_or_label>\n<body>". We accept BOTH integer step_ids and
# string labels — string labels get mapped to a stable per-label integer id
# preserving emit-order (label-tracking happens in the agent loop, not here).
# Header forms supported:
#   STEP_OUTPUT: 3: content...
#   STEP_OUTPUT: catering_plan: content...
#   STEP_OUTPUT: 3 - content...     (rare alt separator)
#   STEP_OUTPUT: catering_plan      (no inline content; body follows)
STEP_OUTPUT_HEAD_RE = re.compile(
    r"^\s*STEP_OUTPUT\s*:\s*([^\n:\-]+?)(?:\s*[:\-]\s*(.*))?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
DIRECTIVE_LINE_RE = re.compile(
    r"^\s*(THINKING|PROBE|STEP_OUTPUT|DONE)\b",
    re.MULTILINE | re.IGNORECASE,
)


def parse_step_outputs(raw: str) -> list[dict[str, Any]]:
    """Parse STEP_OUTPUT: <id_or_label>[: <content>] blocks.

    Content extends from after the head line until the next directive line.
    Returns dicts with 'raw_label' (string from the model) and 'content'.
    """
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    heads = list(STEP_OUTPUT_HEAD_RE.finditer(raw))
    for m in heads:
        raw_label = (m.group(1) or "").strip()
        first_line_content = (
            (m.group(2) or "").strip() if m.lastindex and m.lastindex >= 2 else ""
        )
        # Find next directive line strictly after the head's end.
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


# ---------- System prompt (adapted from SA-full mt for bounded WM setting) ----------

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

THINKING: <2-6 sentences. Restate load-bearing facts you've surfaced, name \
what's still open, what you'll chase next. This is your persistent record \
against thread truncation.>
PROBE: <a specific retrieval query>
PROBE: <another query, different angle>
... up to 4 PROBE lines per turn ...
STEP_OUTPUT: <integer_step_id>: <draft deliverable for this sub-decision: \
real names, real numbers, real constraints surfaced from memory. Drafts \
are revisable — emit one as soon as you have something useful, even if \
uncertain. A later STEP_OUTPUT with the same integer step_id supersedes \
earlier ones.>
... emit multiple STEP_OUTPUT lines when ready ...

CRITICAL: the step_id MUST be a small integer (1, 2, 3, ...). Do NOT use \
string labels like `draft_plan` or `catering_plan` — use integers. Each \
distinct sub-decision in the task gets its own integer; revisions of the \
same sub-decision keep the same integer.

Emit `DONE` on its own line when:
(a) you've drafted STEP_OUTPUTs for all sub-decisions AND recent probes \
return no new useful content, OR
(b) you've probed for plausible candidates (optimistic cues) and confirmed \
memory is silent on the remaining sub-decisions.

You may emit any combination of THINKING / PROBE / STEP_OUTPUT lines per \
turn. After your turn, the system parses PROBE lines, runs retrieval, and \
returns the hits as the next user message. STEP_OUTPUT lines are recorded \
as deliverables. Without one of (a) or (b), DONE is premature — drafts \
under uncertainty beat skipping a sub-decision.
"""


USER_INITIAL = """\
TASK:
{task_prompt}

This is turn 1. You have no probes yet. Emit THINKING about what binding \
facts the task likely needs from memory, then PROBE lines to surface them. \
You may also emit STEP_OUTPUT lines for sub-decisions you can already draft \
based on the task text alone (mark uncertainty inline).
"""


USER_FOLLOWUP = """\
Turn {turn}. New probe results from memory:
---
{snippets}
---

(Probes used so far this run: {n_probes_total}.)

Continue: THINKING, then PROBE/STEP_OUTPUT lines. Emit DONE when saturated.
"""


# ---------- Thread truncation ----------


def truncate_thread(messages: list[dict[str, str]], cap: int) -> dict[str, Any]:
    """Drop oldest user/assistant pairs from the front until messages fit cap.

    The system message (index 0) is preserved. We drop in PAIRS: a user msg
    and the assistant response that follows it, so the dialogue stays
    well-formed. If only a trailing user msg remains beyond the system + one
    pair, we drop that user msg too as a singleton (rare edge case).

    Returns a dict with stats: dropped_pairs, dropped_msgs, tokens_before,
    tokens_after.
    """
    stats = {
        "dropped_pairs": 0,
        "dropped_msgs": 0,
        "tokens_before": messages_tokens(messages),
        "tokens_after": messages_tokens(messages),
    }
    while messages_tokens(messages) > cap and len(messages) >= 3:
        # messages[0] = system, messages[1] = oldest user, messages[2] = oldest assistant
        # Drop the (user, assistant) pair starting at index 1.
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
            # Singleton trailing message.
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
    # step_outputs are keyed by integer id; if model uses string labels we
    # assign a stable next-int id per unique label, preserving the original
    # label for visibility/scoring.
    step_outputs_by_id: dict[int, dict[str, Any]] = {}
    label_to_int: dict[str, int] = {}
    next_label_int = 1
    step_outputs_log: list[dict[str, Any]] = []
    n_probes_total = 0
    seen_chat_ids: set[str] = set()
    done_emitted = False

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

        # Parse step_outputs. The parser returns raw_label (string from
        # model). If it's an integer string, use it directly; otherwise map
        # the string label to a stable per-label int.
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
        # Cap to 4 probes per turn.
        probes = probes[:4]
        log.probes = probes

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

            mt_messages.append(
                {
                    "role": "user",
                    "content": USER_FOLLOWUP.format(
                        turn=turn + 1,
                        snippets=snippets_block,
                        n_probes_total=n_probes_total,
                    ),
                }
            )
        else:
            # No probes this turn. If not done, prompt the model with an empty
            # followup so it has another shot. If done, we'll exit below.
            if not done_emitted:
                mt_messages.append(
                    {
                        "role": "user",
                        "content": USER_FOLLOWUP.format(
                            turn=turn + 1,
                            snippets="(no probes emitted last turn)",
                            n_probes_total=n_probes_total,
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

    return {
        "scenario_id": sid,
        "variant": "mtmsg_capped",
        "trace": [td_to_dict(t) for t in trace],
        "step_outputs": final_step_outputs,
        "step_outputs_log": step_outputs_log,
        "n_turns": len(trace),
        "n_probes_total": n_probes_total,
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

    print(f"  [{sid}] starting variant=mtmsg_capped", flush=True)
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
    n_retrieves = sum(len(t.get("probes", [])) for t in agent_result["trace"])
    print(
        f"  [{sid}] mtmsg_capped: cov={cov} | full_R@5={full} | cond_R@5={cond} | "
        f"turns={agent_result['n_turns']} | step_emits={n_step_emits} | "
        f"final_steps={len(agent_result['step_outputs'])} | "
        f"probes_total={agent_result['n_probes_total']} | "
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
    # All 10 hard scenarios — indices 10-19 of mid_execution_scenarios.json
    # (same set used by shared_harness_v3 and prior v4 experiments).
    SCENARIO_INDICES = list(range(10, 20))
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

    sqlite_path = RESULTS_DIR / "eventmemory_shared_harness_v4_mtmsg_full.sqlite3"
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
            print(f"[run] {sid} (variant=mtmsg_capped)", flush=True)
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
        "step_emits_per_turn": round(n_step_emits_total / n_turns_total, 4)
        if n_turns_total
        else None,
        "retrieves_per_turn": round(n_retrieves_grand / n_turns_total, 4)
        if n_turns_total
        else None,
    }

    summary_path = THIS_DIR / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== Cross-scenario means (v4_mtmsg_full, n={len(results)}) ===")
    m = summary["per_variant_means"]["mtmsg_capped"]
    print(
        f"  mtmsg_capped: cov={m['coverage_mean']} | full_R@5={m['full_R@5_mean']} | "
        f"full_R@10={m['full_R@10_mean']} | cond_R@5={m['cond_R@5_mean']} | "
        f"step_emits/turn={m['step_emits_per_turn']} | "
        f"retrieves/turn={m['retrieves_per_turn']} | "
        f"max_thread_mean={m['max_thread_tokens_mean']} | "
        f"trunc_pairs_total={m['truncate_pairs_total']}"
    )
    print("\n=== References ===")
    print("  v3 op (10 hard):     cov=0.718 | full_R@5=0.302")
    print("  v1 op (10 hard):     cov=0.581 | full_R@5=0.434")
    print("  SA-full (10 hard):   cov=0.876 | full_R@5=0.646")
    print(f"\nWrote {summary_path}")
    print(f"Per-scenario files in {RESULTS_OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
