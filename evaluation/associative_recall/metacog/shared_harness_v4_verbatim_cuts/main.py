"""Shared harness v4 — bounded WM agent with VERBATIM CUTS-ONLY compaction.

ONE change vs v3 (shared_harness_v3/main.py):
  - Replaces the LLM-paraphrasing compactor (tier-3 compressed summaries) with
    an LLM that ONLY marks items for removal. Items kept are kept VERBATIM.
    No paraphrasing, no summarization. The compactor outputs `DROP <id>`
    lines; anything not dropped stays verbatim. There is no tier-3 compressed
    store — items are either verbatim in WM or gone (and re-retrievable from
    external memory).

Same priming as v3 (saturation-needs-evidence, drafts-not-commitments,
citation contract). Only the compactor mechanism changes.

Reference: v3 op cov=0.718 / R@5=0.302; v1 op cov=0.581 / R@5=0.434;
SA-full ref cov=0.876 / R@5=0.646.

Usage:
    uv run python evaluation/associative_recall/metacog/shared_harness_v4_verbatim_cuts/main.py
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

# Make the parent dir importable like mid_execution_eval_e2.py does.
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

# WM caps
WM_HARD_CAP = 10_000
TIER2_MAX_ITEMS = 15  # active verbatim cap; trigger compactor when exceeded

MAX_TURNS = 14
RETRIEVE_K = 5
MAX_COMPLETION_TOKENS = 4500
SCORE_K_LIST = [1, 3, 5, 10]

# Reasoning compression: first ~80 tokens verbatim per turn, last 3 turns shown.
REASONING_TOKEN_BUDGET = 80
REASONING_RECENT_N = 3
CUE_EXCERPT_CHARS = 40

try:
    ENC = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def first_n_tokens(text: str, n: int) -> str:
    if not text:
        return ""
    ids = ENC.encode(text)
    if len(ids) <= n:
        return text.strip()
    return ENC.decode(ids[:n]).strip()


# ---------- Items / IDs ----------


@dataclass
class Item:
    item_id: str
    kind: str
    content: str
    last_referenced_turn: int = 0

    def render_verbatim(self) -> str:
        return f"[{self.item_id}] ({self.kind}, last_ref=t{self.last_referenced_turn}):\n{self.content}"

    def size_tokens(self) -> int:
        return n_tokens(self.render_verbatim())


def make_id(kind: str, n: int) -> str:
    return f"{kind}-{n}"


# ---------- LLM ----------

CITATION_RE = re.compile(r"\[([a-zA-Z]+-\d+)\]")


def parse_citations(text: str) -> list[str]:
    if not text:
        return []
    return list(dict.fromkeys(CITATION_RE.findall(text)))


def parse_json_output(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    s = t.find("{")
    e = t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(t[s : e + 1])
    except json.JSONDecodeError:
        return None


async def llm_chat(
    openai_client,
    system: str,
    user: str,
    *,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
) -> str:
    resp = await openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max(MAX_COMPLETION_TOKENS, max_completion_tokens),
    )
    return (resp.choices[0].message.content or "").strip()


# ---------- Priming prompts (v3, with WM-tier description updated for v4) ----------

POST_RETRIEVAL_REFLECTION_BLOCK = """\
--- IMMEDIATE-POST-RETRIEVAL REFLECTION (always) ---

Treat hits as evidence, not just data. Immediately after each retrieve action's hits arrive, write a reflection in your reasoning:
- What did the hits actually contain, vs what you were looking for?
- What's still missing for the sub-decision you're working on?
- What does the gap between cue and hits tell you about how memory is encoded for this kind of fact?

This reflection is your persistent record. The raw hits will age out of your context; your reflection won't. Your next cue should build on the reflection, not paraphrase your last cue."""


SELF_INSPECTION_BLOCK = """\
--- SELF-INSPECTION (always) ---

You can recognize behavioral patterns in your own work. Each turn, the BEHAVIORAL HISTORY and RECENT REASONING blocks show you what you've been doing across turns. Patterns worth looking for:
- Cue paraphrasing: are recent retrievals re-asking the same thing in different words? Same retrieval neighborhood = same hits. Switch shape, not wording.
- Reasoning circles: are you saying the same thing across turns ("still need X")? You may be stuck on one missing piece while ignoring drafts you could already write.
- Action skew: a high retrieve count with zero step_outputs suggests you're researching to avoid writing. Writing under partial information is harder than retrieving — but writing is what produces the deliverable.
- Diminishing returns: low new-fact rate over recent retrievals means more retrieval is unlikely to help; the next draft will benefit more from being WRITTEN than from another probe.

When you spot a pattern, change tactics: switch retrieval shape, draft a step_output even if uncertain, mark uncertainty inline ("[?]" or "approximately"), or commit a partial answer and refine later. The system gives you the data; you do the diagnosis. Don't wait for an explicit instruction to change — your own observation IS the instruction.

When a cue's hits don't match what you needed, the gap is informative. Look at the hits closely. Compare them to what you wanted. Diagnose the gap specifically, in your own words. Then change your cue's shape based on the diagnosis. A different wording in the same shape will return the same kind of hits."""


DRAFTS_BLOCK = """\
--- DRAFTS, NOT COMMITMENTS ---

step_outputs are DRAFTS, not commitments. Emit one as soon as you have enough to write a useful answer, even if uncertain about parts. A later step_output for the same step_id supersedes an earlier one — drafts are revisable, missing drafts are not.

--- SATURATION ---

Saturation requires EVIDENCE. You cannot claim "no more useful info to retrieve" without having actually retrieved. The task description alone does NOT contain the binding facts — they live in memory. If you haven't retrieved, you don't know what's there. Probe first, claim saturation second.

Two valid grounds for emitting `done`:
(a) You have drafted step_outputs for all sub-decisions in the task, AND recent retrievals consistently surface no new useful content.
(b) You have probed for plausible candidate answers (optimistic cues — concrete specific possibilities, not abstract questions) and confirmed memory is silent on the remaining sub-decisions, having actively tried to surface them.

Without one of these, `done` is premature. Drafts under uncertainty are still better than skipping the sub-decision.

Optimistic cue example (principle-level — apply the shape to your task):
- Don't only probe "what is X's preference?" If that fails, also probe specific plausible answers ("X prefers 30 minutes", "X prefers Thursday", etc.) — if memory contains the fact in any framing, one of these will surface. Only when concrete-candidate probing also fails can you treat memory as silent."""


OUTPUT_SCHEMA_BLOCK = """\
--- OUTPUT SCHEMA (strict JSON, every turn — output ONLY the JSON object) ---

{
  "reasoning": "<free prose with inline citations like [chat-3]. Include immediate-post-retrieval reflections. Note self-inspected patterns when you see them.>",
  "actions": [
    {"type": "retrieve", "cue": "<short content-bearing query>", "reason": "<why>"},
    {"type": "assess",   "re": "<item-id>", "did_advance": true_or_false, "why": "<short>"},
    {"type": "step_output", "step_id": <int>, "label": "<short>", "content": "<draft deliverable: real names, real numbers, citing items used; mark uncertain spans>"},
    {"type": "done"}
  ]
}

You may emit multiple actions per turn."""


PRIMING_OPERATOR = (
    """\
You are a memory-augmented agent solving a multi-step task with bounded working memory. You have a retrieval tool over a past-conversation memory that contains specific facts (constraints, preferences, allergies, dates, numbers, identities) the user has shared previously. THESE FACTS MATERIALLY CHANGE THE TASK OUTPUT — without retrieving them first you will write generic placeholder answers that miss real binding constraints.

The task description ALONE does not contain those facts. They live in the memory tool. Your job: probe memory to surface those facts AND draft step_outputs that USE the surfaced facts (real names, real numbers, real constraints from memory).

--- WORKING MEMORY ---

Your working memory has two tiers, shown to you each turn:
- TIER 1 (transient retrieval buffer): the most recent retrieval's full hits, each shown verbatim with its chat-N id. Replaced each retrieval round.
- TIER 2 (active LRU, verbatim): items you've recently CITED. The system tracks what you cite to keep relevant items active.

When WM exceeds budget, a separate compactor LLM marks items for REMOVAL (verbatim cuts only — nothing is summarized or paraphrased). Items not marked for removal stay VERBATIM. Dropped items are gone from WM but remain re-retrievable from memory.

THE CITATION CONTRACT — central mechanism:
- Cite an item like [chat-12] or [ret-3] when you USE that item's content. Citation = "I am using this right now".
- When you cite, the system refreshes the item's LRU position. Items you stop citing are eligible for removal at compaction time.
- INLINE-QUOTE load-bearing content next to the citation marker. The [id] is for traceability; the inline quoted content is for clarity AND for robustness against compaction (so your reasoning survives even if the cited item is later DROPPED). Example: "Maya has a strict shellfish allergy [chat-43]" — the words before [chat-43] make the reasoning survive if [chat-43] is later removed from WM.
- Cite REAL ids visible in the WM. Do not invent.

"""
    + POST_RETRIEVAL_REFLECTION_BLOCK
    + "\n\n"
    + SELF_INSPECTION_BLOCK
    + "\n\n"
    + DRAFTS_BLOCK
    + "\n\n"
    + OUTPUT_SCHEMA_BLOCK
)


PRIMING_BASELINE = (
    """\
You are a memory-augmented agent solving a multi-step task with bounded working memory. You have a retrieval tool over a past-conversation memory that contains specific facts (constraints, preferences, allergies, dates, numbers, identities) the user has shared previously. THESE FACTS MATERIALLY CHANGE THE TASK OUTPUT — without retrieving them first you will write generic placeholder answers that miss real binding constraints.

The task description ALONE does not contain those facts. They live in the memory tool. Your job: probe memory to surface those facts AND draft step_outputs that USE the surfaced facts (real names, real numbers, real constraints from memory).

Your working memory shows newest items at the top. When WM exceeds the token budget, oldest items get FIFO-evicted. Cite items like [chat-3] when you rely on them so your traceability survives.

"""
    + POST_RETRIEVAL_REFLECTION_BLOCK
    + "\n\n"
    + SELF_INSPECTION_BLOCK
    + "\n\n"
    + DRAFTS_BLOCK
    + "\n\n"
    + OUTPUT_SCHEMA_BLOCK
)


# ---------- Compactor (verbatim cuts only) ----------

COMPACTOR_SYSTEM = """\
You are a working-memory compactor. The agent's working memory is over budget. Your ONLY job: identify items in the current WM that are no longer load-bearing for the remaining work — facts already used and won't be needed again, retrievals whose hits proved irrelevant, drafts already superseded, reasoning that has already produced its insight.

Output ONLY a list of `DROP <item-id>` lines, one per line. Do NOT paraphrase. Do NOT summarize. Do NOT rewrite. Anything not dropped stays VERBATIM in the agent's working memory.

Rules:
- Only emit `DROP <id>` lines. No prose. No explanations. No headers. No code fences.
- Use exact item ids as shown (e.g., `DROP chat-12`, `DROP reason-3`).
- Be conservative when an item might still be needed; the agent can re-retrieve from external memory if you drop something useful, but losing a load-bearing fact mid-task can stall the agent.
- Strongly prefer dropping: stale retrieve-action records (`ret-N`) whose hits proved irrelevant, old reasoning entries (`reason-N`) whose insights are already captured in later reasoning, superseded step_output items where a later step_output for the same step_id exists, assess records that are no longer informative.
- Strongly prefer KEEPING: chat items containing concrete facts (names, numbers, dates, allergies, preferences) that match the task; the user's original task statement (chat-0); the latest reasoning that the agent is actively building on; latest step_outputs for each step_id.
- Drop enough to bring WM meaningfully under budget. If unsure, drop more rather than fewer — the agent can re-retrieve."""


COMPACTOR_USER_TEMPLATE = """\
TASK (for context, so you know what is load-bearing):
{task_prompt}

CURRENT WORKING MEMORY (verbatim, with item ids):
{wm_text}

CURRENT WM TOKENS: {wm_tokens}
BUDGET: {budget}

Output ONLY `DROP <id>` lines, one per line. No other output."""


DROP_LINE_RE = re.compile(r"^\s*DROP\s+([a-zA-Z]+-\d+)\s*$", re.IGNORECASE)


def parse_drop_ids(text: str) -> list[str]:
    if not text:
        return []
    ids: list[str] = []
    for line in text.splitlines():
        m = DROP_LINE_RE.match(line)
        if m:
            ids.append(m.group(1))
    return list(dict.fromkeys(ids))


# ---------- WM rendering ----------


def render_wm_operator(state: OperatorState) -> str:
    parts = []
    parts.append("--- TIER 1 (transient retrieval buffer; latest hits) ---")
    if state.tier1:
        for it in state.tier1:
            parts.append(it.render_verbatim())
    else:
        parts.append("(empty)")
    parts.append("\n--- TIER 2 (active LRU, verbatim) ---")
    if state.tier2:
        for it in state.tier2:
            parts.append(it.render_verbatim())
    else:
        parts.append("(empty)")
    return "\n".join(parts)


def render_wm_for_compactor(state: OperatorState) -> str:
    """Renders the current WM contents (tier1 + tier2) verbatim with ids,
    for the compactor LLM to inspect."""
    parts = []
    parts.append("--- TIER 1 (transient retrieval buffer) ---")
    if state.tier1:
        for it in state.tier1:
            parts.append(it.render_verbatim())
    else:
        parts.append("(empty)")
    parts.append("\n--- TIER 2 (active LRU) ---")
    if state.tier2:
        for it in state.tier2:
            parts.append(it.render_verbatim())
    else:
        parts.append("(empty)")
    return "\n".join(parts)


def render_wm_baseline(state: BaselineState) -> str:
    parts = ["--- WORKING MEMORY (newest first; older items evicted at cap) ---"]
    if not state.wm:
        parts.append("(empty)")
    else:
        for it in state.wm:
            parts.append(it.render_verbatim())
    return "\n".join(parts)


# ---------- Behavioral history rendering ----------


def render_behavioral_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "--- BEHAVIORAL HISTORY ---\n(no prior turns)"
    lines = [
        "--- BEHAVIORAL HISTORY ---",
        f"{'turn':>4}  {'retrieves':>9}  {'assesses':>8}  {'step_outs':>9}  cue_excerpts",
    ]
    for h in history:
        cue_str = " / ".join(f'"{c}"' for c in h["cue_excerpts"]) or "(none)"
        lines.append(
            f"{h['turn']:>4}  {h['retrieves']:>9}  {h['assesses']:>8}  "
            f"{h['step_outs']:>9}  {cue_str}"
        )
    return "\n".join(lines)


def render_recent_reasoning(
    history: list[dict[str, Any]], n: int = REASONING_RECENT_N
) -> str:
    if not history:
        return f"--- RECENT REASONING (last {n}, ~{REASONING_TOKEN_BUDGET} tokens each) ---\n(no prior turns)"
    recent = history[-n:]
    lines = [
        f"--- RECENT REASONING (last {n}, ~{REASONING_TOKEN_BUDGET} tokens each) ---"
    ]
    for h in recent:
        compressed = h.get("reasoning_compressed") or "(empty)"
        lines.append(f"  t{h['turn']}: {compressed}")
    return "\n".join(lines)


# ---------- Prompts ----------


def build_user_prompt(
    *,
    task_prompt: str,
    turn: int,
    wm_text: str,
    retrieval_log: list[dict[str, Any]],
    history: list[dict[str, Any]],
    step_outputs_visible: list[dict[str, Any]],
) -> str:
    parts = []
    parts.append(f"TASK:\n{task_prompt}\n")
    parts.append(f"TURN: {turn}")
    parts.append("")
    parts.append(render_behavioral_history(history))
    parts.append("")
    parts.append(render_recent_reasoning(history))
    parts.append("")
    parts.append(wm_text)
    parts.append("")
    parts.append("--- RETRIEVAL LOG (last 6) ---")
    if retrieval_log:
        for entry in retrieval_log[-6:]:
            ids_part = ", ".join(entry.get("hit_ids", []))
            parts.append(
                f"  t{entry['turn']}: cue={entry['cue']!r} -> "
                f"{entry['n_hits']} hits ({ids_part})"
            )
    else:
        parts.append("(empty)")
    parts.append("")
    parts.append(f"--- STEP OUTPUTS SO FAR ({len(step_outputs_visible)}) ---")
    if step_outputs_visible:
        for so in step_outputs_visible:
            parts.append(
                f"  step_id={so['step_id']} label={so['label']!r} (turn={so['turn']})"
            )
    else:
        parts.append("(none yet — drafts welcome)")
    parts.append("")
    parts.append("Respond with the JSON object now.")
    return "\n".join(parts)


# ---------- Operator state (no tier3) ----------


@dataclass
class OperatorState:
    tier1: list[Item] = field(default_factory=list)
    tier2: list[Item] = field(default_factory=list)
    all_items: dict[str, Item] = field(default_factory=dict)
    retrieval_log: list[dict[str, Any]] = field(default_factory=list)
    # Track all DROP decisions made by the compactor across the run.
    drop_log: list[dict[str, Any]] = field(default_factory=list)

    def register(self, item: Item) -> None:
        self.all_items[item.item_id] = item

    def add_to_tier2(self, item: Item) -> None:
        self.tier2.insert(0, item)
        self.register(item)

    def replace_tier1(self, items: list[Item]) -> None:
        # Items removed from tier1 are dropped entirely (they are short-lived
        # retrieval buffer items; if they were cited they'd already have been
        # promoted into tier2).
        for old in self.tier1:
            if old not in self.tier2:
                self.all_items.pop(old.item_id, None)
        self.tier1 = list(items)
        for it in items:
            self.register(it)

    def cite(self, item_id: str, turn: int) -> str:
        if item_id not in self.all_items:
            return "unknown"
        it = self.all_items[item_id]
        it.last_referenced_turn = turn
        if it in self.tier1:
            self.tier1.remove(it)
            self.tier2.insert(0, it)
            return "tier1->tier2"
        if it in self.tier2:
            self.tier2.remove(it)
            self.tier2.insert(0, it)
            return "tier2->tier2(refresh)"
        return "untracked"

    def transitive_propagate(
        self, cited_ids: list[str], turn: int, depth: int = 2
    ) -> list[str]:
        touched: list[str] = []
        seen = set(cited_ids)
        frontier = list(cited_ids)
        for _ in range(depth):
            next_frontier: list[str] = []
            for cid in frontier:
                it = self.all_items.get(cid)
                if not it:
                    continue
                inner = parse_citations(it.content)
                for cc in inner:
                    if cc in seen:
                        continue
                    seen.add(cc)
                    res = self.cite(cc, turn)
                    if res != "unknown":
                        touched.append(cc)
                        next_frontier.append(cc)
            frontier = next_frontier
            if not frontier:
                break
        return touched

    def verbatim_tokens(self) -> int:
        s = "\n".join(i.render_verbatim() for i in self.tier1 + self.tier2)
        return n_tokens(s)

    def drop_items(self, ids_to_drop: list[str]) -> list[str]:
        """Remove items from tier2 (and tier1, defensively) by id. chat-0 is
        protected (always keep the original task statement)."""
        actually_dropped: list[str] = []
        for did in ids_to_drop:
            if did == "chat-0":
                continue
            it = self.all_items.get(did)
            if not it:
                continue
            removed = False
            if it in self.tier2:
                self.tier2.remove(it)
                removed = True
            if it in self.tier1:
                self.tier1.remove(it)
                removed = True
            if removed:
                self.all_items.pop(did, None)
                actually_dropped.append(did)
        return actually_dropped

    async def compact_via_llm(
        self,
        *,
        task_prompt: str,
        turn: int,
        openai_client,
    ) -> dict[str, Any]:
        """Run the verbatim-cuts compactor: ask LLM to mark items for removal.
        No paraphrasing — items are kept verbatim or dropped entirely."""
        wm_text = render_wm_for_compactor(self)
        wm_tokens = self.verbatim_tokens()
        user = COMPACTOR_USER_TEMPLATE.format(
            task_prompt=task_prompt,
            wm_text=wm_text,
            wm_tokens=wm_tokens,
            budget=WM_HARD_CAP,
        )
        try:
            raw = await llm_chat(
                openai_client,
                COMPACTOR_SYSTEM,
                user,
                max_completion_tokens=2000,
            )
        except Exception as exc:
            return {
                "turn": turn,
                "trigger_tokens": wm_tokens,
                "trigger_tier2_count": len(self.tier2),
                "raw": f"ERROR: {exc!r}",
                "proposed_drops": [],
                "actually_dropped": [],
                "tokens_after": self.verbatim_tokens(),
            }
        proposed = parse_drop_ids(raw)
        actually_dropped = self.drop_items(proposed)
        return {
            "turn": turn,
            "trigger_tokens": wm_tokens,
            "trigger_tier2_count": len(self.tier2),
            "raw": raw[:1500],
            "proposed_drops": proposed,
            "actually_dropped": actually_dropped,
            "tokens_after": self.verbatim_tokens(),
        }

    def fallback_lru_drop(
        self, *, target_count: int | None, target_tokens: int | None
    ) -> list[str]:
        """Drop oldest tier2 items by last_referenced_turn until under target.
        chat-0 protected. Used when LLM compactor fails or under-prunes."""
        dropped: list[str] = []
        # Sort by last_ref ascending = oldest first
        eligible = [it for it in self.tier2 if it.item_id != "chat-0"]
        eligible.sort(key=lambda x: x.last_referenced_turn)
        for victim in eligible:
            need_more = False
            if target_count is not None and len(self.tier2) > target_count:
                need_more = True
            if target_tokens is not None and self.verbatim_tokens() > target_tokens:
                need_more = True
            if not need_more:
                break
            self.tier2.remove(victim)
            self.all_items.pop(victim.item_id, None)
            dropped.append(victim.item_id)
        return dropped

    async def enforce_caps(
        self,
        *,
        task_prompt: str,
        turn: int,
        openai_client,
    ) -> dict[str, Any]:
        """If over caps, run the LLM compactor (verbatim cuts only). If still
        over caps after compaction, fall back to LRU drop to make room."""
        cap_log: dict[str, Any] = {
            "compactor_invoked": False,
            "compactor_record": None,
            "fallback_dropped": [],
        }
        over_count = len(self.tier2) > TIER2_MAX_ITEMS
        over_tokens = self.verbatim_tokens() > WM_HARD_CAP
        if not (over_count or over_tokens):
            return cap_log
        cap_log["compactor_invoked"] = True
        rec = await self.compact_via_llm(
            task_prompt=task_prompt,
            turn=turn,
            openai_client=openai_client,
        )
        self.drop_log.append(rec)
        cap_log["compactor_record"] = rec
        # Fallback if still over caps
        if len(self.tier2) > TIER2_MAX_ITEMS or self.verbatim_tokens() > WM_HARD_CAP:
            fb = self.fallback_lru_drop(
                target_count=TIER2_MAX_ITEMS,
                target_tokens=WM_HARD_CAP,
            )
            cap_log["fallback_dropped"] = fb
        return cap_log


# ---------- Baseline FIFO state (unchanged) ----------


@dataclass
class BaselineState:
    wm: list[Item] = field(default_factory=list)
    all_items: dict[str, Item] = field(default_factory=dict)
    retrieval_log: list[dict[str, Any]] = field(default_factory=list)

    def push(self, item: Item) -> None:
        self.wm.insert(0, item)
        self.all_items[item.item_id] = item

    def tokens(self) -> int:
        s = "\n".join(i.render_verbatim() for i in self.wm)
        return n_tokens(s)

    def enforce_cap(self) -> dict[str, Any]:
        evicted: list[str] = []
        max_iters = 200
        while self.tokens() > WM_HARD_CAP and self.wm and max_iters > 0:
            victim = self.wm.pop()
            self.all_items.pop(victim.item_id, None)
            evicted.append(victim.item_id)
            max_iters -= 1
        return {"evicted": evicted}


# ---------- Hit-rendering helpers ----------


def hit_to_chat_item_text(hit, fallback_idx: int) -> tuple[str, str]:
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
    wm_tokens_before: int
    wm_tokens_after: int
    parse_ok: bool
    raw_excerpt: str
    reasoning_excerpt: str
    reasoning_full: str
    citations: list[str]
    transitive_touched: list[str]
    actions_summary: list[dict[str, Any]]
    tier1_ids: list[str] = field(default_factory=list)
    tier2_ids: list[tuple[str, int]] = field(default_factory=list)
    promoted: list[tuple[str, str]] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    compactor_invoked: bool = False
    compactor_proposed: list[str] = field(default_factory=list)
    compactor_dropped: list[str] = field(default_factory=list)
    fallback_dropped: list[str] = field(default_factory=list)
    new_facts: int = 0
    n_retrieves: int = 0
    n_assesses: int = 0
    n_step_outs: int = 0
    cue_excerpts: list[str] = field(default_factory=list)


async def run_agent_loop(
    *,
    scenario: dict,
    memory,
    openai_client,
    variant: str,
) -> dict[str, Any]:
    task_prompt = scenario["task_prompt"]
    sid = scenario["scenario_id"]

    if variant == "operator_lru":
        state: Any = OperatorState()
        priming = PRIMING_OPERATOR
    elif variant == "baseline_fifo":
        state = BaselineState()
        priming = PRIMING_BASELINE
    else:
        raise ValueError(variant)

    counters = {"chat": 1000, "ret": 0, "reason": 0, "step": 0, "assess": 0}

    trace: list[TurnLog] = []
    step_outputs_by_id: dict[int, dict[str, Any]] = {}
    step_outputs_log: list[dict[str, Any]] = []
    new_facts_per_turn: list[int] = []
    parse_failures = 0
    citations_per_turn: list[int] = []
    seen_chat_ids: set[str] = set()

    behavioral_history: list[dict[str, Any]] = []

    chat0 = Item(
        item_id="chat-0",
        kind="chat",
        content=f"USER TASK: {task_prompt}",
        last_referenced_turn=0,
    )
    if isinstance(state, OperatorState):
        state.add_to_tier2(chat0)
    else:
        state.push(chat0)
    seen_chat_ids.add("chat-0")

    done_emitted = False

    for turn in range(1, MAX_TURNS + 1):
        if isinstance(state, OperatorState):
            wm_text = render_wm_operator(state)
        else:
            wm_text = render_wm_baseline(state)

        step_outputs_visible = sorted(
            step_outputs_by_id.values(), key=lambda s: s["step_id"]
        )

        user_prompt = build_user_prompt(
            task_prompt=task_prompt,
            turn=turn,
            wm_text=wm_text,
            retrieval_log=state.retrieval_log,
            history=behavioral_history,
            step_outputs_visible=step_outputs_visible,
        )

        try:
            raw = await llm_chat(openai_client, priming, user_prompt)
        except Exception as exc:
            raw = ""
            log = TurnLog(
                turn=turn,
                wm_tokens_before=state.verbatim_tokens()
                if isinstance(state, OperatorState)
                else state.tokens(),
                wm_tokens_after=state.verbatim_tokens()
                if isinstance(state, OperatorState)
                else state.tokens(),
                parse_ok=False,
                raw_excerpt=f"LLM ERROR: {exc!r}"[:300],
                reasoning_excerpt="",
                reasoning_full="",
                citations=[],
                transitive_touched=[],
                actions_summary=[],
            )
            trace.append(log)
            new_facts_per_turn.append(0)
            citations_per_turn.append(0)
            parse_failures += 1
            behavioral_history.append(
                {
                    "turn": turn,
                    "retrieves": 0,
                    "assesses": 0,
                    "step_outs": 0,
                    "cue_excerpts": [],
                    "reasoning_compressed": "(LLM error)",
                }
            )
            continue

        parsed = parse_json_output(raw)
        wm_before = (
            state.verbatim_tokens()
            if isinstance(state, OperatorState)
            else state.tokens()
        )
        log = TurnLog(
            turn=turn,
            wm_tokens_before=wm_before,
            wm_tokens_after=wm_before,
            parse_ok=parsed is not None,
            raw_excerpt=raw[:400],
            reasoning_excerpt="",
            reasoning_full="",
            citations=[],
            transitive_touched=[],
            actions_summary=[],
        )
        if isinstance(state, OperatorState):
            log.tier1_ids = [i.item_id for i in state.tier1]
            log.tier2_ids = [(i.item_id, i.last_referenced_turn) for i in state.tier2]

        if parsed is None:
            parse_failures += 1
            new_facts_per_turn.append(0)
            citations_per_turn.append(0)
            counters["reason"] += 1
            r_id = make_id("reason", counters["reason"])
            r_item = Item(
                item_id=r_id,
                kind="reason",
                content=raw[:400] or "(no output)",
                last_referenced_turn=turn,
            )
            if isinstance(state, OperatorState):
                state.add_to_tier2(r_item)
                cap_log = await state.enforce_caps(
                    task_prompt=task_prompt,
                    turn=turn,
                    openai_client=openai_client,
                )
                log.compactor_invoked = cap_log["compactor_invoked"]
                if cap_log["compactor_record"]:
                    log.compactor_proposed = cap_log["compactor_record"][
                        "proposed_drops"
                    ]
                    log.compactor_dropped = cap_log["compactor_record"][
                        "actually_dropped"
                    ]
                log.fallback_dropped = cap_log["fallback_dropped"]
                log.dropped = log.compactor_dropped + log.fallback_dropped
                log.wm_tokens_after = state.verbatim_tokens()
            else:
                state.push(r_item)
                ev = state.enforce_cap()
                log.dropped = ev["evicted"]
                log.wm_tokens_after = state.tokens()
            trace.append(log)
            behavioral_history.append(
                {
                    "turn": turn,
                    "retrieves": 0,
                    "assesses": 0,
                    "step_outs": 0,
                    "cue_excerpts": [],
                    "reasoning_compressed": "(parse failure)",
                }
            )
            continue

        reasoning_text = (parsed.get("reasoning") or "").strip()
        actions = parsed.get("actions") or []
        if not isinstance(actions, list):
            actions = []
        log.reasoning_excerpt = reasoning_text[:300]
        log.reasoning_full = reasoning_text

        counters["reason"] += 1
        r_id = make_id("reason", counters["reason"])
        r_item = Item(
            item_id=r_id,
            kind="reason",
            content=reasoning_text or "(empty reasoning)",
            last_referenced_turn=turn,
        )
        if isinstance(state, OperatorState):
            state.add_to_tier2(r_item)
        else:
            state.push(r_item)

        all_text_for_cite = reasoning_text or ""
        for a in actions:
            if not isinstance(a, dict):
                continue
            for fld in ("re", "why", "content", "label", "cue", "reason"):
                v = a.get(fld)
                if isinstance(v, str):
                    all_text_for_cite += f" [{v}]" if fld == "re" else f" {v}"
        cited = parse_citations(all_text_for_cite)
        log.citations = cited
        citations_per_turn.append(len(cited))

        if isinstance(state, OperatorState):
            for cid in cited:
                where = state.cite(cid, turn)
                if where != "unknown":
                    log.promoted.append((cid, where))
            touched = state.transitive_propagate(cited, turn, depth=2)
            log.transitive_touched = touched

        new_facts = 0
        n_retrieves = 0
        n_assesses = 0
        n_step_outs = 0
        cue_excerpts: list[str] = []

        for a in actions:
            if not isinstance(a, dict):
                continue
            atype = a.get("type", "")
            if atype == "retrieve":
                cue = (a.get("cue") or "").strip()
                if not cue:
                    log.actions_summary.append(
                        {"type": "retrieve", "skipped": "empty_cue"}
                    )
                    continue
                n_retrieves += 1
                cue_excerpts.append(cue[:CUE_EXCERPT_CHARS])
                try:
                    hits = await probe(memory, cue, RETRIEVE_K)
                except Exception as exc:
                    log.actions_summary.append(
                        {"type": "retrieve", "error": repr(exc)[:120]}
                    )
                    continue
                hit_items: list[Item] = []
                hit_ids: list[str] = []
                for idx, h in enumerate(hits):
                    chat_id, content = hit_to_chat_item_text(h, idx)
                    if not content:
                        continue
                    hit_ids.append(chat_id)
                    if h.plant_id and chat_id not in seen_chat_ids:
                        new_facts += 1
                    seen_chat_ids.add(chat_id)
                    if isinstance(state, OperatorState) and chat_id in state.all_items:
                        existing = state.all_items[chat_id]
                        hit_items.append(existing)
                        continue
                    if (
                        not isinstance(state, OperatorState)
                        and chat_id in state.all_items
                    ):
                        existing = state.all_items[chat_id]
                        hit_items.append(existing)
                        continue
                    item = Item(
                        item_id=chat_id,
                        kind="chat",
                        content=content,
                        last_referenced_turn=turn,
                    )
                    hit_items.append(item)
                counters["ret"] += 1
                ret_id = make_id("ret", counters["ret"])
                ret_text = (
                    f"cue={cue!r}; reason={a.get('reason', '')[:120]!r}; "
                    f"surfaced=[{', '.join(hit_ids)}]"
                )
                ret_item = Item(
                    item_id=ret_id,
                    kind="ret",
                    content=ret_text,
                    last_referenced_turn=turn,
                )
                if isinstance(state, OperatorState):
                    state.replace_tier1(hit_items)
                    state.add_to_tier2(ret_item)
                else:
                    for it in hit_items:
                        state.push(it)
                    state.push(ret_item)
                state.retrieval_log.append(
                    {
                        "turn": turn,
                        "cue": cue[:200],
                        "n_hits": len(hits),
                        "hit_ids": hit_ids,
                    }
                )
                log.actions_summary.append(
                    {
                        "type": "retrieve",
                        "cue": cue[:120],
                        "n_hits": len(hits),
                        "hit_ids": hit_ids,
                    }
                )
            elif atype == "assess":
                ref = a.get("re") or ""
                why = a.get("why") or ""
                did_advance = a.get("did_advance")
                n_assesses += 1
                counters["assess"] += 1
                aid = make_id("assess", counters["assess"])
                content = f"re={ref} did_advance={did_advance} why={why}"
                a_item = Item(
                    item_id=aid,
                    kind="assess",
                    content=content,
                    last_referenced_turn=turn,
                )
                if isinstance(state, OperatorState):
                    state.add_to_tier2(a_item)
                else:
                    state.push(a_item)
                log.actions_summary.append(
                    {"type": "assess", "re": ref[:60], "did_advance": did_advance}
                )
            elif atype == "step_output":
                step_id_raw = a.get("step_id")
                if isinstance(step_id_raw, int):
                    step_id = step_id_raw
                elif isinstance(step_id_raw, str) and step_id_raw.isdigit():
                    step_id = int(step_id_raw)
                else:
                    step_id = (max(step_outputs_by_id) + 1) if step_outputs_by_id else 1
                label = a.get("label") or ""
                content = a.get("content") or ""
                n_step_outs += 1
                counters["step"] += 1
                sid_item = make_id("step", counters["step"])
                s_item = Item(
                    item_id=sid_item,
                    kind="step",
                    content=f"[STEP {step_id}] {label}\n{content}",
                    last_referenced_turn=turn,
                )
                if isinstance(state, OperatorState):
                    state.add_to_tier2(s_item)
                else:
                    state.push(s_item)
                step_outputs_by_id[step_id] = {
                    "step_id": step_id,
                    "label": str(label)[:200],
                    "content": str(content),
                    "item_id": sid_item,
                    "turn": turn,
                }
                step_outputs_log.append(
                    {
                        "step_id": step_id,
                        "label": str(label)[:200],
                        "content": str(content),
                        "item_id": sid_item,
                        "turn": turn,
                        "is_revision": step_id in step_outputs_by_id
                        and any(e["step_id"] == step_id for e in step_outputs_log),
                    }
                )
                log.actions_summary.append(
                    {
                        "type": "step_output",
                        "step_id": step_id,
                        "label": str(label)[:60],
                    }
                )
            elif atype == "done":
                done_emitted = True
                log.actions_summary.append({"type": "done"})
            else:
                log.actions_summary.append({"type": "unknown", "raw_type": atype})

        new_facts_per_turn.append(new_facts)
        log.new_facts = new_facts
        log.n_retrieves = n_retrieves
        log.n_assesses = n_assesses
        log.n_step_outs = n_step_outs
        log.cue_excerpts = cue_excerpts

        compressed = first_n_tokens(reasoning_text, REASONING_TOKEN_BUDGET)
        behavioral_history.append(
            {
                "turn": turn,
                "retrieves": n_retrieves,
                "assesses": n_assesses,
                "step_outs": n_step_outs,
                "cue_excerpts": cue_excerpts,
                "reasoning_compressed": compressed,
            }
        )

        if isinstance(state, OperatorState):
            cap_log = await state.enforce_caps(
                task_prompt=task_prompt,
                turn=turn,
                openai_client=openai_client,
            )
            log.compactor_invoked = cap_log["compactor_invoked"]
            if cap_log["compactor_record"]:
                log.compactor_proposed = cap_log["compactor_record"]["proposed_drops"]
                log.compactor_dropped = cap_log["compactor_record"]["actually_dropped"]
            log.fallback_dropped = cap_log["fallback_dropped"]
            log.dropped = log.compactor_dropped + log.fallback_dropped
            log.wm_tokens_after = state.verbatim_tokens()
        else:
            ev = state.enforce_cap()
            log.dropped = ev["evicted"]
            log.wm_tokens_after = state.tokens()

        trace.append(log)
        if done_emitted:
            break

    final_step_outputs = sorted(step_outputs_by_id.values(), key=lambda s: s["step_id"])

    drop_log = state.drop_log if isinstance(state, OperatorState) else []

    return {
        "scenario_id": sid,
        "variant": variant,
        "trace": [td_to_dict(t) for t in trace],
        "step_outputs": final_step_outputs,
        "step_outputs_log": step_outputs_log,
        "behavioral_history": behavioral_history,
        "parse_failures": parse_failures,
        "n_turns": len(trace),
        "max_wm_tokens": max((t.wm_tokens_after for t in trace), default=0),
        "citations_per_turn": citations_per_turn,
        "drop_log": drop_log,
    }


def td_to_dict(t: TurnLog) -> dict[str, Any]:
    return {
        "turn": t.turn,
        "wm_tokens_before": t.wm_tokens_before,
        "wm_tokens_after": t.wm_tokens_after,
        "parse_ok": t.parse_ok,
        "raw_excerpt": t.raw_excerpt,
        "reasoning_excerpt": t.reasoning_excerpt,
        "reasoning_full": t.reasoning_full,
        "citations": t.citations,
        "transitive_touched": t.transitive_touched,
        "actions_summary": t.actions_summary,
        "tier1_ids": t.tier1_ids,
        "tier2_ids": t.tier2_ids,
        "promoted": t.promoted,
        "dropped": t.dropped,
        "compactor_invoked": t.compactor_invoked,
        "compactor_proposed": t.compactor_proposed,
        "compactor_dropped": t.compactor_dropped,
        "fallback_dropped": t.fallback_dropped,
        "new_facts": t.new_facts,
        "n_retrieves": t.n_retrieves,
        "n_assesses": t.n_assesses,
        "n_step_outs": t.n_step_outs,
        "cue_excerpts": t.cue_excerpts,
    }


# ---------- Coverage judge ----------

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


# ---------- Score one variant on one scenario ----------


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
    steps_by_id: dict[int, dict[str, Any]] = {}
    for so in step_outputs:
        steps_by_id[int(so["step_id"])] = so

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
    variants: list[str],
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

    out: dict[str, Any] = {
        "scenario_id": sid,
        "category": scenario.get("category", ""),
        "ingest_time_s": round(ingest_time, 2),
        "ingest_info": ingest_info,
        "K_list": K_list,
        "per_variant": {},
    }

    async def _run_one(variant: str) -> tuple[str, dict[str, Any]]:
        print(f"  [{sid}] starting variant={variant}", flush=True)
        agent_result = await run_agent_loop(
            scenario=scenario,
            memory=memory,
            openai_client=openai_client,
            variant=variant,
        )
        score = await score_variant(
            scenario=scenario,
            agent_result=agent_result,
            memory=memory,
            openai_client=openai_client,
            K_list=K_list,
        )
        per = {**agent_result, **score}
        fp = RESULTS_OUT_DIR / f"{sid}_{variant}.json"
        fp.write_text(json.dumps(per, indent=2, default=str))
        agg = score["aggregates"]
        cov = agg["coverage_rate"]
        full = agg.get("triggered_recall_full@5", "n/a")
        cond = agg.get("recall_given_covered@5", "n/a")
        n_step_emits = sum(t.get("n_step_outs", 0) for t in agent_result["trace"])
        n_compactor = sum(
            1 for t in agent_result["trace"] if t.get("compactor_invoked")
        )
        n_dropped = sum(
            len(t.get("compactor_dropped", [])) for t in agent_result["trace"]
        )
        print(
            f"  [{sid}] {variant}: cov={cov} | full_R@5={full} | cond_R@5={cond} | "
            f"turns={agent_result['n_turns']} | step_emits={n_step_emits} | "
            f"final_steps={len(agent_result['step_outputs'])} | "
            f"parse_fail={agent_result['parse_failures']} | "
            f"max_wm={agent_result['max_wm_tokens']} | "
            f"compactor_calls={n_compactor} | items_dropped={n_dropped}",
            flush=True,
        )
        return variant, per

    pairs = await asyncio.gather(*[_run_one(v) for v in variants])
    for variant, per in pairs:
        out["per_variant"][variant] = per
    return out


async def main() -> None:
    # v4 quick test: only two scenarios.
    SCENARIO_IDS = ["world-knowledge-bridge-01", "multi-hop-banquet-01"]
    K_list = SCORE_K_LIST
    variants = ["baseline_fifo", "operator_lru"]

    scenarios_all = load_scenarios()
    scenarios = [s for s in scenarios_all if s["scenario_id"] in SCENARIO_IDS]
    if len(scenarios) != len(SCENARIO_IDS):
        found = {s["scenario_id"] for s in scenarios}
        missing = set(SCENARIO_IDS) - found
        raise SystemExit(f"Missing scenarios: {missing}")
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

    sqlite_path = RESULTS_DIR / "eventmemory_shared_harness_v4.sqlite3"
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
    SCEN_SEM = asyncio.Semaphore(int(os.getenv("SH_SCEN_CONCURRENCY", "2")))

    async def _run_scen(scenario: dict) -> dict[str, Any]:
        async with SCEN_SEM:
            sid = scenario["scenario_id"]
            print(f"[run] {sid} (variants={variants})", flush=True)
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
                    variants=variants,
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
        row = {"scenario_id": r["scenario_id"], "category": r.get("category", "")}
        for variant in variants:
            agg = r["per_variant"][variant]["aggregates"]
            agent_result = r["per_variant"][variant]
            n_step_emits = sum(t.get("n_step_outs", 0) for t in agent_result["trace"])
            n_retrieves_total = sum(
                t.get("n_retrieves", 0) for t in agent_result["trace"]
            )
            n_compactor = sum(
                1 for t in agent_result["trace"] if t.get("compactor_invoked")
            )
            n_dropped = sum(
                len(t.get("compactor_dropped", [])) for t in agent_result["trace"]
            )
            n_fallback = sum(
                len(t.get("fallback_dropped", [])) for t in agent_result["trace"]
            )
            row[variant] = {
                "coverage_rate": agg["coverage_rate"],
                "full_R@5": agg.get("triggered_recall_full@5"),
                "cond_R@5": agg.get("recall_given_covered@5"),
                "full_R@10": agg.get("triggered_recall_full@10"),
                "n_turns": agent_result["n_turns"],
                "parse_failures": agent_result["parse_failures"],
                "max_wm_tokens": agent_result["max_wm_tokens"],
                "n_step_emits": n_step_emits,
                "n_step_finals": len(agent_result["step_outputs"]),
                "n_retrieves_total": n_retrieves_total,
                "compactor_calls": n_compactor,
                "compactor_dropped": n_dropped,
                "fallback_dropped": n_fallback,
            }
        summary["scenarios"].append(row)

    for variant in variants:
        cov_vals = []
        full5_vals = []
        full10_vals = []
        cond5_vals = []
        wm_vals = []
        n_step_emits_total = 0
        n_retrieves_grand = 0
        n_turns_total = 0
        parse_failures_total = 0
        compactor_calls_total = 0
        compactor_dropped_total = 0
        fallback_dropped_total = 0
        for r in results:
            if "error" in r:
                continue
            agg = r["per_variant"][variant]["aggregates"]
            agent_result = r["per_variant"][variant]
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
            wm_vals.append(agent_result["max_wm_tokens"])
            parse_failures_total += agent_result["parse_failures"]
            n_turns_total += agent_result["n_turns"]
            n_step_emits_total += sum(
                t.get("n_step_outs", 0) for t in agent_result["trace"]
            )
            n_retrieves_grand += sum(
                t.get("n_retrieves", 0) for t in agent_result["trace"]
            )
            compactor_calls_total += sum(
                1 for t in agent_result["trace"] if t.get("compactor_invoked")
            )
            compactor_dropped_total += sum(
                len(t.get("compactor_dropped", [])) for t in agent_result["trace"]
            )
            fallback_dropped_total += sum(
                len(t.get("fallback_dropped", [])) for t in agent_result["trace"]
            )
        summary["per_variant_means"][variant] = {
            "coverage_mean": round(sum(cov_vals) / len(cov_vals), 4)
            if cov_vals
            else None,
            "full_R@5_mean": round(sum(full5_vals) / len(full5_vals), 4)
            if full5_vals
            else None,
            "full_R@10_mean": round(sum(full10_vals) / len(full10_vals), 4)
            if full10_vals
            else None,
            "cond_R@5_mean": round(sum(cond5_vals) / len(cond5_vals), 4)
            if cond5_vals
            else None,
            "max_wm_tokens_mean": round(sum(wm_vals) / len(wm_vals), 1)
            if wm_vals
            else None,
            "max_wm_tokens_max": max(wm_vals) if wm_vals else None,
            "parse_failures_total": parse_failures_total,
            "n_turns_total": n_turns_total,
            "n_step_emits_total": n_step_emits_total,
            "n_retrieves_total": n_retrieves_grand,
            "step_emits_per_turn": round(n_step_emits_total / n_turns_total, 4)
            if n_turns_total
            else None,
            "retrieves_per_turn": round(n_retrieves_grand / n_turns_total, 4)
            if n_turns_total
            else None,
            "compactor_calls_total": compactor_calls_total,
            "compactor_dropped_total": compactor_dropped_total,
            "fallback_dropped_total": fallback_dropped_total,
        }

    summary_path = THIS_DIR / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== Cross-scenario means (v4 verbatim cuts) ===")
    for variant in variants:
        m = summary["per_variant_means"][variant]
        print(
            f"  {variant}: cov={m['coverage_mean']} | full_R@5={m['full_R@5_mean']} | "
            f"full_R@10={m['full_R@10_mean']} | cond_R@5={m['cond_R@5_mean']} | "
            f"step_emits/turn={m['step_emits_per_turn']} | "
            f"retrieves/turn={m['retrieves_per_turn']} | "
            f"max_wm_mean={m['max_wm_tokens_mean']} | parse_fail={m['parse_failures_total']} | "
            f"compactor_calls={m['compactor_calls_total']} | dropped={m['compactor_dropped_total']}"
        )
    print("\n=== Reference ===")
    print("  v3 op:    cov=0.7181 | full_R@5=0.3015")
    print("  v1 op:    cov=0.581  | full_R@5=0.434")
    print("  SA-full:  cov=0.876  | full_R@5=0.646")
    print(f"\nWrote {summary_path}")
    print(f"Per-scenario per-variant files in {RESULTS_OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
