"""Shared harness — bounded WM agent loop vs SA-full reference.

Tests two variants of a bounded-WM-with-retrieval architecture against the
existing 10-hard mid-execution scenarios with REAL EM and LoCoMo distractors.

Variants:
  - baseline_fifo : flat FIFO WM (no LRU/citations) at 10k token cap.
  - operator_lru  : two-tier WM (Tier 1 transient retrieval / Tier 2 active LRU
                    verbatim cap=15 / Tier 3 compressed older cap=35) +
                    citation-driven LRU + transitive citation propagation.

Reference: SA-full (mid_execution_eval_e2.py spreading_activation_full mode)
on the same 10 hard scenarios reported cov 0.876 / full_R@5 0.646.

Usage:
    uv run python evaluation/associative_recall/metacog/shared_harness/main.py
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
TIER2_MAX_ITEMS = 15
TIER3_MAX_ITEMS = 35  # ranks 16-50

MAX_TURNS = 14
RETRIEVE_K = 5
MAX_COMPLETION_TOKENS = 4500
SCORE_K_LIST = [1, 3, 5, 10]

# tiktoken
try:
    ENC = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


# ---------- Items / IDs ----------


@dataclass
class Item:
    item_id: str  # e.g. "chat-3", "ret-7", "reason-2", "step-1", "assess-4"
    kind: str  # chat | ret | reason | step | assess
    content: str  # full text content (verbatim form)
    last_referenced_turn: int = 0
    summary: str | None = None  # set when demoted to Tier 3

    def render_verbatim(self) -> str:
        return f"[{self.item_id}] ({self.kind}, last_ref=t{self.last_referenced_turn}):\n{self.content}"

    def render_summary(self) -> str:
        s = self.summary or self.content[:140]
        return f"[{self.item_id}] ({self.kind}, last_ref=t{self.last_referenced_turn}): {s}"

    def size_tokens(self) -> int:
        if self.summary:
            return n_tokens(self.render_summary())
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


# ---------- Priming prompts ----------

PRIMING_OPERATOR = """\
You are a memory-augmented agent solving a multi-step task with bounded \
working memory. You have a retrieval tool over a past-conversation memory \
that contains specific facts (constraints, preferences, allergies, dates, \
numbers, identities) the user has shared previously. THESE FACTS MATERIALLY \
CHANGE THE TASK OUTPUT — without retrieving them first you will write \
generic placeholder answers that miss real binding constraints.

The task description ALONE does not contain those facts. They live in the \
memory tool. Your job: probe memory aggressively in the first few turns to \
surface those facts, THEN write step_outputs that USE the surfaced facts \
(real names, real numbers, real constraints from memory).

WORKFLOW EXPECTATION:
- Turns 1-4: mostly retrieve actions. Cast a wide net with diverse cues. \
Each step in the task likely has a memory-grounded constraint — probe for \
it. Probe for: people involved (names, allergies, preferences), dates / \
deadlines, vendors / venues, budget / numbers, prior incidents, supersession \
(later messages overriding earlier ones). Use 1-3 retrieve actions per turn.
- Turns 4-10: mix retrieval with step_output actions as facts surface. \
Continue probing for missing dimensions (re-read the task — what dimensions \
haven't you probed yet?).
- Final turns: finish remaining step_outputs and emit done.

DO NOT emit `done` on turn 1 unless you've made multiple retrieve calls \
and the surfaced facts are fully reflected in your step_outputs.

Your working memory has three tiers, shown to you each turn:
- TIER 1 (transient retrieval buffer): the most recent retrieval's full hits, \
each shown verbatim with its chat-N id. Replaced each retrieval round.
- TIER 2 (active LRU, verbatim, cap ~15): items you've recently CITED. The \
system tracks what you cite to keep relevant items active.
- TIER 3 (compressed older items): one-line summaries of items you stopped \
citing. IDs are preserved so you can re-promote them by citing.

THE CITATION CONTRACT — central mechanism:
- Cite an item like [chat-12] or [ret-3] when you USE that item's content. \
Citation = "I am using this right now".
- When you cite, the system refreshes the item's LRU position. Items you \
stop citing fall out as new items get cited.
- INLINE-QUOTE load-bearing content next to the citation marker. The [id] \
is for traceability; the inline quoted content is for clarity AND for \
robustness against compression. Example: "Maya has a strict shellfish \
allergy [chat-43]" — the words before [chat-43] make the reasoning survive \
if [chat-43] later compresses.
- Cite REAL ids visible in the WM. Do not invent.

OUTPUT SCHEMA (strict JSON, every turn — output ONLY the JSON object, no \
markdown fences, no preamble):
{
  "reasoning": "<free prose with inline citations like [chat-3]. State \
what you've learned, what's missing, what to retrieve next or emit next.>",
  "actions": [
    {"type": "retrieve", "cue": "<short content-bearing query>", "reason": "<why>"},
    {"type": "assess",   "re": "<item-id>", "did_advance": true_or_false, "why": "<short>"},
    {"type": "step_output", "step_id": <int>, "label": "<short>", "content": "<the deliverable: real names, real numbers, citing the items you used>"},
    {"type": "done"}
  ]
}

Action-type guidance:
- retrieve: ask EM with a content-bearing query. Hits land in Tier 1.
- assess: note whether a retrieval advanced you. Tracks productivity.
- step_output: emit the deliverable for one sub-step. 1-3 sentences. Use \
SPECIFIC values surfaced from memory (cite them inline). Avoid generic \
placeholders if a real fact is in memory.
- done: when all sub-decisions are addressed AND retrieval has saturated.

You may emit multiple actions per turn. Common patterns:
- Early turn: 2-3 retrieve actions exploring different dimensions.
- Mid turn: 1-2 retrieve + 1-2 step_output as facts solidify.
- Late turn: step_outputs + done.

EFFORT SIGNALS visible each turn (rounds elapsed, new-fact rate trend, WM \
tokens used) inform whether to keep retrieving or finalize.

SATURATION: stop retrieving when probes consistently surface no new content. \
Keep going while productive."""


PRIMING_BASELINE = """\
You are a memory-augmented agent solving a multi-step task with bounded \
working memory. You have a retrieval tool over a past-conversation memory \
that contains specific facts (constraints, preferences, allergies, dates, \
numbers, identities) the user has shared previously. THESE FACTS MATERIALLY \
CHANGE THE TASK OUTPUT — without retrieving them first you will write \
generic placeholder answers that miss real binding constraints.

The task description ALONE does not contain those facts. They live in the \
memory tool. Your job: probe memory aggressively in the first few turns to \
surface those facts, THEN write step_outputs that USE the surfaced facts.

WORKFLOW EXPECTATION:
- Turns 1-4: mostly retrieve. Probe for: people involved (names, \
allergies, preferences), dates / deadlines, vendors / venues, budget / \
numbers, prior incidents, supersession.
- Turns 4-10: mix retrieve with step_output as facts surface. Keep \
probing for missing dimensions.
- Final turns: finish step_outputs, emit done.

DO NOT emit `done` on turn 1 unless you've made multiple retrieve calls \
and surfaced facts are reflected in step_outputs.

Your working memory shows newest items at the top. When WM exceeds the \
token budget, oldest items get FIFO-evicted.

Cite items like [chat-3] or [ret-7] when you rely on them. INLINE-QUOTE \
the load-bearing content next to the citation marker so your reasoning \
survives if items fall off. Cite REAL ids visible in WM — don't invent.

OUTPUT SCHEMA (strict JSON, every turn — output ONLY the JSON object, no \
markdown fences, no preamble):
{
  "reasoning": "<free prose with inline citations like [chat-3]. State \
what you learned, what's missing, what to retrieve or emit next.>",
  "actions": [
    {"type": "retrieve", "cue": "<short content-bearing query>", "reason": "<why>"},
    {"type": "assess",   "re": "<item-id>", "did_advance": true_or_false, "why": "<short>"},
    {"type": "step_output", "step_id": <int>, "label": "<short>", "content": "<the deliverable: real names, real numbers from memory>"},
    {"type": "done"}
  ]
}

Action-type guidance:
- retrieve: ask EM. Hits land at the top of WM.
- assess: note whether a retrieval advanced you.
- step_output: 1-3 sentences. Use SPECIFIC values surfaced from memory.
- done: when sub-decisions covered and retrieval has saturated.

You may emit multiple actions per turn. Use effort signals (rounds, \
new-fact rate, WM tokens) to decide retrieve vs. finalize. If probes \
saturate, stop retrieving."""


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
    parts.append("\n--- TIER 3 (compressed older, ids preserved) ---")
    if state.tier3:
        for it in state.tier3:
            parts.append(it.render_summary())
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


# ---------- Prompts ----------


def build_user_prompt(
    *,
    task_prompt: str,
    turn: int,
    wm_text: str,
    retrieval_log: list[dict[str, Any]],
    effort: dict[str, Any],
    step_outputs: list[dict[str, Any]],
) -> str:
    parts = []
    parts.append(f"TASK:\n{task_prompt}\n")
    parts.append(f"TURN: {turn}")
    parts.append(
        f"EFFORT SIGNALS: rounds_elapsed={effort['rounds']}, "
        f"new_fact_rate_last3={effort['new_fact_rate']:.2f}, "
        f"wm_tokens={effort['wm_tokens']} (cap={WM_HARD_CAP})"
    )
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
    parts.append(f"--- STEP OUTPUTS SO FAR ({len(step_outputs)}) ---")
    if step_outputs:
        for so in step_outputs:
            parts.append(f"  step_id={so['step_id']} label={so['label']!r}")
    else:
        parts.append("(none yet)")
    parts.append("")
    parts.append("Respond with the JSON object now.")
    return "\n".join(parts)


# ---------- Operator state ----------


@dataclass
class OperatorState:
    tier1: list[Item] = field(default_factory=list)
    tier2: list[Item] = field(default_factory=list)
    tier3: list[Item] = field(default_factory=list)
    all_items: dict[str, Item] = field(default_factory=dict)
    retrieval_log: list[dict[str, Any]] = field(default_factory=list)

    def register(self, item: Item) -> None:
        self.all_items[item.item_id] = item

    def add_to_tier2(self, item: Item) -> None:
        self.tier2.insert(0, item)
        self.register(item)

    def replace_tier1(self, items: list[Item]) -> None:
        # Items in old Tier 1 that were never cited get dropped from registry.
        for old in self.tier1:
            self.all_items.pop(old.item_id, None)
        self.tier1 = list(items)
        for it in items:
            self.register(it)

    def cite(self, item_id: str, turn: int) -> str:
        if item_id not in self.all_items:
            return "unknown"
        it = self.all_items[item_id]
        it.last_referenced_turn = turn
        # promote
        if it in self.tier1:
            self.tier1.remove(it)
            self.tier2.insert(0, it)
            return "tier1->tier2"
        if it in self.tier2:
            self.tier2.remove(it)
            self.tier2.insert(0, it)
            return "tier2->tier2(refresh)"
        if it in self.tier3:
            # rehydrate verbatim
            it.summary = None
            self.tier3.remove(it)
            self.tier2.insert(0, it)
            return "tier3->tier2(rehydrate)"
        return "untracked"

    def transitive_propagate(
        self, cited_ids: list[str], turn: int, depth: int = 2
    ) -> list[str]:
        """For each cited id, parse its content and touch any [id] within. Depth-limited."""
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
        s += "\n" + "\n".join(i.render_summary() for i in self.tier3)
        return n_tokens(s)

    def enforce_caps(self, turn: int, summarizer) -> dict[str, Any]:
        cap_log: dict[str, Any] = {"demoted_to_t3": [], "dropped": []}
        # 1. Tier3 cap: drop oldest beyond TIER3_MAX_ITEMS
        # Tier 3 is naturally in newest-first order (insert at 0)
        while len(self.tier3) > TIER3_MAX_ITEMS:
            victim = self.tier3.pop()  # oldest
            self.all_items.pop(victim.item_id, None)
            cap_log["dropped"].append(victim.item_id)
        # 2. Tier 2 by item count (sort by last_referenced desc and pop tail)
        if len(self.tier2) > TIER2_MAX_ITEMS:
            self.tier2.sort(key=lambda x: -x.last_referenced_turn)
            while len(self.tier2) > TIER2_MAX_ITEMS:
                victim = self.tier2.pop()  # oldest
                self._demote(victim, summarizer)
                cap_log["demoted_to_t3"].append(victim.item_id)
        # 3. Token cap: if over, demote oldest tier2 items
        max_iters = 30
        while self.verbatim_tokens() > WM_HARD_CAP and self.tier2 and max_iters > 0:
            self.tier2.sort(key=lambda x: -x.last_referenced_turn)
            victim = self.tier2.pop()
            self._demote(victim, summarizer)
            cap_log["demoted_to_t3"].append(victim.item_id)
            max_iters -= 1
        # 4. If still over, drop tier3 tail
        max_iters = 30
        while self.verbatim_tokens() > WM_HARD_CAP and self.tier3 and max_iters > 0:
            victim = self.tier3.pop()
            self.all_items.pop(victim.item_id, None)
            cap_log["dropped"].append(victim.item_id)
            max_iters -= 1
        # Re-cap tier3 again after potential demotions
        while len(self.tier3) > TIER3_MAX_ITEMS:
            victim = self.tier3.pop()
            self.all_items.pop(victim.item_id, None)
            cap_log["dropped"].append(victim.item_id)
        return cap_log

    def _demote(self, item: Item, summarizer) -> None:
        if not item.summary:
            item.summary = summarizer(item)
        self.tier3.insert(0, item)


# ---------- Baseline FIFO state ----------


@dataclass
class BaselineState:
    wm: list[Item] = field(default_factory=list)  # newest-first
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
            victim = self.wm.pop()  # oldest
            self.all_items.pop(victim.item_id, None)
            evicted.append(victim.item_id)
            max_iters -= 1
        return {"evicted": evicted}


def simple_summarize(item: Item) -> str:
    """1-line summary preserving the item-id structure. Cheap, no LLM call.

    Note: in the real architecture we'd call a small LLM here, but for keeping
    the harness fast and deterministic we use a content-preserving truncation.
    The architecture's load-bearing claim is INLINE-QUOTE PRIMING — content the
    model wrote inline near a citation survives compression. This summarizer
    keeps the first 140 chars which usually covers the inline-quoted content.
    """
    text = item.content.replace("\n", " ").strip()
    if len(text) <= 140:
        return text
    return text[:137] + "..."


# ---------- Hit-rendering helpers ----------


def hit_to_chat_item_text(hit, fallback_idx: int) -> tuple[str, str]:
    """Convert a probe Hit into (chat_id, content_text).

    Use the chat ID = "chat-{turn_id}" so when the model cites [chat-NN], it
    refers to the underlying chat turn, NOT a separate ret-N id. This is the
    unified-id-namespace fix from the prior test.
    """
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
    citations: list[str]
    transitive_touched: list[str]
    actions_summary: list[dict[str, Any]]
    tier1_ids: list[str] = field(default_factory=list)
    tier2_ids: list[tuple[str, int]] = field(default_factory=list)
    tier3_ids: list[tuple[str, int]] = field(default_factory=list)
    promoted: list[tuple[str, str]] = field(default_factory=list)  # (id, where)
    demoted: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    new_facts: int = 0


async def run_agent_loop(
    *,
    scenario: dict,
    memory,
    openai_client,
    variant: str,
) -> dict[str, Any]:
    """One agent loop run. variant in {"baseline_fifo", "operator_lru"}."""
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

    # Counters per kind
    counters = {"chat": 1000, "ret": 0, "reason": 0, "step": 0, "assess": 0}
    # Reserve "chat-{turn_id}" for chat ids; counters['chat'] only used for non-EM chat events (none here).

    trace: list[TurnLog] = []
    step_outputs: list[dict[str, Any]] = []
    new_facts_per_turn: list[int] = []
    parse_failures = 0
    citations_per_turn: list[int] = []
    seen_chat_ids: set[str] = set()

    # Seed the user task as chat-0 so the model can cite it.
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
        # --- effort signals ---
        recent = sum(new_facts_per_turn[-3:])
        denom = max(1, min(3, turn - 1))
        effort = {
            "rounds": turn - 1,
            "new_fact_rate": (recent / denom) if turn > 1 else 0.0,
            "wm_tokens": state.verbatim_tokens()
            if isinstance(state, OperatorState)
            else state.tokens(),
        }

        if isinstance(state, OperatorState):
            wm_text = render_wm_operator(state)
        else:
            wm_text = render_wm_baseline(state)

        user_prompt = build_user_prompt(
            task_prompt=task_prompt,
            turn=turn,
            wm_text=wm_text,
            retrieval_log=state.retrieval_log,
            effort=effort,
            step_outputs=step_outputs,
        )

        try:
            raw = await llm_chat(openai_client, priming, user_prompt)
        except Exception as exc:
            raw = ""
            log = TurnLog(
                turn=turn,
                wm_tokens_before=effort["wm_tokens"],
                wm_tokens_after=effort["wm_tokens"],
                parse_ok=False,
                raw_excerpt=f"LLM ERROR: {exc!r}"[:300],
                reasoning_excerpt="",
                citations=[],
                transitive_touched=[],
                actions_summary=[],
            )
            trace.append(log)
            new_facts_per_turn.append(0)
            citations_per_turn.append(0)
            parse_failures += 1
            continue

        parsed = parse_json_output(raw)
        log = TurnLog(
            turn=turn,
            wm_tokens_before=effort["wm_tokens"],
            wm_tokens_after=effort["wm_tokens"],
            parse_ok=parsed is not None,
            raw_excerpt=raw[:400],
            reasoning_excerpt="",
            citations=[],
            transitive_touched=[],
            actions_summary=[],
        )
        if isinstance(state, OperatorState):
            log.tier1_ids = [i.item_id for i in state.tier1]
            log.tier2_ids = [(i.item_id, i.last_referenced_turn) for i in state.tier2]
            log.tier3_ids = [(i.item_id, i.last_referenced_turn) for i in state.tier3]

        if parsed is None:
            parse_failures += 1
            new_facts_per_turn.append(0)
            citations_per_turn.append(0)
            # Don't lose the turn entirely — emit a degraded reason item
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
                cap_log = state.enforce_caps(turn, simple_summarize)
                log.demoted = cap_log["demoted_to_t3"]
                log.dropped = cap_log["dropped"]
                log.wm_tokens_after = state.verbatim_tokens()
            else:
                state.push(r_item)
                ev = state.enforce_cap()
                log.dropped = ev["evicted"]
                log.wm_tokens_after = state.tokens()
            trace.append(log)
            continue

        reasoning_text = (parsed.get("reasoning") or "").strip()
        actions = parsed.get("actions") or []
        if not isinstance(actions, list):
            actions = []
        log.reasoning_excerpt = reasoning_text[:300]

        # Insert reason item
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

        # Parse citations from reasoning + action fields
        cite_strs = list(reasoning_text or "")
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

        # Apply citations (LRU promotion) + transitive propagation (operator only)
        if isinstance(state, OperatorState):
            for cid in cited:
                where = state.cite(cid, turn)
                if where != "unknown":
                    log.promoted.append((cid, where))
            touched = state.transitive_propagate(cited, turn, depth=2)
            log.transitive_touched = touched

        # Execute actions
        new_facts = 0
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
                try:
                    hits = await probe(memory, cue, RETRIEVE_K)
                except Exception as exc:
                    log.actions_summary.append(
                        {"type": "retrieve", "error": repr(exc)[:120]}
                    )
                    continue
                # Build chat-id-keyed items
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
                    # Reuse prior chat item if already registered
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
                # Also record a "ret" item — tracking the cue itself
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
                    # baseline pushes hits onto top of WM individually
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
                step_id = a.get("step_id") or (len(step_outputs) + 1)
                label = a.get("label") or ""
                content = a.get("content") or ""
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
                step_outputs.append(
                    {
                        "step_id": int(step_id)
                        if isinstance(step_id, (int, str)) and str(step_id).isdigit()
                        else len(step_outputs) + 1,
                        "label": str(label)[:200],
                        "content": str(content),
                        "item_id": sid_item,
                        "turn": turn,
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

        # Enforce caps
        if isinstance(state, OperatorState):
            cap_log = state.enforce_caps(turn, simple_summarize)
            log.demoted = cap_log["demoted_to_t3"]
            log.dropped = cap_log["dropped"]
            log.wm_tokens_after = state.verbatim_tokens()
        else:
            ev = state.enforce_cap()
            log.dropped = ev["evicted"]
            log.wm_tokens_after = state.tokens()

        trace.append(log)
        if done_emitted:
            break

    return {
        "scenario_id": sid,
        "variant": variant,
        "trace": [td_to_dict(t) for t in trace],
        "step_outputs": step_outputs,
        "parse_failures": parse_failures,
        "n_turns": len(trace),
        "max_wm_tokens": max((t.wm_tokens_after for t in trace), default=0),
        "citations_per_turn": citations_per_turn,
    }


def td_to_dict(t: TurnLog) -> dict[str, Any]:
    return {
        "turn": t.turn,
        "wm_tokens_before": t.wm_tokens_before,
        "wm_tokens_after": t.wm_tokens_after,
        "parse_ok": t.parse_ok,
        "raw_excerpt": t.raw_excerpt,
        "reasoning_excerpt": t.reasoning_excerpt,
        "citations": t.citations,
        "transitive_touched": t.transitive_touched,
        "actions_summary": t.actions_summary,
        "tier1_ids": t.tier1_ids,
        "tier2_ids": t.tier2_ids,
        "tier3_ids": t.tier3_ids,
        "promoted": t.promoted,
        "demoted": t.demoted,
        "dropped": t.dropped,
        "new_facts": t.new_facts,
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
    """Coverage judge + per-gold retrieval probe; mirrors run_scenario_e2."""
    plants_by_id = {
        p["plant_id"]: p for p in scenario["preamble_turns"] if p.get("plant_id")
    }

    step_outputs = agent_result["step_outputs"]
    steps_by_id: dict[int, dict[str, Any]] = {}
    for so in step_outputs:
        steps_by_id[int(so["step_id"])] = so

    # Build transcript: PLAN-style listing + step outputs body
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
        # Per-scenario per-variant file
        fp = RESULTS_OUT_DIR / f"{sid}_{variant}.json"
        fp.write_text(json.dumps(per, indent=2, default=str))
        agg = score["aggregates"]
        cov = agg["coverage_rate"]
        full = agg.get("triggered_recall_full@5", "n/a")
        cond = agg.get("recall_given_covered@5", "n/a")
        print(
            f"  [{sid}] {variant}: cov={cov} | full_R@5={full} | cond_R@5={cond} | "
            f"turns={agent_result['n_turns']} | parse_fail={agent_result['parse_failures']} | "
            f"max_wm={agent_result['max_wm_tokens']}",
            flush=True,
        )
        return variant, per

    pairs = await asyncio.gather(*[_run_one(v) for v in variants])
    for variant, per in pairs:
        out["per_variant"][variant] = per
    return out


async def main() -> None:
    HARD_INDICES = list(range(10, 20))
    K_list = SCORE_K_LIST
    variants = ["baseline_fifo", "operator_lru"]

    scenarios_all = load_scenarios()
    scenarios = [scenarios_all[i] for i in HARD_INDICES]
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

    sqlite_path = RESULTS_DIR / "eventmemory_shared_harness.sqlite3"
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
    # Concurrency limit so we don't spike OpenAI/Qdrant. Each scenario runs
    # both variants in parallel internally (see run_one_scenario), so SCEN
    # concurrency=3 means up to 6 agent loops in flight.
    SCEN_SEM = asyncio.Semaphore(int(os.getenv("SH_SCEN_CONCURRENCY", "3")))

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

    # Cross-scenario summary
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
            row[variant] = {
                "coverage_rate": agg["coverage_rate"],
                "full_R@5": agg.get("triggered_recall_full@5"),
                "cond_R@5": agg.get("recall_given_covered@5"),
                "full_R@10": agg.get("triggered_recall_full@10"),
                "n_turns": r["per_variant"][variant]["n_turns"],
                "parse_failures": r["per_variant"][variant]["parse_failures"],
                "max_wm_tokens": r["per_variant"][variant]["max_wm_tokens"],
            }
        summary["scenarios"].append(row)

    for variant in variants:
        cov_vals = []
        full5_vals = []
        full10_vals = []
        cond5_vals = []
        wm_vals = []
        parse_failures_total = 0
        n_turns_total = 0
        for r in results:
            if "error" in r:
                continue
            agg = r["per_variant"][variant]["aggregates"]
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
            wm_vals.append(r["per_variant"][variant]["max_wm_tokens"])
            parse_failures_total += r["per_variant"][variant]["parse_failures"]
            n_turns_total += r["per_variant"][variant]["n_turns"]
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
        }

    summary_path = THIS_DIR / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== Cross-scenario means ===")
    for variant in variants:
        m = summary["per_variant_means"][variant]
        print(
            f"  {variant}: cov={m['coverage_mean']} | full_R@5={m['full_R@5_mean']} | "
            f"full_R@10={m['full_R@10_mean']} | cond_R@5={m['cond_R@5_mean']} | "
            f"max_wm_mean={m['max_wm_tokens_mean']} | max_wm_max={m['max_wm_tokens_max']} | "
            f"parse_fail={m['parse_failures_total']}"
        )
    print("\n=== Reference (SA-full on 10 hard) ===")
    print("  cov=0.876 | full_R@5=0.646")
    print(f"\nWrote {summary_path}")
    print(f"Per-scenario per-variant files in {RESULTS_OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
