"""Two-tier WM + citation-driven LRU + transient retrieval buffer.

Architecture under test (variant=operator):
  - Tier 1 (Transient): last retrieval's full hits. Replaced on each retrieval.
  - Tier 2 (Active LRU): cap = 15 items, verbatim. Sorted by last_referenced_turn desc.
                         Updated ONLY when the model cites the item ([item-id]).
  - Tier 3 (Compressed older): positions 16-50, 1-line summaries with IDs preserved.
  - Dropped (>50). Still retrievable from external memory if chat-derived.

Variant=baseline: naive FIFO. Each retrieval pushes hits into a flat WM list at top;
oldest items get compressed when WM tokens > 10k.

Run: uv run python evaluation/associative_recall/metacog/lru_citation/main.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI

THIS_DIR = Path(__file__).resolve().parent
ENV_PATH = THIS_DIR.parent.parent / ".env"
load_dotenv(ENV_PATH)

MODEL = "gpt-5-mini"
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Architecture caps
WM_HARD_CAP = 10_000
TIER2_MAX_ITEMS = 15
TIER3_MAX_ITEMS = 35  # positions 16-50

MAX_TURNS = 14

try:
    ENC = tiktoken.encoding_for_model("gpt-5-mini")
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


# -------------------- Test cases -------------------- #

# Each case = a multi-step task whose solution requires retrieving 4-7 specific
# facts from an external memory of mixed relevant + noise content.

# ---- Case 1: Plan a constrained 5-day trip ---- #
CASE1_REAL = [
    "BOOKING CONSTRAINT (binding): The Tokyo apartment booking is confirmed for "
    "nights of Mon-Fri (5 nights). Check-in is from 16:00 local time on Monday; "
    "check-out is by 11:00 on Saturday. Reservation #TK-2026-44910. Late check-in "
    "after 21:00 incurs a 8000 JPY fee per the host policy.",
    "TRANSIT CONSTRAINT (binding): The 7-day JR Pass was activated on the Sunday "
    "prior, so it is valid through the following Saturday. The pass covers all JR "
    "lines including the Narita Express, but does NOT cover the Tokyo Metro or "
    "the Toei Subway. Local subway fares average 220 JPY per trip and must be paid "
    "via Suica/Pasmo IC card.",
    "DIETARY CONSTRAINT (binding): Travel companion Maya has a strict shellfish "
    "allergy (anaphylaxis risk). All restaurant choices must confirm shellfish-free "
    "kitchens; izakaya with shared fryers are unsafe. Maya carries an epinephrine "
    "auto-injector; the nearest English-speaking ER is St. Luke's International "
    "Hospital in Tsukiji.",
    "BUDGET CONSTRAINT (binding): The total trip food+activities budget is 75000 "
    "JPY for the two travelers across 5 days. This excludes lodging (already paid) "
    "and the JR Pass (already paid). Daily soft target: 15000 JPY. Two splurge "
    "meals up to 9000 JPY each are pre-approved.",
    "CALENDAR CONSTRAINT (binding): A non-negotiable work call is scheduled for "
    "Wednesday 09:00-10:30 local Tokyo time. The apartment WiFi has been pre-tested "
    "at 180 Mbps. No external excursions on Wednesday morning before 11:00. The "
    "Wednesday afternoon is free.",
    "EVENT CONSTRAINT (binding): Tickets to the teamLab Borderless exhibit are "
    "pre-purchased for Thursday 14:00 entry; the slot is non-refundable and non-"
    "transferable. The venue is in Azabudai Hills; closest station is Kamiyacho on "
    "the Hibiya line (Tokyo Metro, NOT JR-covered).",
]

CASE1_TASK = (
    "Plan a 5-day Tokyo itinerary for two travelers (Mon-Fri) that respects all "
    "binding booking, transit, dietary, budget, calendar, and event constraints. "
    "For each day, propose: morning activity, lunch, afternoon activity, dinner. "
    "Flag any decision that depends on a binding constraint by citing the "
    "constraint."
)

CASE1_GOLD = [
    {
        "qid": "g1",
        "q": "What is the apartment check-in time and reservation number?",
        "keys": ["16:00", "4 pm", "TK-2026-44910"],
    },
    {
        "qid": "g2",
        "q": "Does the JR Pass cover the Tokyo Metro?",
        "keys": ["does not cover", "not cover", "no", "metro is not"],
    },
    {
        "qid": "g3",
        "q": "What is Maya's dietary restriction and where is the recommended ER?",
        "keys": ["shellfish", "St. Luke", "Tsukiji"],
    },
    {
        "qid": "g4",
        "q": "What is the total food+activities budget for the 5 days?",
        "keys": ["75000", "75,000", "75 000", "75k"],
    },
    {
        "qid": "g5",
        "q": "When is the Wednesday work call and what is the implication for excursions?",
        "keys": ["09:00", "9:00", "10:30", "no excursion", "no external", "before 11"],
    },
    {
        "qid": "g6",
        "q": "What pre-purchased event is on Thursday and at what time?",
        "keys": ["teamLab", "14:00", "2 pm", "thursday"],
    },
]


# ---- Case 2: Design a meeting agenda matching multiple stakeholders' constraints ---- #
CASE2_REAL = [
    "STAKEHOLDER A (binding): Priya from Engineering can only attend 60 minutes "
    "max, must leave by 15:00 sharp for a customer escalation. She owns the "
    "PostgresMigration agenda item and refuses to defer it. She is OK presenting "
    "first or last but NOT in the middle.",
    "STAKEHOLDER B (binding): Marcus from Finance requires 20 minutes uninterrupted "
    "for the budget reforecast walkthrough. Marcus has a hard stop at 15:30. He "
    "cannot present before Priya because his numbers depend on her capacity "
    "estimates being shared first.",
    "STAKEHOLDER C (binding): Lin from Product needs 15 minutes for the Q2 "
    "roadmap reprioritization. Lin is remote from Singapore (timezone GMT+8); the "
    "meeting at 14:00 New York time is 02:00 in Singapore — Lin will join but "
    "asked to present in the FIRST 30 minutes to minimize her late-night exposure.",
    "STAKEHOLDER D (binding): Diego from Legal needs only 5 minutes for the "
    "compliance signoff on the data-residency change. Diego is flexible on slot "
    "but MUST be on the recording so the legal trail is preserved. The meeting "
    "platform records automatically once the host clicks 'Start recording'.",
    "ROOM CONSTRAINT (binding): Conference room Olympus is booked 14:00-15:30 "
    "(90 min) on Friday. The room has a 14-person capacity, hybrid AV, and a "
    "30-minute auto-release — if the room sits empty for 30 minutes between "
    "sessions, the booking auto-releases. Late start beyond 14:15 risks loss.",
    "ATTENDANCE NOTE (binding): The CEO Aiko will drop in for the last 10 minutes "
    "for a brief Q&A. Her arrival is fixed at 15:20. The agenda must reserve "
    "15:20-15:30 for CEO Q&A and not let earlier items run over into that slot.",
]

CASE2_TASK = (
    "Design the agenda for Friday's 14:00-15:30 leadership sync that respects "
    "every stakeholder's constraints (presence windows, ordering dependencies, "
    "duration needs), the room booking limits, and the CEO's drop-in. For each "
    "agenda item, list: start time, end time, owner, topic. Flag any decision "
    "that depends on a binding constraint by citing the constraint."
)

CASE2_GOLD = [
    {
        "qid": "g1",
        "q": "What is Priya's hard departure time and which item must she present?",
        "keys": ["15:00", "3 pm", "PostgresMigration", "postgres migration"],
    },
    {
        "qid": "g2",
        "q": "Why must Marcus present after Priya?",
        "keys": ["depend", "capacity", "numbers"],
    },
    {
        "qid": "g3",
        "q": "Why must Lin present in the first 30 minutes?",
        "keys": ["singapore", "02:00", "2:00", "late", "timezone", "late-night"],
    },
    {
        "qid": "g4",
        "q": "What is Diego's strict requirement for the meeting?",
        "keys": ["recording", "record", "legal trail"],
    },
    {
        "qid": "g5",
        "q": "What is the room auto-release rule?",
        "keys": ["30 minute", "30-minute", "auto-release", "auto release", "empty"],
    },
    {
        "qid": "g6",
        "q": "When does the CEO arrive and how long is reserved for her?",
        "keys": ["15:20", "10 minute", "10-minute", "Q&A"],
    },
]


def _filler_paragraph(topic: str, seed: int) -> str:
    rng = random.Random(seed)
    sentences = [
        f"In {topic} planning, the canonical reference advises a buffer between "
        f"sessions and a written record of decisions, keyed to a stable identifier.",
        f"Veteran practitioners in {topic} often recommend defaulting to the most "
        f"resilient option when constraints are unclear, then narrowing later.",
        f"A 2018 retrospective on {topic} found a 12% reduction in escalations "
        f"when the team published a constraint sheet ahead of the work block.",
        f"Tools commonly used for {topic} include shared calendars, lightweight "
        f"checklists, and a single accountable owner per item.",
        f"In urban {topic} contexts, transit overhead is usually the dominant "
        f"variable; in rural contexts, the dominant variable shifts to logistics.",
        f"A common failure mode in {topic} is over-stuffing the schedule, leaving "
        f"no slack for inevitable late starts and information surprises.",
        f"Documentation conventions for {topic} tend to favor agenda-with-owners "
        f"over freeform notes, because owners create accountability.",
        f"Cross-functional {topic} discussions can stall when one stakeholder's "
        f"data depends on another's not-yet-shared estimate; sequence matters.",
        f"For {topic}, the most underrated step is the final summary email — "
        f"a written record of decisions that takes <5 minutes but pays for itself.",
        f"Quarterly reviews of {topic} typically include throughput, defect rate, "
        f"reopen rate, and stakeholder satisfaction.",
    ]
    rng.shuffle(sentences)
    return " ".join(sentences[:5]) + f" (filler-{seed})"


def _build_chunks(
    topic: str, real_chunks: list[str], seed_base: int, target_tokens: int = 30_000
) -> list[str]:
    chunks = list(real_chunks)
    i = 0
    while sum(n_tokens(c) for c in chunks) < target_tokens:
        chunks.append(_filler_paragraph(topic, seed_base + i))
        i += 1
        if i > 400:
            break
    rng = random.Random(seed_base)
    rng.shuffle(chunks)
    return chunks


@dataclass
class Case:
    case_id: str
    task: str
    chunks: list[str]
    gold_facts: list[dict[str, Any]]


def build_cases() -> list[Case]:
    return [
        Case(
            "tokyo_trip",
            CASE1_TASK,
            _build_chunks("travel-planning", CASE1_REAL, 1001, 28_000),
            CASE1_GOLD,
        ),
        Case(
            "leadership_sync",
            CASE2_TASK,
            _build_chunks("meeting-facilitation", CASE2_REAL, 2002, 28_000),
            CASE2_GOLD,
        ),
    ]


# -------------------- External memory retrieval -------------------- #


class ExternalMemory:
    _STOP = {
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "at",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "as",
        "this",
        "that",
        "what",
        "which",
        "who",
        "how",
        "do",
        "does",
        "from",
        "into",
        "we",
        "our",
        "must",
        "should",
        "case",
        "step",
        "item",
    }

    def __init__(self, chunks: list[str]):
        # Each chunk gets a stable id "ch-{i}"
        self.chunks = [(f"ch-{i}", c) for i, c in enumerate(chunks)]

    @classmethod
    def _toks(cls, s: str) -> list[str]:
        return [
            t
            for t in re.findall(r"[a-zA-Z0-9_\-]+", s.lower())
            if t not in cls._STOP and len(t) > 2
        ]

    def query(self, q: str, k: int = 4) -> list[tuple[str, str]]:
        q_terms = self._toks(q)
        if not q_terms:
            return []
        scored = []
        for cid, ch in self.chunks:
            ch_terms = self._toks(ch)
            if not ch_terms:
                continue
            overlap = sum(1 for t in q_terms if t in ch_terms)
            if overlap == 0:
                continue
            score = overlap / (len(set(ch_terms)) ** 0.5)
            scored.append((score, cid, ch))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(cid, ch) for _, cid, ch in scored[:k]]


# -------------------- Item model -------------------- #


@dataclass
class Item:
    item_id: str
    type: str  # ChatEvent | Retrieval | Assessment | Reasoning | Output
    content: str
    last_referenced_turn: int
    size_tokens: int = 0
    summary: str | None = None  # set when in Tier 3 (compressed)

    def __post_init__(self) -> None:
        if self.size_tokens == 0:
            self.size_tokens = n_tokens(self.content)


def gen_id(prefix: str, n: int) -> str:
    return f"{prefix}-{n}"


# -------------------- LLM helper -------------------- #


async def llm(system: str, user: str, max_tokens: int = 4500) -> str:
    # gpt-5-mini consumes reasoning tokens out of max_completion_tokens.
    # Bump the floor aggressively so the JSON output isn't starved.
    budget = max(max_tokens, 4500)
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=budget,
    )
    return (resp.choices[0].message.content or "").strip()


# -------------------- Citation parsing -------------------- #

CITATION_RE = re.compile(r"\[([a-zA-Z]+-\d+)\]")


def parse_citations(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(text)))


# -------------------- Priming prompts -------------------- #

PRIMING_OPERATOR = """You are a memory-augmented agent solving a multi-step task with a bounded working memory and a retrieval tool.

Your working memory has three tiers shown to you each turn:
- TIER 1 (transient retrieval buffer): the most recent retrieval's full hits. Replaced each retrieval.
- TIER 2 (active LRU): a small set of items, kept verbatim. The system tracks which of these you are CURRENTLY USING via your citations.
- TIER 3 (compressed older): one-line summaries of items you stopped citing recently. IDs preserved so you can re-promote them by citing.

CITATION CONTRACT (the central mechanism):
- When you cite an item like [r-3] or [chat-7] in your reasoning, the system records it as "I am USING this item right now" and refreshes its position in the LRU.
- Items you do NOT cite age out as new items get cited. That is intentional: not citing means "I am not currently using it."
- Retrievals deliver candidates into Tier 1. They are NOT auto-promoted to Tier 2. ONLY citing them promotes them.
- Cite what you are actually reasoning about. The structure tracks attention; do not pad citations.

OUTPUT SCHEMA (strict JSON, every turn):
{
  "reasoning": "<free prose, 50-300 tokens. natural reasoning. inline citations like [r-3] when relying on item r-3.>",
  "actions": [
    {"type": "retrieve", "cue": "<short content-bearing query>", "reason": "<why now>"},
    {"type": "assess", "re": "<item-id>", "did_advance": true_or_false, "why": "<short>"},
    {"type": "answer", "text": "<final user-facing answer; only when ready>"}
  ]
}

You may output 1-3 actions per turn. Common patterns:
- Early turns: usually one retrieve action.
- Mid turns: retrieve + assess of the prior retrieval.
- Final turn: answer (you may include reasoning + answer).

EFFORT SIGNALS visible to you each turn (rounds elapsed, new-fact rate, WM tokens) inform your decisions: keep retrieving vs. wrap up. If new-fact rate is dropping, consider answering.

Be concise. Cite real item IDs. Output ONLY the JSON object — no markdown fences, no preamble."""

PRIMING_BASELINE = """You are a memory-augmented agent solving a multi-step task with a bounded working memory and a retrieval tool.

Your working memory shows the most recent items at the top. Older items are compressed when WM exceeds a budget.

OUTPUT SCHEMA (strict JSON, every turn):
{
  "reasoning": "<free prose, 50-300 tokens. natural reasoning.>",
  "actions": [
    {"type": "retrieve", "cue": "<short content-bearing query>", "reason": "<why now>"},
    {"type": "assess", "re": "<item-id>", "did_advance": true_or_false, "why": "<short>"},
    {"type": "answer", "text": "<final user-facing answer; only when ready>"}
  ]
}

You may output 1-3 actions per turn. Be concise. Output ONLY the JSON object — no markdown fences, no preamble."""


def render_item(item: Item, mode: str = "verbatim") -> str:
    if mode == "summary":
        s = item.summary or item.content[:120]
        return f"[{item.item_id}] ({item.type}, last_ref=t{item.last_referenced_turn}): {s}"
    return f"[{item.item_id}] ({item.type}, last_ref=t{item.last_referenced_turn}):\n{item.content}"


def build_prompt_user(
    task: str,
    turn: int,
    tier1: list[Item],
    tier2: list[Item],
    tier3: list[Item],
    retrieval_log: list[dict[str, Any]],
    effort: dict[str, Any],
    last_user_input: str,
) -> str:
    parts = []
    parts.append(f"TASK:\n{task}\n")
    parts.append(f"TURN: {turn}\n")
    parts.append(
        f"EFFORT SIGNALS: rounds_elapsed={effort['rounds']}, new_fact_rate_last3={effort['new_fact_rate']:.2f}, wm_tokens={effort['wm_tokens']}, target_cap={WM_HARD_CAP}"
    )

    parts.append("\n--- TIER 1 (transient retrieval buffer; latest hits) ---")
    if tier1:
        for it in tier1:
            parts.append(render_item(it, "verbatim"))
    else:
        parts.append("(empty)")

    parts.append("\n--- TIER 2 (active LRU, verbatim) ---")
    if tier2:
        for it in tier2:
            parts.append(render_item(it, "verbatim"))
    else:
        parts.append("(empty)")

    parts.append("\n--- TIER 3 (compressed older items, ids preserved) ---")
    if tier3:
        for it in tier3:
            parts.append(render_item(it, "summary"))
    else:
        parts.append("(empty)")

    parts.append("\n--- RETRIEVAL LOG (history of cues issued) ---")
    if retrieval_log:
        for entry in retrieval_log[-6:]:
            parts.append(
                f"  t{entry['turn']}: cue={entry['cue']!r} -> {entry['n_hits']} hits ({', '.join(entry['hit_ids'])})"
            )
    else:
        parts.append("(empty)")

    parts.append(f"\n--- LAST USER INPUT ---\n{last_user_input}")
    parts.append("\nRespond with the JSON object now.")
    return "\n".join(parts)


def build_baseline_prompt(
    task: str,
    turn: int,
    wm_items: list[Item],
    retrieval_log: list[dict[str, Any]],
    effort: dict[str, Any],
    last_user_input: str,
) -> str:
    parts = []
    parts.append(f"TASK:\n{task}\n")
    parts.append(f"TURN: {turn}\n")
    parts.append(
        f"EFFORT SIGNALS: rounds_elapsed={effort['rounds']}, new_fact_rate_last3={effort['new_fact_rate']:.2f}, wm_tokens={effort['wm_tokens']}, target_cap={WM_HARD_CAP}"
    )

    parts.append(
        "\n--- WORKING MEMORY (newest first; older items may be compressed) ---"
    )
    if wm_items:
        for it in wm_items:
            mode = "summary" if it.summary else "verbatim"
            parts.append(render_item(it, mode))
    else:
        parts.append("(empty)")

    parts.append("\n--- RETRIEVAL LOG ---")
    if retrieval_log:
        for entry in retrieval_log[-6:]:
            parts.append(
                f"  t{entry['turn']}: cue={entry['cue']!r} -> {entry['n_hits']} hits"
            )
    else:
        parts.append("(empty)")

    parts.append(f"\n--- LAST USER INPUT ---\n{last_user_input}")
    parts.append("\nRespond with the JSON object now.")
    return "\n".join(parts)


# -------------------- JSON parsing -------------------- #


def parse_json_output(text: str) -> dict[str, Any] | None:
    """Strict-ish JSON parse. Strips ```json fences if present."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        # strip fences
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    # find first { ... last }
    first = t.find("{")
    last = t.rfind("}")
    if first == -1 or last == -1 or last < first:
        return None
    candidate = t[first : last + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # try a more lenient approach: replace single quotes with doubles is too risky.
        return None


# -------------------- Operator (citation-driven LRU) state -------------------- #


@dataclass
class OperatorState:
    tier1: list[Item] = field(default_factory=list)  # transient
    tier2: list[Item] = field(default_factory=list)  # active LRU verbatim, len <= 15
    tier3: list[Item] = field(default_factory=list)  # compressed, len <= 35
    all_items: dict[str, Item] = field(default_factory=dict)
    retrieval_log: list[dict[str, Any]] = field(default_factory=list)

    def register(self, item: Item) -> None:
        self.all_items[item.item_id] = item

    def cite(self, item_id: str, turn: int) -> str:
        """Promote item by citation. Returns where it lives now."""
        if item_id not in self.all_items:
            return "unknown"
        it = self.all_items[item_id]
        it.last_referenced_turn = turn
        # if in tier1, lift into tier2
        if it in self.tier1:
            self.tier1.remove(it)
            self.tier2.insert(0, it)
            return "tier1->tier2"
        # if already in tier2, just refresh order
        if it in self.tier2:
            self.tier2.remove(it)
            self.tier2.insert(0, it)
            return "tier2->tier2(refresh)"
        # if in tier3, lift back to tier2 (rehydrate verbatim)
        if it in self.tier3:
            self.tier3.remove(it)
            self.tier2.insert(0, it)
            return "tier3->tier2(rehydrate)"
        return "untracked"

    def add_chat_event(self, item: Item) -> None:
        """Insert chat-event item into Tier 2 at front (it is the latest event)."""
        self.tier2.insert(0, item)
        self.register(item)

    def replace_tier1(self, items: list[Item]) -> None:
        # Move any unciteed Tier1 contents into Tier3 if user had cited them in past, else drop.
        # But by design tier1 is transient and not citable to history (only via citing).
        # Items not cited disappear from Tier1.
        for old in self.tier1:
            # purge from tracking — they were never cited, never promoted.
            self.all_items.pop(old.item_id, None)
        self.tier1 = items
        for it in items:
            self.register(it)

    def enforce_caps(self, turn: int) -> dict[str, Any]:
        """If Tier2 over cap, demote LRU (oldest last_referenced_turn) to Tier3.
        If Tier3 over cap, drop oldest. Also enforce token cap by demotion."""
        promoted_demoted = {"demoted_to_t3": [], "dropped": []}

        # Sort tier2 by last_referenced_turn desc
        self.tier2.sort(key=lambda x: -x.last_referenced_turn)

        # Demote excess by item count
        while len(self.tier2) > TIER2_MAX_ITEMS:
            victim = self.tier2.pop()  # oldest
            self._demote_to_tier3(victim)
            promoted_demoted["demoted_to_t3"].append(victim.item_id)

        # Token-budget enforcement: if total prompt tokens (verbatim parts) > cap, demote more.
        while self._verbatim_tokens() > WM_HARD_CAP and self.tier2:
            victim = self.tier2.pop()  # oldest
            self._demote_to_tier3(victim)
            promoted_demoted["demoted_to_t3"].append(victim.item_id)

        # Tier3 cap
        while len(self.tier3) > TIER3_MAX_ITEMS:
            victim = self.tier3.pop()
            self.all_items.pop(victim.item_id, None)
            promoted_demoted["dropped"].append(victim.item_id)

        return promoted_demoted

    def _demote_to_tier3(self, item: Item) -> None:
        # generate 1-line summary if not already
        if not item.summary:
            item.summary = self._summarize_one_line(item)
        self.tier3.insert(0, item)

    @staticmethod
    def _summarize_one_line(item: Item) -> str:
        text = item.content.replace("\n", " ").strip()
        if len(text) <= 140:
            return text
        return text[:137] + "..."

    def _verbatim_tokens(self) -> int:
        # tier1 + tier2 verbatim cost
        s = "\n".join(render_item(i, "verbatim") for i in self.tier1 + self.tier2)
        s += "\n".join(render_item(i, "summary") for i in self.tier3)
        return n_tokens(s)


# -------------------- Baseline state -------------------- #


@dataclass
class BaselineState:
    wm_items: list[Item] = field(default_factory=list)  # newest first
    all_items: dict[str, Item] = field(default_factory=dict)
    retrieval_log: list[dict[str, Any]] = field(default_factory=list)

    def register(self, item: Item) -> None:
        self.all_items[item.item_id] = item

    def push_top(self, item: Item) -> None:
        self.wm_items.insert(0, item)
        self.register(item)

    def enforce_cap(self) -> dict[str, Any]:
        compressed = []
        # While total tokens > cap, compress the oldest (last) item if it's still verbatim.
        # If already compressed, drop it.
        while self._tokens() > WM_HARD_CAP and self.wm_items:
            victim = self.wm_items[-1]
            if victim.summary is None:
                victim.summary = OperatorState._summarize_one_line(victim)
                compressed.append(victim.item_id)
            else:
                # already compressed; drop
                self.wm_items.pop()
        return {"compressed": compressed}

    def _tokens(self) -> int:
        s = "\n".join(
            render_item(i, "summary" if i.summary else "verbatim")
            for i in self.wm_items
        )
        return n_tokens(s)


# -------------------- Run loop -------------------- #


async def run_operator(case: Case) -> dict[str, Any]:
    state = OperatorState()
    em = ExternalMemory(case.chunks)

    item_counters = {"r": 0, "ret": 0, "as": 0, "an": 0, "out": 0, "chat": 0}
    trace: list[dict[str, Any]] = []
    facts_seen_per_turn: list[int] = []

    # seed: store task itself as a chat event so model can cite it
    chat0 = Item(
        item_id="chat-0",
        type="ChatEvent",
        content=f"USER: {case.task}",
        last_referenced_turn=0,
    )
    state.add_chat_event(chat0)
    item_counters["chat"] = 1

    parse_failures = 0
    citations_per_turn: list[int] = []
    answers: list[str] = []

    last_user_input = case.task

    for turn in range(1, MAX_TURNS + 1):
        # Compute effort signals
        recent_facts = sum(facts_seen_per_turn[-3:])
        effort = {
            "rounds": turn - 1,
            "new_fact_rate": recent_facts / max(1, min(3, turn - 1))
            if turn > 1
            else 0.0,
            "wm_tokens": state._verbatim_tokens(),
        }

        prompt = build_prompt_user(
            case.task,
            turn,
            state.tier1,
            state.tier2,
            state.tier3,
            state.retrieval_log,
            effort,
            last_user_input,
        )

        # Call LLM
        out = await llm(PRIMING_OPERATOR, prompt, max_tokens=2500)
        parsed = parse_json_output(out)

        turn_log: dict[str, Any] = {
            "turn": turn,
            "wm_tokens_before": effort["wm_tokens"],
            "tier1_ids": [i.item_id for i in state.tier1],
            "tier2_ids": [(i.item_id, i.last_referenced_turn) for i in state.tier2],
            "tier3_ids": [(i.item_id, i.last_referenced_turn) for i in state.tier3],
        }

        if parsed is None:
            parse_failures += 1
            turn_log["parse_failure"] = True
            turn_log["raw_output"] = out[:500]
            trace.append(turn_log)
            facts_seen_per_turn.append(0)
            citations_per_turn.append(0)
            # store reasoning as a degraded item so we don't lose the turn
            r_id = gen_id("r", item_counters["r"])
            item_counters["r"] += 1
            r_item = Item(r_id, "Reasoning", out[:500] or "(no output)", turn)
            state.tier2.insert(0, r_item)
            state.register(r_item)
            state.enforce_caps(turn)
            continue

        reasoning_text = parsed.get("reasoning", "")
        actions = parsed.get("actions", []) or []

        # Add reasoning as item
        r_id = gen_id("r", item_counters["r"])
        item_counters["r"] += 1
        r_item = Item(r_id, "Reasoning", reasoning_text, turn)
        state.tier2.insert(0, r_item)
        state.register(r_item)

        # Parse citations
        cites = parse_citations(reasoning_text)
        # also check assess.re
        for a in actions:
            if isinstance(a, dict) and a.get("type") == "assess":
                ref = a.get("re")
                if isinstance(ref, str):
                    cites.extend(parse_citations(f"[{ref}]"))
        # dedup
        cites = list(dict.fromkeys(cites))
        citations_per_turn.append(len(cites))

        cite_results = []
        for cid in cites:
            res = state.cite(cid, turn)
            cite_results.append((cid, res))

        # Execute actions
        new_facts = 0
        action_log = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            atype = a.get("type", "")
            if atype == "retrieve":
                cue = a.get("cue", "")
                reason = a.get("reason", "")
                hits = em.query(cue, k=4)
                # build retrieval items
                hit_items = []
                real_set = {r.strip() for r in case_real_for(case)}
                for cid, ctext in hits:
                    rid = gen_id("ret", item_counters["ret"])
                    item_counters["ret"] += 1
                    hit_item = Item(rid, "Retrieval", f"hit_id={cid}\n{ctext}", turn)
                    hit_items.append(hit_item)
                    if ctext.strip() in real_set:
                        new_facts += 1
                state.replace_tier1(hit_items)
                state.retrieval_log.append(
                    {
                        "turn": turn,
                        "cue": cue,
                        "reason": reason,
                        "n_hits": len(hits),
                        "hit_ids": [c for c, _ in hits],
                    }
                )
                action_log.append({"type": "retrieve", "cue": cue, "n_hits": len(hits)})
            elif atype == "assess":
                ref = a.get("re", "")
                aid = gen_id("as", item_counters["as"])
                item_counters["as"] += 1
                a_item = Item(
                    aid,
                    "Assessment",
                    f"re={ref} did_advance={a.get('did_advance')} why={a.get('why', '')}",
                    turn,
                )
                state.tier2.insert(0, a_item)
                state.register(a_item)
                action_log.append({"type": "assess", "re": ref})
            elif atype == "answer":
                text = a.get("text", "")
                aid = gen_id("an", item_counters["an"])
                item_counters["an"] += 1
                a_item = Item(aid, "Output", text, turn)
                state.tier2.insert(0, a_item)
                state.register(a_item)
                answers.append(text)
                action_log.append({"type": "answer", "len": len(text)})

        facts_seen_per_turn.append(new_facts)

        # Enforce caps
        cap_log = state.enforce_caps(turn)

        turn_log.update(
            {
                "citations": cites,
                "cite_results": cite_results,
                "actions": action_log,
                "reasoning_excerpt": reasoning_text[:300],
                "wm_tokens_after": state._verbatim_tokens(),
                "demoted_to_t3": cap_log["demoted_to_t3"],
                "dropped": cap_log["dropped"],
                "new_facts_seen": new_facts,
            }
        )
        trace.append(turn_log)

        # Stop if model produced an answer
        if any(a.get("type") == "answer" for a in actions):
            break

    # Score: ask the same model gold questions using the final WM as context
    final_score = await score_model(case, state.tier2 + state.tier3, answers)

    return {
        "case_id": case.case_id,
        "variant": "operator",
        "turns_used": len(trace),
        "parse_failures": parse_failures,
        "parse_rate": (len(trace) - parse_failures) / max(1, len(trace)),
        "citations_per_turn": citations_per_turn,
        "avg_citations_per_turn": sum(citations_per_turn)
        / max(1, len(citations_per_turn)),
        "max_wm_tokens": max(
            (t.get("wm_tokens_after", t.get("wm_tokens_before", 0)) for t in trace),
            default=0,
        ),
        "score": final_score,
        "answers": answers,
        "trace": trace,
    }


def case_real_for(case: Case) -> list[str]:
    if case.case_id == "tokyo_trip":
        return CASE1_REAL
    return CASE2_REAL


async def run_baseline(case: Case) -> dict[str, Any]:
    state = BaselineState()
    em = ExternalMemory(case.chunks)

    item_counters = {"r": 0, "ret": 0, "as": 0, "an": 0, "chat": 0}
    trace: list[dict[str, Any]] = []
    facts_seen_per_turn: list[int] = []
    parse_failures = 0
    citations_per_turn: list[int] = []
    answers: list[str] = []

    chat0 = Item("chat-0", "ChatEvent", f"USER: {case.task}", last_referenced_turn=0)
    state.push_top(chat0)
    item_counters["chat"] = 1

    last_user_input = case.task
    for turn in range(1, MAX_TURNS + 1):
        recent_facts = sum(facts_seen_per_turn[-3:])
        effort = {
            "rounds": turn - 1,
            "new_fact_rate": recent_facts / max(1, min(3, turn - 1))
            if turn > 1
            else 0.0,
            "wm_tokens": state._tokens(),
        }
        prompt = build_baseline_prompt(
            case.task,
            turn,
            state.wm_items,
            state.retrieval_log,
            effort,
            last_user_input,
        )
        out = await llm(PRIMING_BASELINE, prompt, max_tokens=2500)
        parsed = parse_json_output(out)

        turn_log: dict[str, Any] = {
            "turn": turn,
            "wm_tokens_before": effort["wm_tokens"],
            "wm_ids": [
                (i.item_id, "summary" if i.summary else "verbatim")
                for i in state.wm_items
            ],
        }
        if parsed is None:
            parse_failures += 1
            turn_log["parse_failure"] = True
            turn_log["raw_output"] = out[:500]
            trace.append(turn_log)
            facts_seen_per_turn.append(0)
            citations_per_turn.append(0)
            continue

        reasoning_text = parsed.get("reasoning", "")
        actions = parsed.get("actions", []) or []
        r_id = gen_id("r", item_counters["r"])
        item_counters["r"] += 1
        r_item = Item(r_id, "Reasoning", reasoning_text, turn)
        state.push_top(r_item)
        cites = parse_citations(reasoning_text)
        citations_per_turn.append(len(cites))

        new_facts = 0
        action_log = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            atype = a.get("type", "")
            if atype == "retrieve":
                cue = a.get("cue", "")
                hits = em.query(cue, k=4)
                # Baseline pushes ALL hits onto top of WM as flat items
                for cid, ctext in hits:
                    rid = gen_id("ret", item_counters["ret"])
                    item_counters["ret"] += 1
                    hit_item = Item(rid, "Retrieval", f"hit_id={cid}\n{ctext}", turn)
                    state.push_top(hit_item)
                    if ctext.strip() in [r.strip() for r in case_real_for(case)]:
                        new_facts += 1
                state.retrieval_log.append(
                    {
                        "turn": turn,
                        "cue": cue,
                        "n_hits": len(hits),
                        "hit_ids": [c for c, _ in hits],
                    }
                )
                action_log.append({"type": "retrieve", "cue": cue, "n_hits": len(hits)})
            elif atype == "assess":
                aid = gen_id("as", item_counters["as"])
                item_counters["as"] += 1
                a_item = Item(
                    aid,
                    "Assessment",
                    f"re={a.get('re')} did_advance={a.get('did_advance')} why={a.get('why', '')}",
                    turn,
                )
                state.push_top(a_item)
                action_log.append({"type": "assess", "re": a.get("re")})
            elif atype == "answer":
                text = a.get("text", "")
                aid = gen_id("an", item_counters["an"])
                item_counters["an"] += 1
                a_item = Item(aid, "Output", text, turn)
                state.push_top(a_item)
                answers.append(text)
                action_log.append({"type": "answer", "len": len(text)})

        facts_seen_per_turn.append(new_facts)
        compress_log = state.enforce_cap()

        turn_log.update(
            {
                "citations": cites,
                "actions": action_log,
                "reasoning_excerpt": reasoning_text[:300],
                "wm_tokens_after": state._tokens(),
                "compressed": compress_log["compressed"],
                "new_facts_seen": new_facts,
            }
        )
        trace.append(turn_log)

        if any(a.get("type") == "answer" for a in actions):
            break

    # Score: feed the final WM (verbatim+summary) into a fixed scorer
    final_score = await score_model(case, state.wm_items, answers)

    return {
        "case_id": case.case_id,
        "variant": "baseline",
        "turns_used": len(trace),
        "parse_failures": parse_failures,
        "parse_rate": (len(trace) - parse_failures) / max(1, len(trace)),
        "citations_per_turn": citations_per_turn,
        "avg_citations_per_turn": sum(citations_per_turn)
        / max(1, len(citations_per_turn)),
        "max_wm_tokens": max(
            (t.get("wm_tokens_after", t.get("wm_tokens_before", 0)) for t in trace),
            default=0,
        ),
        "score": final_score,
        "answers": answers,
        "trace": trace,
    }


# -------------------- Scoring -------------------- #


async def score_model(
    case: Case, final_wm_items: list[Item], answers: list[str]
) -> dict[str, Any]:
    """Hand the final WM (verbatim+summary) plus produced answers to a scorer that
    asks the gold questions. Score by substring-match on answer keys, case-insensitive."""

    wm_text = "\n".join(
        render_item(i, "summary" if i.summary else "verbatim") for i in final_wm_items
    )
    if answers:
        wm_text += "\n\n--- AGENT FINAL ANSWERS ---\n" + "\n\n".join(answers)

    qs = "\n".join(f"{i + 1}. {gf['q']}" for i, gf in enumerate(case.gold_facts))
    sys = (
        "You are answering recall questions about a task using ONLY the working memory "
        "and final answers shown. Output one numbered answer per question. If WM does not "
        "contain enough, say UNKNOWN for that question. Be concise."
    )
    user = f"WORKING MEMORY:\n{wm_text}\n\nQUESTIONS:\n{qs}\n\nAnswer each question:"
    out = await llm(sys, user, max_tokens=1500)

    score = 0
    per_q = []
    for i, gf in enumerate(case.gold_facts):
        prefix = f"{i + 1}."
        line = ""
        for ln in out.splitlines():
            ln_s = ln.strip()
            if ln_s.startswith(prefix) or ln_s.startswith(f"{i + 1})"):
                line = ln_s.lower()
                break
        if not line:
            line = out.lower()
        hit = any(k.lower() in line for k in gf["keys"])
        if hit:
            score += 1
        per_q.append({"qid": gf["qid"], "q": gf["q"], "hit": hit})
    return {
        "score": score,
        "n": len(case.gold_facts),
        "per_q": per_q,
        "scorer_output": out,
    }


# -------------------- Main -------------------- #


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, Item):
        return asdict(obj)
    return obj


async def main() -> None:
    import sys

    only = None
    if len(sys.argv) > 1:
        only = sys.argv[1]  # "baseline" | "operator"
    cases = build_cases()
    summary: list[dict[str, Any]] = []
    full_traces: list[dict[str, Any]] = []

    variants = [
        ("baseline", run_baseline),
        ("operator", run_operator),
    ]
    if only:
        variants = [(n, r) for n, r in variants if n == only]

    for case in cases:
        for variant_name, runner in variants:
            print(
                f"=== running case={case.case_id} variant={variant_name} ===",
                flush=True,
            )
            try:
                res = await runner(case)
            except Exception as e:
                print(f"  ERROR {e!r}", flush=True)
                summary.append(
                    {
                        "case_id": case.case_id,
                        "variant": variant_name,
                        "error": repr(e),
                    }
                )
                continue
            summary.append(
                {
                    "case_id": res["case_id"],
                    "variant": res["variant"],
                    "turns_used": res["turns_used"],
                    "parse_rate": res["parse_rate"],
                    "parse_failures": res["parse_failures"],
                    "avg_citations_per_turn": res["avg_citations_per_turn"],
                    "max_wm_tokens": res["max_wm_tokens"],
                    "score": res["score"]["score"],
                    "n": res["score"]["n"],
                    "per_q": res["score"]["per_q"],
                }
            )
            full_traces.append(res)

    # write outputs
    suffix = f".{only}" if only else ""
    (THIS_DIR / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    (THIS_DIR / f"trace{suffix}.json").write_text(
        json.dumps(to_serializable(full_traces), indent=2, default=str)
    )

    # console summary
    print("\n=== SUMMARY ===")
    print(
        f"{'case':24s} {'variant':10s} {'turns':>6s} {'parse_rate':>10s} {'avg_cites':>10s} {'max_wm':>8s} {'score':>8s}"
    )
    for s in summary:
        if "error" in s:
            print(f"{s['case_id']:24s} {s['variant']:10s} ERROR {s['error']}")
            continue
        print(
            f"{s['case_id']:24s} {s['variant']:10s} {s['turns_used']:>6d} "
            f"{s['parse_rate']:>10.2f} {s['avg_citations_per_turn']:>10.2f} "
            f"{s['max_wm_tokens']:>8d} {s['score']:>2d}/{s['n']:<2d}"
        )


if __name__ == "__main__":
    asyncio.run(main())
