"""Cross-task procedural-memory metacognitive operator.

ARCHITECTURE
- Bounded working memory (WM) per step: hard cap ~10k tokens. WM is compacted /
  evicted between steps.
- External per-task chat memory: 30-100k tokens of synthetic chat events,
  retrievable on demand by top-K embedding (here: cheap TF-IDF cosine, since the
  question is memory-INTERACTION, not embed-quality). Full chat is NEVER
  passed in the prompt.
- External CASE BASE: cross-task store of small (~200 tok) procedure cards.
  Retrievable by LLM-judged structural-pattern similarity. Persists across
  tasks (operator variant only).

Per task:
  Phase A. (operator only) Probe case base for similar past tasks → seed WM
           with adapted lessons.
  Phase B. Plan: produce a numbered plan. WM holds: task prompt + retrieved
           cards + on-demand chat-memory snippets (top-K). Strict 10k cap.
  Phase C. Execute: walk plan steps, fetch additional chat snippets per step
           from external memory. Old WM compacts.
  Phase D. (operator only) Emit a procedure card → append to case base.

5 synthetic tasks share STRUCTURAL pattern (multi-stakeholder coordination
under uncertainty) but differ in domain. Lessons are PRINCIPLE-level.

Run:
    uv run python evaluation/associative_recall/metacog/procedural_memory/main.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL = "gpt-5-mini"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# Token bookkeeping (rough char/4 heuristic — sufficient for budget tracking)
# ---------------------------------------------------------------------------

WM_CAP_TOKENS = 10_000  # POC; production ceiling 50k per the harness spec.
CARD_BUDGET_TOKENS = 50_000
CARD_TARGET_TOKENS = 220


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


# ---------------------------------------------------------------------------
# Synthetic task suite — 5 tasks, shared structural pattern, different domains.
# Each task has a chat-memory generator that produces 30-100k tokens of
# realistic-ish chat from multiple speakers, with the principle-level
# constraints embedded as needles.
#
# Principle pitfalls (P1..P5) used to score every task:
#   P1: probe stakeholders for non-obvious constraints
#   P2: avoid assuming uniform expertise / familiarity
#   P3: define a fallback / contingency for the most likely failure mode
#   P4: pick a communication channel/format calibrated to the audience
#   P5: name an upstream→downstream coordination handoff
# ---------------------------------------------------------------------------


@dataclass
class ChatEvent:
    eid: str
    speaker: str
    text: str
    is_needle: bool = False
    needle_pid: str | None = None  # which pitfall this evidence supports

    def render(self) -> str:
        return f"[{self.eid}] {self.speaker}: {self.text}"


@dataclass
class Task:
    id: str
    task_type: str
    prompt: str
    chat: list[ChatEvent]
    pitfall_checklist: dict[str, str]


# ----- chat-memory generator -----------------------------------------------

# Filler phrases — boring, high-volume distractor chatter that pads each
# scenario to >30k tokens without leaking pitfall information.
FILLER_TEMPLATES = [
    "Quick status check, {who}: nothing urgent on my end today.",
    "{who}: weather looks fine. Coffee machine is acting up again.",
    "{who}: parked in lot B, the elevator was slow.",
    "{who}: I was on PTO last week, catching up on emails.",
    "{who}: did anyone see the game last night? Wild ending.",
    "{who}: my kid had a recital, sorry for late reply.",
    "{who}: printer on floor 3 jammed again, IT ticket open.",
    "{who}: meeting room 4 has a flickering light, FYI.",
    "{who}: lunch options nearby are slim today, food trucks elsewhere.",
    "{who}: dog walker rescheduled, slight delay this morning.",
    "{who}: train was 12 minutes late, normal Tuesday.",
    "{who}: small typo in last memo, fixed.",
    "{who}: cleaning crew rearranged the supply closet again.",
    "{who}: my mic was muted for the first 5 min, sorry.",
    "{who}: phone battery low, will switch to laptop.",
    "{who}: gym was crowded, normal week.",
    "{who}: ordered new whiteboard markers, ETA Friday.",
    "{who}: budget code for travel reimbursement is the usual one.",
    "{who}: the front-door badge reader was glitchy this morning.",
    "{who}: nothing to flag from yesterday's standup.",
]


def _filler_chat(
    speakers: list[str], n_events: int, eid_prefix: str, rng: random.Random
) -> list[ChatEvent]:
    out: list[ChatEvent] = []
    for i in range(n_events):
        who = rng.choice(speakers)
        text = rng.choice(FILLER_TEMPLATES).format(who=who)
        out.append(ChatEvent(eid=f"{eid_prefix}{i:04d}", speaker=who, text=text))
    return out


def _build_task_chat(
    task_id: str,
    speakers: list[str],
    needles: list[tuple[str, str, str]],  # (pid, speaker, text)
    rng: random.Random,
    target_filler: int = 2400,
) -> list[ChatEvent]:
    """Mix needles into a sea of filler. Needles are interleaved at random
    positions so the agent must retrieve, not scan.

    target_filler ~2400 events × ~13 tok/event ≈ 30-32k tokens of external
    chat memory per task, well over the 10k WM cap.
    """
    filler = _filler_chat(speakers, target_filler, eid_prefix=f"{task_id}-f", rng=rng)
    needle_events = [
        ChatEvent(
            eid=f"{task_id}-n{i:02d}",
            speaker=sp,
            text=txt,
            is_needle=True,
            needle_pid=pid,
        )
        for i, (pid, sp, txt) in enumerate(needles)
    ]
    # Interleave needles at random positions in filler.
    chat = list(filler)
    for ne in needle_events:
        pos = rng.randrange(len(chat) + 1)
        chat.insert(pos, ne)
    # Reassign event ids in order so they stay monotonic.
    for i, ev in enumerate(chat):
        ev.eid = f"{task_id}-{i:05d}"
    return chat


def build_tasks(seed: int = 7) -> list[Task]:
    rng = random.Random(seed)

    tasks: list[Task] = []

    # Task 1: banquet
    tasks.append(
        Task(
            id="banquet",
            task_type="small-group-event-coordination",
            prompt=(
                "Plan a 30-person celebratory banquet for a community soccer team next "
                "Saturday at 7pm at a rented hall. Output a numbered plan."
            ),
            chat=_build_task_chat(
                "banq",
                speakers=["alex", "maria", "coach", "treasurer", "venue_mgr"],
                needles=[
                    (
                        "P1",
                        "maria",
                        "FYI three players have nut allergies and one keeps strict halal — please confirm with families before menu locks.",
                    ),
                    (
                        "P2",
                        "coach",
                        "Half the families have never been to this hall; veterans know the layout but rookies will need a map.",
                    ),
                    (
                        "P3",
                        "venue_mgr",
                        "Heads up: our backup caterer dropped out last month; if our main one cancels we are exposed.",
                    ),
                    (
                        "P4",
                        "treasurer",
                        "Email RSVPs are slow; the parents respond reliably to the WhatsApp group, not email.",
                    ),
                    (
                        "P5",
                        "alex",
                        "Setup at 5pm is on the venue staff but cleanup by 11pm is on us — we need a captain for that handoff.",
                    ),
                ],
                rng=rng,
            ),
            pitfall_checklist={
                "P1": "Probes guests for dietary restrictions / allergies / cultural needs",
                "P2": "Notes mixed familiarity with the venue across families",
                "P3": "Has a fallback for caterer/vendor cancellation",
                "P4": "Picks a channel that the audience actually responds to",
                "P5": "Names who handles cleanup / venue handoff",
            },
        )
    )

    # Task 2: school field trip
    tasks.append(
        Task(
            id="field_trip",
            task_type="small-group-event-coordination",
            prompt=(
                "Plan a one-day field trip for 24 fifth-graders to a regional natural "
                "history museum. Bus departs school 8am. Output a numbered plan."
            ),
            chat=_build_task_chat(
                "ftrip",
                speakers=["ms_lin", "principal", "nurse", "parent_rep", "bus_co"],
                needles=[
                    (
                        "P1",
                        "nurse",
                        "Two students have severe peanut allergies (epi-pens in office), one has a seizure protocol — chaperones MUST be briefed.",
                    ),
                    (
                        "P2",
                        "ms_lin",
                        "Reading levels in this class span grades 2 to 7; museum's standard tour script will lose half the kids.",
                    ),
                    (
                        "P3",
                        "bus_co",
                        "Last month our backup bus was 90 min late when the primary broke down. Always reconfirm 24h prior.",
                    ),
                    (
                        "P4",
                        "parent_rep",
                        "Paper permission slips never come back. Parents ONLY respond to ClassDojo notifications, not email.",
                    ),
                    (
                        "P5",
                        "principal",
                        "Need a clear handoff between teachers and museum docent — last year the docent thought we were 30 min later than we were.",
                    ),
                ],
                rng=rng,
            ),
            pitfall_checklist={
                "P1": "Probes for medical / allergy / behavioral student needs",
                "P2": "Accounts for varying reading / attention levels among kids",
                "P3": "Fallback for transport delay or lost child",
                "P4": "Permission-slip + parent comms channel chosen explicitly",
                "P5": "Chaperone roles + museum-staff handoff named",
            },
        )
    )

    # Task 3: hospital ward shift handoff
    tasks.append(
        Task(
            id="ward_handoff",
            task_type="professional-shift-handoff-coordination",
            prompt=(
                "Plan a structured shift handoff for a 12-bed hospital medical ward at "
                "7pm change-of-shift. Outgoing nurses brief incoming nurses. "
                "Output a numbered plan."
            ),
            chat=_build_task_chat(
                "ward",
                speakers=["dr_hall", "nurse_kim", "charge", "pharmacy", "social"],
                needles=[
                    (
                        "P1",
                        "nurse_kim",
                        "Bed 4 is contact-precaution MRSA, bed 7 is DNR (family confirmed today), bed 9 needs an interpreter for any consent — incoming staff WILL miss these without explicit handoff.",
                    ),
                    (
                        "P2",
                        "charge",
                        "Two of tonight's incoming nurses are floats from cardiology — assume zero familiarity with our specific patients.",
                    ),
                    (
                        "P3",
                        "dr_hall",
                        "Bed 11 has been borderline septic all afternoon; if they crash mid-handoff, ICU bridge plan must be predefined.",
                    ),
                    (
                        "P4",
                        "charge",
                        "Our hallway free-form verbal handoffs miss things constantly; please use a structured SBAR template tonight.",
                    ),
                    (
                        "P5",
                        "pharmacy",
                        "Two patients need meds redosed at 8pm — pharmacy needs the verbal handoff from outgoing nurse, not just a chart note.",
                    ),
                ],
                rng=rng,
            ),
            pitfall_checklist={
                "P1": "Probes for non-obvious patient constraints (allergies, isolation, code status, language)",
                "P2": "Doesn't assume incoming nurses know each patient",
                "P3": "Fallback for unstable patient mid-handoff",
                "P4": "Uses a structured handoff format (e.g., SBAR)",
                "P5": "Pharmacy / physician / family contact handoff named",
            },
        )
    )

    # Task 4: software service migration
    tasks.append(
        Task(
            id="migration_kickoff",
            task_type="cross-team-technical-rollout",
            prompt=(
                "Plan the kickoff meeting and first-week rollout for migrating an "
                "internal billing service from a legacy monolith to a new microservice. "
                "Six engineering teams depend on it. Output a numbered plan."
            ),
            chat=_build_task_chat(
                "migr",
                speakers=["tech_lead", "sre", "pm", "compliance", "team_a", "team_b"],
                needles=[
                    (
                        "P1",
                        "compliance",
                        "Two of the six teams have SOX-relevant flows; their integrations CANNOT lose audit trail data even for 5 minutes — surface this before kickoff.",
                    ),
                    (
                        "P2",
                        "tech_lead",
                        "Team B has used the new service in staging for months; teams D and F have never seen it. A single kickoff slide deck won't land for both groups.",
                    ),
                    (
                        "P3",
                        "sre",
                        "If the cutover fails, we need a documented dual-run / rollback path. Last migration in 2023 had no rollback and we lost an afternoon.",
                    ),
                    (
                        "P4",
                        "pm",
                        "Engineers ignore email blasts; they respond to Slack channel pings and a one-page doc. Press-releases land, decks don't.",
                    ),
                    (
                        "P5",
                        "sre",
                        "On-call rotation handoff between platform team and downstream service owners must be explicit — last time PagerDuty routing was ambiguous for 3 hours.",
                    ),
                ],
                rng=rng,
            ),
            pitfall_checklist={
                "P1": "Probes downstream teams for hidden integrations / SLAs / compliance",
                "P2": "Doesn't assume uniform familiarity with the new system",
                "P3": "Has a rollback / dual-run / canary plan",
                "P4": "Picks comms calibrated to engineers (Slack/doc vs deck)",
                "P5": "On-call / incident-response handoff named",
            },
        )
    )

    # Task 5: museum exhibit opening (largest audience)
    tasks.append(
        Task(
            id="exhibit_opening",
            task_type="public-event-coordination",
            prompt=(
                "Plan the opening night of a new interactive science exhibit at a city "
                "museum. Expected: ~200 mixed-age public attendees, press, donors. "
                "Output a numbered plan."
            ),
            chat=_build_task_chat(
                "exhib",
                speakers=["curator", "ops", "pr", "dev_office", "security"],
                needles=[
                    (
                        "P1",
                        "ops",
                        "We have wheelchair guests confirmed, two donors who only speak Mandarin, and a press contingent that needs early access; surface ALL of these before signage finalizes.",
                    ),
                    (
                        "P2",
                        "pr",
                        "The same exhibit script that wows kids is going to bore donors and confuse press. Prepare TIERED material — kid handout, donor brief, press kit — not one master deck.",
                    ),
                    (
                        "P3",
                        "curator",
                        "Two of the interactive stations had touchscreen failures during dress rehearsal. If a station goes down opening night, we need a docent fallback script.",
                    ),
                    (
                        "P4",
                        "dev_office",
                        "Donors expect a personal printed brief at their seats. Press want digital embargoed materials 24h before. Public just need clear floor signage. Three different channels.",
                    ),
                    (
                        "P5",
                        "security",
                        "Crowd flow handoff between front-door security, gallery docents, and curator-led tours must be choreographed; last opening had a 20-min dead-zone where nobody owned the second-floor crowd.",
                    ),
                ],
                rng=rng,
                target_filler=3200,  # bigger event → more chatter
            ),
            pitfall_checklist={
                "P1": "Probes for accessibility / language / safety constraints",
                "P2": "Plans for very mixed expertise (kids, donors, press)",
                "P3": "Fallback for technical exhibit failure / over-attendance",
                "P4": "Differentiates press kit / donor brief / public signage",
                "P5": "Security / staff / curator handoff during run-of-show",
            },
        )
    )

    return tasks


# ---------------------------------------------------------------------------
# External per-task chat memory: cheap TF-IDF retrieval (no embedding API
# call needed; the architecture point is "queryable external memory", not
# embedding quality).
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(s)]


class ChatMemory:
    """External, queryable per-task chat memory. NOT passed in prompt."""

    def __init__(self, events: list[ChatEvent]):
        self.events = events
        # Build per-doc term counts and idf.
        self._tf: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for ev in events:
            toks = _tokenize(ev.text)
            self._tf.append(Counter(toks))
            for t in set(toks):
                df[t] += 1
        N = max(1, len(events))
        self._idf: dict[str, float] = {
            t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()
        }
        self._norms: list[float] = []
        for tf in self._tf:
            s = 0.0
            for t, c in tf.items():
                w = c * self._idf.get(t, 0.0)
                s += w * w
            self._norms.append(math.sqrt(s) or 1.0)

    def __len__(self) -> int:
        return len(self.events)

    def total_tokens(self) -> int:
        return sum(approx_tokens(ev.text) for ev in self.events)

    def query(self, q: str, k: int = 6) -> list[tuple[ChatEvent, float]]:
        q_tf = Counter(_tokenize(q))
        if not q_tf:
            return []
        # Weighted query vector
        q_w = {t: c * self._idf.get(t, 0.0) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_w.values())) or 1.0
        scored: list[tuple[ChatEvent, float]] = []
        for i, tf in enumerate(self._tf):
            dot = 0.0
            for t, qv in q_w.items():
                if t in tf:
                    dot += qv * tf[t] * self._idf.get(t, 0.0)
            if dot <= 0:
                continue
            sim = dot / (q_norm * self._norms[i])
            scored.append((self.events[i], sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------------------------
# Procedure-card case base (cross-task).
# ---------------------------------------------------------------------------


@dataclass
class ProcedureCard:
    task_id: str
    task_type: str
    text: str  # JSON-as-string

    def approx_tokens(self) -> int:
        return approx_tokens(self.text)


class CaseBase:
    def __init__(self, budget_tokens: int = CARD_BUDGET_TOKENS):
        self.cards: list[ProcedureCard] = []
        self.budget = budget_tokens

    def total_tokens(self) -> int:
        return sum(c.approx_tokens() for c in self.cards)

    def add(self, card: ProcedureCard) -> None:
        self.cards.append(card)
        # Drop oldest if over budget (compaction stub).
        while self.total_tokens() > self.budget and len(self.cards) > 1:
            self.cards.pop(0)

    def __len__(self) -> int:
        return len(self.cards)


# ---------------------------------------------------------------------------
# Bounded WM with compaction.
# ---------------------------------------------------------------------------


@dataclass
class WMSlot:
    label: str  # e.g. "TASK", "CARD:ward_handoff", "RETR:step3"
    text: str
    pinned: bool = False  # pinned slots survive compaction


@dataclass
class WorkingMemory:
    cap_tokens: int = WM_CAP_TOKENS
    slots: list[WMSlot] = field(default_factory=list)
    compactions: int = 0

    def total_tokens(self) -> int:
        return sum(approx_tokens(s.text) for s in self.slots)

    def render(self) -> str:
        return "\n\n".join(f"## {s.label}\n{s.text}" for s in self.slots)

    def add(self, slot: WMSlot) -> None:
        self.slots.append(slot)
        self._maybe_compact()

    def replace(self, label: str, text: str, pinned: bool = False) -> None:
        for i, s in enumerate(self.slots):
            if s.label == label:
                self.slots[i] = WMSlot(label, text, pinned)
                self._maybe_compact()
                return
        self.add(WMSlot(label, text, pinned))

    def _maybe_compact(self) -> None:
        while self.total_tokens() > self.cap_tokens and len(self.slots) > 1:
            # Evict oldest non-pinned slot. If only pinned slots remain,
            # truncate the largest pinned slot.
            for i, s in enumerate(self.slots):
                if not s.pinned:
                    self.slots.pop(i)
                    self.compactions += 1
                    break
            else:
                # Truncate the largest pinned slot to half.
                idx = max(range(len(self.slots)), key=lambda i: len(self.slots[i].text))
                t = self.slots[idx].text
                self.slots[idx] = WMSlot(
                    self.slots[idx].label, t[: len(t) // 2], pinned=True
                )
                self.compactions += 1


# ---------------------------------------------------------------------------
# Prompts — principled, no domain-specific recipes.
# ---------------------------------------------------------------------------

WHY_PROCEDURAL = (
    "WHY procedural memory matters: most agents start each task fresh and "
    "repeat the same mistakes. Humans accumulate cross-task experience as "
    'principle-level lessons ("for tasks of shape X the typical '
    'decomposition is Y; common pitfalls are Z"). Without this, no skill '
    "growth: the 1000th task is solved no better than the first. Bounded "
    "card size keeps the case base manageable AND forces principle-level "
    "compression (recipes rot, principles transfer)."
)

CARD_GUIDE = (
    "HOW to write a procedure card. Keep it ≤ ~200 tokens. Fields:\n"
    "  task_type — short abstract label (e.g., 'multi-stakeholder-coordination-under-uncertainty')\n"
    "  decomposition — 3-5 abstract sub-steps you used\n"
    "  what_worked — 2-3 PRINCIPLE-level bullets\n"
    "  what_failed_or_was_risky — 2-3 PRINCIPLE-level bullets\n"
    "  lessons — 2-3 adaptable rules\n"
    "Principle-level test: 'always check stakeholders' hidden constraints' is "
    "principle-level; 'always check Sam's allergies' is not. If you cite a "
    "specific name, vendor, system, or location, you are doing it wrong — "
    "abstract it. If you cannot abstract it, drop it."
)

USE_GUIDE = (
    "HOW to USE retrieved cards. (a) Identify the structural pattern shared "
    "with the past task. (b) For each lesson, locate the ADAPTATION POINT in "
    "the new task — what plays the role of 'stakeholder', 'fallback', "
    "'handoff' here? (c) Do NOT blindly copy specifics; surface domain "
    "differs. (d) If a past lesson does not apply, drop it explicitly."
)


PLAN_SYSTEM_BASELINE = f"""You are a careful planning agent solving a coordination task.

{WHY_PROCEDURAL}

You have NO access to past cases. You DO have a queryable external chat memory
that has already been probed; relevant snippets are in your working memory.

Working memory is BOUNDED (~10k tokens). Older content has been compacted.
Use only what is in WM.

Write a numbered plan (5-9 steps). Each step concrete enough to act on, but the
plan as a whole should reflect principled coordination thinking
(stakeholder constraints, fallbacks, audience-calibrated communication,
handoffs)."""


PLAN_SYSTEM_OPERATOR = f"""You are a careful planning agent solving a coordination task.

{WHY_PROCEDURAL}

You have access to a CASE BASE of procedure cards from past tasks; the most
structurally-similar ones have been retrieved and seeded into WM. You also
have a queryable external chat memory that has been probed; relevant snippets
are in WM.

{USE_GUIDE}

Working memory is BOUNDED (~10k tokens). Older content has been compacted.
Use only what is in WM.

Write a numbered plan (5-9 steps). Adapt principle-level lessons; do not copy
specifics. If a past lesson clearly does not transfer to this task's surface,
drop it."""


CARD_SYSTEM = f"""You are an agent emitting a procedure card after solving a task.

{WHY_PROCEDURAL}

{CARD_GUIDE}

Output STRICT JSON with keys: task_type, decomposition, what_worked,
what_failed_or_was_risky, lessons. Each field is a string or a short list of
strings. NO domain-specific names — abstract everything."""


SIMILARITY_SYSTEM = """You are a retrieval judge. Given a NEW task description
and a list of past procedure cards (with task_type + lessons), rank cards by
how transferable their lessons are to the new task. Match on STRUCTURAL
PATTERN (e.g., 'coordinating multiple stakeholders under uncertainty'),
NOT surface domain. Output STRICT JSON: {"ranking": [card_index, ...]} where
card_index is the integer index in the input list, most relevant first.
Include only cards whose lessons plausibly transfer; drop the rest."""


CUE_SYSTEM = """You generate retrieval cues for an external chat memory. You
will be given a task prompt (and optionally retrieved procedure-card lessons).
Emit 3-6 short retrieval queries (one per line) that, if probed against the
chat memory, are most likely to surface the constraints / hidden information
the planner needs. Queries should be CONCEPT-level, not verbatim — e.g.,
'dietary restrictions allergies guests', 'fallback if vendor cancels', 'how
families respond to communications'. No numbering, no commentary, just one
query per line."""


JUDGE_SYSTEM = """You are a strict plan evaluator. You score a plan against a
fixed pitfall checklist and an overall quality rubric.

Output STRICT JSON with keys:
  pitfalls_avoided: object mapping each pitfall id (e.g. "P1") to true/false
                    (true = plan addresses that pitfall)
  pitfall_evidence: object mapping each pitfall id to a 1-line quote/paraphrase
  quality_1_10: integer 1-10 for overall plan quality (clarity, coverage,
                actionability, principled coordination thinking)
  principles_invoked: integer count of distinct PRINCIPLE-level statements
                      (e.g., \"don't assume uniform familiarity\") in the plan
  quality_rationale: one sentence

Be strict. A vague gesture toward a topic is NOT addressing the pitfall — the
plan must concretely act on it."""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


async def llm(system: str, user: str, max_tokens: int = 3000) -> str:
    resp = await CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Cue gen + chat retrieval
# ---------------------------------------------------------------------------


async def gen_cues(task: Task, retrieved_cards: list[ProcedureCard]) -> list[str]:
    cards_block = (
        "\n---\n".join(c.text for c in retrieved_cards) if retrieved_cards else "(none)"
    )
    user = (
        f"TASK:\n{task.prompt}\n\nRETRIEVED CARDS (lessons may suggest what to "
        f"probe for):\n{cards_block}\n\nEmit 3-6 retrieval queries now."
    )
    raw = await llm(CUE_SYSTEM, user, max_tokens=2000)
    lines = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]
    return lines[:6]


def retrieve_chat(
    mem: ChatMemory, cues: list[str], k_per_cue: int = 4
) -> list[ChatEvent]:
    seen: dict[str, ChatEvent] = {}
    for q in cues:
        for ev, _ in mem.query(q, k=k_per_cue):
            if ev.eid not in seen:
                seen[ev.eid] = ev
    return list(seen.values())


# ---------------------------------------------------------------------------
# Card store + retrieval (case base)
# ---------------------------------------------------------------------------


async def emit_card(task: Task, plan_text: str) -> ProcedureCard:
    user = (
        f"TASK PROMPT:\n{task.prompt}\n\nYOUR PLAN:\n{plan_text}\n\n"
        "Now emit the procedure card as STRICT JSON. Principle-level only. "
        f"Target ≤ {CARD_TARGET_TOKENS} tokens."
    )
    raw = await llm(CARD_SYSTEM, user, max_tokens=1500)
    parsed = _extract_json(raw)
    if parsed is None:
        body = raw[: CARD_TARGET_TOKENS * 4]
        task_type = task.task_type
    else:
        task_type = (parsed.get("task_type") or task.task_type)[:80]
        body = json.dumps(parsed, indent=2)
        if approx_tokens(body) > CARD_TARGET_TOKENS * 1.5:
            body = body[: int(CARD_TARGET_TOKENS * 1.5) * 4]
    return ProcedureCard(task_id=task.id, task_type=task_type, text=body)


async def retrieve_cards(
    case_base: CaseBase, task: Task, top_k: int = 2
) -> list[ProcedureCard]:
    if not case_base.cards:
        return []
    listing = "\n".join(
        f"[{i}] task_type={c.task_type}\n{c.text}"
        for i, c in enumerate(case_base.cards)
    )
    user = (
        f"NEW TASK:\n{task.prompt}\n\nPAST CARDS:\n{listing}\n\n"
        "Return the JSON ranking now."
    )
    raw = await llm(SIMILARITY_SYSTEM, user, max_tokens=800)
    parsed = _extract_json(raw)
    if not parsed or "ranking" not in parsed:
        return case_base.cards[-1:]
    indices = parsed["ranking"][:top_k]
    out: list[ProcedureCard] = []
    for idx in indices:
        if isinstance(idx, int) and 0 <= idx < len(case_base.cards):
            out.append(case_base.cards[idx])
    return out


# ---------------------------------------------------------------------------
# Planner driver (with bounded WM + retrieval-on-demand)
# ---------------------------------------------------------------------------


async def plan_with_wm(
    task: Task,
    chat_mem: ChatMemory,
    retrieved_cards: list[ProcedureCard],
    variant: str,
    log: list[dict[str, Any]],
) -> str:
    wm = WorkingMemory(cap_tokens=WM_CAP_TOKENS)

    # 1. Pin task prompt.
    wm.add(WMSlot(label="TASK", text=task.prompt, pinned=True))

    # 2. Operator: pin retrieved cards as compact text.
    if variant == "operator" and retrieved_cards:
        cards_text = "\n\n---\n\n".join(
            f"PAST CARD (task_type={c.task_type}):\n{c.text}" for c in retrieved_cards
        )
        wm.add(WMSlot(label="CARDS", text=cards_text, pinned=True))

    # 3. Cue-gen → chat retrieval. Snippets enter WM (NOT pinned, can be evicted).
    cues = await gen_cues(task, retrieved_cards if variant == "operator" else [])
    snippets = retrieve_chat(chat_mem, cues, k_per_cue=4)
    snippet_text = "\n".join(s.render() for s in snippets[:24])
    wm.add(WMSlot(label="CHAT_SNIPPETS", text=snippet_text))

    log.append(
        {
            "phase": "plan_setup",
            "variant": variant,
            "n_cues": len(cues),
            "cues": cues,
            "n_snippets": len(snippets),
            "wm_tokens": wm.total_tokens(),
            "wm_cap": WM_CAP_TOKENS,
            "wm_compactions": wm.compactions,
            "external_chat_tokens": chat_mem.total_tokens(),
        }
    )

    # 4. Plan.
    sys_prompt = PLAN_SYSTEM_OPERATOR if variant == "operator" else PLAN_SYSTEM_BASELINE
    user = f"WORKING MEMORY:\n{wm.render()}\n\nWrite the numbered plan now."
    plan = await llm(sys_prompt, user, max_tokens=3000)

    log.append(
        {
            "phase": "plan_done",
            "wm_tokens_at_plan": wm.total_tokens(),
            "wm_compactions_at_plan": wm.compactions,
            "plan_tokens": approx_tokens(plan),
        }
    )
    return plan


async def judge(task: Task, plan_text: str) -> dict[str, Any]:
    checklist = "\n".join(
        f"  {pid}: {desc}" for pid, desc in task.pitfall_checklist.items()
    )
    user = (
        f"TASK:\n{task.prompt}\n\nPITFALL CHECKLIST:\n{checklist}\n\n"
        f"PLAN TO EVALUATE:\n{plan_text}\n\nOutput strict JSON now."
    )
    raw = await llm(JUDGE_SYSTEM, user, max_tokens=4000)
    parsed = _extract_json(raw) or {}
    if not parsed:
        # Diagnostic: stash the raw judge output so failures are debuggable.
        parsed = {"_raw_head": raw[:500]}
    return parsed


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    task_id: str
    variant: str
    plan: str
    judge: dict[str, Any]
    n_pitfalls_avoided: int
    quality: int
    principles_invoked: int
    retrieved_card_ids: list[str]


def _count_pitfalls(judge_out: dict[str, Any]) -> int:
    pa = judge_out.get("pitfalls_avoided") or {}
    return sum(1 for v in pa.values() if v is True)


async def run_variant(
    variant: str, tasks: list[Task]
) -> tuple[list[TaskResult], CaseBase, list[dict[str, Any]]]:
    case_base = CaseBase()
    results: list[TaskResult] = []
    trace: list[dict[str, Any]] = []
    for i, task in enumerate(tasks):
        chat_mem = ChatMemory(task.chat)
        task_log: list[dict[str, Any]] = []
        task_log.append(
            {
                "phase": "task_start",
                "task_id": task.id,
                "variant": variant,
                "task_idx": i,
                "external_chat_events": len(chat_mem),
                "external_chat_tokens": chat_mem.total_tokens(),
                "case_base_size_before": len(case_base),
                "case_base_tokens_before": case_base.total_tokens(),
            }
        )

        retrieved: list[ProcedureCard] = []
        if variant == "operator" and i > 0 and case_base.cards:
            retrieved = await retrieve_cards(case_base, task, top_k=2)
            task_log.append(
                {
                    "phase": "case_base_retrieval",
                    "n_retrieved": len(retrieved),
                    "retrieved_ids": [c.task_id for c in retrieved],
                }
            )

        plan = await plan_with_wm(task, chat_mem, retrieved, variant, task_log)
        judge_out = await judge(task, plan)
        quality = int(judge_out.get("quality_1_10") or 0)
        principles = int(judge_out.get("principles_invoked") or 0)
        n_avoid = _count_pitfalls(judge_out)
        results.append(
            TaskResult(
                task_id=task.id,
                variant=variant,
                plan=plan,
                judge=judge_out,
                n_pitfalls_avoided=n_avoid,
                quality=quality,
                principles_invoked=principles,
                retrieved_card_ids=[c.task_id for c in retrieved],
            )
        )

        # Emit + (operator only) store card.
        card = await emit_card(task, plan)
        task_log.append(
            {
                "phase": "card_emitted",
                "card_tokens": card.approx_tokens(),
                "card_text_head": card.text[:300],
            }
        )
        if variant == "operator":
            case_base.add(card)

        task_log.append(
            {
                "phase": "task_end",
                "pitfalls_avoided": n_avoid,
                "quality": quality,
                "principles_invoked": principles,
                "case_base_size_after": len(case_base),
                "case_base_tokens_after": case_base.total_tokens(),
            }
        )
        trace.append({"task_id": task.id, "variant": variant, "events": task_log})
        print(
            f"  [{variant}] {task.id}: pitfalls={n_avoid}/5 q={quality} "
            f"principles={principles} retrieved={[c.task_id for c in retrieved]} "
            f"cb={len(case_base)}c/{case_base.total_tokens()}t "
            f"chat={chat_mem.total_tokens()}t"
        )
    return results, case_base, trace


async def main() -> None:
    tasks = build_tasks()
    print(f"Tasks: {[t.id for t in tasks]}")
    for t in tasks:
        print(
            f"  {t.id}: {len(t.chat)} chat events, "
            f"~{ChatMemory(t.chat).total_tokens()} tokens external memory"
        )

    print("\n--- BASELINE ---")
    baseline_results, _, baseline_trace = await run_variant("baseline", tasks)
    print("\n--- OPERATOR ---")
    operator_results, final_cb, operator_trace = await run_variant("operator", tasks)

    def agg(rs: list[TaskResult], slice_: slice | None = None) -> dict[str, float]:
        sel = rs[slice_] if slice_ else rs
        n = len(sel)
        if n == 0:
            return {
                "n": 0,
                "pitfalls_avg": 0.0,
                "quality_avg": 0.0,
                "principles_avg": 0.0,
            }
        return {
            "n": n,
            "pitfalls_avg": sum(r.n_pitfalls_avoided for r in sel) / n,
            "quality_avg": sum(r.quality for r in sel) / n,
            "principles_avg": sum(r.principles_invoked for r in sel) / n,
        }

    summary = {
        "n_tasks": len(tasks),
        "model": MODEL,
        "wm_cap_tokens": WM_CAP_TOKENS,
        "case_base_budget_tokens": CARD_BUDGET_TOKENS,
        "external_chat_tokens_per_task": [
            {"task": t.id, "tokens": ChatMemory(t.chat).total_tokens()} for t in tasks
        ],
        "baseline_overall": agg(baseline_results),
        "operator_overall": agg(operator_results),
        "baseline_tasks_2plus": agg(baseline_results, slice(1, None)),
        "operator_tasks_2plus": agg(operator_results, slice(1, None)),
        "baseline_tasks_4plus": agg(baseline_results, slice(3, None)),
        "operator_tasks_4plus": agg(operator_results, slice(3, None)),
        "case_base_final_tokens": final_cb.total_tokens(),
        "case_base_size": len(final_cb),
        "per_task_baseline": [asdict(r) for r in baseline_results],
        "per_task_operator": [asdict(r) for r in operator_results],
    }

    out_path = RESULTS_DIR / "run.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Per-task token-trace logs.
    ts = time.strftime("%Y%m%d_%H%M%S")
    for tr in baseline_trace + operator_trace:
        path = LOGS_DIR / f"{ts}_{tr['variant']}_{tr['task_id']}.json"
        path.write_text(json.dumps(tr, indent=2))

    print("\n=== SUMMARY ===")
    print(f"Baseline overall:    {summary['baseline_overall']}")
    print(f"Operator overall:    {summary['operator_overall']}")
    print(f"Baseline tasks 2+:   {summary['baseline_tasks_2plus']}")
    print(f"Operator tasks 2+:   {summary['operator_tasks_2plus']}")
    print(f"Baseline tasks 4-5:  {summary['baseline_tasks_4plus']}")
    print(f"Operator tasks 4-5:  {summary['operator_tasks_4plus']}")
    print(
        f"Final case base: {summary['case_base_size']} cards, "
        f"{summary['case_base_final_tokens']} tokens"
    )
    print(f"Wrote {out_path}")
    print(f"Per-task token-traces in {LOGS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
