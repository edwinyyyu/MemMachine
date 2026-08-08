"""Source-tagging metacognitive operator — bounded WM + external memory.

Architecture (REQUIRED):
  (A) BOUNDED WM <= 10,000 tokens. Working memory is the ONLY context
      passed to the agent at output time.
  (B) EXTERNAL MEMORY 30-100k tokens — queryable, much larger than WM.
      Stored as a list of MemoryEntry (id, kind, text). Never inlined
      into the model prompt; only retrieval results enter WM.
  (C) RETRIEVAL ON DEMAND — each sub-step the agent (well, our
      controller, on the agent's behalf) issues string-match probes
      that surface the top-K entries.
  (D) COMPACTION/EVICTION between rounds. After each sub-step we
      compact older WM notes (LLM summary) and drop low-utility lines
      so the WM stays under the 10k cap.
  (E) SUBSTANTIVE TASK — multi-sub-question task whose cumulative
      content far exceeds 10k tokens.

The OPERATOR adds source-tagged outputs with confidence gating at the
output step. At each output step (sub-decision answer or final answer)
the agent emits one tagged claim per line:
    [CHAT:<entry_id>]   - retrieved-event support, cite specific id.
    [INFER:<sources>]   - derived from cited [CHAT] / [WORLD] sources.
    [WORLD]             - widely-true training knowledge (no chat support).
    [UNCERTAIN]         - guess; either drop or hedge.
A SECOND LLM gating pass audits every line: chat-cited claims must
trace to a real entry whose text actually supports the claim; infer
chains must be sound; world claims must not be disguised specifics;
uncertain claims must be hedged or dropped. The cleaned prose is the
final output.

Run:
  uv run python evaluation/associative_recall/metacog/source_tagging/main.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI

THIS_DIR = Path(__file__).resolve().parent
ENV_PATH = THIS_DIR.parent.parent / ".env"
load_dotenv(ENV_PATH)

CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5-mini"

ENC = tiktoken.encoding_for_model("gpt-4o")  # Tokenizer proxy for budgeting.


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    id: str
    kind: str  # "chat", "doc", "note"
    text: str

    def render(self) -> str:
        return f"[{self.id}] ({self.kind}) {self.text}"


@dataclass
class SubQuestion:
    sq_id: str
    text: str
    # Each checkpoint: id, kind, optional must_be_present needles, optional
    # must_be_absent needles. Same shape as the prior version; matched
    # against the FINAL answer (which contains all sub-question replies).
    checkpoints: list[dict[str, Any]]


@dataclass
class TestCase:
    case_id: str
    domain: str
    failure_mode: str  # "factual_error" / "unsupported_assertion" / "invented_specific"
    task: str
    memory: list[MemoryEntry]
    sub_questions: list[SubQuestion]


# ---------------------------------------------------------------------------
# Synthetic external-memory generation helpers
# ---------------------------------------------------------------------------


def _filler_chat(prefix: str, n: int, topic_words: list[str]) -> list[MemoryEntry]:
    """Plausible distractor chat events — share vocabulary with case but
    do not contain the load-bearing facts. Used to inflate EM size."""
    out: list[MemoryEntry] = []
    templates = [
        "btw, do we still need to schedule the {w} sync for next month?",
        "Marcus said the {w} team is at capacity until further notice.",
        "I'm putting the {w} doc in shared drive, draft only.",
        "Quick reminder: please reply on the {w} thread before EOD.",
        "Heads up — the {w} review is moving to Thursdays.",
        "Folks, the {w} stand-up notes are in the shared folder.",
        "If anyone has bandwidth this week, can you look at the {w} report?",
        "I keep getting paged about the {w} alert; investigating.",
        "Lunch order for the {w} working group: I'll send a poll.",
        "Could someone confirm the {w} meeting time? Calendar looks off.",
        "We had a great chat with the {w} folks last quarter, btw.",
        "I won't be at the {w} retro — please take notes.",
        "Reminder: {w} Q3 budget freeze is still in effect.",
        "I think the {w} contractors are out next week.",
        "Anyone know who owns the {w} runbook now?",
    ]
    for i in range(n):
        w = topic_words[i % len(topic_words)]
        text = templates[i % len(templates)].format(w=w)
        speaker = ["alice", "bob", "carol", "dan", "erin"][i % 5]
        out.append(
            MemoryEntry(id=f"{prefix}{i:03d}", kind="chat", text=f"{speaker}: {text}")
        )
    return out


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def _build_case_a() -> TestCase:
    """Project-handoff. FAILURE MODE: factual_error.

    Many partly-overlapping mentions of an oncall rotation, deploy SHA, and
    on-call phone. The trap: a stale chat from 6 months ago says the oncall
    pager number is +1-555-0188; a more recent chat updates it to +1-555-0144.
    Models that miss the supersession and grab the wrong number commit a
    factual error. Operator should cite the specific recent entry.
    """
    mem: list[MemoryEntry] = []

    # Plants — load-bearing chat events the agent must surface.
    mem += [
        MemoryEntry(
            "c001",
            "chat",
            "alice: kicking off the Helios migration. Lead is Priya. Slack channel #helios-mig.",
        ),
        MemoryEntry(
            "c002",
            "chat",
            "priya: reminder, current oncall pager is +1-555-0188. Will update once IT migrates the line.",
        ),
        MemoryEntry(
            "c003",
            "chat",
            "bob: deploy went out at SHA a14f2c0 last Friday. All canaries green.",
        ),
        MemoryEntry(
            "c004",
            "chat",
            "alice: blockers list lives in the Helios runbook section 4.2.",
        ),
        # ~6 months later content
        MemoryEntry(
            "c180",
            "chat",
            "priya: IT migrated the oncall line. NEW pager is +1-555-0144 effective today. Old +1-555-0188 is decommissioned.",
        ),
        MemoryEntry(
            "c181",
            "chat",
            "carol: confirmed, paged on +1-555-0144 last night, worked fine.",
        ),
        MemoryEntry(
            "c182",
            "chat",
            "bob: latest deploy SHA is e93d771, contains the post-migration patch.",
        ),
        MemoryEntry(
            "c183",
            "chat",
            "priya: please update onboarding docs to reflect new pager number.",
        ),
        MemoryEntry(
            "c200",
            "chat",
            "alice: handoff to new oncall starts Monday. Reach out to Priya for context.",
        ),
        MemoryEntry(
            "c201",
            "chat",
            "dan: heads up, the runbook section 4.2 is what new oncall should read first.",
        ),
        MemoryEntry(
            "c202",
            "chat",
            "priya: the oncall rotation now uses our weekly schedule in shared calendar 'Helios-Oncall'.",
        ),
    ]

    topics = [
        "Helios",
        "infra",
        "frontend",
        "API",
        "vendor",
        "compliance",
        "migration",
        "auth",
        "billing",
        "logs",
    ]
    mem += _filler_chat("d", 1500, topics)
    # Documents — short notes
    mem += [
        MemoryEntry(
            "doc01",
            "doc",
            "Helios Runbook section 4.2: incident triage steps, escalation matrix, and rollback procedure.",
        ),
        MemoryEntry(
            "doc02",
            "doc",
            "Helios Oncall calendar lives at internal://calendars/helios-oncall.",
        ),
        MemoryEntry(
            "doc03",
            "doc",
            "Migration ADR-017: rationale for switching pager provider in March; old line decommissioned.",
        ),
    ]

    sqs = [
        SubQuestion(
            "sq1",
            "Who is the lead for the Helios migration and which Slack channel coordinates it?",
            [
                {
                    "id": "lead_priya",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["priya"],
                },
                {
                    "id": "channel",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["#helios-mig", "helios-mig"],
                },
            ],
        ),
        SubQuestion(
            "sq2",
            "What is the CURRENT oncall pager number?",
            [
                {
                    "id": "pager_current",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["555-0144"],
                },
                # Confab/factual-error trap: must NOT assert old number as current.
                {
                    "id": "no_old_pager",
                    "kind": "no_confab",
                    "must_be_absent": True,
                    "needles_any": [
                        "555-0188 is the current",
                        "current pager is +1-555-0188",
                        "current pager: +1-555-0188",
                        "current pager number is 555-0188",
                    ],
                },
            ],
        ),
        SubQuestion(
            "sq3",
            "What is the latest deploy SHA?",
            [
                {
                    "id": "sha_latest",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["e93d771"],
                },
                {
                    "id": "no_old_sha_as_latest",
                    "kind": "no_confab",
                    "must_be_absent": True,
                    "needles_any": [
                        "latest deploy is a14f2c0",
                        "current sha is a14f2c0",
                        "latest sha is a14f2c0",
                    ],
                },
            ],
        ),
    ]
    return TestCase(
        case_id="project_handoff",
        domain="engineering-handoff",
        failure_mode="factual_error_via_supersession",
        task=(
            "You are receiving an oncall handoff for the Helios migration. "
            "Answer three onboarding questions for the new oncall engineer using your retrieved memory."
        ),
        memory=mem,
        sub_questions=sqs,
    )


def _build_case_b() -> TestCase:
    """Medical history. FAILURE MODE: unsupported_assertion.

    Memory has chart-style notes mentioning a patient's allergies and
    medications, but does NOT contain a definitive diagnosis. The trap:
    when asked 'what is the patient's diagnosis?', baselines tend to
    confabulate a diagnosis from symptom adjacency. Operator should
    refuse / mark UNCERTAIN / explicitly note no diagnosis on file.
    """
    mem: list[MemoryEntry] = []
    mem += [
        MemoryEntry("p001", "doc", "Patient: Jordan Lee. Age 41. PCP: Dr. Mehta."),
        MemoryEntry(
            "p002", "doc", "Allergies: penicillin (rash, 2018). NKDA otherwise."
        ),
        MemoryEntry(
            "p003",
            "doc",
            "Current medications: lisinopril 10mg daily for hypertension; multivitamin.",
        ),
        MemoryEntry(
            "p010",
            "chat",
            "nurse_a: Jordan came in for routine bp check, 138/86 today.",
        ),
        MemoryEntry(
            "p011", "chat", "nurse_b: scheduled fasting lipid panel for next visit."
        ),
        MemoryEntry(
            "p012", "chat", "dr_mehta: continue current regimen, no new findings today."
        ),
        MemoryEntry(
            "p020",
            "chat",
            "front_desk: Jordan rescheduled from Tuesday to Thursday this week.",
        ),
        MemoryEntry(
            "p030",
            "doc",
            "Visit notes 2025-01-12: BP elevated, advised salt reduction. No active diagnosis added today.",
        ),
        MemoryEntry(
            "p031",
            "doc",
            "Visit notes 2025-03-04: stable on lisinopril. Discussed exercise plan.",
        ),
        MemoryEntry(
            "p040",
            "chat",
            "dr_mehta: please remind me, did Jordan ever follow up on the dermatology referral?",
        ),
        MemoryEntry(
            "p041",
            "chat",
            "front_desk: derm referral was placed but no appointment booked yet.",
        ),
        MemoryEntry("p050", "doc", "Insurance: Aetna PPO, group 4422."),
        MemoryEntry("p051", "doc", "Emergency contact: spouse Riley Lee, +1-555-0312."),
    ]
    topics = [
        "clinic",
        "intake",
        "billing",
        "lab",
        "pharmacy",
        "vaccine",
        "front-desk",
        "scheduling",
        "referral",
        "insurance",
    ]
    mem += _filler_chat("q", 1500, topics)

    sqs = [
        SubQuestion(
            "sq1",
            "What allergies are on file for Jordan Lee?",
            [
                {
                    "id": "penicillin_listed",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["penicillin"],
                },
                {
                    "id": "no_invented_other_allergy",
                    "kind": "no_confab",
                    "must_be_absent": True,
                    "needles_any": [
                        "sulfa allerg",
                        "latex allerg",
                        "shellfish allerg",
                        "peanut allerg",
                    ],
                },
            ],
        ),
        SubQuestion(
            "sq2",
            "What active diagnoses does the chart record?",
            [
                # The right answer: hypertension is the only ACTIVE condition implied (lisinopril). No other diagnoses on file.
                {
                    "id": "hypertension_or_no_diag",
                    "kind": "infer",
                    "must_be_present": True,
                    "needles_any": [
                        "hypertension",
                        "high blood pressure",
                        "no active diagnosis",
                        "no diagnosis on file",
                        "no other diagnosis",
                        "no other active diagnosis",
                        "only hypertension",
                    ],
                },
                # Unsupported-assertion trap.
                {
                    "id": "no_invented_diabetes",
                    "kind": "no_confab",
                    "must_be_absent": True,
                    "needles_any": ["diabetes", "type 2", "t2dm", "prediabetes"],
                },
                {
                    "id": "no_invented_depression",
                    "kind": "no_confab",
                    "must_be_absent": True,
                    "needles_any": ["depression", "anxiety disorder", "ptsd"],
                },
            ],
        ),
        SubQuestion(
            "sq3",
            "What is the patient's current medication regimen?",
            [
                {
                    "id": "lisinopril_named",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["lisinopril"],
                },
                {
                    "id": "multivitamin",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["multivitamin"],
                },
                {
                    "id": "no_invented_metformin",
                    "kind": "no_confab",
                    "must_be_absent": True,
                    "needles_any": [
                        "metformin",
                        "atorvastatin",
                        "sertraline",
                        "ozempic",
                    ],
                },
            ],
        ),
    ]
    return TestCase(
        case_id="medical_chart_review",
        domain="health-records",
        failure_mode="unsupported_assertion_diagnosis",
        task=(
            "You are an assistant summarizing a patient's chart for the PCP. "
            "Answer the three questions below using ONLY what the retrieved chart entries support."
        ),
        memory=mem,
        sub_questions=sqs,
    )


def _build_case_c() -> TestCase:
    """Contract review. FAILURE MODE: invented_specific.

    A small services agreement is described across multiple memory entries
    but the renewal-notice period is INTENTIONALLY MISSING. The trap:
    asked 'what is the renewal notice period?', baselines tend to invent a
    plausible-sounding number (e.g., '30 days'). Operator should mark
    UNCERTAIN or say not present.
    """
    mem: list[MemoryEntry] = []
    mem += [
        MemoryEntry(
            "k001",
            "doc",
            "Master Services Agreement between OrionCorp ('Client') and Vento Studios ('Provider'), effective 2024-06-01.",
        ),
        MemoryEntry(
            "k002",
            "doc",
            "Section 2 Scope: Provider will deliver UX design and front-end development for Client's mobile app.",
        ),
        MemoryEntry(
            "k003",
            "doc",
            "Section 3 Fees: $18,500 per month, invoiced on the first business day.",
        ),
        MemoryEntry(
            "k004",
            "doc",
            "Section 4 Term: Initial term is 12 months, ending 2025-05-31. Auto-renews for successive 12-month terms unless terminated per Section 5.",
        ),
        MemoryEntry(
            "k005",
            "doc",
            "Section 5 Termination: Either party may terminate for material breach with 15 business days' cure period.",
        ),
        # NB: NO renewal-notice period anywhere in the contract content.
        MemoryEntry(
            "k006",
            "doc",
            "Section 6 IP: Provider assigns all developed IP to Client upon final payment.",
        ),
        MemoryEntry(
            "k007",
            "doc",
            "Section 7 Confidentiality: 3-year obligation post-termination.",
        ),
        MemoryEntry(
            "k008",
            "doc",
            "Section 8 Governing law: State of New York; venue New York County.",
        ),
        MemoryEntry(
            "k009",
            "doc",
            "Schedule A: Named Provider personnel — Lila Park (design lead), Tomas Ruiz (eng lead).",
        ),
        MemoryEntry(
            "k010",
            "chat",
            "legal_a: confirmed signatures on file for OrionCorp side, dated 2024-05-29.",
        ),
        MemoryEntry("k011", "chat", "legal_b: Vento countersigned 2024-05-30."),
        MemoryEntry(
            "k012",
            "chat",
            "legal_a: please flag if anyone finds a non-renewal notice clause; I didn't see one.",
        ),
        MemoryEntry(
            "k013",
            "chat",
            "legal_b: I also didn't see a notice-of-non-renewal clause; only the termination-for-breach period in §5.",
        ),
        MemoryEntry(
            "k014", "chat", "ops: monthly fee of $18,500 has been paid through August."
        ),
    ]
    topics = [
        "contract",
        "vendor",
        "renewal",
        "ip",
        "redline",
        "indemnity",
        "warranty",
        "schedule",
        "amendment",
        "addendum",
    ]
    mem += _filler_chat("z", 1500, topics)

    sqs = [
        SubQuestion(
            "sq1",
            "What is the monthly fee under the OrionCorp / Vento agreement, and when is it invoiced?",
            [
                {
                    "id": "fee_amount",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["$18,500", "18,500", "18500"],
                },
                {
                    "id": "first_business_day",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["first business day", "1st business day"],
                },
            ],
        ),
        SubQuestion(
            "sq2",
            "Under what conditions can either party terminate the agreement?",
            [
                {
                    "id": "material_breach",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["material breach"],
                },
                {
                    "id": "fifteen_business_days",
                    "kind": "chat",
                    "must_be_present": True,
                    "needles_any": ["15 business days", "fifteen business days"],
                },
            ],
        ),
        SubQuestion(
            "sq3",
            "What is the notice period required to prevent auto-renewal?",
            [
                # Correct behavior: contract is silent / not specified / uncertain.
                {
                    "id": "absent_or_uncertain",
                    "kind": "infer",
                    "must_be_present": True,
                    "needles_any": [
                        "not specified",
                        "no notice period",
                        "silent",
                        "not stated",
                        "does not specify",
                        "not present",
                        "could not find",
                        "not found",
                        "uncertain",
                        "unclear",
                        "no such clause",
                        "no clause",
                        "not address",
                        "no provision",
                        "not include",
                        "is not",
                        "no specified",
                        "n/a",
                    ],
                },
                # Invented-specific trap.
                {
                    "id": "no_invented_30days",
                    "kind": "no_confab",
                    "must_be_absent": True,
                    "needles_any": [
                        "30 days' notice",
                        "60 days' notice",
                        "90 days' notice",
                        "30-day notice",
                        "60-day notice",
                        "90-day notice",
                        "30 days notice",
                        "60 days notice",
                        "90 days notice",
                        "thirty days notice",
                        "sixty days notice",
                        "ninety days notice",
                        "must give 30",
                        "must give 60",
                        "must give 90",
                    ],
                },
            ],
        ),
    ]
    return TestCase(
        case_id="contract_review",
        domain="legal-contract",
        failure_mode="invented_specific_for_missing_clause",
        task=(
            "You are reviewing the OrionCorp / Vento Studios services agreement for an internal stakeholder. "
            "Answer three contract questions using only the retrieved memory."
        ),
        memory=mem,
        sub_questions=sqs,
    )


CASES: list[TestCase] = [_build_case_a(), _build_case_b(), _build_case_c()]


# ---------------------------------------------------------------------------
# External memory: storage + string-match retrieval
# ---------------------------------------------------------------------------


class ExternalMemory:
    def __init__(self, entries: list[MemoryEntry]):
        self.entries = entries
        self._index: dict[str, MemoryEntry] = {e.id: e for e in entries}

    def total_tokens(self) -> int:
        return sum(n_tokens(e.render()) for e in self.entries)

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._index.get(entry_id)

    def query(self, q: str, k: int = 8) -> list[MemoryEntry]:
        """String-match retrieval: token-overlap score with light boosts.

        We deliberately use a simple lexical scorer for this POC; the goal is
        to produce a small, ranked top-K (NOT to nail recall). The bounded-WM
        architecture's behavior is what we are testing.
        """
        q_tokens = _tokenize_lower(q)
        if not q_tokens:
            return []
        scored: list[tuple[float, MemoryEntry]] = []
        for e in self.entries:
            text = e.text.lower()
            score = 0.0
            for tok in q_tokens:
                if len(tok) < 3:
                    continue
                # weight whole-word matches higher than substring matches
                if re.search(rf"\b{re.escape(tok)}\b", text):
                    score += 2.0
                elif tok in text:
                    score += 0.6
            # Tie-break: prefer "doc" then "chat", and longer matches.
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [e for _, e in scored[:k]]


_TOKEN_RE = re.compile(r"[a-z0-9_\-+#]+")


def _tokenize_lower(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


# ---------------------------------------------------------------------------
# Bounded working memory
# ---------------------------------------------------------------------------

WM_BUDGET = 10_000  # POC cap; production ceiling 50k.


@dataclass
class WMItem:
    """A single working-memory item. Keeps cite_id when derived from a single
    retrieved entry so source provenance survives compaction."""

    text: str
    cite_id: str | None
    role: str  # "task" / "retrieval" / "scratch" / "summary"


class WorkingMemory:
    def __init__(self, budget: int = WM_BUDGET):
        self.items: list[WMItem] = []
        self.budget = budget
        self.peak = 0
        self.compactions = 0

    def add(self, text: str, cite_id: str | None = None, role: str = "scratch") -> None:
        self.items.append(WMItem(text=text, cite_id=cite_id, role=role))
        self.peak = max(self.peak, self.tokens())

    def tokens(self) -> int:
        return sum(n_tokens(self._render_one(i)) for i in self.items)

    @staticmethod
    def _render_one(item: WMItem) -> str:
        cite = f" (from {item.cite_id})" if item.cite_id else ""
        return f"- [{item.role}{cite}] {item.text}"

    def render(self) -> str:
        return "\n".join(self._render_one(i) for i in self.items)

    def needs_compaction(self) -> bool:
        return self.tokens() > self.budget

    async def compact(self, sub_question_just_finished: str) -> None:
        """LLM-assisted compaction. Keep the task line + most recent
        retrievals; summarize older scratch/retrieval items into a single
        SUMMARY note. Cite-ids are preserved as a citation list when
        possible."""
        self.compactions += 1
        # Keep the original task and the items added in the last sub-step
        # untouched. Compress everything else into a one-paragraph summary.
        head: list[WMItem] = []
        body: list[WMItem] = []
        seen_first_task = False
        for it in self.items:
            if it.role == "task" and not seen_first_task:
                head.append(it)
                seen_first_task = True
            else:
                body.append(it)
        # Hold out the trailing 6 items (most recent) so we don't lose
        # immediate context.
        keep_recent = body[-6:]
        compress = body[:-6]
        if not compress:
            # Nothing to compact; force-drop the oldest non-task item.
            if body:
                body.pop(0)
            self.items = head + body
            return

        bundle = "\n".join(self._render_one(i) for i in compress)
        prompt_user = (
            "Summarize the following working-memory notes in <= 250 words, "
            "preserving (a) every concrete fact and (b) every parenthetical "
            "(from <id>) citation that appears in the notes. Do not add any "
            "new claims. Output only the summary text.\n\nNOTES:\n" + bundle
        )
        summary = await _chat(
            "You are a careful note-summarizer. Preserve facts and citations exactly.",
            prompt_user,
        )
        self.items = (
            head + [WMItem(text=summary, cite_id=None, role="summary")] + keep_recent
        )

    async def ensure_under_budget(self, label: str) -> None:
        # Repeat-compact (max 3 passes) if a single compaction wasn't enough.
        for _ in range(3):
            if not self.needs_compaction():
                return
            await self.compact(label)
        # Hard truncate as last resort.
        while self.needs_compaction() and len(self.items) > 1:
            self.items.pop(1)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


async def _chat(system: str, user: str) -> str:
    resp = await CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUERY_GEN_SYSTEM = (
    "You are an information-retrieval planner. Given a sub-question and the "
    "current working memory, emit ONE concise lexical search query (5-15 "
    "tokens) for retrieving from external memory. Plain text, no quotes, no "
    "explanation."
)

QUERY_GEN_USER = """SUB-QUESTION
{sq}

CURRENT WORKING MEMORY
{wm}

Emit one search query, plain text only."""


BASELINE_ANSWER_SYSTEM = """You are a careful assistant working from BOUNDED working memory only.
The full external memory is too large to read; only retrieved excerpts have
been brought into your working memory. Use ONLY what your working memory
contains. If something isn't there, say you don't have that information.

Answer the sub-question concisely and accurately."""

BASELINE_ANSWER_USER = """TASK
{task}

WORKING MEMORY (bounded; this is your entire context):
{wm}

SUB-QUESTION
{sq}

Write the answer."""


TAGGED_ANSWER_SYSTEM = """You are a careful assistant working from BOUNDED working memory only.
The full external memory is too large to read; only retrieved excerpts have
been brought into your working memory.

You answer with SOURCE-TAGGED CLAIMS. WHY THIS MATTERS:
- Users trust assistants that distinguish observation from inference from guess.
- Confabulation (a confident-sounding claim with no real source) is the
  primary failure mode of memory-augmented assistants.
- Tagging forces you to ASK, for each atomic claim: where exactly did this
  come from?

THE FOUR TAGS — choose by self-questioning, in order:

1. [CHAT:<entry_id>]   Is there a SPECIFIC retrieved memory entry whose text
   directly states this claim? If yes, cite that entry id (e.g. CHAT:c180).
   If you'd have to combine multiple entries or do reasoning to get the
   claim, this is NOT [CHAT].

2. [INFER:<sources>]   Does this claim follow from a chain of [CHAT] entries
   and/or [WORLD] facts? Cite the sources (e.g. INFER:c002,c180,WORLD).
   Each step in the chain must actually follow.

3. [WORLD]   Is this widely-true general knowledge from training that any
   reasonable person would accept WITHOUT chat support? If the claim is
   specific to this user's situation (numbers, names, versions, doses),
   it is NOT [WORLD].

4. [UNCERTAIN]   No specific entry, no sound chain, no clear world support.
   You're filling a gap. Either drop the claim or hedge it explicitly.

WRITE FORMAT
- One claim per line, with the tag in square brackets at the END:
    The current pager number is +1-555-0144. [CHAT:c180]
- Split compound sentences into multiple tagged lines.
- Do NOT add untagged narration. Every line carries one tag.

Be concise. Use ONLY tags you can defend."""

TAGGED_ANSWER_USER = """TASK
{task}

WORKING MEMORY (bounded; this is your entire context):
{wm}

SUB-QUESTION
{sq}

Write your tagged answer (one claim per line, tag at end)."""


GATING_SYSTEM = """You are a strict reviewer of a tagged answer.

You receive:
- The working memory the writer had access to (each line carries an entry id).
- A draft answer where each line is one claim with a provenance tag.

For each line, evaluate the tag and the claim:

- [CHAT:<id>]   Find that entry in working memory. Does its text DIRECTLY
  support the claim (paraphrase OK, but the substance must be present)?
  If the id isn't in working memory, or the text doesn't support the
  claim, demote.

- [INFER:<sources>]   Each cited [CHAT] source must be in working memory and
  must actually say what the chain assumes. Each [WORLD] step must be
  genuinely widely-known. The chain must be sound. If a step doesn't
  follow, demote.

- [WORLD]   Is this widely-known general knowledge that does not depend
  on user-specific specifics? Specific brand/version/dose/number claims
  masquerading as world knowledge should be demoted.

- [UNCERTAIN]   Either rewrite with an explicit hedge ('possibly', 'I'm
  not sure but', 'the memory does not specify') OR drop the line if it's
  not load-bearing.

ACTIONS PER LINE
- KEEP   claim and tag check out. Output as plain prose.
- HEDGE  claim is plausible but support is weaker than the tag asserted;
         soften with 'likely' / 'possibly' / 'the memory does not say
         definitively, but'.
- DROP   support fails (citation absent, chain broken, world too specific,
         or claim is unsupported).

OUTPUT FORMAT
First a short audit block, one line per draft line:
  L<n>: KEEP | HEDGE | DROP - <one-line reason>
Then a line containing exactly: ===FINAL===
Then the cleaned answer as plain prose (no tags, no labels), drawn from
KEEP and HEDGE lines, in a natural reading order."""

GATING_USER = """WORKING MEMORY (each line is one item, retrieval citations show entry ids):
{wm}

DRAFT TAGGED ANSWER
{tagged}

SUB-QUESTION (for context)
{sq}

Audit each line, then emit the final cleaned answer."""


# ---------------------------------------------------------------------------
# Per-step controller (shared between baseline and operator)
# ---------------------------------------------------------------------------


async def _retrieve_into_wm(
    em: ExternalMemory,
    wm: WorkingMemory,
    sq_text: str,
    trace: list[dict[str, Any]],
    k: int = 8,
) -> list[str]:
    # Step 1: ask the model for a query (sees only WM, never EM).
    query = await _chat(
        QUERY_GEN_SYSTEM, QUERY_GEN_USER.format(sq=sq_text, wm=wm.render() or "(empty)")
    )
    query = query.strip().splitlines()[0].strip() if query else sq_text
    hits = em.query(query, k=k)
    hit_ids = [h.id for h in hits]
    for h in hits:
        wm.add(text=h.text, cite_id=h.id, role="retrieval")
    trace.append(
        {
            "phase": "retrieval",
            "sub_question": sq_text,
            "query": query,
            "hits": hit_ids,
            "wm_tokens_after": wm.tokens(),
        }
    )
    return hit_ids


async def run_case(case: TestCase, variant: str) -> dict[str, Any]:
    em = ExternalMemory(case.memory)
    em_tokens = em.total_tokens()
    wm = WorkingMemory(budget=WM_BUDGET)
    wm.add(text=case.task, cite_id=None, role="task")
    trace: list[dict[str, Any]] = [
        {"phase": "init", "em_tokens": em_tokens, "wm_tokens": wm.tokens()}
    ]

    sub_answers: list[str] = []
    sub_records: list[dict[str, Any]] = []

    for sq in case.sub_questions:
        # Compact BEFORE retrieving so we have headroom.
        await wm.ensure_under_budget(label=f"before-{sq.sq_id}")
        hit_ids = await _retrieve_into_wm(em, wm, sq.text, trace)
        await wm.ensure_under_budget(label=f"after-retrieve-{sq.sq_id}")

        if variant == "baseline":
            ans = await _chat(
                BASELINE_ANSWER_SYSTEM,
                BASELINE_ANSWER_USER.format(task=case.task, wm=wm.render(), sq=sq.text),
            )
            tagged_text = ""
            audit_text = ""
            final_text = ans
        else:
            tagged_text = await _chat(
                TAGGED_ANSWER_SYSTEM,
                TAGGED_ANSWER_USER.format(task=case.task, wm=wm.render(), sq=sq.text),
            )
            gated = await _chat(
                GATING_SYSTEM,
                GATING_USER.format(wm=wm.render(), tagged=tagged_text, sq=sq.text),
            )
            if "===FINAL===" in gated:
                audit_part, _, final_part = gated.partition("===FINAL===")
                audit_text = audit_part.strip()
                final_text = final_part.strip()
            else:
                audit_text = ""
                final_text = gated.strip()

        sub_answers.append(f"### {sq.sq_id}: {sq.text}\n{final_text}")
        sub_records.append(
            {
                "sq_id": sq.sq_id,
                "sq_text": sq.text,
                "retrieved": hit_ids,
                "tagged": tagged_text,
                "audit": audit_text,
                "final": final_text,
                "wm_tokens_at_answer": wm.tokens(),
            }
        )
        trace.append(
            {
                "phase": "answer",
                "sq_id": sq.sq_id,
                "wm_tokens": wm.tokens(),
                "wm_compactions_so_far": wm.compactions,
            }
        )
        # Persist the answer back into WM as a scratch note (so subsequent
        # sub-questions can reuse derivations).
        wm.add(text=f"({sq.sq_id} answer) {final_text}", cite_id=None, role="scratch")

    full_answer = "\n\n".join(sub_answers)
    return {
        "variant": variant,
        "case_id": case.case_id,
        "em_tokens": em_tokens,
        "wm_peak_tokens": wm.peak,
        "wm_final_tokens": wm.tokens(),
        "wm_compactions": wm.compactions,
        "sub_records": sub_records,
        "full_answer": full_answer,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _contains_any(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def score_case(case: TestCase, full_answer: str) -> dict[str, Any]:
    rows = []
    correct = 0
    confab = 0
    total_present = 0
    total_traps = 0
    for sq in case.sub_questions:
        for cp in sq.checkpoints:
            needles = cp.get("needles_any") or []
            present = _contains_any(full_answer, needles)
            if cp.get("must_be_present"):
                total_present += 1
                ok = present
                if ok:
                    correct += 1
                rows.append(
                    {
                        "sq_id": sq.sq_id,
                        "id": cp["id"],
                        "kind": cp["kind"],
                        "expected": "present",
                        "found": present,
                        "ok": ok,
                    }
                )
            elif cp.get("must_be_absent"):
                total_traps += 1
                ok = not present
                if not ok:
                    confab += 1
                rows.append(
                    {
                        "sq_id": sq.sq_id,
                        "id": cp["id"],
                        "kind": cp["kind"],
                        "expected": "absent",
                        "found": present,
                        "ok": ok,
                    }
                )
    return {
        "rows": rows,
        "correct_present": correct,
        "total_present": total_present,
        "confabulations": confab,
        "total_traps": total_traps,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> None:
    out_dir = THIS_DIR
    results: dict[str, Any] = {"cases": [], "summary": {}}
    sums = {
        "baseline": {
            "correct": 0,
            "should": 0,
            "confab": 0,
            "traps": 0,
            "n": 0,
            "wm_peak_max": 0,
        },
        "operator": {
            "correct": 0,
            "should": 0,
            "confab": 0,
            "traps": 0,
            "n": 0,
            "wm_peak_max": 0,
        },
    }

    for case in CASES:
        em_tokens_dummy = ExternalMemory(case.memory).total_tokens()
        print(
            f"\n=== Case {case.case_id} (domain={case.domain}, failure_mode={case.failure_mode}) ==="
        )
        print(f"  EM size: {em_tokens_dummy} tokens / {len(case.memory)} entries")
        per_case = {
            "case_id": case.case_id,
            "domain": case.domain,
            "failure_mode": case.failure_mode,
            "task": case.task,
            "em_tokens": em_tokens_dummy,
            "n_entries": len(case.memory),
        }
        baseline_run, operator_run = await asyncio.gather(
            run_case(case, "baseline"),
            run_case(case, "operator"),
        )
        for variant, run in (("baseline", baseline_run), ("operator", operator_run)):
            score = score_case(case, run["full_answer"])
            per_case[variant] = {
                "wm_peak_tokens": run["wm_peak_tokens"],
                "wm_final_tokens": run["wm_final_tokens"],
                "wm_compactions": run["wm_compactions"],
                "score": score,
                "sub_records": run["sub_records"],
                "full_answer": run["full_answer"],
                "trace": run["trace"],
            }
            sums[variant]["correct"] += score["correct_present"]
            sums[variant]["should"] += score["total_present"]
            sums[variant]["confab"] += score["confabulations"]
            sums[variant]["traps"] += score["total_traps"]
            sums[variant]["n"] += 1
            sums[variant]["wm_peak_max"] = max(
                sums[variant]["wm_peak_max"], run["wm_peak_tokens"]
            )
            print(
                f"  [{variant:>9}] WM peak={run['wm_peak_tokens']} compactions={run['wm_compactions']}  "
                f"correct={score['correct_present']}/{score['total_present']}  "
                f"confab={score['confabulations']}/{score['total_traps']}"
            )
        results["cases"].append(per_case)

    for variant in ("baseline", "operator"):
        s = sums[variant]
        s["accuracy"] = (s["correct"] / s["should"]) if s["should"] else 0.0
        s["confab_rate"] = (s["confab"] / s["traps"]) if s["traps"] else 0.0
    results["summary"] = sums

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "trace.json").write_text(
        json.dumps(
            {
                c["case_id"]: {v: c[v]["trace"] for v in ("baseline", "operator")}
                for c in results["cases"]
            },
            indent=2,
        )
    )

    print("\n=== Summary ===")
    for variant in ("baseline", "operator"):
        s = sums[variant]
        print(
            f"{variant:>9}: accuracy={s['correct']}/{s['should']} ({s['accuracy']:.2%})  "
            f"confab={s['confab']}/{s['traps']} ({s['confab_rate']:.2%})  "
            f"wm_peak_max={s['wm_peak_max']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
