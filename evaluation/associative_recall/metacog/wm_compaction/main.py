"""Metacognitive WM compaction-controller experiment.

Tests ONE operator: a working-memory budget controller that runs every step
and (when WM tokens exceed an 8k threshold) invokes an LLM compaction pass
that returns a structured compacted WM (<= 5k tokens). The operator teaches
four operations (DROP / CHUNK / REFRESH / OFFLOAD) and is asked to preserve
goal-relevant content + key intermediate results.

ARCHITECTURE (both variants):
  - Bounded WM <= 10k tokens (POC ceiling).
  - External memory: 30-100k token corpus (chunked, queryable). NOT in prompt.
  - Loop: probe (LLM emits a query) -> retrieve top-K chunks -> WM update
          -> compaction-controller pass.
  - Final question step: WM only -> answer load-bearing-fact questions.

VARIANTS:
  - baseline: append-only WM with FIFO drop when over 10k. No relevance.
  - operator: at WM > 8k threshold, LLM compactor returns structured sections
              (LOAD-BEARING-FACTS / RECENT-OBSERVATIONS / TASK-CONTEXT / NOTES)
              <= 5k tokens.

Run: uv run python evaluation/associative_recall/metacog/wm_compaction/main.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
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

# Token budgets
WM_HARD_CAP = 10_000  # bounded WM ceiling (POC)
WM_THRESHOLD = 8_000  # operator triggers above this
WM_TARGET = 5_000  # operator-compacted WM target

# Loop budgets
MAX_STEPS = 12
TOP_K = 3  # chunks per retrieval

try:
    ENC = tiktoken.encoding_for_model("gpt-5-mini")
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


# -------------------- Synthetic external-memory corpora -------------------- #
#
# Each case has:
#   - a multi-step task (5-10 sub-decisions)
#   - 30-100k tokens of external memory chunks (corpus)
#   - 4-7 LOAD-BEARING gold facts hidden in the corpus, surfaced across the run
#   - final questions verifying recall of those facts
#
# We pad the corpus with topical-but-irrelevant filler so retrieval brings back
# real context, and total raw retrieval volume across the run greatly exceeds
# the 10k WM budget (forcing real compaction pressure).


@dataclass
class GoldFact:
    fact_id: str
    text: str  # planted into the corpus as an unmistakable load-bearing chunk
    question: str  # final-step question
    answer_keys: list[str]  # any of these substrings in the answer = correct


@dataclass
class Case:
    case_id: str
    task: str
    sub_steps: list[str]  # the agent will iterate over these as anchor probes
    chunks: list[str]  # external memory; each ~200-600 tokens
    gold_facts: list[GoldFact]


def _filler_paragraph(topic: str, seed: int) -> str:
    """Generate a topical-but-irrelevant filler chunk via deterministic templates."""
    rng = random.Random(seed)
    sentences = [
        f"In the {topic} domain, the standard reference text describes the canonical "
        f"workflow as a sequence of intake, normalization, review, and sign-off.",
        f"Practitioners in {topic} often debate whether per-item heuristics outperform "
        f"end-to-end statistical models; published evaluations are mixed.",
        f"A 2019 retrospective on {topic} found that organizations underestimate "
        f"calibration drift between rollout milestones and steady-state operations.",
        f"Common terminology in {topic} can confuse newcomers because acronyms vary "
        f"between regional offices and partner agencies.",
        f"Documentation conventions in {topic} typically favor numbered checklists "
        f"over prose, though prose remains common for rationale and exceptions.",
        f"Audits of {topic} programs have repeatedly highlighted the need for "
        f"clearer escalation paths when unusual cases appear mid-cycle.",
        f"Training cohorts in {topic} use a mix of shadowing, paired review, and "
        f"reflective practice journals over the first six weeks.",
        f"Tooling vendors in the {topic} space differ chiefly in licensing model, "
        f"data residency, and how they handle long tails of edge cases.",
        f"Quarterly metrics for {topic} typically include throughput, defect rate, "
        f"reopen rate, and cycle time, though weighting varies by team.",
        f"Historical records from {topic} programs show that early adopters "
        f"often paid a coordination tax that paid off only after 9-12 months.",
    ]
    rng.shuffle(sentences)
    return " ".join(sentences[:6]) + f" (filler-{seed})"


def _build_chunks(
    topic: str,
    real_chunks: list[str],
    n_filler: int,
    seed_base: int,
    target_tokens: int = 50_000,
) -> list[str]:
    """Mix real (load-bearing) chunks with filler chunks; expand filler until target."""
    chunks: list[str] = list(real_chunks)
    i = 0
    while True:
        chunk = _filler_paragraph(topic, seed_base + i)
        chunks.append(chunk)
        i += 1
        if i >= n_filler and sum(n_tokens(c) for c in chunks) >= target_tokens:
            break
        if i > 400:
            break
    rng = random.Random(seed_base)
    rng.shuffle(chunks)
    return chunks


# ---- Case 1: clinical trial protocol drafting ---- #
CASE1_REAL = [
    "PROTOCOL ADDENDUM v2.3 (binding): The IRB requires the new vasodilator study to use "
    "a primary endpoint of mean systolic blood pressure reduction at 12 weeks. The previous "
    "endpoint (24-week MACE composite) was rejected as underpowered for the proposed N=180. "
    "All consent forms and statistical analysis plans MUST cite the 12-week endpoint.",
    "INVESTIGATIONAL DRUG NOTE (binding): Compound VX-714 has a known photosensitivity "
    "interaction with tetracycline-class antibiotics. Concomitant tetracycline use within "
    "14 days of dosing is an absolute exclusion criterion for this trial. This was added "
    "after the Phase 1 dermatology signal report.",
    "RECRUITMENT ADVISORY (binding): The Boston site (PI: Dr. Reyes) cannot enroll patients "
    "under age 21 because the local Boston IRB amendment limits the trial to adults 21+. "
    "Other sites use the protocol's 18+ floor. Site-specific eligibility tables must "
    "distinguish Boston (>=21) from all other sites (>=18).",
    "RANDOMIZATION DECISION (binding): The DSMB approved 2:1 active:placebo allocation "
    "instead of the original 1:1 plan, after the sponsor agreed to fund an additional "
    "30 active-arm slots. Randomization software config and the SAP must reflect 2:1.",
    "BLINDING PROCEDURE (binding): This is a double-blind trial. Unblinding is permitted "
    "ONLY in suspected unexpected serious adverse reaction (SUSAR) cases, and ONLY by the "
    "unblinded medical monitor (Dr. Chen, not the site PI). No site staff may unblind.",
    "DATA HANDOFF (binding): Source data verification cycles run on a 6-week cadence, NOT "
    "the original 4-week cadence. The CRO (PharmaTech Services) negotiated this in the "
    "Master Services Agreement amendment dated 2025-11-18.",
]
CASE1_TASK = (
    "Draft a clinical-trial protocol summary document for the VX-714 vasodilator Phase 2 "
    "study. Cover, in order: (1) primary endpoint definition, (2) inclusion criteria, "
    "(3) exclusion criteria including drug interactions, (4) site-specific eligibility "
    "differences, (5) randomization scheme, (6) blinding and unblinding procedures, "
    "(7) data-management cadence, (8) DSMB and IRB oversight."
)
CASE1_STEPS = [
    "primary endpoint definition",
    "inclusion criteria",
    "exclusion criteria and drug interactions",
    "site-specific eligibility differences",
    "randomization scheme",
    "blinding and unblinding procedures",
    "data-management cadence",
    "DSMB and IRB oversight",
]
CASE1_GOLD = [
    GoldFact(
        "g1",
        CASE1_REAL[0],
        "What is the primary endpoint of the VX-714 trial and at what timepoint?",
        ["12 week", "12-week", "12  week", "twelve week", "systolic"],
    ),
    GoldFact(
        "g2",
        CASE1_REAL[1],
        "What concomitant medication class is an absolute exclusion criterion, and over what window?",
        ["tetracycline", "14 day", "14-day", "fourteen day"],
    ),
    GoldFact(
        "g3",
        CASE1_REAL[2],
        "What site-specific minimum-age difference applies, and which site has the higher floor?",
        ["21", "Boston", "Reyes"],
    ),
    GoldFact(
        "g4",
        CASE1_REAL[3],
        "What is the randomization allocation ratio (active:placebo)?",
        ["2:1", "2 to 1", "two to one"],
    ),
    GoldFact(
        "g5",
        CASE1_REAL[4],
        "Who is authorized to unblind a patient in a SUSAR, and from what role?",
        ["Chen", "medical monitor", "unblinded medical monitor"],
    ),
    GoldFact(
        "g6",
        CASE1_REAL[5],
        "What is the source-data verification cadence?",
        ["6 week", "6-week", "six week"],
    ),
]

# ---- Case 2: software-architecture migration runbook ---- #
CASE2_REAL = [
    "MIGRATION CONSTRAINT (binding): The legacy billing service uses MySQL 5.7 with "
    "binlog-based CDC. The target Postgres 15 cluster requires logical replication. The "
    "team built a custom Debezium-to-pgoutput bridge; cutover MUST go through this bridge, "
    "NOT direct dump-and-load, to preserve idempotent retry semantics for in-flight charges.",
    "DOWNTIME WINDOW (binding): The maintenance window is Saturday 2025-12-13 02:00-04:00 "
    "America/New_York. Total allowed read-write downtime is 90 minutes; reads can stay up "
    "via the standby replica during the entire window. Anything exceeding 90 minutes "
    "triggers customer credits per SLA section 4.2.",
    "DEPENDENT SERVICE NOTE (binding): The fraud-scoring service caches the billing "
    "service's account-tier table in Redis with a 30-minute TTL. After cutover, the "
    "fraud team needs an explicit cache flush at T+5 minutes to avoid stale-tier "
    "false positives. Coordinate with on-call fraud engineer Priya.",
    "ROLLBACK CRITERION (binding): If post-cutover error rate on the /charges endpoint "
    "exceeds 0.5% sustained over 10 minutes (measured by the prod-monitoring SLO board), "
    "the runbook MUST trigger DNS-level rollback to the legacy MySQL endpoint. The "
    "decision authority is the on-call SRE lead, not the migration owner.",
    "AUTH MIGRATION DETAIL (binding): The auth tokens are bcrypt-hashed in MySQL but the "
    "Postgres schema expects argon2id. A wrapper accepts EITHER hash on first login and "
    "re-hashes to argon2id on success; do not bulk-rehash, that breaks active sessions. "
    "The wrapper is enabled by feature flag billing.auth.dual_hash=true.",
    "OBSERVABILITY HANDOFF (binding): Datadog dashboards for the legacy service are "
    "indexed by service:billing-legacy. Post-cutover, the new dashboards live at "
    "service:billing-pg and the alert routing rules need an explicit update; otherwise "
    "PagerDuty pages still go to the legacy on-call rotation.",
    "DATA-INTEGRITY CHECK (binding): Before cutover, run the reconciliation script "
    "tools/recon_charges.py with --window=24h on a warm replica; expected output is "
    "delta=0 rows. Any non-zero delta blocks cutover until the discrepancy is "
    "explained in writing by the data-eng on-call.",
]
CASE2_TASK = (
    "Write a cutover runbook for migrating the billing service from MySQL 5.7 to "
    "Postgres 15. Cover, in order: (1) replication / CDC mechanism, (2) maintenance window "
    "and downtime SLA, (3) dependent-service coordination, (4) rollback criteria, "
    "(5) auth-credential handling, (6) observability handoff, (7) pre-cutover data "
    "integrity check."
)
CASE2_STEPS = [
    "replication and CDC mechanism",
    "maintenance window and downtime SLA",
    "dependent-service coordination",
    "rollback criteria",
    "auth-credential handling",
    "observability handoff",
    "pre-cutover data integrity check",
]
CASE2_GOLD = [
    GoldFact(
        "g1",
        CASE2_REAL[0],
        "What replication mechanism must be used for the cutover, and what is the alternative that is forbidden?",
        ["Debezium", "pgoutput", "bridge", "logical replication"],
    ),
    GoldFact(
        "g2",
        CASE2_REAL[1],
        "What is the maximum allowed read-write downtime?",
        ["90 minute", "90-minute", "ninety minute"],
    ),
    GoldFact(
        "g3",
        CASE2_REAL[2],
        "What dependent-service action is required at T+5 minutes after cutover?",
        ["cache flush", "Redis", "fraud", "Priya"],
    ),
    GoldFact(
        "g4",
        CASE2_REAL[3],
        "What error-rate threshold on /charges triggers rollback, and over what window?",
        ["0.5%", "0.5 %", "10 minute", "10-minute"],
    ),
    GoldFact(
        "g5",
        CASE2_REAL[4],
        "How are auth credentials migrated (no bulk rehash) and via what feature flag?",
        ["dual_hash", "argon2", "wrapper", "feature flag"],
    ),
    GoldFact(
        "g6",
        CASE2_REAL[5],
        "What service tag identifies the new Postgres dashboards in Datadog?",
        ["billing-pg", "service:billing-pg"],
    ),
    GoldFact(
        "g7",
        CASE2_REAL[6],
        "What pre-cutover script is run, and what output is expected?",
        ["recon_charges", "delta=0", "delta = 0", "reconciliation"],
    ),
]

# ---- Case 3: festival logistics planning ---- #
CASE3_REAL = [
    "PERMITTING NOTE (binding): The Riverside Park amphitheater permit caps amplified "
    "sound at 95 dB measured at the FOH mixing position, with a hard 22:00 local-time "
    "curfew. The headliner set therefore must end by 21:50 to allow for outro and "
    "venue handover. The permit is filed under city application #RP-2026-0331.",
    "ARTIST RIDER ALERT (binding): The headliner act, Mira Solano, has a contractual "
    "rider requiring an all-vegan green-room catering setup, no flash photography "
    "during the first three songs, and a 90-minute soundcheck slot ending at least "
    "60 minutes before doors. Violations risk contractual penalties.",
    "VENDOR LOGISTICS (binding): The food-truck row uses 30A twist-lock power per stall; "
    "vendors arriving with 50A or standard-edison cables will not be served. The site "
    "manager (Theo) has 12 30A-to-50A adapters in the trailer, but only as last-resort "
    "loaners. Power layout published in the vendor packet on 2026-03-12.",
    "WEATHER CONTINGENCY (binding): Forecast shows 60% chance of evening thunderstorms. "
    "The lightning-detection threshold is any strike within 8 miles; on detection, the "
    "site goes to LEVEL-2 evacuation (audience clears the lawn to the indoor pavilion) "
    "within 15 minutes. The PA must broadcast a pre-recorded LEVEL-2 message twice.",
    "MEDICAL STAFFING (binding): On-site EMS is two ALS units staffed continuously; "
    "the medical lead is Dr. Aiyana Bear. The closest hospital with an ED capable of "
    "handling cardiac cases is Memorial East, 14 minutes by ambulance — NOT the closer "
    "Riverside Community urgent care, which does NOT have a cath lab.",
    "TICKETING POLICY (binding): All entry is digital-only via QR ticket on the festival "
    "app. Box-office cash sales are NOT allowed under the city permit; turn-aways must "
    "be directed to the online portal. Customer-service script v3 covers the most common "
    "objections; supervisors hold escalation discretion for ADA accommodations only.",
    "INSURANCE & RELEASE (binding): The general-liability insurance certificate must "
    "name the city of Riverside as additionally insured at $5M aggregate. The signed "
    "performer waiver is required for all artists; missing waivers result in pulled "
    "credentials at the talent gate (handled by talent coordinator Jamal).",
]
CASE3_TASK = (
    "Build the operations brief for Saturday's headlining day at the Riverside Music "
    "Festival. Cover, in order: (1) sound and curfew permit constraints, (2) headliner "
    "rider compliance, (3) food-vendor power logistics, (4) severe-weather contingency, "
    "(5) on-site medical and hospital routing, (6) ticketing and entry policy, "
    "(7) insurance and waiver compliance."
)
CASE3_STEPS = [
    "sound and curfew permit",
    "headliner rider compliance",
    "food-vendor power logistics",
    "severe-weather contingency",
    "medical and hospital routing",
    "ticketing and entry policy",
    "insurance and waivers",
]
CASE3_GOLD = [
    GoldFact(
        "g1",
        CASE3_REAL[0],
        "What is the dB cap (and at what measurement point) and what is the hard curfew time?",
        ["95 dB", "95dB", "22:00", "10pm", "10:00 pm"],
    ),
    GoldFact(
        "g2",
        CASE3_REAL[1],
        "Name two binding requirements from the headliner Mira Solano's rider.",
        ["vegan", "no flash", "flash photography", "soundcheck"],
    ),
    GoldFact(
        "g3",
        CASE3_REAL[2],
        "What is the required power connector for food trucks, and what fallback exists?",
        ["30A", "twist-lock", "twist lock", "adapter", "Theo"],
    ),
    GoldFact(
        "g4",
        CASE3_REAL[3],
        "What is the lightning-strike radius that triggers a LEVEL-2 evacuation, and within what time?",
        ["8 mile", "8-mile", "eight mile", "15 minute", "15-minute"],
    ),
    GoldFact(
        "g5",
        CASE3_REAL[4],
        "Which hospital should cardiac cases be routed to, and how far is it?",
        ["Memorial East", "14 minute", "14-minute"],
    ),
    GoldFact(
        "g6",
        CASE3_REAL[5],
        "Are cash sales allowed at the box office? What is the sole entry mechanism?",
        ["digital", "QR", "no cash", "not allowed"],
    ),
    GoldFact(
        "g7",
        CASE3_REAL[6],
        "What is the required general-liability aggregate limit and who must be named?",
        ["$5M", "5M", "5 million", "Riverside"],
    ),
]


def build_case1() -> Case:
    return Case(
        case_id="clinical_trial_protocol",
        task=CASE1_TASK,
        sub_steps=CASE1_STEPS,
        chunks=_build_chunks(
            "clinical-research", CASE1_REAL, 80, seed_base=1001, target_tokens=40_000
        ),
        gold_facts=CASE1_GOLD,
    )


def build_case2() -> Case:
    return Case(
        case_id="db_migration_runbook",
        task=CASE2_TASK,
        sub_steps=CASE2_STEPS,
        chunks=_build_chunks(
            "site-reliability-engineering",
            CASE2_REAL,
            80,
            seed_base=2002,
            target_tokens=40_000,
        ),
        gold_facts=CASE2_GOLD,
    )


def build_case3() -> Case:
    return Case(
        case_id="festival_ops_brief",
        task=CASE3_TASK,
        sub_steps=CASE3_STEPS,
        chunks=_build_chunks(
            "event-operations", CASE3_REAL, 80, seed_base=3003, target_tokens=40_000
        ),
        gold_facts=CASE3_GOLD,
    )


CASES = [build_case1(), build_case2(), build_case3()]


# -------------------- External-memory retrieval -------------------- #


class ExternalMemory:
    """Simple keyword-overlap retrieval over chunk corpus.

    Not in-prompt: the agent must issue queries; only top-K chunks are returned.
    """

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
        "site",
        "case",
        "step",
    }

    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    @classmethod
    def _toks(cls, s: str) -> list[str]:
        return [
            t
            for t in re.findall(r"[a-zA-Z0-9_\-]+", s.lower())
            if t not in cls._STOP and len(t) > 2
        ]

    def query(self, q: str, k: int = TOP_K) -> list[str]:
        q_terms = self._toks(q)
        if not q_terms:
            return []
        scored: list[tuple[float, int, str]] = []
        for i, ch in enumerate(self.chunks):
            ch_terms = self._toks(ch)
            if not ch_terms:
                continue
            overlap = sum(1 for t in q_terms if t in ch_terms)
            if overlap == 0:
                continue
            # length-normalized overlap
            score = overlap / (len(set(ch_terms)) ** 0.5)
            scored.append((score, i, ch))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [c for _, _, c in scored[:k]]


# -------------------- Prompts -------------------- #

PROBE_SYSTEM = (
    "You are a memory-augmented agent solving a multi-step task. Each step you "
    "must issue ONE retrieval query against the external memory to gather any "
    "task-relevant information for the current sub-step.\n\n"
    "Output exactly one line:\n"
    "QUERY: <a short, content-bearing query string focused on the sub-step>"
)

PROBE_USER_TEMPLATE = """TASK:
{task}

WORKING MEMORY (current):
{wm}

CURRENT SUB-STEP:
{step}

Issue your retrieval query now.
"""

WM_INTEGRATION_SYSTEM = (
    "You are a memory-augmented agent. After issuing a query, integrate the new "
    "retrieval results into your working memory. Output a short (<= 120-token) "
    "OBSERVATION block summarizing what you learned that is task-relevant. "
    "Do NOT just copy the chunks. Capture binding constraints and concrete values."
)

WM_INTEGRATION_USER_TEMPLATE = """TASK:
{task}

CURRENT SUB-STEP:
{step}

NEW RETRIEVAL RESULTS (top-{k}):
{hits}

Write the OBSERVATION block now.
"""

# Operator compactor prompt. PRINCIPLE-LEVEL: teaches WHY, the four operations,
# and HOW to decide. NO example-specific recipes.
COMPACTOR_SYSTEM = (
    "You are a working-memory budget controller for a long-running agent. The "
    "agent has a HARD token budget on its working memory — beyond that, content "
    "is forcibly evicted by recency, which loses load-bearing facts. To keep WM "
    "bounded forever no matter how long the agent runs, you compact it.\n\n"
    "WHY this matters:\n"
    "  - Token budgets are real. Model context, latency, and cost all degrade "
    "linearly with WM size. Cheap maintenance prevents catastrophic forgetting.\n"
    "  - Relevance beats recency. Recent ≠ load-bearing. Old ≠ irrelevant. The "
    "controller picks what stays based on the active task and remaining work.\n"
    "  - The agent's job is many steps; your job is to keep the smallest WM that "
    "still lets it finish those steps without re-querying for facts it already "
    "saw.\n\n"
    "OPERATIONS available to you:\n"
    "  - DROP: remove items irrelevant to the active task / remaining work.\n"
    "  - CHUNK: collapse multi-piece related items into a structured summary "
    "(prefer principle-level over verbatim).\n"
    "  - REFRESH: re-mention important items so they remain salient and resist "
    "stale-displacement (especially load-bearing facts surfaced many steps ago).\n"
    "  - OFFLOAD: write to external memory and replace WM content with a short "
    "pointer (only when a re-query would be cheap and reliable).\n\n"
    "HOW to decide:\n"
    "  - Read the TASK and REMAINING WORK. Anything that is a binding constraint, "
    "concrete value (numbers, names, IDs, thresholds, deadlines), or load-bearing "
    "decision must SURVIVE compaction.\n"
    "  - Anything that is filler, restatement of the task, or low-density prose "
    "should be dropped or chunked.\n"
    "  - Use dense structured formats (short bullet lines), not prose paragraphs.\n\n"
    "OUTPUT FORMAT (REQUIRED). Return EXACTLY these four sections, in this order:\n"
    "  ## LOAD-BEARING-FACTS\n"
    "  - <fact 1>\n"
    "  - <fact 2>\n"
    "  ## RECENT-OBSERVATIONS\n"
    "  - <recent observation 1>\n"
    "  ## TASK-CONTEXT\n"
    "  - <terse restatement of the active task and what remains>\n"
    "  ## NOTES\n"
    "  - <any operator notes; can be empty>\n\n"
    "BOUND: the entire compacted WM must be <= 5000 tokens. Aim for ~2-4k tokens. "
    "Do NOT include any other text. Do NOT include preamble or explanation."
)

COMPACTOR_USER_TEMPLATE = """TASK:
{task}

REMAINING WORK (sub-steps not yet done):
{remaining}

CURRENT WORKING MEMORY (~{wm_tokens} tokens, exceeds threshold):
---
{wm}
---

Produce the compacted WM now, in the four-section format. <= 5000 tokens.
"""

ANSWER_SYSTEM = (
    "You are a memory-augmented agent answering final questions about the task. "
    "You may rely ONLY on your working memory. Do NOT speculate; if the WM does "
    "not contain enough to answer, say UNKNOWN.\n\n"
    "Output one numbered answer per question, each on its own line."
)

ANSWER_USER_TEMPLATE = """TASK:
{task}

WORKING MEMORY:
{wm}

QUESTIONS:
{questions}

Answer each question concisely.
"""


# -------------------- LLM helper -------------------- #


async def llm(system: str, user: str, max_tokens: int = 2048) -> str:
    # gpt-5-mini consumes reasoning tokens out of max_completion_tokens. Bump
    # the floor so short outputs aren't starved by reasoning overhead.
    budget = max(max_tokens, 2000)
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=budget,
    )
    return (resp.choices[0].message.content or "").strip()


# -------------------- WM management -------------------- #


@dataclass
class WMState:
    """Append-ordered list of WM blocks. Each block is a string."""

    blocks: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)  # token trace per step

    def text(self) -> str:
        return "\n\n".join(self.blocks)

    def tokens(self) -> int:
        return n_tokens(self.text())


def fifo_truncate(wm: WMState, hard_cap: int) -> tuple[WMState, int]:
    """Baseline: drop oldest blocks until total <= hard_cap. Returns (wm, dropped_n)."""
    dropped = 0
    while wm.blocks and wm.tokens() > hard_cap:
        wm.blocks.pop(0)
        dropped += 1
    return wm, dropped


async def operator_compact(
    wm: WMState, task: str, remaining: list[str], target: int = WM_TARGET
) -> tuple[str, int]:
    """Operator: invoke LLM compactor; return (compacted_text, tokens_after)."""
    user = COMPACTOR_USER_TEMPLATE.format(
        task=task,
        remaining="\n".join(f"  - {s}" for s in remaining) if remaining else "  (none)",
        wm_tokens=wm.tokens(),
        wm=wm.text(),
    )
    out = await llm(COMPACTOR_SYSTEM, user, max_tokens=4096)
    # Hard-enforce ceiling: if compactor went over, fall back to head-truncating its output.
    out_tokens = n_tokens(out)
    if out_tokens > WM_HARD_CAP:
        # Truncate by tokens
        ids = ENC.encode(out)[: WM_HARD_CAP - 500]
        out = ENC.decode(ids)
        out_tokens = n_tokens(out)
    return out, out_tokens


# -------------------- Run loop -------------------- #


def parse_query(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("QUERY:"):
            return line.split(":", 1)[1].strip()
    return text.strip().splitlines()[0][:200] if text.strip() else ""


async def run_case(case: Case, variant: str, log_lines: list[str]) -> dict[str, Any]:
    em = ExternalMemory(case.chunks)
    wm = WMState()
    # seed WM with task statement so the agent knows it
    wm.blocks.append(f"TASK: {case.task}")

    total_raw_retrieved_tokens = 0
    compactor_calls = 0
    total_steps = min(MAX_STEPS, len(case.sub_steps))

    log_lines.append(f"\n=== case={case.case_id} variant={variant} ===")
    log_lines.append(f"corpus_size_tokens={sum(n_tokens(c) for c in case.chunks)}")
    log_lines.append(f"corpus_n_chunks={len(case.chunks)}")
    log_lines.append(f"initial_wm_tokens={wm.tokens()}")

    for step_idx, step in enumerate(case.sub_steps[:total_steps]):
        # --- probe ---
        probe_user = PROBE_USER_TEMPLATE.format(task=case.task, wm=wm.text(), step=step)
        probe_out = await llm(PROBE_SYSTEM, probe_user, max_tokens=200)
        query = parse_query(probe_out)

        # --- retrieve ---
        hits = em.query(query, k=TOP_K)
        hits_text = "\n---\n".join(hits) if hits else "(no hits)"
        retrieved_tokens = n_tokens(hits_text)
        total_raw_retrieved_tokens += retrieved_tokens

        # --- integrate (small obs block, NOT raw chunks) ---
        integ_user = WM_INTEGRATION_USER_TEMPLATE.format(
            task=case.task, step=step, k=TOP_K, hits=hits_text
        )
        observation = await llm(WM_INTEGRATION_SYSTEM, integ_user, max_tokens=400)
        wm.blocks.append(
            f"[step {step_idx + 1}: {step}] QUERY={query}\nOBS: {observation}"
        )

        # ALSO append the raw retrieval to apply pressure on the WM (mimics an
        # agent that captures retrieval into WM, not just summaries).
        wm.blocks.append(f"[step {step_idx + 1} RAW]\n{hits_text}")

        before_tokens = wm.tokens()
        action = "noop"
        dropped = 0
        compacted = 0

        # --- compaction controller ---
        if variant == "baseline":
            if before_tokens > WM_HARD_CAP:
                wm, dropped = fifo_truncate(wm, WM_HARD_CAP)
                action = f"FIFO_drop x{dropped}"
        else:  # operator
            if before_tokens > WM_THRESHOLD:
                remaining = list(case.sub_steps[step_idx + 1 :])
                compacted_text, compacted_tokens = await operator_compact(
                    wm, case.task, remaining, target=WM_TARGET
                )
                wm.blocks = [compacted_text]
                compactor_calls += 1
                compacted = compacted_tokens
                action = f"COMPACT->{compacted_tokens}t"
                # safety net: if compactor failed to shrink, fall back to FIFO on the result
                if wm.tokens() > WM_HARD_CAP:
                    wm, dropped = fifo_truncate(wm, WM_HARD_CAP)
                    action += f"+FIFO_drop x{dropped}"

        after_tokens = wm.tokens()
        wm.history.append(
            {
                "step": step_idx + 1,
                "name": step,
                "query": query,
                "retrieved_tokens": retrieved_tokens,
                "wm_before": before_tokens,
                "wm_after": after_tokens,
                "action": action,
            }
        )
        log_lines.append(
            f"  step {step_idx + 1:2d} ({step[:40]:40s}) "
            f"q={query[:50]!r:52s} "
            f"retr={retrieved_tokens:5d} "
            f"wm {before_tokens:5d}->{after_tokens:5d} action={action}"
        )

        # invariant check: never exceed hard cap going into next step
        assert wm.tokens() <= WM_HARD_CAP + 200, (
            f"WM EXCEEDED HARD CAP: {wm.tokens()} on step {step_idx + 1}"
        )

    # --- final answer step ---
    questions_block = "\n".join(
        f"{i + 1}. {gf.question}" for i, gf in enumerate(case.gold_facts)
    )
    answer_user = ANSWER_USER_TEMPLATE.format(
        task=case.task, wm=wm.text(), questions=questions_block
    )
    answer = await llm(ANSWER_SYSTEM, answer_user, max_tokens=1500)

    # --- score ---
    correct = []
    answer_lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    # crude: for each Q, look for any answer line whose number prefix matches and check answer-keys
    score = 0
    per_q: list[dict[str, Any]] = []
    for i, gf in enumerate(case.gold_facts):
        idx_prefix = f"{i + 1}."
        line_lower = ""
        for ln in answer_lines:
            if ln.startswith(idx_prefix) or ln.startswith(f"{i + 1})"):
                line_lower = ln.lower()
                break
        if not line_lower:
            # fallback: search whole answer
            line_lower = answer.lower()
        hit = any(k.lower() in line_lower for k in gf.answer_keys)
        if hit:
            score += 1
        per_q.append(
            {
                "q": gf.question,
                "answer_keys": gf.answer_keys,
                "hit": hit,
            }
        )

    log_lines.append(f"  final_wm_tokens={wm.tokens()}")
    log_lines.append(f"  total_raw_retrieved_tokens={total_raw_retrieved_tokens}")
    log_lines.append(f"  compactor_calls={compactor_calls}")
    log_lines.append(f"  score={score}/{len(case.gold_facts)}")

    return {
        "case_id": case.case_id,
        "variant": variant,
        "score": score,
        "n_questions": len(case.gold_facts),
        "compactor_calls": compactor_calls,
        "total_raw_retrieved_tokens": total_raw_retrieved_tokens,
        "final_wm_tokens": wm.tokens(),
        "max_wm_tokens": max((h["wm_before"] for h in wm.history), default=wm.tokens()),
        "history": wm.history,
        "answer": answer,
        "per_q": per_q,
    }


async def main() -> None:
    log_lines: list[str] = []
    log_lines.append("WM Compaction Operator Experiment")
    log_lines.append(
        f"WM_HARD_CAP={WM_HARD_CAP} WM_THRESHOLD={WM_THRESHOLD} WM_TARGET={WM_TARGET}"
    )
    log_lines.append(f"MODEL={MODEL}")

    results: list[dict[str, Any]] = []
    for case in CASES:
        for variant in ("baseline", "operator"):
            try:
                res = await run_case(case, variant, log_lines)
                results.append(res)
            except Exception as e:
                log_lines.append(
                    f"  ERROR case={case.case_id} variant={variant}: {e!r}"
                )
                results.append(
                    {
                        "case_id": case.case_id,
                        "variant": variant,
                        "error": repr(e),
                    }
                )

    # summary table
    log_lines.append("\n=== SUMMARY ===")
    log_lines.append(
        f"{'case':30s} {'variant':10s} {'score':>10s} {'compactions':>12s} {'raw_retr':>10s} {'final_wm':>10s} {'max_wm':>8s}"
    )
    for r in results:
        if "error" in r:
            log_lines.append(
                f"{r['case_id']:30s} {r['variant']:10s} ERROR {r['error']}"
            )
            continue
        log_lines.append(
            f"{r['case_id']:30s} {r['variant']:10s} "
            f"{r['score']:>4d}/{r['n_questions']:<4d}    "
            f"{r['compactor_calls']:>10d}   "
            f"{r['total_raw_retrieved_tokens']:>10d} "
            f"{r['final_wm_tokens']:>10d} "
            f"{r['max_wm_tokens']:>8d}"
        )

    # write outputs
    log_path = THIS_DIR / "token_trace.log"
    log_path.write_text("\n".join(log_lines))
    res_path = THIS_DIR / "results.json"
    res_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {log_path}")
    print(f"wrote {res_path}")
    print("\n".join(log_lines[-(2 + len(results)) :]))


if __name__ == "__main__":
    asyncio.run(main())
