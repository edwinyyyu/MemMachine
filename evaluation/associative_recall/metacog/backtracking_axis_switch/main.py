"""Backtracking with axis switching — metacognitive operator test.

ARCHITECTURE
============
(A) BOUNDED WM <= 10_000 tokens (POC ceiling). Architecturally enforced: after
    every step, if WM exceeds the cap, an LLM compactor summarizes older entries
    into a compressed summary block.
(B) EXTERNAL MEMORY — large queryable Python list of fact strings (>= 30k
    tokens raw). Held in plain Python; never injected wholesale into a prompt.
(C) RETRIEVAL ON DEMAND — each step the agent emits a probe; an embedding-free
    lexical-overlap retriever returns the top-K facts. Only those K facts ever
    enter WM.
(D) COMPACTION — LLM-summarize compaction is triggered between steps when WM
    > 10k tokens. Older WM entries get folded into a SUMMARY block; recent
    rounds (snippets verbatim) are preserved.
(E) SUBSTANTIVE TASK — multi-sub-decision plans where cumulative raw memory
    >> 10k tokens.

OPERATOR
========
Backtracking-with-axis-switching: between rounds the LLM judges whether the
last probe ADVANCED or returned LOW_INFO; on LOW_INFO the LLM marks that AXIS
(general decomposition lens, not topic phrase) DEAD and switches to a
structurally different unused axis.

BASELINE: same loop without the axis tracker — next probe is just an LLM
paraphrase from prior context.

Run:  uv run python -m evaluation.associative_recall.metacog.backtracking_axis_switch.main
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
ENV_PATH = THIS_DIR.parent.parent / ".env"
load_dotenv(ENV_PATH)

MODEL = "gpt-5-mini"
N_ROUNDS = 6  # max probe rounds per sub-decision
TOP_K = 5  # snippets per probe (only these enter WM)
WM_TOKEN_CAP = 10_000  # POC bounded working memory ceiling
COMPACTION_TRIGGER = 8_500  # compact when WM exceeds this

CLIENT = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def _approx_tokens(text: str) -> int:
    """Cheap token estimate: ~0.75 tokens per word + small overhead."""
    if not text:
        return 0
    # 4 chars/token is the standard rough heuristic.
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# External memory: gold facts + a large pool of plausible distractors
# ---------------------------------------------------------------------------

# Per-case gold facts indexed by axis. Each case task has multiple sub-decisions
# whose answers live in DIFFERENT axes — operator wins by switching axes; the
# baseline tends to paraphrase a single axis and miss the off-axis golds.

CASES_RAW: list[dict] = [
    {
        "id": "team_dinner",
        "task": (
            "Plan a team dinner for our 5-person engineering squad next week. "
            "Pick a venue, pick a day and time, and decide how long it should run."
        ),
        "gold_subdecisions": [
            {
                "q": "What kind of cuisine / venue should we pick?",
                "axis": "preferences",
                "gold_facts": [
                    "Alice on the engineering team is vegan and avoids dairy products entirely.",
                    "Bob is allergic to shellfish but otherwise eats anything.",
                    "Carla on the engineering team prefers Italian cuisine over Asian food.",
                ],
            },
            {
                "q": "Which day of the week to schedule the team dinner on?",
                "axis": "timing",
                "gold_facts": [
                    "Carla has a recurring physical-therapy appointment every Tuesday at 6pm.",
                    "Dan flies out Friday morning at 7am for a conference.",
                    "Eve teaches an evening class on Wednesdays from 6pm to 9pm.",
                ],
            },
            {
                "q": "How long should the dinner run?",
                "axis": "history",
                "gold_facts": [
                    "Last quarter's team dinner ran 3 hours; people complained it was too long.",
                    "HR reminded us that team dinners should not exceed 2.5 hours per company policy.",
                ],
            },
        ],
        # Per-case distractor pool — used to inflate the external memory.
        "distractor_topics": [
            "company laptop refresh policy",
            "Q3 roadmap",
            "office plant care schedule",
            "vendor contracts",
            "401k matching changes",
            "open-source licensing",
            "the broken espresso machine",
            "the new bike rack",
            "expense-reimbursement policy",
            "annual hackathon sponsorship",
            "the building fire-drill schedule",
        ],
    },
    {
        "id": "press_release",
        "task": (
            "Draft and send a press release announcing our new product line. "
            "Decide content, decide approvals, decide timing."
        ),
        "gold_subdecisions": [
            {
                "q": "When should we publish the press release?",
                "axis": "timing",
                "gold_facts": [
                    "The competing product from Acme launches publicly on May 15.",
                    "Our CEO is on vacation May 10-20 and unreachable for direct quotes.",
                    "Industry trade show DevWorld runs May 12-14 and absorbs press attention.",
                ],
            },
            {
                "q": "What approvals or sign-offs are needed before sending?",
                "axis": "stakeholders",
                "gold_facts": [
                    "Legal flagged that any product claims need their explicit sign-off before publication.",
                    "Our PR firm Northwind handles all external announcements; their account lead is Priya.",
                    "Investor relations must review any release that mentions revenue or growth.",
                ],
            },
            {
                "q": "What content must we be careful to AVOID in the release?",
                "axis": "exceptions",
                "gold_facts": [
                    "Last press release misnamed the product and required a humiliating retraction.",
                    "An NDA with our manufacturing partner restricts disclosing production volume.",
                    "Investor-relations rules forbid forward-looking revenue claims pre-earnings.",
                ],
            },
        ],
        "distractor_topics": [
            "office snack budget",
            "VP travel plans",
            "the new logo redesign",
            "a hiring freeze rumor",
            "open-plan-office complaints",
            "the broken HVAC unit",
            "company-wide diversity training",
            "annual benefits enrollment window",
            "internal Slack-vs-Teams debate",
            "the security-awareness training rollout",
        ],
    },
    {
        "id": "open_source_release",
        "task": (
            "Cut a 1.0 release of our open-source library and announce it to the community. "
            "Decide blockers, decide whom to notify, decide announcement timing."
        ),
        "gold_subdecisions": [
            {
                "q": "What blockers must be resolved BEFORE we tag 1.0?",
                "axis": "exceptions",
                "gold_facts": [
                    "There's an unresolved deadlock bug in the threading module reported two weeks ago.",
                    "A maintainer left a TODO 'breaking change before 1.0' in the auth module that nobody addressed.",
                    "Two contributors have outstanding PRs they expected to land before 1.0.",
                ],
            },
            {
                "q": "Whom should we notify, and when?",
                "axis": "stakeholders",
                "gold_facts": [
                    "Our biggest enterprise user, ZetaCorp, asked to be notified 48 hours before any 1.0 cut.",
                    "Lead maintainer prefers we batch release announcements with a blog post, not a raw GitHub release.",
                    "The package's Debian maintainer asked to be looped in before the upload.",
                ],
            },
            {
                "q": "How should we time the public announcement?",
                "axis": "timing",
                "gold_facts": [
                    "PyCon US is next Friday; releasing during the conference would maximize visibility.",
                    "Our docs are hosted on Read the Docs, which has a known outage scheduled Wednesday.",
                    "We promised the Python Weekly newsletter a one-week heads-up before the release.",
                ],
            },
        ],
        "distractor_topics": [
            "the Slack-archive incident",
            "the Sentry quota overage",
            "the contributor licensing agreement debate",
            "GitHub Actions billing changes",
            "the package logo redesign",
            "the conf-room booking system",
            "an unrelated CI vendor migration",
            "the 0.7 RC user-survey results",
            "open-collective grant applications",
            "documentation theme bikeshed",
        ],
    },
]


def _gen_distractor_for_topic(topic: str, case_id: str, rng: random.Random) -> str:
    """Synthesize a plausible distractor sentence for a topic.

    These are intentionally on-topic-sounding but bear NO relevance to the
    case's gold sub-decisions. Lexically they share enough generic vocabulary
    with axis-A probes to compete for retrieval slots, which is what makes
    axis-switching valuable.
    """
    templates = [
        "Last week's discussion of {t} stretched over multiple meetings without consensus.",
        "An email from finance about {t} arrived with several spreadsheet attachments.",
        "Several team members raised concerns about {t} in the all-hands Q&A.",
        "The internal wiki page on {t} was last updated more than a year ago and is stale.",
        "A Slack thread on {t} ran to 200+ messages with widely conflicting opinions.",
        "The budget review covered {t} but action items were assigned to nobody.",
        "An external auditor asked about {t} during the procedural review last quarter.",
        "Documentation for {t} lives in three places and disagrees with itself.",
        "A retro item filed against {t} was closed without resolution months ago.",
        "An offsite breakout session on {t} produced sticky notes but no decisions.",
        "Our vendor for {t} renewed its contract with mostly cosmetic changes to the SLA.",
        "Quarterly OKRs touched {t} only at the level of an indirect dependency.",
        "Engineering sent a memo about {t} that HR found ambiguous and asked to revise.",
        "A working group on {t} was formed but met only once before disbanding.",
        "Recent metrics on {t} show no statistically meaningful trend either way.",
    ]
    template = rng.choice(templates)
    return f"[{case_id}] {template.format(t=topic)}"


def build_external_memory(
    case: dict, target_tokens: int = 35_000, seed: int = 0
) -> list[str]:
    """Build a large external memory list: gold facts + many distractors.

    Returns a flat list of fact strings. Token count >= target_tokens (raw).
    """
    rng = random.Random(seed + hash(case["id"]) % 2**31)
    facts: list[str] = []
    # gold first
    for sub in case["gold_subdecisions"]:
        for f in sub["gold_facts"]:
            facts.append(f)
    # distractors generated from the per-case topics (paraphrastic to push lexical overlap)
    topics = case["distractor_topics"]
    while sum(_approx_tokens(f) for f in facts) < target_tokens:
        topic = rng.choice(topics)
        facts.append(_gen_distractor_for_topic(topic, case["id"], rng))
    rng.shuffle(facts)
    return facts


# ---------------------------------------------------------------------------
# Retrieval — lexical-overlap top-K (small, deterministic, dependency-light)
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "which",
    "should",
    "are",
    "our",
    "you",
    "your",
    "have",
    "has",
    "had",
    "but",
    "not",
    "can",
    "will",
    "would",
    "into",
    "out",
    "over",
    "all",
    "any",
    "some",
    "more",
    "less",
    "than",
    "about",
    "very",
    "much",
    "many",
    "few",
    "they",
    "them",
    "their",
    "his",
    "her",
    "him",
    "she",
    "team",
    "next",
}


def _toks(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if len(w) >= 3 and w not in _STOP}


def retrieve(probe: str, memory: list[str], top_k: int = TOP_K) -> list[str]:
    pt = _toks(probe)
    if not pt:
        return []
    scored: list[tuple[float, str]] = []
    for fact in memory:
        ft = _toks(fact)
        if not ft:
            continue
        overlap = len(pt & ft)
        if overlap == 0:
            continue
        score = overlap / (1 + 0.25 * len(ft))
        scored.append((score, fact))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [f for _, f in scored[:top_k]]


# ---------------------------------------------------------------------------
# Working memory: bounded, with LLM-compaction
# ---------------------------------------------------------------------------


@dataclass
class WMEntry:
    kind: str  # "task", "subq", "round", "summary"
    text: str

    @property
    def tokens(self) -> int:
        return _approx_tokens(self.text)


class WorkingMemory:
    def __init__(self, cap: int = WM_TOKEN_CAP) -> None:
        self.cap = cap
        self.entries: list[WMEntry] = []
        self.compactions: int = 0
        self.trace: list[dict] = []

    def add(self, kind: str, text: str) -> None:
        self.entries.append(WMEntry(kind=kind, text=text))
        self._record("add")

    def total_tokens(self) -> int:
        return sum(e.tokens for e in self.entries)

    def render(self) -> str:
        return "\n\n".join(f"[{e.kind}] {e.text}" for e in self.entries)

    def _record(self, op: str) -> None:
        self.trace.append(
            {
                "op": op,
                "n_entries": len(self.entries),
                "tokens": self.total_tokens(),
                "compactions_so_far": self.compactions,
            }
        )

    async def maybe_compact(self) -> None:
        """If WM exceeds COMPACTION_TRIGGER, summarize older `round` entries."""
        if self.total_tokens() <= COMPACTION_TRIGGER:
            return
        # Keep task + subq + the LAST round entry verbatim; summarize the rest.
        keep_kinds = {"task", "subq"}
        head = [e for e in self.entries if e.kind in keep_kinds]
        rest = [e for e in self.entries if e.kind not in keep_kinds]
        if len(rest) <= 1:
            return  # nothing to compact yet
        to_summarize = rest[:-1]
        keep_tail = rest[-1:]

        merged = "\n\n".join(f"[{e.kind}] {e.text}" for e in to_summarize)
        compaction_user = (
            "Compress the following retrieval-loop history into <= 200 words. "
            "Preserve concrete facts (names, dates, numbers) and which axes were "
            "tried with what result. Drop verbose probes/snippets that did not "
            "advance the sub-decision.\n\n"
            f"HISTORY:\n{merged}"
        )
        try:
            summary = await _chat(
                "You compress agent working memory while keeping load-bearing facts.",
                compaction_user,
            )
        except Exception as exc:
            summary = f"(compaction failed: {exc}; entries dropped)"
        self.compactions += 1
        self.entries = (
            head + [WMEntry(kind="summary", text=summary.strip())] + keep_tail
        )
        self._record("compact")
        # Hard truncate as a last resort
        while self.total_tokens() > self.cap and len(self.entries) > 2:
            # drop oldest non-task/subq/summary
            for i, e in enumerate(self.entries):
                if e.kind not in {"task", "subq"}:
                    del self.entries[i]
                    break
            self._record("truncate")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BUDGET_NOTE = (
    f"You operate under a HARD working-memory cap of ~{WM_TOKEN_CAP} tokens. "
    "Older history may be replaced by a [summary] block; that compression is "
    "intentional. Keep probes short (<= 12 words)."
)

BASELINE_SYSTEM = f"""You are a memory-augmented planning agent.

{BUDGET_NOTE}

You have access to an external memory store via short text probes. Each probe
returns the top-{TOP_K} most-related facts (NOT the full memory).

For the active SUB-DECISION you will iteratively emit one probe per round.
After each probe you'll see the snippets that came back, then you'll emit the
next probe. Goal: surface facts needed to make the sub-decision.

Output ONLY a single line of the form:
PROBE: <short probe text, <= 12 words>
"""

OPERATOR_SYSTEM = f"""You are a memory-augmented planning agent equipped with a
metacognitive BACKTRACKING-WITH-AXIS-SWITCHING operator.

{BUDGET_NOTE}

You have access to an external memory store via short text probes. Each probe
returns the top-{TOP_K} most-related facts (NOT the full memory).

WHY NAIVE ITERATION FAILS
-------------------------
Naive iterative probing tends to recur on a single CONCEPT-AXIS. If a probe on
"dietary needs" returned nothing useful, asking "food preferences" is just a
PARAPHRASE — same axis, same retrieval neighborhood, same dead end. Genuine
reformulation requires a STRUCTURALLY DIFFERENT angle on the sub-decision.

AXES — GENERAL DECOMPOSITION LENSES (not topic-specific keywords)
----------------------------------------------------------------
You may use axis labels from this lens-set (or invent others in the same
spirit):
  - entities involved
  - timing / schedule / deadlines
  - stakeholders / approvers
  - constraints / budgets / policies
  - environment / location / venue
  - goals / desired outcomes
  - conflicts / risks / blockers
  - preferences / tastes
  - history / past patterns / what happened before
  - exceptions / edge cases / things to avoid

NOTE: "food preferences" and "dietary needs" are the SAME axis (preferences).
"When are people free" is a DIFFERENT axis (timing). Axis-switching is about
STRUCTURAL ANGLE, not topical synonyms.

EACH ROUND, OUTPUT THIS STRICT FORMAT
-------------------------------------
JUDGEMENT: ADVANCED | LOW_INFO | INITIAL
DEAD_AXIS: <axis label or NONE>
NEXT_AXIS: <a SHORT axis label, e.g. "timing", "stakeholders", "exceptions">
PROBE: <short probe text under 12 words framed FROM the next axis>

How to judge LOW_INFO: snippets are off-topic, repeat earlier hits, or do not
bear on the sub-decision. When LOW_INFO, mark the just-tried axis DEAD and
pick a structurally DIFFERENT unused axis from the lens-set above.
"""


def _format_snippets(snips: list[str]) -> str:
    if not snips:
        return "(no snippets returned)"
    return "\n".join(f"- {s}" for s in snips)


def _format_tracker(tracker: list[dict]) -> str:
    if not tracker:
        return "(empty — first round)"
    lines = []
    for entry in tracker:
        status = entry.get("status", "open")
        snip_preview = "; ".join(s[:60] for s in entry["snippets"][:2]) or "(no hits)"
        lines.append(
            f'- axis={entry["axis"]} [{status}] probe="{entry["probe"]}" -> {snip_preview}'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_CHAT_SEM = asyncio.Semaphore(6)


async def _chat(system: str, user: str) -> str:
    async with _CHAT_SEM:
        for attempt in range(3):
            try:
                resp = await CLIENT.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 + attempt * 2)
    raise RuntimeError("unreachable")


_PROBE_RE = re.compile(r"^\s*PROBE:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_AXIS_RE = re.compile(r"^\s*NEXT_AXIS:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_DEAD_RE = re.compile(r"^\s*DEAD_AXIS:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_JUDGE_RE = re.compile(r"^\s*JUDGEMENT:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def _parse_probe(text: str) -> str:
    m = _PROBE_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: last non-empty line
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    surfaced: set[str] = field(default_factory=set)
    rounds: list[dict] = field(default_factory=list)
    wm_trace: list[dict] = field(default_factory=list)
    compactions: int = 0


async def run_baseline(task: str, subq: str, memory: list[str]) -> RunResult:
    out = RunResult()
    wm = WorkingMemory()
    wm.add("task", f"TASK: {task}")
    wm.add("subq", f"SUB-DECISION: {subq}")

    last_snips: list[str] = []
    for r in range(N_ROUNDS):
        await wm.maybe_compact()
        if r == 0:
            user = wm.render() + "\n\nFirst round — emit your initial PROBE."
        else:
            user = (
                wm.render()
                + f"\n\nRound {r + 1}. Last probe's snippets:\n"
                + _format_snippets(last_snips)
                + "\n\nEmit your next PROBE."
            )
        text = await _chat(BASELINE_SYSTEM, user)
        probe = _parse_probe(text)
        snips = retrieve(probe, memory)
        out.surfaced.update(snips)
        out.rounds.append({"round": r + 1, "probe": probe, "snippets": snips})
        wm.add(
            "round", f"R{r + 1} PROBE: {probe}\nSNIPPETS:\n" + _format_snippets(snips)
        )
        last_snips = snips
    out.wm_trace = wm.trace
    out.compactions = wm.compactions
    return out


async def run_operator(task: str, subq: str, memory: list[str]) -> RunResult:
    out = RunResult()
    wm = WorkingMemory()
    wm.add("task", f"TASK: {task}")
    wm.add("subq", f"SUB-DECISION: {subq}")

    tracker: list[dict] = []
    last_snips: list[str] = []
    last_axis: str | None = None
    last_probe: str | None = None

    for r in range(N_ROUNDS):
        await wm.maybe_compact()
        user_lines = [
            wm.render(),
            "",
            "TRACKER (axes tried so far):",
            _format_tracker(tracker),
            "",
        ]
        if r == 0:
            user_lines += [
                "First round. Pick an initial axis and probe.",
                "JUDGEMENT must be INITIAL; DEAD_AXIS must be NONE.",
            ]
        else:
            user_lines += [
                f'Last probe (axis={last_axis}): "{last_probe}"',
                "Last probe's snippets:",
                _format_snippets(last_snips),
                "",
                "Judge whether the last probe ADVANCED or was LOW_INFO. If LOW_INFO,"
                " mark its axis DEAD and switch to a structurally different unused axis.",
            ]
        text = await _chat(OPERATOR_SYSTEM, "\n".join(user_lines))
        judge_m = _JUDGE_RE.search(text)
        dead_m = _DEAD_RE.search(text)
        axis_m = _AXIS_RE.search(text)
        probe = _parse_probe(text)
        judgement = (
            judge_m.group(1).strip()
            if judge_m
            else ("INITIAL" if r == 0 else "UNKNOWN")
        )
        dead_axis = dead_m.group(1).strip() if dead_m else "NONE"
        next_axis = axis_m.group(1).strip() if axis_m else "unspecified"

        if r > 0 and tracker:
            prev = tracker[-1]
            if "DEAD" in dead_axis.upper() or judgement.upper() == "LOW_INFO":
                prev["status"] = "dead"
            elif judgement.upper() == "ADVANCED":
                prev["status"] = "advanced"

        snips = retrieve(probe, memory)
        out.surfaced.update(snips)
        tracker.append(
            {"axis": next_axis, "probe": probe, "snippets": snips, "status": "open"}
        )
        out.rounds.append(
            {
                "round": r + 1,
                "axis": next_axis,
                "dead_axis": dead_axis,
                "judgement": judgement,
                "probe": probe,
                "snippets": snips,
            }
        )
        wm.add(
            "round",
            f"R{r + 1} axis={next_axis} judge={judgement} dead={dead_axis}\n"
            f"PROBE: {probe}\nSNIPPETS:\n" + _format_snippets(snips),
        )
        last_snips = snips
        last_axis = next_axis
        last_probe = probe
    out.wm_trace = wm.trace
    out.compactions = wm.compactions
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_subq(surfaced: Iterable[str], gold: list[str]) -> tuple[int, int]:
    surfaced_set = set(surfaced)
    hits = sum(1 for g in gold if g in surfaced_set)
    return hits, len(gold)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_case(case: dict, memory: list[str]) -> dict:
    rows = []
    for sub in case["gold_subdecisions"]:
        baseline = await run_baseline(case["task"], sub["q"], memory)
        operator = await run_operator(case["task"], sub["q"], memory)
        b_hits, n_gold = score_subq(baseline.surfaced, sub["gold_facts"])
        o_hits, _ = score_subq(operator.surfaced, sub["gold_facts"])
        rows.append(
            {
                "case": case["id"],
                "subq": sub["q"],
                "axis": sub["axis"],
                "n_gold": n_gold,
                "baseline_hits": b_hits,
                "operator_hits": o_hits,
                "baseline_compactions": baseline.compactions,
                "operator_compactions": operator.compactions,
                "baseline_rounds": baseline.rounds,
                "operator_rounds": operator.rounds,
                "baseline_wm_trace": baseline.wm_trace,
                "operator_wm_trace": operator.wm_trace,
            }
        )
    return {"case": case["id"], "rows": rows}


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not loaded — check evaluation/associative_recall/.env"
        )

    # Build external memories — > 30k tokens each
    memories: dict[str, list[str]] = {}
    raw_token_counts: dict[str, int] = {}
    for case in CASES_RAW:
        mem = build_external_memory(case, target_tokens=35_000, seed=42)
        memories[case["id"]] = mem
        raw_token_counts[case["id"]] = sum(_approx_tokens(f) for f in mem)
        print(
            f"[memory] case={case['id']} facts={len(mem)} ~tokens={raw_token_counts[case['id']]}"
        )

    t0 = time.time()
    results = await asyncio.gather(*(run_case(c, memories[c["id"]]) for c in CASES_RAW))
    elapsed = time.time() - t0

    out_path = THIS_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))

    # ---- token-trace logs ----
    trace_path = THIS_DIR / "token_trace.json"
    trace_blob = {
        "wm_cap_tokens": WM_TOKEN_CAP,
        "compaction_trigger": COMPACTION_TRIGGER,
        "raw_external_memory_tokens": raw_token_counts,
        "per_case": [],
    }
    for case_result in results:
        per_case = {"case": case_result["case"], "subq_traces": []}
        for row in case_result["rows"]:
            per_case["subq_traces"].append(
                {
                    "subq": row["subq"],
                    "axis": row["axis"],
                    "baseline_compactions": row["baseline_compactions"],
                    "operator_compactions": row["operator_compactions"],
                    "baseline_wm_peak_tokens": max(
                        (t["tokens"] for t in row["baseline_wm_trace"]), default=0
                    ),
                    "operator_wm_peak_tokens": max(
                        (t["tokens"] for t in row["operator_wm_trace"]), default=0
                    ),
                }
            )
        trace_blob["per_case"].append(per_case)
    trace_path.write_text(json.dumps(trace_blob, indent=2))

    # ---- aggregate ----
    print(f"\n{'case':<22} {'subq-axis':<14} {'gold':>5} {'base':>5} {'oper':>5}")
    print("-" * 60)
    tot_n = tot_b = tot_o = 0
    axis_b: dict[str, list[int]] = {}
    axis_o: dict[str, list[int]] = {}
    axis_n: dict[str, int] = {}
    peak_baseline = peak_operator = 0
    total_compactions_b = total_compactions_o = 0
    for case_result in results:
        for row in case_result["rows"]:
            print(
                f"{row['case']:<22} {row['axis']:<14} "
                f"{row['n_gold']:>5} {row['baseline_hits']:>5} {row['operator_hits']:>5}"
            )
            tot_n += row["n_gold"]
            tot_b += row["baseline_hits"]
            tot_o += row["operator_hits"]
            axis_b.setdefault(row["axis"], []).append(row["baseline_hits"])
            axis_o.setdefault(row["axis"], []).append(row["operator_hits"])
            axis_n[row["axis"]] = axis_n.get(row["axis"], 0) + row["n_gold"]
            peak_baseline = max(
                peak_baseline,
                max((t["tokens"] for t in row["baseline_wm_trace"]), default=0),
            )
            peak_operator = max(
                peak_operator,
                max((t["tokens"] for t in row["operator_wm_trace"]), default=0),
            )
            total_compactions_b += row["baseline_compactions"]
            total_compactions_o += row["operator_compactions"]
    print("-" * 60)
    print(f"{'TOTAL':<22} {'':<14} {tot_n:>5} {tot_b:>5} {tot_o:>5}")
    print(f"recall: baseline={tot_b / tot_n:.3f}  operator={tot_o / tot_n:.3f}")
    print(
        f"\nWM peak tokens: baseline={peak_baseline} operator={peak_operator}  cap={WM_TOKEN_CAP}"
    )
    print(f"Compactions: baseline={total_compactions_b} operator={total_compactions_o}")
    print(f"Elapsed: {elapsed:.1f}s")

    print("\nPer-axis recall:")
    for ax in sorted(axis_n):
        b = sum(axis_b[ax])
        o = sum(axis_o[ax])
        n = axis_n[ax]
        print(f"  {ax:<14} n={n:<3} base={b / n:.2f}  oper={o / n:.2f}")

    summary = {
        "total_gold": tot_n,
        "baseline_hits": tot_b,
        "operator_hits": tot_o,
        "baseline_recall": tot_b / tot_n,
        "operator_recall": tot_o / tot_n,
        "wm_cap_tokens": WM_TOKEN_CAP,
        "wm_peak_baseline": peak_baseline,
        "wm_peak_operator": peak_operator,
        "compactions_baseline": total_compactions_b,
        "compactions_operator": total_compactions_o,
        "raw_external_memory_tokens": raw_token_counts,
        "by_axis": {
            ax: {
                "n": axis_n[ax],
                "baseline_recall": sum(axis_b[ax]) / axis_n[ax],
                "operator_recall": sum(axis_o[ax]) / axis_n[ax],
            }
            for ax in axis_n
        },
        "elapsed_seconds": elapsed,
    }
    (THIS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
