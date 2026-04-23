"""Timestamp-aware temporal scoring substrate for LongMemEval.

Uses LongMemEval's per-session `haystack_dates` and per-query `question_date`
(metadata not available on LoCoMo) as a pure-metadata scoring channel to
attack the temporal-reasoning ceiling.

Two-part substrate:

1) LLM parser (gpt-5-mini, one call per query, cached) produces a structured
   temporal constraint:
       {
         "has_temporal_constraint": bool,
         "temporal_type": "before"|"after"|"during"|
                          "relative-past"|"relative-future"|null,
         "reference_date": "YYYY-MM-DD"|null,
         "relative_window_days": int|null,
         "uses_question_date_as_reference": bool,
       }
   If parsing fails, falls back to no-constraint (safe).

2) Temporal compatibility scorer: given a turn's session date, the question
   date, and the parsed constraint, return a score in [0,1] (or -1 for hard
   incompatible).

Integration is done by `tsscore_eval.py`, which applies the scorer to v2f's
candidate pool (confidence-gated displacement pattern).

Cache: `tsscore_llm_cache.json` (reads from lmehard cache as warm-start but
writes only here).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

CACHE_DIR = Path(__file__).resolve().parent / "cache"
TSSCORE_LLM_CACHE = CACHE_DIR / "tsscore_llm_cache.json"
LMEHARD_LLM_CACHE = CACHE_DIR / "lmehard_llm_cache.json"

MODEL = "gpt-5-mini"


# ---------------------------------------------------------------------------
# Temporal constraint structure
# ---------------------------------------------------------------------------
@dataclass
class TemporalConstraint:
    has_temporal_constraint: bool = False
    temporal_type: str | None = None
    reference_date: str | None = None  # YYYY-MM-DD
    relative_window_days: int | None = None
    uses_question_date_as_reference: bool = False
    raw_llm_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw_llm_response", None)
        return d


# ---------------------------------------------------------------------------
# LLM parser
# ---------------------------------------------------------------------------
TEMPORAL_PARSE_PROMPT = """\
Task: Parse a user question into a temporal-constraint JSON that describes \
when, on the past-conversation timeline, the event or information the \
question asks about was most likely RECORDED (mentioned in a chat \
session). The constraint will score turns whose session dates fall in a \
matching time range.

Key insight: users talk about events soon after they happen. So if a \
question says "N days/weeks/months ago I did X", the turn where the user \
mentioned doing X is almost certainly in a session dated about N days \
before question_date. Fire the constraint in this case.

Output ONE JSON object, no prose, no code fences. Schema:
{{
  "has_temporal_constraint": true|false,
  "temporal_type": "before"|"after"|"during"|"relative-past"|"relative-future"|null,
  "reference_date": "YYYY-MM-DD" or null,
  "relative_window_days": integer or null,
  "uses_question_date_as_reference": true|false
}}

FIRE has_temporal_constraint=true when:

- "N days/weeks/months ago did I X" → relative-past (window = N days).
- "last Saturday / last Tuesday / last week" → during, \
reference_date = question_date minus 7 days, window = 3.
- "last month / in the past month" → relative-past, window = 30.
- "recently / lately" → relative-past, window = 30.
- "a month ago / a week ago" → relative-past, window = 30 or 7.
- "on Valentine's day" → during, reference_date = that Feb 14, window=2.
- "in December" / "in June" → during, reference_date=1st of month, \
window = 15.
- "last Saturday" → during with reference_date = the most recent Saturday \
strictly before question_date, window=2.

DO NOT FIRE has_temporal_constraint for:
- "Which happened first, X or Y?" — retrieval needs BOTH events, not a \
window. Return false.
- "How long have I been doing X" / "How long had I been doing X when Y" — \
needs both the start and end events spanning potentially all of history. \
Return false.
- "How old was I when X happened" — needs the event mention, which could \
be anywhere in history. Return false.
- "Which X did I start using most recently" — needs comparisons across \
all events. Return false.
- "What was the first Y I had after X" — ordering question, return false.
- "How many days/months had passed between X and Y" or "since X" — needs \
both events, can be anywhere. Return false.

Special clarification — these DO fire (retrieval window = around \
question_date - N):
- "N days/weeks/months ago did I X" — mention is in session ~N days ago.
- "last Saturday / last Tuesday / last week / last month / recently" — \
mention in recent session.
- "a month ago / a couple weeks ago" — recent relative window.

Important: when the question contains both an event and a relative time \
phrase (e.g. "what did I do with Rachel on the Wednesday two months ago"), \
fire with the relative phrase.

Date arithmetic: today is the given question_date. Use the Gregorian \
calendar. For "N weeks ago" use 7*N days. For "N months ago" use 30*N \
days. For "last <weekday>" use the most recent past weekday before \
question_date, window=2.

Examples:

Q: "What did I do with Rachel on the Wednesday two months ago?"
question_date=2023-05-30
{{
  "has_temporal_constraint": true,
  "temporal_type": "during",
  "reference_date": "2023-03-29",
  "relative_window_days": 3,
  "uses_question_date_as_reference": false
}}

Q: "How many weeks ago did I start using the cashback app 'Ibotta'?"
question_date=2023-06-10
{{
  "has_temporal_constraint": true,
  "temporal_type": "relative-past",
  "reference_date": null,
  "relative_window_days": 30,
  "uses_question_date_as_reference": true
}}

Q: "How many weeks ago did I attend a bird watching workshop at the local \
Audubon society?"
question_date=2023-06-10
{{
  "has_temporal_constraint": true,
  "temporal_type": "relative-past",
  "reference_date": null,
  "relative_window_days": 30,
  "uses_question_date_as_reference": true
}}

Q: "Who did I meet with during the lunch last Tuesday?"
question_date=2023-06-10
{{
  "has_temporal_constraint": true,
  "temporal_type": "during",
  "reference_date": "2023-06-06",
  "relative_window_days": 2,
  "uses_question_date_as_reference": false
}}

Q: "How many months have passed since I last visited a museum with a friend?"
question_date=2023-06-10
{{
  "has_temporal_constraint": false,
  "temporal_type": null,
  "reference_date": null,
  "relative_window_days": null,
  "uses_question_date_as_reference": false
}}

Q: "Which event happened first, the meeting with Rachel or the pride parade?"
question_date=2023-06-10
{{
  "has_temporal_constraint": false,
  "temporal_type": null,
  "reference_date": null,
  "relative_window_days": null,
  "uses_question_date_as_reference": false
}}

Q: "What was the airline that I flied with on Valentine's day?"
question_date=2023-06-10
{{
  "has_temporal_constraint": true,
  "temporal_type": "during",
  "reference_date": "2023-02-14",
  "relative_window_days": 3,
  "uses_question_date_as_reference": false
}}

Q: "How old was I when I moved to the United States?"
question_date=2023-06-10
{{
  "has_temporal_constraint": false,
  "temporal_type": null,
  "reference_date": null,
  "relative_window_days": null,
  "uses_question_date_as_reference": false
}}

Q: "What was the social media activity I participated 5 days ago?"
question_date=2023-04-10
{{
  "has_temporal_constraint": true,
  "temporal_type": "relative-past",
  "reference_date": null,
  "relative_window_days": 5,
  "uses_question_date_as_reference": true
}}

Now your turn.

question_date: {question_date}
question: {question}

JSON:"""


class TemporalParseCache:
    """Dedicated tsscore parse cache.

    Warm-starts from LME-hard llm cache (read-only), writes only to
    `tsscore_llm_cache.json`.
    """

    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        # Warm-start from the shared lmehard cache (read-only).
        if LMEHARD_LLM_CACHE.exists():
            try:
                with open(LMEHARD_LLM_CACHE) as f:
                    for k, v in json.load(f).items():
                        if v:
                            self._cache[k] = v
            except (OSError, json.JSONDecodeError):
                pass
        # Then dedicated cache (may override).
        if TSSCORE_LLM_CACHE.exists():
            try:
                with open(TSSCORE_LLM_CACHE) as f:
                    for k, v in json.load(f).items():
                        if v:
                            self._cache[k] = v
            except (OSError, json.JSONDecodeError):
                pass
        self._new: dict[str, str] = {}

    def _key(self, model: str, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"\0")
        h.update(prompt.encode())
        return h.hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(self._key(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        k = self._key(model, prompt)
        self._cache[k] = response
        self._new[k] = response

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if TSSCORE_LLM_CACHE.exists():
            try:
                with open(TSSCORE_LLM_CACHE) as f:
                    existing = json.load(f)
            except (OSError, json.JSONDecodeError):
                existing = {}
        existing.update(self._new)
        tmp = TSSCORE_LLM_CACHE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(TSSCORE_LLM_CACHE)
        self._new = {}


def _extract_json_block(text: str) -> str | None:
    # Find first {...} JSON object substring via brace matching.
    if not text:
        return None
    text = text.strip()
    # Strip code fences if any.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def parse_temporal_constraint(
    client: OpenAI,
    cache: TemporalParseCache,
    question: str,
    question_date: dt.date,
    model: str = MODEL,
) -> TemporalConstraint:
    """Ask gpt-5-mini to parse a structured temporal constraint."""
    prompt = TEMPORAL_PARSE_PROMPT.format(
        question_date=question_date.isoformat(),
        question=question,
    )
    raw = cache.get(model, prompt)
    if raw is None or not raw.strip():
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:  # pragma: no cover - network failures
            return TemporalConstraint(raw_llm_response=f"ERROR: {e}")
        if raw.strip():
            cache.put(model, prompt, raw)

    blk = _extract_json_block(raw)
    if blk is None:
        return TemporalConstraint(raw_llm_response=raw)
    try:
        obj = json.loads(blk)
    except json.JSONDecodeError:
        return TemporalConstraint(raw_llm_response=raw)

    tc = TemporalConstraint(
        has_temporal_constraint=bool(obj.get("has_temporal_constraint", False)),
        temporal_type=obj.get("temporal_type"),
        reference_date=obj.get("reference_date"),
        relative_window_days=obj.get("relative_window_days"),
        uses_question_date_as_reference=bool(
            obj.get("uses_question_date_as_reference", False)
        ),
        raw_llm_response=raw,
    )
    # Normalize: if temporal_type is set but constraint flag wasn't, treat as
    # constraint. If types/structure are wrong, fall back.
    if tc.temporal_type not in (
        None, "before", "after", "during",
        "relative-past", "relative-future",
    ):
        return TemporalConstraint(raw_llm_response=raw)
    if tc.temporal_type is None:
        tc.has_temporal_constraint = False
    if tc.relative_window_days is not None:
        try:
            tc.relative_window_days = int(tc.relative_window_days)
        except (TypeError, ValueError):
            tc.relative_window_days = None
    # Validate reference_date format.
    if tc.reference_date:
        try:
            dt.date.fromisoformat(tc.reference_date)
        except (TypeError, ValueError):
            tc.reference_date = None
    return tc


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


def parse_lme_date(s: str) -> dt.date | None:
    """Parse LME date strings like '2023/05/20 (Sat) 02:21' → date."""
    if not s:
        return None
    m = _DATE_RE.search(s)
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Temporal compatibility scorer
# ---------------------------------------------------------------------------
def temporal_score(
    turn_date: dt.date | None,
    constraint: TemporalConstraint,
    question_date: dt.date,
) -> float:
    """Return a score in [0, 1] (1 = fully compatible, 0 = incompatible).

    If constraint is absent, returns 0 (caller should skip applying this
    channel when the question has no temporal constraint).
    """
    if not constraint.has_temporal_constraint:
        return 0.0
    if turn_date is None:
        return 0.0

    t = constraint.temporal_type
    if t == "before":
        ref = (
            dt.date.fromisoformat(constraint.reference_date)
            if constraint.reference_date else None
        )
        if ref is None:
            return 0.0
        return 1.0 if turn_date < ref else 0.0

    if t == "after":
        ref = (
            dt.date.fromisoformat(constraint.reference_date)
            if constraint.reference_date else None
        )
        if ref is None:
            return 0.0
        return 1.0 if turn_date > ref else 0.0

    if t == "during":
        ref = (
            dt.date.fromisoformat(constraint.reference_date)
            if constraint.reference_date else None
        )
        if ref is None:
            return 0.0
        w = constraint.relative_window_days or 7
        delta = abs((turn_date - ref).days)
        if delta <= w:
            return 1.0
        # Linear decay to 0 over 2x window.
        decay = max(0.0, 1.0 - (delta - w) / max(w, 1))
        return decay

    if t == "relative-past":
        # Interpret relative_window_days as the expected age of the event
        # in days. Score 1.0 in a symmetric half-band around
        # (question_date - N) with half-width max(N/2, 7); decay to 0 over
        # an outer band. Hard-zero outside past & future caps.
        n = constraint.relative_window_days or 30
        center = question_date - dt.timedelta(days=n)
        half_band = max(n // 2, 7)
        outer = max(n, 21)
        delta = abs((turn_date - center).days)
        # Reject future turns (after question_date).
        if turn_date > question_date:
            return 0.0
        if delta <= half_band:
            return 1.0
        if delta <= outer:
            return max(0.0, 1.0 - (delta - half_band) / max(outer - half_band, 1))
        return 0.0

    if t == "relative-future":
        n = constraint.relative_window_days or 30
        center = question_date + dt.timedelta(days=n)
        half_band = max(n // 2, 7)
        outer = max(n, 21)
        delta = abs((turn_date - center).days)
        if turn_date < question_date:
            return 0.0
        if delta <= half_band:
            return 1.0
        if delta <= outer:
            return max(0.0, 1.0 - (delta - half_band) / max(outer - half_band, 1))
        return 0.0

    return 0.0


# ---------------------------------------------------------------------------
# Turn-to-date map builders (per-question)
# ---------------------------------------------------------------------------
def build_turn_to_date_map(
    src_question: dict,
) -> dict[int, dt.date | None]:
    """From a source LME question dict (with haystack_session_ids,
    haystack_dates, haystack_sessions), build {turn_id: date}."""
    mp: dict[int, dt.date | None] = {}
    sess_ids = src_question.get("haystack_session_ids") or []
    dates = src_question.get("haystack_dates") or []
    sessions = src_question.get("haystack_sessions") or []
    t = 0
    for _sid, date_str, turns in zip(sess_ids, dates, sessions):
        d = parse_lme_date(date_str)
        for _ in turns:
            mp[t] = d
            t += 1
    return mp


def build_all_turn_to_date(
    src_questions: Iterable[dict],
    hard_ids: set[str],
) -> tuple[
    dict[str, dict[int, dt.date | None]],
    dict[str, dt.date],
]:
    """Build per-conversation turn→date map and question→date for a set of
    hard question_ids from the source LME corpus.
    """
    turn_date: dict[str, dict[int, dt.date | None]] = {}
    q_date: dict[str, dt.date] = {}
    for q in src_questions:
        qid = q["question_id"]
        if qid not in hard_ids:
            continue
        turn_date[qid] = build_turn_to_date_map(q)
        qd = parse_lme_date(q.get("question_date", "") or "")
        if qd is None:
            # Fall back to last haystack date, or epoch.
            dates = [parse_lme_date(d) for d in q.get("haystack_dates", [])]
            dates = [d for d in dates if d]
            qd = max(dates) if dates else dt.date(2023, 1, 1)
        q_date[qid] = qd
    return turn_date, q_date
