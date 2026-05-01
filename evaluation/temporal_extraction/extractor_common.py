"""Shared base for improved extractor versions (v2-v6).

Provides an ``ImprovedExtractor`` class that the per-version files subclass.
Each version overrides prompts and optionally adds extra passes.

Cache: per-version directory ``cache/extractor_v{N}/`` keyed by (model,
prompt_hash). Reuses v1's shared llm_cache.json where prompt matches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from resolver import ResolverError, post_process
from schema import TimeExpression, time_expression_from_dict

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
UPPER_MODEL = "gpt-5"
CACHE_ROOT = Path(__file__).resolve().parent / "cache"
CACHE_ROOT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Cache (per-version)
# ---------------------------------------------------------------------------
class LLMCache:
    def __init__(self, cache_file: Path) -> None:
        self.path = cache_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        if cache_file.exists():
            with cache_file.open() as f:
                self._cache = json.load(f)
        self._new: dict[str, str] = {}

    @staticmethod
    def _key(model: str, prompt_key: str) -> str:
        return hashlib.sha256(f"{model}|{prompt_key}".encode()).hexdigest()

    def get(self, model: str, prompt_key: str) -> str | None:
        return self._cache.get(self._key(model, prompt_key))

    def put(self, model: str, prompt_key: str, response: str) -> None:
        k = self._key(model, prompt_key)
        self._cache[k] = response
        self._new[k] = response

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if self.path.exists():
            with self.path.open() as f:
                existing = json.load(f)
        existing.update(self._new)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(existing, f)
        tmp.replace(self.path)
        self._new.clear()


# ---------------------------------------------------------------------------
# Reference-time context block
# ---------------------------------------------------------------------------
def full_ref_context(ref_time: datetime) -> str:
    """Produce the verbose reference-time block for v2+ prompts."""
    wk = ref_time.strftime("%A")
    iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Day arithmetic
    yest = ref_time - timedelta(days=1)
    tom = ref_time + timedelta(days=1)
    # ISO week (Monday start): compute start of this week & last week
    iso_weekday = ref_time.isoweekday()  # Mon=1..Sun=7
    this_week_start = (ref_time - timedelta(days=iso_weekday - 1)).date()
    this_week_end = this_week_start + timedelta(days=6)
    last_week_start = this_week_start - timedelta(days=7)
    last_week_end = last_week_start + timedelta(days=6)
    next_week_start = this_week_start + timedelta(days=7)
    next_week_end = next_week_start + timedelta(days=6)

    # Months
    def month_label(dt: datetime) -> str:
        return dt.strftime("%B %Y")

    this_month = month_label(ref_time)
    # last month
    if ref_time.month == 1:
        last_month_dt = ref_time.replace(year=ref_time.year - 1, month=12, day=1)
    else:
        last_month_dt = ref_time.replace(month=ref_time.month - 1, day=1)
    # next month
    if ref_time.month == 12:
        next_month_dt = ref_time.replace(year=ref_time.year + 1, month=1, day=1)
    else:
        next_month_dt = ref_time.replace(month=ref_time.month + 1, day=1)
    this_quarter = (ref_time.month - 1) // 3 + 1
    this_year = ref_time.year
    return (
        f"Reference time: {iso_ref} ({wk}).\n"
        f"Today = {ref_time.strftime('%A, %B %-d, %Y')}. "
        f"Yesterday = {yest.strftime('%A, %b %-d, %Y')}. "
        f"Tomorrow = {tom.strftime('%A, %b %-d, %Y')}.\n"
        f"This week = {this_week_start.strftime('%b %-d')}"
        f"-{this_week_end.strftime('%b %-d, %Y')} (Mon-Sun).\n"
        f"Last week = {last_week_start.strftime('%b %-d')}"
        f"-{last_week_end.strftime('%b %-d, %Y')}. "
        f"Next week = {next_week_start.strftime('%b %-d')}"
        f"-{next_week_end.strftime('%b %-d, %Y')}.\n"
        f"This month = {this_month}. "
        f"Last month = {month_label(last_month_dt)}. "
        f"Next month = {month_label(next_month_dt)}.\n"
        f"This quarter = Q{this_quarter} {this_year}. "
        f"This year = {this_year}. Last year = {this_year - 1}. "
        f"Next year = {this_year + 1}."
    )


# ---------------------------------------------------------------------------
# Trigger-word gazetteer (domain-neutral, does NOT paraphrase synth corpus)
# ---------------------------------------------------------------------------
TRIGGER_GAZETTEER = """Common temporal trigger patterns to scan for:
- Named-relative calendar units: "yesterday", "today", "tomorrow"; "last/this/next + week/month/year/quarter/weekend"; "earlier/later this + week/month/year/quarter"; "this morning/afternoon/evening"; "tonight".
- Counted-relative: "N second/minute/hour/day/week/month/year/decade(s) ago", "in N ...", "N ... from now", "N ... later", "N ... earlier", "N ... before/after [event]".
- Approximate anchors: "about/around/roughly/approximately/nearly + <time>", "a few/couple of ...", "some time ago", "recently", "lately", "soon".
- Absolute dates: year literals like 1995 or 2026; month-day like "March 5", "Mar 5, 2026", "5 March 2026", "5/3/2026", "2026-03-05"; ISO timestamps; quarter labels "Q3 2025".
- Day-of-week names (Monday...Sunday), possibly with "last/this/next" or "on".
- Decades/eras: "the 80s", "the 1990s", "mid-90s", "the early 2000s".
- Times of day: "at 9am", "around 3 pm", "noon", "midnight", "dawn", "dusk".
- Intervals: "from X to Y", "between X and Y", "during <period>", "throughout <period>".
- Partial-period phrases: "the first/second/third/last week of <month>", "early/mid/late <month>".
- Embedded references in natural speech: "when I was in college", "back in X", "during the pandemic" (era) — extract only if the reference is concrete (year-anchored or context-anchored).
- Recurrences: "every <day>", "daily", "weekly", "monthly", "annually", "on weekends", "each morning".
- Durations (unanchored length): "for N hours/days/weeks/...", "three hours long", "half a day".
Do NOT emit: "once", "always", "often", bare seasons without a year, or purely descriptive adjectives ("recent", "old") without a concrete referent."""


# ---------------------------------------------------------------------------
# Few-shot hard-case examples (domain-neutral — NOT from synth corpus)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """Examples of hard-to-catch temporal references (domain-neutral):

Passage: "My cousin moved out last month and I haven't heard from her since."
Extractions: [{"surface": "last month", "kind_guess": "instant", "context_hint": "calendar month prior"}]

Passage: "Earlier this month we repainted the kitchen, and the week before that we ordered the cabinets."
Extractions: [
  {"surface": "Earlier this month", "kind_guess": "instant", "context_hint": "within the current calendar month, earlier half"},
  {"surface": "the week before that", "kind_guess": "instant", "context_hint": "one week before 'earlier this month'"}
]

Passage: "I saw her last Tuesday at the coffee shop."
Extractions: [{"surface": "last Tuesday", "kind_guess": "instant", "context_hint": "most recent past Tuesday"}]

Passage: "Back in the 90s, dial-up was the norm; by around 2008 most homes had broadband."
Extractions: [
  {"surface": "the 90s", "kind_guess": "interval", "context_hint": "decade 1990-1999"},
  {"surface": "around 2008", "kind_guess": "instant", "context_hint": "approximately the year 2008"}
]

Passage: "Every Thursday at 3pm I have a standup, but next Thursday it's cancelled."
Extractions: [
  {"surface": "Every Thursday at 3pm", "kind_guess": "recurrence", "context_hint": "weekly Thursday at 15:00"},
  {"surface": "next Thursday", "kind_guess": "instant", "context_hint": "upcoming Thursday (cancelled instance)"}
]

Passage: "Sometime in the first week of July 2020, we moved to the city."
Extractions: [{"surface": "the first week of July 2020", "kind_guess": "interval", "context_hint": "first calendar week of July 2020"}]"""


# ---------------------------------------------------------------------------
# Pass 2 system — identical across v1..v6 (resolution rules are stable).
# ---------------------------------------------------------------------------
PASS2_SYSTEM = """You resolve ONE temporal reference into absolute wall-clock form.

Reference time is given; use it to resolve all relative expressions.
Compute carefully - check weekday alignment, month length, year rollovers.

Output JSON matching this schema exactly. Omit fields not relevant to the
kind by setting them to null.

{
  "kind": "instant" | "interval" | "duration" | "recurrence",
  "surface": string,
  "confidence": float in [0,1],
  "instant": { "earliest": ISO, "latest": ISO, "best": ISO|null, "granularity": string } | null,
  "interval": { "start": {...instant...}, "end": {...instant...} } | null,
  "duration": { "seconds": int } | null,
  "recurrence": {
    "rrule": string,
    "dtstart": {...instant...},
    "until": {...instant...} | null,
    "exdates": [{...instant...}]
  } | null
}

Granularity is one of: second, minute, hour, day, week, month, quarter, year, decade, century.

Rules:
- Use UTC ISO 8601 with "Z" suffix for all datetimes.
- earliest is inclusive, latest is exclusive.
- For "about"/"around"/"roughly"/"a few"/"a couple", widen the interval by one granularity level; set best to the centered estimate.
- "yesterday" at ref R -> day bracket of R-1d, granularity=day.
- "N weeks ago" at ref R -> day bracket of R-7N days, granularity=day.
- "last week" -> week bracket of the ISO week before R, granularity=week.
- "last month" -> MONTH bracket of R.month-1 (wrapping year if needed), granularity=month. earliest=first day of that month at 00:00, latest=first day of the FOLLOWING month at 00:00.
- "this month" / "earlier this month" / "later this month" -> MONTH bracket of R.month, granularity=month. earliest=first day at 00:00, latest=first day of next month at 00:00. Use best = approx centered estimate.
- "last year" -> year bracket of R.year-1, granularity=year.
- "this year" -> year bracket of R.year, granularity=year.
- "in YYYY" -> year bracket of YYYY, granularity=year.
- "the 90s"/"the 1990s" -> decade bracket [1990, 2000), granularity=decade.
- "the 2010s" -> decade bracket [2010, 2020), granularity=decade.
- "around YYYY" -> year bracket of YYYY plus +/- 2 years, granularity=year.
- "the first week of <month> <year>" -> interval: start=first day of that month 00:00, end=7 days later 00:00. granularity=week on both ends. Use kind="interval".
- For recurrences "every Thursday": rrule "FREQ=WEEKLY;BYDAY=TH". No trailing semicolon. No DTSTART inside the rrule string (put it in dtstart instead).
- "every day at 9am": rrule "FREQ=DAILY;BYHOUR=9;BYMINUTE=0".
- For recurrences with no explicit end, set until to null.
- Always fill the instant.earliest/latest/best triple. Best is the typical point estimate.
"""

PASS2_JSON_SCHEMA: dict[str, Any] = {
    "name": "time_expression",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["instant", "interval", "duration", "recurrence"],
            },
            "surface": {"type": "string"},
            "confidence": {"type": "number"},
            "instant": {
                "type": ["object", "null"],
                "properties": {
                    "earliest": {"type": "string"},
                    "latest": {"type": "string"},
                    "best": {"type": ["string", "null"]},
                    "granularity": {"type": "string"},
                },
            },
            "interval": {
                "type": ["object", "null"],
                "properties": {
                    "start": {"type": "object"},
                    "end": {"type": "object"},
                },
            },
            "duration": {
                "type": ["object", "null"],
                "properties": {"seconds": {"type": "number"}},
            },
            "recurrence": {
                "type": ["object", "null"],
                "properties": {
                    "rrule": {"type": "string"},
                    "dtstart": {"type": "object"},
                    "until": {"type": ["object", "null"]},
                    "exdates": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
            },
        },
        "required": ["kind", "surface"],
    },
}


# ---------------------------------------------------------------------------
# Regex patterns for v5 pre-pass
# ---------------------------------------------------------------------------
REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "named_rel_day",
        re.compile(r"\b(?:yesterday|today|tomorrow|tonight)\b", re.IGNORECASE),
    ),
    (
        "named_rel_wmyq",
        re.compile(
            r"\b(?:last|this|next|earlier\s+this|later\s+this)\s+"
            r"(?:week|weekend|month|year|quarter|decade)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "named_rel_dow",
        re.compile(
            r"\b(?:last|this|next)\s+"
            r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "counted_rel",
        re.compile(
            r"\b(?:(?:about|around|roughly|approximately|a\s+few|a\s+couple\s+of)\s+)?"
            r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(?:second|minute|hour|day|week|month|year|decade)s?\s+"
            r"(?:ago|from\s+now|later|earlier|before|after)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "counted_in",
        re.compile(
            r"\bin\s+(?:about|around|roughly|approximately|a\s+few)?\s*"
            r"\d+\s+(?:second|minute|hour|day|week|month|year|decade)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "about_month_ago",
        re.compile(
            r"\b(?:about|around|roughly|approximately|a\s+few|a\s+couple\s+of|a)\s+"
            r"(?:second|minute|hour|day|week|month|year|decade)s?\s+ago\b",
            re.IGNORECASE,
        ),
    ),
    (
        "year_literal",
        re.compile(r"\b(?:around\s+|in\s+|about\s+)?(?:19\d{2}|20\d{2})\b"),
    ),
    (
        "decade_literal",
        re.compile(
            r"\bthe\s+(?:(?:early|mid|late)[-\s]+)?"
            r"(?:\d{1,3}0)s\b",
            re.IGNORECASE,
        ),
    ),
    (
        "month_day",
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2}(?:,\s*\d{4})?\b"
        ),
    ),
    (
        "month_year",
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
        ),
    ),
    (
        "first_week_of",
        re.compile(
            r"\b(?:the\s+)?(?:first|second|third|fourth|last)\s+week\s+of\s+"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"(?:\s+\d{4})?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "iso_date",
        re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?Z?)?\b"),
    ),
    (
        "time_of_day",
        re.compile(
            r"\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\b|"
            r"\b(?:noon|midnight|dawn|dusk|morning|afternoon|evening|night)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "every_recurrence",
        re.compile(
            r"\bevery\s+(?:\w+\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|day|week|month|year|morning|afternoon|evening|night)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "from_to",
        re.compile(
            r"\bfrom\s+[^.,;]+\s+to\s+[^.,;]+\b",
            re.IGNORECASE,
        ),
    ),
    (
        "during_period",
        re.compile(
            r"\bduring\s+(?:the\s+)?(?:(?:first|second|third|last|early|mid|late)\s+)?(?:week|month|year|quarter|\w+\s+\d{4}|\d{4})\b",
            re.IGNORECASE,
        ),
    ),
]


def regex_candidates(text: str) -> list[tuple[str, int, int]]:
    """Return (surface, start, end) tuples for regex candidates, deduped."""
    hits: list[tuple[str, int, int]] = []
    seen_spans: set[tuple[int, int]] = set()
    for _, pat in REGEX_PATTERNS:
        for m in pat.finditer(text):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            hits.append((m.group(0), m.start(), m.end()))
    # Sort by start offset and prefer longest non-overlapping candidates,
    # but keep all — the LLM filters.
    hits.sort(key=lambda x: (x[1], -x[2]))
    return hits


# ---------------------------------------------------------------------------
# Base extractor class
# ---------------------------------------------------------------------------
class BaseImprovedExtractor:
    """Base class with shared LLM call, cache, pass-2 logic, and post-process.

    Subclasses override:
    - ``VERSION`` (int): 2..6
    - ``pass1`` (return list of ref dicts)
    - optionally ``extract`` to add passes.
    """

    VERSION = 0

    def __init__(
        self,
        concurrency: int = 10,
        model: str = MODEL,
        cache_subdir: str | None = None,
    ) -> None:
        self.client = AsyncOpenAI()
        self.sem = asyncio.Semaphore(concurrency)
        subdir = cache_subdir or f"extractor_v{self.VERSION}"
        self.cache = LLMCache(CACHE_ROOT / subdir / "llm_cache.json")
        # Shared pass-2 cache across all improved versions (pass-2 system
        # prompt is identical, so resolution of the same surface+ref is the
        # same regardless of pass-1 version).
        self.shared_pass2_cache = LLMCache(
            CACHE_ROOT / "extractor_shared_pass2" / "llm_cache.json"
        )
        self.usage: dict[str, int] = {"input": 0, "output": 0}
        self.model = model

    async def _call(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        json_object: bool = False,
        max_completion_tokens: int = 6000,
        model: str | None = None,
        share_cache: bool = False,
    ) -> str:
        use_model = model or self.model
        prompt_key = f"{hashlib.sha256(system.encode()).hexdigest()[:16]}|{user}"
        # Check version-local cache first, then shared pass-2 cache if
        # ``share_cache`` is enabled (pass-2 resolution is stable across
        # pass-1 variations).
        cached = self.cache.get(use_model, prompt_key)
        if cached is None and share_cache:
            cached = self.shared_pass2_cache.get(use_model, prompt_key)
            if cached is not None:
                self.cache.put(use_model, prompt_key, cached)
        if cached is not None:
            return cached

        async def _do_call(max_toks: int) -> Any:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            kwargs: dict[str, Any] = {
                "model": use_model,
                "messages": messages,
                "max_completion_tokens": max_toks,
            }
            if json_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json_schema,
                }
            elif json_object:
                kwargs["response_format"] = {"type": "json_object"}
            async with self.sem:
                return await self.client.chat.completions.create(**kwargs)

        # Try with requested budget; if we hit a max-tokens 400, double once.
        resp = None
        toks = max_completion_tokens
        for attempt in range(2):
            try:
                resp = await _do_call(toks)
                break
            except Exception as e:
                msg = str(e)
                if "max_tokens" in msg or "output limit" in msg:
                    toks = min(toks * 2, 16000)
                    if attempt == 0:
                        continue
                # Propagate on non-max-tokens errors (caller catches at
                # the extract() level).
                raise
        assert resp is not None
        usage = resp.usage
        if usage:
            self.usage["input"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["output"] += getattr(usage, "completion_tokens", 0) or 0
        content = resp.choices[0].message.content or ""
        self.cache.put(use_model, prompt_key, content)
        if share_cache:
            self.shared_pass2_cache.put(use_model, prompt_key, content)
        return content

    async def _call_json(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> dict | list | None:
        """LLM call that parses JSON; retries once with error-hint on
        JSONDecodeError."""
        raw = await self._call(system, user, json_object=True, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            # Retry once with error context appended
            retry_user = (
                f"{user}\n\n[Previous response was not valid JSON: {e}. "
                "Please return a valid JSON object ONLY.]"
            )
            raw2 = await self._call(system, retry_user, json_object=True, **kwargs)
            try:
                return json.loads(raw2)
            except json.JSONDecodeError:
                return None

    # ------------------------------------------------------------------
    # Pass 1 — subclasses override
    # ------------------------------------------------------------------
    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Pass 2 — resolve a single reference (shared)
    # ------------------------------------------------------------------
    async def pass2(
        self,
        surface: str,
        kind_guess: str,
        context_hint: str,
        surrounding: str,
        ref_time: datetime,
        *,
        error_hint: str = "",
    ) -> dict[str, Any] | None:
        wk = ref_time.strftime("%A")
        iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        user = (
            f"Reference time: {iso_ref} ({wk})\n"
            f"Surrounding context: {surrounding}\n"
            f'Reference: "{surface}"\n'
            f"Kind hint: {kind_guess}\n"
            f"Context hint: {context_hint}\n"
        )
        if error_hint:
            user += f"\nCorrection hint: {error_hint}\n"
        user += "\nReturn JSON matching the schema."
        raw = await self._call(
            PASS2_SYSTEM,
            user,
            json_schema=PASS2_JSON_SCHEMA,
            max_completion_tokens=3000,
            share_cache=True,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # One retry with error hint appended
            user2 = (
                user
                + "\n\n[Your previous response was invalid JSON. Return ONLY the JSON object.]"
            )
            raw2 = await self._call(
                PASS2_SYSTEM,
                user2,
                json_schema=PASS2_JSON_SCHEMA,
                max_completion_tokens=3000,
                share_cache=True,
            )
            try:
                return json.loads(raw2)
            except json.JSONDecodeError:
                return None

    # ------------------------------------------------------------------
    # Validation & retry hook (v6 overrides)
    # ------------------------------------------------------------------
    def validate_resolution(self, pred: dict[str, Any], surface: str) -> str | None:
        """Return an error-hint string if the resolution fails a check,
        else None. Base class: no validation."""
        return None

    async def pass2_with_retry(
        self,
        surface: str,
        kind_guess: str,
        context_hint: str,
        surrounding: str,
        ref_time: datetime,
    ) -> dict[str, Any] | None:
        pred = await self.pass2(
            surface, kind_guess, context_hint, surrounding, ref_time
        )
        if pred is None:
            return None
        err = self.validate_resolution(pred, surface)
        if err is None:
            return pred
        # Retry once with correction hint.
        pred2 = await self.pass2(
            surface,
            kind_guess,
            context_hint,
            surrounding,
            ref_time,
            error_hint=err,
        )
        return pred2 or pred

    # ------------------------------------------------------------------
    # Main extract pipeline
    # ------------------------------------------------------------------
    async def extract(self, text: str, ref_time: datetime) -> list[TimeExpression]:
        refs = await self.pass1(text, ref_time)
        # Dedupe by lowercased surface (keep first occurrence per surface).
        seen: set[str] = set()
        unique_refs: list[dict[str, Any]] = []
        for ref in refs:
            surf = (ref.get("surface") or "").strip()
            if not surf:
                continue
            key = surf.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_refs.append(ref)

        coros = []
        surfaces_ordered: list[str] = []
        for ref in unique_refs:
            surface = ref["surface"].strip()
            kind_guess = ref.get("kind_guess", "instant")
            context_hint = ref.get("context_hint", "")
            surfaces_ordered.append(surface)
            coros.append(
                self.pass2_with_retry(surface, kind_guess, context_hint, text, ref_time)
            )
        results = await asyncio.gather(*coros)
        out: list[TimeExpression] = []
        for surface, pred in zip(surfaces_ordered, results):
            if pred is None:
                continue
            pred["reference_time"] = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                te = time_expression_from_dict(pred)
            except Exception:
                continue
            idx = text.find(surface)
            if idx >= 0:
                te.span_start = idx
                te.span_end = idx + len(surface)
            try:
                te, _warnings = post_process(te, auto_correct=True)
            except ResolverError:
                continue
            out.append(te)
        return out


# ---------------------------------------------------------------------------
# Public convenience
# ---------------------------------------------------------------------------
async def extract_many(
    extractor: BaseImprovedExtractor,
    items: list[tuple[str, str, datetime]],
) -> dict[str, list[TimeExpression]]:
    results: dict[str, list[TimeExpression]] = {}

    async def one(iid: str, text: str, ref_time: datetime) -> None:
        try:
            results[iid] = await extractor.extract(text, ref_time)
        except Exception as e:
            print(f"  extract failed for {iid}: {e}")
            results[iid] = []

    await asyncio.gather(*(one(i, t, r) for i, t, r in items))
    extractor.cache.save()
    extractor.shared_pass2_cache.save()
    return results
