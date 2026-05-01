"""F5 — Allen-relation extractor.

Pass 1 (Allen): one structured-output call per passage that returns, for
every temporal reference it finds:

    {
        "surface": <time substring, verbatim>,
        "kind_guess": instant|interval|duration|recurrence,
        "relation": before|after|during|overlaps|contains|null,
        "anchor_span": <surface of anchor> | null,
        "anchor_kind": event|time|null,
    }

``relation`` is null for purely absolute references ("March 15, 2026").
For relational references ("before the meeting", "during my wedding"),
``relation`` names the Allen relation and ``anchor_span`` is the surface
of the referenced anchor ("the meeting", "my wedding").

Pass 2 reuses :mod:`extractor`'s pass 2 to resolve the time surface into
an absolute bracket. If ``anchor_kind == "time"`` the anchor span is
also resolved with pass 2. If ``anchor_kind == "event"`` the anchor is
left unresolved here — :mod:`event_resolver` fills it in.

Uses gpt-5-mini, cached under cache/allen/.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from advanced_common import JSONCache
from allen_schema import AllenAnchor, AllenExpression
from dotenv import load_dotenv
from extractor import PASS2_JSON_SCHEMA, PASS2_SYSTEM
from openai import AsyncOpenAI
from resolver import ResolverError, post_process
from schema import (
    TimeExpression,
    time_expression_from_dict,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "allen"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
LLM_CACHE_FILE = CACHE_DIR / "llm_cache.json"


ALLEN_PASS1_SYSTEM = """You are a meticulous temporal-reference extractor
that ALSO labels qualitative temporal relations (Allen's interval
algebra, 5-relation subset).

A "temporal reference" is ANY of:
- An absolute/relative time expression: "March 15, 2026", "yesterday",
  "the 90s".
- A RELATIONAL reference: a phrase that positions something in time
  relative to another event or time. RELATIONAL REFERENCES COUNT AS
  TEMPORAL REFERENCES even if the anchor is a named event with no
  absolute date in the passage (e.g., "before my wedding", "during
  the conference", "the month after my promotion", "overlapping with
  Alice's graduation").

For each reference, output:

- surface: the exact verbatim substring. For relational references,
  surface is the WHOLE clause INCLUDING the relational preposition —
  e.g., "before my wedding", "during my wedding weekend", "the month
  after my promotion".
- kind_guess: instant | interval | duration | recurrence.
- relation: one of [before, after, during, overlaps, contains] if the
  reference positions itself qualitatively against another event/time;
  otherwise null.
  - before:  "before X", "prior to X", "leading up to X", "the week
    before X", "N weeks before X" → strictly earlier than X.
  - after:   "after X", "following X", "the month after X", "N days
    after X", "right after X", "shortly after X" → strictly later.
  - during:  "during X", "in the middle of X", "at X" (when X is a
    span), "while X was happening", "on X day" → fully contained
    within X.
  - overlaps: "around X", "coinciding with X", "overlapping X",
    "from late A into early B" (when it partially hits X) → partial
    overlap.
  - contains: "from A to B which included X", "the trip that
    contained X" → d contains X.
- anchor_span: the surface of the anchor the reference is relative to
  (e.g., "my wedding", "the meeting", "March 15, 2020"). null if
  relation is null.
- anchor_kind: "event" if the anchor is a named event or noun phrase
  ("my wedding", "the conference", "Alice's graduation"); "time" if
  the anchor is itself an absolute time expression ("March 15, 2020",
  "January"); null if relation is null.

CRITICAL: in a question like "What happened before my wedding?",
"before my wedding" IS the temporal reference. Emit it with
relation="before", anchor_span="my wedding", anchor_kind="event".
Do NOT skip relational references just because they have no explicit
date.

Do NOT try to resolve event anchors to absolute time — leave the
anchor as text only.

A single temporal reference with NO qualitative relation (e.g., just
"March 15, 2026" or "yesterday") emits relation=null,
anchor_span=null, anchor_kind=null.

Output a single JSON object: {"refs": [ {surface, kind_guess,
relation, anchor_span, anchor_kind}, ... ]}. If no temporal
references, output {"refs": []}.
"""

ALLEN_PASS1_SCHEMA = {
    "name": "allen_refs",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "surface": {"type": "string"},
                        "kind_guess": {"type": "string"},
                        "relation": {"type": ["string", "null"]},
                        "anchor_span": {"type": ["string", "null"]},
                        "anchor_kind": {"type": ["string", "null"]},
                    },
                    "required": ["surface"],
                },
            }
        },
        "required": ["refs"],
    },
}


VALID_RELATIONS = {"before", "after", "during", "overlaps", "contains"}
VALID_ANCHOR_KINDS = {"event", "time"}


class AllenExtractor:
    """Two-pass extractor producing AllenExpressions.

    Pass 1 detects time spans + optional (relation, anchor). Pass 2
    resolves each time/anchor surface into an absolute bracket.
    """

    def __init__(self, concurrency: int = 10) -> None:
        self.client = AsyncOpenAI()
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = JSONCache(LLM_CACHE_FILE)
        self.usage: dict[str, int] = {"input": 0, "output": 0}

    async def _call(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        json_object: bool = False,
        max_completion_tokens: int = 2000,
        cache_tag: str = "",
    ) -> str:
        pkey = JSONCache.key(
            MODEL,
            cache_tag,
            hashlib.sha256(system.encode()).hexdigest()[:16],
            user,
        )
        cached = self.cache.get(pkey)
        if cached is not None:
            return cached
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_completion_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        elif json_object:
            kwargs["response_format"] = {"type": "json_object"}
        current_tokens = max_completion_tokens
        last_err: Exception | None = None
        for _ in range(3):
            try:
                kwargs["max_completion_tokens"] = current_tokens
                async with self.sem:
                    resp = await self.client.chat.completions.create(**kwargs)
                if resp.usage:
                    self.usage["input"] += getattr(resp.usage, "prompt_tokens", 0) or 0
                    self.usage["output"] += (
                        getattr(resp.usage, "completion_tokens", 0) or 0
                    )
                content = resp.choices[0].message.content or ""
                if content:
                    self.cache.put(pkey, content)
                    return content
                current_tokens = min(current_tokens * 2, 8000)
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.3)
        if last_err is not None:
            print(f"  Allen LLM failed: {last_err}")
        return ""

    # ------------------------------------------------------------------
    # Pass 1: detect spans + relation + anchor
    # ------------------------------------------------------------------
    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
        iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        wk = ref_time.strftime("%A")
        user = (
            f"Reference time: {iso_ref} ({wk})\n"
            f"Passage:\n{text}\n\n"
            'Return {"refs": [...]}.'
        )
        raw = await self._call(
            ALLEN_PASS1_SYSTEM,
            user,
            json_schema=ALLEN_PASS1_SCHEMA,
            max_completion_tokens=2000,
            cache_tag="allen_p1",
        )
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return list(d.get("refs") or [])

    # ------------------------------------------------------------------
    # Pass 2: resolve a single surface into a TimeExpression
    # ------------------------------------------------------------------
    async def pass2_resolve(
        self,
        surface: str,
        kind_guess: str,
        surrounding: str,
        ref_time: datetime,
    ) -> TimeExpression | None:
        iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        wk = ref_time.strftime("%A")
        user = (
            f"Reference time: {iso_ref} ({wk})\n"
            f"Surrounding context: {surrounding}\n"
            f'Reference: "{surface}"\n'
            f"Kind hint: {kind_guess}\n"
            f"Context hint: (none)\n\n"
            "Return JSON matching the schema."
        )
        raw = await self._call(
            PASS2_SYSTEM,
            user,
            json_schema=PASS2_JSON_SCHEMA,
            max_completion_tokens=1500,
            cache_tag="allen_p2",
        )
        try:
            pred = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not pred:
            return None
        pred["reference_time"] = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            te = time_expression_from_dict(pred)
        except Exception:
            return None
        try:
            te, _ = post_process(te, auto_correct=True)
        except ResolverError:
            return None
        return te

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------
    async def extract(self, text: str, ref_time: datetime) -> list[AllenExpression]:
        refs = await self.pass1(text, ref_time)
        out: list[AllenExpression] = []
        # We run pass-2 for time surfaces (always) and for ``time``-kind
        # anchor spans (resolved inline).  We skip pass-2 for event
        # anchors — event_resolver handles those lazily at query time.
        tasks: list[tuple[int, str, asyncio.Task]] = []
        # Prepare
        staged: list[dict[str, Any]] = []
        for ref in refs:
            surface = (ref.get("surface") or "").strip()
            if not surface:
                continue
            kind_guess = ref.get("kind_guess") or "instant"
            rel = ref.get("relation")
            if rel not in VALID_RELATIONS:
                rel = None
            anchor_span = ref.get("anchor_span")
            anchor_kind = ref.get("anchor_kind")
            if anchor_kind not in VALID_ANCHOR_KINDS:
                anchor_kind = None
            # If relation is labeled but anchor is missing, drop the
            # relation — we need an anchor to reason about it.
            if rel is not None and (not anchor_span or not anchor_kind):
                rel = None
                anchor_span = None
                anchor_kind = None
            staged.append(
                {
                    "surface": surface,
                    "kind_guess": kind_guess,
                    "relation": rel,
                    "anchor_span": anchor_span,
                    "anchor_kind": anchor_kind,
                }
            )

        # Time-expression pass-2 tasks (one per ref).
        # The surface passed to pass-2 is just the *time* portion — strip
        # the relational preposition if present so the resolver doesn't
        # widen wrongly. We derive the "time portion" by stripping a
        # leading "before|after|during|around" word.
        def strip_relation(surface: str, rel: str | None) -> str:
            if rel is None:
                return surface
            # Heuristic: drop the first word or two.
            lower = surface.lower()
            for pref in (
                "right before ",
                "right after ",
                "just before ",
                "just after ",
                "the month after ",
                "the week after ",
                "the year after ",
                "the month before ",
                "the week before ",
                "the year before ",
                "shortly before ",
                "shortly after ",
                "leading up to ",
                "following ",
                "prior to ",
                "before ",
                "after ",
                "during ",
                "around ",
                "throughout ",
                "spanning ",
            ):
                if lower.startswith(pref):
                    return surface[len(pref) :].strip()
            return surface

        t_time_tasks: list[Any] = []
        t_anchor_tasks: list[Any] = []
        # Track which refs need pass-2 at all. For pure event-anchor
        # relational refs in QUERIES ("what happened before my wedding?"),
        # there is no absolute time in the surface — only the event name
        # remains after stripping. We SKIP pass-2 in that case and
        # construct a sentinel TimeExpression so the AllenExpression
        # still carries (relation, anchor).
        skip_time_p2: list[bool] = []
        for s in staged:
            time_surface = strip_relation(s["surface"], s["relation"])
            # Decide whether pass-2 on the time surface is worthwhile.
            # It's worthwhile only if `time_surface` likely names a time
            # expression on its own (contains a digit, a month name, a
            # weekday, or a relative cue like "ago"/"yesterday"/"last").
            skip = False
            if s["relation"] is not None and s["anchor_kind"] == "event":
                # Heuristic: if after stripping, the remainder IS the
                # anchor_span or a trivial variant, skip pass-2.
                anchor = (s["anchor_span"] or "").strip().lower()
                ts_lower = time_surface.strip().lower()
                if ts_lower == anchor or anchor in ts_lower:
                    skip = True
                if not _looks_like_time_surface(ts_lower):
                    skip = True
            skip_time_p2.append(skip)
            if skip:
                t_time_tasks.append(_null_coro())
            else:
                t_time_tasks.append(
                    self.pass2_resolve(time_surface, s["kind_guess"], text, ref_time)
                )
            if s["relation"] is not None and s["anchor_kind"] == "time":
                t_anchor_tasks.append(
                    self.pass2_resolve(s["anchor_span"], "instant", text, ref_time)
                )
            else:
                t_anchor_tasks.append(_null_coro())

        time_resolutions = await asyncio.gather(*t_time_tasks)
        anchor_resolutions = await asyncio.gather(*t_anchor_tasks)

        for s, time_te, anchor_te, skip in zip(
            staged, time_resolutions, anchor_resolutions, skip_time_p2
        ):
            if time_te is None and not skip:
                # Pass-2 failed; drop this ref (can't resolve).
                continue
            if skip:
                # Build a sentinel TimeExpression; retrieval will rely on
                # (relation, anchor) rather than the time field.
                time_te = _sentinel_time(ref_time, s["surface"])
            # Attach span offsets
            idx = text.find(s["surface"])
            if idx >= 0:
                time_te.span_start = idx
                time_te.span_end = idx + len(s["surface"])
            anchor = None
            if s["relation"] is not None and s["anchor_span"]:
                anchor = AllenAnchor(
                    kind=s["anchor_kind"],
                    span=s["anchor_span"],
                    resolved=(anchor_te if s["anchor_kind"] == "time" else None),
                )
            out.append(
                AllenExpression(time=time_te, relation=s["relation"], anchor=anchor)
            )
        return out

    def save(self) -> None:
        self.cache.save()

    def cost_usd(self) -> float:
        return (
            self.usage["input"] * 0.25 / 1_000_000
            + self.usage["output"] * 2.0 / 1_000_000
        )


async def _null_coro() -> None:
    return None


_TIME_CUES = (
    "ago",
    "yesterday",
    "tomorrow",
    "last ",
    "next ",
    "this ",
    "weeks",
    "months",
    "years",
    "days",
    "hours",
    "minutes",
    "seconds",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "morning",
    "evening",
    "afternoon",
    "night",
)


def _looks_like_time_surface(s: str) -> bool:
    """True if ``s`` looks like it names a time expression on its own."""
    sl = s.lower()
    if any(c.isdigit() for c in sl):
        return True
    for cue in _TIME_CUES:
        if cue in sl:
            return True
    return False


def _sentinel_time(ref_time: datetime, surface: str) -> TimeExpression:
    """A placeholder TimeExpression for relational refs with no
    standalone time portion (e.g., "before my wedding").

    We populate instant=earliest=latest=best=ref_time with granularity
    "day" so retrieval scorers that look at ``time`` don't crash; the
    real retrieval path (``allen_retrieve``) uses the RESOLVED anchor,
    not this sentinel.
    """
    from schema import FuzzyInstant

    return TimeExpression(
        kind="instant",
        surface=surface,
        reference_time=ref_time,
        instant=FuzzyInstant(
            earliest=ref_time,
            latest=ref_time,
            best=ref_time,
            granularity="day",
        ),
    )


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------
async def extract_many(
    items: list[tuple[str, str, datetime]],
) -> tuple[dict[str, list[AllenExpression]], dict[str, int]]:
    ex = AllenExtractor()
    results: dict[str, list[AllenExpression]] = {}

    async def one(iid: str, text: str, ref_time: datetime) -> None:
        try:
            results[iid] = await ex.extract(text, ref_time)
        except Exception as e:
            print(f"  allen extract failed for {iid}: {e}")
            results[iid] = []

    await asyncio.gather(*(one(i, t, r) for i, t, r in items))
    ex.save()
    return results, ex.usage
