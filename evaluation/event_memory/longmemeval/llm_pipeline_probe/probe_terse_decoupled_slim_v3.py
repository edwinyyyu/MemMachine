"""Slimmed terse-decoupled segmenter v3 -- slim_v2 + restored date section.

Prompt-only change vs terse-decoupled-v2. v2's prompt had accreted to
~75 lines / 878 tok (the "particular" list 3x, the retrieval objective
3x, a 19-line date section, "FAILURE" 7x).

Iteration history:
- slim_v1 (556 tok): removed redundancy but also cut the
  anti-fragmentation guardrail -> over-segmented (~9-11k segs vs 5.2k).
- slim_v2 (629 tok): restored anti-fragmentation as an objective
  TOPIC-not-sentence principle. Granularity fixed -- but the gpt-5 judge
  showed a temporal (cat-2) regression: 86-88 vs v2's 89-90. The
  19->6-line date-section compression had cut load-bearing date rules.
- slim_v3 (726 tok, -17% vs v2): restores the explicit date section --
  9 relative-reference examples, the EQUALS/DIFFERS branch, and
  precision-matched ISO output (day/month/year) -- while keeping every
  other slim_v2 simplification (those left cat-1/3/4 unchanged).

See PROMPT_BLOAT_ANALYSIS.md for the redundancy audit.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    DecoupledRetrievalContext,
    Event,
    ProducerContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel

# Slimmed prompt v3 (slim_v2 + restored date section): redundancy-removed + objective anti-fragmentation
# rule. See PROMPT_BLOAT_ANALYSIS.md.
PROMPT_SLIM_V3 = """\
You convert one chat MESSAGE into memory items for later retrieval.

Each item is ONE TOPIC the speaker raises -- a single occasion, plan, \
fact, opinion, or preference -- carrying EVERY detail the message gives \
about that topic. Keeping a topic whole is the priority: several \
sentences about one topic are ONE item, and a run of particulars about \
one thing is ONE item. Never split by sentence or by particular. Open a \
separate item only when the speaker genuinely turns to a different \
topic; most messages raise just one. Emit NOTHING for interchangeable \
filler -- greetings, sign-offs, thanks, acknowledgements, reassurance, \
pleasantries, generic reflections, generic questions.

A PARTICULAR is any detail that makes a topic specific rather than \
generic: names, places, dates, numbers, identifiers, decisions, plans, \
preferences and opinions (with their direction), quoted wording, \
attached-media details. Every particular in the message must reach the \
output -- in an item's statement or its queries. Losing a particular \
is the main failure to avoid.

Each item has three fields:

(A) "memory" -- a third-person statement of the topic, about \
{speaker}. State the content itself, not the act of communicating it \
(drop "{speaker} said that ..." wrappers, unless the communicative act \
IS the point, e.g. a promise or apology). Use {speaker}'s name; \
resolve "I"/"my" to {speaker}, "you" to the person addressed, and \
this/that/there to what they refer to. Keep every particular.

Dates in the statement: the message's own date ({date}) is attached \
automatically when this memory is surfaced, so the statement text must \
never contain {date}. Resolve every relative time reference -- \
"yesterday", "last week", "three years ago", "next Friday", "the \
weekend", "today", "recently", "now", "just" -- to an absolute date \
anchored at {date}.
  - If the resolved date EQUALS {date}, the statement carries no date \
and no relative phrase.
  - If it DIFFERS from {date}, delete the relative phrase and weave the \
absolute date into the prose at its true precision: "on 2024-03-15" for \
a day, "in March 2024" for a month, "in 2024" for a year. Never leave a \
relative phrase beside the resolved date, and never write a date as a \
bracketed, parenthetical, or sentence-prefixed tag.

(B) "terse" -- field (A) rewritten in the fewest words that stay \
unambiguous: drop articles, filler and hedges; keep every particular \
and the same date handling. Write tight readable prose -- a full \
sentence, not a headline or telegraphic fragment. This is the ONLY \
text retrieval shows to a reader, so it must stand on its own.

(C) "queries" -- 1 to 3 short questions a user might later ask that \
this item answers. Vary their angle (who/what/when/where/why/how). \
Phrase them as a person would ask; never include the answer.

Treat any NEIGHBORING TURNS shown as context for resolving references \
only; never emit items for them.

Return JSON: {{"items": [{{"memory": "...", "terse": "...", \
"queries": ["...", "..."]}}, ...]}}. Return an empty list if the \
message has no topic worth remembering.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


# Ablation: "verbatim" date handling -- replaces the date subsection of
# PROMPT_SLIM_V3 with an instruction to copy date references straight
# from the source. The LLM does no date math. The .replace() asserts the
# old block is found, so the constants stay in sync if PROMPT_SLIM_V3
# changes.
_DATE_BLOCK_RESOLVE = """Dates in the statement: the message's own date ({date}) is attached automatically when this memory is surfaced, so the statement text must never contain {date}. Resolve every relative time reference -- "yesterday", "last week", "three years ago", "next Friday", "the weekend", "today", "recently", "now", "just" -- to an absolute date anchored at {date}.
  - If the resolved date EQUALS {date}, the statement carries no date and no relative phrase.
  - If it DIFFERS from {date}, delete the relative phrase and weave the absolute date into the prose at its true precision: "on 2024-03-15" for a day, "in March 2024" for a month, "in 2024" for a year. Never leave a relative phrase beside the resolved date, and never write a date as a bracketed, parenthetical, or sentence-prefixed tag."""

_DATE_BLOCK_VERBATIM = """Dates: copy date and time references verbatim from the message. Don't resolve relative phrases like "yesterday", "last week", or "next Friday" to absolute dates, don't rewrite a stated date into a different format, and don't drop a date because it matches the event date. The event date ({date}) is attached automatically when this memory is surfaced, so relative phrases stay meaningful in context."""

# Hybrid: resolve the relative phrase to an absolute date (same as RESOLVE)
# but match the source's register on output format instead of prescribing
# ISO. Chat / prose -> "March 15, 2024", "March 2024", "2024". ISO source
# ("2024-03-15") -> stay ISO. Precision stays at what the speaker stated.
_DATE_BLOCK_RESOLVE_NATURAL = """Dates in the statement: the message's own date ({date}) is attached automatically when this memory is surfaced, so the statement text must never contain {date}. Resolve every relative time reference -- "yesterday", "last week", "three years ago", "next Friday", "the weekend", "today", "recently", "now", "just" -- to an absolute date anchored at {date}.
  - If the resolved date EQUALS {date}, the statement carries no date and no relative phrase.
  - If it DIFFERS from {date}, delete the relative phrase and weave the absolute date into the prose. Match the source's register: use ISO-like dates ("2024-03-15") only if the source itself uses ISO; for chat or prose, use natural language ("on March 15, 2024", "in March 2024", "in 2024"). Match the precision the speaker stated (don't invent a day if they only said a month). Never leave a relative phrase beside the resolved date, and never write a date as a bracketed, parenthetical, or sentence-prefixed tag."""

PROMPT_SLIM_V3_VERBATIM_DATES = PROMPT_SLIM_V3.replace(
    _DATE_BLOCK_RESOLVE, _DATE_BLOCK_VERBATIM
)
assert _DATE_BLOCK_VERBATIM in PROMPT_SLIM_V3_VERBATIM_DATES, (
    "Date-block swap failed: _DATE_BLOCK_RESOLVE not found in PROMPT_SLIM_V3"
)

PROMPT_SLIM_V3_NATURAL_DATES = PROMPT_SLIM_V3.replace(
    _DATE_BLOCK_RESOLVE, _DATE_BLOCK_RESOLVE_NATURAL
)
assert _DATE_BLOCK_RESOLVE_NATURAL in PROMPT_SLIM_V3_NATURAL_DATES, (
    "Date-block swap failed: _DATE_BLOCK_RESOLVE not found in PROMPT_SLIM_V3"
)


_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
# ISO date / ISO month occurrences inside a memory statement.
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def _date_aliases(event_date: datetime, memory_text: str) -> str:
    """Natural-language aliases for every absolute date a segment carries.

    Always includes the event's own date (the rendered timestamp).
    Also expands any ISO date/month resolved into the memory text. Each
    date yields a "Month YYYY" alias and, when day-precise, a
    "Month D, YYYY" alias -- so a query phrased "in August 2023" or
    "on August 15, 2023" lexically and semantically matches.
    """
    dates: set[tuple[int, int, int]] = {
        (event_date.year, event_date.month, event_date.day)
    }
    for match in _ISO_RE.finditer(memory_text):
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3)) if match.group(3) else 0
        if 1 <= month <= 12:
            dates.add((year, month, day))

    parts: list[str] = []
    seen: set[str] = set()
    for year, month, day in sorted(dates):
        month_name = _MONTHS[month - 1]
        for alias in (
            f"{month_name} {year}",
            f"{month_name} {day}, {year}" if day else None,
        ):
            if alias and alias not in seen:
                seen.add(alias)
                parts.append(alias)
    return "; ".join(parts)


def _long_form(year: int, month: int, day: int) -> str:
    """CLDR-style long form for en-US, one canonical string per date.

    Day-precise -> "Monday, January 15, 2024" (matches CLDR 'full');
    month-precise -> "January 2024".
    """
    if day:
        return datetime(year, month, day).strftime("%A, %B %d, %Y")
    return f"{_MONTHS[month - 1]} {year}"


def _date_aliases_cldr_all(event_date: datetime, memory_text: str) -> str:
    """One CLDR-long form per date; regex still extracts dates from memory.

    BM25 tokenizes the long form into its component lemmas, so the
    abbreviated "Month YYYY" alias is a strict subset and unnecessary --
    keep exactly one canonical form per date.
    """
    dates: set[tuple[int, int, int]] = {
        (event_date.year, event_date.month, event_date.day)
    }
    for match in _ISO_RE.finditer(memory_text):
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3)) if match.group(3) else 0
        if 1 <= month <= 12:
            dates.add((year, month, day))
    parts: list[str] = []
    seen: set[str] = set()
    for year, month, day in sorted(dates):
        alias = _long_form(year, month, day)
        if alias not in seen:
            seen.add(alias)
            parts.append(alias)
    return "; ".join(parts)


def _date_aliases_cldr_event(event_date: datetime, memory_text: str) -> str:
    """Single CLDR-long form for the event's own timestamp only -- no regex.

    Drops the ISO-from-memory extraction: any other date the LLM wove
    into `memory` already tokenizes inline, no extra alias for it.
    """
    return _long_form(
        event_date.year, event_date.month, event_date.day
    )


def _date_aliases_verbose_event(event_date: datetime, memory_text: str) -> str:
    """2-form alias (TF-boost preserved) for the event timestamp only.

    Same shape as `verbose` -- "Month YYYY; Month D, YYYY" -- but with no
    regex extraction. Isolates whether the regex's ISO-from-memory
    contribution is doing useful work in 2-form mode.
    """
    year, month, day = event_date.year, event_date.month, event_date.day
    parts: list[str] = [f"{_MONTHS[month - 1]} {year}"]
    if day:
        parts.append(f"{_MONTHS[month - 1]} {day}, {year}")
    return "; ".join(parts)


_ALIAS_FNS = {
    "verbose": _date_aliases,
    "cldr-all": _date_aliases_cldr_all,
    "cldr-event": _date_aliases_cldr_event,
    "verbose-event": _date_aliases_verbose_event,
}


class _MemoryItem(BaseModel):
    memory: str
    terse: str
    queries: list[str]


class _RewriteResponse(BaseModel):
    items: list[_MemoryItem]


def _format_neighbors(before: list, after: list) -> str:
    lines: list[str] = []
    if before:
        lines.append("PRIOR TURNS (context only, do not emit):")
        for ev in before:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    if after:
        lines.append("LATER TURNS (context only, do not emit):")
        for ev in after:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


class TerseDecoupledSegmenter(Segmenter):
    """Slimmed terse-decoupled segmenter (redundancy-removed prompt)."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_SLIM_V3,
        chunk_size: int = 1500,
        max_attempts: int = 3,
        include_raw_chunk_in_embed: bool = True,
        date_aliases_in_embed: bool = True,
        date_aliases_in_bm25: bool = True,
        date_alias_mode: str = "verbose",
        date_handling: str = "resolve",
    ) -> None:
        if date_alias_mode not in _ALIAS_FNS:
            raise ValueError(
                f"date_alias_mode={date_alias_mode!r} not in {list(_ALIAS_FNS)}"
            )
        if date_handling not in ("resolve", "verbatim", "natural"):
            raise ValueError(
                f"date_handling={date_handling!r} not in ('resolve', 'verbatim', 'natural')"
            )
        # If the caller didn't override prompt_template, dispatch on
        # date_handling. Explicit prompt_template wins.
        if prompt_template is PROMPT_SLIM_V3 and date_handling == "verbatim":
            prompt_template = PROMPT_SLIM_V3_VERBATIM_DATES
        elif prompt_template is PROMPT_SLIM_V3 and date_handling == "natural":
            prompt_template = PROMPT_SLIM_V3_NATURAL_DATES
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
        # v3 ablation: when False, text_to_embed is memory + queries (+ dates)
        # only -- no raw speaker-prefixed chunk. With BM25 decoupled onto its
        # own clean text, the raw chunk's lexical-surface role is redundant
        # and its vocatives/filler may dilute the embedding vector.
        self._include_raw_chunk_in_embed = include_raw_chunk_in_embed
        # Ablation: gate the programmatic "Dates:" alias line per channel.
        # Default both True (production). When a channel is False, dates
        # survive there only as ISO strings the LLM weaves into `memory`.
        self._date_aliases_in_embed = date_aliases_in_embed
        self._date_aliases_in_bm25 = date_aliases_in_bm25
        # Ablation: alias-generation mode.
        #   "verbose"   -- current (1-2 forms/date, regex on)
        #   "cldr-all"  -- 1 long-form/date, regex on
        #   "cldr-event"-- 1 long-form, event timestamp only, no regex
        self._date_alias_mode = date_alias_mode
        # Ablation: date_handling.
        #   "resolve"  -- LLM resolves relative refs + rewrites in mixed
        #                 ISO/natural at precision (current production).
        #   "verbatim" -- LLM copies date refs from source as-said; no
        #                 resolution, no format rewriting.
        self._date_handling = date_handling
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=[
                "\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
                ", ", " ", "",
            ],
            keep_separator="end",
        )

    async def _rewrite_chunk(
        self, chunk: str, speaker: str, date: str, neighbors_block: str
    ) -> list[_MemoryItem]:
        prompt = self._prompt_template.format(
            speaker=speaker,
            date=date,
            passage=chunk,
            neighbors_block=neighbors_block,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return [
            item
            for item in response.items
            if item.memory and item.memory.strip()
        ]

    def _build_embed_text(
        self, memory: str, queries: list[str], original_chunk: str, speaker: str
    ) -> str:
        q = " ".join(q.strip() for q in queries if q and q.strip())
        parts = [memory]
        if q:
            parts.append(f"Queries: {q}")
        if self._include_raw_chunk_in_embed:
            parts.append(f"{speaker}: {original_chunk}")
        return "\n".join(parts)

    @override
    async def segment(self, event: Event) -> list[Segment]:
        match event.context:
            case SurroundingEventsContext(
                producer=producer, before=before, after=after
            ):
                speaker = producer
                neighbors_block = _format_neighbors(before, after)
            case ProducerContext(producer=producer):
                speaker = producer
                neighbors_block = ""
            case _:
                speaker = "the speaker"
                neighbors_block = ""
        date_str = event.timestamp.strftime("%Y-%m-%d")

        segments: list[Segment] = []
        for block_index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
                    chunks = (
                        self._splitter.split_text(text)
                        if len(text) > self._chunk_size
                        else [text]
                    )
                    offset = 0
                    for chunk in chunks:
                        chunk_stripped = chunk.strip()
                        if not chunk_stripped:
                            continue
                        items = await self._rewrite_chunk(
                            chunk_stripped, speaker, date_str, neighbors_block
                        )
                        for item in items:
                            memory = item.memory.strip()
                            terse = item.terse.strip() or memory
                            embed_text = self._build_embed_text(
                                memory,
                                item.queries,
                                chunk_stripped,
                                speaker,
                            )
                            bm25_text = memory
                            if (self._date_aliases_in_embed
                                    or self._date_aliases_in_bm25):
                                aliases = _ALIAS_FNS[self._date_alias_mode](
                                    event.timestamp, memory
                                )
                                if aliases:
                                    if self._date_aliases_in_embed:
                                        embed_text = (
                                            f"{embed_text}\nDates: {aliases}"
                                        )
                                    if self._date_aliases_in_bm25:
                                        bm25_text = (
                                            f"{bm25_text}\nDates: {aliases}"
                                        )
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=terse),
                                    context=DecoupledRetrievalContext(
                                        producer=speaker,
                                        text_to_embed=embed_text,
                                        text_to_score_bm25=bm25_text,
                                    ),
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
