"""REVIEW v2: add LLM-resolved dates to bm25 text + use terse for memory.

REVIEW v1 diagnosis: the LLM correctly fills `date_resolutions` and uses
them in `terse`, but ignores them in `memory`. Since `memory` becomes
the bm25_text, BM25 retrieval misses queries about absolute dates.

Two changes from REVIEW v1:
  1. block.text uses `memory` (the long-form rewrite that may have
     unresolved phrases). Actually still use terse — terse uses
     resolutions and is shown to answerer.
  2. bm25_text incorporates date_resolutions as a "Resolved:" line so
     BM25 can lexically match queries about the absolute date even when
     the memory field kept the relative phrase.

Schema (unchanged from v1):
  relative_phrases, date_resolutions, memory, terse, queries
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


# Same prompt as REVIEW v1
PROMPT_REVIEW2 = """\
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

Each item has FIVE fields (in this exact generation order):

(A) "relative_phrases" -- list every relative time reference from the \
source for THIS item, exactly as the speaker wrote it. Examples: \
"yesterday", "last week", "the other day", "two weekends after", \
"three years ago", "next Friday", "recently", "just now". Empty list \
if none.

(B) "date_resolutions" -- for each phrase in (A), in the same order, \
write the absolute date anchored at {date}. Match the precision the \
speaker stated. Use natural format ("April 8, 2022", "March 2024", \
"2024") for chat / prose source; ISO ("2024-03-15") only if the source \
itself uses ISO. Empty list if (A) is empty.

(C) "memory" -- a third-person statement of the topic, about \
{speaker}. State the content itself, not the act of communicating it \
(drop "{speaker} said that ..." wrappers, unless the communicative act \
IS the point, e.g. a promise or apology). Use {speaker}'s name; \
resolve "I"/"my" to {speaker}, "you" to the person addressed, and \
this/that/there to what they refer to. Keep every particular.

Dates in the statement: the message's own date ({date}) is attached \
automatically when this memory is surfaced, so the statement text must \
never contain {date}, and must never contain any phrase listed in (A). \
For each phrase in (A), substitute the corresponding date from (B) \
into the prose (if the resolution equals {date}, drop the phrase \
entirely with no date in its place). Never leave a relative phrase \
beside a resolved date, and never write a date as a bracketed, \
parenthetical, or sentence-prefixed tag.

(D) "terse" -- field (C) rewritten in the fewest words that stay \
unambiguous: drop articles, filler and hedges; keep every particular \
and the same date handling. Write tight readable prose -- a full \
sentence, not a headline or telegraphic fragment. This is the ONLY \
text retrieval shows to a reader, so it must stand on its own.

(E) "queries" -- 1 to 3 short questions a user might later ask that \
this item answers. Vary their angle (who/what/when/where/why/how). \
Phrase them as a person would ask; never include the answer.

Treat any NEIGHBORING TURNS shown as context for resolving references \
only; never emit items for them.

Return JSON: {{"items": [{{"relative_phrases": [...], "date_resolutions": [...], "memory": "...", "terse": "...", "queries": ["...", "..."]}}, ...]}}. Return an empty list if the message has no topic worth remembering.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def _date_aliases_verbose(event_date: datetime, *extra_texts: str) -> str:
    """Natural aliases for event date + any ISO dates in the extra texts.

    Adds aliases for ISO/month patterns found in any of the supplied
    texts (memory, resolution strings, etc.) so BM25 has a uniform alias
    surface.
    """
    dates: set[tuple[int, int, int]] = {
        (event_date.year, event_date.month, event_date.day)
    }
    for text in extra_texts:
        for match in _ISO_RE.finditer(text or ""):
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


class _MemoryItem(BaseModel):
    relative_phrases: list[str]
    date_resolutions: list[str]
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


class Review2Segmenter(Segmenter):
    """REVIEW v2: date_resolutions added to bm25 text for retrieval."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REVIEW2,
        chunk_size: int = 1500,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                ", ",
                " ",
                "",
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
        return [item for item in response.items if item.memory and item.memory.strip()]

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
                            # Concatenate date_resolutions into a single
                            # text the alias function can scan for any
                            # ISO or month references. Also pass `terse`
                            # since it's where the LLM consistently uses
                            # resolved dates.
                            resolutions_text = "; ".join(
                                r.strip()
                                for r in item.date_resolutions
                                if r and r.strip()
                            )
                            aliases = _date_aliases_verbose(
                                event.timestamp,
                                memory,
                                terse,
                                resolutions_text,
                            )
                            # bm25_text: memory + resolved dates (so BM25
                            # can lexically match queries about absolute
                            # dates even when memory kept relative phrase).
                            # The Resolved line lists the absolute dates
                            # for each relative phrase in memory.
                            bm25_text = memory
                            if resolutions_text:
                                bm25_text = f"{bm25_text}\nResolved: {resolutions_text}"
                            if aliases:
                                bm25_text = f"{bm25_text}\nDates: {aliases}"
                            q = " ".join(
                                q.strip() for q in item.queries if q and q.strip()
                            )
                            embed_parts = [memory]
                            if q:
                                embed_parts.append(f"Queries: {q}")
                            embed_parts.append(f"{speaker}: {chunk_stripped}")
                            embed_text = "\n".join(embed_parts)
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
