"""Slimmed terse-decoupled segmenter -- redundancy-removed prompt.

Identical architecture and date-alias logic to terse-decoupled-v2; the
ONLY change is the prompt. PROMPT_TERSE_DECOUPLED_V2 had accreted over
many iterations to ~75 lines with the "specific particular" list stated
three times, the retrieval objective framed three times, a 19-line date
section carrying two redundant enumerations (10 relative-reference
examples; 4 "forbidden date forms"), and "FAILURE" repeated 7x.

This prompt states every load-bearing rule exactly once, principle-first
(~38 lines, ~half the length). Load-bearing behaviour preserved:
keep every particular; one item per event; drop interchangeable filler;
third-person content-not-speech-act rewrite; reference resolution;
relative->absolute date resolution woven inline; never repeat the
message's own date; terse = memory minus filler; 1-3 retrieval queries.

See PROMPT_BLOAT_ANALYSIS.md for the line-by-line redundancy audit.
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

# Slimmed, redundancy-removed prompt. See PROMPT_BLOAT_ANALYSIS.md.
PROMPT_SLIM_V1 = """\
You convert one chat MESSAGE into memory items for later retrieval.

Split the message into EVENTS -- distinct things that happened or were \
decided, planned, stated, or preferred. Emit ONE item per event; a \
multi-sentence elaboration of a single event stays one item. Emit \
NOTHING for interchangeable filler: greetings, sign-offs, \
acknowledgements, pleasantries, generic questions.

A PARTICULAR is any detail that makes an event specific rather than \
generic: names, places, dates, numbers, identifiers, decisions, plans, \
preferences and opinions (with their direction), quoted wording, \
attached-media details. Every particular in the message must reach the \
output -- in an item's statement or its queries. Losing a particular \
is the main failure to avoid.

Each item has three fields:

(A) "memory" -- a third-person statement of the event, about \
{speaker}. State the content itself, not the act of communicating it \
(drop "{speaker} said that ..." wrappers, unless the communicative act \
IS the event, e.g. a promise or apology). Use {speaker}'s name; \
resolve "I"/"my" to {speaker}, "you" to the person addressed, and \
this/that/there to what they refer to. Keep every particular. \
Dates: the message's own date ({date}) is attached automatically, so \
never put it in the text. Convert every relative time reference \
("yesterday", "last spring", "in two weeks") into an absolute date \
anchored on {date}, drop the relative wording, and write the date \
inline as ordinary words ("on 2024-03-15", "in March 2024") -- not as \
a bracketed, parenthetical, or sentence-prefixed tag. If the converted \
date equals {date}, write no date at all.

(B) "terse" -- field (A) rewritten in the fewest words that stay \
unambiguous: drop articles, filler and hedges; keep every particular \
and the same date handling. This is the ONLY text retrieval shows to a \
reader, so it must stand on its own.

(C) "queries" -- 1 to 3 short questions a user might later ask that \
this item answers. Vary their angle (who/what/when/where/why/how). \
Phrase them as a person would ask; never include the answer.

Treat any NEIGHBORING TURNS shown as context for resolving references \
only; never emit items for them.

Return JSON: {{"items": [{{"memory": "...", "terse": "...", \
"queries": ["...", "..."]}}, ...]}}. Return an empty list if the \
message has no particular worth remembering.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


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
        prompt_template: str = PROMPT_SLIM_V1,
        chunk_size: int = 1500,
        max_attempts: int = 3,
        include_raw_chunk_in_embed: bool = True,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
        # v3 ablation: when False, text_to_embed is memory + queries (+ dates)
        # only -- no raw speaker-prefixed chunk. With BM25 decoupled onto its
        # own clean text, the raw chunk's lexical-surface role is redundant
        # and its vocatives/filler may dilute the embedding vector.
        self._include_raw_chunk_in_embed = include_raw_chunk_in_embed
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
                            aliases = _date_aliases(event.timestamp, memory)
                            if aliases:
                                embed_text = f"{embed_text}\nDates: {aliases}"
                                bm25_text = f"{bm25_text}\nDates: {aliases}"
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
