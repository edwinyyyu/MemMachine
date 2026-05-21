"""Terse-display decoupled segmenter v2 -- date-alias retrieval enrichment.

v1 result: terse `block.text` is ~accuracy-neutral and ~15% cheaper than
qkey-min3p's `memory`; the Pareto win is modest because the accuracy
CEILING (~87 @ high K, mini judge) is set by broad retrieval misses --
~82% of hard failures have the answer evidence absent from the top-12.

A confirmed slice of those misses is temporal surface-form mismatch:
the question says "in August 2023" but the memory / timestamp carries
"2023-08-15", so BM25 cannot lexically bridge "August" to "08" and the
embedder treats the ISO date as a weak token. Example miss: "Which
city was Calvin visiting in August 2023?" -> retrieved/answered the
wrong-dated instance.

v2 fix (deterministic, no extra LLM cost, generalizable): enrich the
RETRIEVAL texts -- and only those -- with natural-language date
aliases. For every segment the event date is always known (the
timestamp); any extra absolute dates resolved into the memory text are
also expanded. Aliases ("August 2023", "August 15, 2023") are appended
to ``text_to_embed`` and ``text_to_score_bm25``. ``block.text`` (the
terse answer text) is untouched -- the rendered header already carries
the date, so the answerer pays nothing.

Everything else is v1: one LLM call per turn -> {memory, terse,
queries}; retrieval design otherwise identical to qkey-min3p.
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

# Prompt is unchanged from v1 (v2 is a deterministic post-processing
# change, not a prompt change).
PROMPT_TERSE_DECOUPLED_V2 = """\
Rewrite the MESSAGE into a JSON list of memory items. Each item \
contains a third-person memory statement, a compressed version of it, \
and retrieval queries. A future user querying any specific content in \
the message should find at least one item whose statement OR queries \
contain that content.

KEEP specific content (names, places, dates, numbers, decisions, plans, \
preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions). Dropping specific content is a FAILURE; \
emitting an item for interchangeable content is a FAILURE.

ONE item per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of the same occurrence is ONE item that \
contains all of its particulars. Distinct events (different times, \
different occasions, different actions) each get their own item.

EACH ITEM has:

(A) "memory": a third-person memory statement about {speaker}. Reports \
content, not the speech-act of conveying it -- ``{speaker} said that \
...`` / ``{speaker} told X that ...`` wrappers are dropped unless the \
speech-act itself is the event (a promise, an apology, an \
announcement). Refer to {speaker} by name; resolve first-person to \
{speaker}'s name on first occurrence; resolve ``you`` to addressee's \
name when known. Resolve demonstratives to concrete referents. \
Preserve every concrete particular from the message verbatim -- names, \
numbers, identifiers, distinctive phrasing.

DATES in the memory: the framework prepends the message timestamp \
automatically when surfacing the statement, so the statement text \
MUST NOT contain {date} in any form. Resolve every relative time \
reference (``yesterday``, ``last week``, ``three years ago``, \
``next Friday``, ``the weekend``, ``today``, ``tonight``, \
``recently``, ``now``, ``just``) to an absolute date anchored at \
{date}.
  - If the resolved date EQUALS {date}, the memory contains NO date \
and NO relative phrase.
  - If the resolved date DIFFERS from {date}, weave it into prose as \
``on YYYY-MM-DD`` (for day-precision) or ``in YYYY-MM`` (for month) \
or ``in YYYY`` (for year) and DELETE the original relative phrase. \
The relative phrase appearing alongside the resolved date is a \
FAILURE.

Forbidden date forms (each is a FAILURE): sentence-prefix \
``On YYYY-MM-DD, ...``; parenthetical ``(Date: ...)`` or \
``(Event date: ...)``; square-bracket ``[YYYY-MM-DD]``; \
``as of YYYY-MM-DD``.

(B) "terse": the SAME memory compressed to the fewest words a reader \
still fully understands. This is the only form a future reader sees, \
so it must answer-ready. Keep EVERY concrete particular from "memory" \
verbatim -- every name, place, date, number, identifier, decision, \
plan, preference, opinion, quoted phrase, attached-media detail, and \
the polarity/direction of each. Cut ONLY filler: articles and \
connectives where meaning survives, hedges, redundancy, and any \
scaffolding clause that carries no particular. Tight readable prose, \
not a headline and not telegraphic fragments. Dropping or blurring \
any particular that "memory" states is a FAILURE; an ambiguous \
"terse" is a FAILURE. Follow the SAME date rules as "memory".

(C) "queries": a list of 1-3 short queries (each 5-12 words). Each \
query is a way a future user might ask to retrieve this memory. \
Queries should target DIFFERENT angles on the particulars \
(who/what/when/where/how/why). Queries MUST NOT contain the answer; \
they should be phrased like a real user query. Queries that exactly \
restate the memory are a FAILURE.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

Output: a JSON object with field "items" whose value is a list of \
{{ "memory": "...", "terse": "...", "queries": ["...", "..."] }} \
objects. Use an empty list when the message contains no specific \
content.

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
    """v2: terse decoupled segmenter with date-alias retrieval enrichment."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_TERSE_DECOUPLED_V2,
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
