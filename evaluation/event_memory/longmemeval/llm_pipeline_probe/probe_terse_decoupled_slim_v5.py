"""Terse-decoupled segmenter v5 -- slim_v4 with the `memory` field dropped.

slim_v3/v4 produce three fields per item: memory (M, a full 3rd-person
statement), terse (T, M compressed), queries (Q). The pipeline only ever
uses T and Q -- the T-anchor ablation showed T works as the embedding and
BM25 anchor as well as M does, so M is never stored or retrieved against.

That leaves one question (Q3): does producing M *help the model produce
a good T*? M is written before T in the output, so it is a built-in
chain-of-thought scaffold -- write the careful full statement, then
compress. slim_v5 removes it: output only {statement, queries}, where
`statement` carries all of M's content rules (3rd-person, reference
resolution, keep-every-particular, date handling) AND T's terseness in
one field, produced in one shot with no M to compress from.

slim_v5 vs slim_v4 isolates exactly the memory-field removal -- same
keep/drop rule, same date handling, same everything else. Tie => M was
not a load-bearing scaffold, drop it. Regression => writing M first is
load-bearing even though M is never stored.
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

# slim_v5: slim_v4 keep/drop rule, single content field (no M scaffold).
PROMPT_SLIM_V5 = """\
You convert one chat MESSAGE into memory items for later retrieval.

A memory item records a fact about {speaker}'s own life -- never a move \
in the conversation. A fact about a life is an event that happened, a \
plan, decision or commitment, a circumstance, a possession, a \
relationship, or a preference {speaker} holds. A move in the \
conversation carries nothing beyond the chat itself; emit NOTHING for \
one -- a greeting or sign-off, thanks, agreement or acknowledgement, \
praise, encouragement or reassurance, a reaction or feeling about what \
was just said, a generic observation true of anyone. When unsure, ask: \
would this be worth knowing about {speaker} a year from now? If not, \
drop it. A message can be entirely such moves -- then emit an empty \
list.

Each item kept is ONE TOPIC -- a single occasion, plan, fact, or \
preference -- carrying EVERY detail the message gives about that topic. \
Keeping a topic whole is the priority: several sentences about one \
topic are ONE item, and a run of particulars about one thing is ONE \
item. Never split by sentence or by particular. Open a separate item \
only when the speaker genuinely turns to a different topic; most \
messages raise just one.

A PARTICULAR is any detail that makes a topic specific rather than \
generic: names, places, dates, numbers, identifiers, decisions, plans, \
preferences and opinions (with their direction), quoted wording, \
attached-media details. Every particular in the message must reach the \
output -- in an item's statement or its queries. Losing a particular \
is the main failure to avoid.

Each item has two fields:

(A) "statement" -- a third-person record of the topic, about \
{speaker}, in the fewest words that stay unambiguous. State the content \
itself, not the act of communicating it (drop "{speaker} said that \
..." wrappers, unless the communicative act IS the point, e.g. a \
promise or apology). Use {speaker}'s name; resolve "I"/"my" to \
{speaker}, "you" to the person addressed, and this/that/there to what \
they refer to. Keep every particular. Drop articles, filler and \
hedges; write one tight readable sentence -- not a headline or \
telegraphic fragment. This is the ONLY text retrieval shows to a \
reader, so it must stand on its own.

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

(B) "queries" -- 1 to 3 short questions a user might later ask that \
this item answers. Vary their angle (who/what/when/where/why/how). \
Phrase them as a person would ask; never include the answer.

Treat any NEIGHBORING TURNS shown as context for resolving references \
only; never emit items for them.

Return JSON: {{"items": [{{"statement": "...", \
"queries": ["...", "..."]}}, ...]}}. Return an empty list if the \
message has no topic worth remembering.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def _date_aliases(event_date: datetime, memory_text: str) -> str:
    """Natural-language aliases for every absolute date a segment carries."""
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
    statement: str
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
    """slim_v5 segmenter: single content field, no M scaffold."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_SLIM_V5,
        chunk_size: int = 1500,
        max_attempts: int = 3,
        include_raw_chunk_in_embed: bool = True,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
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
            if item.statement and item.statement.strip()
        ]

    def _build_embed_text(
        self, statement: str, queries: list[str], original_chunk: str,
        speaker: str,
    ) -> str:
        q = " ".join(q.strip() for q in queries if q and q.strip())
        parts = [statement]
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
                            statement = item.statement.strip()
                            embed_text = self._build_embed_text(
                                statement,
                                item.queries,
                                chunk_stripped,
                                speaker,
                            )
                            bm25_text = statement
                            aliases = _date_aliases(
                                event.timestamp, statement
                            )
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
                                    block=TextBlock(text=statement),
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
