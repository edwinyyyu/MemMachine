"""REVIEW v3: stronger imperatives on memory-field date substitution.

REVIEW v1 had relative_phrases/date_resolutions BEFORE memory in the
schema, so the LLM had the resolutions in autoregressive context when
writing memory. It still left "last week" unresolved in memory while
using the resolution in terse. Diagnosis: instruction wasn't strong
enough; the LLM treats memory as paraphrase of source and terse as
explicit compression.

v3 fixes this with stronger prompt language only (no new examples — the
TEMPORAL variant showed examples reshuffle retrieval). Three changes:
  - Frame (A) "relative_phrases" as a *worklist of items to substitute
    out of (C)*.
  - Add a required *substitution check* to (C): each phrase from (A)
    must appear ZERO times in memory; each (B) resolution must appear
    in (C) in its place.
  - Use strong imperatives ("MUST", "verify before submitting").

Schema unchanged from v1.
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


PROMPT_REVIEW3 = """\
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

(A) "relative_phrases" -- a worklist of every relative time reference \
in the source for THIS item, exactly as the speaker wrote it. Examples: \
"yesterday", "last week", "the other day", "two weekends after", \
"three years ago", "next Friday", "recently", "just now". These phrases \
are what you will SUBSTITUTE OUT of field (C) below; this field is your \
extraction step. Empty list if none.

(B) "date_resolutions" -- the absolute date for each phrase in (A), in \
the same order, anchored at {date}. Match the precision the speaker \
stated. Natural format ("April 8, 2022", "March 2024", "2024") for \
chat / prose source; ISO ("2024-03-15") only if the source itself uses \
ISO. This field is your resolution step — these resolved dates are \
what you will SUBSTITUTE INTO field (C). Empty list if (A) is empty.

(C) "memory" -- a third-person statement of the topic, about \
{speaker}. State the content itself, not the act of communicating it \
(drop "{speaker} said that ..." wrappers, unless the communicative act \
IS the point, e.g. a promise or apology). Use {speaker}'s name; \
resolve "I"/"my" to {speaker}, "you" to the person addressed, and \
this/that/there to what they refer to. Keep every particular.

REQUIRED date substitution (apply mechanically before submitting (C)):

  1. For each phrase in (A), write the corresponding date from (B) into \
the prose of (C) where that phrase would have appeared, woven into the \
sentence naturally.
  2. Verify (C) contains ZERO occurrences of any phrase listed in (A). \
If you wrote "last week" or "the other day" or "next Friday" or any \
similar relative phrase in (C), you violated this rule — go back and \
substitute the resolution from (B) before submitting.
  3. Special case: if the resolved date in (B) equals the event date \
({date}), drop the phrase entirely with no replacement (no date in its \
place).
  4. Never write {date} itself in (C). Never write a date as a \
bracketed, parenthetical, or sentence-prefixed tag. The (B) resolutions \
must be woven into the sentence naturally.

The LLM you generated (A) and (B) is the same one writing (C). Use \
what you already wrote.

(D) "terse" -- field (C) rewritten in the fewest words that stay \
unambiguous: drop articles, filler and hedges; keep every particular \
and the same date handling (no phrases from (A); only resolutions from \
(B)). Write tight readable prose -- a full sentence, not a headline or \
telegraphic fragment. This is the ONLY text retrieval shows to a \
reader, so it must stand on its own.

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


def _date_aliases_verbose(event_date: datetime, memory_text: str) -> str:
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


class Review3Segmenter(Segmenter):
    """REVIEW v3: stronger imperatives on memory-field date substitution."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REVIEW3,
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
                            aliases = _date_aliases_verbose(event.timestamp, memory)
                            bm25_text = memory
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
