"""First-person variant of slim_v3 / bo-natural.

Hypothesis: writing memory in the speaker's first-person voice ("I",
"my") makes temporal resolution easier for the LLM (because the speaker
naturally thinks in first person) and improves single-hop fact
fidelity. Tests whether the cat2/cat4 raw-events gap closes when the
rewrite is first-person instead of third-person.

Differences from bo-natural:
  - (A) "memory" field is written in {speaker}'s voice (I/my/me).
  - (B) "terse" mirrors (A) in first-person.
  - block.text (rendered to answerer) is prefixed with the speaker name
    so the answerer can attribute statements across mixed-speaker memory.
  - bm25_text and embed_text are also speaker-prefixed.
  - Defaults match bo-natural: date_handling="natural",
    date_aliases_in_embed=False, date_aliases_in_bm25=True.

Kept independent from probe_terse_decoupled_slim_v3.py so the working
bo-natural code path can't accidentally regress.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    FirstPersonDecoupledRetrievalContext,
    ProducerContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel


PROMPT_FP_NATURAL = """\
You convert one chat MESSAGE from {speaker} into memory items for later \
retrieval.

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

(A) "memory" -- a FIRST-person statement of the topic, in {speaker}'s \
voice. Use "I", "my", "me" naturally -- write as if {speaker} is \
recounting it afterward. Drop "I said that ..." wrappers (unless the \
communicative act IS the point, e.g. a promise or apology). Refer to \
other people by name. Keep every particular. The speaker's name will \
be attached at retrieval time, so I never write "{speaker}" in the \
statement.

Dates in the statement: the message's own date ({date}) is attached \
automatically when this memory is surfaced, so the statement text must \
never contain {date}. Resolve every relative time reference -- \
"yesterday", "last week", "three years ago", "next Friday", "the \
weekend", "today", "recently", "now", "just", "the other day", \
"two weekends after" -- to an absolute date anchored at {date}.
  - If the resolved date EQUALS {date}, the statement carries no date \
and no relative phrase.
  - If it DIFFERS from {date}, delete the relative phrase and weave the \
absolute date into the prose. Match the source's register: use ISO-like \
dates ("2024-03-15") only if the source itself uses ISO; for chat or \
prose, use natural language ("on March 15, 2024", "in March 2024", \
"in 2024"). Match the precision the speaker stated (don't invent a day \
if they only said a month). Never leave a relative phrase beside the \
resolved date, and never write a date as a bracketed, parenthetical, \
or sentence-prefixed tag.

(B) "terse" -- field (A) rewritten in the fewest words that stay \
unambiguous: drop articles, filler and hedges; keep every particular \
and the same date handling. Write tight readable prose -- a full \
sentence, not a headline or telegraphic fragment. First-person ("I", \
"my"). This is the ONLY text retrieval shows to a reader (with my name \
attached separately), so it must stand on its own.

(C) "queries" -- 1 to 3 short questions someone else might later ask \
about {speaker} that this item answers. Phrase in third person about \
{speaker} ("When did {speaker} adopt Ned?"). Vary their angle \
(who/what/when/where/why/how). Never include the answer.

Treat any NEIGHBORING TURNS shown as context for resolving references \
only; never emit items for them.

Return JSON: {{"items": [{{"memory": "...", "terse": "...", \
"queries": ["...", "..."]}}, ...]}}. Return an empty list if the \
message has no topic worth remembering.

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
    """Natural-language aliases for the event date + any ISO dates in memory.

    Each date yields a "Month YYYY" alias and, when day-precise, a
    "Month D, YYYY" alias.
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


class FirstPersonSegmenter(Segmenter):
    """First-person memory-rewrite segmenter (bo-natural variant)."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_FP_NATURAL,
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
                            memory_fp = item.memory.strip()
                            terse_fp = item.terse.strip() or memory_fp
                            # Use FirstPersonDecoupledRetrievalContext so
                            # EventMemory._segment_header adds the producer
                            # prefix to the rendered block.text (same as
                            # ProducerContext). The 1p block.text itself
                            # contains no speaker name, so EventMemory's
                            # render is what attributes it.
                            block_text = terse_fp
                            # bm25_text and embed_text DO include speaker so
                            # name-based queries lexically/semantically match
                            # (the 1p memory itself has no name otherwise).
                            aliases = _date_aliases_verbose(event.timestamp, memory_fp)
                            bm25_text = f"{speaker}: {memory_fp}"
                            if aliases:
                                bm25_text = f"{bm25_text}\nDates: {aliases}"
                            q = " ".join(
                                q.strip() for q in item.queries if q and q.strip()
                            )
                            embed_parts = [f"{speaker}: {memory_fp}"]
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
                                    block=TextBlock(text=block_text),
                                    context=FirstPersonDecoupledRetrievalContext(
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
