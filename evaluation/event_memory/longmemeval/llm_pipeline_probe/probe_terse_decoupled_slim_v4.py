"""Terse-decoupled segmenter v4 -- slim_v3 + disambiguated keep/drop rule.

Prompt-only change vs slim_v3. The slim_v3->v4 motivation: a cross-model
diagnosis (gpt-5-nano vs gpt-5-mini segmenter on identical source) showed
the weak model (nano) keeps ~19% more source messages and emits terse
fields ~13% longer. The extra material is conversational filler -- praise,
agreement, encouragement, reactions, generic reflections -- that mini
correctly drops. Root cause: slim_v3's keep/drop rule is a closed
enumeration ("greetings, sign-offs, thanks, acknowledgements, ...") and
it lists "opinion" as a keep-type while listing "generic reflections" as
a drop-type, with no objective test to resolve the two. The weak model
defaults to keep.

slim_v4 replaces that with one objective dichotomy -- a fact about the
speaker's LIFE vs a move in the CONVERSATION -- plus a diagnosis test
("worth knowing a year from now?"). "opinion" is dropped as a standalone
keep-type (an opinion tied to a real topic survives as a particular; a
bare generic opinion does not). Everything else (date section, terse,
queries, anti-fragmentation, code) is byte-identical to slim_v3, so the
nano-vs-mini gap change is attributable to this rule alone.
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

# slim_v4: slim_v3 with the keep/drop rule reframed as an objective
# life-fact vs conversation-move dichotomy. See module docstring.
PROMPT_SLIM_V4 = """\
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
    """Slimmed terse-decoupled segmenter (slim_v4 keep/drop rule)."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_SLIM_V4,
        chunk_size: int = 1500,
        max_attempts: int = 3,
        include_raw_chunk_in_embed: bool = True,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
        # When False, text_to_embed is memory + queries (+ dates) only --
        # no raw speaker-prefixed chunk.
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
