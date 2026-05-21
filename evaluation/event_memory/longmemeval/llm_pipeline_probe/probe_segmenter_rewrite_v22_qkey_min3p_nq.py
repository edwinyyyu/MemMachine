"""v22-qkey-min3p -- combine qkey (queries in embedding) with min3p tight date.

Hypothesis: qkey K=7 add0.1 lifts c1234 by +0.78pp but pulls 16t over
budget because longer events match the query-augmented embedding. The
min3p prompt produces SHORTER memory statements (drops redundant inline
date when message-date equals event-date) which should reduce embedding
token bias toward longer events.

text_to_embed = "{memory}\\nQueries: {q1} {q2}\\n{speaker}: {raw_chunk}"
where {memory} follows min3p's tighter date conventions.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    RewriteContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel


PROMPT_REWRITE_V22_QKEY_MIN3P = """\
Rewrite the MESSAGE into a JSON list of memory items. Each item \
contains a third-person memory statement plus retrieval queries. \
A future user querying any specific content in the message should \
find at least one item whose statement OR queries contain that content.

KEEP specific content (names, places, dates, numbers, decisions, plans, \
preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions). Dropping specific content is a FAILURE; \
emitting an item for interchangeable content is a FAILURE.

A QUESTION the speaker poses to the other person is NOT a retrievable \
fact -- even when it names specific people, places, teams, titles, or \
events. ``Did you watch the Liverpool vs Chelsea match?``, ``Have you \
been to Paris?``, ``What game are you playing?`` are speech acts of \
ASKING, not facts about {speaker}; they share entity words and \
question words with future user queries and crowd out the \
answer-bearing item, so emitting an item for them is a FAILURE. The \
ONLY exception: a question that itself states a concrete fact about \
{speaker} or a described event (e.g. ``Want to hear about the \
half-marathon I ran yesterday?`` states that {speaker} ran a \
half-marathon) -- emit the embedded fact, never the act of asking.

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

(B) "queries": a list of 1-3 short queries (each 5-12 words). Each \
query is a way a future user might ask to retrieve this memory. \
Queries should target DIFFERENT angles on the particulars \
(who/what/when/where/how/why). Queries MUST NOT contain the answer; \
they should be phrased like a real user query. Queries that exactly \
restate the memory are a FAILURE.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

Output: a JSON object with field "items" whose value is a list of \
{{ "memory": "...", "queries": ["...", "..."] }} objects. Use an \
empty list when the message contains no specific content.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _MemoryItem(BaseModel):
    memory: str
    queries: list[str]


class _RewriteResponse(BaseModel):
    items: list[_MemoryItem]


def _format_neighbors(before: list, after: list, current_speaker: str) -> str:
    lines = []
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


class RewriteSegmenter(Segmenter):
    """v22-qkey-min3p -- qkey embedding format + min3p tight date rule."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_QKEY_MIN3P,
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
        return [
            item
            for item in response.items
            if item.memory and item.memory.strip()
        ]

    @staticmethod
    def _build_embed_text(
        memory: str, queries: list[str], original_chunk: str, speaker: str
    ) -> str:
        q = " ".join(q.strip() for q in queries if q and q.strip())
        if q:
            return f"{memory}\nQueries: {q}\n{speaker}: {original_chunk}"
        return f"{memory}\n{speaker}: {original_chunk}"

    @override
    async def segment(self, event: Event) -> list[Segment]:
        match event.context:
            case SurroundingEventsContext(
                producer=producer, before=before, after=after
            ):
                speaker = producer
                neighbors_block = _format_neighbors(before, after, producer)
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
                            embed_text = RewriteSegmenter._build_embed_text(
                                item.memory.strip(),
                                item.queries,
                                chunk_stripped,
                                speaker,
                            )
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=item.memory.strip()),
                                    context=RewriteContext(text_to_embed=embed_text),
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
