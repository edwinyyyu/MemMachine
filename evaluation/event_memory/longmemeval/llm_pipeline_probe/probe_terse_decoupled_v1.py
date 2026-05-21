"""Terse-display decoupled segmenter (v1).

Goal: get qkey-min3p's K=10 retrieval coverage (~87% mini-judge) inside
the ~340-350 token answer budget, where qkey-min3p only affords K=6
(~85%). The lever is per-segment answer-token cost.

qkey-min3p stores ONE text per event and shows it to the answerer
(``block.text``), embeds an augmented form, and BM25-scores the same
display text. Here the answerer-visible text is DECOUPLED from the
retrieval texts via ``DecoupledRetrievalContext``:

  - ``block.text``          = ``terse``  -- a maximally compressed but
        fully faithful statement; the ONLY text that costs answer
        tokens. Cutting it ~25% lets ~25% more segments (higher K) fit
        the budget.
  - ``text_to_embed``       = ``{memory}\\nQueries: {q}\\n{speaker}: {chunk}``
        -- byte-identical to qkey-min3p's embedding input.
  - ``text_to_score_bm25``  = ``memory`` -- the full statement, exactly
        what qkey-min3p's BM25 scored.

So retrieval (both channels) is the qkey-min3p design unchanged; the
only delta is that the answerer reads ``terse`` instead of ``memory``.
A clean isolation of "does a terser answer text Pareto-win by buying K".

The LLM emits, per event, ``{memory, terse, queries}`` in one call --
same cost as qkey-min3p's one-call-per-turn. Paired deriver:
``WholeTextDeriver`` (passes ``text_to_embed`` through).
"""

from __future__ import annotations

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

# Extends qkey-min3p's prompt with a third per-item field, "terse".
PROMPT_TERSE_DECOUPLED_V1 = """\
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
    """qkey-min3p retrieval design, with a terser decoupled display text."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_TERSE_DECOUPLED_V1,
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
                            embed_text = (
                                TerseDecoupledSegmenter._build_embed_text(
                                    memory,
                                    item.queries,
                                    chunk_stripped,
                                    speaker,
                                )
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
                                        text_to_score_bm25=memory,
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
