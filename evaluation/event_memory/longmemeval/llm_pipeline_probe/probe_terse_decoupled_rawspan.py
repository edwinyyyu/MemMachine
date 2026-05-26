"""Terse-decoupled segmenter -- raw-span variant.

slim_v3 appends the WHOLE source message to every item's text_to_embed,
shared across all topic-items from that message: a 3-topic message
dilutes each item's embedding with the other two topics' raw text.

This variant asks the LLM to ALSO return, per item, the verbatim raw
span of the message that the item covers -- so each item's embedding
anchor carries only ITS OWN raw text. Segmentation (the raw span) and
the rewrite happen in one LLM call.

Flags:
  span_first -- field/generation order of the structured output:
    True  -> {source, memory, terse, queries}  (commit to the span,
             then rewrite)
    False -> {memory, terse, queries, source}  (rewrite, then point
             back to the source span)
  block_uses_span -- when True, block.text (answerer-visible, counted
    against the token budget) is the raw span instead of `terse`.
    Phase-2 test: only meaningful if the per-item span helps the
    embedding.

If a returned `source` is not a (whitespace-normalized) substring of
the message, it falls back to the whole chunk -- guards against
hallucinated spans. Everything else matches slim_v3.
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

# --- field descriptions (referenced by name, so order is free) ---------

_DESC_SOURCE = """\
"source" -- the verbatim slice of the MESSAGE below that this item is \
drawn from. Copy it exactly, character for character: no paraphrase, no \
edits, no words added or removed. If the item covers the whole message, \
copy the whole message. If the speaker raises several topics, each \
item's "source" is the consecutive run of message text for that topic. \
"source" is raw message text; the other fields are your rewrite of it."""

_DESC_MEMORY = """\
"memory" -- a third-person statement of the topic, about {speaker}. \
State the content itself, not the act of communicating it (drop \
"{speaker} said that ..." wrappers, unless the communicative act IS the \
point, e.g. a promise or apology). Use {speaker}'s name; resolve \
"I"/"my" to {speaker}, "you" to the person addressed, and this/that/\
there to what they refer to. Keep every particular.

Dates in "memory": the message's own date ({date}) is attached \
automatically when this memory is surfaced, so the text must never \
contain {date}. Resolve every relative time reference -- "yesterday", \
"last week", "three years ago", "next Friday", "the weekend", "today", \
"recently", "now", "just" -- to an absolute date anchored at {date}.
  - If the resolved date EQUALS {date}, the statement carries no date \
and no relative phrase.
  - If it DIFFERS from {date}, delete the relative phrase and weave the \
absolute date into the prose at its true precision: "on 2024-03-15" for \
a day, "in March 2024" for a month, "in 2024" for a year. Never leave a \
relative phrase beside the resolved date, and never write a date as a \
bracketed, parenthetical, or sentence-prefixed tag."""

_DESC_TERSE = """\
"terse" -- "memory" rewritten in the fewest words that stay \
unambiguous: drop articles, filler and hedges; keep every particular \
and the same date handling. Write tight readable prose -- a full \
sentence, not a headline or telegraphic fragment. This is the text \
retrieval shows to a reader, so it must stand on its own."""

_DESC_QUERIES = """\
"queries" -- 1 to 3 short questions a user might later ask that this \
item answers. Vary their angle (who/what/when/where/why/how). Phrase \
them as a person would ask; never include the answer."""

_INTRO = """\
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

Each item has four fields:"""

_OUTRO_HEAD = (
    "Treat any NEIGHBORING TURNS shown as context for resolving "
    "references only; never emit items for them.\n\nReturn JSON: "
)
_OUTRO_TAIL = (
    ". Return an empty list if the message has no topic worth "
    "remembering.\n\n{neighbors_block}MESSAGE FROM {speaker} on "
    "{date}:\n{passage}"
)

_JSON_KEYS = {
    True: '"source": "...", "memory": "...", "terse": "...", '
    '"queries": ["...", "..."]',
    False: '"memory": "...", "terse": "...", "queries": ["...", "..."], '
    '"source": "..."',
}


def _build_template(span_first: bool) -> str:
    """Assemble the prompt; one call-time .format() resolves everything.

    The JSON example keeps doubled braces (escaped for that .format());
    {speaker}/{date}/{passage}/{neighbors_block} stay as placeholders.
    """
    if span_first:
        fields = [_DESC_SOURCE, _DESC_MEMORY, _DESC_TERSE, _DESC_QUERIES]
    else:
        fields = [_DESC_MEMORY, _DESC_TERSE, _DESC_QUERIES, _DESC_SOURCE]
    field_block = "\n\n".join(fields)
    json_example = '{{"items": [{{' + _JSON_KEYS[span_first] + '}}, ...]}}'
    outro = _OUTRO_HEAD + json_example + _OUTRO_TAIL
    return _INTRO + "\n\n" + field_block + "\n\n" + outro


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


def _normalized(text: str) -> str:
    return " ".join(text.split())


class _ItemSpanFirst(BaseModel):
    source: str
    memory: str
    terse: str
    queries: list[str]


class _ItemSpanLast(BaseModel):
    memory: str
    terse: str
    queries: list[str]
    source: str


class _ResponseSpanFirst(BaseModel):
    items: list[_ItemSpanFirst]


class _ResponseSpanLast(BaseModel):
    items: list[_ItemSpanLast]


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
    """Terse-decoupled segmenter that also emits a per-item raw span."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        span_first: bool = True,
        block_uses_span: bool = False,
        chunk_size: int = 1500,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._span_first = span_first
        self._block_uses_span = block_uses_span
        self._prompt_template = _build_template(span_first)
        self._response_model = (
            _ResponseSpanFirst if span_first else _ResponseSpanLast
        )
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
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
    ) -> list:
        prompt = self._prompt_template.format(
            speaker=speaker,
            date=date,
            passage=chunk,
            neighbors_block=neighbors_block,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=self._response_model,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return [
            item for item in response.items if item.memory and item.memory.strip()
        ]

    def _resolve_span(self, raw_source: str, chunk: str) -> str:
        """Use the LLM span if it is verbatim from the chunk; else fall back."""
        span = (raw_source or "").strip()
        if span and _normalized(span) in _normalized(chunk):
            return span
        return chunk

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
                            span = self._resolve_span(
                                item.source, chunk_stripped
                            )
                            q = " ".join(
                                qq.strip()
                                for qq in item.queries
                                if qq and qq.strip()
                            )
                            embed_parts = [memory]
                            if q:
                                embed_parts.append(f"Queries: {q}")
                            embed_parts.append(f"{speaker}: {span}")
                            embed_text = "\n".join(embed_parts)
                            bm25_text = memory
                            aliases = _date_aliases(event.timestamp, memory)
                            if aliases:
                                embed_text = f"{embed_text}\nDates: {aliases}"
                                bm25_text = f"{bm25_text}\nDates: {aliases}"
                            block_text = (
                                span if self._block_uses_span else terse
                            )
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=block_text),
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
