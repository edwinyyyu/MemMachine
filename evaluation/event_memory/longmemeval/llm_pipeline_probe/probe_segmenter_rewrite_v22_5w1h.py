"""v22-5w1h -- v22 with explicit who/what/when/where/how/why labels in embedding.

Hypothesis: rather than ONE running statement, give the embedding 6 \
structurally-tagged aspect-slots. Each slot fills in concisely if the \
message has that information. Tagged structure makes the embedding \
match query types directly: "who" queries match WHO content, "when" \
queries match WHEN content, etc.

text_to_embed = "WHO: {who}; WHAT: {what}; WHEN: {when}; WHERE: \
{where}; HOW: {how}; WHY: {why}\\n{speaker}: {raw_chunk}"

block.text remains the running 3p statement (used for BM25 + display \
fallback). Display under --answer-with-raw-events is the raw event.

In-scope: gpt-5.4-nano + text-embedding-3-small.
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


PROMPT_REWRITE_V22_5W1H = """\
Rewrite the MESSAGE into a JSON list of memory items. Each item \
captures one event in the message. A future user querying any \
specific content in the message should find at least one item that \
contains that content.

KEEP what is specific to this message; DROP what is interchangeable \
across similar messages. Specific content includes: names of people, \
places, organizations, brands, named objects, named activities; \
dates, times, durations, quantities; identifiers, titles, quoted \
phrases, proper nouns; decisions, plans, preferences, opinions, \
relationships, roles, emotional states tied to events; described \
events (something that happened or will happen); attached-media \
descriptions. Interchangeable content has none of these. Dropping \
specific content is a FAILURE; emitting an item for interchangeable \
content (bare greeting, "thanks", "lol") is a FAILURE.

EACH ITEM has TWO fields:

(A) "summary": a running third-person statement of the event, \
preserving every concrete particular verbatim (names, dates, numbers, \
identifiers, distinctive phrasing, attached-media descriptions). \
Resolves first-person to {speaker}'s name, second-person to the \
addressee's name when known. Resolves relative time references to \
absolute dates anchored at {date} (point references to YYYY-MM-DD; \
broader span references to YYYY-MM or YYYY). Anchored to the event's \
date. Reports content as fact, not as speech-act. One date per \
summary.

(B) "facets": a JSON object with these short string fields (each \
field 0-15 words; "" if the message does not contain that aspect):
- who: the named subjects/actors/attendees of the event (resolved \
to names)
- what: the action, occurrence, decision, or state expressed
- when: the date, time, duration, or temporal anchor of the event \
(use the resolved absolute form, e.g. 2023-04 for a month-precision \
reference)
- where: the place, venue, or location of the event
- how: the manner, method, or instrumentation
- why: the motivation, cause, or reason

Each facet is a SHORT phrase, not a sentence — just the queryable \
particular. Facets MUST be derived solely from the message; never \
invent.

NEIGHBORING TURNS appear before and after the message strictly to \
help resolve second-person addressees, demonstratives, anaphora, \
and unresolved relative references. Content drawn from the \
neighbors is NEVER emitted -- only content drawn from the message \
itself.

Output: a JSON object with field "items" whose value is a list of \
{{ "summary": "...", "facets": {{ "who": "...", "what": "...", \
"when": "...", "where": "...", "how": "...", "why": "..." }} }} \
objects. Use an empty list when the message contains no specific \
content.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _Facets(BaseModel):
    who: str = ""
    what: str = ""
    when: str = ""
    where: str = ""
    how: str = ""
    why: str = ""


class _MemoryItem(BaseModel):
    summary: str
    facets: _Facets


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


def _format_facets(f: _Facets) -> str:
    parts = []
    for label, val in (
        ("WHO", f.who), ("WHAT", f.what), ("WHEN", f.when),
        ("WHERE", f.where), ("HOW", f.how), ("WHY", f.why),
    ):
        v = (val or "").strip()
        if v:
            parts.append(f"{label}: {v}")
    return "; ".join(parts)


class RewriteSegmenter(Segmenter):
    """v22-5w1h -- summary + tagged 5W1H facets in embedding text.

    block.text = summary (running 3p memory statement, same as v22).
    text_to_embed = facets_line + "\\n" + speaker + ": " + raw_chunk
    (NO leading summary in embed; facets carry the structured retrieval signal).

    Hypothesis: structured tagged facets cluster better with structured
    user queries (who/what/when/where queries) than a running statement.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_5W1H,
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
            if item.summary and item.summary.strip()
        ]

    @staticmethod
    def _build_embed_text(
        summary: str, facets: _Facets, original_chunk: str, speaker: str
    ) -> str:
        f_line = _format_facets(facets)
        if f_line:
            return f"{summary}\n{f_line}\n{speaker}: {original_chunk}"
        return f"{summary}\n{speaker}: {original_chunk}"

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
                                item.summary.strip(),
                                item.facets,
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
                                    block=TextBlock(text=item.summary.strip()),
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
