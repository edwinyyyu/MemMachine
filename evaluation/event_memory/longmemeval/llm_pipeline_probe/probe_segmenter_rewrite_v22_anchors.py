"""v22-anchors -- v22 with extracted key terms in embedding text.

Hypothesis: embedding text whose surface form is a focused list of \
salient noun phrases will match shorter, keyword-dense events better \
than full sentences. Diff from qkey (questions) and paraphrase \
(restatement): just the queryable particulars as anchors.

text_to_embed = "{memory}\nKeywords: {kw1}, {kw2}, ..., {kwN}\n{speaker}: {raw_chunk}"

block.text remains the memory statement.

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


PROMPT_REWRITE_V22_ANCHORS = """\
Rewrite the MESSAGE into a JSON list of memory items. Each item \
contains a third-person memory statement plus a list of keyword \
anchors. A future user querying any specific content in the message \
should find at least one item whose memory OR keyword list contains \
that content.

KEEP specific content (names, places, dates, numbers, decisions, \
plans, preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions). Dropping specific content is a FAILURE; \
emitting an item for interchangeable content is a FAILURE.

ONE item per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of the same occurrence is ONE item that \
contains all of its particulars. Distinct events (different times, \
different occasions, different actions) each get their own item.

EACH ITEM has TWO fields:

(A) "memory": a third-person memory statement about {speaker}. \
Reports content, not the speech-act of conveying it -- ``{speaker} \
said that ...`` / ``{speaker} told X that ...`` wrappers are dropped \
unless the speech-act itself is the event (a promise, an apology, an \
announcement). Refer to {speaker} by name; resolve first-person to \
{speaker}'s name on first occurrence; resolve ``you`` to addressee's \
name when known. Resolve demonstratives to concrete referents. \
Preserve every concrete particular from the message verbatim -- names, \
numbers, identifiers, distinctive phrasing.

Resolve every relative time reference (``yesterday``, ``last week``, \
``three years ago``, ``next Friday``, ``the weekend``, ``today``, \
``tonight``, ``recently``, ``now``, ``just``) to an absolute date \
anchored at {date} (point references to YYYY-MM-DD; broader span \
references to YYYY-MM or YYYY). Use one date per memory; the framework \
prepends the message timestamp automatically when surfacing the \
statement.

(B) "keywords": a list of 3-8 short keyword anchors. Each anchor is a \
noun phrase or proper noun (1-4 words) that captures a queryable \
particular: a person, place, organization, named object, activity, \
date, quantity, decision, or distinctive phrase. Do NOT include verbs \
or sentence connectors. Each anchor must be a substring or close \
paraphrase of content in the memory. Anchors MUST NOT duplicate each \
other.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

Output: a JSON object with field "items" whose value is a list of \
{{ "memory": "...", "keywords": ["...", "..."] }} objects. Use an \
empty list when the message contains no specific content.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _MemoryItem(BaseModel):
    memory: str
    keywords: list[str]


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
    """v22-anchors -- keyword anchors in embed."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_ANCHORS,
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
        memory: str, keywords: list[str], original_chunk: str, speaker: str
    ) -> str:
        kws = [k.strip() for k in keywords if k and k.strip()]
        if kws:
            kw_line = ", ".join(kws)
            return f"{memory}\nKeywords: {kw_line}\n{speaker}: {original_chunk}"
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
                                item.keywords,
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
