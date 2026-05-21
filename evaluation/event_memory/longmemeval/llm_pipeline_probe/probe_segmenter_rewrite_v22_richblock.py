"""v22-richblock -- expand block.text (BM25 input) with keywords.

Hypothesis: with --answer-with-raw-events, block.text feeds BM25 \
scoring but NOT the displayed answer context. We can therefore \
enrich block.text with extra anchors (keywords, hypothetical \
question) so the lexical channel of fusion gets more match surface \
WITHOUT bloating ctx tokens.

block.text = "{memory}\\nQ: {question}\\nTerms: {kw1}, {kw2}, ..."

text_to_embed = "{memory}\\nQ: {question}\\nTerms: {kw1}, {kw2}, ...\\n{speaker}: {raw_chunk}"

Both channels (vector + BM25) get the rich signal. The raw_chunk \
remains in text_to_embed for dual-text embedding, omitted from \
block.text since BM25 already gets the raw event content via the \
displayed timestamped raw event line.

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


PROMPT_REWRITE_V22_RICHBLOCK = """\
Rewrite the MESSAGE into a JSON list of memory items. Each item \
contains a third-person memory statement, a hypothetical user \
question, and a list of keyword anchors.

KEEP specific content (names, places, dates, numbers, decisions, \
plans, preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions). Dropping specific content is a FAILURE; \
emitting an item for interchangeable content is a FAILURE.

ONE item per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of the same occurrence is ONE item that \
contains all of its particulars. Distinct events (different times, \
different occasions, different actions) each get their own item.

EACH ITEM has THREE fields:

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

(B) "question": ONE hypothetical question (8-18 words) that a future \
user might ask to retrieve this memory. The question MUST NOT contain \
the answer or restate the memory; it should be phrased like a real \
user query. Use natural question forms (``What did X do when ...``, \
``Why does X ...``, ``When did X ...``, ``Where is X ...``, ``Who \
told X about ...``, ``How does X feel about ...``). End with a \
question mark.

(C) "keywords": a list of 3-8 short keyword anchors. Each anchor is \
a noun phrase or proper noun (1-4 words) that captures a queryable \
particular: a person, place, organization, named object, activity, \
date, quantity, decision, or distinctive phrase. Do NOT include \
sentence connectors. Each anchor must be a substring or close \
paraphrase of content in the memory. Anchors MUST NOT duplicate \
each other.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

Output: a JSON object with field "items" whose value is a list of \
{{ "memory": "...", "question": "...", "keywords": ["...", "..."] }} \
objects. Use an empty list when the message contains no specific \
content.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _MemoryItem(BaseModel):
    memory: str
    question: str
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


def _build_rich_block(memory: str, question: str, keywords: list[str]) -> str:
    parts = [memory.strip()]
    q = (question or "").strip()
    if q:
        parts.append(f"Q: {q}")
    kws = [k.strip() for k in (keywords or []) if k and k.strip()]
    if kws:
        parts.append(f"Terms: {', '.join(kws)}")
    return "\n".join(parts)


class RewriteSegmenter(Segmenter):
    """v22-richblock -- expand block.text with question + keywords."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_RICHBLOCK,
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
                            rich_block = _build_rich_block(
                                item.memory, item.question, item.keywords
                            )
                            embed_text = f"{rich_block}\n{speaker}: {chunk_stripped}"
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=rich_block),
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
