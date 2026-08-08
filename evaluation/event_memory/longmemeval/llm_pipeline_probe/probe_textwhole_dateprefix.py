"""Date-prefixed text-whole probe — LLM-less baseline with [date] prepend.

Identical to the production text-whole pipeline (RecursiveCharacterText
chunking + whole-text pass-through) except the embed text gets a
prepended date in brackets and the chunk is quoted:

    text_to_embed = '[Saturday, February 24, 2024] Audrey: "<chunk>"'

block.text and bm25 stay as the standard speaker-prefixed chunk —
this isolates the embed-format change to the semantic-retrieval
channel only.

Used to test the claim that "[<date>] <speaker>: \"<chunk>\"" raises
recall on raw text for the LLM-less stack.
"""

from __future__ import annotations

from datetime import datetime
from typing import override
from uuid import uuid4

from babel.dates import format_date
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


_DATE_STYLE = "full"  # "Saturday, February 24, 2024"
_LOCALE = "en_US"


def _full_date(ts: datetime) -> str:
    return format_date(ts, format=_DATE_STYLE, locale=_LOCALE)


class TextWholeDatePrefixSegmenter(Segmenter):
    """Recursive raw-text chunker that emits segments with
    DecoupledRetrievalContext, prepending [date] to the embed channel.
    Pair with WholeTextDeriver in the ingest pipeline."""

    def __init__(
        self,
        *,
        chunk_size: int = 1500,
    ) -> None:
        self._chunk_size = chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                "​",
                " ",
                "",
            ],
            keep_separator="end",
        )

    @override
    async def segment(self, event: Event) -> list[Segment]:
        match event.context:
            case SurroundingEventsContext(producer=producer):
                speaker = producer
            case ProducerContext(producer=producer):
                speaker = producer
            case _:
                speaker = "the speaker"

        date_str = _full_date(event.timestamp)

        segments: list[Segment] = []
        for block_index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
                    chunks = (
                        self._splitter.split_text(text)
                        if len(text) > self._chunk_size
                        else [text]
                    )
                    for offset, chunk in enumerate(chunks):
                        chunk_stripped = chunk.strip()
                        if not chunk_stripped:
                            continue
                        display_text = f"{speaker}: {chunk_stripped}"
                        embed_text = (
                            f'[{date_str}] {speaker}: "{chunk_stripped}"'
                        )
                        bm25_text = display_text
                        segments.append(
                            Segment(
                                uuid=uuid4(),
                                event_uuid=event.uuid,
                                index=block_index,
                                offset=offset,
                                timestamp=event.timestamp,
                                block=TextBlock(text=display_text),
                                context=DecoupledRetrievalContext(
                                    producer=speaker,
                                    text_to_embed=embed_text,
                                    text_to_score_bm25=bm25_text,
                                ),
                                properties=event.properties,
                            )
                        )
                case _:
                    raise NotImplementedError(
                        f"Unsupported block: {type(block).__name__}"
                    )
        return segments
