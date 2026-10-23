"""Segmenters for events containing TextBlocks."""

from collections.abc import Iterable
from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    FormatOptions,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)


class TextSegmenter(Segmenter):
    """Segments events via recursive character splitting.

    A text block's segments, joined in offset order, are its text exactly.
    Whitespace between two segments ends the earlier one, so only a block's
    first segment begins with whitespace.
    """

    def __init__(self, max_chunk_length: int = 500) -> None:
        """
        Initialize the segmenter.

        Args:
            max_chunk_length (int):
                Max code-point length for text chunks, not counting
                the whitespace at their ends
                (default: 500).
        """
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_length,
            chunk_overlap=0,
            separators=[
                "\n\n",
                "],\n",
                "},\n",
                "),\n",
                "]\n",
                "}\n",
                ")\n",
                ",\n",
                "\uff1f\n",  # Fullwidth question mark
                "?\n",
                "\uff01\n",  # Fullwidth exclamation mark
                "!\n",
                "\u3002\n",  # Ideographic full stop
                ".\n",
                "\uff1f",  # Fullwidth question mark
                "? ",
                "\uff01",  # Fullwidth exclamation mark
                "! ",
                "\u3002",  # Ideographic full stop
                ". ",
                "; ",
                ": ",
                "—",
                "--",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                ", ",
                "\u200b",  # Zero-width space
                " ",
                "",
            ],
            keep_separator="end",
            strip_whitespace=False,
        )

    @override
    async def segment(
        self,
        event: Event,
        *,
        format_options: FormatOptions | None = None,
    ) -> list[Segment]:
        segments: list[Segment] = []
        for index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
                    chunks = _move_leading_whitespace_back(
                        self._text_splitter.split_text(text)
                    )
                    segments.extend(
                        Segment(
                            uuid=uuid4(),
                            event_uuid=event.uuid,
                            index=index,
                            offset=offset,
                            timestamp=event.timestamp,
                            block=TextBlock(text=chunk),
                            context=event.context,
                            properties=event.properties,
                        )
                        for offset, chunk in enumerate(chunks)
                    )
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments


def _move_leading_whitespace_back(chunks: Iterable[str]) -> list[str]:
    """Move whitespace that begins a chunk to the end of the chunk before it.

    Whitespace before any content begins the first chunk instead, so only
    the first chunk begins with whitespace, and only text of whitespace
    alone yields a chunk of whitespace alone.
    """
    moved: list[str] = []
    leading = ""
    for chunk in chunks:
        content = chunk.lstrip()
        whitespace = chunk[: len(chunk) - len(content)]
        if moved:
            moved[-1] += whitespace
        else:
            leading += whitespace
        if content:
            moved.append(leading + content)
            leading = ""
    if leading:
        moved.append(leading)
    return moved
