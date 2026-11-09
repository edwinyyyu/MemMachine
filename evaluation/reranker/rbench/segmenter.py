"""EventMemory's TextSegmenter split, copied so the recursive chunker is exactly MemMachine's.

Source: MemMachine #1855 (7e46f8b2a), plus a bare "\\n" separator right after ".\\n" (the
design approved on 2026-10-26). Segments join back to the text exactly, and none is longer
than the limit, whitespace included. Whitespace between two segments ends the earlier one
through its last newline; the rest (a line's indentation) begins the later one. Embedding
and reranking use each segment with its edge whitespace stripped, as EventMemory's
TextDeriver does.
"""

import re
from collections.abc import Sequence
from itertools import pairwise

# Split points in order of preference; the empty string splits between
# characters. A separator stays at the end of the text it follows.
SEPARATORS = (
    "\n\n",
    "],\n",
    "},\n",
    "),\n",
    "]\n",
    "}\n",
    ")\n",
    ",\n",
    "？\n",  # Fullwidth question mark
    "?\n",
    "！\n",  # Fullwidth exclamation mark
    "!\n",
    "。\n",  # Ideographic full stop
    ".\n",
    "\n",
    "？",  # Fullwidth question mark
    "? ",
    "！",  # Fullwidth exclamation mark
    "! ",
    "。",  # Ideographic full stop
    ". ",
    "; ",
    ": ",
    "—",
    "--",
    "，",  # Fullwidth comma
    "、",  # Ideographic comma
    ", ",
    "​",  # Zero-width space
    " ",
    "",
)


def split(text: str, max_length: int, separators: Sequence[str] = SEPARATORS) -> list[str]:
    """Split text into chunks of at most `max_length` that join to it.

    Cuts at `_cuts` for the first separator the text contains, then merges
    adjacent pieces while they fit. A piece of `max_length` or more is split
    again at the separators after that one. The last separator must be the
    empty string, which cuts between characters; a piece still over
    `max_length` there is cut every `max_length` code points.
    """
    position, separator = next(
        (position, separator)
        for position, separator in enumerate(separators)
        if not separator or separator in text
    )
    remaining = separators[position + 1 :]

    chunks: list[str] = []
    chunk = ""
    for start, end in pairwise([0, *_cuts(text, separator), len(text)]):
        piece = text[start:end]
        if len(piece) >= max_length:
            if chunk:
                chunks.append(chunk)
                chunk = ""
            if remaining:
                chunks.extend(split(piece, max_length, remaining))
            else:
                chunks.extend(
                    piece[offset : offset + max_length]
                    for offset in range(0, len(piece), max_length)
                )
        elif len(chunk) + len(piece) > max_length:
            chunks.append(chunk)
            chunk = piece
        else:
            chunk += piece
    if chunk:
        chunks.append(chunk)
    return chunks


def _cuts(text: str, separator: str) -> list[int]:
    """Cut after each occurrence of `separator`, in ascending order.

    A cut that touches whitespace moves to where `_whitespace_cut` cuts that
    whitespace, and is dropped if the whitespace has no cut.
    """
    cuts: list[int] = []
    run_end = -1
    run_cut: int | None = None
    for match in re.finditer(re.escape(separator), text):
        cut: int | None = match.end()
        if (cut > 0 and text[cut - 1].isspace()) or (
            cut < len(text) and text[cut].isspace()
        ):
            if cut > run_end:
                run_end, run_cut = _whitespace_cut(text, cut)
            cut = run_cut
        if cut is not None and 0 < cut < len(text) and (not cuts or cut > cuts[-1]):
            cuts.append(cut)
    return cuts


def _whitespace_cut(text: str, position: int) -> tuple[int, int | None]:
    """Find where the whitespace run at `position` ends and where it is cut.

    The text before the run takes it through its last newline, or all of it
    if it has none. A run at either end of the text belongs to the text it
    touches, so it has no cut (None).
    """
    start = position
    while start > 0 and text[start - 1].isspace():
        start -= 1
    end = position
    while end < len(text) and text[end].isspace():
        end += 1
    if start == 0 or end == len(text):
        return end, None
    return end, text.rfind("\n", start, end) + 1 or end
