"""SENTENCE: N segments per message, one per sentence; per-sentence rewrite+queries.

Architecture
------------

Single LLM call per message returns a structured per-sentence list.
Each sentence becomes its OWN Segment:

  block.text                = verbatim sentence (answerer sees raw quotes)
  text_to_embed             = "{sentence}\\n{rewrite}\\nQueries: {qs}"
  text_to_score_bm25        = "{rewrite}\\nQueries: {qs}\\nDates: {da}"

The rewrite is context-resolved using ~8 neighboring turns passed in
the prompt, AND resolves relative time phrases to absolute dates inline
(the rewrite is never shown to the answerer, only embedded; block.text
preserves the speaker's verbatim phrasing).

Single derivative per segment (standard WholeTextDeriver), so no
shared/duplicated components across sentence-segments from the same
message -- a hazard the prior SIDECAR-MULTI design fell into when the
shared whole-message chunk appeared as a duplicated derivative across
N segments from the same chunk.

K (max_num_segments) should be rescaled UP relative to the per-message
pipelines: each segment is sentence-sized, so to hit the same ~340
token budget, K~25-30 fits where K=10 fit at the message level.

Schema
------

class _SentenceItem:
    sentence: str       # verbatim slice of MESSAGE (becomes block.text)
    rewrite: str        # 3p, context-resolved, ABSOLUTE dates inline
    queries: list[str]  # 1-3 short questions
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    FirstPersonDecoupledRetrievalContext,
    ProducerContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel


PROMPT_SENTENCE = """\
You convert one chat MESSAGE into per-sentence retrieval anchors. \
Each anchor stands alone: a future search using natural-language \
questions should retrieve THE SENTENCE that answers it.

Each item corresponds to a sentence whose claim names a particular \
subject (the speaker, the addressee, or another entity nameable \
without reference to the prior turns) AND asserts a specific \
attribute, action, relation, or state of that subject that is not \
already asserted by the prior turns. Output items in source order; \
the list may be empty.

Each item has THREE fields:

(A) "sentence" -- the VERBATIM slice of the MESSAGE this item covers. \
Copy it exactly, character-for-character. No paraphrase, no edits. The \
slice should be a contiguous run of message text. This becomes the text \
shown to the reader for THIS item.

(B) "rewrite" -- a third-person, context-resolved statement of (A) \
about {speaker}. Resolve "I"/"my" to {speaker}; resolve "you" to the \
person addressed; resolve this/that/there/then to what they refer to \
(use the NEIGHBORING TURNS shown above the message for resolution \
context). Keep every particular from (A) -- names, places, numbers, \
identifiers, decisions, opinions, attached-media details, quoted \
wording. Date handling: the message's own date ({date}) is attached \
automatically at display time of the SENTENCE -- the rewrite is seen \
ONLY by the retriever. Resolve every relative time reference in the \
rewrite to an absolute date: "yesterday" → "on YYYY-MM-DD"; "last \
week" → "in week-of YYYY-MM-DD"; "three years ago" → "in YYYY"; \
"next Friday" → "on YYYY-MM-DD"; etc. Tight readable prose, full \
sentence; no narrative connective tissue.

(C) "queries" -- 1 to 3 short questions a user might later ask that \
THIS sentence answers. Vary their angle (who/what/when/where/why/how). \
Phrase them as a person would ask; include the subject's name when \
relevant so the query stands alone. Never include the answer.

Output items in source order. Empty list if the message has no kept \
content.

Treat any NEIGHBORING TURNS shown as context for resolving references \
only; never emit items for them.

Return JSON: {{"items": [{{"sentence": "...", "rewrite": "...", \
"queries": ["...", "..."]}}, ...]}}.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def _date_aliases_from_text(event_date: datetime, text: str) -> str:
    """Natural-form date aliases for any ISO date in `text` + event date."""
    dates: set[tuple[int, int, int]] = {
        (event_date.year, event_date.month, event_date.day)
    }
    for match in _ISO_RE.finditer(text):
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


class _SentenceItem(BaseModel):
    sentence: str
    rewrite: str
    queries: list[str]


class _RewriteResponse(BaseModel):
    items: list[_SentenceItem]


_WS_RE = re.compile(r"\s+")


def _locate(needle: str, haystack: str) -> int:
    """Find ``needle`` in ``haystack`` with whitespace tolerance.

    Returns the start index in ``haystack`` of the first match, or -1.
    Tries exact substring first, then a whitespace-collapsed pass that
    maps positions back to the original ``haystack``.
    """
    needle = needle.strip()
    if not needle:
        return -1
    idx = haystack.find(needle)
    if idx != -1:
        return idx
    # Whitespace-tolerant: collapse both, find, map back.
    haystack_norm = _WS_RE.sub(" ", haystack)
    needle_norm = _WS_RE.sub(" ", needle)
    norm_idx = haystack_norm.find(needle_norm)
    if norm_idx == -1:
        return -1
    # Map norm_idx back to original haystack by walking and counting
    # non-whitespace runs.
    src = 0
    dst = 0
    while src < len(haystack) and dst < norm_idx:
        c = haystack[src]
        if c.isspace():
            if dst > 0 and haystack_norm[dst - 1] != " ":
                # shouldn't happen but defensive
                pass
            # Skip whitespace in source; normalized has a single space
            # at the same dst position when first encountered.
            if dst < len(haystack_norm) and haystack_norm[dst] == " ":
                dst += 1
            while src < len(haystack) and haystack[src].isspace():
                src += 1
        else:
            src += 1
            dst += 1
    return src


def _assign_block_texts(
    items: list["_SentenceItem"], chunk_text: str
) -> list[tuple["_SentenceItem", str]]:
    """Greedy consumer: each item's block_text spans [pos_i, pos_{i+1}).

    Each block_text includes its own sentence + any trailing
    whitespace/punctuation/inter-sentence text up to the next kept
    item's start position. The text BEFORE the first kept item
    (i.e. chunk_text[0:positions[0]]) is DROPPED -- per the
    "don't include leading" rule. Items whose sentence cannot be
    located in ``chunk_text`` are dropped.

    Concatenation of all returned block_texts reproduces
    ``chunk_text[positions[0]:]`` -- the verbatim source from the
    first kept sentence onward (inter-sentence content preserved,
    pre-first-sentence content discarded).
    """
    located: list[tuple[int, "_SentenceItem"]] = []
    for item in items:
        pos = _locate(item.sentence, chunk_text)
        if pos == -1:
            continue
        located.append((pos, item))
    located.sort(key=lambda x: x[0])
    if not located:
        return []
    result: list[tuple["_SentenceItem", str]] = []
    for i, (pos, item) in enumerate(located):
        start = pos
        end = located[i + 1][0] if i + 1 < len(located) else len(chunk_text)
        block_text = chunk_text[start:end]
        if block_text.strip():
            result.append((item, block_text))
    return result


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


class SentenceSegmenter(Segmenter):
    """N segments per message; one per sentence emitted by the LLM.

    Each sentence-segment carries its OWN unique embed/bm25 string
    (sentence + rewrite + queries). No shared whole-message component
    -- the rewrite already carries resolved-context meaning, and a
    shared component across sentence-segments from the same message
    would create duplicate vectors in the index.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_SENTENCE,
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
    ) -> list[_SentenceItem]:
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
            if item.sentence
            and item.sentence.strip()
            and item.rewrite
            and item.rewrite.strip()
        ]

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
                        # Greedy consumer: each item gets the verbatim
                        # span [pos_i, pos_{i+1}) from chunk_stripped.
                        # Leading garbage absorbs into the first item;
                        # any whitespace/punctuation between sentences
                        # stays attached to the prior item. Concat of
                        # all block_texts reconstructs the source.
                        assigned = _assign_block_texts(items, chunk_stripped)
                        for item, block_text in assigned:
                            sentence = item.sentence.strip()
                            rewrite = item.rewrite.strip()
                            if not sentence or not rewrite:
                                continue
                            qs = " ".join(
                                q.strip() for q in item.queries if q and q.strip()
                            )
                            # Per-sentence date aliases derived from
                            # the absolute dates the LLM emitted in
                            # rewrite. Event-date alias always present.
                            date_aliases = _date_aliases_from_text(
                                event.timestamp, rewrite
                            )
                            # Embed text: sentence + rewrite ONLY.
                            # Queries are excluded from the embedding
                            # because for short sentence-level segments
                            # the query text dominates the centroid and
                            # pulls the embedding away from the
                            # sentence's specific content. Queries
                            # still go into BM25 (lexical match).
                            embed_text = f"{sentence}\n{rewrite}"
                            # BM25 text: rewrite + queries + dates
                            # (rewrite has names + absolute dates;
                            # queries add natural-language tokens).
                            bm25_parts = [rewrite]
                            if qs:
                                bm25_parts.append(f"Queries: {qs}")
                            if date_aliases:
                                bm25_parts.append(f"Dates: {date_aliases}")
                            bm25_text = "\n".join(bm25_parts)
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=block_text),
                                    context=FirstPersonDecoupledRetrievalContext(
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
