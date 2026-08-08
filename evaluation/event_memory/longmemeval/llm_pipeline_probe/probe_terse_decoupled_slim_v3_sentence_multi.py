"""SENTENCE-MULTI: per-sentence LLM rewrite + queries as separate embeddings.

Architecture
------------

The user's hypothesis: on LoCoMo, sentence-level retrieval (vector
match on individual sentences) was a strong baseline -- but plain
sentence splitting loses neighboring context (pronouns unresolved,
deictic phrases unanchored). Augment each sentence's embedding with a
context-resolved LLM rewrite + queries, so the sentence becomes
retrievable on its own. Display: still retrieve at the whole-MESSAGE
granularity (block.text = whole verbatim message, like rawev).

Implementation choice: single LLM call per message with structured
output containing per-sentence items. Avoids the N-fold input-token
cost of one-call-per-sentence; lets the LLM cross-reference sentences
within the message. Surrounding events (~8 prior, 8 later) are passed
as context for reference resolution but no items are emitted for them.

Schema (per segment / message)
------------------------------

  block.text                  = whole verbatim message (answerer-visible)
  text_to_score_bm25         = concatenated rewrites + dates
  text_to_embed (fallback)   = same as bm25 text
  properties:
    embed_n                  = N (number of sentences)
    embed_0                  = "{rewrite_0}\\nQueries: {q1} {q2} {q3}"
    embed_1                  = "{rewrite_1}\\nQueries: ..."
    ...

The MultiComponentDeriver picks up embed_0..embed_{N-1} and emits one
Derivative per sentence; the vector store ends up with N rows per
segment, all mapped to the same segment_uuid. Retrieval dedups by
segment_uuid (best score wins), so a strong sentence match returns
the whole message. Block.text is shown to the answerer verbatim.
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


PROMPT_SENTENCE_MULTI = """\
You convert one chat MESSAGE into per-sentence retrieval anchors. \
Each anchor is a short rewrite + queries for ONE sentence-sized claim \
from the message, written so a future search can retrieve THAT \
sentence's content even if the question phrases things differently.

For each sentence in the MESSAGE, emit an item if the sentence \
asserts a fact about the world that exists independent of the act of \
speaking it, and that fact is not already established by the prior \
turns. Emit items in source order; the list may be empty.

Also emit a `summary` field: a single, coherent third-person \
statement that synthesizes all the items into one topic-level \
sentence about {speaker}. The summary should mention every \
particular from the items in resolved-pronoun form. Empty string \
if there are no items.

Each item has THREE fields:

(A) "sentence" -- the verbatim slice of the MESSAGE this item covers. \
Copy it exactly. Use this as a span anchor; the LLM should be able to \
locate this string inside the message if asked.

(B) "rewrite" -- a third-person, context-resolved statement of (A) \
about {speaker}. Resolve "I"/"my" to {speaker}; resolve "you" to the \
person addressed; resolve this/that/there/then to what they refer to \
(use the NEIGHBORING TURNS shown above the message for resolution \
context). Keep every particular from (A) -- names, places, numbers, \
identifiers, decisions, opinions, attached-media details, quoted \
wording. Date handling: the message's own date ({date}) is attached \
automatically at display time, so the rewrite must never write {date}; \
KEEP every relative time phrase verbatim ("yesterday", "last week", \
"three years ago", "next Friday", "the weekend", "today", "recently", \
"now", "just"). Tight readable prose, full sentence; no narrative \
connective tissue.

(C) "queries" -- 1 to 3 short questions a user might later ask that \
THIS sentence answers. Vary their angle (who/what/when/where/why/how). \
Phrase them as a person would ask; never include the answer. The \
queries are seen only by the retriever, so include the subject name \
explicitly when relevant -- the queries should be answerable from the \
rewrite alone.

Output items in source order. If the message has no kept content, \
return an empty list.

Treat any NEIGHBORING TURNS shown as context for resolving references \
only; never emit items for them.

Return JSON: {{"items": [{{"sentence": "...", "rewrite": "...", \
"queries": ["...", "..."]}}, ...], "summary": "..."}}.

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


def _date_aliases_verbose(event_date: datetime, joined_text: str) -> str:
    dates: set[tuple[int, int, int]] = {
        (event_date.year, event_date.month, event_date.day)
    }
    for match in _ISO_RE.finditer(joined_text):
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
    summary: str = ""  # whole-message LLM synthesis (single coherent topic statement)


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


EMBED_COMPONENT_COUNT_KEY = "embed_n"
EMBED_COMPONENT_KEY_PREFIX = "embed_"


class SentenceMultiSegmenter(Segmenter):
    """Single LLM call per message → per-sentence items → multi-embed.

    One Segment per message (block.text = verbatim message). N
    components stored in segment.properties for the
    MultiComponentDeriver to emit N derivatives.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_SENTENCE_MULTI,
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
    ) -> tuple[list[_SentenceItem], str]:
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
            return [], ""
        items = [
            item for item in response.items if item.rewrite and item.rewrite.strip()
        ]
        return items, (response.summary or "").strip()

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
                        items, summary = await self._rewrite_chunk(
                            chunk_stripped, speaker, date_str, neighbors_block
                        )
                        if not items:
                            continue
                        # Build per-sentence components.
                        components: list[str] = []
                        rewrites_joined: list[str] = []
                        for item in items:
                            rewrite = item.rewrite.strip()
                            if not rewrite:
                                continue
                            qs = " ".join(
                                q.strip() for q in item.queries if q and q.strip()
                            )
                            comp = rewrite if not qs else (f"{rewrite}\nQueries: {qs}")
                            components.append(comp)
                            rewrites_joined.append(rewrite)
                        if not components:
                            continue
                        # Hybrid multi-vector: also add a single
                        # whole-message-synthesis component from the
                        # LLM's `summary` field (a single coherent
                        # third-person statement, like SIDECAR's
                        # `memory` field). Falls back to concatenation
                        # if the summary is empty.
                        if len(rewrites_joined) > 1:
                            # Synthesis = joined rewrites + verbatim source
                            # message. The rewrites give multi-aspect
                            # resolved-pronoun coverage; the verbatim
                            # gives lexical fidelity for c4 matches.
                            synth_parts = list(rewrites_joined)
                            synth_parts.append(chunk_stripped)
                            components.append("\n".join(synth_parts))
                        # Date aliases (computed over all rewrites for
                        # absolute-date BM25/embedding tokens).
                        all_rewrites = " ".join(rewrites_joined)
                        date_aliases = _date_aliases_verbose(
                            event.timestamp, all_rewrites
                        )
                        # BM25 text = joined rewrites + dates aliases
                        # (single string, single BM25 score per segment).
                        bm25_parts = list(rewrites_joined)
                        if date_aliases:
                            bm25_parts.append(f"Dates: {date_aliases}")
                        bm25_text = "\n".join(bm25_parts)
                        # text_to_embed fallback (used only if the
                        # deriver can't read embed_N keys -- shouldn't
                        # fire when --deriver multi is used).
                        embed_text = "\n".join(components)
                        # Properties: pack components for the deriver.
                        properties = dict(event.properties)
                        properties[EMBED_COMPONENT_COUNT_KEY] = str(len(components))
                        for i, comp in enumerate(components):
                            properties[f"{EMBED_COMPONENT_KEY_PREFIX}{i}"] = comp
                        # Display text = whole verbatim message
                        # (single segment per chunk; answerer sees
                        # the conversational source like rawev).
                        segments.append(
                            Segment(
                                uuid=uuid4(),
                                event_uuid=event.uuid,
                                index=block_index,
                                offset=offset,
                                timestamp=event.timestamp,
                                block=TextBlock(text=chunk_stripped),
                                context=DecoupledRetrievalContext(
                                    producer=speaker,
                                    text_to_embed=embed_text,
                                    text_to_score_bm25=bm25_text,
                                ),
                                properties=properties,
                            )
                        )
                        offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
