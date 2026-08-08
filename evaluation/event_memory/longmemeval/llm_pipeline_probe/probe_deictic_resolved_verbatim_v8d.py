"""Deictic-resolved verbatim segmenter -- v8d.

V8C + two WHEN TO EDIT eligibility checks moved to the front of the
question list (and the now-redundant autoregression bullet removed
from HOW TO EDIT):

  - Fixed-expression pronoun check: "to top it off", "take it easy",
    "you know", etc. The pronoun is part of a frozen multi-word
    expression and has no real antecedent. This catches the D28:3
    FULL idiom-corruption pattern at the WHEN gate rather than
    relying on the model to reason about it later.
  - Already-resolved check: if an earlier same-message occurrence of
    the same reference has already been substituted, subsequent
    occurrences are already anchored within the output and stay as
    the original pronouns. This catches the over-substitution-
    repetition pattern (D30:14 3x "Dave's picture", D2:6 3-4x
    "Calvin's car", D28:20 3x "my car", etc.) at the WHEN gate.

HOW TO EDIT now has three principles: topic-shift attention,
ownership-inference guard, output-full-message preservation. The
autoregressive co-reference principle is dropped because its
operative effect is now the WHEN-TO-EDIT "already resolved" check.
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


PROMPT_DEICTIC_RESOLVED_V8D = """\
You will edit a chat MESSAGE for indexing in an information \
retrieval system. The version you produce — not the original — is \
what gets indexed and later retrieved. The reader at retrieval \
time will have no access to the surrounding context.

An indexed message is useful at retrieval time only if it \
accurately preserves the meaning and nuances of what the speaker \
wrote. Your edited version preserves the speaker's wording, \
except for edits that resolve references. Each such edit anchors \
a reference ("you", "it", "they", "this", "that", etc.) to its \
concrete antecedent from the surrounding context, making it \
standalone. Edit only when the resolution is obvious; otherwise \
the reference stays as the speaker wrote it.

WHEN TO EDIT — for each pronoun or demonstrative in the MESSAGE, \
first check whether it is eligible for editing:
- Is the pronoun or demonstrative part of a fixed expression that \
has no actual referent? (e.g., "it's raining", "you know", "this \
and that") If yes, leave it unchanged.
- Would substituting the reference require copying a passage from \
the surrounding context rather than a single noun phrase? (e.g., \
"that said", "in light of this") If yes, leave it unchanged.
- Has an earlier occurrence of the same reference in this message \
already been resolved? If yes, leave it unchanged — the entity is \
already named in your output and the reader will follow.

If the reference is eligible, ask:
- Is the antecedent not just plausible, but obvious, from the \
surrounding context? If not, leave it unchanged.
- Would another careful reader pick the same antecedent? If not, \
leave it unchanged.
- Is there any alternative interpretation a careful reader might \
prefer? If yes, leave it unchanged.

WHAT EDITS TO MAKE — when a reference passes the checks above, \
apply the matching pattern:
- A "you" addressed to a specific person identifiable from the \
surrounding context gains a vocative with that person's name, at \
the start or end of the sentence. For example, if the addressee is \
Alice, "Do you like pizza?" becomes "Do you like pizza, Alice?", \
not "Do you, Alice, like pizza?". The original "you" stays.
- A third-person pronoun or demonstrative ("it", "they", "this", \
"that", "those", etc.) with one unambiguous antecedent in the \
surrounding context is replaced by the established noun phrase.

HOW TO EDIT — apply these principles to every edit you consider:
- Pay attention to subtle or discreet topic changes in the \
surrounding context. When the conversation shifts subject — a \
follow-up question reframes the topic, an aside introduces a new \
subject, a back-channel acknowledgement returns to an earlier one \
— a noun phrase tied to the prior topic is not the antecedent of \
a reference in the new topic.
- Do not infer ownership unless the surrounding context makes it \
abundantly clear with words like "my", "your", "their". \
Demonstratives like "this" or "that" do not establish ownership, \
and neither does someone simply sharing a photo or describing \
something. When ownership is not stated, use a neutral description \
("the X", "a X") rather than an ownership form ("my X", "your X").
- Output the full edited message. Substitutions replace specific \
words; they do not delete or compress content. Every sentence and \
clause of the original message appears in your output.

WHAT STAYS UNCHANGED — everything in the message outside these \
edits stays as the speaker wrote it. The speaker and timestamp \
are part of the message's metadata, so references anchored to \
them remain as written:
- First-person voice ("I", "my", "we", "our") — anchored by the \
speaker.
- Temporal references ("yesterday", "last week", "today") — \
anchored by the timestamp.

Output the resulting message text and nothing else — no speaker \
prefix, no timestamp, no commentary or preamble. Preserve the \
line structure of the original message.

SURROUNDING CONTEXT:
{neighbors_block}MESSAGE BY {speaker}:
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


def _date_aliases(event_date: datetime, text: str) -> str:
    dates: set[tuple[int, int, int]] = {
        (event_date.year, event_date.month, event_date.day)
    }
    for m in _ISO_RE.finditer(text):
        y, mo = int(m.group(1)), int(m.group(2))
        d = int(m.group(3)) if m.group(3) else 0
        if 1 <= mo <= 12:
            dates.add((y, mo, d))
    parts: list[str] = []
    seen: set[str] = set()
    for y, mo, d in sorted(dates):
        name = _MONTHS[mo - 1]
        for alias in (f"{name} {y}", f"{name} {d}, {y}" if d else None):
            if alias and alias not in seen:
                seen.add(alias)
                parts.append(alias)
    return "; ".join(parts)


def _format_neighbors(before: list, after: list) -> str:
    lines: list[str] = []
    for ev in list(before) + list(after):
        lines.append(f"- {ev.producer}: {ev.text}")
    return "\n".join(lines) + ("\n" if lines else "")


class DeicticResolvedV8DSegmenter(Segmenter):
    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DEICTIC_RESOLVED_V8D,
        chunk_size: int = 1500,
        max_attempts: int = 3,
    ) -> None:
        self._lm = language_model
        self._prompt = prompt_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            keep_separator="end",
        )

    async def _resolve(
        self,
        chunk: str,
        speaker: str,
        neighbors_block: str,
    ) -> str:
        prompt = self._prompt.format(
            speaker=speaker,
            passage=chunk,
            neighbors_block=neighbors_block,
        )
        text, _ = await self._lm.generate_response(
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        return text.strip() if text else ""

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
                        resolved = await self._resolve(
                            chunk_stripped, speaker, neighbors_block
                        )
                        if not resolved:
                            continue
                        date_aliases = _date_aliases(event.timestamp, resolved)
                        display_text = f"{speaker}: {resolved}"
                        embed_text = display_text
                        bm25_text = display_text
                        if date_aliases:
                            bm25_text = f"{bm25_text}\nDates: {date_aliases}"
                            embed_text = f"{embed_text}\nDates: {date_aliases}"
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
                        offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block: {type(block).__name__}"
                    )
        return segments
