"""Deictic-resolved verbatim segmenter -- v2.

v1 inspection showed the LLM treats "verbatim" as primary and skips the
substitution rules in ~90% of cases. When it does substitute, it picks
wrong referents (e.g. replacing generic "you" with the speaker's own
name; grabbing addressee for "this car" when antecedent is in earlier
context).

v2 fixes:
  - Reframe: reference resolution is the PRIMARY task; verbatim applies
    only to unlisted material.
  - Explicit "DO NOT substitute the speaker's own name" guard.
  - "you" exception: generic "one"/"anyone" usage stays unchanged.
  - For "you" addressing the other speaker, use VOCATIVE INSERTION
    ("Are you, Joanna, excited?") rather than pronoun replacement --
    grammatical AND injects the name token for embedding.
  - "with a clear referent in neighboring turns" guard everywhere
    discourages hallucination on ambiguous referents.
  - For self-reference via 3p pronoun, rewrite to FIRST PERSON (not
    speaker's name) -- avoids the "Alice said Alice went home"
    awkwardness; the display prefix "{speaker}: " keeps attribution.

Block.text / text_to_embed / text_to_score_bm25 all use the same
resolved-verbatim text. Speaker prefix prepended to display.
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


PROMPT_DEICTIC_RESOLVED_V2 = """\
You receive one chat MESSAGE from {speaker} in a two-person chat. The \
other speaker (the addressee) appears as a producer name in the \
NEIGHBORING TURNS.

Your task is REFERENCE RESOLUTION. Make every personal and spatial \
deictic in the message point to a concrete referent the reader can \
understand without seeing the neighbors. Apply the resolutions below; \
keep everything else unchanged.

REQUIRED resolutions:

1. "you" / "your" / "yours" / "yourself" addressing the OTHER speaker \
(the addressee shown in neighboring turns): insert a vocative tag with \
the addressee's name, KEEPING the pronoun in place. This adds the \
name token for retrieval while staying grammatical. DO NOT replace \
"you" with the speaker's own name. Examples:
- {speaker}: "Are you excited?" addressed to Joanna → "Are you, \
Joanna, excited?"
- {speaker}: "What did you get?" addressed to Andrew → "What did \
you, Andrew, get?"
- {speaker}: "Have you ever tried it?" addressed to Audrey → \
"Have you, Audrey, ever tried it?"
- EXCEPTION: generic "you" meaning "one" / "anyone" / "a person" \
("you have to be careful in life"): KEEP unchanged.

2. Third-person pronouns ("he"/"she"/"they"/"him"/"her"/"them"/"his"\
/"hers"/"theirs") with a clear referent in the neighboring turns:
- If the referent is {speaker} themselves → rewrite to FIRST \
PERSON "I"/"my"/"me". DO NOT substitute the speaker's own name. The \
display prefixes "{speaker}: " so first person reads correctly. \
Example: "Alice mentioned she went home" with "she" = Alice (the \
speaker) → "Alice mentioned I went home".
- If the referent is a NAMED PERSON OR THING in neighboring turns \
→ replace with the name. Example: "they said they wanted to \
hang out" + neighbor named "the guys at the tournament" → "the \
guys at the tournament said they wanted to hang out".
- If the referent is uncertain (no clear antecedent in neighbors) \
→ KEEP the pronoun unchanged. Do not guess.

3. "this" / "that" / "these" / "those" / "here" / "there" with a \
clear referent in the neighboring turns (an object, photo, place, \
plan, idea) → replace with the referent's noun phrase. Examples:
- "I love this series" + neighbors describe fantasy books → "I \
love this fantasy book series"
- "Where was that taken?" + neighbor shared a photo → "Where was \
the photo taken?"
- "going to that spot" + neighbor named Boston Common → "going \
to Boston Common"
- If the referent is uncertain → KEEP unchanged.

KEEP UNCHANGED (do not modify):

- "I" / "my" / "me" / "mine" / "myself" referring to {speaker}.
- TEMPORAL references: "yesterday", "last week", "tomorrow", "next \
Friday", "the weekend", "today", "tonight", "now", "just", "recently", \
"three years ago", "this morning", etc. The event date is rendered \
alongside the message; dateinstr resolves these at answer time.
- VOCATIVES at the start of the message ("Hey Andrew!", "Calvin, ..."): \
they already identify the addressee.
- All other content: wording, punctuation, capitalization, emoji, \
attached-media descriptions. No paraphrasing, compression, expansion, \
reordering, or commentary.

If the message is interchangeable filler with no content (bare \
greeting, sign-off, thanks, acknowledgement, pleasantry), return \
{{"text": ""}}.

Output JSON: {{"text": "..."}}.

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


class _Response(BaseModel):
    text: str


def _format_neighbors(before: list, after: list) -> str:
    lines: list[str] = []
    if before:
        lines.append("PRIOR TURNS (context only):")
        for ev in before:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    if after:
        lines.append("LATER TURNS (context only):")
        for ev in after:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


class DeicticResolvedV2Segmenter(Segmenter):
    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DEICTIC_RESOLVED_V2,
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
        self, chunk: str, speaker: str, date: str, neighbors_block: str
    ) -> str:
        prompt = self._prompt.format(
            speaker=speaker,
            date=date,
            passage=chunk,
            neighbors_block=neighbors_block,
        )
        resp = await self._lm.generate_parsed_response(
            output_format=_Response,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        return resp.text.strip() if resp else ""

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
                        resolved = await self._resolve(
                            chunk_stripped, speaker, date_str, neighbors_block
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
