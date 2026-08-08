"""Deictic-resolved verbatim segmenter -- v3.

v2 failures and v3 fixes:

  - v2 inserted SPEAKER name as vocative (D28:32, D14:14) because
    {speaker} appears 11x in the template and the model latches on the
    salient name. v3 derives {addressee} explicitly from neighbors and
    passes it as its own template variable -- no "OTHER speaker"
    indirection.

  - v2 ignored the generic-"you" exception. v3 splits the "you" rule
    into two numbered rules (addressed vs generic) with a sharp
    discriminator: "addressed you" makes a claim ABOUT the addressee
    (their action / state / possession) or asks them a question;
    "generic you" is in advice / general statements about anyone.

  - v2 ignored 3p antecedent resolution. v3 promotes 3p / demonstrative
    resolution to numbered rules with worked examples that show the
    antecedent name explicitly in the surrounding turns.

  - v2 burdened the model with JSON schema. v3 uses plain-text
    completion -- one rewritten line, no JSON envelope.

  - "NEVER substitute {speaker}" is hoisted to its own rule and placed
    AFTER the resolution rules so it serves as a final guard.
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


PROMPT_DEICTIC_RESOLVED_V3 = """\
SPEAKER: {speaker}
ADDRESSEE: {addressee}

{speaker} sent the message below to {addressee} in their two-person \
chat. Rewrite the message so a reader who has not seen the surrounding \
turns understands every reference. Apply the resolution rules below; \
keep everything else word-for-word.

Resolution rules:

1. ADDRESSED "you" — when "you" / "your" / "yours" / "yourself" says \
something about {addressee} (their action, their state, their \
possession, or a question to them), insert "{addressee}" as a vocative \
beside the pronoun. The pronoun stays; the vocative just adds the \
name token.
   - "Are you excited?" → "Are you, {addressee}, excited?"
   - "What did you get?" → "What did you, {addressee}, get?"
   - "Have you ever tried sushi?" → "Have you, {addressee}, ever \
tried sushi?"
   - "your car looks awesome" → "your, {addressee}'s, car looks \
awesome"

2. GENERIC "you" — when "you" means "anyone" / "a person" / "one" \
(typical in advice or general statements), KEEP it unchanged. Test: \
the sentence stays true if "you" is replaced with "a person".
   - "All you need is a gamepad" → unchanged (a person needs)
   - "You have to be patient in life" → unchanged (a person has to)
   - "It's incredible when you get those moments of joy" → unchanged

3. THIRD-PERSON pronouns ("he" / "she" / "they" / "him" / "her" / \
"them" / "his" / "hers" / "theirs") with a clear antecedent in the \
surrounding turns — replace with the antecedent's name or noun \
phrase. Antecedents are the most recent named person or group that \
the surrounding turns introduced.
   - "they said they wanted to hang out" + surrounding turn \
"I talked to the guys at the tournament" → "the guys at the \
tournament said the guys at the tournament wanted to hang out"
   - "she has a new dog" + surrounding turn naming Alice → "Alice \
has a new dog"

4. DEMONSTRATIVES "this" / "that" / "these" / "those" / "here" / \
"there" with a clear referent in the surrounding turns (an object, a \
photo, a place, a plan) — replace with the noun phrase that names \
the referent.
   - "I love this series" + surrounding turn describing a fantasy \
book series → "I love the fantasy book series"
   - "Where was that taken?" + surrounding turn that shared a photo \
→ "Where was the photo taken?"
   - "we are going to that spot" + surrounding turn naming Boston \
Common → "we are going to Boston Common"
   - "I made this car look like a beast" + surrounding turns about \
{speaker}'s Subaru restoration project → "I made my Subaru look like \
a beast" (use the OWNERSHIP relationship the surrounding turns \
established — do NOT replace with the addressee's name)

5. AMBIGUOUS referent — when a 3p pronoun or demonstrative has no \
clear antecedent in the surrounding turns, KEEP it unchanged. Do not \
guess.

6. NEVER substitute "{speaker}" anywhere in the rewrite. If a 3p \
pronoun refers to {speaker} themselves, rewrite to first person \
"I" / "my" / "me" (NOT the speaker's name). The display prefixes \
"{speaker}: " so first person reads correctly.
   - "{speaker} mentioned she was excited" with "she" = {speaker} → \
"{speaker} mentioned I was excited"

7. KEEP UNCHANGED:
   - "I" / "my" / "me" / "mine" / "myself" referring to {speaker}.
   - TEMPORAL references: "yesterday", "last week", "tomorrow", "next \
Friday", "the weekend", "today", "tonight", "now", "just", "recently", \
"three years ago", "this morning", etc. The event date is rendered \
alongside the message; downstream tooling resolves these at answer \
time.
   - Vocatives that are already there at the start of the message \
("Hey {addressee}!", "{addressee}, ..."): they already identify the \
addressee.
   - All other content: wording, punctuation, capitalization, emoji, \
attached-media descriptions. No paraphrasing, compression, expansion, \
reordering, or commentary.

OUTPUT FORMAT: write ONLY the rewritten message on a single line. No \
JSON, no quotes, no preamble, no commentary. If the message is \
interchangeable filler (a bare greeting, sign-off, thanks, \
acknowledgement, pleasantry) with no informational content, output \
nothing (an empty line).

{neighbors_block}MESSAGE TO REWRITE (from {speaker} to {addressee}):
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


def _derive_addressee(speaker: str, before: list, after: list) -> str:
    """Two-person chat: addressee = the producer in neighbors who isn't speaker."""
    for ev in list(before) + list(after):
        if ev.producer and ev.producer != speaker:
            return ev.producer
    return "the other person"


class DeicticResolvedV3Segmenter(Segmenter):
    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DEICTIC_RESOLVED_V3,
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
        addressee: str,
        neighbors_block: str,
    ) -> str:
        prompt = self._prompt.format(
            speaker=speaker,
            addressee=addressee,
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
                addressee = _derive_addressee(speaker, before, after)
                neighbors_block = _format_neighbors(before, after)
            case ProducerContext(producer=producer):
                speaker = producer
                addressee = "the other person"
                neighbors_block = ""
            case _:
                speaker = "the speaker"
                addressee = "the other person"
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
                            chunk_stripped, speaker, addressee, neighbors_block
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
