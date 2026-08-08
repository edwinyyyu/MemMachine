"""Deictic-resolved verbatim segmenter -- v4.

v3 examples leaked test-domain structure (e.g. "Have you ever tried
sushi?", "All you need is a gamepad", "I love this series" + fantasy
books, "I made this car... my Subaru") -- the LLM was partially
pattern-matching its outputs against the prompt examples rather than
reasoning from rules. v3's pass rate was inflated.

v4 design:

  - State each rule as a PRINCIPLE, no sentence-level examples.
  - Where the principle is hard to verbalize (e.g. exactly where to
    place a vocative), give only the PRONOUN-LEVEL transformation
    pattern -- "you" -> "you, {addressee}" -- no surrounding sentence.
  - Generic-you exception expressed by a SELF-CHECKABLE TEST: replace
    "you" with "a person"; if the sentence is still true / makes
    sense, "you" is generic.
  - Ownership for demonstratives is principle-stated: if the
    surrounding turns establish the referent BELONGS TO {speaker},
    use first-person possessive; otherwise use the noun phrase the
    surrounding turns established.
  - Hardest single rule kept: rule 6 NEVER substitute speaker name,
    rewrite self-3p to first person.
  - Output: plain-text completion (no JSON, no pydantic).
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


PROMPT_DEICTIC_RESOLVED_V4 = """\
SPEAKER: {speaker}
ADDRESSEE: {addressee}

{speaker} sent the message below to {addressee} in their two-person \
chat. Rewrite the message so a reader who has not seen the surrounding \
turns understands every reference. Apply the resolution rules below; \
keep everything else word-for-word.

Resolution rules:

1. ADDRESSED "you". When "you" / "your" / "yours" / "yourself" makes \
a claim about {addressee} (their action, state, possession, opinion) \
or asks them a question, ADD "{addressee}" beside the pronoun as a \
vocative; KEEP the pronoun. Use these pronoun-level patterns:
- "you"        -> "you, {addressee}"
- "your X"     -> "your, {addressee}'s, X"
- "yours"      -> "yours, {addressee}'s"
- "yourself"   -> "yourself, {addressee}"

2. GENERIC "you". When "you" means "anyone" / "a person" (typical in \
advice, hypotheticals, general truths), KEEP it unchanged. Self-check \
test: substitute "a person" for "you"; if the sentence still makes \
sense as a general statement, the "you" is generic and stays \
unchanged.

3. THIRD-PERSON pronouns ("he" / "she" / "they" / "him" / "her" / \
"them" / "his" / "hers" / "theirs") that refer to a person or thing \
named in the surrounding turns: replace with the name or noun phrase \
the surrounding turns established. The antecedent is the most \
recently introduced named subject in the surrounding turns that fits \
the pronoun's number and animacy.

4. DEMONSTRATIVES "this" / "that" / "these" / "those" / "here" / \
"there" that refer to a specific object, place, plan, or event named \
in the surrounding turns: replace with the noun phrase the \
surrounding turns established for that referent. If the surrounding \
turns establish that the referent BELONGS TO {speaker} (they made it, \
own it, built it, restored it, planted it, drew it), use first-person \
possessive ("my X") instead of the noun phrase.

5. AMBIGUOUS referent. When a third-person pronoun or demonstrative \
has no clear antecedent in the surrounding turns, KEEP it unchanged. \
Do not guess.

6. NEVER substitute "{speaker}" anywhere in the rewrite. When a \
third-person pronoun refers to {speaker} themselves, rewrite it to \
first person "I" / "my" / "me" (NOT the speaker's name). The display \
prepends "{speaker}: " so first person reads correctly. Pronoun-level \
patterns:
- "he" / "she" / "they"  -> "I"
- "him" / "her" / "them" -> "me"
- "his" / "her" / "their" -> "my"
- "himself" / "herself" / "themself" -> "myself"

7. KEEP UNCHANGED:
- "I" / "my" / "me" / "mine" / "myself" referring to {speaker}.
- TEMPORAL references such as "yesterday", "last week", "tomorrow", \
"next Friday", "the weekend", "today", "tonight", "now", "just", \
"recently", "X years ago", "this morning". The event date is \
rendered alongside; downstream tooling resolves these at answer time.
- Vocatives already present at the start of the message ("Hey \
{addressee}!", "{addressee}, ..."): they already identify the \
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
    for ev in list(before) + list(after):
        if ev.producer and ev.producer != speaker:
            return ev.producer
    return "the other person"


class DeicticResolvedV4Segmenter(Segmenter):
    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DEICTIC_RESOLVED_V4,
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
