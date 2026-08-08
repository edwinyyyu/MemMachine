"""Deictic-resolved verbatim segmenter -- MULTIPASS.

The single-pass v2..v7 each accumulated 7+ rules and the model at
gpt-5.4-nano @ low could only respect 3-4 at a time. Defensive
patches conflicted in attention and the model paraphrased.

This variant decomposes the resolution into THREE sequential passes,
each with one focused transformation. Autoregression argument: when
the prompt says "apply ONLY this transformation, preserve every other
word verbatim", the model commits to the source word stream early and
only intervenes at the targeted token type.

Pass 1 -- "you" handling: addressed vs generic discrimination plus
vocative insertion. Other tokens copy through.

Pass 2 -- third-person pronoun handling: substitute names from
neighbors, or first-person if the referent is the speaker. Other
tokens copy through.

Pass 3 -- demonstrative handling: substitute noun phrases from
neighbors, or "my X" if the referent is owned by the speaker. Other
tokens copy through.

Cost: 3 LLM calls per chunk instead of 1. Each prompt is tiny.
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


PROMPT_PASS_1_YOU = """\
SPEAKER: {speaker}
ADDRESSEE: {addressee}

You will rewrite the MESSAGE below. Your ONLY job is to handle \
"you" / "your" / "yours" / "yourself" pronouns. Every other token in \
the MESSAGE must appear in your output unchanged.

For each "you" pronoun, decide:

ADDRESSED (the "you" refers specifically to {addressee}):
- Question to {addressee}: sentence starts with auxiliary + you \
("Are/Do/Did/Have/Has/Can/Will/Would/Could/Should you ...").
- Claim about {addressee}'s SPECIFIC recent action / state / \
possession that the SURROUNDING TURNS just discussed.
For an ADDRESSED "you", INSERT ", {addressee}, " (with surrounding \
commas) immediately AFTER the pronoun. The pronoun itself stays in \
your output, unchanged.
Patterns:
- "you"      becomes  "you, {addressee},"
- "your X"   becomes  "your, {addressee}'s, X"
- "yourself" becomes  "yourself, {addressee},"

GENERIC (the "you" means "anyone" / "a person"):
- "you have to / need to / want to [verb]" (advice)
- "all you need is [noun]"
- "when you [verb]" / "if you [verb]" (hypothetical)
- "you get [noun]" (general experience)
- Generic imperative advice ("have faith in yourself")
For a GENERIC "you", keep the pronoun unchanged. No vocative.

Default to GENERIC if uncertain.

HARD CONSTRAINTS:
- At most one vocative per pronoun token.
- The original "you" / "your" / "yours" / "yourself" tokens must \
appear in your output word-for-word.
- Every other word in the MESSAGE must appear in your output \
unchanged. Do not paraphrase, compress, or rearrange.

OUTPUT: write ONLY the rewritten MESSAGE on a single line. If the \
MESSAGE is interchangeable filler (a bare greeting/sign-off/thanks \
with no informational content), output an empty line.

SURROUNDING TURNS (context only):
{neighbors_block}MESSAGE FROM {speaker} TO {addressee}:
{passage}"""


PROMPT_PASS_2_THIRDPERSON = """\
SPEAKER: {speaker}
ADDRESSEE: {addressee}

You will rewrite the MESSAGE below. Your ONLY job is to handle \
third-person pronouns: "he" / "she" / "they" / "him" / "her" / \
"them" / "his" / "hers" / "theirs" / "himself" / "herself" / \
"themself". Every other token in the MESSAGE must appear in your \
output unchanged.

For each third-person pronoun, decide its antecedent by checking the \
SURROUNDING TURNS:

(a) If the antecedent is {speaker} themselves, substitute first \
person:
- "he" / "she" / "they"   becomes  "I"
- "him" / "her" / "them"  becomes  "me"
- "his" / "her" / "their" becomes  "my"
- "himself" / "herself" / "themself" becomes "myself"
Never write "{speaker}" as a self-reference replacement.

(b) If the antecedent is another person or a thing named in the \
SURROUNDING TURNS, REPLACE the pronoun with the name or noun phrase \
the surrounding turns established. The substituted phrase must \
appear WORD-FOR-WORD in the SURROUNDING TURNS block above. Copy the \
exact phrase from the block.

(c) If you cannot find the antecedent's name or noun phrase word-for-\
word in the surrounding turns, KEEP the pronoun unchanged.

DO NOT TOUCH:
- "you" / "your" / "yours" / "yourself" -- already handled in a prior \
pass.
- "I" / "my" / "me" / "mine" / "myself" referring to {speaker}.
- "we" / "us" / "our" / "ours" / "ourselves" -- joint references.
- "it" -- handled in a later pass.
- Demonstratives ("this" / "that" / etc.) -- handled in a later pass.
- Vocatives like ", {addressee}," that may already be in the message.
- Every other word: keep unchanged.

HARD CONSTRAINTS:
- Never invent a name or noun phrase not present in the SURROUNDING \
TURNS block. If neighbors describe a group without naming \
individuals (e.g. "my furry friends"), use that exact phrase; do not \
invent specific names.
- Never write "{speaker}: " as a prefix anywhere -- the display adds \
it automatically.

OUTPUT: write ONLY the rewritten MESSAGE on a single line. If the \
MESSAGE is empty filler, output an empty line.

SURROUNDING TURNS (context only):
{neighbors_block}MESSAGE FROM {speaker} TO {addressee}:
{passage}"""


PROMPT_PASS_3_DEMONSTRATIVES = """\
SPEAKER: {speaker}
ADDRESSEE: {addressee}

You will rewrite the MESSAGE below. Your ONLY job is to handle \
demonstratives: "this" / "that" / "these" / "those" / "here" / \
"there" and the bare pronoun "it". Every other token in the MESSAGE \
must appear in your output unchanged.

For each demonstrative or bare "it", decide its referent by checking \
the SURROUNDING TURNS:

(a) If the SURROUNDING TURNS establish that the referent BELONGS TO \
{speaker} (the surrounding turns describe {speaker} as the maker / \
owner / builder / restorer / planter / creator of the referent), \
REPLACE with "my [noun]" where [noun] is copied from the surrounding \
turns word-for-word.

(b) Otherwise, if the referent's noun phrase appears WORD-FOR-WORD \
in the SURROUNDING TURNS block, REPLACE the demonstrative (or bare \
"it") with that noun phrase.

(c) If you cannot find the referent's noun phrase word-for-word in \
the surrounding turns, KEEP the demonstrative unchanged.

DO NOT TOUCH:
- "I" / "my" / "me" / "mine" / "myself" referring to {speaker}.
- "we" / "us" / "our" / "ours" / "ourselves".
- "you" / "your" / "yours" / "yourself" -- handled in a prior pass.
- Third-person pronouns -- handled in a prior pass.
- "it" when it is part of an idiom ("isn't it", "it's nice to", \
"how it goes") -- those bare "it"s have no concrete referent.
- TEMPORAL references like "today", "tomorrow", "yesterday", "next \
week", "this morning", "this weekend" -- KEEP unchanged.
- Vocatives already inserted in the message (", {addressee},").
- Every other word: keep unchanged.

HARD CONSTRAINTS:
- Never invent a referent noun phrase not present in the SURROUNDING \
TURNS block.
- Never substitute when the referent is unclear; KEEP unchanged \
instead.
- Never write "{speaker}: " as a prefix -- the display adds it \
automatically.

OUTPUT: write ONLY the rewritten MESSAGE on a single line. If the \
MESSAGE is empty filler, output an empty line.

SURROUNDING TURNS (context only):
{neighbors_block}MESSAGE FROM {speaker} TO {addressee}:
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


class DeicticResolvedMultipassSegmenter(Segmenter):
    def __init__(
        self,
        *,
        language_model: LanguageModel,
        chunk_size: int = 1500,
        max_attempts: int = 3,
    ) -> None:
        self._lm = language_model
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            keep_separator="end",
        )

    async def _call(
        self,
        template: str,
        chunk: str,
        speaker: str,
        addressee: str,
        neighbors_block: str,
    ) -> str:
        prompt = template.format(
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

    async def _resolve(
        self,
        chunk: str,
        speaker: str,
        addressee: str,
        neighbors_block: str,
    ) -> str:
        # Pass 1 -- "you" handling.
        draft1 = await self._call(
            PROMPT_PASS_1_YOU,
            chunk,
            speaker,
            addressee,
            neighbors_block,
        )
        if not draft1:
            return ""
        # Pass 2 -- 3p pronoun handling.
        draft2 = await self._call(
            PROMPT_PASS_2_THIRDPERSON,
            draft1,
            speaker,
            addressee,
            neighbors_block,
        )
        if not draft2:
            return draft1
        # Pass 3 -- demonstratives.
        draft3 = await self._call(
            PROMPT_PASS_3_DEMONSTRATIVES,
            draft2,
            speaker,
            addressee,
            neighbors_block,
        )
        return draft3 or draft2

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
