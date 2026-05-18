"""LLM rewriting segmenter v11 — v7 prompt + quality-check / fix pass.

User-explicit last-resort experiment: after the v7 rewrite, run a
SECOND LLM call that compares the rewrites against the source chunk
and lists any specific named entities, numbers, dates, quoted
phrases, or proper nouns that appear in the source but are missing
from every rewrite. If any are missing, run a FIX pass that re-emits
the rewrites with explicit instructions to add the missing items.

This doubles ingest cost per chunk in the worst case. The goal is to
test whether the v7 rewrite occasionally drops specifics that the
dual-surface embed *retrieves* (via verbatim chunk) but the answerer
LLM *can't use* (because the segment text is just the rewrite).

Architecture identical to v7 in every other respect.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    RewriteContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel

PROMPT_REWRITE_V11 = """\
Rewrite a single conversational message into a JSON list of \
self-contained third-person memory statements. The memory system \
stores these statements and later retrieves them by semantic search; \
each statement must be findable and meaningful on its own.

CONTEXT:
- The message was spoken by {speaker} on {date}.
- You only see this one message. Do not reference what came before \
or after.

RULES:
1. THIRD PERSON. Use the speaker's name in place of first-person \
references ("I" -> "{speaker}"). Use a descriptive name in place of \
second-person references when the addressee is not known to you \
("you" -> "the other party"). Replace ambiguous pronouns with their \
concrete referents.

2. SELF-CONTAINED. A statement must be understandable in isolation, \
without the surrounding conversation. State who is involved, what \
happened or what is being claimed, and any time, location, or \
quantity that grounds the statement.

3. PRESERVE EVERY SPECIFIC. Names, dates, numbers, quantities, titles, \
brands, places, named objects, named activities, quoted phrases, and \
proper nouns must appear in the statement exactly as written in the \
message. Replacing a specific with a vague category is a failure: \
"Ferrari 488 GTB" stays "Ferrari 488 GTB" (not "a car"); \
"promoted to assistant manager" stays "assistant manager" (not \
"manager"); "416 pages" stays "416 pages" (not "about 400 pages"); \
"watched 'Eternal Sunshine of the Spotless Mind'" keeps the full \
title.

4. PRESERVE MEANING. Mirror the exact direction, polarity, and \
emotional content of the original. "Used to enjoy X" means no longer \
enjoys X. "Didn't get to bed until 2 AM" means LATE BEDTIME, not late \
wakeup. "Can't stop X-ing" means doing X frequently. Emotional states \
("scared but reassured", "happy and thankful"), motivations \
("inspired by her own journey"), and subjective descriptions \
("therapeutic", "nerve-wracking") are part of the meaning and stay.

5. TIME RESOLUTION. Convert every relative time reference into an \
absolute date or interval, using {date} as the anchor.
   - Past-tense markers ("yesterday", "last week", "ago", verbs in \
past tense, "used to") resolve BACKWARD from {date}.
   - Future-tense markers ("tomorrow", "next week", "upcoming", \
"will", "expecting", "this coming", verbs in future tense) resolve \
FORWARD from {date}.
   - Ambiguous bare month/day references ("in March", "on the 5th") \
resolve to the nearest occurrence consistent with the surrounding \
tense -- backward for past tense, forward for future tense.
   - Absolute dates and explicit durations stay verbatim -- "18 days" \
stays "18 days".
   - After resolution, the original relative phrase MUST NOT appear \
in the statement. Replace "yesterday" with the resolved date; do not \
keep "yesterday" alongside the date.

6. DATE-ANCHOR EACH STATEMENT. Every statement should be anchored to \
a specific date drawn from the message context. When the statement \
describes an event, action, observation, or claim made on the \
observation date {date}, include "{date}" in the statement (e.g., \
"On {date}, {speaker} said ..." or "{speaker} attended X on \
{date}"). When the statement is about an event that happened on a \
different date (already in the message or resolved per rule 5), use \
that date instead. The date may appear once per statement and should \
sit naturally inside the prose, not as a redundant prefix.

7. ONE TOPIC PER STATEMENT. If the message covers multiple unrelated \
topics, produce one statement per topic. If a single topic carries \
several details (e.g. "I got promoted at Shopify last week after two \
years"), capture them together in one rich statement.

8. NO FABRICATION. Every detail in a statement must trace back to \
words in the message. Do not infer attributes (gender, age, \
ethnicity, role) from names or context. Do not import information \
from outside the message.

9. NO META-COMMENTARY. Extract the CONTENT of what was said, not the \
fact that it was said. "{speaker} asked about X" is wrong unless the \
question itself carries an incidental fact. If the message contains \
incidental personal facts inside a question or request, extract \
those facts.

10. DROP PURE FILLER. Greetings ("Hi"), sign-offs ("Bye"), and bare \
acknowledgments ("Sounds good", "Thanks", "Got it") that carry no \
specifics produce no statement. A message whose every part is \
interchangeable filler -- no name, date, number, decision, plan, \
preference, opinion, event, hobby, relationship detail, or attached \
media description -- emits an empty list.

11. NO DUPLICATES. Do not emit two statements that carry the same \
information in different words. Each fact appears exactly once.

Output: a JSON object {{ "memories": [...] }} where memories is a \
list of strings. Empty list if the message has no extractable \
specifics.

MESSAGE FROM {speaker} on {date}:
{passage}"""


PROMPT_QUALITY_CHECK = """\
A SOURCE message and a list of REWRITES extracted from it are below.

Identify every CONCRETE SPECIFIC from the source that does NOT \
appear (verbatim) in any rewrite. Concrete specifics are: proper \
names (people, places, organizations, brands, products), titles \
(book/movie/song titles in quotes), exact numbers and quantities \
("416 pages", "two years", "$50"), exact dates and durations, \
distinctive quoted phrases (e.g. an emotional reaction in quotes), \
and named activities or named objects (e.g. "Ferrari 488 GTB", \
"Eternal Sunshine of the Spotless Mind").

Ignore: generic words ("happy", "the meeting"), pronouns, common \
adjectives, function words, filler tokens like "Hi"/"Bye"/"yes", and \
specifics already covered (verbatim or by clear synonym) in some \
rewrite.

If every concrete specific from the source IS covered by at least \
one rewrite, return an empty list.

SOURCE MESSAGE FROM {speaker} on {date}:
{passage}

REWRITES:
{rewrites}

Output: a JSON object {{ "missing": [...] }} where missing is a list \
of the concrete specifics that are absent from every rewrite. Each \
list item is a single specific exactly as it appears in the source."""


PROMPT_REWRITE_FIX = """\
The earlier rewrites of the SOURCE message below dropped some \
concrete specifics that must be searchable. Update the rewrites so \
that every item in MUST_INCLUDE appears verbatim in at least one \
rewrite. You may add new statements OR weave the missing items into \
existing statements, whichever keeps each statement self-contained \
and date-anchored. Follow these rules:

- Every original rule still applies: third-person, self-contained, \
preserve specifics verbatim, preserve meaning, time-resolved, \
date-anchored, no fabrication, no meta-commentary, drop pure filler, \
no duplicates.
- Every MUST_INCLUDE item must appear verbatim somewhere.
- Statements already in the original rewrites can be kept, edited, \
or dropped — produce the FINAL list, not a diff.

SOURCE MESSAGE FROM {speaker} on {date}:
{passage}

ORIGINAL_REWRITES:
{rewrites}

MUST_INCLUDE:
{missing}

Output: a JSON object {{ "memories": [...] }} where memories is the \
revised list of rewrite statements."""


class _RewriteResponse(BaseModel):
    memories: list[str]


class _MissingResponse(BaseModel):
    missing: list[str]


class RewriteSegmenter(Segmenter):
    """v11 — v7 rewrite + LLM quality-check/fix pass per chunk."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V11,
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

    async def _initial_rewrite(self, chunk: str, speaker: str, date: str) -> list[str]:
        prompt = self._prompt_template.format(speaker=speaker, date=date, passage=chunk)
        response = await self._language_model.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return [m.strip() for m in response.memories if m and m.strip()]

    async def _find_missing(
        self, chunk: str, speaker: str, date: str, rewrites: list[str]
    ) -> list[str]:
        rewrites_block = "\n".join(f"- {m}" for m in rewrites) if rewrites else "(none)"
        prompt = PROMPT_QUALITY_CHECK.format(
            speaker=speaker, date=date, passage=chunk, rewrites=rewrites_block
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_MissingResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return [m.strip() for m in response.missing if m and m.strip()]

    async def _fix_rewrites(
        self,
        chunk: str,
        speaker: str,
        date: str,
        rewrites: list[str],
        missing: list[str],
    ) -> list[str]:
        rewrites_block = "\n".join(f"- {m}" for m in rewrites) if rewrites else "(none)"
        missing_block = "\n".join(f"- {m}" for m in missing)
        prompt = PROMPT_REWRITE_FIX.format(
            speaker=speaker,
            date=date,
            passage=chunk,
            rewrites=rewrites_block,
            missing=missing_block,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return rewrites
        revised = [m.strip() for m in response.memories if m and m.strip()]
        return revised or rewrites

    async def _rewrite_chunk(self, chunk: str, speaker: str, date: str) -> list[str]:
        rewrites = await self._initial_rewrite(chunk, speaker, date)
        if not rewrites:
            return rewrites
        missing = await self._find_missing(chunk, speaker, date, rewrites)
        if not missing:
            return rewrites
        return await self._fix_rewrites(chunk, speaker, date, rewrites, missing)

    @staticmethod
    def _build_embed_text(rewrite: str, original_chunk: str, speaker: str) -> str:
        return f"{rewrite}\n{speaker}: {original_chunk}"

    @override
    async def segment(self, event: Event) -> list[Segment]:
        speaker = (
            event.context.producer
            if isinstance(event.context, ProducerContext)
            else "the speaker"
        )
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
                        memories = await self._rewrite_chunk(
                            chunk_stripped, speaker, date_str
                        )
                        for memory in memories:
                            embed_text = RewriteSegmenter._build_embed_text(
                                memory, chunk_stripped, speaker
                            )
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=memory),
                                    context=RewriteContext(text_to_embed=embed_text),
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
