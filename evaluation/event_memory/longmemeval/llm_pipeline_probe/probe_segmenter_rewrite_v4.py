"""LLM rewriting segmenter v4 — preserve stated reasons verbatim.

Investigating whether v2's 88.2% floor on LoCoMo conv 0 l11 can be
pushed toward 90%. The remaining 18 failures split into:
  - ~5-6 cross-message inference (Sweden only in necklace context;
    Becoming Nicole / Matt Patterson exist in DB but rank below top-11)
  - ~5-6 retrieval semantic-mismatch (right segment exists but query
    phrasing doesn't match)
  - ~5-6 detail/reason DROPPED during rewrite

The third bucket is the only one addressable at the segmenter level
without breaking the single-event constraint. Failure pattern:
  - Gold: "she wanted to catch the eye and make people smile"
    v2 output: "she used colors for self-expression"
  - Gold: "strength and motivation"
    v2 output: "love"

The LLM smooths specific stated reasons into a more abstract version
of the same idea. v4 adds a rule that explicitly forbids this:
when the speaker states a REASON, MOTIVATION, or PURPOSE for an
action, that exact phrasing (the answer to "why?" or "what for?") must
appear in the statement verbatim.

This is principle-level, not example-driven. v3 added similar
detail to rule 4 and regressed -3pp because it expanded the
existing rule and starved time-resolution attention. v4 adds the
reason rule as a NEW separate rule (rule 4a) so it has its own
attention slot, and keeps every other rule textually identical to v2.
"""

from __future__ import annotations

import asyncio
import os
from typing import override
from uuid import uuid4

import openai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_REWRITE_V4 = """\
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

4. PRESERVE STATED REASONS. When the speaker explicitly states a \
reason, motivation, purpose, or "why" for an action, decision, choice, \
preference, or feeling, that stated reason must appear in the \
statement using the speaker's own concrete phrasing. Do not substitute \
a more abstract or generalized version of the same idea. If the \
speaker says they did X "to catch the eye and make people smile", the \
statement keeps "to catch the eye and make people smile" verbatim, \
not "for self-expression". If the speaker says their family gives them \
"strength and motivation", the statement keeps "strength and \
motivation", not "love". The answer to a "why did X?" question is the \
exact phrasing the speaker used, not a paraphrase to a related theme.

5. PRESERVE MEANING. Mirror the exact direction, polarity, and \
emotional content of the original. "Used to enjoy X" means no longer \
enjoys X. "Didn't get to bed until 2 AM" means LATE BEDTIME, not late \
wakeup. "Can't stop X-ing" means doing X frequently. Emotional states \
("scared but reassured", "happy and thankful"), motivations \
("inspired by her own journey"), and subjective descriptions \
("therapeutic", "nerve-wracking") are part of the meaning and stay.

6. TIME RESOLUTION. Convert every relative time reference into an \
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

7. NO REDUNDANT DATE PREFIX. Do not prepend "On {date}, ..." or \
"{speaker} on {date}: ..." to statements. The observation date \
{date} is metadata used for resolving relative time; it does not \
need to appear in every statement. A statement should contain a date \
only when that date is part of the content (a resolved relative \
reference, an explicit date from the message, or the only \
identifying time for the event).

8. ONE TOPIC PER STATEMENT. If the message covers multiple unrelated \
topics, produce one statement per topic. If a single topic carries \
several details (e.g. "I got promoted at Shopify last week after two \
years"), capture them together in one rich statement.

9. NO FABRICATION. Every detail in a statement must trace back to \
words in the message. Do not infer attributes (gender, age, \
ethnicity, role) from names or context. Do not import information \
from outside the message.

10. NO META-COMMENTARY. Extract the CONTENT of what was said, not \
the fact that it was said. "{speaker} asked about X" is wrong unless \
the question itself carries an incidental fact. If the message \
contains incidental personal facts inside a question or request, \
extract those facts.

11. DROP PURE FILLER. Greetings ("Hi"), sign-offs ("Bye"), and bare \
acknowledgments ("Sounds good", "Thanks", "Got it") that carry no \
specifics produce no statement. A message whose every part is \
interchangeable filler -- no name, date, number, decision, plan, \
preference, opinion, event, or attached media description -- emits \
an empty list.

12. NO DUPLICATES. Do not emit two statements that carry the same \
information in different words. Each fact appears exactly once.

Output: a JSON object {{ "memories": [...] }} where memories is a \
list of strings. Empty list if the message has no extractable \
specifics.

MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    memories: list[str]


class RewriteSegmenter(Segmenter):
    """v4 — adds rule 4 (PRESERVE STATED REASONS) as its own slot."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V4,
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

    async def _rewrite_chunk(self, chunk: str, speaker: str, date: str) -> list[str]:
        prompt = self._prompt_template.format(speaker=speaker, date=date, passage=chunk)
        response = await self._language_model.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return [m.strip() for m in response.memories if m and m.strip()]

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
                        if not chunk.strip():
                            continue
                        memories = await self._rewrite_chunk(chunk, speaker, date_str)
                        for memory in memories:
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=memory),
                                    context=NullContext(),
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments


# Validation harness — focus on the specific failure cases.
SAMPLES: list[tuple[str, str, str]] = [
    (
        "Melanie",
        "2023-10-13",
        "I picked these colors and patterns because I wanted to catch "
        "the eye and make people smile when they see the bowl. The "
        "colors are bright and the patterns are intricate so it feels "
        "joyful.",
    ),
    (
        "Caroline",
        "2023-08-17",
        "My family really gives me strength and motivation to keep going. "
        "Just knowing they're behind me makes everything easier.",
    ),
    (
        "Caroline",
        "2023-07-12",
        "I want to pursue counseling because my own journey and the "
        "support I got was pivotal. Seeing how counseling helps changed "
        "everything for me.",
    ),
    # Sanity check that other rules still hold
    (
        "User",
        "2023-05-08",
        "Hey! I'm Marcus. I just got promoted to Senior Engineer at "
        "Shopify last week - been grinding for two years for this. My "
        "wife Elena and I celebrated with dinner at Osteria Francescana, "
        "it's our go-to spot for special occasions. We're also expecting "
        "our first baby in March!",
    ),
    (
        "Caroline",
        "2023-05-08",
        "Hi Mel!",
    ),
]


MODEL_CONFIGS: list[tuple[str, str, str]] = [
    ("gpt-5.4-nano @ low", "gpt-5.4-nano", "low"),
]


async def _validate() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for label, model, effort in MODEL_CONFIGS:
        print(f"\n{'=' * 60}\nMODEL: {label}\n{'=' * 60}")
        lm = OpenAIResponsesLanguageModel(
            OpenAIResponsesLanguageModelParams(
                client=client,
                model=model,
                reasoning_effort=effort,
            )
        )
        seg = RewriteSegmenter(language_model=lm)
        for speaker, date, text in SAMPLES:
            event = Event(
                uuid=uuid4(),
                timestamp=__import__("datetime").datetime.fromisoformat(
                    f"{date}T12:00:00+00:00"
                ),
                context=ProducerContext(producer=speaker),
                blocks=[TextBlock(text=text)],
            )
            try:
                out = await seg.segment(event)
            except Exception as exc:
                print(f"\n  [{speaker}] {text[:60]}... -> ERROR {exc}")
                continue
            print(f"\n  [{speaker}] {text[:70]}{'...' if len(text) > 70 else ''}")
            for s in out:
                print(f"    -> {s.block.text}")
            if not out:
                print("    -> (no memories)")


if __name__ == "__main__":
    asyncio.run(_validate())
