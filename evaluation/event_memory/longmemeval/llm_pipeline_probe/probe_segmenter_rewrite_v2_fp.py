"""LLM rewriting segmenter v2-FP — first-person variant of v2.

The user explicitly offered two architectural choices:
  - third person (NullContext) — tested as v2, winner at 88.2% l11
  - first person (keep ProducerContext) — UNTESTED, this file

Hypothesis: first-person retains the speaker's natural language hooks
("I moved from my home country" stays "I moved from my home country"
verbatim) while still compressing filler. Semantic search may match
queries phrased in first-person better, and downstream cross-segment
inference (Caroline-as-subject queries) becomes simpler because the
speaker name appears in the segment header instead of repeated in
every statement.

Trade-off: speaker header `[date] Caroline:` adds ~10 tok/seg vs
NullContext header `[date]`. At l11 that's +110 tok, so token-matched
comparison gets fewer segments for first-person.

v2-FP changes from v2 third-person:
  - Rule 1: KEEP first-person ("I", "my", "me"). Resolve only
    second-person ("you" -> addressee name or "the other party").
  - Output context: ProducerContext (segment header will carry
    speaker label).
  - All other rules identical to v2.
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
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_REWRITE_V2_FP = """\
Rewrite a single conversational message into a JSON list of \
self-contained first-person memory statements. The memory system \
stores these statements with a speaker tag and timestamp; each \
statement must be findable and meaningful on its own.

CONTEXT:
- The message was spoken by {speaker} on {date}.
- You only see this one message. Do not reference what came before \
or after.
- The stored memory will be prefixed at display time with the \
speaker's name. Your output should remain in first person.

RULES:
1. FIRST PERSON KEPT. Keep first-person references ("I", "my", "me") \
verbatim -- the speaker tag in the stored record carries the identity. \
Resolve only second-person references ("you", "your") to the addressee's \
name if known, otherwise to "the other party". Replace ambiguous \
third-person pronouns with their concrete referents.

2. SELF-CONTAINED. A statement must be understandable in isolation, \
without the surrounding conversation. State what happened or what is \
being claimed, plus any time, location, or quantity that grounds the \
statement.

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
("inspired by my own journey"), and subjective descriptions \
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
tense — backward for past tense, forward for future tense.
   - Absolute dates and explicit durations stay verbatim — "18 days" \
stays "18 days".
   - After resolution, the original relative phrase MUST NOT appear \
in the statement. Replace "yesterday" with the resolved date; do not \
keep "yesterday" alongside the date.

6. NO REDUNDANT DATE PREFIX. Do not prepend "On {date}, ..." to \
statements. The observation date {date} is already in the segment \
header; it does not need to appear in every statement. A statement \
should contain a date only when that date is part of the content (a \
resolved relative reference, an explicit date from the message, or \
the only identifying time for the event).

7. ONE TOPIC PER STATEMENT. If the message covers multiple unrelated \
topics, produce one statement per topic. If a single topic carries \
several details (e.g. "I got promoted at Shopify last week after two \
years"), capture them together in one rich statement.

8. NO FABRICATION. Every detail in a statement must trace back to \
words in the message. Do not infer attributes (gender, age, \
ethnicity, role) from names or context. Do not import information \
from outside the message.

9. NO META-COMMENTARY. Extract the CONTENT of what was said, not the \
fact that it was said. "I asked about X" is wrong unless the question \
itself carries an incidental fact. If the message contains incidental \
personal facts inside a question or request, extract those facts.

10. DROP PURE FILLER. Greetings ("Hi"), sign-offs ("Bye"), and bare \
acknowledgments ("Sounds good", "Thanks", "Got it") that carry no \
specifics produce no statement. A message whose every part is \
interchangeable filler -- no name, date, number, decision, plan, \
preference, opinion, event, or attached media description -- emits \
an empty list.

11. NO DUPLICATES. Do not emit two statements that carry the same \
information in different words. Each fact appears exactly once.

Output: a JSON object {{ "memories": [...] }} where memories is a \
list of strings. Empty list if the message has no extractable \
specifics.

MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    memories: list[str]


def _splitter_for_chunks(chunk_size: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
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


def _speaker_label(context: object) -> str:
    if isinstance(context, ProducerContext):
        return context.producer
    return "the speaker"


class RewriteSegmenterFP(Segmenter):
    """First-person rewriting segmenter — keeps ProducerContext."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V2_FP,
        chunk_size: int = 1500,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
        self._splitter = _splitter_for_chunks(chunk_size)

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
        speaker = _speaker_label(event.context)
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
                                    # KEEP ProducerContext so speaker
                                    # appears in segment header.
                                    context=event.context,
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments


SAMPLES: list[tuple[str, str, str]] = [
    (
        "Caroline",
        "2023-05-08",
        "I went to a LGBTQ support group yesterday and it was so "
        "powerful. The transgender stories were so inspiring! I was so "
        "happy and thankful for all the support.",
    ),
    (
        "Caroline",
        "2023-05-08",
        "I moved here 4 years ago from my home country.",
    ),
    (
        "Caroline",
        "2023-09-12",
        "Thanks, Melanie! This necklace is super special to me - a gift "
        "from my grandma in my home country, Sweden. She gave it to me "
        "when I was young, and it stands for love, faith and strength.",
    ),
    (
        "Melanie",
        "2023-08-17",
        "By the way, my son was in a car accident on the road trip last "
        "weekend -- airbags deployed, dashboard got dented. He was okay, "
        "but the other kids were scared so I had to reassure them. Family "
        "is super important; they mean the world to me.",
    ),
    (
        "Caroline",
        "2023-05-08",
        "Hi Mel!",
    ),
    (
        "User",
        "2024-02-03",
        "Sounds good!",
    ),
]


MODEL_CONFIGS: list[tuple[str, str, str]] = [
    ("gpt-5.4-nano @ low", "gpt-5.4-nano", "low"),
    ("gpt-5-mini @ low", "gpt-5-mini", "low"),
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
        seg = RewriteSegmenterFP(language_model=lm)
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
