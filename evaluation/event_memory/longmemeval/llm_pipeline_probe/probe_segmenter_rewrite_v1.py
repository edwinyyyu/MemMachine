"""LLM rewriting segmenter v1 — deterministic split + LLM rewrite.

Architecture rationale (per /loop directive):
  - Prior LLM-segmenter line (v33 / v46) lets the LLM both split AND
    select. The model is bad at splitting -- it overfragments
    conversational turns at sentence boundaries.
  - This segmenter instead:
      1. Deterministically splits the event text via
         RecursiveCharacterTextSplitter into bounded chunks.
      2. Calls the LLM ONCE per chunk to REWRITE the chunk into a list
         of self-contained third-person memory statements.
  - Output is in third-person prose (like mem0's V3 ADDITIVE_EXTRACTION),
    so segments get context=NullContext (the speaker name is embedded
    in the prose itself).
  - The segmenter sees ONE event at a time; no cross-event context.

Goal: improve scores at lower token budgets by trading verbatim
fidelity for write-time synthesis density. Mem0's ~38 tok/memory at
k=10 beats our ~57 tok/seg at l7 by 4pp on conv0.

Prompt design constraints (per /loop directive):
  - Cross-model: gpt-5-nano, gpt-5-mini, gpt-5.4-nano at low or medium.
  - Principles > examples > biased language.
  - Each rule is a procedural test, not a soft suggestion.
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


# v1 prompt -- principles-only, no examples, no biased language.
# The speaker's name and the observation date are injected so the LLM
# can attribute correctly and resolve relative time references.
PROMPT_REWRITE_V1 = """\
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

5. RESOLVE RELATIVE TIME. When the message says "yesterday", "last \
week", "in March", "next month", or similar, convert to an absolute \
date using {date} as the anchor. "yesterday" -> the day before \
{date}. "last week" -> the week before {date}. "in March" -> the \
March nearest before {date} unless context makes a later March \
unambiguous. Absolute dates and durations stay absolute -- "18 days" \
stays "18 days".

6. ONE TOPIC PER STATEMENT. If the message covers multiple unrelated \
topics, produce one statement per topic. If a single topic carries \
several details (e.g. "I got promoted at Shopify last week after two \
years"), capture them together in one rich statement.

7. NO FABRICATION. Every detail in a statement must trace back to \
words in the message. Do not infer attributes (gender, age, \
ethnicity, role) from names or context. Do not import information \
from outside the message.

8. NO META-COMMENTARY. Extract the CONTENT of what was said, not the \
fact that it was said. "{speaker} asked about X" is wrong unless the \
question itself carries an incidental fact. If the message contains \
incidental personal facts inside a question or request, extract \
those facts.

9. DROP PURE FILLER. Greetings ("Hi"), sign-offs ("Bye"), and bare \
acknowledgments ("Sounds good", "Thanks", "Got it") that carry no \
specifics produce no statement. A message whose every part is \
interchangeable filler -- no name, date, number, decision, plan, \
preference, opinion, event, or attached media description -- emits an \
empty list.

10. NO DUPLICATES. Do not emit two statements that carry the same \
information in different words. Each fact appears exactly once.

Output: a JSON object {{ "memories": [...] }} where memories is a \
list of strings. Empty list if the message has no extractable \
specifics.

MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    """Structured response from the rewriting language model."""

    memories: list[str]


def _splitter_for_chunks(chunk_size: int) -> RecursiveCharacterTextSplitter:
    """Recursive splitter sized to keep most LME turns as a single chunk.

    For LME turns averaging ~1-3 paragraphs, chunk_size=1500 keeps the
    whole turn together. The splitter only fires on outlier long
    messages.
    """
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
    """Extract a human-readable speaker label from event context."""
    if isinstance(context, ProducerContext):
        return context.producer
    return "the speaker"


class RewriteSegmenter(Segmenter):
    """Segmenter that deterministically chunks then asks the LLM to rewrite.

    Output segments are third-person prose with NullContext. The
    speaker name is embedded in the rewritten prose itself.

    Args:
        language_model: LanguageModel used to rewrite each chunk.
            Configure the model and reasoning effort at construction
            of the LanguageModel itself.
        prompt_template: A `.format(speaker=..., date=..., passage=...)`
            template producing the full prompt.
        chunk_size: Maximum characters per deterministic chunk before
            calling the LLM.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V1,
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
        # Filter out empty / whitespace-only strings, but preserve order.
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


# ---------------------------------------------------------------------------
# Quick validation harness -- run a few sample turns through the segmenter
# under three model configs and print the output for human inspection.
# Usage:
#     uv run --project /Users/eyu/edwinyyyu/mmcc/segment_store python \
#         -m llm_pipeline_probe.probe_segmenter_rewrite_v1
# ---------------------------------------------------------------------------


SAMPLES: list[tuple[str, str, str]] = [
    # (speaker, date, text)
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
        "I went to a LGBTQ support group yesterday and it was so "
        "powerful. The transgender stories were so inspiring! I was so "
        "happy and thankful for all the support.",
    ),
    (
        "Caroline",
        "2023-05-08",
        "Hi Mel!",
    ),
    (
        "Melanie",
        "2023-08-17",
        "By the way, my son was in a car accident on the road trip "
        "last weekend -- airbags deployed, dashboard got dented. He was "
        "okay, but the other kids were scared so I had to reassure them. "
        "Family is super important; they mean the world to me.",
    ),
    (
        "Assistant",
        "2024-01-15",
        "Here are some Netflix documentaries known for storytelling: "
        '1) "Formula 1: Drive to Survive" (behind the scenes of '
        'Formula 1 racing) 2) "Athlete A" (investigative look at USA '
        'Gymnastics) 3) "The Battered Bastards of Baseball" '
        "(independent baseball story).",
    ),
    (
        "User",
        "2024-02-03",
        "Sounds good!",
    ),
]


MODEL_CONFIGS: list[tuple[str, str, str]] = [
    ("gpt-5-nano", "gpt-5-nano", "low"),
    ("gpt-5-mini", "gpt-5-mini", "low"),
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
