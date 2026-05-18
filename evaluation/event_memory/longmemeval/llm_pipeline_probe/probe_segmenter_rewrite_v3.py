"""LLM rewriting segmenter v3 — preserve distinctive subjective phrasing verbatim.

v2 result on LoCoMo conv 0 (152 q, mem0-bench, gpt-5.4-nano @ low):
  - l8 (339 tok):   82.9% | mh 22/32 | od 9/13 | sh 59/70 | tmp 36/37
  - l11 (461 tok):  88.2% | mh 25/32 | od 11/13 | sh 61/70 | tmp 37/37
  - l40 (1560 tok): 93.4% | mh 27/32 | od 12/13 | sh 67/70 | tmp 36/37

vs text-seg l8 (474 tok, 84.2%): v2 l11 wins +4.0pp at matched tokens.

Two residual regressions vs text-seg:
  - Open-domain at l8 (-3): rewrite paraphrases emotional/causal phrases
    that ground subjective inferences. Example: gold "Would Caroline
    pursue counseling without support?" → answer hinges on Caroline's
    own quote "MY journey and the support I got was PIVOTAL." Rewrite
    produced "Caroline wanted counseling" — lost the causal "because of
    support." text-seg kept the exact quote.
  - Multi-hop at l40 (-4 vs text-seg l40): verbatim wins back recall at
    higher budget when token cost stops being the binding constraint.

v3 changes:
  - Rule 4 (PRESERVE MEANING) gets a new clause: VERBATIM SUBJECTIVE
    PHRASES. Distinctive descriptive phrases the speaker uses to
    describe a feeling, evaluation, motivation, or causal connection
    must appear in the statement verbatim, inside quotes. The model
    paraphrases generic words ("happy" → "happy"); it does not
    paraphrase distinctive ones ("pivotal", "insane", "scary",
    "rewarding", "soul-crushing", "transformative").
  - Rule 4 also adds CAUSAL/MOTIVATIONAL CONNECTIONS. When the
    speaker says X happened BECAUSE of Y, or X is motivated by Y,
    that "because" link must survive in the statement.
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


PROMPT_REWRITE_V3 = """\
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

3. PRESERVE EVERY SPECIFIC. Names, dates, numbers, quantities, \
titles, brands, places, named objects, named activities, quoted \
phrases, and proper nouns must appear in the statement exactly as \
written in the message. Replacing a specific with a vague category \
is a failure: "Ferrari 488 GTB" stays "Ferrari 488 GTB" (not "a \
car"); "promoted to assistant manager" stays "assistant manager" \
(not "manager"); "416 pages" stays "416 pages" (not "about 400 \
pages"); "watched 'Eternal Sunshine of the Spotless Mind'" keeps \
the full title.

4. PRESERVE MEANING. Mirror the exact direction, polarity, and \
emotional content of the original.
   - DIRECTION AND POLARITY. "Used to enjoy X" means no longer \
enjoys X. "Didn't get to bed until 2 AM" means LATE BEDTIME, not \
late wakeup. "Can't stop X-ing" means doing X frequently.
   - VERBATIM SUBJECTIVE PHRASES. When the speaker uses a \
distinctive word or phrase to describe a feeling, evaluation, \
motivation, atmosphere, or experience -- words like "pivotal", \
"insane", "scary", "rewarding", "soul-crushing", "transformative", \
"liberating", "nerve-wracking", "magical", "devastating", "powerful" \
-- include that exact word in the statement, inside quotes. Do not \
paraphrase to a generic synonym ("scary" never becomes "frightening" \
or "negative"; "pivotal" never becomes "important"; "insane" never \
becomes "wild").
   - CAUSAL AND MOTIVATIONAL LINKS. When the speaker says A because \
of B, or A is motivated by B, or A inspired B, the link must survive \
in the statement. The statement must show both the effect and the \
cause/motivation. Example: "Caroline wants to pursue counseling \
because her own journey and the support she received were 'pivotal'" \
is correct; "Caroline wants to pursue counseling" alone is wrong.
   - EMOTIONAL STATES STAY. "scared but reassured", "happy and \
thankful", "liberated and empowered" -- keep them in the statement.

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

6. NO REDUNDANT DATE PREFIX. Do not prepend "On {date}, ..." or \
"{speaker} on {date}: ..." to statements. The observation date \
{date} is metadata used for resolving relative time; it does not \
need to appear in every statement. A statement should contain a date \
only when that date is part of the content (a resolved relative \
reference, an explicit date from the message, or the only \
identifying time for the event).

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
    """Structured response from the rewriting language model."""

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


class RewriteSegmenter(Segmenter):
    """v3 — strengthened rule 4 for verbatim subjective/causal preservation."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V3,
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


# Validation harness — focus on cases where v2 lost on open-domain.
SAMPLES: list[tuple[str, str, str]] = [
    (
        "Caroline",
        "2023-07-13",
        "I really want to do counseling because my own journey and the "
        "support I got was pivotal. Seeing how counseling helps changed "
        "everything for me.",
    ),
    (
        "Melanie",
        "2023-10-20",
        "The roadtrip was insane! My son was in an accident -- airbags "
        "deployed, dashboard got dented. It was a real scary experience. "
        "He's okay but I'm shaken.",
    ),
    (
        "Caroline",
        "2023-05-08",
        "I've been here in this country for 4 years -- since I moved "
        "from my home country. The transition was tough.",
    ),
    # Multi-topic sanity check
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
