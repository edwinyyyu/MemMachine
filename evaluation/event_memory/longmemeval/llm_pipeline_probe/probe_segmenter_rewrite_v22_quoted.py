"""v22-quoted RewriteSegmenter -- single-channel quoted-attribution rewrite.

Hypothesis
----------

v22 baseline embeds via ``RewriteContext(text_to_embed="{3p_rewrite}\\n
{speaker}: {raw_chunk}")`` -- two text channels concatenated so queries
phrased in either person can find the segment. The v22-fp-min variant
dropped the raw-chunk channel (ProducerContext only) and lost retrieval
signal on c124 (-1.42pp combined on g3+g4).

This variant tests whether a denser SINGLE-channel encoding can
preserve both channels' value without two texts: each statement is
written as

  ``<speaker> said: "<raw-ish first-person utterance>"``

The 3p attribution lives outside the quote (good for queries that
mention the speaker by name in third-person); the verbatim-ish 1p
speech lives inside the quote (good for queries that mirror raw
conversational phrasing). One string, both surfaces, ONE embed
channel via ``NullContext``.

Design
------

  - segment.block.text format: ``Alice said: "..."``. The speaker name
    is already inside the segment text, so the segment uses
    ``NullContext`` (NOT ProducerContext, which would prepend the
    speaker AGAIN and duplicate it).
  - Inside the quote, the speaker speaks in their own ``I`` / ``my``
    / ``me`` / ``we`` -- preserves the raw 1p surface.
  - Cross-speaker pronouns inside the quote resolve to addressee NAMES
    when known (``you`` -> ``Bob`` if Bob is the addressee).
  - Dates: resolve every relative reference to an absolute date INSIDE
    the quote. Single canonical form ``on YYYY-MM-DD``. Drop the bare
    relative phrase after resolution. Omit the date entirely when it
    equals {date}.
  - One event per statement; one date per statement.
  - Neutral names (Alice/Bob/Charlie/Dana) and neutral domains
    (cockatiels/mandolin/Half Dome/Big Sur/cake) in the examples; no
    LoCoMo entities.
  - Prompt budgeted to <= ~1000 tokens (cuts the long enumerations
    that the model handles for free at gpt-5.4-nano).
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    NullContext,
    ProducerContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel


PROMPT_REWRITE_V22_QUOTED = """\
Rewrite the MESSAGE into a JSON list of standalone quoted-attribution \
memory statements. Each statement is stored verbatim and retrieved \
later by semantic search.

FORMAT. Every statement is exactly:

  {speaker} said: "<utterance>"

The quoted utterance preserves {speaker}'s own first-person voice and \
verbatim phrasing as much as possible -- {speaker} speaks in their \
own ``I`` / ``my`` / ``me`` / ``myself`` / ``we`` inside the quotes. \
Inside the quote, cross-speaker references resolve to addressee NAMES \
when known from context (e.g. ``you`` -> ``Bob`` when Bob is the \
addressee). Demonstratives (``this``, ``that``, ``it``, ``they``) \
resolve to their concrete referents inside the quote.

KEEP specific content (names, places, dates, numbers, decisions, \
plans, preferences, opinions, described events, attached media). \
DROP interchangeable content (bare greetings, sign-offs, \
acknowledgments, reactions, generic questions).

ONE statement per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of the same occurrence is ONE statement \
whose quote contains all of its particulars. Distinct events \
(different times, different occasions, different actions) each get \
their own statement. Report content, not the speech-act of conveying \
it -- ``I said that ...`` / ``I told X that ...`` wrappers are \
dropped from the quoted utterance unless the speech-act itself is \
the event (a promise, an apology, an announcement).

DATES. The MESSAGE was sent on {date}. The framework prepends the \
message timestamp automatically, so the quoted utterance MUST NOT \
contain {date} in any form. Resolve every relative time reference \
(``yesterday``, ``last week``, ``three years ago``, ``next Friday``, \
``the weekend``, ``today``, ``tonight``, ``recently``, ``now``, \
``just``) to an absolute date anchored at {date}, INSIDE the quote.
  - If the resolved date EQUALS {date}, the quote contains NO date \
and NO relative phrase.
  - If the resolved date DIFFERS from {date}, weave it into the quote \
as ``on YYYY-MM-DD`` and DELETE the original relative phrase. The \
relative phrase appearing alongside the resolved date is a FAILURE.
One event date per statement; split multi-date messages into \
multiple statements.

Forbidden date forms (each is a FAILURE): sentence-prefix \
``On YYYY-MM-DD, ...``; parenthetical ``(Date: ...)`` or \
``(Event date: ...)``; square-bracket ``[YYYY-MM-DD]``; \
``as of YYYY-MM-DD``. Only ``on YYYY-MM-DD`` woven inline is \
allowed, and only when the resolved date differs from {date}.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content as their own statements.

EXAMPLES (neutral names and domains).

Example 1 -- message-time event; no date in the quote.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- \
the tonkotsu is incredible.
->
{{ "memories": ["Alice said: \\"I am finally trying the ramen place \
on Castro Street and the tonkotsu is incredible.\\""] }}

Example 2 -- relative reference resolves to a DIFFERENT date; the \
relative phrase is removed.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["Bob said: \\"I adopted my two cockatiels on \
2023-04-10, right before I moved to Portland.\\""] }}

Example 3 -- explicit absolute date; cross-speaker ``you`` resolves \
to addressee NAME inside the quote.
MESSAGE FROM Charlie on 2026-05-02 (addressed to Dana):
Your wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["Charlie said: \\"Dana's wedding on 2025-06-14 was \
the best party I attended in 2025.\\""] }}

Example 4 -- speech-act IS the event; multi-particular single event.
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll \
be there every week from now on.
->
{{ "memories": ["Dana said: \\"I promise to stop missing the \
Thursday mandolin practice and to attend every week going \
forward.\\""] }}

Example 5 -- two distinct events get two statements; the second has \
a resolved different date.
MESSAGE FROM Alice on 2026-05-18:
I summited Half Dome this morning and I'm baking a chocolate cake \
for Bob's birthday tomorrow.
->
{{ "memories": ["Alice said: \\"I summited Half Dome this \
morning.\\"", "Alice said: \\"I am baking a chocolate cake for Bob's \
birthday on 2026-05-19.\\""] }}

Example 6 -- bare filler; emit an empty list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Output: a JSON object {{ "memories": [...] }}. The list is empty when \
the message contains no specific content. Every non-empty entry MUST \
have the shape ``{speaker} said: "<utterance>"``.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    """Structured response from the rewriting language model."""

    memories: list[str]


def _format_neighbors(before: list, after: list) -> str:
    lines: list[str] = []
    if before:
        lines.append("PRIOR TURNS (context only, do not emit):")
        for ev in before:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    if after:
        lines.append("LATER TURNS (context only, do not emit):")
        for ev in after:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


class RewriteSegmenter(Segmenter):
    """Quoted-attribution rewrite segmenter (single-channel embed).

    Each emitted statement has the shape ``{speaker} said: "..."`` --
    the 3p attribution lives outside the quote, the raw 1p speech lives
    inside it. The segment block text IS the embed text (one channel,
    both surfaces). ``NullContext`` is used so the framework does not
    prepend the speaker again.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_QUOTED,
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

    async def _rewrite_chunk(
        self,
        chunk: str,
        speaker: str,
        date: str,
        neighbors_block: str,
    ) -> list[str]:
        prompt = self._prompt_template.format(
            speaker=speaker,
            date=date,
            passage=chunk,
            neighbors_block=neighbors_block,
        )
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
                        memories = await self._rewrite_chunk(
                            chunk_stripped, speaker, date_str, neighbors_block
                        )
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
