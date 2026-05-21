"""v22-quoted-v2 RewriteSegmenter -- terse speaker-prefix rewrite.

Hypothesis
----------

v22-quoted v1 used ``{speaker} said: "<utterance>"`` and regressed g3
by -7.45pp vs v22 baseline (78.72% vs 86.17%), with the worst hit on
cat4 single-hop direct retrieval (-10.5pp). Per-segment ``Alice said:
"I live in Portland."`` distributes embedding attention across the
5-8 token ``said: "..."`` wrapper, weakening cosine match against
short direct queries like ``Where does Alice live?``. The verbose
quote also encouraged the rewriter to pad each utterance.

v2 drops the wrapper. Format becomes just the conversational-turn
prefix:

  ``{speaker}: <utterance>``

This matches v22 baseline's ``RewriteContext.text_to_embed`` raw-chunk
half (``{speaker}: {raw_chunk}``) but as a single denser channel:
3p attribution by speaker name + 1p verbatim-ish speech. No
``said`` / no quote marks -- fewer wrapper tokens, more signal
density per segment.

Also adds a LIST FAITHFULNESS rule (verbatim from fp-min-v2): list
and enumeration content is preserved item-for-item so queries that
target a specific item still hit the segment.

Design
------

  - segment.block.text format: ``Alice: I live in Portland.``
  - Inside the utterance, speaker uses their own ``I`` / ``my``.
  - Cross-speaker references (``you``) resolve to addressee NAMES.
  - Dates resolved to ``on YYYY-MM-DD``; omitted entirely when the
    resolved date equals {date}.
  - LIST FAITHFULNESS rule preserves enumerations.
  - One event per statement; one date per statement.
  - ``NullContext`` -- the speaker name is already in the segment
    text; do not duplicate via ProducerContext.
  - Neutral names (Alice/Bob/Charlie/Dana) and neutral domains
    (cockatiels/mandolin/Half Dome/Big Sur/cake).
  - Prompt budgeted to <= ~1000 tokens.
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


PROMPT_REWRITE_V22_QUOTED_V2 = """\
Rewrite the MESSAGE into a JSON list of standalone speaker-prefixed \
memory statements. Each statement is stored verbatim and retrieved \
later by semantic search.

FORMAT. Every statement is exactly:

  {speaker}: <utterance>

No quote marks, no ``said``, no other wrapper. The utterance \
preserves {speaker}'s own first-person voice -- {speaker} speaks in \
their own ``I`` / ``my`` / ``me`` / ``myself`` / ``we``. Cross-speaker \
references resolve to addressee NAMES when known from context (e.g. \
``you`` -> ``Bob`` when Bob is the addressee). Demonstratives \
(``this``, ``that``, ``it``, ``they``) resolve to their concrete \
referents.

KEEP specific content (names, places, dates, numbers, decisions, \
plans, preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions).

LIST FAITHFULNESS. When the message enumerates items (a list of \
foods, names, places, steps, options, etc.), the utterance preserves \
EVERY item by its exact surface form. Do not summarize ``A, B, and C`` \
as ``several things`` or drop tail items. A query targeting any one \
item must still hit the statement.

ONE statement per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of one occurrence is ONE statement whose \
utterance contains all its particulars. Distinct events (different \
times, different occasions, different actions) each get their own \
statement. Report content, not the speech-act of conveying it -- \
``I said that ...`` / ``I told X that ...`` wrappers are dropped \
unless the speech-act itself is the event (a promise, an apology, an \
announcement).

DATES. The MESSAGE was sent on {date}. The framework prepends the \
message timestamp automatically, so the utterance MUST NOT contain \
{date} in any form. Resolve every relative time reference \
(``yesterday``, ``last week``, ``three years ago``, ``next Friday``, \
``the weekend``, ``today``, ``tonight``, ``recently``, ``now``, \
``just``) to an absolute date anchored at {date}.
  - If the resolved date EQUALS {date}, the utterance contains NO \
date and NO relative phrase.
  - If the resolved date DIFFERS from {date}, weave it in as \
``on YYYY-MM-DD`` and DELETE the original relative phrase. Keeping \
the relative phrase alongside the resolved date is a FAILURE.
One event date per statement; split multi-date messages into multiple \
statements.

Forbidden date forms (each is a FAILURE): sentence-prefix \
``On YYYY-MM-DD, ...``; parenthetical ``(Date: ...)`` or \
``(Event date: ...)``; square-bracket ``[YYYY-MM-DD]``; \
``as of YYYY-MM-DD``. Only ``on YYYY-MM-DD`` woven inline is allowed, \
and only when the resolved date differs from {date}.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content as their own statements.

EXAMPLES (neutral names and domains).

Example 1 -- message-time event; no date in the utterance.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- \
the tonkotsu is incredible.
->
{{ "memories": ["Alice: I am finally trying the ramen place on Castro \
Street and the tonkotsu is incredible."] }}

Example 2 -- relative reference resolves to a DIFFERENT date.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["Bob: I adopted my two cockatiels on 2023-04-10, \
right before I moved to Portland."] }}

Example 3 -- explicit absolute date; cross-speaker ``you`` resolves \
to addressee NAME.
MESSAGE FROM Charlie on 2026-05-02 (addressed to Dana):
Your wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["Charlie: Dana's wedding on 2025-06-14 was the best \
party I attended in 2025."] }}

Example 4 -- speech-act IS the event; single multi-particular event.
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll \
be there every week from now on.
->
{{ "memories": ["Dana: I promise to stop missing the Thursday \
mandolin practice and to attend every week going forward."] }}

Example 5 -- two distinct events get two statements.
MESSAGE FROM Alice on 2026-05-18:
I summited Half Dome this morning and I'm baking a chocolate cake \
for Bob's birthday tomorrow.
->
{{ "memories": ["Alice: I summited Half Dome this morning.", "Alice: \
I am baking a chocolate cake for Bob's birthday on 2026-05-19."] }}

Example 6 -- list faithfulness; every item preserved.
MESSAGE FROM Bob on 2026-05-18:
For the camping trip I packed my tent, sleeping bag, headlamp, \
pocket knife, and trail mix.
->
{{ "memories": ["Bob: For the camping trip I packed my tent, sleeping \
bag, headlamp, pocket knife, and trail mix."] }}

Example 7 -- bare filler; emit an empty list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Output: a JSON object {{ "memories": [...] }}. The list is empty when \
the message contains no specific content. Every non-empty entry MUST \
have the shape ``{speaker}: <utterance>``.

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
    """Terse speaker-prefix rewrite segmenter (single-channel embed).

    Each emitted statement has the shape ``{speaker}: <utterance>`` --
    speaker name as prefix (3p attribution surface) + first-person
    verbatim-ish speech (1p raw-conversational surface). No ``said``,
    no quote marks: minimal wrapper, maximal signal density. The
    segment block text IS the embed text (one channel, both surfaces).
    ``NullContext`` is used so the framework does not prepend the
    speaker again.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_QUOTED_V2,
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
