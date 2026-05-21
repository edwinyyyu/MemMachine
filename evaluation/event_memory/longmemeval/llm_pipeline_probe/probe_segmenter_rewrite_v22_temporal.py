"""v22-temporal RewriteSegmenter — date-bracket-prefix 3p rewrite.

Hypothesis
----------

Queries about events at specific dates ("What did Alice do on October
21?" / "What happened in November 2022?") benefit when the event date
is a STRUCTURED, BM25-friendly token at the front of each segment
rather than embedded mid-sentence. v22's natural-prose
``on YYYY-MM-DD`` suffix is recoverable by the embedder but BM25
weights it as just another token mid-document. Promoting the date to
a square-bracket prefix gives BM25 a strong anchor (``YYYY-MM-DD``
recovered as a single token after the bracket is stripped as a
separator), and gives the embedder a consistent leading temporal
context that should improve cat3 / cat4 (single-hop + temporal)
retrieval at the small-K Mem0-comparable budget.

Per-segment shape
-----------------

    ``[YYYY-MM-DD] <speaker> <past-tense verb phrase>.``

The bracket carries the EVENT date (resolved if relative; preserved
if absolute). Events that coincide with the message-sending moment
use ``{date}`` as the bracket date -- v22 dropped this; v22-temporal
adds it back as a STRUCTURED tag so it does not pollute the prose
channel but is still indexable.

Embed channel preserved
-----------------------

``RewriteContext(text_to_embed = f"{bracket_segment_text}\n{speaker}:
{raw_chunk}")`` -- same dual-text strategy as v22 (rewrite + raw
verbatim), only the rewrite line now leads with a bracketed date tag.

Rules kept from v22 (in min-style brevity)
------------------------------------------

  - KEEP specific content / DROP interchangeable content
  - ONE statement per EVENT (anti-fragmentation)
  - polarity preservation
  - speech-act-only-when-event
  - neighbors are context only

Rules flipped
-------------

  - emit ``[YYYY-MM-DD] ...`` prefix on EVERY segment
  - forbid any other inline date form (no ``on YYYY-MM-DD`` suffix,
    no parentheticals, no ``as of``)
  - single event per segment so date is unambiguous
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
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel


PROMPT_REWRITE_V22_TEMPORAL = """\
Rewrite the MESSAGE into a JSON list of standalone third-person memory \
statements about {speaker}. Each statement is stored verbatim and \
retrieved later by semantic search and by keyword (BM25) search. Every \
statement leads with a structured date tag for retrieval anchoring.

FORMAT (every statement): ``[YYYY-MM-DD] <sentence about {speaker}>.``
- The ``[YYYY-MM-DD]`` prefix carries the EVENT date (the date the \
event actually occurred / will occur / is occurring), not the date the \
message was sent.
- Always include the bracket prefix. Exactly one event per statement \
so the date is unambiguous; split multi-date messages into multiple \
statements.

DATES.
- The MESSAGE was sent on {date}.
- If the event is happening at the moment of sending (``right now``, \
``today``, present-tense state), the bracket date is {date}.
- Resolve every relative time reference (``yesterday``, ``last week``, \
``three years ago``, ``next Friday``, ``the weekend``, ``tonight``, \
``recently``, ``just``) to an absolute YYYY-MM-DD anchored at {date}, \
then DELETE the relative phrase from the sentence -- the bracket \
carries the date now.
- Preserve absolute dates that already appear in the MESSAGE; put them \
in the bracket, not in the sentence.
- Forbidden in the sentence body (each is a FAILURE): ``on \
YYYY-MM-DD``, ``(Date: ...)``, ``(Event date: ...)``, \
``as of YYYY-MM-DD``. The bracket prefix is the ONLY date form.
- If the message contains no concrete event (bare filler), emit \
nothing -- do NOT fabricate a bracket date.

KEEP specific content (names, places, numbers, decisions, plans, \
preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions). Preserve polarity, direction, and \
emotional tone -- ``used to`` implies no longer; ``didn't get to bed \
until 2 AM`` implies a late end, not a late start.

ONE statement per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of the same occurrence is ONE statement. \
Distinct events (different times, different occasions, different \
actions) each get their own statement. Report content, not the \
speech-act of conveying it -- ``{speaker} said that ...`` / \
``{speaker} told X that ...`` wrappers are dropped unless the \
speech-act itself is the event (a promise, an apology, an \
announcement).

PERSON: refer to the speaker by name ({speaker}). Resolve \
first-person self-references (``I`` / ``me`` / ``my`` / ``mine`` / \
``myself``) to {speaker}'s name on first occurrence; subsequent \
references within the same statement use {speaker}'s pronouns. \
Resolve ``you`` / ``your`` to the addressee's NAME when known from \
context; resolve demonstratives to their concrete referents.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

EXAMPLES (neutral names; every statement leads with [YYYY-MM-DD]).

Example 1 -- message-time event; bracket = {{date}}.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- the \
tonkotsu is incredible.
->
{{ "memories": ["[2026-05-18] Alice ate tonkotsu ramen at the ramen \
place on Castro Street and found it incredible."] }}

Example 2 -- relative reference resolves to a different date; the \
relative phrase is removed.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["[2023-04-10] Bob adopted his two cockatiels, right \
before he moved to Portland."] }}

Example 3 -- explicit absolute date in the message; date goes in \
bracket only.
MESSAGE FROM Charlie on 2026-05-02:
Dana's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["[2025-06-14] Charlie attended Dana's wedding and \
considered it the best party he went to in 2025."] }}

Example 4 -- future plan; bracket carries the future date.
MESSAGE FROM Charlie on 2026-05-18:
I'm flying to Tokyo next Friday for the AI alignment conference.
->
{{ "memories": ["[2026-05-22] Charlie will fly to Tokyo for an AI \
alignment conference."] }}

Example 5 -- bare filler; emit an empty list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Output: a JSON object {{ "memories": [...] }}. The list is empty when \
the message contains no specific event.

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
    """Third-person rewrite segmenter with date-bracket-prefix format.

    Each emitted segment carries
    ``RewriteContext(text_to_embed = f"{bracket_segment}\\n{speaker}:
    {raw_chunk}")`` -- v22's dual-text embed channel is preserved so
    the indexed string carries BOTH the date-anchored rewrite and the
    speaker-prefixed verbatim chunk. The segment's stored block text
    is the rewrite alone (with bracket prefix) so the framework's
    same-event merge under one ``[<timestamp>]`` header does not
    duplicate content.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_TEMPORAL,
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

    @staticmethod
    def _build_embed_text(rewrite: str, original_chunk: str, speaker: str) -> str:
        return f"{rewrite}\n{speaker}: {original_chunk}"

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
