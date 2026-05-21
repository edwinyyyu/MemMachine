"""v22-temporal-v2 RewriteSegmenter — hybrid date format + LIST FAITHFULNESS.

Hypothesis
----------

v22-temporal v1 ties baseline on g3 (162/188, 86.17%) and slightly
regresses on g4 (-0.61pp). Per-cat shows the bracket prefix
``[YYYY-MM-DD]`` HELPED cat1 multi-hop on both groups (+5.41 g3 / +3.22
g4) but HURT cat2 temporal on g3 (-12.50pp) and cat4 single-hop on g4
(-2.81pp).

Diagnosis
---------

- Cat2 (temporal) regressed because LongMemEval temporal queries phrase
  dates in natural prose (``on 2024-03-12`` / ``in March 2024``). The
  bracket prefix is novel to text-embedding-3-small -- it does not match
  the natural ``on YYYY-MM-DD`` phrasing in the query embedding.
- Cat4 (single-hop) regressed because the bracket prefix dilutes the
  subject-verb-object embedding signal that direct single-hop matches
  need; the date token absorbs a non-trivial share of the embedding's
  energy.

v2 fix
------

Use a HYBRID date format -- BOTH the bracket prefix AND a natural-prose
``on YYYY-MM-DD`` inline. The bracket anchors cat1 multi-hop (kept
across both groups in v1); the inline natural form anchors cat2
temporal (matches query phrasing) and reduces cat4 dilution by giving
the embedder a familiar grammatical slot for the date token.

Also add the LIST FAITHFULNESS rule from fp-min-v2 -- LongMemEval cat1
multi-hop frequently requires enumerable evidence (``Which X has the
speaker done?``) and the v1 rewrite still loses items.

Per-segment shape
-----------------

    ``[YYYY-MM-DD] <speaker> <past-tense verb phrase> on YYYY-MM-DD.``

The date appears TWICE: once as a structured bracket prefix, once as
natural prose inside the sentence. Both forms carry the SAME event
date (resolved if relative, preserved if absolute).

Embed channel preserved
-----------------------

``RewriteContext(text_to_embed = f"{bracket_segment_text}\n{speaker}:
{raw_chunk}")`` -- v1 dual-text strategy unchanged.

Anticipated cost
----------------

The date appears twice per segment (~10 extra tokens per segment).
Across a 500-segment session that is ~5k extra tokens; modest relative
to the embedding budget.
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


PROMPT_REWRITE_V22_TEMPORAL_V2 = """\
Rewrite the MESSAGE into a JSON list of standalone third-person memory \
statements about {speaker}. Each statement is stored verbatim and \
retrieved later by semantic search and by keyword (BM25) search. Every \
statement carries the event date TWICE: as a structured bracket prefix \
for retrieval anchoring, and as natural prose inside the sentence for \
query-phrasing match.

FORMAT (every statement): ``[YYYY-MM-DD] <sentence about {speaker}> on \
YYYY-MM-DD.`` -- the bracket date and the inline ``on YYYY-MM-DD`` are \
the SAME date (the event date). Exactly one event per statement so the \
date is unambiguous; split multi-date messages into multiple \
statements.

DATES.
- The MESSAGE was sent on {date}.
- If the event is happening at the moment of sending (``right now``, \
``today``, present-tense state), both the bracket date and the inline \
``on YYYY-MM-DD`` are {date}.
- Resolve every relative time reference (``yesterday``, ``last week``, \
``three years ago``, ``next Friday``, ``the weekend``, ``tonight``, \
``recently``, ``just``) to an absolute YYYY-MM-DD anchored at {date}, \
then DELETE the relative phrase -- the bracket plus the inline \
``on YYYY-MM-DD`` carry the date.
- Preserve absolute dates that already appear in the MESSAGE; emit \
them in BOTH the bracket and the inline form.
- The ONLY date forms allowed in the sentence body are inline \
``on YYYY-MM-DD`` (matching the bracket date). Forbidden in the \
sentence body (each is a FAILURE): ``(Date: ...)``, \
``(Event date: ...)``, ``as of YYYY-MM-DD``, sentence-prefix \
``On YYYY-MM-DD, ...``.
- If the message contains no concrete event (bare filler), emit \
nothing -- do NOT fabricate a date.

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

LIST FAITHFULNESS. When the raw message contains an enumeration (a \
list of three or more comparable items: names, books, places, foods, \
brands, hobbies, languages, dates, etc.), the statement MUST preserve \
EVERY listed item verbatim. Summaries that collapse the list into \
``various``, ``several``, ``a few``, ``many``, ``some``, or a partial \
sample are FAILURES. If the list itself is the queryable content, the \
statement is a single enumeration -- not multiple statements. Two \
comma-separated phrases that describe the SAME event are not an \
enumeration; only three-or-more comparable items count.

PERSON: refer to the speaker by name ({speaker}). Resolve first-person \
self-references (``I`` / ``me`` / ``my`` / ``mine`` / ``myself``) to \
{speaker}'s name on first occurrence; subsequent references within the \
same statement use {speaker}'s pronouns. Resolve ``you`` / ``your`` to \
the addressee's NAME when known from context; resolve demonstratives \
to their concrete referents.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

EXAMPLES (neutral names; every statement carries [YYYY-MM-DD] prefix \
AND inline ``on YYYY-MM-DD``).

Example 1 -- message-time event; both date slots = {{date}}.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- the \
tonkotsu is incredible.
->
{{ "memories": ["[2026-05-18] Alice ate tonkotsu ramen at the ramen \
place on Castro Street on 2026-05-18 and found it incredible."] }}

Example 2 -- relative reference resolves to a different date; relative \
phrase removed; bracket and inline date agree.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["[2023-04-10] Bob adopted his two cockatiels on \
2023-04-10, right before he moved to Portland."] }}

Example 3 -- explicit absolute date in the message.
MESSAGE FROM Charlie on 2026-05-02:
Dana's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["[2025-06-14] Charlie attended Dana's wedding on \
2025-06-14 and considered it the best party he went to in 2025."] }}

Example 4 -- future plan; both date slots carry the future date.
MESSAGE FROM Charlie on 2026-05-18:
I'm flying to Tokyo next Friday for the AI alignment conference.
->
{{ "memories": ["[2026-05-22] Charlie will fly to Tokyo on 2026-05-22 \
for an AI alignment conference."] }}

Example 5 -- bare filler; emit an empty list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Example 6 -- LIST FAITHFULNESS: a single enumeration of comparable \
items must keep every item verbatim.
MESSAGE FROM Bob on 2026-05-18:
For book club next month I'm picking between Piranesi, Tomorrow and \
Tomorrow and Tomorrow, The Overstory, Sea of Tranquility, and Klara \
and the Sun.
->
{{ "memories": ["[2026-05-18] Bob is picking his book club selection \
on 2026-05-18 between Piranesi, Tomorrow and Tomorrow and Tomorrow, \
The Overstory, Sea of Tranquility, and Klara and the Sun."] }}

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
    """Third-person rewrite segmenter with hybrid date format.

    Each emitted segment carries
    ``RewriteContext(text_to_embed = f"{rewrite}\\n{speaker}:
    {raw_chunk}")`` -- v1 dual-text embed channel preserved. The
    rewrite line carries the event date BOTH as a bracket prefix
    ``[YYYY-MM-DD]`` (BM25 anchor) and as inline natural prose
    ``on YYYY-MM-DD`` (semantic match with query phrasing). The
    segment's stored block text is the rewrite alone so the framework's
    same-event merge under one ``[<timestamp>]`` header does not
    duplicate content.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_TEMPORAL_V2,
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
