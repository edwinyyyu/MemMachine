"""Raw-segmenter + LLM-deriver, NO DATE HANDLING (v4).

Hypothesis under test
---------------------

In our current rawseg architecture, the answerer always sees the raw event
formatted with a framework header ``[<timestamp>] {producer}: {raw_text}``.
The timestamp is therefore ALWAYS available to the answerer for "when"
questions, regardless of what the deriver writes.

Past evidence has shown that adding inline date tokens to the deriver's
output (v22 baseline behaviour) appears to help retrieval:
  - nomsgdate (drop redundant inline message-date): -57q
  - nodaterule (drop v22 segmenter date rules): -37q
  - naturaltime-v2 (preserve speaker wording, no resolution): -17q at K=9

But: those experiments showed that REMOVING the date rule without a
replacement leaves the model handling dates INCONSISTENTLY (sometimes
adding "on YYYY-MM-DD", sometimes not, sometimes wrong). That
inconsistency itself plausibly drives the loss, not absence-of-date.

Counter-hypothesis (user-driven): if NEARLY EVERY segment ends with the
same kind of inline date token, embeddings cluster around dates and lose
differentiation. "When" queries get biased toward dated segments even
though the framework timestamp would have answered them.

This variant tests the cleanest counterfactual: a deriver prompt that
explicitly forbids ANY date emission. Not "preserve verbatim" (which
still leaks dates the speaker said). Not "resolve" (which adds dates).
Just: do not include date or time information in the rewrite, at all.

If date tokens are pure noise -> this matches or beats v22-rawev at
equal token budget. If dates do real retrieval work -> this loses
again, and we can stop pursuing this lever.

Architecture
------------

Segmenter: RawChunkSegmenter (same as v1/v3)
Deriver: emits a SINGLE clean 3p rewrite per non-filler segment.
        The rewrite preserves every concrete particular EXCEPT dates,
        times, and date-anchored phrases. Pure filler -> [] (segment
        invisible to retrieval, but raw event still in storage).

block.text = raw chunk (UNCHANGED from source)
context = RawSegmentEventContext(producer, before, current_event_text)
derivative.text = 3p rewrite without dates

Embedding input (via _format_with_context for RawSegmentEventContext):
  "{producer}: {3p_rewrite_no_dates}"

Display (framework formatting):
  "[<timestamp>] {producer}: {raw_chunk}"  -- timestamp + speaker + raw

The answerer thus sees the timestamp; the embedder does not.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    RawSegmentEventContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import (
    Deriver,
)
from pydantic import BaseModel


PROMPT_DERIVE_V4_NODATE = """\
You are generating a SEARCH KEY for a message that has already been \
stored verbatim. Your output is what semantic search will match against \
when a future user asks a question. The user will see the ORIGINAL \
message text (with its timestamp) at answer time, not your output. Your \
job is solely to maximize the chance that future queries about any \
specific content in the message match this segment.

OUTPUT a JSON object {{ "rewrite": "..." }}. The rewrite is a single \
string, one or more sentences, in the third-person voice about \
{speaker}.

KEEP what is specific to this message; DROP what is interchangeable \
across similar messages. Specific content includes: names of people, \
places, organizations, brands, named objects, named activities; \
quantities, durations, identifiers, titles, quoted phrases, proper \
nouns; decisions, plans, preferences, opinions, relationships, roles, \
emotional states tied to events; described events (something that \
happened or will happen); attached-media descriptions.

NO DATES OR TIMES IN THE REWRITE. Do not include any year, month, day, \
date, time-of-day, "yesterday"/"today"/"tomorrow", relative time \
phrases ("last week", "next month", "three years ago"), or "on \
YYYY-MM-DD" style anchors. The framework already provides the message \
timestamp to the answerer. The rewrite should describe WHAT happened \
and WHO/WHERE was involved, not WHEN.

RESOLVE pronouns: first-person ``I``, ``my``, ``me`` -> {speaker}. \
Second-person ``you`` -> the addressee's name when known from context. \
Demonstratives and ambiguous pronouns -> their concrete referents. The \
first occurrence of any queryable entity is named; subsequent \
references within the same rewrite may use natural pronouns.

PRESERVE every CONCRETE PARTICULAR verbatim -- names, places, numbers \
(other than dates), identifiers, distinctive phrasing, quoted phrases, \
attached-media descriptions. Generic abstractions or stock paraphrases \
for specifics are FAILURES.

PRESERVE polarity, direction, and emotional tone. "Used to" implies no \
longer; "didn't get to bed until 2 AM" implies a late end.

REPORT content as facts, not as speech acts. Drop ``{speaker} said \
that ...``, ``{speaker} told X that ...``, ``{speaker} mentioned that \
...`` framing unless the speech-act itself is the event (a promise, an \
apology, an explicit announcement, a question whose phrasing is the \
searchable particular).

DROP PURE FILLER. If the message is ENTIRELY a phatic opener or closer \
with no specific content (``Hi``, ``Hey``, ``Bye``, ``Take care``, \
``Cool!``, ``Lol``, standalone ``Yes``/``No``/``Thanks``/``Got it``), \
output an empty string ``""``. The segment will then be invisible to \
retrieval, which is correct. A response that CARRIES specific content \
even if short (``Yes, the purple one``, ``I researched adoption \
agencies``) is NOT pure filler -- include the specific content.

NEIGHBORING TURNS appear before the message strictly to help resolve \
addressees, demonstratives, anaphora, and unresolved references. \
Content drawn from neighbors is NEVER emitted -- only content drawn \
from this message.

{neighbors_block}MESSAGE FROM {speaker}:
{passage}"""


class _DeriveResponse(BaseModel):
    rewrite: str


def _format_neighbors(before) -> str:
    lines: list[str] = []
    if before:
        lines.append("PRIOR TURNS (resolution context only, do not emit):")
        for ev in before:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


class NoDateLLMRewriteDeriver(Deriver):
    """v4 deriver: single 3p rewrite per non-filler segment, NO dates.

    Tests whether removing date tokens from the embedding text closes
    the loss seen in past no-date variants (nodaterule, naturaltime-v2)
    -- by routing date answering through the framework timestamp shown
    to the answerer at display time, rather than through the embedding.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DERIVE_V4_NODATE,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        if not isinstance(segment.block, TextBlock):
            return []

        if not isinstance(segment.context, RawSegmentEventContext):
            return [
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=segment.context,
                    block=TextBlock(text=segment.block.text),
                    properties=segment.properties,
                )
            ]

        producer = segment.context.producer
        before = segment.context.before
        event_text = segment.context.current_event_text
        neighbors_block = _format_neighbors(before)

        prompt = self._prompt_template.format(
            speaker=producer,
            passage=event_text,
            neighbors_block=neighbors_block,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_DeriveResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        rewrite = response.rewrite.strip()
        if not rewrite:
            return []

        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=rewrite),
                properties=segment.properties,
            )
        ]
