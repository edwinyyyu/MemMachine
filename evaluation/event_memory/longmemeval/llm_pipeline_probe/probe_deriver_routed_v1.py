"""Content-routed deriver v1.

Routes derivative SHAPE based on the segment's content type rather
than applying a single uniform shape across all segments. Always
emits the whole-text derivative (WholeTextDeriver baseline) and adds
0-N routed derivatives selected by the content classification.

Hypothesis
----------

Prior single-shape variants (tag-suffix, multi-axis, list-extraction,
topic-cue, generic-v1, subjectline, stable, qshape) each helped one
question pattern but hurt another. Cross-checking 8 variants on
LoCoMo g3 + g4 produced an empirical routing table -- each question
pattern has ONE shape that helps without regression, but no single
shape helps across all patterns:

  date_in_q          --> tag-suffix on dates (Pattern A: paraphrase
                         hurts; preserving date in the embed +
                         category-noun tags is the only no-regression
                         move)
  superlative /
  stable preference  --> multi-axis NPs (1-2 angles, each a noun
                         phrase; helps preference + multi-hop on g4)
  enumerable list
  (3+ items)         --> single list-extraction NP ("X's pet names",
                         "Y's hobbies"); helps list-shape queries
                         without enum crowding
  one-time event,
  no date            --> NONE (whole only); broad anchors crowd top-K
                         (Pattern B)
  filler / chitchat  --> NONE (no retrievable content;
                         Pattern C/D risk)

The hypothesis is that a SINGLE LLM call can both (a) classify the
segment's content type and (b) emit the appropriate derivative shape
in one structured response, achieving the per-pattern Pareto front
without paying for multiple shape passes. Most segments in
conversational data should route to NONE (one-time-event / filler),
so the average extra-derivative count per segment is low.

Routing table (implemented in PROMPT_ROUTED_V1)
-----------------------------------------------

  Segment content                         Routed shape         Count
  --------------------------------------- -------------------- -----
  Date / event-on-date                    tag_suffix            1
  Comparison / preference / favorite /
    stable attribute                      multi_axis            1-2
  Enumerable list (3+ items)              list_extraction       1
  Single one-time event without date      none                  0
  Generic filler (greetings, ack.)        none                  0

A segment may match more than one row -- e.g. "Alice's favorite
recipe is the chickpea curry her grandmother taught her in 2019" has
both a preference axis AND a date. The prompt permits the LLM to emit
multiple shapes in one response when independent axes exist, but
encourages a single dominant shape to avoid crowding.

Output schema
-------------

  { "derivatives": [
      { "shape": "tag_suffix", "values": ["category tag", ...] },
      { "shape": "multi_axis", "values": ["NP1", "NP2"] },
      { "shape": "list_extraction", "values": ["X's pet names"] },
      { "shape": "none", "values": [] }
    ]
  }

The "values" field's meaning depends on the shape:
- tag_suffix: each value is a 1-3-word lowercase category noun;
  ALL values are appended to the whole-text derivative as
  `[tags: t1, t2, ...]`. The shape is realized AS A SUFFIX on the
  whole-text derivative, not as separate derivatives. Empty values
  means no suffix.
- multi_axis: each value is a 4-12-word NP; each value becomes its
  own additional derivative (NullContext, segment_uuid shared).
- list_extraction: a single value (the list-shape NP); becomes one
  additional derivative.
- none: no extra derivatives, no suffix. Whole-text only.

The deriver always emits the whole-text derivative (optionally with
tag-suffix). Routed shapes other than tag_suffix add additional
derivatives that share segment_uuid for post-pool deduplication.

Anticipated failure modes
-------------------------

- MISCLASSIFICATION. The LLM picks the wrong shape (e.g. routes a
  preference segment to tag_suffix, missing the multi-axis lift).
  Mitigation: the prompt's diagnostic question makes content-type
  observable ("is the answer a SPECIFIC VALUE the user could have
  named?", "does the segment mention a date?", "is there a LIST of
  3+ items?"). Each example shows the diagnostic answer alongside
  the routing choice. Cross-model sanity-check planned.

- OVER-EMISSION (multiple shapes). The LLM emits two or three
  shapes for a segment that should only get one. This is partly
  intentional (Pattern A + multi-axis can compound for date-anchored
  preferences) but risks Pattern B crowding. The prompt's count
  rule: prefer one shape; emit multiple shapes only when independent
  axes exist; multi_axis values still capped at 2.

- UNDER-EMISSION (NONE everywhere). The LLM routes most segments
  to "none" out of conservatism, collapsing to WholeTextDeriver
  baseline. The prompt frames "none" as the default for ONLY two
  cases (single one-time event without date, filler) and gives
  positive examples for each productive shape so the LLM has
  pattern-matchable templates.

- SHAPE CONFUSION. The LLM emits a tag_suffix value with a date
  ("baking activity on 2024-09-12") or a multi_axis NP that includes
  the specific answer value. Mitigation: the prompt's "values" rules
  per shape are explicit and the examples model the right granularity.

- LIST FALSE POSITIVES. The LLM emits list_extraction for a segment
  with 2 items (not 3+) or for a single fact phrased with a comma
  ("Alice's hobbies are reading"). Mitigation: explicit "3 or more
  items" rule with a negative example for the 1-2-item case.

- CROWDING REGRESSION. If even 10-15% of one-time-event segments
  get routed to a productive shape by mistake, top-K gets diluted
  (Pattern B). The whole-text always being emitted means worst-case
  regression is bounded -- the bare-whole baseline survives in the
  pool even when extra derivatives mislead retrieval.

Expected behavior
-----------------

On LoCoMo conversational data:
- ~50-65% of segments route to NONE (one-time events / filler /
  greetings). These get the whole-text only.
- ~20-30% route to tag_suffix (date-anchored facts). These get the
  whole-text + suffix in a single derivative (no extra vector).
- ~10-15% route to multi_axis (stable preferences, comparisons).
  These add 1-2 extra derivatives.
- ~3-7% route to list_extraction (genuine 3+ item lists, e.g.
  pet names, hobby lists, language lists). These add 1 derivative.

Average extra-derivative count per segment: ~0.4-0.6 (lower than
multi-axis-everywhere baseline of ~1.6), with the additions targeted
at segments that empirically benefit from a routed shape.

Generalizability argument
-------------------------

The routing decision is made on PRINCIPLES (content-type observable
from segment text), not benchmark-specific keywords. The shape
prescriptions encode "what kind of query targets this content":
- date-bearing segments need date preserved + category-noun anchors
  because date-anchored queries need the date present in the embed
- preference/comparison segments need angle-noun phrases because
  superlative queries name the angle but not the value
- enumerable lists need a list-shape NP because list queries
  ("what dogs does X have?") embed nowhere near a long prose list
- one-time events without date are best left as whole-text because
  any added anchor competes with their already-specific embedding

These properties apply to any conversational long-term-memory corpus,
not just LoCoMo. Examples use neutral domains (Alice/Bob/Charlie/
Dana, mandolin/cake/dog/team) so the model generalizes the routing
rule across domains.

Cost
----

One LLM call per segment at ingest. Parallelizable. Storage cost is
lower than uniform multi-axis (NONE routes save extra vectors).
Query time unchanged.

Caller guidance
---------------

Pair with the v22 RewriteSegmenter + SurroundingEventsContext
(before-window=8). The router sees the segment text only; the
classification decision is local to the segment's content, not the
surrounding conversation.
"""

from __future__ import annotations

from typing import Literal, override
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    Derivative,
    NullContext,
    ProducerContext,
    RewriteContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import (
    Deriver,
)


PROMPT_ROUTED_V1 = """\
Classify the SEGMENT's content type and emit the appropriate \
derivative SHAPE(s) for retrieval.

A future user querying memory targets different anchors for \
different content types. Picking the wrong anchor either dilutes \
the segment's embedding (over-broad anchors compete with unrelated \
segments) or drops a load-bearing token (e.g. dropping a date from \
a date-bearing segment). The right shape per content type is:

  CONTENT TYPE                             SHAPE
  --------------------------------------- --------------------
  Date / event-on-date                    tag_suffix
  Comparison / preference / favorite /
    stable attribute                      multi_axis
  Enumerable list (3+ items)              list_extraction
  Single one-time event WITHOUT date      none
  Generic filler (greetings, acks)        none

Diagnostic questions to ask BEFORE picking a shape:
  Q1: "Does the segment mention a specific calendar date or named \
event-day?" If YES, lean tag_suffix.
  Q2: "Is the segment about a stable preference, opinion, favorite, \
comparison, or stable attribute the entity holds?" If YES, lean \
multi_axis.
  Q3: "Does the segment enumerate 3 or more comparable items \
(names, hobbies, languages, places, etc.)?" If YES, lean \
list_extraction.
  Q4: If none of Q1-Q3 fits, route to none.

A segment may match more than one diagnostic -- e.g. a dated \
preference disclosure. In that case emit BOTH shapes in the same \
response. Default to one shape; emit a second only when the second \
diagnostic is genuinely independent of the first.

Shape rules:

  tag_suffix:
    - "values" is a list of 1-6 lowercase 1-3-word CATEGORY NOUNS \
that name what the segment is ABOUT (not the answer value).
    - Tags are appended as `[tags: t1, t2, ...]` to the whole-text \
embed. They are lexical anchors for keyword search; they should be \
GENERIC enough that an unrelated segment about a different specific \
instance could share the same tag.
    - DO NOT include the answer value (no proper-noun titles, no \
exact dates, no exact numbers) in tag values.

  multi_axis:
    - "values" is a list of 1 or 2 SHORT NOUN PHRASES (under 12 \
words) naming the INDEPENDENT retrieval axes the segment carries.
    - Each axis is a "{{entity}}'s {{category noun}}" or a similar \
generic-shape NP. KEEP entity names + category nouns; DROP the \
specific answer value.
    - Emit 2 axes ONLY when the segment carries TWO independent \
queryable axes (e.g. a preference AND an opinion about a different \
target). If a candidate second axis is just a paraphrase of the \
first, emit only 1.
    - Each axis becomes a separate derivative.

  list_extraction:
    - "values" is a single-element list with the list-shape NP \
naming WHAT IS BEING ENUMERATED.
    - Phrase as "{{entity}}'s {{list category noun}}" -- the noun \
that names the list itself, not a sample of the items.
    - Only when the segment enumerates 3+ items.

  none:
    - "values" is an empty list.
    - Use when the segment is filler, or when the segment is a \
single one-time event without a date.

Output format:
  {{ "derivatives": [
      {{ "shape": "tag_suffix", "values": ["..."] }},
      {{ "shape": "multi_axis", "values": ["..."] }}
    ]
  }}

You may include 0, 1, or 2 entries in the "derivatives" array. If \
you include 0 entries, the segment routes entirely to NONE (whole-\
text only). NEVER duplicate the same shape twice in one response.

Examples:

SEGMENT: "Alice baked a chocolate-cardamom cake for Bob's birthday \
on 2024-09-12 and said it was her favorite recipe right now."
DIAGNOSTIC: Q1 YES (date present, 2024-09-12). Q2 YES (favorite \
recipe right now). Q3 NO. --> tag_suffix + multi_axis.
RESPONSE:
  {{ "derivatives": [
      {{ "shape": "tag_suffix", "values": ["cake recipe", \
"birthday gift", "baking activity"] }},
      {{ "shape": "multi_axis", "values": ["Alice's favorite \
recipe"] }}
    ]
  }}

SEGMENT: "Charlie said his dogs are named Pepper, Precious, and \
Panda, and that he adopted all three from the local shelter."
DIAGNOSTIC: Q1 NO (no date). Q2 NO (not a preference). Q3 YES \
(3 dog names enumerated). --> list_extraction.
RESPONSE:
  {{ "derivatives": [
      {{ "shape": "list_extraction", "values": ["Charlie's dog \
names"] }}
    ]
  }}

SEGMENT: "Dana said Edinburgh would be a better city for the team \
retreat than Glasgow because the venues are closer to the airport."
DIAGNOSTIC: Q1 NO. Q2 YES (comparison + preference between two \
cities). Q3 NO. --> multi_axis.
RESPONSE:
  {{ "derivatives": [
      {{ "shape": "multi_axis", "values": ["Dana's preferred \
retreat city", "Dana's reasoning on retreat venue"] }}
    ]
  }}

SEGMENT: "Bob mentioned he ran into Alice at the coffee shop \
yesterday and they chatted for a few minutes about her new book."
DIAGNOSTIC: Q1 NO (yesterday is not a calendar anchor in this \
embed). Q2 NO. Q3 NO. Single one-time encounter. --> none.
RESPONSE:
  {{ "derivatives": [] }}

SEGMENT: "Dana knows Spanish, Portuguese, French, and a little \
Italian besides English."
DIAGNOSTIC: Q1 NO. Q2 NO. Q3 YES (4 languages enumerated). --> \
list_extraction.
RESPONSE:
  {{ "derivatives": [
      {{ "shape": "list_extraction", "values": ["Dana's languages \
known"] }}
    ]
  }}

SEGMENT: "Charlie said hi."
DIAGNOSTIC: filler. --> none.
RESPONSE:
  {{ "derivatives": [] }}

SEGMENT: "Alice's appointment at the dental clinic is scheduled for \
2025-03-15 at 9 a.m."
DIAGNOSTIC: Q1 YES (calendar date). Q2 NO. Q3 NO. --> tag_suffix.
RESPONSE:
  {{ "derivatives": [
      {{ "shape": "tag_suffix", "values": ["dental appointment", \
"upcoming appointment"] }}
    ]
  }}

SEGMENT: "Bob's favorite mandolin teacher is Charlie because his \
lessons cover both technique and ear training."
DIAGNOSTIC: Q1 NO. Q2 YES (favorite teacher). Q3 NO. --> multi_axis.
RESPONSE:
  {{ "derivatives": [
      {{ "shape": "multi_axis", "values": ["Bob's mandolin \
teacher"] }}
    ]
  }}

SEGMENT: "Dana mentioned her hobbies include rock climbing, \
mandolin, cooking, and gardening."
DIAGNOSTIC: Q1 NO. Q2 NO (not a preference between options, just \
an enumeration). Q3 YES (4 hobbies). --> list_extraction.
RESPONSE:
  {{ "derivatives": [
      {{ "shape": "list_extraction", "values": ["Dana's hobbies"] }}
    ]
  }}

SEGMENT: "yeah totally"
DIAGNOSTIC: filler, no entity, no content. --> none.
RESPONSE:
  {{ "derivatives": [] }}

SEGMENT: {segment_text}"""


class _RoutedShape(BaseModel):
    shape: Literal["tag_suffix", "multi_axis", "list_extraction", "none"]
    values: list[str] = Field(default_factory=list)


class _RoutedResponse(BaseModel):
    derivatives: list[_RoutedShape] = Field(default_factory=list)


def _format_with_context(context: Context, text: str) -> str:
    """Mirror WholeTextDeriver's context formatting.

    For RewriteContext, returns text_to_embed (the
    rewrite+speaker+raw-chunk dual-text embed). For other contexts,
    returns the speaker-prefixed text or the bare text.
    """
    match context:
        case ProducerContext(producer=producer):
            return f"{producer}: {text}"
        case SurroundingEventsContext(producer=producer):
            return f"{producer}: {text}"
        case NullContext():
            return text
        case RewriteContext(text_to_embed=text_to_embed):
            return text_to_embed
        case _:
            raise NotImplementedError(
                f"Unsupported context type: {type(context).__name__}"
            )


def _clean_tag_values(raw_values: list[str]) -> list[str]:
    """Strip, lowercase, dedupe, cap at 6 for tag-suffix values."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        if not isinstance(raw, str):
            continue
        tag = raw.strip().lower()
        if not tag:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
        if len(cleaned) >= 6:
            break
    return cleaned


def _clean_axis_values(raw_values: list[str], cap: int) -> list[str]:
    """Strip, drop empties, dedupe (case-insensitive), cap at `cap`."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        if not isinstance(raw, str):
            continue
        axis = raw.strip()
        if not axis:
            continue
        key = axis.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(axis)
        if len(cleaned) >= cap:
            break
    return cleaned


class GenericDeriver(Deriver):
    """Content-routed deriver: whole-text + 0-N shape-routed derivatives.

    Single LLM call per segment classifies content type and emits a
    structured response listing one or more derivative shapes. The
    deriver always emits the whole-text derivative (optionally with a
    tag-suffix appended) and additionally emits one derivative per
    multi_axis value and one derivative per list_extraction value.

    Routing table:
      tag_suffix       --> APPENDED to whole-text as `[tags: ...]`
                           (no extra vector)
      multi_axis       --> 1-2 extra derivatives (NullContext)
      list_extraction  --> 1 extra derivative (NullContext)
      none             --> whole-text only

    All derivatives share segment_uuid; the vector store ingests them
    independently and retrieval dedupes by segment_uuid post-pool so
    the top-K slot count stays unchanged.

    Args:
        language_model: LanguageModel used to generate the routed
            response. Configure model + reasoning_effort at
            construction.
        prompt_template: A ``.format(segment_text=...)`` template.
            Defaults to PROMPT_ROUTED_V1.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_ROUTED_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    async def _generate_routed(
        self, segment_text: str
    ) -> tuple[list[str], list[str], list[str]]:
        """Return (tag_values, multi_axis_values, list_extraction_values).

        Each list is independently cleaned and capped per shape rules.
        Any shape that was not emitted (or `none`) yields an empty
        list. The deriver caller uses the three lists to assemble the
        final derivative set.
        """
        prompt = self._prompt_template.format(segment_text=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_RoutedResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return ([], [], [])

        tag_values: list[str] = []
        multi_axis_values: list[str] = []
        list_extraction_values: list[str] = []

        # Each shape may appear at most once; if the LLM duplicates a
        # shape, last-write-wins.
        for shape_entry in response.derivatives:
            match shape_entry.shape:
                case "tag_suffix":
                    tag_values = _clean_tag_values(shape_entry.values)
                case "multi_axis":
                    multi_axis_values = _clean_axis_values(
                        shape_entry.values, cap=2
                    )
                case "list_extraction":
                    list_extraction_values = _clean_axis_values(
                        shape_entry.values, cap=1
                    )
                case "none":
                    # `none` carries no values; if the LLM emitted it
                    # alongside another shape, ignore it (the other
                    # shape's emit stands).
                    continue

        return (tag_values, multi_axis_values, list_extraction_values)

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                pass
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )

        whole_text = _format_with_context(segment.context, text)

        tag_values, multi_axis_values, list_extraction_values = (
            await self._generate_routed(text)
        )

        # Whole-text derivative, with optional tag suffix appended.
        suffix = (
            f" [tags: {', '.join(tag_values)}]" if tag_values else ""
        )
        whole_embed_text = whole_text + suffix

        derivatives: list[Derivative] = [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=whole_embed_text),
                properties=segment.properties,
            )
        ]

        # Multi-axis derivatives (0-2). Each axis NP is its own
        # NullContext derivative; the segment_uuid links them back
        # to the source segment for post-pool dedup.
        for axis in multi_axis_values:
            derivatives.append(
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=NullContext(),
                    block=TextBlock(text=axis),
                    properties=segment.properties,
                )
            )

        # List-extraction derivative (0 or 1).
        for list_np in list_extraction_values:
            derivatives.append(
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=NullContext(),
                    block=TextBlock(text=list_np),
                    properties=segment.properties,
                )
            )

        return derivatives
