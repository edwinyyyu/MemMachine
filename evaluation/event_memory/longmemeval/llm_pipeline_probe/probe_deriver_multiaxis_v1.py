"""Multi-axis generic-derivative deriver v1.

Generalizes the v1 generic deriver from ONE generic-shape derivative
per segment to ONE-TO-THREE generic-shape derivatives, each targeting
a different retrieval AXIS the segment could plausibly be queried
along. The whole-text derivative is still emitted unchanged.

Hypothesis: why multi-axis should beat single-axis
--------------------------------------------------

The v22 segmenter produces consolidated third-person segments that
preserve every concrete particular ("Pepper", "Edinburgh", "The
Wolves"). Diagnosis on g4 showed the dominant retrieval failure mode
(~50% of REAL fails) is MULTI-AXIS CONTENT: a single segment carries
two or more independent retrieval axes, but a single embedding
vector can only anchor ONE axis well. Example:

  Segment: "Bob said the Riverside Hawks are solid and that
            Chen-Wei's skills and leadership are amazing on
            2023-12-11."
  Plausible queries this segment is the gold for:
    Axis 1: "Which basketball team does Bob support?"
            -> generic anchor: "Bob's basketball team affiliation"
    Axis 2: "What does Bob think of Chen-Wei?"
            -> generic anchor: "Bob's opinion on Chen-Wei"
    Axis 3 (often subsumed): "What did Bob and his friend discuss?"
            -> generic anchor: "Bob's basketball conversation topic"

v1 emits only Axis 1 OR Axis 2 -- whichever the LLM picks. The other
axis is unanchored in generic-shape embedding space and gets
displaced by adjacent specific segments. v2 emits both axes as
separate derivatives sharing the same segment_uuid; whichever axis
matches the query fires, and post-pool segment_uuid dedup keeps the
top-K=7 slot accounting honest.

Risk control: the "don't emit garbage" gate
-------------------------------------------

Indexing extra generics costs vector-search budget. Two adjacent
paraphrases of the same axis ("Bob's basketball team" /
"Bob's NBA team allegiance") would crowd a separate gold-bearing
segment out of the top-K pool for OTHER queries while adding zero
new coverage. The prompt enforces three explicit gates:

  G1. ONE-AXIS DEFAULT. If the segment carries a single fact along
      a single axis ("Alice's favorite coffee is oat-milk latte"),
      emit exactly ONE generic. Multi-emit is opt-in, not default.

  G2. INDEPENDENT-AXIS TEST. Before adding a second generic, the
      model must answer "could a query match generic A without
      matching generic B?". If no, the second generic is a paraphrase
      and is suppressed.

  G3. HARD CAP AT 3. Even segments with many specifics (long trip
      recaps, multi-topic dinner conversations) emit at most 3
      generics. The cap forces the model to pick the most likely-
      to-be-queried axes rather than enumerating everything.

The output schema is a JSON list, and the prompt explicitly says
"emit 1, 2, or 3 -- decide based on how many independent axes the
fact has". Count is a model decision, not a fixed number.

Anticipated failure modes (honest)
----------------------------------

F1. Spurious second axes. The model may invent an axis that isn't
    really queryable from the segment ("Alice's opinion on her
    coffee" when the segment only states the preference). Mitigation:
    rule 4 says "each axis must be a question someone would
    plausibly ask given only the entity + category nouns in the
    segment, not a question the segment happens to ALSO answer".
    Residual risk: model still confuses "could be asked" with
    "queryable independently".

F2. Paraphrases sneaking through. "Alice's coffee preference" and
    "Alice's favorite drink" are paraphrases. G2 mitigates but model
    judgment is imperfect. Cost: vector-search slot waste; recall
    not actively hurt for the right axis.

F3. Under-emit on genuinely multi-axis content. Conservative model
    falls back to 1 when 2 would be correct. This is the SAFE
    failure direction -- worst case is parity with v1, not
    regression.

F4. Long-tail segments with 4+ axes (a 5-topic dinner recap). Cap
    forces a triage. The two axes left on the table are unanchored
    in generic space, but they're also the lowest-likelihood query
    targets per model judgment. Acceptable.

F5. Cross-axis bleed in single-vector embedding. Each generic IS
    its own embedding, so this is not actually a failure mode of
    THIS deriver -- but the WHOLE-text derivative still suffers
    from it. That's a different patch (see SurroundingEventsContext
    work).

Why neutral examples should generalize
--------------------------------------

The examples use placeholder names (Alice/Bob/Charlie/Dana/Erin) and
generic domains (puppy, mandolin, book club, gardening, hiking,
cooking class). This avoids the prior overfitting failure where
prompt examples sharing literal nouns with bench cases caused
gpt-5-nano to pattern-match prompt entities at inference time.

The RULES (G1/G2/G3 + the entity/category-noun/relational-verb
keep-vs-drop schema) carry the generalization: a segment about
"Charlie's mandolin lessons and his teacher Dana" hits the same
rules as a segment about "Tim's basketball discussion and his
trainer". Examples illustrate one-axis (rule G1 in action),
two-axis (G2 satisfied), three-axis (G3 boundary), and the empty
case (filler segment).

Caller guidance
---------------

Use this deriver paired with the v22 RewriteSegmenter +
SurroundingEventsContext with before-window=8. Cost: same as v1
(one LLM call per segment) since the prompt emits a list in a
single response. Storage cost: average ~1.6 generics per segment
(estimated from g4-shape distributions); vector storage grows ~30%
over v1, ~160% over WholeTextDeriver baseline. Query time
unchanged.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from pydantic import BaseModel

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


PROMPT_MULTIAXIS_V1 = """\
Rephrase the FACT into 1-3 generic-shape derivatives for retrieval.

A future user querying with a generic-shape question ("What X does Y \
do?", "Which Z does W use?", "What does Y think of Z?") may not name \
the specific answer value. Each derivative matches such queries by \
KEEPING entity names + category nouns + relational verbs and \
DROPPING the specific answer values (proper-noun titles, exact \
dates, lists of items, numbers).

A single fact can carry MULTIPLE independent retrieval axes. For \
example, "Alice said the Mountain Goats are great and that Bob's \
guitar work is incredible" carries TWO axes: (1) Alice's favorite \
band, (2) Alice's opinion on Bob's playing. Each deserves its own \
generic derivative because a query may match one without the other.

Rules:
- KEEP entity names (the OWNER/subject of each axis).
- KEEP category nouns the axis is ABOUT (preference, opinion, plan, \
location, recipe, hobby, language, achievement, advice, name, role, \
relationship).
- KEEP relational verbs in their natural form (likes, owns, \
recommends, plans, visited, knows, said, thinks, hopes).
- DROP specific values that would be the ANSWER content (proper-noun \
titles, exact dates, exact numbers, listed items).
- Each derivative is a short noun phrase under 12 words. No leading \
"The fact is" or wrapper text -- just the phrase.
- Count rule (READ CAREFULLY):
  * Default is ONE derivative. Emit a second or third ONLY if the \
fact carries genuinely independent retrieval axes.
  * Independent-axis test: a second derivative is justified ONLY \
if some plausible query would match it but NOT the first. If a \
candidate second derivative is just a paraphrase or a broader/\
narrower restatement of the first, DROP it.
  * Hard cap of 3. Even fact-dense segments emit at most 3. Pick \
the axes most likely to be queried.
- If the fact has no retrievable structure (pure filler, no entity, \
no category noun), emit an empty list.

Examples:

FACT: "Alice's favorite coffee is an oat-milk latte from the corner \
cafe."
GENERICS: ["Alice's coffee preference"]
(One axis. The fact is only about a preference.)

FACT: "Bob said the Mountain Goats are great and that Charlie's \
guitar work is incredible on 2024-02-08."
GENERICS: ["Bob's favorite band", "Bob's opinion on Charlie's \
guitar playing"]
(Two axes. A query for "what band does Bob like" matches the first; \
a query for "what does Bob think of Charlie's playing" matches the \
second. Independent.)

FACT: "Dana visited the Outer Hebrides for two weeks, said the \
ferry from Oban cost about ninety pounds, and stayed with her \
cousin Erin who lives in Stornoway."
GENERICS: ["Dana's recent travel destination", "cost of Dana's \
ferry trip", "Dana's relative living abroad"]
(Three axes. Trip location, trip cost, and family-living-there are \
independently queryable.)

FACT: "Charlie joined a book club that meets Tuesdays at the public \
library."
GENERICS: ["Charlie's hobby"]
(One axis. Time and place are details of the same hobby, not \
separate axes.)

FACT: "Erin recommended the chickpea curry recipe and said her \
grandmother taught it to her in 2019."
GENERICS: ["Erin's recommended recipe", "origin of Erin's recipe"]
(Two axes. The recommendation and the family origin are \
independent.)

FACT: "Alice told Bob that her puppy is named Mochi and weighs \
about six pounds."
GENERICS: ["Alice's puppy's name", "Alice's puppy's weight"]
(Two axes. Name and weight are independently queryable category \
nouns, both anchored on the same entity.)

FACT: "yeah totally"
GENERICS: []
(No entity, no category noun, no retrievable structure.)

Output: a JSON object {{ "generics": ["...", ...] }}. The list has \
1, 2, or 3 entries, or is empty when the fact has no retrievable \
structure.

FACT: {segment_text}"""


class _MultiAxisResponse(BaseModel):
    generics: list[str]


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


class GenericDeriver(Deriver):
    """Emits whole-text + 0-3 generic-shape axis derivatives per segment.

    Output:
      1. WHOLE -- block.text = _format_with_context(segment.context,
         segment.block.text). Identical to WholeTextDeriver.
      2. GENERICS -- 0 to 3 derivatives, one per independent
         retrieval axis the segment carries. Count is decided by
         the LLM per the prompt's count rule + independent-axis
         test.

    All derivatives share segment_uuid; the vector store ingests
    them independently and retrieval dedupes by segment_uuid
    post-pool so the top-K=7 slot count stays unchanged.

    Args:
        language_model: LanguageModel used to generate the generic
            paraphrases. Configure model + reasoning_effort at
            construction.
        prompt_template: A ``.format(segment_text=...)`` template.
            Defaults to PROMPT_MULTIAXIS_V1.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_MULTIAXIS_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    async def _generate_generics(self, segment_text: str) -> list[str]:
        prompt = self._prompt_template.format(segment_text=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_MultiAxisResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        cleaned: list[str] = []
        for generic in response.generics or []:
            stripped = (generic or "").strip()
            if stripped:
                cleaned.append(stripped)
        # Enforce the prompt's hard cap defensively, in case the LLM
        # disregards the cap of 3.
        return cleaned[:3]

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

        # Always emit the whole-text derivative.
        derivatives: list[Derivative] = [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=whole_text),
                properties=segment.properties,
            )
        ]

        # Generate 0-3 generic-shape axis derivatives.
        generics = await self._generate_generics(text)
        for generic in generics:
            derivatives.append(
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=NullContext(),
                    block=TextBlock(text=generic),
                    properties=segment.properties,
                )
            )

        return derivatives
