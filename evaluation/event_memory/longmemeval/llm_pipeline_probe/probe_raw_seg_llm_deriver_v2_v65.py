"""Raw-segmenter + v65-style LLM-deriver (v2).

v2 uses the SHIPPED v65 LongMemEval deriver prompt verbatim, applied to
LoCoMo raw-message segments. v65 has the same "free deriver" philosophy
as v1 (empty list for pure filler, one-per-event for content) but with
battle-tested rules for:

- compound identifier preservation
- multi-derivative scope-naming (entities + scope on every derivative)
- non-prose surface handling
- bare-term derivatives

v65 was developed on LongMemEval segmenter output (LLMTextSegmenter)
where segments are coherent passages. Applied to LoCoMo where each
"segment" is a single raw message, the prompt should still work well
but lacks LoCoMo-specific neighbor-context for addressee resolution.

If v65 outperforms v1 on LoCoMo, the resolution gap may not matter or
v65's structural rules dominate it.

Architecture
------------

Segmenter: same RawChunkSegmenter as v1 (raw chunk = block.text)
Deriver: v65 prompt operating on segment text alone (no neighbors,
  no current_event_text -- since segment.block.text IS the event text
  for LoCoMo's single-chunk events)
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


PROMPT_DERIVER_V65 = """\
A SEGMENT is stored verbatim in a retrieval system. Your DERIVATIVES are \
additional strings embedded for semantic search alongside it.

Goal: a future user querying anything in this segment should find it \
via at least one derivative. Generate the FEWEST derivatives that cover \
every searchable thing. Splitting a focused thought, paraphrasing one \
fact, listing items already inline, or atomizing a single-topic \
narrative are FAILURES.

DERIVATIVES COVERAGE:
- PURE FILLER -- content-free short responses only meaningful with the \
prior message ("yes", "no", "ok", "thanks", "lol", "great point") -> \
emit ZERO derivatives (return an empty list). When such a response \
ALSO carries concrete content ("ok, leaving Tuesday at 5", "no, I \
changed my mind about Tuesday"), it is NOT pure filler -- emit \
derivatives for the content.
- A passage centered on ONE central subject -- a person, place, event, \
project, process, product, relationship, preference, opinion, \
decision, or concept -- across any number of sentences is ONE \
searchable thing -> ONE derivative that still covers EVERY specific \
detail (names, dates, numbers, distinctive phrases). ONE derivative \
is one searchable string, not a short summary. Co-mentioned \
particulars (a city, a friend, a brand) are ATTRIBUTES of the central \
subject, not new topics -- they don't each get their own derivative. \
Emitting one derivative per sentence of a single-subject narrative is \
a FAILURE.
- Several genuinely independent particulars a future query could ask \
separately -> one derivative per particular.
- A single focused statement (a function signature, a commit message, \
a definition) -> ONE near-original derivative. Do NOT split its \
clauses; do NOT paraphrase.
- A bare standalone term ("Paris") -> ONE derivative with just that \
term.
- A list of items sharing one predicate (versions of a model, rows of \
a table) -> ONE derivative naming the parent and listing items inline.
- When unsure between ONE and several derivatives, prefer ONE.

EACH DERIVATIVE:
- Is a full grammatical sentence (except bare-entity), using the \
segment's wording verbatim for CONCRETE PARTICULARS -- names, places, \
dates, numbers, identifiers, decisions, plans, preferences, opinions, \
relationships, emotional states tied to events, constraints, and \
distinctive phrasing. Drop generic abstractions or stock phrases.
- Preserves compound identifiers. When a name takes a disambiguating \
role or descriptor ("team lead Alice", "the engineering library on \
campus"), keep the whole phrase together every time -- splitting loses \
the query-binding.
- Stands alone as a search result. When you emit multiple derivatives \
for one segment, each must name the scope (the trip, project, period, \
conversation, artifact) and anyone or anything else involved -- with \
compound identifiers intact ("my wife Anne", "team lead Alice"), \
never as pronouns ("it", "they", "we") for the scope or anyone/anything \
involved. A query for the scope or anyone/anything involved finds \
only the derivatives where they were named, not the ones using \
pronouns.
- Never invents content the segment lacks.

NON-PROSE SURFACES (code, tables, encoded text, JSON, log lines, \
markup):
- Emit one DESCRIPTION derivative naming what the segment IS and what \
it is ABOUT.
- Decode encodings (Caesar, base64) into a separate prose derivative.
- For non-prose aggregates with multiple distinct entries (table rows, \
JSON array elements, log batches), optionally emit one prose \
derivative per entry.
- Never preserve pipes, code syntax, or brackets.

Output: a JSON object {{ "derivatives": [...] }} and nothing else.

SEGMENT:
{segment}"""


class _DeriveResponse(BaseModel):
    derivatives: list[str]


class V65DeriverProbe(Deriver):
    """v65 deriver prompt applied to LoCoMo raw segments."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DERIVER_V65,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        if not isinstance(segment.block, TextBlock):
            return []
        # For RawSegmentEventContext we use current_event_text as the segment
        # passage (= the raw event text). For LoCoMo single-chunk events,
        # this equals segment.block.text. For multi-chunk events it gives
        # the deriver full event context. For other context types, fall back
        # to segment.block.text.
        segment_text = segment.block.text
        if isinstance(segment.context, RawSegmentEventContext):
            segment_text = segment.context.current_event_text

        prompt = self._prompt_template.format(segment=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_DeriveResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        derivs = [d.strip() for d in response.derivatives if d and d.strip()]
        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=d),
                properties=segment.properties,
            )
            for d in derivs
        ]
