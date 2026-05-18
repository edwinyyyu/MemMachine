"""LLM-driven text deriver.

The deriver shows the segment's raw text to a `LanguageModel` and asks
for one or more derivative strings that will be embedded for semantic
search alongside the segment. The prompt is the validated v65
(see `evaluation/event_memory/longmemeval/llm_pipeline_probe/probe_deriver_v65_completeness.py`).
v65 = v64 + an explicit completeness reminder on the "ONE central
subject -> ONE derivative" trigger ("...that still covers EVERY
specific detail (names, dates, numbers, distinctive phrases). ONE
derivative is one searchable string, not a short summary."). Closes a
v64 failure mode where the model would emit a one-sentence summary
that dropped specific details. N=40 content-drop stress on
gpt-5-nano @ low: v64 had 3 catastrophic + 7 partial drops; v65 has 0.
Rule placement matters -- the reminder must be adjacent to the
"ONE derivative" trigger; moving it to EACH DERIVATIVE regressed.

Each derivative is wrapped with the segment's context the same way
`SentenceTextDeriver` / `WholeTextDeriver` wrap theirs -- a
`ProducerContext` prepends "Speaker: ", a `NullContext` is a no-op.
The LLM only sees the segment text (without speaker), so the wrapping
is applied to whatever it returns.
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
from memmachine_server.episodic_memory.event_memory.deriver.deriver import Deriver

# v65 prompt -- see project_deriver_design.md memory for the
# iteration history. Frozen here so the production class is standalone.
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


class _DeriverResponse(BaseModel):
    """Structured response from the deriver language model."""

    derivatives: list[str]


def _format_with_context(context: Context, text: str) -> str:
    """Wrap text with its producer prefix when relevant."""
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


class LLMTextDeriver(Deriver):
    """Derives derivatives from TextBlock segments via a LanguageModel.

    Args:
        language_model: The LanguageModel used to produce derivatives.
            Configure model and reasoning effort at LanguageModel
            construction time; this deriver calls
            `generate_parsed_response(output_format=_DeriverResponse, ...)`.
        prompt_template: A `.format(segment=...)` template producing the
            full prompt. Defaults to the validated v65 prompt.
        emit_at_least_one: When True (default), if the LLM returns no
            derivatives, fall back to a single derivative containing the
            segment's text. This preserves the invariant that every
            segment is searchable.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DERIVER_V65,
        emit_at_least_one: bool = True,
        max_attempts: int = 3,
    ) -> None:
        """Initialize the deriver; see class docstring for arguments."""
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._emit_at_least_one = emit_at_least_one
        self._max_attempts = max_attempts

    async def _derive_texts(self, segment_text: str) -> list[str]:
        prompt = self._prompt_template.format(segment=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_DeriverResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return list(response.derivatives)

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                pass
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )

        derivative_texts = await self._derive_texts(text)
        if not derivative_texts and self._emit_at_least_one:
            derivative_texts = [text]

        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(
                    text=_format_with_context(segment.context, derivative_text)
                ),
                properties=segment.properties,
            )
            for derivative_text in derivative_texts
        ]
