"""v22-saysmin RewriteSegmenter -- minimalist says-framed rewrite prompt.

Hypothesis
----------

v22-fp-min (first-person, minimalist, ~918 tokens, ~1.10 segs/turn) won
on g3 c124 by +1.60pp over v22-fp but lost on g4 by -4.88pp. v22-says
(third-person ``{speaker} says ...``, relative-time preserved verbatim,
RewriteContext dual-text embed, full-length prompt) tied v22-fp at 81.91%
on g3 c1 -- the speech-act framing alone did not help.

This variant combines:

  - min's STRUCTURE: short prompt, ~5 examples, terse KEEP/DROP, one-
    event-per-statement, no per-rule WHY explanations
  - says's VOICE: ``{speaker} says ...`` 3rd-person framing; first-
    person references in the raw resolve to the speaker's NAME inside
    the says clause (he/she/they); cross-speaker addressee references
    resolve to the addressee's name
  - says's RELATIVE-DATE policy: preserve relative time references
    (``yesterday``, ``last week``, ``next month``) VERBATIM. The
    framework prepends the message date as a timestamp at retrieval
    and chronological ordering lets the answerer interpret them.
    Preserve explicit absolute dates verbatim too.

Embed channel: ``ProducerContext(producer=speaker)`` (same as v22-fp-
min), single-channel embed. This isolates the variable under test
(prompt voice + date policy) from the embed-channel change tested by
v22-says's RewriteContext.

Target: <=1000 tokens for the prompt body.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel


PROMPT_REWRITE_V22_SAYSMIN = """\
Rewrite the MESSAGE into a JSON list of standalone third-person memory \
statements about what {speaker} said. Each statement is stored verbatim \
and retrieved later by semantic search.

KEEP specific content (names, places, dates, numbers, decisions, plans, \
preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions).

ONE statement per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of the same occurrence is ONE statement \
that contains all of its particulars. Distinct events (different \
times, different occasions, different actions) each get their own \
statement. Report content, not the speech-act of conveying it -- \
``X said that ...`` / ``X told Y that ...`` wrappers are dropped \
unless the speech-act itself is the event (a promise, an apology, an \
announcement).

VOICE: frame each statement as ``{speaker} says ...`` or ``{speaker} \
says that ...``. Resolve first-person references in the raw message \
(``I``, ``me``, ``my``, ``myself``) to {speaker}'s NAME and \
appropriate third-person pronouns (he/she/they, him/her/them, \
his/her/their) WITHIN the says clause. Resolve second-person \
references (``you``, ``your``) to the addressee's NAME when known \
from context. Resolve demonstratives (``this``, ``that``, ``it``, \
``they``) to their concrete referents.

DATES. The MESSAGE was sent on {date}. The framework prepends the \
message timestamp automatically when surfacing the statement, so the \
statement text MUST NOT contain {date} in any form. PRESERVE relative \
time references from the raw message VERBATIM (``yesterday``, ``last \
week``, ``three years ago``, ``next Friday``, ``the weekend``, \
``today``, ``tonight``, ``recently``, ``now``, ``just``) -- the \
framework's chronological ordering and timestamp prefix let the \
answerer interpret them. PRESERVE explicit absolute dates from the \
message verbatim (e.g., ``March 12, 2024``, ``June 14``, ``2021``).

Forbidden date forms (each is a FAILURE): sentence-prefix \
``On YYYY-MM-DD, ...``; parenthetical ``(Date: ...)`` or \
``(Event date: ...)``; square-bracket ``[YYYY-MM-DD]``; \
``as of YYYY-MM-DD``; any echo of {date} in the statement text.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

EXAMPLES (neutral names; speech-act framing; relative phrases kept).

Example 1 -- message-time event; relative ``right now`` kept verbatim.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- the \
tonkotsu is incredible.
->
{{ "memories": ["Alice says she is right now eating tonkotsu ramen at \
the ramen place on Castro Street and finds it incredible."] }}

Example 2 -- relative reference kept verbatim; no absolute resolution.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["Bob says he adopted his two cockatiels three years \
ago, right before he moved to Portland."] }}

Example 3 -- explicit absolute date in the message kept verbatim.
MESSAGE FROM Charlie on 2026-05-02:
Dana's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["Charlie says Dana's wedding on June 14, 2025 was \
the best party he attended last year."] }}

Example 4 -- speech-act IS the event; addressee resolved by context.
PRIOR TURNS (context only, do not emit):
- Alice: Can you commit to showing up every week from now on?
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll be \
there every week from now on.
->
{{ "memories": ["Dana promises Alice she will stop missing the \
Thursday mandolin practice and will attend every week from now on."] }}

Example 5 -- bare filler; emit an empty list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Output: a JSON object {{ "memories": [...] }}. The list is empty when \
the message contains no specific content.

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
    """Says-framed minimalist rewrite segmenter.

    Each emitted segment carries ``ProducerContext(producer=speaker)``
    (single-channel embed, same as v22-fp-min). The block text is the
    third-person ``{speaker} says ...`` rewrite with relative time
    references preserved verbatim from the raw message.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_SAYSMIN,
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
                                    context=ProducerContext(producer=speaker),
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
