"""v22-min3p-keepdate RewriteSegmenter -- min3p prompt with v22-style inline-date rule.

Hypothesis
----------

min3p K=7 vs baseline K=7 cat-by-cat (gpt-5-mini OLD stack, fb c124):
- cat1: 236 vs 230 -- min3p WINS +6 (multi-hop)
- cat2: 287 vs 289 -- min3p LOSES -2 (temporal)
- cat3:  70 vs  69 -- min3p WINS +1 (open-domain)
- cat4: 765 vs 768 -- min3p LOSES -3 (single-hop)
- c124: 1288 vs 1287 -- TIE +1 (within noise)

The cat2 regression (-2) correlates with min3p's date-drop rule (drops
inline `on YYYY-MM-DD` when the event date equals the message date,
relying on the framework's `[<timestamp>] ` prefix). Diagnosed elsewhere
(project_fb_iteration_dashboard.md, project_pareto_frontier_breakthrough.md)
that the inline absolute `on YYYY-MM-DD` is LOAD-BEARING for cat2 because
the framework header is in natural-language format
("[Friday, December 8, 2023, 7:42 PM]") that doesn't BM25-match
YYYY-MM-DD-style query phrasings.

v22-min3p-keepdate fix
----------------------

Replace min3p's drop-date rule with v22 baseline's keep-date rule:
include inline `on YYYY-MM-DD` regardless of whether it equals the
message date. The 7-15 token cost per message-time segment is worth it
if it restores cat2 retrieval signal.

Everything else from min3p unchanged (KEEP/DROP, ONE statement per
EVENT, PERSON rule with name resolution, NEIGHBORING TURNS framing).
ProducerContext + WholeTextDeriver pipeline unchanged.

Anticipated outcome
-------------------

If hypothesis holds:
- cat2 lifts by ~2 (back to baseline 289 territory or close)
- cat1 stays at +6 (anti-fragmentation structure unchanged)
- cat4 might also lift slightly (dates help "when did X" queries)
- Net: meaningful c124 win over baseline at K=7

If hypothesis fails:
- cat2 may stay flat or improve slightly without lifting the overall
- diagnoses min3p's regression as more than just date-drop
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


PROMPT_REWRITE_V22_MIN3P_KEEPDATE = """\
Rewrite the MESSAGE into a JSON list of standalone third-person memory \
statements about {speaker}. Each statement is stored verbatim and \
retrieved later by semantic search.

KEEP specific content (names, places, dates, numbers, decisions, plans, \
preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions).

ONE statement per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of the same occurrence is ONE statement \
that contains all of its particulars. Distinct events (different \
times, different occasions, different actions) each get their own \
statement. Report content, not the speech-act of conveying it -- \
``{speaker} said that ...`` / ``{speaker} told X that ...`` wrappers \
are dropped unless the speech-act itself is the event (a promise, an \
apology, an announcement).

PERSON: refer to the speaker by name ({speaker}). Resolve \
first-person self-references (``I`` / ``me`` / ``my`` / ``mine`` / \
``myself``) in the raw to {speaker}'s name on first occurrence; \
subsequent in-statement references use {speaker}'s pronouns \
(he/she/they + his/her/their + him/her/them). Everyone else also \
stays third-person -- resolve ``you`` / ``your`` to the addressee's \
NAME when known from context, and resolve demonstratives (``this``, \
``that``, ``it``, ``they``) to their concrete referents.

DATES. The MESSAGE was sent on {date}. Resolve every relative time \
reference (``yesterday``, ``last week``, ``three years ago``, \
``next Friday``, ``the weekend``, ``today``, ``tonight``, \
``recently``, ``now``, ``just``) to an absolute date anchored at \
{date}. The original relative phrase MUST NOT appear after resolution. \
The statement is ANCHORED to the date of the event it describes: \
``on YYYY-MM-DD`` woven into natural prose -- ``{date}`` for events \
that happened during the message, or the resolved date for events \
that happened on a different date. One event date per statement; \
split multi-date messages into multiple statements.

Forbidden date forms (each is a FAILURE): sentence-prefix \
``On YYYY-MM-DD, ...``; parenthetical ``(Date: ...)`` or \
``(Event date: ...)``; square-bracket ``[YYYY-MM-DD]``; \
``as of YYYY-MM-DD``.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

EXAMPLES (neutral names; the SPEAKER is referred to by name).

Example 1 -- message-time event; inline date carries the message date.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- the \
tonkotsu is incredible.
->
{{ "memories": ["Alice is eating tonkotsu ramen at the ramen place on \
Castro Street on 2026-05-18 and finds it incredible."] }}

Example 2 -- relative reference resolves to a DIFFERENT date; the \
relative phrase is removed.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["Bob adopted his two cockatiels on 2023-04-10, right \
before he moved to Portland."] }}

Example 3 -- explicit absolute date in the message.
MESSAGE FROM Charlie on 2026-05-02:
Dana's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["Charlie considers Dana's wedding on 2025-06-14 the \
best party he attended in 2025."] }}

Example 4 -- speech-act IS the event; date anchored to message date.
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll be \
there every week from now on.
->
{{ "memories": ["Dana promises on 2026-05-18 to stop missing the \
Thursday mandolin practice and to attend every week going forward."] }}

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
    """min3p + v22-style always-include-inline-date rule.

    ProducerContext + WholeTextDeriver pipeline same as min3p.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_MIN3P_KEEPDATE,
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
