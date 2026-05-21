"""LLM rewriting segmenter.

Pairs a deterministic chunker with an LLM that rewrites each chunk into
a list of self-contained third-person memory statements. Each emitted
segment carries a ``RewriteContext`` whose ``text_to_embed`` combines
the rewrite with the speaker-prefixed original chunk, so retrieval
matches against either the formal narrative or the raw conversational
phrasing.

The split/rewrite split is deliberate. Earlier LLM-driven segmenters
(see ``llm_text_segmenter.py`` v33) asked the LLM to do BOTH the split
and the selection in one call. The model is a poor splitter under low
reasoning: it over-fragments conversational turns at sentence
boundaries (LoCoMo conv 0 averaged 2.8 segments per message vs the
deterministic baseline's 1.0). This segmenter delegates the split to a
RecursiveCharacterTextSplitter and asks the LLM to do what it does
well: paraphrase, compress, and resolve relative references.

Output style: third-person rewrite for the segment block text (shown
to the answerer at retrieval time); a ``RewriteContext`` carrying
``text_to_embed = "{rewrite}\\n{speaker}: {original_chunk}"`` for the
embed channel. The dual-text embed gives BOTH the answer-bearing
narrative AND the raw turn to retrieval -- queries that lexically
match the conversational surface still surface the segment.

Neighbor context: when the caller populates ``event.context`` with a
``SurroundingEventsContext``, the prior (and optionally future) turns
are exposed to the LLM as anaphor/referent-resolution context only --
never emitted as their own facts. With deeper prior-context windows
the LLM resolves "the recipe" -> "the Chicken Pot Pie recipe", "the
people" -> "her dogs", and bakes inherited specifier framing from a
preceding question into the answer segment.

Empirical results on LoCoMo (1444 c124 questions, mem0-bench judge,
gpt-5.4-nano @ low reasoning, K=7, OLD retrieval stack with BM25
fusion additive 0.5, no reranker, vector-search-limit 28,
expand-context 0):

  ============================  =======  =======  =======  =======
  Caller neighbor configuration   c1       c2       c4      c124
  ============================  =======  =======  =======  =======
  no neighbors                  81.91%   85.36%   86.21%   85.18%
  1 prior                       79.43%   86.92%   87.87%   86.01%
  2 prior                       80.85%   87.54%   88.70%   86.91%
  4 prior                       82.62%   86.60%   89.66%   87.60%
  **8 prior (recommended)**     81.56%   **90.03%**   **91.32%**   **89.13%**
  16 prior                      82.27%   87.54%   90.96%   88.50%
  32 prior                      81.91%   85.05%   90.49%   87.60%
  bidirectional (2+2)           81.56%   86.92%   88.82%   86.98%
  8 prior + 1 after             77.30%   88.16%   90.84%   87.60%
  8 prior + 2 after             83.69%   87.23%   90.73%   88.57%
  8 prior + 4 after             81.21%   86.60%   90.61%   87.88%
  8 prior + 8 after             80.14%   86.60%   88.47%   86.43%
  ============================  =======  =======  =======  =======

Prior-context depth peaks at 8 turns; deeper regresses (c2 is most
sensitive; 16-prior loses -2.49pp on c2 vs 8-prior). After-neighbors
never help on aggregate c124, so production callers should populate
only the ``before`` field of ``SurroundingEventsContext`` -- this
also enables streaming/online ingestion since the segmenter never
needs future context.

The dual-text embed combined with the deeper-prior-context anchors
yields +2.15pp on c124 over the prior bidirectional default at
essentially the same token budget (+25t/q).

The segmenter sees ONE event at a time -- this is a hard constraint
to keep the segmenter trivially parallelizable and to avoid leaking
content across events.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

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


# Frozen v22 prompt. See
# evaluation/event_memory/longmemeval/llm_pipeline_probe/
# probe_segmenter_rewrite_v22.py for the iteration trail (v20 added
# neighbor context + the KEEP/DROP enumeration; v22 reframes the event
# rule so the unit of granularity is the EVENT not the SENTENCE, with
# explicit FAILURE markers on per-sentence and per-particular splits).
PROMPT_REWRITE_V22_SAYS = """\
Rewrite the MESSAGE into a JSON list of standalone third-person \
memory statements. Each statement is stored verbatim and later \
retrieved by semantic search. A future user querying any specific \
content in the message should find at least one statement that \
contains that content.

KEEP what is specific to this message; DROP what is interchangeable \
across similar messages. Specific content includes: names of people, \
places, organizations, brands, named objects, named activities; \
dates, times, durations, quantities; identifiers, titles, quoted \
phrases, proper nouns; decisions, plans, preferences, opinions, \
relationships, roles, emotional states tied to events; described \
events (something that happened or will happen); attached-media \
descriptions. Interchangeable content has none of these -- bare \
greetings, sign-offs, acknowledgments, reactions, and questions \
whose phrasing introduces nothing from the specific list above. \
Dropping specific content is a FAILURE; emitting a statement for \
interchangeable content is a FAILURE.

EACH STATEMENT:
- Corresponds to one EVENT in the message, not to one sentence. An \
event is a single occurrence, decision, plan, observation, state, \
or preference at one point or span in time. A multi-sentence \
elaboration of the same event (subject in one sentence, reason in \
the next, outcome after that) is ONE statement that contains all of \
those sentences' particulars. Emitting one statement per sentence \
is a FAILURE; emitting one statement per concrete particular is a \
FAILURE; merging two distinct events into one statement is a \
FAILURE. Distinct events (different times, different occasions, \
different actions) each get their own statement, even when they \
share participants or topics.
- Contains every concrete particular the message gives about its \
event -- subject, action, time, place, attendees, motivation, \
outcome, attached media.
- Frames the speaker's content as ``{speaker} says ...`` (or \
``{speaker} says that ...``) in third-person. The speaker is the \
SUBJECT of the speech-act verb. First-person self-references in the \
raw message resolve to the speaker's name + appropriate third-person \
pronouns inside the ``says`` clause; second-person references \
(``you``, ``your``) resolve to the addressee's name when one is \
known; demonstrative and ambiguous pronouns resolve to their concrete \
referents. The first occurrence of any queryable entity is named; \
subsequent references within the same statement may use natural \
pronouns and possessives.
- Preserves every CONCRETE PARTICULAR from the message verbatim -- \
names, dates, numbers, identifiers, decisions, plans, preferences, \
opinions, relationships, emotional states tied to events, \
distinctive phrasing, attached-media descriptions. Generic \
abstractions or stock paraphrases for specifics are FAILURES.
- Preserves polarity, direction, and emotional tone. "Used to" \
implies no longer; "didn't get to bed until 2 AM" implies a late \
end, not a late start.
- The system AUTOMATICALLY prepends the message date {date} to each \
statement at retrieval time as ``[<long-date>] ``. DO NOT include the \
message date in the statement text. Segments are sorted \
chronologically so the answerer can interpret relative time \
references against the framework's timestamp prefix.
- PRESERVE relative time references from the raw message verbatim: \
``yesterday``, ``last week``, ``next month``, ``three years ago``, \
``the weekend``, ``tonight``, etc. Do NOT resolve them to absolute \
dates -- the framework prefix + chronological ordering carry the \
anchor, and the answerer can compute the absolute date from there.
- PRESERVE absolute dates and date-like phrases that appear \
explicitly in the message (e.g., ``March 12, 2024``, ``June 14``, \
``2021``). These are content the speaker stated and should be \
verbatim in the statement.
- Traces every detail back to words in the message. Inferred \
attributes (a title from context, a gender from a name, an age, an \
ethnicity, an unstated role) are FAILURES; explicitly stated \
attributes are preserved.
- Reports the content the message conveys, not the speech-act of \
conveying it. "X said that ...", "X told Y that ...", "X mentioned \
that ..." framing is included ONLY when the speech-act itself is \
the event (an apology, a promise, an explicit announcement).

NEIGHBORING TURNS appear before and after the message strictly to \
help resolve second-person addressees, demonstratives, anaphora, \
and unresolved relative references. Content drawn from the \
neighbors is NEVER emitted -- only content drawn from the message \
itself.

Output: a JSON object {{ "memories": [...] }}. The list is empty \
when the message contains no specific content.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    """Structured response from the rewriting language model."""

    memories: list[str]


def _format_neighbors(before: list, after: list) -> str:
    """Render the surrounding-events block fed to the LLM.

    Empty string when both lists are empty -- the prompt template
    tolerates an empty block. The block is labeled "PRIOR TURNS" /
    "LATER TURNS" so the LLM treats them as resolution context, not as
    content to emit.
    """
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
    """Deterministic split + LLM rewrite into third-person statements,
    indexed against a dual-text embed channel (rewrite + raw chunk).

    Args:
        language_model: The LanguageModel used to rewrite each chunk.
            Configure the model and reasoning effort at construction
            of the LanguageModel itself.
        prompt_template: A ``.format(speaker=..., date=..., passage=...,
            neighbors_block=...)`` template producing the full prompt.
            Defaults to the validated v22 prompt.
        chunk_size: Maximum characters per deterministic chunk before
            calling the LLM. Defaults to 1500, which keeps typical
            single-turn messages as one chunk while still chunking
            outlier-long inputs deterministically rather than relying
            on the LLM to split.
        max_attempts: Retries on retryable language-model errors.

    Caller guidance: populate ``event.context`` with a
    ``SurroundingEventsContext`` whose ``before`` field contains the
    most recent prior turns of the conversation (recommended depth: 8
    turns). Leave ``after`` empty for streaming-friendly ingestion --
    after-neighbors never help on the LoCoMo benchmark and force the
    pipeline to buffer for future context.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_SAYS,
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
        """Embed-channel text: rewrite + raw speaker-prefixed chunk.

        The dual-text embed lets retrieval match either the
        third-person narrative (good for query phrasings that mirror
        the rewrite) or the original conversational surface (good for
        query phrasings that mirror the raw turn).
        """
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
                                    context=RewriteContext(
                                        text_to_embed=embed_text
                                    ),
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
