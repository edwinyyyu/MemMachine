"""v22-says v3: says framing + absolute-date resolution + LIST FAITHFULNESS.

Hypothesis under test
---------------------
v22-says v2 reverted v1's relative-phrase preservation back to v22
baseline's absolute-date resolution and recovered most of g3, but
still loses g4 cat1 by -6.46pp (74.19 vs 80.65). The empirical shape
of the regression -- g3 up, g4 cat1 down -- is the same pattern that
v22-fp-min v1 showed before the LIST FAITHFULNESS rule was added in
v22-fp-min v2. There, sampled cat1 failures showed the terse rewrite
voice tempting the model to collapse enumerations into ``various`` /
``several`` / a partial sample, dropping list items that cat1
multi-hop questions then ask about.

The third-person ``{speaker} says ...`` wrapper inherits the same
compression failure mode: the wrapper itself adds 3-4 tokens, which
makes the model more eager to abbreviate the content inside. A list
of brands, books, or hobbies inside a ``says`` clause is exactly the
shape that gets summarized into ``X says she likes several books`` or
``X says he was offered various endorsements``.

v3 fix
------
Adopt the LIST FAITHFULNESS rule from v22-fp-min v2 verbatim (only
voice-neutral phrasing -- the rule itself talks about the
``statement`` not the ``first-person voice``, so it ports cleanly
into the 3p says framing). Everything else from v22-says v2 -- the
``{speaker} says ...`` wrapper, the absolute-date resolution clause,
the speech-act-only-when-event rule, the neighbors-as-context-only
clause, the addressee/demonstrative resolution rule, the dual-text
embed channel -- is unchanged.

Neutral names (Alice/Bob/Charlie/Dana) and neutral domains (book
club, climbing gym, mandolin, puppy training, ramen) only, so the
prompt does not telegraph the LoCoMo character set.
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


PROMPT_REWRITE_V22_SAYS_V3 = """\
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
elaboration of the same event is ONE statement that contains all of \
its particulars. Emitting one statement per sentence is a FAILURE; \
emitting one statement per concrete particular is a FAILURE; merging \
two distinct events into one statement is a FAILURE. Distinct events \
(different times, different occasions, different actions) each get \
their own statement, even when they share participants or topics.
- Contains every concrete particular the message gives about its \
event -- subject, action, time, place, attendees, motivation, \
outcome, attached media.
- Frames the speaker's content as ``{speaker} says ...`` (or \
``{speaker} says that ...``) in third-person. First-person \
self-references resolve to the speaker's name + appropriate \
third-person pronouns inside the ``says`` clause; second-person \
references (``you``, ``your``) resolve to the addressee's name when \
known; demonstratives and ambiguous pronouns resolve to their \
concrete referents. The first occurrence of any queryable entity is \
named; subsequent references within the same statement may use \
natural pronouns. Example: ``I finally finished The Buried Giant \
last night`` from Alice on 2026-04-09 becomes ``Alice says she \
finished reading The Buried Giant on 2026-04-08.``
- Preserves every concrete particular verbatim -- names, dates, \
numbers, identifiers, decisions, plans, preferences, opinions, \
relationships, emotional states, distinctive phrasing, attached-media \
descriptions. Generic abstractions or stock paraphrases are FAILURES.
- Preserves polarity, direction, and emotional tone. ``Used to`` \
implies no longer; ``didn't get to bed until 2 AM`` implies a late \
end, not a late start.

LIST FAITHFULNESS. When the raw message contains an enumeration (a \
list of three or more comparable items: names, books, places, foods, \
brands, hobbies, languages, dates, etc.), the statement MUST preserve \
EVERY listed item verbatim. Summaries that collapse the list into \
``various``, ``several``, ``a few``, ``many``, ``some``, or a partial \
sample are FAILURES. If the list itself is the queryable content, the \
statement is a single enumeration -- not multiple statements. Two \
comma-separated phrases that describe the SAME event are not an \
enumeration; only three-or-more comparable items count. Example: \
``Bob says he is picking his book club selection between Piranesi, \
Tomorrow and Tomorrow and Tomorrow, The Overstory, Sea of \
Tranquility, and Klara and the Sun.``

DATES. Resolve every relative time reference into an absolute date or \
interval anchored at {date}: past-tense and backward markers go \
backward; future-tense and forward markers go forward; bare month/day \
references resolve to the nearest occurrence consistent with the \
surrounding tense; bare durations resolve to a span ending at \
{date}; bare shortcuts (``the weekend``, ``the holidays``) resolve \
to the nearest past occurrence when no tense marker is present. The \
original relative phrase MUST NOT appear after resolution. Resolved \
dates appear inline in canonical ``on YYYY-MM-DD`` form (or ``from \
YYYY-MM-DD to YYYY-MM-DD`` for spans). One date per statement; split \
multi-date messages into multiple statements. When the resolved event \
date equals {date}, omit the inline date phrase (the framework prefix \
already carries it); when it differs from {date}, write the inline \
``on YYYY-MM-DD`` phrase explicitly. Example: ``Bob says he and \
Charlie hit the climbing gym yesterday`` from a message on 2026-04-09 \
becomes ``Bob says he and Charlie went to the climbing gym on \
2026-04-08.``

SPEECH-ACT FRAMING. Routine mentions, recounts, and opinions take \
the plain ``says`` wrapper. Other speech-act framings -- ``X \
promised that ...``, ``X apologized for ...``, ``X announced that \
...`` -- are used ONLY when that specific speech-act itself is the \
event (an apology, a promise, an explicit announcement). Example: \
``Dana says her mandolin lesson is moving to Thursdays.`` (routine); \
``Charlie apologized to Alice for missing the book club meeting.`` \
(apology IS the event).

Trace every detail back to words in the message. Inferred attributes \
(a title from context, a gender from a name, an age, an unstated \
role) are FAILURES; explicitly stated attributes are preserved.

NEIGHBORING TURNS appear before and after the message strictly to \
help resolve second-person addressees, demonstratives, anaphora, and \
unresolved relative references. Content drawn from the neighbors is \
NEVER emitted -- only content drawn from the message itself.

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
    """v22-says v3: says framing + absolute-date + LIST FAITHFULNESS.

    Identical to v22-says v2 except a verbatim copy of the LIST
    FAITHFULNESS rule from v22-fp-min v2 is added, targeting the g4
    cat1 enumeration-collapse failure mode.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_SAYS_V3,
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
