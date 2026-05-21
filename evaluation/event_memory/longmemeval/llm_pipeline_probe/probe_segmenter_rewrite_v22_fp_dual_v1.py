"""v22-fp-dual-v1 RewriteSegmenter -- 1p body + v22 dual-text embed architecture.

Hypothesis
----------

Existing 1p variants (fp, fp-min, fp-min-v2, fp-min-v3) all use
ProducerContext, which gives:
  - BM25 input: "[<timestamp>] {speaker}: {1p_text}"   (speaker via header)
  - EMBED input: "{speaker}: {1p_text}"                (single text)

v22 baseline (3p RewriteContext) gives:
  - BM25 input: "[<timestamp>] {3p_text}"              (subject inline in 3p text)
  - EMBED input: "{3p_text}\\n{speaker}: {raw_chunk}"  (DUAL text)

The EMBED asymmetry is the diagnosed mechanism behind 1p's fb regression:
sub-fact details (a photo, an aside, a quoted phrase) in the raw_chunk
get embedded by v22 baseline through the dual-text channel; the 1p
paraphrase may drop those details from its rewrite, and the
ProducerContext single-text embed has no recovery path.

Concrete evidence (fb regression sample for "What did Nate share a photo
of when mentioning unwinding at home?", gold = bookcase with DVDs):
  - Baseline retrieved one segment containing BOTH "Nate found that
    playing video games or watching movies helps him unwind" AND
    "Nate attached a photo of...".
  - fp-min-v2 retrieved a segment with just the unwind sentence; the
    photo description got split into a different segment that didn't
    make top-7.

Hypothesis: the dual-text embed (with raw_chunk) anchors retrieval on
sub-facts the rewrite might omit. Adding dual-text embed to the 1p
paradigm should restore retrieval parity with baseline on cat2/cat4
without losing 1p's natural-prose advantage on cat3 inferential.

Architecture
------------

Context: ``RewriteContext(text_to_embed=f"{1p_text_with_self_id}\\n{speaker}: {raw_chunk}")``

block.text: ``{1p_text_with_self_id}`` (clean 1p body, NOT speaker-prefixed)

Because RewriteContext has NO speaker prefix in the rendered header
(unlike ProducerContext), BM25 only sees ``[<timestamp>] {block.text}``.
To anchor BM25 on the speaker's name, we add a SPEAKER SELF-IDENTIFICATION
rule to the prompt: the FIRST first-person pronoun in each statement is
followed by ``(<speaker name>)``. This puts the name in block.text so
BM25 can score speaker-named queries.

Display rendering: ``[<timestamp>] {1p_text_with_self_id}`` — clean 1p
text with one parenthetical name per statement. Less awkward than
"X says X went to..." framing and still QA-readable.

Generalizability: this is a fundamental architecture pattern (dual-text
embed + retrieval-side speaker anchoring); not LongMemEval-specific.

Anticipated failure modes
-------------------------

1. Self-id over-injection: model writes "(Caroline)" on every pronoun,
   inflating block.text. Mitigation: explicit rule + counter-example.
2. Embed-text dominates BM25 in the wrong direction: BM25 is over the
   block.text alone (no raw_chunk), so if 1p text is shorter than 3p
   baseline's block.text, BM25 surface area shrinks. Mitigation: the
   1p body should be roughly the same length as 3p (semantic content
   preserved via FAILURE clauses), so BM25 surface should be comparable.
3. Display readability: the "I (Caroline)" parenthetical may confuse
   the QA model. Mitigation: the QA model handles named-entity
   parentheticals fine in standard prose ("She lives in Seattle (WA).").

Everything else copied from probe_segmenter_rewrite_v22_fp.py:
- Full v22 KEEP/DROP enumeration
- PERSON RULES preserving 1p, resolving 2p/3p
- MESSAGE DATE vs EVENT-MENTIONED DATE discipline
- RELATIVE-TIME RESOLUTION with FAILURE clauses
- CANONICAL DATE FORMAT
- EACH STATEMENT FAILURE clauses (anti-fragmentation, concrete particulars,
  polarity, no-inferred-attributes, speech-act-only-when-event)
- NEIGHBORING TURNS framing
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


PROMPT_REWRITE_V22_FP_DUAL_V1 = """\
Rewrite the MESSAGE into a JSON list of standalone first-person memory \
statements from {speaker}'s point of view. Each statement is stored \
verbatim and later retrieved by semantic search. A future user querying \
any specific content in the message should find at least one statement \
that contains that content.

KEEP what is specific to this message; DROP what is interchangeable \
across similar messages. Specific content includes: names of people, \
places, organizations, brands, named objects, named activities; dates, \
times, durations, quantities; identifiers, titles, quoted phrases, \
proper nouns; decisions, plans, preferences, opinions, relationships, \
roles, emotional states tied to events; described events (something \
that happened or will happen); attached-media descriptions. \
Interchangeable content has none of these -- bare greetings, sign-offs, \
acknowledgments, reactions, and questions whose phrasing introduces \
nothing from the specific list above. Dropping specific content is a \
FAILURE; emitting a statement for interchangeable content is a FAILURE.

PERSON RULES (first-person from {speaker}'s perspective):
- Preserve {speaker}'s own ``I`` / ``me`` / ``my`` / ``mine`` / \
``myself`` from the raw message verbatim. Do NOT replace them with the \
speaker's name.
- SPEAKER SELF-IDENTIFICATION. The FIRST first-person pronoun in each \
statement is immediately followed by the speaker's name in parentheses: \
``I ({speaker})`` / ``my ({speaker}'s)`` / ``me ({speaker})`` / \
``myself ({speaker})``. Subsequent ``I``/``me``/``my``/``myself`` in \
the SAME statement stay bare. The parenthetical holds the speaker's \
name only; no titles, no roles. Adding the parenthetical to every \
pronoun rather than only the first is a FAILURE.
- Names other than {speaker} stay in third-person: addressees, \
third-party participants, places, brands, organizations, named \
objects. Resolve a second-person reference (``you``, ``your``) to the \
addressee's NAME when the addressee is known from the conversation \
context; otherwise leave the second-person form intact.
- Resolve demonstrative or ambiguous pronouns (``this``, ``that``, \
``it``, ``they``) to their concrete referents using neighboring turns \
when needed.
- First-person plural (``we``, ``our``) refers to {speaker} together \
with one or more named others; preserve the plural and name the \
others when identifiable.

MESSAGE DATE vs EVENT-MENTIONED DATE:
- The MESSAGE was sent on {date}. The retrieval system automatically \
prepends the message timestamp to each statement when surfacing it for \
question answering. DO NOT include the message date {date} in your \
statement text in any form -- not as a suffix, not as a sentence \
prefix, not parenthesized, not bracketed.
- Any date that appears in your statement text refers to an EVENT \
MENTIONED in the message, NOT the message itself.

RELATIVE-TIME RESOLUTION:
- Resolve every relative time reference (``yesterday``, ``last week``, \
``this summer``, ``three years ago``, ``next month``, ``the weekend``) \
into an ABSOLUTE date anchored at the message date {date}. Past-tense \
and backward markers go backward; future-tense and forward markers go \
forward; bare month/day references resolve to the nearest occurrence \
consistent with the surrounding tense; bare durations ending ``now`` \
resolve to a span ending at the message date; bare shortcuts \
(``the weekend``, ``the holidays``) resolve to the nearest past \
occurrence when no tense marker is present.
- If the resolved date EQUALS the message date {date} (the mentioned \
event coincides with the message-sending moment), DROP the date from \
the statement. The framework's automatic prefix carries it. Common \
triggers for "resolves to message date": ``today``, ``tonight``, \
``right now``, ``just``, ``earlier today``, no temporal marker at all \
when the verb is present-tense or simple-past describing the current \
day. In all these cases the output statement must contain NO date.
- If the resolved date DIFFERS from the message date, include it \
inline as ``on YYYY-MM-DD`` in natural prose -- for example, \
``My tournament was on 2026-03-14``.
- A bare relative phrase (``last week``, ``next month``, ``yesterday``, \
``three years ago``, ``the weekend``, ``last year``, ``today``, \
``tonight``, ``recently``) appearing in the output AFTER resolution \
is a FAILURE. After you resolve the relative phrase to an absolute \
date, REMOVE the relative phrase from the sentence -- it must not \
appear alongside the resolved date.
  FAIL:  ``I took a solo trip last year on 2022-05-04.``
  PASS:  ``I (Alice) took a solo trip on 2022-05-04.``
  FAIL:  ``My laptop crashed last week and I lost all my work.``
  PASS (with absolute date kept):  ``My (Alice's) laptop crashed on \
2022-09-07 and I lost all my work.``  (here the message was sent \
2022-09-14 and "last week" resolves to ~2022-09-07).
  PASS (when resolved date equals message date, dropping both the \
relative phrase AND the date):  ``I (Alice) checked out an art show \
with a friend and found it inspiring.``  (here the message was sent \
on the event day; the framework prefix carries it).

CANONICAL DATE FORMAT:
- Use exactly ONE inline date form: ``on YYYY-MM-DD`` woven into \
natural prose.
- Forbidden forms (each is a FAILURE): a sentence-prefix \
``On YYYY-MM-DD, ...``; a parenthetical ``(Date: YYYY-MM-DD)`` or \
``(Event date: YYYY-MM-DD)``; a square-bracket ``[YYYY-MM-DD]``; an \
``as of YYYY-MM-DD`` qualifier; any suffix attaching the message date \
to a statement.
- One event date per statement. A statement that describes events \
occurring on multiple different dates must be split into separate \
statements, each carrying its own event date.

EACH STATEMENT:
- Corresponds to one EVENT in the message, not to one sentence. An \
event is a single occurrence, decision, plan, observation, state, or \
preference at one point or span in time. A multi-sentence elaboration \
of the same event (subject in one sentence, reason in the next, \
outcome after that) is ONE statement that contains all of those \
sentences' particulars. Emitting one statement per sentence is a \
FAILURE; emitting one statement per concrete particular is a FAILURE; \
merging two distinct events into one statement is a FAILURE. Distinct \
events (different times, different occasions, different actions) each \
get their own statement, even when they share participants or topics.
- Contains every concrete particular the message gives about its \
event -- subject, action, time, place, attendees, motivation, outcome, \
attached media.
- Preserves every CONCRETE PARTICULAR from the message verbatim -- \
names, dates other than the dropped message date, numbers, \
identifiers, decisions, plans, preferences, opinions, relationships, \
emotional states tied to events, distinctive phrasing, attached-media \
descriptions. Generic abstractions or stock paraphrases for specifics \
are FAILURES.
- Preserves polarity, direction, and emotional tone. ``Used to`` \
implies no longer; ``didn't get to bed until 2 AM`` implies a late \
end, not a late start.
- Traces every detail back to words in the message. Inferred \
attributes (a title from context, a gender from a name, an age, an \
ethnicity, an unstated role) are FAILURES; explicitly stated \
attributes are preserved.
- Reports the content the message conveys, not the speech-act of \
conveying it. ``I said that ...``, ``I told Y that ...``, ``I \
mentioned that ...`` framing is included ONLY when the speech-act \
itself is the event (an apology, a promise, an explicit announcement). \
For mere conversational reporting, drop the speech-act wrapper.

NEIGHBORING TURNS appear before and after the message strictly to \
help resolve second-person addressees, demonstratives, anaphora, and \
unresolved relative references. Content drawn from the neighbors is \
NEVER emitted -- only content drawn from the message itself.

EXAMPLES (neutral names Alice/Bob/Charlie/Dana, neutral domains; the \
SPEAKER in each example owns the first-person voice).

Example 1 -- message-time event; self-id on the first ``I``.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- the \
tonkotsu is incredible.
->
{{ "memories": ["I (Alice) am eating tonkotsu ramen at the ramen place \
on Castro Street and find it incredible."] }}

Example 2 -- mentioned event with a relative reference resolved to a \
different date.
MESSAGE FROM Alice on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["I (Alice) adopted my two cockatiels on 2023-04-10, \
right before I moved to Portland."] }}

Example 3 -- mentioned event with an EXPLICIT absolute date; first \
first-person pronoun is ``My``.
MESSAGE FROM Bob on 2026-05-02:
Charlie's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["My (Bob's) most enjoyable party in 2025 was Charlie's \
wedding on 2025-06-14."] }}

Example 4 -- speech-act-as-event.
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll be \
there every week from now on.
->
{{ "memories": ["I (Dana) promise to stop missing the Thursday \
mandolin practice and to attend every week going forward."] }}

Example 5 -- conversational reporting that should DROP the speech-act \
wrapper.
MESSAGE FROM Alice on 2026-05-18:
I mentioned to Bob that I went hiking in Big Sur last weekend with my \
book club.
->
{{ "memories": ["I (Alice) went hiking in Big Sur with my book club \
on 2026-05-16."] }}

Example 6 -- bare question with no specific content. Emit an empty \
list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Example 7 -- multi-event message splits into multiple statements; \
self-id on first first-person pronoun in EACH statement.
MESSAGE FROM Charlie on 2026-05-18:
I climbed Half Dome two months ago and I'm flying to Tokyo next Friday \
for a conference on AI alignment.
->
{{ "memories": ["I (Charlie) climbed Half Dome on 2026-03-18.", \
"I (Charlie) am flying to Tokyo on 2026-05-22 for a conference on AI \
alignment."] }}

Example 8 -- second-person addressee resolution. Neighbor turn names \
the addressee.
PRIOR TURNS (context only, do not emit):
- Bob: Hey Alice, what are you up to this weekend?

MESSAGE FROM Alice on 2026-05-18:
Just baking your favorite cake for our potluck tomorrow.
->
{{ "memories": ["I (Alice) am baking Bob's favorite cake for our \
potluck on 2026-05-19."] }}

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
    """First-person rewrite segmenter with dual-text embed (RewriteContext).

    Each emitted segment carries
    ``RewriteContext(text_to_embed=f"{1p_text}\\n{speaker}: {raw_chunk}")``.
    The block.text is the bare 1p_text (with speaker self-id parenthetical),
    so BM25 sees ``[<timestamp>] {1p_text}`` and the EMBEDDING sees the
    dual-text concatenation. Mirrors v22 baseline architecture but in 1p.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_FP_DUAL_V1,
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
