"""Question-shape derivative deriver v1.

Adds a second retrieval anchor per segment: a literal QUESTION-SHAPED
paraphrase that mirrors how a future user is likely to phrase the
query (reverse-HyDE / "Hypothetical Query"). The original whole-text
derivative is still emitted so specifics-anchored queries continue to
match the segment.

Hypothesis: why question-shape > noun-phrase generic
-----------------------------------------------------

The v1 generic-derivative prompt produces noun-phrase paraphrases
("Alice's cycling team affiliation", "languages Bob knows"). These
work as long as the user query embeds near the same noun-phrase
region of the embedding space. But real benchmark queries are almost
always full INTERROGATIVE sentences:

  "Which cycling team does Alice support?"
  "What languages does Bob know besides French?"
  "What are the names of Charlie's puppies?"

Question-shape strings sit in a measurably different region of
text-embedding-3-small space than noun-phrase summaries. Embeddings
encode syntactic register: declarative noun phrases cluster with
catalog/title text; WH-fronted interrogatives cluster with other
questions. By emitting a hypothetical query rather than a category
phrase, the derivative lives in the SAME register as the test-time
query, which should narrow the cosine gap.

Secondary benefits:

- WH-words ("what", "which", "who", "where", "when", "how many")
  carry the answer-axis explicitly. "Which cycling team does Alice
  support?" embeds the team-axis more strongly than "Alice's cycling
  team affiliation" because the WH-word is grammatically bound to
  the category noun.

- Real LongMemEval queries (and most enum_which_what benchmarks) are
  written by humans as full questions. A noun-phrase derivative is
  always an approximate match; a question-shape derivative is closer
  to an EXACT register match for at least some queries.

- The question-shape register is harder for the LLM to accidentally
  pollute with the answer value: a well-formed WH-question naturally
  ends with a question mark before the answer would appear, whereas
  a noun phrase can drift into answer-disclosing modifiers ("Alice's
  cycling team, the Wolves").

Anticipated failure modes
--------------------------

1. ANSWER LEAKAGE. The LLM appends the specific answer to the
   question ("Which cycling team does Alice support, the Wolves?").
   This kills the derivative's purpose: the embedding now sits in the
   specifics-anchored region with the original segment. Mitigation:
   explicit rule "the question must NOT contain the answer", and the
   rule reframes the task as "write the question BEFORE you knew the
   answer". Examples never contain answers.

2. WRONG AXIS. The fact bundles multiple axes (Alice talked about
   cycling teams AND her training schedule AND her coach), and the
   LLM picks an axis that doesn't match the gold query. Mitigation:
   we can't predict which axis the query will ask about, so the rule
   asks the LLM to pick the axis that is "most specific to this
   passage" -- i.e., the answer-content that's most distinctive. The
   query in practice tends to ask about the most distinctive axis.
   Genuine multi-axis segments are an unsolved residual failure mode
   for a single-derivative-per-segment design; future v2 can emit
   2-3 questions.

3. OVER-VAGUE QUESTIONS. The LLM produces "What did Alice talk
   about?" or "What did Bob say?". These embed nowhere useful.
   Mitigation: rule requires the question to name the SUBJECT and
   the CATEGORY NOUN. "What did Alice talk about" violates the
   category-noun rule.

4. WRONG INTERROGATIVE. The LLM uses "Did Alice play mandolin?"
   (yes/no) when the gold query is "What instrument does Alice
   play?". Yes/no questions don't expose the answer axis.
   Mitigation: rule requires WH-questions only (what / which / who /
   where / when / how many / how much / how long). Yes/no is banned.

5. STRUCTURAL MISMATCH FOR NON-FACT SEGMENTS. Plans, opinions, and
   first-person feelings ("Alice is excited about the trip") don't
   have a clean WH-question form. Mitigation: rule offers verb
   templates per segment type (events: "What did X do?"; opinions:
   "What does X think about Y?"; plans: "What does X plan to do
   about Y?"; attributes: "What is X's Y?"; lists: "What are X's
   Y?"). If no template fits, emit "" rather than force a bad
   question.

6. PROPER-NOUN ECHO IN THE OBJECT. The question mentions a proper
   noun that effectively names the answer ("Which puppy is named
   Mochi?" -- the answer IS Mochi). Mitigation: rule says proper
   nouns may appear ONLY as the SUBJECT (the entity who owns the
   fact). Proper nouns that are themselves answer-content go in the
   dropped set.

7. ENTITY NAME OMITTED. The LLM writes "What basketball team does
   he support?" without naming the subject. The pronoun renders the
   derivative useless for cross-segment retrieval. Mitigation:
   explicit rule "name the subject entity in the question".

8. OVER-SPECIFIC PROPER-NOUN CONTEXT. The LLM keeps a date or
   location that's incidental to the question axis ("What cycling
   team does Alice support on 2024-03-11?"). Mitigation: rule
   explicitly drops dates, times, and incidental locations from the
   question text -- they belong with the segment timestamp, not the
   query embedding.

Why neutral examples + the rule set should generalize
------------------------------------------------------

Examples use Alice / Bob / Charlie / Dana / Erin and unrelated
domains (puppy / mandolin / book club / climbing / pottery / pen-pal
in Lyon / chickpea curry). None overlap with LongMemEval or LoCoMo
evaluation entities or topics. The rule set is principle-based
(category noun, subject name, WH-word, drop answers, drop dates),
not pattern-based, so the LLM should apply the same shape to unseen
domains. Cross-segment-type coverage (event / opinion / plan /
attribute / list / list-with-count) is demonstrated explicitly with
one example per type, so the LLM has a template for each segment
shape it will see at ingest.

Output
------

Two derivatives per segment:

  1. WHOLE -- block.text = _format_with_context(segment.context,
     segment.block.text). Identical to WholeTextDeriver and to
     GenericDeriver v1 -- preserves specifics-anchored retrieval.

  2. QSHAPE -- block.text = LLM-generated hypothetical WH-question
     about the segment's most distinctive answer axis. If the LLM
     returns empty (segment has no clean WH-question form), no
     second derivative is emitted.

Both derivatives share segment_uuid. The vector store ingests them
independently; retrieval dedupes by segment_uuid post-pool, so the
top-K slot count stays unchanged.

Cost
----

One small LLM call per segment at ingest. Parallelizable across the
corpus. With gpt-5-nano @ low, ingest cost roughly doubles over the
WholeTextDeriver baseline; vector storage doubles per segment.
Query time unchanged.

Caller guidance
---------------

Use this deriver paired with the v22 RewriteSegmenter and a
SurroundingEventsContext with before-window=8 (recommended segmenter
configuration). The question-shape derivative is generated from the
segment TEXT only -- it doesn't see neighbors, since the segment is
already self-contained after segmentation.
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


PROMPT_QSHAPE_V1 = """\
Rewrite the FACT as a single hypothetical QUESTION that a future \
user might ask, where the FACT contains the answer. The question is \
a retrieval anchor: it lives next to real user queries in embedding \
space so that the FACT's segment can be retrieved when that query \
arrives.

Imagine you are the user BEFORE you knew the fact. You know the \
subject entity and the category, but you do NOT know the specific \
answer value. Write the question you would ask.

Rules:
- The question must be a WH-question. Allowed openers: what, which, \
who, where, when, how many, how much, how long, how often. Yes/no \
questions are banned.
- NAME the subject entity (the owner of the fact) by their proper \
name in the question. Do not use pronouns ("he", "she", "they") and \
do not omit the subject.
- NAME the category noun the question is about (team, instrument, \
recipe, language, city, plan, opinion, achievement, name, count, \
duration, location, ailment, etc.). The category noun is what makes \
the question targetable.
- The question MUST NOT contain the answer value. Specifically drop: \
proper-noun titles that are the answer (band names, team names, \
book titles, city names that ARE the answer), exact lists of items, \
exact numbers that are the answer, exact dates, exact durations \
that are the answer. If you find yourself writing the answer into \
the question, you are doing it wrong -- rewrite without it.
- Drop incidental context that isn't part of the question axis: \
dates of the conversation, timestamps, secondary participants, the \
fact that "X told Y" (just ask about X's something).
- Pick the SINGLE most distinctive answer axis in the fact. If the \
fact bundles multiple axes, pick the one whose answer value is most \
specific (a rare proper noun beats a generic verb). Do not try to \
cover all axes in one question.
- Proper nouns may appear ONLY as the subject entity. Proper nouns \
that name the answer go in the dropped set.
- Under 14 words. One sentence ending in a question mark.
- If the fact has no targetable answer axis (pure filler, greeting, \
no entity, no category noun), emit "".

Segment-type templates (use as starting points, adapt the WH-word):
- EVENT (X did/said/visited Z): "What did X do at/about <category>?" \
or "Where did X go for <category>?" or "When did X <verb>?"
- OPINION (X thinks/likes/recommends Z): "What does X think about \
<category>?" or "Which <category> does X recommend?"
- PLAN (X plans/hopes/intends Z): "What does X plan to do about \
<category>?"
- ATTRIBUTE (X is/owns/works as Z): "What is X's <category>?" or \
"Which <category> does X have?"
- LIST (X's Z values are A, B, C): "What are the names of X's \
<category>?" or "What <category> does X have?"
- COUNT (X has N of Z): "How many <category> does X have?"
- DURATION (X has done Z for N time): "How long has X been doing \
<category>?"

Examples:

FACT: "Alice told Bob that the band she joined last spring is The \
Lantern Choir on 2024-03-11."
QUESTION: "Which band did Alice join last spring?"

FACT: "Bob said his three puppies are named Mochi, Pebble, and \
Clover on 2025-01-09."
QUESTION: "What are the names of Bob's puppies?"

FACT: "Charlie recommended the small bookshop in Lyon called Le \
Coin Tranquille on 2023-07-22."
QUESTION: "Which bookshop did Charlie recommend in Lyon?"

FACT: "Dana plans to take a six-week pottery course at the \
community center starting in October."
QUESTION: "What course does Dana plan to take this fall?"

FACT: "Erin said the mandolin was easier to pick up than she \
expected and she practices twice a week."
QUESTION: "How often does Erin practice mandolin?"

FACT: "Alice has a pen-pal in Lyon named Margaux who writes her \
every other Sunday."
QUESTION: "Who is Alice's pen-pal in Lyon?"

FACT: "Bob mentioned that his book club is reading translated \
Korean fiction this quarter."
QUESTION: "What genre is Bob's book club reading this quarter?"

FACT: "Charlie's chickpea curry recipe uses coconut milk and \
toasted cumin seeds."
QUESTION: "What ingredients are in Charlie's chickpea curry?"

FACT: "Dana climbed her first outdoor 5.10 route at Smith Rock on \
2025-05-04."
QUESTION: "What climbing grade did Dana send outdoors?"

FACT: "Erin said her doctor mentioned her iron levels were low at \
her last checkup."
QUESTION: "What did Erin's doctor flag at her last checkup?"

FACT: "Alice has been studying Portuguese for about eleven months \
since switching from Italian."
QUESTION: "How long has Alice been studying Portuguese?"

FACT: "Bob is excited about the upcoming trip and asked Charlie to \
bring snacks."
QUESTION: ""

Output: a JSON object {{ "question": "..." }}.

FACT: {segment_text}"""


class _QShapeResponse(BaseModel):
    question: str


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
    """Emits TWO derivatives per segment: whole + question-shape paraphrase.

    The class name matches the v1 generic deriver for drop-in
    swappability in the pipeline configuration -- the only difference
    is the prompt template and the response field name.

    Args:
        language_model: LanguageModel used to generate the
            question-shape derivative. Configure model +
            reasoning_effort at construction.
        prompt_template: A ``.format(segment_text=...)`` template.
            Defaults to PROMPT_QSHAPE_V1.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_QSHAPE_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    async def _generate_question(self, segment_text: str) -> str:
        prompt = self._prompt_template.format(segment_text=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_QShapeResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return ""
        return (response.question or "").strip()

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

        # Generate the question-shape derivative; skip if empty.
        question = await self._generate_question(text)
        if question:
            derivatives.append(
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=NullContext(),
                    block=TextBlock(text=question),
                    properties=segment.properties,
                )
            )

        return derivatives
