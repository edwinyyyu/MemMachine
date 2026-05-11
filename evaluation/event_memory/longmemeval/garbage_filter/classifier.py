"""LLM-based cue-worthiness classifier.

Decides whether a single message TEXT (no timestamp, no role label) is
worth embedding into a long-term retrieval store, OR is conversational
mechanics that would be useless out of context.

The contract is asymmetric:
  - Never reject something a human would remember (false-reject is costly).
  - It is fine to keep something a human would discard (false-keep is cheap).

Multiple prompt variants are kept here so we can A/B them on the test set.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import openai

# ----- prompt variants ---------------------------------------------------


# v1: terse principle, no examples.
PROMPT_V1 = """\
You are gating which pieces of text get embedded into a long-term retrieval \
index. Decide if the TEXT below would still be useful, on its own, as a \
retrieval cue — judge it standalone, without any surrounding context.

Reject ONLY if the text is purely conversational or scaffolding plumbing \
with no content of its own: bare acknowledgements, greetings, thanks, \
generic "continue"/"more"/"do another" requests, lone formatting fragments \
(headers, bullets, code fences), or single characters/numbers without \
referent.

Keep everything else. When in doubt, keep. \
Even one named entity, place, person, brand, concept, preference, \
specific question, or unique phrasing is enough to keep.

Reply with exactly one token: KEEP or REJECT.

TEXT:
{text}"""


# v2: principle + asymmetric reasoning + the "would I remember this" framing.
PROMPT_V2 = """\
You are gating which messages get embedded into a long-term memory index for a \
chat assistant. The question is simple:

  If you saw ONLY this text, with no surrounding turns, would it still anchor \
  a memory you'd want to retrieve later?

A message has anchoring power if it carries any of:
  - a specific named entity (person, place, brand, product, work, school,
    dish, concept) — even one word like "Telkomsel" or "ahinsa" qualifies
  - a concrete fact, preference, plan, or experience the user owns
  - a specific question whose subject identifies a topic
  - distinctive phrasing, story content, or roleplay action

A message lacks anchoring power if it is:
  - greeting, sign-off, thanks, apology, generic affect ("Perfect", "got it")
  - acknowledgement ("Acknowledged.", "Yes.", "Disagree.", "Memorized.")
  - generic turn-mechanics ("continue", "more", "next", "regenerate",
    "give me another", "show the rest", "any other ideas?")
  - a generic instruction with no topic ("make it longer", "in a table",
    "simplify", "expand on this")
  - lone formatting (a header like "**Resources:**", "---", "```",
    "1.", "VI. B.", broken table fragments, zero-width characters)
  - a single letter, digit, or placeholder ("C", "7", "test", "spq")

Asymmetric rule: false rejects are EXPENSIVE; false keeps are CHEAP. \
If you are unsure, output KEEP.

Reply with exactly one token: KEEP or REJECT.

TEXT:
{text}"""


# v3: same principle but with a tiny calibration set (diverse, on-purpose).
PROMPT_V3 = """\
You decide whether a single message should be embedded into a long-term \
retrieval index. The message will appear later WITH NO SURROUNDING CONTEXT \
when matched against a search query, so judge it standalone.

Principle: keep the message iff it carries something a human would still \
recall as a clue — a named entity, concrete fact, preference, plan, \
experience, distinctive question, or unique phrasing. Reject only when \
the text is pure conversational mechanics (greetings, acks, thanks, \
generic continue/more/next, generic instructions like "make it longer", \
lone formatting like a section header or bullet, or a single character).

When in doubt, KEEP. False rejects lose memories permanently; false keeps \
just add a little noise.

A few diverse calibrations (different domains on purpose):
  "I drive a Tesla." -> KEEP (concrete fact)
  "Coca-Cola Company" -> KEEP (named entity)
  "ahinsa" -> KEEP (specific concept)
  "Who did Pocahontas marry?" -> KEEP (specific question)
  "I'll play as half elf" -> KEEP (game/roleplay choice)
  "Acknowledged." -> REJECT (pure ack)
  "continue" -> REJECT (turn mechanics)
  "**Tips:**" -> REJECT (lone formatting header)
  "Make it longer" -> REJECT (generic instruction, no topic)
  "Perfect" -> REJECT (affect)
  "C" -> REJECT (single character, no referent)

Reply with exactly one token: KEEP or REJECT.

TEXT:
{text}"""


# v5: principle-only, framed in terms of arbitrary text rather than messages.
PROMPT_V5 = """\
You are gating which pieces of text get embedded into a long-term retrieval \
index. The text below will later be matched against search queries with NO \
surrounding context, so judge it standalone.

Keep the text if a human reading it on its own could still tell what it is \
about: a named entity, a concrete fact, a preference, a plan, a specific \
question, or distinctive phrasing.

Reject the text if it carries no content of its own — it is purely \
conversational plumbing or formatting scaffolding. That includes \
acknowledgement, greeting, thanks, generic affect, a request to keep going \
/ repeat / reshape / quantify the previous output, a stray section header \
or markup fragment, or a single token with no referent. This still \
applies when the text dresses up a generic ask with a filler noun \
(content, ideas, examples, options, things) — if removing the filler \
leaves only generic mechanics, it is plumbing.

Asymmetric rule: false rejects are EXPENSIVE; false keeps are CHEAP. \
When in doubt, KEEP.

Reply with exactly one token: KEEP or REJECT.

TEXT:
{text}"""


# v6: principle-only, "any signal anywhere keeps it" rule. Asymmetric goal
# means a chunk that is mostly filler but contains even one concrete fact /
# entity / plan should still be kept — losing the fact is worse than
# keeping the surrounding noise.
PROMPT_V6 = """\
You are gating which pieces of text get embedded into a long-term retrieval \
index. Decide if the TEXT below would still be useful, on its own, as a \
retrieval cue — judge it standalone, without any surrounding context.

Keep the text if any portion of it could anchor a memory — a named entity, \
place, person, brand, work; a concrete fact, preference, plan, or \
experience; a specific question; distinctive phrasing. Anchor power is \
not diluted by surrounding filler — one piece of content anywhere in the \
text is enough to keep.

Reject ONLY if NOTHING in the text carries content of its own: every \
sentence is generic acknowledgement, greeting, sign-off, thanks, generic \
affect, a generic request to continue / repeat / reshape / quantify, a \
stray header or markup fragment, or a single token with no referent.

Asymmetric rule: false rejects are EXPENSIVE; false keeps are CHEAP. \
When in doubt, KEEP.

Reply with exactly one token: KEEP or REJECT.

TEXT:
{text}"""


# v7: directly frame the rule the way the human would.
# A human reader, given only this text and asked "would I bother
# remembering this as a retrieval cue?", says either KEEP or REJECT.
# The classifier's job:
#   - if a human would KEEP it, the classifier MUST KEEP it (no false rejects)
#   - if a human would REJECT it, the classifier MAY keep or reject (false
#     keeps are cheap; favor REJECT only when confident)
PROMPT_V7 = """\
A piece of text below will be embedded into a long-term retrieval index. \
You are deciding whether to admit it.

Imagine a human reader is shown ONLY this text — no surrounding context — \
and asked "would I bother remembering this as a cue?". They KEEP the text \
when it carries something specific they could later recall — a named \
entity, place, person, brand, work; a concrete fact, preference, plan, \
or experience; a specific question; distinctive phrasing. They REJECT \
it when nothing in the text carries content of its own: every sentence \
is generic acknowledgement, greeting, thanks, affect, a generic request \
to continue / repeat / reshape / quantify the previous output, a stray \
header or markup fragment, or a single token with no referent.

Apply the human's verdict asymmetrically:
  - If the human would KEEP the text, you MUST KEEP it. Never reject \
    something a human would have remembered.
  - If the human would REJECT the text, you may either KEEP or REJECT. \
    Favor REJECT only when you are confident there is no anchor anywhere \
    in the text; otherwise KEEP.

Reply with exactly one token: KEEP or REJECT.

TEXT:
{text}"""


# v8: leans on "identifies something particular vs generic" as the
# distinguishing principle, so distinctive single-word headings still
# pass while category-label headings fail.
PROMPT_V8 = """\
You are gating which pieces of text get embedded into a long-term retrieval \
index. Decide if the TEXT below would still be useful, on its own, as a \
retrieval cue — judge it standalone, without any surrounding context.

The test is whether the text identifies anything in particular. A piece \
of text identifies something in particular when its words pick out a \
specific instance in the world — a name, place, person, brand, work, \
identifier, or a concrete fact, preference, plan, experience, or \
question tied to a particular topic. Distinctiveness matters more than \
length: a single word that picks out a particular thing is enough to \
keep, regardless of formatting wrapping it. A long passage that picks \
out nothing in particular still earns nothing.

Reject ONLY when the entire text is interchangeable with countless other \
generic interactions and identifies nothing in particular: every part is \
acknowledgement, greeting, thanks, generic affect, a generic request to \
continue / repeat / reshape / quantify the previous output, a \
category-label or markup fragment with no specific content, or a \
single token that names no particular thing.

Asymmetric rule: keep anything a careful human would have remembered. \
Never reject something a human would keep. Borderline → KEEP.

Reply with exactly one token: KEEP or REJECT.

TEXT:
{text}"""


# v9: v1 + one clause clarifying that formatting wrapping does not change
# whether the words inside are content. A bold-and-colon header is plumbing
# only when its label is itself a generic category word; if the label
# names something specific, the words inside still count as content.
PROMPT_V9 = """\
You are gating which pieces of text get embedded into a long-term retrieval \
index. Decide if the TEXT below would still be useful, on its own, as a \
retrieval cue — judge it standalone, without any surrounding context.

Reject ONLY if the text is purely scaffolding plumbing with no content of \
its own: bare acknowledgements, greetings, thanks, generic affect, generic \
"continue"/"more"/"do another" requests, or formatting fragments where \
the visible words are themselves generic category labels (the kind any \
document might use as a section divider) rather than naming anything \
particular.

Formatting and length do not change whether the words are content. A \
short fragment with a name, place, brand, work, condition, identity, \
preference, or specific topic is still a cue, regardless of any markup \
wrapping it. A long passage whose words pick out nothing in particular \
is still plumbing.

Keep everything else. When in doubt, KEEP. Even one specific name, \
place, person, brand, concept, preference, plan, experience, specific \
question, or distinctive phrasing is enough to keep.

Reply with exactly one token: KEEP or REJECT.

TEXT:
{text}"""


PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
    "v3": PROMPT_V3,
    "v5": PROMPT_V5,
    "v6": PROMPT_V6,
    "v7": PROMPT_V7,
    "v8": PROMPT_V8,
    "v9": PROMPT_V9,
}


# ----- runner ------------------------------------------------------------


@dataclass
class ClassifyResult:
    text: str
    label: str  # "KEEP" or "REJECT"
    raw: str


def _normalize(raw: str) -> str:
    s = raw.strip().upper()
    if s.startswith("KEEP"):
        return "KEEP"
    if s.startswith("REJECT"):
        return "REJECT"
    # When in doubt, KEEP (asymmetric).
    return "KEEP"


async def classify_one_chat(
    client: openai.AsyncOpenAI,
    text: str,
    *,
    model: str,
    prompt_template: str,
    reasoning_effort: str | None,
) -> ClassifyResult:
    prompt = prompt_template.format(text=text)
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_effort and model.startswith("gpt-5"):
        kwargs["reasoning_effort"] = reasoning_effort
    resp = await client.chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "").strip()
    return ClassifyResult(text=text, label=_normalize(raw), raw=raw)


async def classify_one_responses(
    client: openai.AsyncOpenAI,
    text: str,
    *,
    model: str,
    prompt_template: str,
    reasoning_effort: str | None,
) -> ClassifyResult:
    prompt = prompt_template.format(text=text)
    kwargs: dict = {
        "model": model,
        "input": prompt,
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    resp = await client.responses.create(**kwargs)
    raw = (resp.output_text or "").strip()
    return ClassifyResult(text=text, label=_normalize(raw), raw=raw)


async def classify_many(
    texts: list[str],
    *,
    model: str = "gpt-5-mini",
    prompt: str = "v2",
    reasoning_effort: str | None = "low",
    concurrency: int = 16,
    api: str = "chat",  # "chat" or "responses"
) -> list[ClassifyResult]:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(concurrency)
    template = PROMPTS[prompt]
    one = classify_one_responses if api == "responses" else classify_one_chat

    async def go(text: str) -> ClassifyResult:
        async with sem:
            return await one(
                client,
                text,
                model=model,
                prompt_template=template,
                reasoning_effort=reasoning_effort,
            )

    try:
        return await asyncio.gather(*(go(t) for t in texts))
    finally:
        await client.close()
