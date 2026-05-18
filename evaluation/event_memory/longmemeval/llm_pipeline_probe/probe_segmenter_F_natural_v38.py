"""LLM segmenter, Mode F-natural v38 -- operational topic-shift test.

v33 (shipped) over-fragments LoCoMo conversational data (4/15 on
F4). v36/v37 with abstract clarifications regressed below v33
(3/15). v35 with explicit bias + example moved the needle (7/15)
but used the two tools the user has constrained out.

v38 = v33 + a rule 6 that gives the model a procedural test for
topic boundaries instead of a tendency or example. The test:

  At each potential boundary, ask -- does the next sentence
  introduce a new entity, event, or decision that the previous
  sentence was not about? Yes => segment ends. No => segment
  continues.

Plus enumerated discourse markers ("by the way", "anyway",
"speaking of", "moving on") as objective shift signals. No outcome
prescription, no inline example. Same enumeration shape as rule 2's
"names, places, dates, numbers" enumeration.

All other rules are identical to v33.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V38 = """\
Compress this passage into the parts a human would still want during \
memory reconstruction, broken into a list of standalone memory \
segments. These segments are stored verbatim and shown back when the \
memory is retrieved.

Rules:
1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording -- \
"fabulous" stays "fabulous"; preserve whitespace, newlines, and \
special characters within a segment exactly. The only edits allowed \
are starting and ending the quote at sentence or clause boundaries.
2. KEEP what is specific to this passage; DROP what is interchangeable \
across similar passages. Specific content differentiates this passage \
from any other -- names, places, dates, numbers, identifiers, \
decisions, plans, opinions, preferences, relationships, emotional \
states tied to events, constraints, distinctive phrasing. Multi-line \
non-prose blocks (code, tables, ASCII art, diagrams, encoded text) \
are also specific -- keep each as one segment; dropping such a block \
as decoration is a FAILURE. Interchangeable content has none of \
these specifics -- it is pure framing that could open or close any \
passage regardless of subject (e.g., bare greetings, sign-offs).
3. SHORT RESPONSES that are only meaningful with the prior message \
("yes", "no", "ok", "got it", "sounds good", "acknowledged") are \
KEPT -- they carry the answer or reaction the reconstructed memory \
needs to show.
4. An utterance that begins with a greeting or softener but carries \
substantive content is KEPT -- drop only the leading greeting, not \
the content.
5. PRESERVE original order -- segments appear in the same order as \
their source quotes.
6. SEGMENT BY TOPIC SHIFT. A topic is what the passage is about: \
a subject, event, person, decision, or claim. At each potential \
boundary between consecutive sentences, apply this test -- does the \
next sentence introduce a new entity, event, or decision that the \
previous sentence was not about? If YES, the topic has shifted and \
the segment ends. If NO, the same topic continues and the segment \
extends across the boundary. Objective signals of a shift include \
discourse markers ("by the way", "anyway", "speaking of", "moving \
on", "oh"), a question-to-new-question transition, and a change in \
tense scope (past event -> future plan -> present state). Coherence \
trumps balance: a topic developed across many sentences is one \
segment; several distinct topics in a row are several segments, one \
per topic. Do not artificially split a coherent unit, and do not \
artificially merge unrelated ones.
7. STANDALONE -- each segment reads on its own. If a quote depends on \
a referent introduced earlier ("the trip" for "the Tokyo trip in \
May"), widen the quote to start where the referent is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


def _splitter_for_windows(window_chars: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=window_chars,
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


async def segment(client, model, text, reasoning="low", window_chars=WINDOW_CHARS):
    if len(text) <= window_chars:
        prompt = PROMPT_F_NATURAL_V38.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V38.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
