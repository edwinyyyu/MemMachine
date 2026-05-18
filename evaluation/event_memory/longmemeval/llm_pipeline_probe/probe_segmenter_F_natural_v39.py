"""LLM segmenter, Mode F-natural v39 -- point-being-made topic-shift framing.

v33 (shipped) over-fragments LoCoMo at 4/15 on F4 cases. v36/v37/v38
all hit 3/15 with definition-only / topic-as-unit / operational-test
framings. The "introduces a new entity" hard rule fails because new
entities can be sub-aspects of the current focus (Paris -> Eiffel
Tower is one topic).

v39 = v33 + a rule 6 reframed around the *point being made* -- the
communicative move the passage is currently advancing. Two reciprocal
classifiers replace the hard entity rule:

  CONTINUATION signals -- the next sentence builds, elaborates,
  restates, reacts to, intensifies, or qualifies the same point.
  Same subject of focus, same communicative purpose.

  SHIFT signals -- a different focal subject, a different
  communicative purpose (reaction -> query, statement -> proposal,
  narrative -> commentary), or an explicit discourse marker
  ("by the way", "anyway", "speaking of", "moving on", "oh").

No outcome bias, no inline input->output example. Same enumeration
shape as rule 2's "specific content" list.

All other rules are identical to v33.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V39 = """\
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
6. SEGMENT BY POINT. A topic is the single point the passage is \
currently making -- the subject, claim, reaction, decision, or \
question being advanced. Two consecutive sentences belong to the \
SAME topic when the second builds, elaborates, restates, reacts to, \
intensifies, or qualifies the first about the same focal subject; \
sub-aspects of that subject (its attributes, parts, consequences) \
are continuations, not new topics. Two consecutive sentences mark a \
TOPIC SHIFT when the passage moves to a different focal subject, a \
different communicative purpose (reaction -> query, statement -> \
proposal, narrative -> commentary, question -> answer to a different \
question), or carries an explicit discourse marker of departure \
("by the way", "anyway", "speaking of", "moving on", "oh"). \
Coherence trumps balance: a point developed across many sentences \
is one segment; several distinct points in a row are several \
segments, one per point. Do not artificially split a coherent unit, \
and do not artificially merge unrelated ones.
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
        prompt = PROMPT_F_NATURAL_V39.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V39.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
