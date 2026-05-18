"""LLM segmenter, Mode F-natural v46 -- task-framing reframe.

Diagnostic from v44 (rule 6 deleted entirely): the model produces
near-identical F4 outputs to v33. Rule 6 wording is a no-op at
gpt-5.4-nano @ low; the model's sentence-level default is set by the
TASK FRAMING in the first line, not by rule 6.

v46 changes the framing without changing rule content:
  - "Compress this passage ... segments" -> "Extract from this passage
    ... memory entries"
  - "memory segment" -> "memory entry" throughout
  - "segment" as a verb -> "extract"

The word "segment" carries strong sentence-level associations from
training data. "Entry"/"contribution" implies a coarser unit. If the
default granularity is task-framing-set, this should shift it.

No outcome bias toward fewer entries, no inline example. Rule 6 is the
plain v33 wording (carrying its no-op status with us). All other rules
identical to v33.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V46 = """\
Extract from this passage the noteworthy contributions that should \
be retained as memory entries. Each entry is stored verbatim and \
shown back when the memory is retrieved.

Rules:
1. VERBATIM. Each entry is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording -- \
"fabulous" stays "fabulous"; preserve whitespace, newlines, and \
special characters within an entry exactly. The only edits allowed \
are starting and ending the quote at sentence or clause boundaries.
2. KEEP what is specific to this passage; DROP what is interchangeable \
across similar passages. Specific content differentiates this passage \
from any other -- names, places, dates, numbers, identifiers, \
decisions, plans, opinions, preferences, relationships, emotional \
states tied to events, constraints, distinctive phrasing. Multi-line \
non-prose blocks (code, tables, ASCII art, diagrams, encoded text) \
are also specific -- keep each as one entry; dropping such a block \
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
5. PRESERVE original order -- entries appear in the same order as \
their source quotes.
6. ONE ENTRY PER CONTRIBUTION. A contribution is one thing the \
passage adds to the discourse: one claim, one reaction, one \
announcement, one decision, one question, one answer, one \
description. A contribution may span several sentences when they \
together build, elaborate, restate, react to, or qualify the same \
focal subject. Two consecutive sentences are different contributions \
when they introduce different focal subjects, different communicative \
purposes, or are joined by an explicit discourse marker of departure \
("by the way", "anyway", "speaking of", "moving on").
7. STANDALONE -- each entry reads on its own. If a quote depends on \
a referent introduced earlier ("the trip" for "the Tokyo trip in \
May"), widen the quote to start where the referent is named.

Output: a JSON object {{ "segments": [...] }} and nothing else. \
(The output field name remains "segments" for schema compatibility; \
each item in the list is one memory entry.)

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
        prompt = PROMPT_F_NATURAL_V46.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V46.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
