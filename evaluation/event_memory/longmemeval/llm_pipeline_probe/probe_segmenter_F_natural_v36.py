"""LLM segmenter, Mode F-natural v36 -- over-fragmentation fix.

v33 (the shipped baseline) over-fragments LoCoMo conversational
data: avg 2.8 segments per ingested message vs 1.0 for the
deterministic TextSegmenter baseline. The failure mode is the LLM
treating sentence breaks, exclamations, and clause boundaries inside
a single reaction or acknowledgment as topic shifts. Over-
fragmentation hurts retrieval because `expand_context`'s neighbor
budget gets consumed by intra-event siblings before reaching
neighboring messages.

v36 = v33 + a rule 6 clarification. The clarification defines what
counts as a topic shift -- a change in subject matter, not a
sentence or clause boundary. No outcome bias ("prefer one segment")
and no example are included; the goal is to leave segment count to
the model's coherence judgment, only correcting the
misidentification of punctuation as topic shift.

All other rules are identical to v33.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V36 = """\
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
6. SEGMENT NATURALLY -- break where topics or sub-topics shift. A \
topic shift is a change in subject matter, not a sentence break, an \
exclamation, or a clause boundary. Coherence trumps balance: a long \
passage that stays on one topic is one segment; a passage that \
covers several topics gets one segment per topic. Do not \
artificially split a coherent unit, and do not artificially merge \
unrelated ones.
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
        prompt = PROMPT_F_NATURAL_V36.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V36.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
