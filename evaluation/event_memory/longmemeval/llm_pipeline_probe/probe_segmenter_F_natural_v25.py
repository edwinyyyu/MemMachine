"""LLM segmenter, Mode F-natural v25 — trim examples.

v24 had too many example strings in both rule 2 (drop categories) and
rule 3 (short responses). v25:
  - Rule 2: drop the literal example strings, keep just category names.
    The principle is the load-bearing thing; categories anchor it.
  - Rule 3: trim from 10 short-response examples to 6 canonical ones —
    "yes", "no", "ok", "got it", "sounds good", "acknowledged".
    Covers the pattern (affirmation / negation / agreement / receipt)
    without listing every variant.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V25 = """\
Compress this passage into the parts a human would still want during \
memory reconstruction, broken into a list of standalone memory \
segments. These segments are stored verbatim and shown back when the \
memory is retrieved.

Rules:
  1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording — \
"fabulous" stays "fabulous". The only edits allowed are starting and \
ending the quote at sentence or clause boundaries.
  2. KEEP what is specific to this passage; DROP what is \
interchangeable across similar passages. Specific content \
differentiates this passage from any other — names, places, dates, \
numbers, identifiers, decisions, plans, opinions, preferences, \
relationships, emotional states tied to events, constraints, \
distinctive phrasing. Interchangeable content is stock framing that \
appears across many passages regardless of subject — greetings, \
sign-offs, polite restatements, envelope phrases, bylines, chapter \
prefaces, license headers, and similar boilerplate.
  3. SHORT RESPONSES that are only meaningful with the prior message \
("yes", "no", "ok", "got it", "sounds good", "acknowledged") are \
KEPT — they carry the answer or reaction the reconstructed memory \
needs to show. When such a response ALSO carries content ("ok, \
leaving Tuesday at 5", "no, I changed my mind about Tuesday"), it \
remains KEPT — emit the whole utterance, not a fragment.
  4. An utterance that begins with a greeting or softener but carries \
substantive content is KEPT — drop only the leading greeting, not the \
content.
  5. PRESERVE original order — segments appear in the same order as \
their source quotes.
  6. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  7. STANDALONE — each segment reads on its own. If a quote depends on \
a referent introduced earlier ("the trip" for "the Tokyo trip in May"), \
widen the quote to start where the referent is named.

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
        prompt = PROMPT_F_NATURAL_V25.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V25.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
