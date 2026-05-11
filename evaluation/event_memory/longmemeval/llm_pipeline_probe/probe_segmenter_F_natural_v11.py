"""LLM segmenter, Mode F-natural v11 — restore DROP/KEEP split, trim examples.

v10 merged rule 2 (DROP) and rule 3 (KEEP) into a single SUBSTANCE TEST,
which made the model too conservative — dropped "I love jazz", "Sleep is
overrated", "Hey, that movie was insane!" (real opinions/content treated
as filler).

v11 restores the two-rule split that v9 had, but trims example lists in
both rules. The split keeps DROP and KEEP forces in tension, with neither
dominating. Target: cross-model robust at ~2k chars.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V11 = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording — \
"fabulous" stays "fabulous". The only edits allowed are starting and \
ending the quote at sentence or clause boundaries.
  2. DROP conversational framing — content whose only role is moving \
the conversation along: greetings, sign-offs, restatements of what \
someone just asked, envelope phrases that gesture at content without \
conveying it (e.g., "please find attached"), chat reactions, reactive \
filler ("I love that!", "great point") that echoes a prior turn \
without adding new specifics. Utterances that begin with a greeting \
but carry substance ("Hey, what's the deal with X?") are KEPT — drop \
only the leading greeting.
  3. KEEP every concrete particular — anything specific to this \
passage that a future reader would want to recall: names, places, \
dates, numbers, decisions, plans, opinions, distinctive phrasing.
  4. PRESERVE original order — segments appear in the same order as \
their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. A \
passage on one topic is one segment; a passage covering several topics \
gets one segment per topic. Do not artificially split a coherent unit, \
and do not artificially merge unrelated ones.
  6. STANDALONE — each segment reads on its own. If a quote uses a \
referent introduced earlier ("the trip" for "the Tokyo trip in May"), \
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
        prompt = PROMPT_F_NATURAL_V11.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V11.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
