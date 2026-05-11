"""LLM segmenter, Mode F-natural v12 — v9 compacted, principle-first.

v9 (2555c) works cross-model but has redundancy:
  - Example lists have low-information members ("Hi!" + "Hi team", "Thanks,
    Sarah" + "Best regards", "lol" + "fwiw", etc.)
  - Rule 2 has both a category-list AND a "test" — same idea twice.

v12 keeps the rule-2/3 split that v9 needs (rule 3 = KEEP counter-balance
to rule 2 DROP) but:
  - Leads rule 2 with the principle, lists are illustration only.
  - Keeps the examples that ablation showed load-bearing: "I love your
    reasoning" (pure-agent-support), "please find attached" (email),
    "omg yes!!" (group chat).
  - Drops redundant examples (Hi!/Hi team duplication, etc.)
  - Tighter rule 3, same content.

Target: ~2k chars, no cross-model regression vs v9.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V12 = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording — \
"fabulous" stays "fabulous". The only edits allowed are starting and \
ending the quote at sentence or clause boundaries.
  2. DROP content whose only role is conversational framing — content \
that echoes a prior turn, expresses pure approval/disapproval, or moves \
the conversation along without adding specifics. Common forms: \
greetings ("Hi team"), sign-offs ("Thanks, Sarah"), polite restatements \
of what someone just asked, envelope phrases that gesture at content \
without conveying it ("please find attached"), chat reactions ("omg \
yes!!"), reactive filler ("I love that!", "I love your reasoning"). An \
utterance that begins with a greeting or softener but carries \
substantive content is KEPT — drop only the leading greeting, not the \
content.
  3. KEEP concrete particulars specific to this passage: names, places, \
dates, numbers, decisions, plans, opinions, relationships, distinctive \
phrasing. Drop generic abstractions or stock phrases that would fit \
many situations.
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
        prompt = PROMPT_F_NATURAL_V12.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V12.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
