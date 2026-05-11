"""LLM segmenter, Mode F-natural v13 — v9 with duplicate-shape examples
removed, keep the explicit KEEP/DROP test.

v12 (1860c) was too aggressive — removing the "test for any utterance"
sentence made the model drop real content. v13 keeps that sentence and
ONLY removes redundant duplicate-shape examples:
  - "Hi!" + "Hi team" → "Hi team" (covers both)
  - "Thanks, Sarah" + "Best regards" → "Thanks, Sarah"
  - 5 envelope examples → "please find attached" (the load-bearing one)
  - "omg yes!!" + "lol" + "fwiw" → "omg yes!!"
  - "I love that!" + "great point" + "I love your reasoning" + "sounds
    great!" → "I love that!", "I love your reasoning" (the two ablation
    showed are load-bearing on agent-support and reactive cases)

Rule 3 tightened slightly. Target: ~2.1k chars, no cross-model regression.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V13 = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording — \
"fabulous" stays "fabulous". The only edits allowed are starting and \
ending the quote at sentence or clause boundaries.
  2. DROP content whose role is conversational framing rather than \
substance: greetings ("Hi team"), sign-offs ("Thanks, Sarah"), polite \
restatements of what the other party just asked, envelope phrases that \
gesture at content without conveying it ("please find attached"), chat \
reactions ("omg yes!!"), reactive filler ("I love that!", "I love your \
reasoning"). The test for any utterance: does it contribute information \
someone would want to recall — a fact, name, claim, decision, plan, \
observation, or opinion with backing? If yes, KEEP. If it only echoes \
prior content or expresses pure approval/disapproval without new \
substance, DROP. An utterance that begins with a greeting or softener \
but carries substantive content is KEPT ("Hey, what's the deal with \
X?" — drop only the leading greeting, not the content).
  3. KEEP concrete particulars specific to this passage: names, \
places, dates, numbers, identifiers, decisions, plans, opinions, \
relationships, emotional states tied to events, constraints, \
distinctive phrasing. Drop generic abstractions that fit many \
situations.
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
        prompt = PROMPT_F_NATURAL_V13.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V13.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
