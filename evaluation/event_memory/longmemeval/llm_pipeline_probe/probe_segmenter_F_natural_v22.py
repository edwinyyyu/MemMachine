"""LLM segmenter, Mode F-natural v22 — v21 with envelope category fixed.

v21 said "envelope openers" — but stock-phrase closers like "Let me
know if any questions" are envelope too. v22:
  - "envelope openers" → "envelope phrases" (covers openers + closers)
  - Add "Let me know if any questions" as a closer example
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V22 = """\
Compress this passage into the parts a human would still want during \
memory reconstruction, broken into a list of standalone memory \
segments. These segments are stored verbatim and shown back when the \
memory is retrieved — keep everything that carries content for the \
reconstructed conversation, even if it would not be findable on its \
own (search uses other indexes).

Rules:
  1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording — \
"fabulous" stays "fabulous". The only edits allowed are starting and \
ending the quote at sentence or clause boundaries.
  2. KEEP everything specific to this passage — names, places, dates, \
numbers, identifiers, decisions, plans, opinions, preferences, \
relationships, emotional states tied to events, constraints, \
distinctive phrasing. Short responses that are only meaningful with \
the prior message ("yes", "no", "ok", "sure thing", "got it", "sounds \
good", "great point", "omg yes!!", "lol", "acknowledged") are also \
KEPT — they carry the answer or reaction the reconstructed memory \
needs to show, even though they would not be findable on their own.
  3. DROP only content with zero information of its own — pure social \
plumbing such as greetings ("Hi!", "Hi team"), thanks ("Thanks!", \
"Thank you"), sign-offs ("Best regards", "Take care"), polite \
restatements of what someone just asked, and stock envelope phrases \
that gesture at no content ("As discussed,", "Let me know if any \
questions"). An utterance that begins with a greeting or softener but \
carries substantive content is KEPT — drop only the leading greeting, \
not the content.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment reads on its own. If a quote depends on \
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
        prompt = PROMPT_F_NATURAL_V22.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V22.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
