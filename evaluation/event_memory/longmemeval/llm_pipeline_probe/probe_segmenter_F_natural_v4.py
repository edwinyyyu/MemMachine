"""LLM segmenter, Mode F-natural v4 (iteration artifact).

Change: F1 fix v3: elaboration-as-discriminator framing

This file preserves the v4 prompt body as drafted in this session, per the
user instruction to save each iteration as its own versioned file.
v1 lives in probe_segmenter_F_natural.py; v9 (current shipped) in
probe_segmenter_F_natural_v9.py.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V4 = """\
Compress this passage into the parts a human would still remember weeks later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from the passage. Do NOT change wording: "fabulous" stays "fabulous", "gobsmacked" stays "gobsmacked". Connotation matters — never swap a word for a synonym, never paraphrase, never abstract. The only edits allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP generic scaffolding by simply not including it in any segment: "Hi!", "What a great question!", "I hope this helps!", "Let me know if you have any other questions", a polite restatement of what the other party just asked, AND reactive filler ("I love that!", "great point", "I love your reasoning", "sounds great!") that echoes the prior turn without adding new specifics. Brief love/hate exclamations are reactive filler when the passage offers no elaboration on what they react to — no examples, behavior, prices, dates, or other specifics tied to it. KEEP love/hate statements that come with elaboration ("I love jazz" followed by examples; "I hated the dentist visit. They charged $400.") or that the speaker frames as a position ("I think Sleep is overrated").
  3. KEEP every named entity, place, person, brand, work, date, price, preference, plan, decision, and specific factual claim, plus eccentric or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. Coherence trumps balance: a long passage that stays on one topic is one segment; a passage that covers several topics gets one segment per topic. Do not artificially split a coherent unit, and do not artificially merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote depends on a referent introduced earlier (e.g., uses "the trip" for "the Tokyo trip in May"), widen the quote to start where the referent is named.

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
        prompt = PROMPT_F_NATURAL_V4.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V4.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
