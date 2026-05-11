"""LLM segmenter, Mode F-natural v8 (iteration artifact).

Change: F3 reframe: rule 3 enumeration → principle (concrete particulars)

This file preserves the v8 prompt body as drafted in this session, per the
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


PROMPT_F_NATURAL_V8 = """\
Compress this passage into the parts a human would still remember weeks later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from the passage. Do NOT change wording: "fabulous" stays "fabulous", "gobsmacked" stays "gobsmacked". Connotation matters — never swap a word for a synonym, never paraphrase, never abstract. The only edits allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP content whose role is conversational framing rather than substance. Common forms: standalone greetings ("Hi!", "Hi team"), sign-offs ("Thanks, Sarah", "Best regards"), polite restatements of what the other party just asked, envelope phrases that announce or gesture-toward content without conveying it ("As discussed,", "please find attached", "Let me know if any questions", "What a great question!", "I hope this helps!"), chat reactions ("omg yes!!", "lol", "fwiw"), AND reactive filler ("I love that!", "great point", "I love your reasoning", "sounds great!") that echoes the prior turn without adding new specifics. A substantive utterance is KEPT even when it begins with a greeting or softener word ("Hey, what's the deal with X?" carries the question — drop only the leading greeting, not the content). Brief love/hate exclamations are reactive filler when the passage offers no elaboration on what they react to — no examples, behavior, prices, dates, or other specifics tied to it. KEEP love/hate statements that come with elaboration ("I love jazz" followed by examples; "I hated the dentist visit. They charged $400.") or that the speaker frames as a position ("I think Sleep is overrated").
  3. KEEP concrete particulars — anything specific to this passage that a future reader would want to recall. Names, places, dates, numbers, identifiers, decisions, plans, preferences, relationships, emotional states tied to events, constraints, and distinctive phrasing all qualify. Drop generic abstractions or stock phrases that would fit many situations.
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
        prompt = PROMPT_F_NATURAL_V8.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V8.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
