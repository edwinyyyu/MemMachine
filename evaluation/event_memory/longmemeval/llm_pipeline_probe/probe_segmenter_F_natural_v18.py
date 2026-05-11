"""LLM segmenter, Mode F-natural v18 — drop only social plumbing.

v17 corrected the segment-vs-derivative role split (segments are for
reconstruction; short answers must survive). v18 sharpens the line:

  v17 still treated "Sure thing." / "Got it." as pure acknowledgments,
  but they're actually affirmative commitments ("Yes, I'll do that" /
  "I understand and will act on it"). Anything that could plausibly be
  an affirmative, agreement, commitment, or short content answer should
  be KEPT — the reconstructed memory needs the answer.

  v18 drops ONLY pure social plumbing: greetings, thanks, sign-offs,
  polite restatements, envelope phrases. Keeps every short affirmative
  ("Yes", "No", "Sure thing", "OK", "Got it", "Sounds good", "Will do").
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V18 = """\
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
  2. DROP ONLY pure social plumbing — content whose only signal is "I \
received your message" or "I am being polite": standalone greetings \
("Hi Alice!", "Hi team"), thanks ("Thanks!", "Thanks, Bob"), sign-offs \
("Best regards"), polite restatements of what the other party just \
asked, envelope phrases that gesture at content without conveying it \
("As discussed,", "please find attached", "Let me know if any \
questions", "What a great question!", "I hope this helps!"), and \
reactive filler that just echoes the prior turn with approval and no \
new specifics ("I love that!", "great point", "I love your reasoning"). \
KEEP everything else, including short affirmatives, agreements, or \
commitments ("Yes.", "No.", "Sure thing.", "Will do.", "OK.", "Sounds \
good.", "Got it.", "Mhm."), single-word or single-phrase content \
answers ("Pizza.", "Tokyo.", "Vanilla, definitely.", "I'm vegan."), \
short opinions ("I think Sleep is overrated."), and any sentence with \
specifics. An utterance that begins with a greeting or softener but \
carries substantive content is KEPT ("Hey, what's the deal with X?" — \
drop only the leading greeting).
  3. KEEP concrete particulars — anything specific to this passage \
that a future reader would want to recall. Names, places, dates, \
numbers, identifiers, decisions, plans, preferences, relationships, \
emotional states tied to events, constraints, and distinctive phrasing \
all qualify. Drop generic abstractions or stock phrases that would fit \
many situations.
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
        prompt = PROMPT_F_NATURAL_V18.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V18.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
