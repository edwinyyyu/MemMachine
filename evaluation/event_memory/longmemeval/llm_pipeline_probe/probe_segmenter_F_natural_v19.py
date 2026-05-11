"""LLM segmenter, Mode F-natural v19 — v9 + explicit "keep short answers".

v17/v18 reframed rule 2 too aggressively for the segment-as-reconstruction
use case. v18 kept affirmatives but lost reactive filler ("I love your
reasoning") and chat reactions ("omg yes!!") which were correctly
dropped by v9.

v19 is the minimal change: v9's rule 2 verbatim (preserves filler/
reactions handling), plus a single explicit sentence that short
affirmatives / commitments / content answers are KEPT — they're
answers, not acknowledgments. Reconstruction needs them, even though
they can only be reached via time-expansion from the question.

Distinguishing line:
  KEEP: "Yes.", "No.", "Sure thing.", "OK.", "Got it.", "Sounds good.",
        "Will do.", "Pizza.", "Tokyo.", "I'm vegan." (answers /
        commitments / content)
  DROP: "I love your reasoning", "great point", "omg yes!!", "Thanks!"
        (reactive emotional approval / pure social plumbing)
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V19 = """\
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
  2. DROP content whose role is conversational framing rather than \
substance: standalone greetings ("Hi Alice!", "Hi team"), sign-offs \
("Thanks, Bob", "Best regards"), polite restatements of what the other \
party just asked, envelope phrases that gesture at content without \
conveying it ("As discussed,", "please find attached", "Let me know if \
any questions", "What a great question!", "I hope this helps!"), chat \
reactions ("omg yes!!", "lol", "fwiw"), reactive filler ("I love \
that!", "great point", "I love your reasoning", "sounds great!") that \
echoes the prior turn without adding new specifics. \
\
Short affirmatives, agreements, commitments, or content answers are \
KEPT — they're answers, not pure acknowledgments: "Yes.", "No.", \
"Sure thing.", "OK.", "Sounds good.", "Got it.", "Will do.", \
"Pizza.", "Tokyo.", "Vanilla, definitely.", "I'm vegan." A short \
reply that answers an implicit question or commits to an action is \
substance, not scaffolding. \
\
The test for any utterance: does it contribute information someone \
would want during reconstruction — a fact, name, claim, decision, \
plan, observation, opinion with backing, OR a short answer / \
commitment to whatever was just said? If yes, KEEP. If it only \
expresses pure approval/disapproval of what someone else said without \
adding new substance or answering a question ("great point", "I love \
your reasoning"), DROP. An utterance that begins with a greeting or \
softener but carries substantive content is KEPT ("Hey, what's the \
deal with X?" contains the question — drop only the leading greeting, \
not the content).
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
        prompt = PROMPT_F_NATURAL_V19.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V19.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
