"""LLM segmenter, Mode F-natural v17 — keep short answers, drop only
message-receipt scaffolding.

Architectural correction (vs v9): segments are kept for memory
RECONSTRUCTION, not for self-retrieval. Derivatives handle semantic
search; time-based expansion bridges from a matched segment to its
temporally adjacent neighbors. So a single-event reply like "Yes." or
"Pizza." must be KEPT — it's reachable through the question's segment
plus expansion, and the reply text is the answer the reconstructed
memory needs to show.

What changes vs v9:
  - Rule 2 reframed around the receipt-vs-answer distinction.
  - Bare "Yes." / "No." / short-content answers ("Pizza.", "Tokyo.",
    "Yes please!") are KEPT as substance.
  - Pure message-receipt acknowledgments ("OK.", "Got it.", "Sure
    thing.", "Sounds good.", "Thanks!") are DROPPED — they carry no
    meaning beyond "I received your message".
  - "Reactive filler" still applies to approval-only echoes ("I love
    that!", "great point", "I love your reasoning").

Other rules unchanged from v9.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V17 = """\
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
  2. DROP only content whose role is acknowledging message reception \
or framing the conversation, with no information of its own: \
standalone greetings ("Hi Alice!", "Hi team"), sign-offs ("Thanks, \
Bob", "Best regards"), polite restatements of what the other party \
just asked, envelope phrases that gesture at content without conveying \
it ("As discussed,", "please find attached", "Let me know if any \
questions", "What a great question!", "I hope this helps!"), pure \
acknowledgments ("OK.", "Got it.", "Sure thing.", "Sounds good."), \
chat reactions ("omg yes!!", "lol", "fwiw"), reactive filler ("I love \
that!", "great point", "I love your reasoning", "sounds great!") that \
echoes the prior turn without adding new specifics. KEEP everything \
else, including short answers that carry information — "Yes." and \
"No." (binary answers), single-word or single-phrase replies \
("Pizza.", "Tokyo.", "Vanilla, definitely.", "I'm vegan."), short \
opinions ("I think Sleep is overrated."), and any sentence with \
specifics (names, dates, numbers, decisions). An utterance that begins \
with a greeting or softener but carries substantive content is KEPT \
("Hey, what's the deal with X?" — drop only the leading greeting).
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
        prompt = PROMPT_F_NATURAL_V17.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V17.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
