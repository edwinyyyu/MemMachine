"""LLM segmenter, Mode F-natural v10 — shorter, merged keep/drop test.

Goals:
  - Shorter prompt with minimal example list (v9 had ~12 example strings;
    v10 keeps one example per category).
  - Merge rule 2 (DROP filler) and rule 3 (KEEP particulars) into a single
    SUBSTANCE TEST rule — they were complementary phrasings of the same
    decision.
  - Generalize to any text medium (chat, email, agent log, transcript,
    prose) — no domain-specific idiom lists.
  - Cross-model target: gpt-5-nano, gpt-5.4-nano, gpt-5-mini at low/medium.

Validation pending — see probe_segmenter_feedback_bench.py.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V10 = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Do NOT change wording: "fabulous" stays "fabulous". Connotation \
matters — never swap a word for a synonym, paraphrase, or abstract. The \
only edits allowed are starting and ending the quote at sentence or \
clause boundaries.
  2. SUBSTANCE. For any candidate segment, ask: does it contribute \
specific information a future reader would want to recall — a name, \
place, date, fact, decision, plan, claim, observation, or opinion with \
backing? If yes, KEEP. If it only echoes prior content or expresses \
pure approval/disapproval, DROP. Conversational framing is DROPPED: \
greetings ("Hi!"), sign-offs ("Thanks, Sarah"), polite restatements of \
what someone just asked, envelope phrases that gesture at content \
without conveying it ("please find attached"), chat reactions ("omg \
yes!!"), reactive filler ("I love that!"). An utterance that begins \
with a greeting or softener but carries substantive content is KEPT — \
drop only the leading greeting, not the content.
  3. PRESERVE original order — segments appear in the same order as \
their source quotes.
  4. SEGMENT NATURALLY — break where topics or sub-topics shift. A \
passage that stays on one topic is one segment; a passage covering \
multiple topics gets one segment per topic. Do not artificially split a \
coherent unit, and do not artificially merge unrelated ones.
  5. STANDALONE — each segment reads on its own. If a quote uses a \
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
        prompt = PROMPT_F_NATURAL_V10.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V10.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
