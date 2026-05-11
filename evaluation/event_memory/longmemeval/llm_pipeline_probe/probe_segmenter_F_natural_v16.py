"""LLM segmenter, Mode F-natural v16 — v14 structure + 2 more envelope examples.

v14 (1557c, 390/420) was shortest but lost email-scaffolding on gpt-5-mini
(0/5 vs v9 5/5). v15 (1733c, 397/420) added test sentence + envelope
examples — fixed email-scaffolding but broke F2-peer-dialog keep on
gpt-5.4-nano (0/5).

v16 isolates the variable: take v14, add back ONLY "As discussed,"
and "Let me know if any questions" envelope examples. No test sentence.
Target ~1625 chars.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V16 = """\
Compress this passage into a list of standalone memory segments — the \
parts a human would still remember weeks later.

Each segment must be:
  (a) VERBATIM. A contiguous quote from the passage, unchanged in \
wording. Never paraphrase, swap synonyms, or abstract — "fabulous" \
stays "fabulous". The only edits allowed are starting and ending the \
quote at sentence or clause boundaries.
  (b) SUBSTANTIVE. Content that contributes information a future \
reader would want to recall — a name, place, date, number, fact, \
decision, plan, claim, opinion, relationship, or distinctive phrasing. \
Drop content whose only role is moving the conversation along: \
greetings ("Hi team"), sign-offs ("Thanks, Sarah"), polite restatements \
of what someone just asked, envelope phrases that gesture at content \
without conveying it ("As discussed,", "please find attached", "Let me \
know if any questions"), chat reactions ("omg yes!!"), reactive filler \
("I love that!", "I love your reasoning") that echoes the prior turn \
without adding new specifics. An utterance with a leading greeting plus \
substance ("Hey, what's the deal with X?") is kept — drop only the \
greeting.
  (c) STANDALONE. Reads on its own. If a quote uses a referent \
introduced earlier ("the trip" for "the Tokyo trip in May"), widen the \
quote to start where the referent is named.

Segments appear in the same order as their source quotes. Break \
naturally where topics shift: a passage on one topic is one segment; a \
passage covering several topics gets one segment per topic.

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
        prompt = PROMPT_F_NATURAL_V16.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V16.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
