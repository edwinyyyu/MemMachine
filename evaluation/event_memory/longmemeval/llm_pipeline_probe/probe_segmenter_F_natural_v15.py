"""LLM segmenter, Mode F-natural v15 — v14 structure + restored test
sentence + 2 more envelope examples.

v14 (1557c) was shortest yet, but trade: -16 drop trials vs v9 on
gpt-5-mini's F2-email-scaffolding (0/5 vs v9 5/5). Diagnosis: v14
trimmed envelope examples to just "please find attached" — without
"As discussed,"/"Let me know if any questions" the model doesn't read
the email shape, anchors to "the proposal" and keeps the sentence.

v15 keeps v14's "each segment must be (a)(b)(c)" structure but adds back:
  1. The explicit test sentence ("Does the utterance contribute
     information... If yes KEEP, if echoes/approval DROP").
  2. Two more envelope examples ("As discussed,", "Let me know if any
     questions").

Target ~1850 chars — still 27% shorter than v9.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V15 = """\
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
The test for any utterance: does it contribute new information someone \
would want to recall? If yes, KEEP. If it only echoes prior content or \
expresses pure approval/disapproval without new substance, DROP. \
Common conversational framing to DROP: greetings ("Hi team"), \
sign-offs ("Thanks, Sarah"), polite restatements of what someone just \
asked, envelope phrases that gesture at content without conveying it \
("As discussed,", "please find attached", "Let me know if any \
questions"), chat reactions ("omg yes!!"), reactive filler ("I love \
that!", "I love your reasoning"). An utterance with a leading greeting \
plus substance ("Hey, what's the deal with X?") is KEPT — drop only \
the greeting.
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
        prompt = PROMPT_F_NATURAL_V15.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V15.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
