"""LLM segmenter, Mode F-natural v14 — "each segment must" criteria style.

Iterations v10-v13 all regressed by tweaking v9's 6-rule list structure
(too-aggressive merging, trimmed examples, removed test sentence). v14
takes a different approach: organize as "each segment must be (a) X
(b) Y (c) Z" with logistics in a closing paragraph.

This is structurally simpler — three criteria a segment must satisfy,
plus order/segmentation as logistics. Same requirements as v9: verbatim
preservation (connotation), substance over framing, standalone with
referent-widening. Same examples as v9 for substance test (since
trimming them regressed strong models).
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V14 = """\
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
without conveying it ("please find attached"), chat reactions ("omg \
yes!!"), reactive filler ("I love that!", "I love your reasoning") \
that echoes the prior turn without adding new specifics. An utterance \
with a leading greeting plus substance ("Hey, what's the deal with \
X?") is kept — drop only the greeting.
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
        prompt = PROMPT_F_NATURAL_V14.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V14.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
