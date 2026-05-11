"""LLM segmenter, Mode F-natural v9.

Iterated from v1 (probe_segmenter_F_natural.py) across three feedback areas:
  F1 OPINION-VS-FILLER. Real opinions (with elaboration or position framing)
     vs reactive solidarity ("I love that!", "great point", "I love brindle
     room!" echoing a recommendation).
  F2 SCAFFOLDING DIVERSITY. Broaden beyond user-assistant chat to peer
     dialog, email envelopes, group chat reactions, agent logs.
  F3 RULE 3 FRAMING. Replace enumerated list (entity / place / person /
     brand / work / date / price / preference / plan / decision / factual
     claim / phrasing) with a principle (concrete particulars vs generic
     abstractions).

v9 changes vs v8:
  - Rule 2 unified around a single principle ("does the utterance contribute
    information someone would want to recall?") — no special love/hate case.
  - Removed heterogeneous enumeration ("no examples, behavior, prices, dates")
    that mixed general and specific terms.

Validation: 10-rep head-to-head vs v8 on probe_segmenter_feedback_bench.py
14 cases. v9 wins F1-filler-reply keep (8→10), F2-peer-dialog keep (9→10),
loses F1-pure-agent-support drop (8→6). Net: asymmetrically better — fewer
false-drops on real content.

Run identically to v1:
    from probe_segmenter_F_natural_v9 import segment, PROMPT_F_NATURAL_V9
"""

from __future__ import annotations

import asyncio

import openai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V9 = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP content whose role is conversational framing rather than \
substance: standalone greetings ("Hi Alice!", "Hi team"), sign-offs ("Thanks, \
Bob", "Best regards"), polite restatements of what the other party \
just asked, envelope phrases that gesture at content without conveying \
it ("As discussed,", "please find attached", "Let me know if any \
questions", "What a great question!", "I hope this helps!"), chat \
reactions ("omg yes!!", "lol", "fwiw"), reactive filler ("I love \
that!", "great point", "I love your reasoning", "sounds great!"). The \
test for any utterance: does it contribute information someone would \
want to recall — a fact, name, claim, decision, plan, observation, or \
opinion with backing? If yes, KEEP. If it only echoes prior content or \
expresses pure approval/disapproval without new substance, DROP. An \
utterance that begins with a greeting or softener but carries \
substantive content is KEPT ("Hey, what's the deal with X?" contains \
the question — drop only the leading greeting, not the content).
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
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

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


async def segment(
    client: openai.AsyncOpenAI,
    model: str,
    text: str,
    reasoning: str | None = "low",
    window_chars: int = WINDOW_CHARS,
) -> list[str]:
    """Run F-natural v9; transparently pre-window if input is huge."""
    if len(text) <= window_chars:
        prompt = PROMPT_F_NATURAL_V9.format(passage=text)
        return await call(client, model, prompt, reasoning)

    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V9.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat: list[str] = []
    for r in sub_results:
        flat.extend(r)
    return flat
