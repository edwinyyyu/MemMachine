"""LLM segmenter, Mode F-natural v40 -- detect-shifts-not-cut-points framing.

v33 (shipped) over-fragments LoCoMo conversational data. v36/v37/v38/v39
all hit 0/12 on real F4 over-frag cases (3/15 includes a vacuous
anti-regression case). The framings differed (definition-only,
topic-as-unit, operational test, point-being-made) but the model
defaulted to sentence-level cuts.

v40 reframes the TASK structure of rule 6. v33-v39 all phrase the
rule as "break where topics shift" -- an instruction to choose
cut points. The model's default response is to find them
everywhere. v40 inverts: "identify the topic shifts present in the
passage; each detected shift creates a boundary." The default action
becomes "no shift detected -> one segment". Detection is a positive
test (find evidence) rather than a partitioning act (place cuts).

Continuation and shift signal enumeration unchanged from v39
(point-being-made framing). No outcome bias, no inline example.

All other rules are identical to v33.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V40 = """\
Compress this passage into the parts a human would still want during \
memory reconstruction, broken into a list of standalone memory \
segments. These segments are stored verbatim and shown back when the \
memory is retrieved.

Rules:
1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording -- \
"fabulous" stays "fabulous"; preserve whitespace, newlines, and \
special characters within a segment exactly. The only edits allowed \
are starting and ending the quote at sentence or clause boundaries.
2. KEEP what is specific to this passage; DROP what is interchangeable \
across similar passages. Specific content differentiates this passage \
from any other -- names, places, dates, numbers, identifiers, \
decisions, plans, opinions, preferences, relationships, emotional \
states tied to events, constraints, distinctive phrasing. Multi-line \
non-prose blocks (code, tables, ASCII art, diagrams, encoded text) \
are also specific -- keep each as one segment; dropping such a block \
as decoration is a FAILURE. Interchangeable content has none of \
these specifics -- it is pure framing that could open or close any \
passage regardless of subject (e.g., bare greetings, sign-offs).
3. SHORT RESPONSES that are only meaningful with the prior message \
("yes", "no", "ok", "got it", "sounds good", "acknowledged") are \
KEPT -- they carry the answer or reaction the reconstructed memory \
needs to show.
4. An utterance that begins with a greeting or softener but carries \
substantive content is KEPT -- drop only the leading greeting, not \
the content.
5. PRESERVE original order -- segments appear in the same order as \
their source quotes.
6. IDENTIFY TOPIC SHIFTS, DO NOT PLACE CUTS. Scan the passage for \
topic shifts; each shift you detect creates a segment boundary, and \
the passage between any two boundaries is one segment. A topic shift \
is a transition in the passage from one focal subject, claim, \
reaction, decision, or question to a different one. The evidence for \
a shift is positive and concrete -- one of: a different focal subject \
than the one immediately before, a different communicative purpose \
(reaction -> query, statement -> proposal, narrative -> commentary, \
question -> answer to a different question), or an explicit \
discourse marker of departure ("by the way", "anyway", "speaking \
of", "moving on", "oh"). When no such evidence is present between \
two consecutive sentences, no shift occurred and they belong to one \
segment. Sentences that build, elaborate, restate, react to, \
intensify, or qualify the same point share a topic; sub-aspects of \
the current focal subject (its attributes, parts, consequences) are \
continuations, not shifts. Do not artificially split a coherent \
unit, and do not artificially merge unrelated ones.
7. STANDALONE -- each segment reads on its own. If a quote depends on \
a referent introduced earlier ("the trip" for "the Tokyo trip in \
May"), widen the quote to start where the referent is named.

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
        prompt = PROMPT_F_NATURAL_V40.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V40.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
