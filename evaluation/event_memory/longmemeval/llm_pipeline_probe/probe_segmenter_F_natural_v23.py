"""LLM segmenter, Mode F-natural v23 — unified principle, cross-domain.

v22's DROP rule was conversation-centric (greetings, thanks, sign-offs)
and example-heavy. The general principle is actually just the inverse
of v22's KEEP rule:

  KEEP content specific to this passage; DROP content that is
  interchangeable across similar passages.

That principle covers all media:
  - Conversation: greetings, thanks, sign-offs, polite restatements
  - Email: stock envelope phrases ("As discussed,", "Let me know if
    any questions")
  - News: bylines, datelines, "Read more", "Subscribe", attribution
    boilerplate
  - Textbook / docs: chapter prefaces, "In this chapter we will
    learn", "See also:", license headers, copyright

v23 states this principle once and uses cross-domain examples to
anchor it. Rules 2 (KEEP specific) and 3 (DROP interchangeable) are
now stated as two sides of the same principle.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V23 = """\
Compress this passage into the parts a human would still want during \
memory reconstruction, broken into a list of standalone memory \
segments. These segments are stored verbatim and shown back when the \
memory is retrieved — keep everything that carries content for the \
reconstructed passage, even if it would not be findable on its own \
(search uses other indexes).

Rules:
  1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording — \
"fabulous" stays "fabulous". The only edits allowed are starting and \
ending the quote at sentence or clause boundaries.
  2. KEEP what is specific to this passage and DROP what is \
interchangeable across similar passages. Specific content \
differentiates this passage from any other — names, places, dates, \
numbers, identifiers, decisions, plans, opinions, preferences, \
relationships, emotional states tied to events, constraints, \
distinctive phrasing. Interchangeable content is stock framing that \
appears across many passages of the same kind regardless of subject: \
conversational openings/closings ("Hi!", "Thanks!", "Best regards"), \
polite restatements of what was just asked, stock envelope phrases \
("As discussed,", "Let me know if any questions"), news bylines and \
"Read more" CTAs, textbook chapter prefaces ("In this chapter we will \
learn..."), license headers, and similar boilerplate.
  3. SHORT RESPONSES that are only meaningful with the prior message \
("yes", "no", "ok", "sure thing", "got it", "sounds good", "great \
point", "omg yes!!", "lol", "acknowledged") are KEPT — they carry the \
answer or reaction the reconstructed memory needs to show, even though \
they would not be findable on their own. When such a response ALSO \
carries content ("ok, leaving Tuesday at 5", "no, I changed my mind \
about Tuesday"), it remains KEPT — emit the whole utterance, not a \
fragment.
  4. An utterance that begins with a greeting or softener but carries \
substantive content is KEPT — drop only the leading greeting, not the \
content.
  5. PRESERVE original order — segments appear in the same order as \
their source quotes.
  6. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  7. STANDALONE — each segment reads on its own. If a quote depends on \
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
        prompt = PROMPT_F_NATURAL_V23.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V23.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
