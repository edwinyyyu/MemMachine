"""LLM segmenter, Mode F-natural v20 — drop ONLY pure social plumbing.

User correction (continuing the segment-vs-derivative principle):
  "omg yes!!" contains "yes" — an affirmative. By the same logic,
  "I love your reasoning" / "great point" / "Yeah, cool." all express
  opinion or agreement, not pure receipt acknowledgment. v9-v19 were
  over-aggressive on the DROP side for these.

v20 narrows the DROP list to TRULY content-free utterances:
  - Greetings ("Hi!", "Hi team", "Hello there")
  - Thanks ("Thanks!", "Thank you!")
  - Sign-offs ("Bye!", "Best regards", "Take care", "Talk soon")
  - Polite restatements (re-stating what someone just asked)
  - Envelope-only phrases ("As discussed,", "please find attached",
    "Let me know if any questions")

Everything else KEEPS, including:
  - Short affirmatives ("Yes.", "No.", "Yeah", "Sure thing.", "OK.")
  - Reactions / opinions / agreements ("omg yes!!", "I love that!",
    "great point", "I love your reasoning", "Yeah, cool.",
    "I agree.", "I disagree.")
  - Content answers ("Pizza.", "Tokyo.", "I'm vegan.")
  - Anything with named entities, dates, numbers, decisions, etc.

The principle: for the reconstructed memory, the user wants to see
what each speaker said and how they reacted. Only TRULY content-free
glue (greetings, thanks, sign-offs, envelope-only) is noise.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_F_NATURAL_V20 = """\
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
2. DROP ONLY content with zero information beyond social courtesy or \
message-receipt acknowledgment: greetings ("Hi!", "Hi team"), thanks \
("Thanks!", "Thanks, Bob"), sign-offs ("Best regards", "Take care"), \
polite restatements of what the other party just asked, envelope-only \
phrases that gesture at content without conveying it ("As discussed,", \
"please find attached", "Let me know if any questions", "What a great \
question!", "I hope this helps!"). \
\
KEEP everything else, including short affirmatives ("Yes.", "No.", \
"Sure thing.", "OK.", "Got it.", "Sounds good.", "Will do."), \
reactions and opinions ("omg yes!!", "I love that!", "great point", \
"I love your reasoning", "Yeah, cool.", "I agree completely."), \
content answers ("Pizza.", "Tokyo.", "Vanilla, definitely.", "I'm \
vegan."), and any sentence with specifics. These all carry content the \
reconstructed memory needs. \
\
The test for any utterance: does it convey anything beyond bare \
greeting / thanks / sign-off / envelope phrasing? If yes, KEEP. An \
utterance that begins with a greeting or softener but carries \
substantive content is KEPT ("Hey, what's the deal with X?" — drop \
only the leading greeting, not the content).
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
        prompt = PROMPT_F_NATURAL_V20.format(passage=text)
        return await call(client, model, prompt, reasoning)
    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL_V20.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat = []
    for r in sub_results:
        flat.extend(r)
    return flat
