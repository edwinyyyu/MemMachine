"""Retuned cue-generation prompts for EventMemory (speaker-baked embeddings).

The EventMemory backend bakes speaker names into embedded text via
`MessageContext.source`:
    f"{source}: {text}"  e.g. "Caroline: Yeah, 16 weeks."

The current v2f prompt (`em_architectures.V2F_PROMPT`) was tuned for
plain-text embeddings. This module defines variants that explicitly
mirror the speaker-prefix embedding format so cosine similarity aligns
better with the embedded text distribution.

Variants:
  V2fSpeakerFormat   -- each cue must start with "{speaker}: " where the
                        LLM picks an appropriate speaker per cue.
  V2fMixedSpeakers   -- alternate cues between the two participants.
  V2fRoleTag         -- use role tags "[USER] "/"[ASSISTANT] " instead.
  TypeEnumSpeakerFmt -- type_enumerated variant; each type-cue starts
                        with "{speaker}: ".

Each variant exposes:
    prompt(question, context_section, participants, *) -> str
    parse_cues(response) -> list[str]

The parse function strips trailing "[TYPE]:" style tags inherited from
type_enumerated and preserves the "{speaker}: " prefix.

No framework files, no ingested data touched. All I/O routes through
dedicated emretune_*_cache.json files handled by the caller.
"""

from __future__ import annotations

import re


# --------------------------------------------------------------------------
# V2fSpeakerFormat
# --------------------------------------------------------------------------


V2F_SPEAKERFORMAT_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history between {participant_1} and {participant_2}. Turns in the history \
are embedded in the format:
"<speaker_name>: <chat content>"
Example: "{participant_1}: Yeah, 16 weeks."

Your cues will be embedded and compared via cosine similarity against \
those turns. To align distributions, each cue MUST begin with \
"<speaker_name>: " where <speaker_name> is either {participant_1} or \
{participant_2} -- pick whichever would plausibly have said the content.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is \
this search going? What kind of content is still missing? Should you \
search for similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep \
searching for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns. Do NOT \
write questions ("Did you mention X?"). Write text that would actually \
appear in a chat message, prefixed with the speaker.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


# --------------------------------------------------------------------------
# V2fMixedSpeakers -- alternate cues between the two participants
# --------------------------------------------------------------------------


V2F_MIXEDSPEAKERS_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history between {participant_1} and {participant_2}. Turns are embedded \
in the format "<speaker_name>: <chat content>".

Question: {question}

{context_section}

First, briefly assess the retrieval so far and what is still missing.

Then generate 2 search cues. To maximize speaker coverage, cue 1 MUST \
start with "{participant_1}: " and cue 2 MUST start with \
"{participant_2}: ". Use specific vocabulary that would appear in \
the target conversation turns. Do NOT write questions.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: {participant_1}: <text>
CUE: {participant_2}: <text>
Nothing else."""


# --------------------------------------------------------------------------
# V2fRoleTag -- use [USER] / [ASSISTANT] role tags
# --------------------------------------------------------------------------


V2F_ROLETAG_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history between two participants (a USER and an ASSISTANT). Turns are \
embedded with a role tag prefix like "[USER] <text>" or "[ASSISTANT] <text>".

Question: {question}

{context_section}

First, briefly assess the retrieval so far and what is still missing.

Then generate 2 search cues. Each cue MUST begin with either "[USER] " or \
"[ASSISTANT] " -- pick the role that would plausibly have said the \
content. Use specific vocabulary that would appear in the target turns. \
Do NOT write questions.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: [USER] <text>   or   CUE: [ASSISTANT] <text>
CUE: [USER] <text>   or   CUE: [ASSISTANT] <text>
Nothing else."""


# --------------------------------------------------------------------------
# TypeEnumSpeakerFmt -- type_enumerated with speaker-prefix on each cue
# --------------------------------------------------------------------------


TYPE_ENUM_SPEAKERFMT_PROMPT = """\
Generate cues to find scattered constraints/details in a conversation \
between {participant_1} and {participant_2}. Turns are embedded in the \
format "<speaker_name>: <chat content>". Each cue MUST start with \
"<speaker_name>: " where the speaker is {participant_1} or {participant_2}.

Question: {question}

RETRIEVED SO FAR:
{context_section}

Generate ONE cue per type below. Each cue must mimic how someone would \
ACTUALLY phrase that type of information in chat, prefixed with the \
speaker. Use deictic pronouns (she, he, they) NOT named entities inside \
the content. No quotes around phrases.

[ARRIVAL]: when someone says they arrived/showed up somewhere
[PREFERENCE]: when someone expresses a like/dislike
[CONFLICT]: when a disagreement or issue is discussed
[UPDATE]: informal updates like "oh I forgot to mention" or "just got a message"
[RESOLUTION]: resolutions like "we cleared the air" or "actually it's fine now"
[AFTERTHOUGHT]: casual additions like "wait one more thing" or "btw"
[PHYSICAL]: spatial/physical details like seating, location, position

Format:
CUE: <speaker_name>: <casual chat text for this type>
(7 cues total, one per type)"""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

# Standard: CUE: <rest>  -- but tolerate a leading [TYPE]: tag on
# type_enumerated cues (consistent with em_architectures.TYPE_ENUM_CUE_RE).
CUE_RE = re.compile(
    r"^\s*(?:\[?[A-Z_]+\]?\s*[:\-]\s*)?CUE\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _strip_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'").strip()


def parse_cues(response: str, max_cues: int) -> list[str]:
    cues: list[str] = []
    for m in CUE_RE.finditer(response):
        cue = _strip_quotes(m.group(1))
        if cue:
            cues.append(cue)
        if len(cues) >= max_cues:
            break
    return cues


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def build_v2f_speakerformat_prompt(
    question: str,
    context_section: str,
    participant_1: str,
    participant_2: str,
) -> str:
    return V2F_SPEAKERFORMAT_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=participant_1,
        participant_2=participant_2,
    )


def build_v2f_mixedspeakers_prompt(
    question: str,
    context_section: str,
    participant_1: str,
    participant_2: str,
) -> str:
    return V2F_MIXEDSPEAKERS_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=participant_1,
        participant_2=participant_2,
    )


def build_v2f_roletag_prompt(
    question: str,
    context_section: str,
) -> str:
    return V2F_ROLETAG_PROMPT.format(
        question=question,
        context_section=context_section,
    )


def build_type_enum_speakerfmt_prompt(
    question: str,
    context_section: str,
    participant_1: str,
    participant_2: str,
) -> str:
    return TYPE_ENUM_SPEAKERFMT_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=participant_1,
        participant_2=participant_2,
    )
