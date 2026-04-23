"""Deeper cue-gen tuning variants for EventMemory (speaker-baked embeddings).

Extends `em_retuned_cue_gen.V2F_SPEAKERFORMAT_PROMPT` with length/structure
variants and a properly rewritten type_enumerated variant, plus helpers for
composition variants (alias_expand + speakerformat, two_speaker_filter +
speakerformat cues).

Variants in this module:

Length / structure:
  V2F_SPEAKERFORMAT_SHORT_PROMPT      -- ~=15 words/cue max
  V2F_SPEAKERFORMAT_5CUES_PROMPT      -- 5 short cues per call
  V2F_SPEAKERFORMAT_NATURAL_TURN_PROMPT -- write cues AS the turn itself

Specialist retuning:
  CHAIN_WITH_SCRATCHPAD_SPEAKERFORMAT_PROMPT -- chain-style reasoning
  TYPE_ENUMERATED_EM_RETUNED_PROMPT   -- rewrite of type_enumerated (not SIMPLE
                                         prefix swap) per question intent.

Composition cue generators reuse these prompts; no framework files touched.

All prompts match the cosine-similarity register of speaker-baked
EventMemory embeds:  f"{speaker}: {chat text}".
"""

from __future__ import annotations

import re


# --------------------------------------------------------------------------
# Length / structure variants
# --------------------------------------------------------------------------


V2F_SPEAKERFORMAT_SHORT_PROMPT = """\
You are generating SHORT search text for semantic retrieval over a \
conversation history between {participant_1} and {participant_2}. Turns \
are embedded in the format:
"<speaker_name>: <chat content>"
Example: "{participant_1}: Yeah, 16 weeks."

LoCoMo chat turns are often TERSE (5-15 words). To align cosine distribution \
with embedded turns, each cue MUST:
  1. Begin with "<speaker_name>: " ({participant_1} or {participant_2}).
  2. Be VERY SHORT -- at most 15 words AFTER the speaker prefix.
  3. Use concrete chat vocabulary, no hedging, no questions.

Question: {question}

{context_section}

First, briefly assess retrieval so far and what is still missing.

Then generate 2 short search cues based on your assessment.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <speaker_name>: <short text, <=15 words>
CUE: <speaker_name>: <short text, <=15 words>
Nothing else."""


V2F_SPEAKERFORMAT_5CUES_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history between {participant_1} and {participant_2}. Turns are embedded in \
the format:
"<speaker_name>: <chat content>"
Example: "{participant_1}: Yeah, 16 weeks."

Each cue MUST begin with "<speaker_name>: " ({participant_1} or \
{participant_2}). Use concrete chat vocabulary. Do NOT write questions.

Question: {question}

{context_section}

First, briefly assess retrieval so far.

Then generate 5 DIVERSE short cues. Diversity angles to cover across the 5:
  - different speakers (at least one from each participant if plausible)
  - different phrasings of the same answer (e.g. direct statement vs. \
follow-up affirmation)
  - different temporal/contextual framings (when, where, outcome, reaction)
  - adjacent content (what would be said right before / right after)
Each cue should be SHORT (<=20 words after the prefix), chat-register.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
CUE: <speaker_name>: <text>
Nothing else."""


V2F_SPEAKERFORMAT_NATURAL_TURN_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history between {participant_1} and {participant_2}. Turns are embedded in \
the format:
"<speaker_name>: <chat content>"
Example: "{participant_1}: Yeah, 16 weeks."

Imagine what either {participant_1} or {participant_2} might actually say \
in this chat that would answer or approach the question. Write 2 cues that \
WERE (or could plausibly be) a real turn in that conversation -- not a \
question, not a paraphrase, not a search query. A turn.

Question: {question}

{context_section}

Constraints for each cue:
  - Begin with "<speaker_name>: " ({participant_1} or {participant_2}).
  - First-person voice, chat register, no hedging like "I wonder" or "maybe".
  - Use specific vocabulary that would literally appear in the target turn.
  - Each cue is ONE turn a real speaker would say in this chat.

Format:
ASSESSMENT: <1-2 sentence self-evaluation, what kind of turn to target>
CUE: <speaker_name>: <text that WAS (or could be) a real turn>
CUE: <speaker_name>: <text that WAS (or could be) a real turn>
Nothing else."""


# --------------------------------------------------------------------------
# Specialist retuning
# --------------------------------------------------------------------------


CHAIN_WITH_SCRATCHPAD_SPEAKERFORMAT_PROMPT = """\
You are performing chain-of-evidence retrieval over a conversation between \
{participant_1} and {participant_2}. Turns are embedded in the format \
"<speaker_name>: <chat content>".

Question: {question}

{context_section}

Many LoCoMo questions require connecting MULTIPLE turns: A sets up context, \
B answers, C confirms or extends. Work it out on a scratchpad.

SCRATCHPAD (think step-by-step, keep it brief):
  - What specific clue/link is already visible in retrieved excerpts?
  - What ADJACENT link is still missing? (setup? confirmation? outcome? \
reaction?)
  - Which speaker would plausibly say each link?

Then produce 2 cues. Each cue targets a DIFFERENT link of the chain and \
MUST begin with "<speaker_name>: " ({participant_1} or {participant_2}). \
Cues are chat text (not questions), concrete vocabulary.

Format:
SCRATCHPAD: <your reasoning, 2-4 sentences>
CUE: <speaker_name>: <text targeting link 1>
CUE: <speaker_name>: <text targeting link 2>
Nothing else."""


TYPE_ENUMERATED_EM_RETUNED_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
between {participant_1} and {participant_2}. Turns are embedded in the \
format "<speaker_name>: <chat content>". Each cue MUST begin with \
"<speaker_name>: " where the speaker is {participant_1} or {participant_2}.

Question: {question}

RETRIEVED SO FAR:
{context_section}

Instead of bucketing by generic categories, target THE ANSWER TYPE this \
question implies, then generate 5 cues that together cover the most likely \
shapes of a relevant turn. Pick from the angles below as fits:

  DIRECT_STATEMENT -- a speaker just says the answer as a fact:
      e.g. "Caroline: We decided last Tuesday."
  FIRST_MENTION    -- the moment it first came up:
      e.g. "Caroline: I've been thinking about joining the LGBTQ group."
  CONFIRMATION     -- the other speaker confirms/asks for details:
      e.g. "Melanie: So you're definitely doing it? Monday?"
  OUTCOME          -- result after the event / decision:
      e.g. "Caroline: That meeting yesterday really helped me."
  REFERENCE_BACK   -- a later casual reference using deictic:
      e.g. "Caroline: Remember that thing I told you about?"

For each of the 5 cues, pick whichever angle + speaker would most help \
retrieval. Use deictic pronouns (she, he, they) NOT named entities inside \
the content. Chat register, short sentences.

Format:
CUE: <speaker_name>: <text, angle 1>
CUE: <speaker_name>: <text, angle 2>
CUE: <speaker_name>: <text, angle 3>
CUE: <speaker_name>: <text, angle 4>
CUE: <speaker_name>: <text, angle 5>
(5 cues total)"""


# --------------------------------------------------------------------------
# Prompt builders
# --------------------------------------------------------------------------


def build_speakerformat_short_prompt(
    question: str,
    context_section: str,
    participant_1: str,
    participant_2: str,
) -> str:
    return V2F_SPEAKERFORMAT_SHORT_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=participant_1,
        participant_2=participant_2,
    )


def build_speakerformat_5cues_prompt(
    question: str,
    context_section: str,
    participant_1: str,
    participant_2: str,
) -> str:
    return V2F_SPEAKERFORMAT_5CUES_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=participant_1,
        participant_2=participant_2,
    )


def build_speakerformat_natural_turn_prompt(
    question: str,
    context_section: str,
    participant_1: str,
    participant_2: str,
) -> str:
    return V2F_SPEAKERFORMAT_NATURAL_TURN_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=participant_1,
        participant_2=participant_2,
    )


def build_chain_scratchpad_speakerformat_prompt(
    question: str,
    context_section: str,
    participant_1: str,
    participant_2: str,
) -> str:
    return CHAIN_WITH_SCRATCHPAD_SPEAKERFORMAT_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=participant_1,
        participant_2=participant_2,
    )


def build_type_enum_em_retuned_prompt(
    question: str,
    context_section: str,
    participant_1: str,
    participant_2: str,
) -> str:
    return TYPE_ENUMERATED_EM_RETUNED_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=participant_1,
        participant_2=participant_2,
    )


# --------------------------------------------------------------------------
# Parsing (tolerates optional "SCRATCHPAD:" and "ASSESSMENT:" lines, ignores
# them; captures all CUE: lines). Handles optional "[ANGLE]:" tags like
# type_enumerated / retuned variants.
# --------------------------------------------------------------------------

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
