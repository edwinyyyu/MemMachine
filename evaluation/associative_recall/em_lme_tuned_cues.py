"""LME-tuned cue generation prompts for EventMemory on LongMemEval-hard.

Five variants tested:
  1. v2f_lme_userformat         - cues begin with "User: " or "Assistant: ",
                                  LLM picks which speaker plausibly said it.
  2. v2f_lme_user_only          - all 2 cues prefixed "User: ".
  3. v2f_lme_mixed_7030         - 2 User + 1 Assistant cue (target ~70/30).
  4. type_enumerated_lme_retuned - LME-register categories (USER_FACT,
                                    DECISION, PREFERENCE, TEMPORAL,
                                    QUESTION_ASKED) with speaker-prefix.
  5. ens_2_lme_retuned          - variant 1 + variant 4, sum_cosine.

All run with expand_context=3 (the LME winning recipe).

This module ONLY contains prompts and light parsing helpers.  The eval
script em_lme_tuned_eval.py drives them.
"""

from __future__ import annotations

import re
from pathlib import Path

# =========================================================================
# Variant 1: v2f_lme_userformat — LLM decides speaker per cue
# =========================================================================

V2F_LME_USERFORMAT_PROMPT = """\
You are generating search text for semantic retrieval over a chat log where \
each turn is embedded as "User: <text>" or "Assistant: <text>".

Question (asked by the user about their past conversations): {question}

{context_section}

Briefly assess: what kind of content still needs to be found? In the past \
chat, would this content most plausibly appear in something the USER said \
(stating a fact, preference, event, or decision about themselves) or in \
something the ASSISTANT said (acknowledging, summarizing, or answering)?

Then generate exactly 2 search cues. Each cue MUST begin with either \
`User: ` or `Assistant: ` to match how turns are embedded. Use casual \
first-person register; no quotes; no questions ("did I mention X?"). Write \
text that would actually appear in that speaker's chat turn.

Format:
ASSESSMENT: <1-2 sentences>
CUE: User: <text>   OR   CUE: Assistant: <text>
CUE: User: <text>   OR   CUE: Assistant: <text>
Nothing else."""


# =========================================================================
# Variant 2: v2f_lme_user_only — both cues prefixed "User: "
# =========================================================================

V2F_LME_USER_ONLY_PROMPT = """\
You are generating search text for semantic retrieval over a chat log where \
each turn is embedded as "User: <text>" or "Assistant: <text>".

Question (asked by the user about their own past): {question}

{context_section}

Briefly assess the search state. The gold content is almost always \
user-authored ("I did X", "I prefer Y", "I'm going to Z"), so generate \
cues that look like user-side chat messages.

Then generate exactly 2 search cues. Each cue MUST begin with `User: `. \
Use casual first-person register; no quotes; no questions. Write text that \
would appear verbatim in a user chat message.

Format:
ASSESSMENT: <1-2 sentences>
CUE: User: <text>
CUE: User: <text>
Nothing else."""


# =========================================================================
# Variant 3: v2f_lme_mixed_7030 — 2 User + 1 Assistant cue
# =========================================================================

V2F_LME_MIXED_7030_PROMPT = """\
You are generating search text for semantic retrieval over a chat log where \
each turn is embedded as "User: <text>" or "Assistant: <text>".

Question (asked by the user about their past): {question}

{context_section}

Briefly assess. Most gold is user-authored, but sometimes the answer is \
only explicit in the ASSISTANT's reply (e.g. the assistant summarized or \
named what the user described).

Then generate exactly 3 search cues:
 - 2 cues beginning with `User: ` (user-authored statements)
 - 1 cue beginning with `Assistant: ` (assistant reply/acknowledgement)

Use casual chat register; no quotes; no questions. Each cue should be \
text that would appear verbatim in that speaker's turn.

Format:
ASSESSMENT: <1-2 sentences>
CUE: User: <text>
CUE: User: <text>
CUE: Assistant: <text>
Nothing else."""


# =========================================================================
# Variant 4: type_enumerated_lme_retuned — LME-register categories
# =========================================================================

TYPE_ENUM_LME_RETUNED_PROMPT = """\
Generate cues to find scattered first-person diary/chat content in a \
conversation log. Each turn is embedded as "User: <text>" or \
"Assistant: <text>". Your cues will be embedded the same way.

Question (asked by the user about their past): {question}

RETRIEVED SO FAR:
{context_section}

Generate ONE cue per category below. Each cue MUST begin with `User: ` or \
`Assistant: ` to match how turns are embedded. Use casual first-person chat \
register; no quotes; no questions.

[USER_FACT]: a personal fact the user stated about themselves (identity, \
relationships, possessions, location, job). Prefix `User: `.

[DECISION]: a decision the user made or action they took ("I decided to \
...", "I went ahead and ..."). Prefix `User: `.

[PREFERENCE]: a like, dislike, or rule-out the user expressed ("I love \
...", "I can't stand ...", "I won't ..."). Prefix `User: `.

[TEMPORAL]: a time-stamped event or schedule from the user's life ("last \
week I ...", "on Friday I ...", "I just ..."). Prefix `User: `.

[QUESTION_ASKED]: a specific question the user asked the assistant \
earlier. Prefix `User: ` (the user's original question, not a rephrasing).

Format:
CUE: User: <text>
CUE: User: <text>
CUE: User: <text>
CUE: User: <text>
CUE: User: <text>
(5 cues total, one per category, in the order above)"""


# =========================================================================
# Parsers
# =========================================================================

# Accept CUE: User: xxx  or CUE: Assistant: xxx  or optional [TAG] prefix.
CUE_LINE_RE = re.compile(
    r"^\s*(?:\[?[A-Z_]+\]?\s*[:\-]\s*)?CUE\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

SPEAKER_RE = re.compile(r"^(?:user|assistant)\s*:\s*", re.IGNORECASE)


def parse_speaker_cues(
    response: str,
    *,
    max_cues: int,
    require_speaker_prefix: bool = True,
) -> list[str]:
    """Parse cues that begin with 'User: ' or 'Assistant: '.

    If a cue line is missing the speaker prefix, we prepend 'User: '
    (since it's the more common fallback in LME).

    Returns at most `max_cues` cues, in the order encountered.
    """
    cues: list[str] = []
    for m in CUE_LINE_RE.finditer(response):
        raw = m.group(1).strip().strip('"').strip()
        if not raw:
            continue
        if require_speaker_prefix and not SPEAKER_RE.match(raw):
            # Silent fallback: prepend User:
            raw = "User: " + raw
        # Normalize the speaker prefix casing to "User: " or "Assistant: ".
        mm = SPEAKER_RE.match(raw)
        if mm:
            prefix = raw[: mm.end()]
            rest = raw[mm.end() :].strip()
            if prefix.lower().startswith("user"):
                raw = "User: " + rest
            else:
                raw = "Assistant: " + rest
        cues.append(raw)
        if len(cues) >= max_cues:
            break
    return cues


# =========================================================================
# Convenience: variant registry
# =========================================================================

VARIANT_PROMPTS: dict[str, dict] = {
    "v2f_lme_userformat": {
        "prompt": V2F_LME_USERFORMAT_PROMPT,
        "max_cues": 2,
        "cache_subkey": "v2f_userformat",
    },
    "v2f_lme_user_only": {
        "prompt": V2F_LME_USER_ONLY_PROMPT,
        "max_cues": 2,
        "cache_subkey": "v2f_useronly",
    },
    "v2f_lme_mixed_7030": {
        "prompt": V2F_LME_MIXED_7030_PROMPT,
        "max_cues": 3,
        "cache_subkey": "v2f_mixed7030",
    },
    "type_enumerated_lme_retuned": {
        "prompt": TYPE_ENUM_LME_RETUNED_PROMPT,
        "max_cues": 5,
        "cache_subkey": "te_lme_retuned",
    },
}


# Cache file locations (distinct from existing LME caches).
CACHE_DIR = Path(__file__).resolve().parent / "cache"
LMETUNE_V2F_USERFORMAT_CACHE = CACHE_DIR / "lmetune_v2f_userformat_cache.json"
LMETUNE_V2F_USERONLY_CACHE = CACHE_DIR / "lmetune_v2f_useronly_cache.json"
LMETUNE_V2F_MIXED7030_CACHE = CACHE_DIR / "lmetune_v2f_mixed7030_cache.json"
LMETUNE_TE_RETUNED_CACHE = CACHE_DIR / "lmetune_te_retuned_cache.json"
