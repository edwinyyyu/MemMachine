"""Pure-regex ingestion-side alt-key heuristics.

Implements the 7 heuristics from `results/ingestion_predictability.md` §7.

Each heuristic is a pure function: given the current turn text and the preceding
turn text, decide whether the heuristic fires and emit the alt-key text.

A single turn may fire multiple heuristics. The caller should dedupe alt-keys
by text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Token sets and regexes (compiled once)
# ---------------------------------------------------------------------------
ANAPHORIC_TOKENS = {
    "that",
    "this",
    "those",
    "these",
    "it",
    "they",
    "he",
    "she",
    "him",
    "her",
    "his",
    "its",
    "their",
    "them",
}

SHORT_RESPONSE_TOKENS = {
    "yeah",
    "yes",
    "yep",
    "ok",
    "okay",
    "sure",
    "no",
    "nope",
    "definitely",
    "exactly",
    "right",
    "true",
    "false",
    "maybe",
}

UPDATE_MARKER_RE = re.compile(
    r"^(actually|wait|oh|scratch that|correction|on second thought|update|"
    r"let me correct|turns out|never mind)[\s,.]",
    re.IGNORECASE,
)

KNOWN_UNKNOWN_RE = re.compile(
    r"let me check|circle back|TBD|pending|not sure|waiting on",
    re.IGNORECASE,
)

ALIAS_EVOLUTION_RE = re.compile(
    r"call(ed)? it|\baka\b|also known as|renamed|new name",
    re.IGNORECASE,
)

# structured_fact: keyword list. Interpreting "by (monday|tuesday|...)" as
# "by <weekday>" (case-insensitive), and also catching $, %, v<digits>.
STRUCTURED_FACT_RE = re.compile(
    r"allergy|deadline|prescription|dosage|prefer|"
    r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\$|%|\bv\d+\b",
    re.IGNORECASE,
)

# rare_entity: match capitalized tokens NOT sentence-initial, and version/number tokens.
# Strategy: scan capitalized \b[A-Z][a-zA-Z]{2,}\b but drop the first token of each sentence.
CAP_TOKEN_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Number-like / version tokens (also used for rare_entity):
NUM_VER_RE = re.compile(
    r"\bv\d+(?:\.\d+)*\b"  # v1, v2.1
    r"|\b\d+(?:\.\d+)+\b"  # 1.2.3
    r"|\b[A-Z]+-\d+\b"  # JIRA-4521
    r"|\$\d[\d,\.]*"  # $28K, $30
    r"|\b\d+%\b"  # 25%
    r"|\b\d{1,2}:\d{2}\s*(am|pm)?\b"  # 9:00, 9:00am
    r"|\b\d+[KkMm]\b",  # 28K
    re.IGNORECASE,
)

HEURISTIC_NAMES = (
    "anaphoric",
    "short_response",
    "update_marker",
    "known_unknown",
    "alias_evolution",
    "structured_fact",
    "rare_entity",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class AltKey:
    parent_index: int  # index into SegmentStore.segments
    heuristic: str  # which heuristic fired
    text: str  # the alt-key text to embed


# ---------------------------------------------------------------------------
# Helper: first-token extraction (robust to leading punctuation/whitespace).
# ---------------------------------------------------------------------------
def _first_token_lower(text: str) -> str:
    text = text.lstrip().lstrip("\"'([{<-*").lstrip()
    if not text:
        return ""
    # first whitespace or punctuation chunk
    m = re.match(r"[A-Za-z']+", text)
    return m.group(0).lower() if m else ""


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


# ---------------------------------------------------------------------------
# Heuristic predicates
# ---------------------------------------------------------------------------
def fires_anaphoric(text: str) -> bool:
    return _first_token_lower(text) in ANAPHORIC_TOKENS


def fires_short_response(text: str) -> bool:
    if _word_count(text) <= 4:
        return True
    return _first_token_lower(text) in SHORT_RESPONSE_TOKENS


def fires_update_marker(text: str) -> bool:
    return bool(UPDATE_MARKER_RE.search(text.lstrip()))


def fires_known_unknown(text: str) -> bool:
    return bool(KNOWN_UNKNOWN_RE.search(text))


def fires_alias_evolution(text: str) -> bool:
    return bool(ALIAS_EVOLUTION_RE.search(text))


def fires_structured_fact(text: str) -> bool:
    return bool(STRUCTURED_FACT_RE.search(text))


def _rare_entity_tokens(text: str) -> list[str]:
    """Return capitalized proper-noun-like tokens (not sentence-initial) +
    number/version tokens. Deduped preserving order."""
    found: list[str] = []
    seen: set[str] = set()

    # Number/version tokens anywhere
    for m in NUM_VER_RE.finditer(text):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            found.append(tok)

    # Capitalized tokens, skipping first token of each sentence
    for sent in SENTENCE_SPLIT_RE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        # Find capitalized tokens with their offsets
        matches = list(CAP_TOKEN_RE.finditer(sent))
        if not matches:
            continue
        # Compute offset of the first real letter token
        first_alpha = re.search(r"[A-Za-z]+", sent)
        first_start = first_alpha.start() if first_alpha else -1
        for m in matches:
            if m.start() == first_start:
                # Sentence-initial capitalization is cheap (grammar), skip
                continue
            tok = m.group(0)
            if tok not in seen:
                seen.add(tok)
                found.append(tok)
    return found


def fires_rare_entity(text: str) -> bool:
    return len(_rare_entity_tokens(text)) > 0


# ---------------------------------------------------------------------------
# Alt-key generation per §7 of the analysis
# ---------------------------------------------------------------------------
def generate_alt_keys_for_turn(
    parent_index: int,
    this_turn_text: str,
    preceding_turn_text: str,
) -> list[AltKey]:
    """Fire every applicable heuristic on a turn and emit alt-keys.

    Returns a list of AltKey objects (one per heuristic that fires). Duplicate
    alt-key text across heuristics is deduped by the CALLER (so each heuristic
    still gets credit in fire-counts).
    """
    out: list[AltKey] = []

    this_text = this_turn_text or ""
    prev_text = preceding_turn_text or ""
    joined = (prev_text + " " + this_text).strip()

    if fires_anaphoric(this_text):
        out.append(
            AltKey(
                parent_index=parent_index,
                heuristic="anaphoric",
                text=joined,
            )
        )

    if fires_short_response(this_text):
        out.append(
            AltKey(
                parent_index=parent_index,
                heuristic="short_response",
                text=joined,
            )
        )

    if fires_update_marker(this_text):
        out.append(
            AltKey(
                parent_index=parent_index,
                heuristic="update_marker",
                text=joined,
            )
        )

    if fires_known_unknown(this_text):
        out.append(
            AltKey(
                parent_index=parent_index,
                heuristic="known_unknown",
                text="unresolved question pending check: " + this_text[:200],
            )
        )

    if fires_alias_evolution(this_text):
        out.append(
            AltKey(
                parent_index=parent_index,
                heuristic="alias_evolution",
                text=joined,
            )
        )

    if fires_structured_fact(this_text):
        out.append(
            AltKey(
                parent_index=parent_index,
                heuristic="structured_fact",
                text="structured fact: " + this_text,
            )
        )

    tokens = _rare_entity_tokens(this_text)
    if tokens:
        entity_line = "entities: " + " ".join(tokens) + " | " + this_text[:150]
        out.append(
            AltKey(
                parent_index=parent_index,
                heuristic="rare_entity",
                text=entity_line,
            )
        )

    return out


def generate_alt_keys_for_conversation(
    segments: list,
) -> tuple[list[AltKey], dict[str, int]]:
    """Generate alt-keys for an ordered list of segments (all from the same
    conversation, sorted by turn_id) and return (alt_keys, fire_counts).

    `segments` must be an iterable of objects with `.index` and `.text`.
    """
    alt_keys: list[AltKey] = []
    fire_counts: dict[str, int] = dict.fromkeys(HEURISTIC_NAMES, 0)
    prev_text = ""
    for seg in segments:
        keys = generate_alt_keys_for_turn(
            parent_index=seg.index,
            this_turn_text=seg.text,
            preceding_turn_text=prev_text,
        )
        for k in keys:
            fire_counts[k.heuristic] += 1
        alt_keys.extend(keys)
        prev_text = seg.text
    return alt_keys, fire_counts


def generate_all_alt_keys(
    segments: list,
) -> tuple[list[AltKey], dict[str, int]]:
    """Generate alt-keys for a flat list of segments that may span multiple
    conversations. Groups by conversation_id, orders by turn_id, preserves
    the original segment.index.
    """
    by_conv: dict[str, list] = {}
    for s in segments:
        by_conv.setdefault(s.conversation_id, []).append(s)

    all_keys: list[AltKey] = []
    total_counts: dict[str, int] = dict.fromkeys(HEURISTIC_NAMES, 0)
    for cid, segs in by_conv.items():
        segs_sorted = sorted(segs, key=lambda s: s.turn_id)
        keys, counts = generate_alt_keys_for_conversation(segs_sorted)
        all_keys.extend(keys)
        for k, v in counts.items():
            total_counts[k] += v
    return all_keys, total_counts
