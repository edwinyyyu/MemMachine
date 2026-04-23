"""Reflective-memory architecture for LongMemEval-hard EventMemory.

Per query q, we run 1..N rounds of (cue_gen, retrieve, reflect). The LLM's
reflections are written to a scratch memory (a small local list of (text,
embedding) tuples). From round 2 onward, retrieval scores candidates against
both the EventMemory corpus AND the scratch memory; the scratch memory is
ALSO passed back into the cue-gen prompt so the LLM can steer the next round.

Design goals:
  - Same retrieval substrate: EventMemory.query(..., expand_context=3).
  - Same "User: " prefix convention for queries and cues.
  - Round 1 cues use the best-known LME prompt (em_v2f_lme_mixed_7030:
    70% "User: " cues, 30% "Assistant: ") — reuse the V2F_LME_MIXED_7030
    cache when possible (byte-identical prompt text for round 1).
  - Scratch memory = list of (sentence, np.ndarray). Embedding uses the
    same text-embedding-3-small used for retrieval; scores against scratch
    use plain cosine similarity and are combined via max-per-turn.
  - LLM reflections are parsed as JSON with keys {learned: [str], still_need: [str]}.

This module only contains prompts and helpers. The driver is reflmemlme_eval.py.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


CACHE_DIR = Path(__file__).resolve().parent / "cache"
REFLMEMLME_CUE_ROUND1_CACHE = CACHE_DIR / "reflmemlme_cue_round1_cache.json"
REFLMEMLME_CUE_ROUNDN_CACHE = CACHE_DIR / "reflmemlme_cue_roundn_cache.json"
REFLMEMLME_REFLECT_CACHE = CACHE_DIR / "reflmemlme_reflect_cache.json"


# =========================================================================
# Round 1 cue-gen prompt: copied verbatim from em_v2f_lme_mixed_7030
# so we can reuse that cache (byte-identical prompt text).
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
# Reflection prompt (after a round's retrieval).
# LLM produces JSON: {"learned": [...], "still_need": [...]}.
# =========================================================================

REFLECTION_PROMPT = """\
You are reasoning over retrieved chat turns to build a scratch memory that \
will guide further retrieval for the question below. Scratch-memory entries \
will be embedded and used as additional semantic anchors alongside the chat \
corpus, so each entry should read like a short declarative sentence (not a \
question).

Question (asked by the user about their past): {question}

SCRATCH MEMORY SO FAR:
{scratch_section}

NEWLY RETRIEVED TURNS (from the latest round of cues):
{turns_section}

Your job:
1. Extract up to 5 short declarative sentences stating facts or \
partial-facts that HAVE been confirmed by the retrieved turns. Phrase them \
as 1st-person user statements when the fact is about the user ("I am 32", \
"I plan to visit Paris in April"). Be concrete — include entity names, \
dates, and numbers when they appear.
2. Identify up to 3 gap statements — what information is still MISSING to \
answer the question. Phrase each gap as a short declarative clause that \
would plausibly appear in a user chat turn if the missing info were stated \
("I'm going to Rachel's wedding on <date>", "My current age is <number>"). \
Avoid questions; write the hypothetical statement.

Output STRICT JSON only, no prose:
{{"learned": ["...", "..."], "still_need": ["...", "..."]}}"""


# =========================================================================
# Round-N cue-gen prompt (rounds 2+): uses scratch memory as steering.
# =========================================================================

ROUND_N_CUE_PROMPT = """\
You are generating NEW search text to fill remaining gaps in an iterative \
retrieval over a chat log where each turn is embedded as "User: <text>" or \
"Assistant: <text>".

Question (asked by the user about their past): {question}

SCRATCH MEMORY (what we've learned so far + what's still missing):
{scratch_section}

PREVIOUSLY GENERATED CUES (do NOT repeat these verbatim):
{prev_cues_section}

Given what's still missing, generate exactly 3 NEW search cues that target \
the gaps. Each cue MUST begin with `User: ` or `Assistant: ` to match how \
turns are embedded. Aim for a 2/1 User/Assistant split. Use casual first-\
person register; no quotes; no questions. Each cue should be text that \
would appear verbatim in that speaker's turn, and it should be DIFFERENT \
from the previous cues in either vocabulary, entity, or time window.

Format:
ASSESSMENT: <1 sentence on what's still missing>
CUE: User: <text>
CUE: User: <text>
CUE: Assistant: <text>
Nothing else."""


# =========================================================================
# Parsers
# =========================================================================

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

    Missing speaker prefix falls back to 'User: ' (LME convention).
    """
    cues: list[str] = []
    for m in CUE_LINE_RE.finditer(response):
        raw = m.group(1).strip().strip('"').strip()
        if not raw:
            continue
        if require_speaker_prefix and not SPEAKER_RE.match(raw):
            raw = "User: " + raw
        mm = SPEAKER_RE.match(raw)
        if mm:
            prefix = raw[: mm.end()]
            rest = raw[mm.end():].strip()
            if prefix.lower().startswith("user"):
                raw = "User: " + rest
            else:
                raw = "Assistant: " + rest
        cues.append(raw)
        if len(cues) >= max_cues:
            break
    return cues


def parse_reflection_json(response: str) -> tuple[list[str], list[str]]:
    """Parse LLM JSON reflection. Robust to stray prose or fenced blocks.

    Returns (learned, still_need). Silently returns empty lists on failure.
    """
    if not response:
        return [], []
    # Try raw parse first.
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    # Strip any leading prose before the first '{'.
    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        text = text[i: j + 1]
    try:
        data = json.loads(text)
    except Exception:
        return [], []
    learned = data.get("learned") or []
    still_need = data.get("still_need") or []
    if not isinstance(learned, list):
        learned = []
    if not isinstance(still_need, list):
        still_need = []
    # Keep only strings.
    learned = [str(x).strip() for x in learned if isinstance(x, (str, int, float))]
    still_need = [
        str(x).strip() for x in still_need if isinstance(x, (str, int, float))
    ]
    learned = [x for x in learned if x]
    still_need = [x for x in still_need if x]
    return learned[:5], still_need[:3]


# =========================================================================
# Scratch memory
# =========================================================================


@dataclass
class ScratchEntry:
    text: str
    kind: str  # "learned" | "still_need"
    embedding: np.ndarray  # shape (d,)


@dataclass
class ScratchMemory:
    entries: list[ScratchEntry] = field(default_factory=list)

    def add(self, text: str, kind: str, embedding: np.ndarray) -> None:
        self.entries.append(ScratchEntry(text=text, kind=kind, embedding=embedding))

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def as_text_lines(self) -> list[str]:
        return [f"[{e.kind}] {e.text}" for e in self.entries]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_turn_against_scratch(
    turn_embedding: np.ndarray, scratch: ScratchMemory
) -> float:
    """Max cosine similarity between a turn's embedding and any scratch
    entry's embedding. Returns 0.0 if scratch is empty."""
    if scratch.is_empty():
        return 0.0
    best = 0.0
    for e in scratch.entries:
        s = cosine(turn_embedding, e.embedding)
        if s > best:
            best = s
    return best


# =========================================================================
# Context formatters
# =========================================================================


def format_scratch_section(scratch: ScratchMemory) -> str:
    if scratch.is_empty():
        return "(scratch memory is empty)"
    lines = []
    for e in scratch.entries:
        lines.append(f"- [{e.kind}] {e.text}")
    return "\n".join(lines)


def format_turns_section(
    turns: list[dict],
    *,
    max_items: int = 20,
    max_len: int = 250,
) -> str:
    """Format a batch of retrieved turns for the reflection prompt."""
    if not turns:
        return "(no turns retrieved this round)"
    sorted_turns = sorted(turns, key=lambda t: t.get("turn_id", 0))
    lines = []
    for t in sorted_turns[:max_items]:
        role = t.get("role", "?")
        text = t.get("text", "")
        lines.append(f"[Turn {t.get('turn_id', '?')}, {role}]: {text[:max_len]}")
    return "\n".join(lines)


def format_prev_cues_section(prev_cues: list[str]) -> str:
    if not prev_cues:
        return "(none)"
    return "\n".join(f"- {c}" for c in prev_cues)


def format_primer_context_lme(
    segments: list,
    *,
    max_items: int = 12,
    max_len: int = 250,
) -> str:
    """Primer context formatter matching em_v2f_lme_mixed_7030 exactly so
    the round-1 prompt can hit the existing lmetune_v2f_mixed7030_cache.

    Copied byte-for-byte from em_architectures.format_primer_context.
    """
    if not segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    sorted_segs = sorted(segments, key=lambda s: s["turn_id"])
    lines = []
    for s in sorted_segs[:max_items]:
        lines.append(
            f"[Turn {s['turn_id']}, {s['role']}]: {s['text'][:max_len]}"
        )
    return "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + "\n".join(lines)
