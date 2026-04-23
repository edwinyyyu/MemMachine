"""Salience pruning for retrieval index.

Marks low-salience turns (chitchat, filler, short acks) and either drops them
from the retrieval index or down-weights their cosine scores.

Variants:
    prune_regex_aggressive  — drop all regex-low-salience turns
    prune_regex_conservative — drop only word_count<=3 AND backchannel first token
    prune_llm               — drop turns where LLM classifier says NO
    downweight_regex        — keep, multiply cosine similarity by 0.5

Exposes:
    build_salience_mask(store, variant, ...) -> dict
    PrunedSegmentStore (subclasses SegmentStore)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    DATA_DIR,
    EmbeddingCache,
    LLMCache,
    RetrievalResult,
    Segment,
    SegmentStore,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

CACHE_FILE = CACHE_DIR / "salience_llm_cache.json"


# ---------------------------------------------------------------------------
# Regex heuristic sets
# ---------------------------------------------------------------------------
BACKCHANNEL_FIRST_TOKENS = {
    "yeah", "yes", "yep", "ok", "okay", "sure", "nope", "no",
    "hmm", "hmmm", "lol", "haha", "right", "true", "k", "kk",
    "cool", "got", "nice", "thanks", "ty", "np", "alright",
    "awesome", "great", "fine", "sounds", "wow", "oh",
}

# backchannel regex patterns (match whole text roughly)
PURE_BACKCHANNEL_RE = re.compile(
    r"^\s*(yeah|yes|yep|yup|ok+|okay|sure|nope|no|hmm+|lol|haha|right|true|"
    r"k{1,2}|cool|got\s*it|nice|thanks?|ty|np|alright|awesome|great|fine|"
    r"sounds\s+good)\s*[.!?,:;]*\s*$",
    re.IGNORECASE,
)

# High-salience markers: numbers / $ / % / dates / versions
STRUCTURED_MARKER_RE = re.compile(
    r"\$\d|\b\d+%|\bv\d+(?:\.\d+)*\b|\b\d{1,2}:\d{2}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\b\d+[KkMm]\b",
    re.IGNORECASE,
)

# Capitalized proper-noun-like tokens NOT sentence-initial (rough)
CAP_TOKEN_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _first_token_lower(text: str) -> str:
    t = text.lstrip().lstrip("\"'([{<-*").lstrip()
    if not t:
        return ""
    m = re.match(r"[A-Za-z']+", t)
    return m.group(0).lower() if m else ""


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _has_proper_noun(text: str) -> bool:
    for sent in SENTENCE_SPLIT_RE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        matches = list(CAP_TOKEN_RE.finditer(sent))
        if not matches:
            continue
        first_alpha = re.search(r"[A-Za-z]+", sent)
        first_start = first_alpha.start() if first_alpha else -1
        for m in matches:
            if m.start() != first_start:
                return True
    return False


def is_high_salience_regex(text: str) -> bool:
    """High-salience if it has numbers/dates/$/% OR proper nouns OR long."""
    if STRUCTURED_MARKER_RE.search(text):
        return True
    if _has_proper_noun(text):
        return True
    return False


def is_low_salience_aggressive(text: str) -> bool:
    """Aggressive: word_count <= 4 OR backchannel first-token OR pure backchannel regex.
    AND not counter-marked by high-salience features."""
    wc = _word_count(text)
    ft = _first_token_lower(text)
    fires = (
        wc <= 4
        or ft in BACKCHANNEL_FIRST_TOKENS
        or bool(PURE_BACKCHANNEL_RE.match(text))
    )
    if not fires:
        return False
    # If it has numbers, dates, proper nouns etc., keep it (still useful).
    if is_high_salience_regex(text):
        return False
    return True


def is_low_salience_conservative(text: str) -> bool:
    """Conservative: word_count <= 3 AND backchannel first-token. Not high-salient."""
    wc = _word_count(text)
    ft = _first_token_lower(text)
    if not (wc <= 3 and ft in BACKCHANNEL_FIRST_TOKENS):
        return False
    if is_high_salience_regex(text):
        return False
    return True


# ---------------------------------------------------------------------------
# LLM salience classifier (cached)
# ---------------------------------------------------------------------------
SALIENCE_PROMPT = (
    "You are labeling conversation turns for a retrieval index. "
    "For the turn below, answer YES if this turn contains information "
    "someone could later ask about (a fact, event, decision, name, plan, "
    "number, date, preference, or explanation). Answer NO if it is pure "
    "chitchat, a generic acknowledgement, filler, or a greeting with no "
    "factual content.\n\n"
    "Turn:\n{text}\n\n"
    "Respond with exactly one word: YES or NO."
)


class SalienceLLMCache:
    """Simple disk-cached YES/NO classifier using gpt-5-mini."""

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = CACHE_FILE
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)
        self._new_entries: dict[str, str] = {}
        self.client = OpenAI(timeout=30.0)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    def classify(self, text: str) -> str:
        """Return 'YES' or 'NO'. Cached."""
        key = self._key(text)
        if key in self._cache:
            return self._cache[key]
        prompt = SALIENCE_PROMPT.format(text=text[:2000])
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=10,
            )
            out = (resp.choices[0].message.content or "").strip().upper()
        except Exception as e:
            print(f"  LLM classify failed: {e}")
            out = "YES"  # fail-open
        if "NO" in out and "YES" not in out:
            label = "NO"
        else:
            label = "YES"
        self._cache[key] = label
        self._new_entries[key] = label
        return label

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------
Variant = Literal[
    "prune_regex_aggressive",
    "prune_regex_conservative",
    "prune_llm",
    "downweight_regex",
]


@dataclass
class SalienceMask:
    """Per-index labels for a store.

    low_salience[i] = True  means segment i is low-salience (candidate for prune/downweight).
    """
    variant: str
    low_salience: np.ndarray  # bool array of shape (N,)

    def pruned_count(self) -> int:
        return int(self.low_salience.sum())

    def pruned_fraction(self) -> float:
        return float(self.low_salience.mean()) if len(self.low_salience) else 0.0


def build_mask(
    store: SegmentStore,
    variant: Variant,
    llm_cache: SalienceLLMCache | None = None,
    verbose: bool = False,
    restrict_to_conv_ids: set[str] | None = None,
) -> SalienceMask:
    """Build a bool mask marking low-salience segments.

    restrict_to_conv_ids: if set, only classify/mark segments whose
    conversation_id is in this set. Other segments default to NOT low-salience.
    Useful for the LLM variant to avoid spending on unused conversations.
    """
    N = len(store.segments)
    mask = np.zeros(N, dtype=bool)

    def _is_target(idx: int) -> bool:
        if restrict_to_conv_ids is None:
            return True
        return store.segments[idx].conversation_id in restrict_to_conv_ids

    if variant == "prune_regex_aggressive":
        for i, seg in enumerate(store.segments):
            if _is_target(i) and is_low_salience_aggressive(seg.text):
                mask[i] = True
    elif variant == "prune_regex_conservative":
        for i, seg in enumerate(store.segments):
            if _is_target(i) and is_low_salience_conservative(seg.text):
                mask[i] = True
    elif variant == "downweight_regex":
        for i, seg in enumerate(store.segments):
            if _is_target(i) and is_low_salience_aggressive(seg.text):
                mask[i] = True
    elif variant == "prune_llm":
        if llm_cache is None:
            raise ValueError("prune_llm requires llm_cache")
        todo = [i for i in range(N) if _is_target(i)]
        for j, i in enumerate(todo):
            seg = store.segments[i]
            if not seg.text.strip():
                mask[i] = True
                continue
            label = llm_cache.classify(seg.text)
            if label == "NO":
                mask[i] = True
            if verbose and (j + 1) % 100 == 0:
                print(f"    LLM classify {j+1}/{len(todo)}", flush=True)
        llm_cache.save()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return SalienceMask(variant=variant, low_salience=mask)


# ---------------------------------------------------------------------------
# Pruned store
# ---------------------------------------------------------------------------
class PrunedSegmentStore(SegmentStore):
    """SegmentStore with a salience mask applied to retrieval scores.

    Drop modes: excluded from retrieval entirely (similarity = -1).
    Downweight mode: similarity scaled by `downweight_factor` (default 0.5).
    """

    def __init__(
        self,
        base: SegmentStore,
        mask: SalienceMask,
        mode: Literal["drop", "downweight"] = "drop",
        downweight_factor: float = 0.5,
    ):
        # Share all state with base to avoid re-loading
        self.embeddings = base.embeddings
        self.conversation_ids = base.conversation_ids
        self.turn_ids = base.turn_ids
        self.roles = base.roles
        self.texts = base.texts
        self.normalized_embeddings = base.normalized_embeddings
        self.segments = base.segments
        self._turn_index = base._turn_index

        self.mask = mask
        self.mode = mode
        self.downweight_factor = downweight_factor

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        conversation_id: str | None = None,
        exclude_indices: set[int] | None = None,
    ) -> RetrievalResult:
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
        sims = self.normalized_embeddings @ q

        if self.mode == "downweight":
            sims = np.where(
                self.mask.low_salience, sims * self.downweight_factor, sims
            )
        elif self.mode == "drop":
            sims = np.where(self.mask.low_salience, -1.0, sims)

        if conversation_id is not None:
            conv_mask = self.conversation_ids == conversation_id
            sims = np.where(conv_mask, sims, -1.0)

        if exclude_indices:
            for idx in exclude_indices:
                sims[idx] = -1.0

        top = np.argsort(sims)[::-1][:top_k]
        segs = [self.segments[i] for i in top if sims[i] > -1.0]
        scores = [float(sims[i]) for i in top if sims[i] > -1.0]
        return RetrievalResult(segments=segs, scores=scores)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------
def false_prune_stats(
    store: SegmentStore,
    mask: SalienceMask,
    questions: list[dict],
) -> dict:
    """Count how many gold-turn segments get marked low-salience.

    Gold turns: question['source_chat_ids'] for matching conversation_id.
    Returns: {'gold_total', 'gold_pruned', 'false_prune_rate', 'pruned_gold_samples'}
    """
    # Build (conv_id, turn_id) -> index
    t_idx: dict[tuple[str, int], int] = {}
    for i, seg in enumerate(store.segments):
        t_idx[(seg.conversation_id, seg.turn_id)] = i

    gold_pairs: set[tuple[str, int]] = set()
    for q in questions:
        cid = q["conversation_id"]
        for tid in q.get("source_chat_ids", []):
            gold_pairs.add((cid, int(tid)))

    gold_total = 0
    gold_pruned = 0
    pruned_samples: list[dict] = []
    for pair in gold_pairs:
        idx = t_idx.get(pair)
        if idx is None:
            continue
        gold_total += 1
        if mask.low_salience[idx]:
            gold_pruned += 1
            if len(pruned_samples) < 10:
                seg = store.segments[idx]
                pruned_samples.append({
                    "conv": seg.conversation_id,
                    "turn": seg.turn_id,
                    "role": seg.role,
                    "text": seg.text[:200],
                })

    return {
        "gold_total": gold_total,
        "gold_pruned": gold_pruned,
        "false_prune_rate": (gold_pruned / gold_total) if gold_total else 0.0,
        "pruned_gold_samples": pruned_samples,
    }
