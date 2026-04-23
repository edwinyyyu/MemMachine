"""Dialogue-act tagging + act-routed retrieval.

At ingestion time, an LLM classifier labels each conversation turn by its
speech act (DECISION, COMMITMENT, RETRACTION, UNRESOLVED, CLARIFICATION,
STATEMENT). Non-STATEMENT acts populate a SEPARATE per-act vector store.

At query time, the query is routed to relevant acts (via keyword rules or an
optional LLM call). Top-M hits from the act-specific index (above a
min_score floor) are merged with the main retrieval via the same
always-top-M / additive-bonus pattern that `critical_info_store` validated.

This module provides:
  - DialogueActTagger: parallel LLM classifier with dedicated cache
  - ActIndex: a per-act vector store (reuses original-text embeddings)
  - route_query_keywords: keyword-based query -> act-set routing
  - merge_additive_bonus / merge_always_top_m: re-exposed from critical_info_store
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np
from openai import OpenAI

from associative_recall import CACHE_DIR, Segment, SegmentStore

# Re-export merge helpers so eval code only imports from this module.
from critical_info_store import merge_additive_bonus, merge_always_top_m  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACT_LABELS = (
    "DECISION",
    "COMMITMENT",
    "RETRACTION",
    "UNRESOLVED",
    "CLARIFICATION",
    "STATEMENT",
)

ACT_TAG_PROMPT = """\
Classify this conversation turn as ONE of the following speech acts. \
Output only the label.

- DECISION: states a decision made ("we'll use X", "decided to...")
- COMMITMENT: promises an action with an actor ("I'll handle X", "she'll \
send it by Tuesday")
- RETRACTION: corrects or overrides an earlier statement ("actually, no", \
"scratch that", "I was wrong")
- UNRESOLVED: raises a question or flags uncertainty ("not sure if...", \
"need to check...", "TBD")
- CLARIFICATION: asks a question or requests info
- STATEMENT: everything else (narrative, chitchat, factual sharing)

Turn ({role}): {text}

Output exactly one label: DECISION, COMMITMENT, RETRACTION, UNRESOLVED, \
CLARIFICATION, or STATEMENT."""


QUERY_ROUTE_PROMPT = """\
You will route a retrieval query to dialogue-act sub-indices. The candidate \
acts are: DECISION, COMMITMENT, RETRACTION, UNRESOLVED.

- DECISION: queries about choices made, what was selected/resolved.
- COMMITMENT: queries about promises, who-will-do-what, follow-ups.
- RETRACTION: queries about contradictions, corrections, things that \
changed or were overridden.
- UNRESOLVED: queries about open questions, TBDs, things that were never \
settled, unfinished business.

Query: {question}

Output ONE line with the comma-separated set of relevant acts, or NONE if no \
act is a good match. Examples of valid outputs:
COMMITMENT,UNRESOLVED
DECISION
RETRACTION
NONE"""


# ---------------------------------------------------------------------------
# Dedicated LLM cache (avoids colliding with other agents / critical-info cache)
# ---------------------------------------------------------------------------
class DialactLLMCache:
    """Dialact-specific LLM cache. Reads dialact_llm_cache.json and writes back."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "dialact_llm_cache.json"
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except Exception:
                # raw-decode recovery
                try:
                    with open(self.cache_file, "rb") as f:
                        raw = f.read().decode("utf-8", errors="replace")
                    self._cache, _ = json.JSONDecoder().raw_decode(raw)
                except Exception:
                    self._cache = {}
        self._new_entries: dict[str, str] = {}
        self._save_lock = Lock()

    def _key(self, model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(self._key(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        k = self._key(model, prompt)
        self._cache[k] = response
        self._new_entries[k] = response

    def save(self) -> None:
        with self._save_lock:
            if not self._new_entries:
                return
            existing = {}
            if self.cache_file.exists():
                try:
                    with open(self.cache_file) as f:
                        existing = json.load(f)
                except Exception:
                    try:
                        with open(self.cache_file, "rb") as f:
                            raw = f.read().decode("utf-8", errors="replace")
                        existing, _ = json.JSONDecoder().raw_decode(raw)
                    except Exception:
                        existing = {}
            existing.update(self._new_entries)
            import os
            tmp = self.cache_file.with_suffix(f".json.{os.getpid()}.tmp")
            with open(tmp, "w") as f:
                json.dump(existing, f)
            tmp.replace(self.cache_file)
            self._new_entries.clear()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class TurnActLabel:
    parent_index: int
    conversation_id: str
    turn_id: int
    role: str
    text: str
    raw_response: str
    label: str  # one of ACT_LABELS, or "UNKNOWN"


# ---------------------------------------------------------------------------
# Prompt & response handling
# ---------------------------------------------------------------------------
def build_tag_prompt(role: str, text: str) -> str:
    t = text[:1200]
    return ACT_TAG_PROMPT.format(role=role, text=t)


_LABEL_SET = set(ACT_LABELS)


def parse_tag_response(response: str) -> str:
    """Extract a single label from an LLM response. Returns one of ACT_LABELS
    or 'UNKNOWN' on parse failure."""
    text = (response or "").strip().strip("'\"`")
    if not text:
        return "UNKNOWN"
    # Check direct match (ignore case, strip trailing punctuation / periods)
    first_tok = text.split("\n")[0].strip().strip(".,:;!?\"'`").upper()
    if first_tok in _LABEL_SET:
        return first_tok
    # Search for any label appearing as a standalone word
    up = text.upper()
    for lbl in ACT_LABELS:
        if re.search(rf"\b{lbl}\b", up):
            return lbl
    return "UNKNOWN"


def build_route_prompt(question: str) -> str:
    return QUERY_ROUTE_PROMPT.format(question=question.strip()[:800])


def parse_route_response(response: str) -> set[str]:
    """Parse LLM routing response -> set of act labels (subset of
    {DECISION, COMMITMENT, RETRACTION, UNRESOLVED})."""
    text = (response or "").strip().upper()
    if not text:
        return set()
    # First non-empty line
    for raw in text.split("\n"):
        line = raw.strip().strip(".,:;!?\"'`")
        if not line:
            continue
        if line == "NONE":
            return set()
        out: set[str] = set()
        for tok in re.split(r"[,\s]+", line):
            tok = tok.strip().strip(".,:;!?\"'`")
            if tok in {"DECISION", "COMMITMENT", "RETRACTION", "UNRESOLVED"}:
                out.add(tok)
        return out
    return set()


# ---------------------------------------------------------------------------
# LLM tagger
# ---------------------------------------------------------------------------
class DialogueActTagger:
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "gpt-5-mini",
        max_workers: int = 8,
        cache: DialactLLMCache | None = None,
    ):
        self.client = client or OpenAI(timeout=60.0)
        self.model = model
        self.max_workers = max_workers
        self.cache = cache or DialactLLMCache()
        self._cache_lock = Lock()
        self._counter_lock = Lock()
        self.n_cached = 0
        self.n_uncached = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _cache_key(self, prompt: str) -> str:
        # Namespace so a cache shared with other callers doesn't collide.
        return f"[dialogue_act/v1]\n" + prompt

    def call_one(self, role: str, text: str) -> str:
        prompt = build_tag_prompt(role, text)
        ck = self._cache_key(prompt)

        with self._cache_lock:
            cached = self.cache.get(self.model, ck)
        if cached is not None:
            with self._counter_lock:
                self.n_cached += 1
            return cached

        raw = ""
        pt = 0
        ct = 0
        last_err: Exception | None = None
        # gpt-5-mini uses reasoning tokens that eat the budget before the
        # label is emitted; use 800+ to match critical_info_store.
        for tok_budget in (800, 1600, 3200):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=tok_budget,
                )
                raw = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if "max_tokens" in msg or "output limit" in msg:
                    continue
                if tok_budget == 800:
                    continue
                raise
        if last_err is not None:
            raw = ""

        with self._cache_lock:
            self.cache.put(self.model, ck, raw)
        with self._counter_lock:
            self.n_uncached += 1
            self.total_prompt_tokens += int(pt)
            self.total_completion_tokens += int(ct)
        return raw

    def route_query(self, question: str) -> tuple[set[str], str]:
        """LLM-based query -> act-set routing. Returns (act_set, raw)."""
        prompt = build_route_prompt(question)
        ck = f"[dialogue_act_route/v1]\n" + prompt
        with self._cache_lock:
            cached = self.cache.get(self.model, ck)
        if cached is not None:
            with self._counter_lock:
                self.n_cached += 1
            return parse_route_response(cached), cached

        raw = ""
        pt = 0
        ct = 0
        for tok_budget in (800, 1600, 3200):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=tok_budget,
                )
                raw = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                break
            except Exception:
                if tok_budget == 800:
                    continue
                raise
        with self._cache_lock:
            self.cache.put(self.model, ck, raw)
        with self._counter_lock:
            self.n_uncached += 1
            self.total_prompt_tokens += int(pt)
            self.total_completion_tokens += int(ct)
        return parse_route_response(raw), raw

    def save(self) -> None:
        self.cache.save()


def tag_turns(
    tagger: DialogueActTagger,
    segments: Iterable[Segment],
    log_every: int = 200,
) -> list[TurnActLabel]:
    segs = list(segments)
    n = len(segs)
    out: list[TurnActLabel | None] = [None] * n
    t0 = time.time()
    done = [0]
    last_save = [t0]

    def _do(i: int) -> tuple[int, TurnActLabel]:
        s = segs[i]
        raw = tagger.call_one(s.role, s.text)
        label = parse_tag_response(raw)
        lab = TurnActLabel(
            parent_index=s.index,
            conversation_id=s.conversation_id,
            turn_id=s.turn_id,
            role=s.role,
            text=s.text,
            raw_response=raw,
            label=label,
        )
        done[0] += 1
        if done[0] % log_every == 0:
            el = time.time() - t0
            rate = done[0] / max(el, 1e-6)
            eta = (n - done[0]) / max(rate, 1e-6)
            print(
                f"  [{done[0]}/{n}] cached={tagger.n_cached} "
                f"uncached={tagger.n_uncached} "
                f"rate={rate:.1f}/s eta={eta:.0f}s",
                flush=True,
            )
            if time.time() - last_save[0] > 30:
                tagger.save()
                last_save[0] = time.time()
        return i, lab

    with ThreadPoolExecutor(max_workers=tagger.max_workers) as ex:
        futures = [ex.submit(_do, i) for i in range(n)]
        for f in as_completed(futures):
            i, lab = f.result()
            out[i] = lab

    tagger.save()
    return [x for x in out if x is not None]


# ---------------------------------------------------------------------------
# Per-act separate index
# ---------------------------------------------------------------------------
class ActIndex:
    """A vector store restricted to turns with a specific dialogue-act label.

    Shares the underlying normalized embeddings with the main SegmentStore
    (we look up by parent_index) so no re-embedding is required.
    """

    def __init__(
        self,
        base: SegmentStore,
        parent_indices: list[int],
    ):
        self._base = base
        if len(parent_indices) == 0:
            dim = base.normalized_embeddings.shape[1]
            self.act_normalized = np.zeros((0, dim), dtype=np.float32)
            self.act_parent_index = np.zeros(0, dtype=np.int64)
            self.act_conversation_ids = np.zeros(0, dtype=object)
        else:
            idx = np.array(parent_indices, dtype=np.int64)
            self.act_normalized = base.normalized_embeddings[idx].astype(np.float32)
            self.act_parent_index = idx
            convs = [base.segments[i].conversation_id for i in parent_indices]
            self.act_conversation_ids = np.array(convs, dtype=object)

    def search_per_parent(
        self,
        query_embedding: np.ndarray,
        top_m: int,
        conversation_id: str,
        min_score: float = -1.0,
    ) -> list[tuple[int, float, Segment]]:
        """Return up to top_m (parent_index, score, Segment) tuples — one per
        parent — scored as cosine similarity against the turn's own embedding.

        Results are filtered to `conversation_id` and to scores >= min_score.
        Sorted descending by score.
        """
        if self.act_normalized.shape[0] == 0:
            return []
        q = query_embedding.astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-10)
        sims = self.act_normalized @ q  # (N_act,)

        mask = self.act_conversation_ids == conversation_id
        if not np.any(mask):
            return []
        sims = np.where(mask, sims, -np.inf)

        order = np.argsort(sims)[::-1]
        out: list[tuple[int, float, Segment]] = []
        for j in order:
            sc = float(sims[int(j)])
            if sc == -np.inf or sc < min_score:
                break
            parent_idx = int(self.act_parent_index[int(j)])
            out.append((parent_idx, sc, self._base.segments[parent_idx]))
            if len(out) >= top_m:
                break
        return out


def build_act_indices(
    base: SegmentStore,
    labels: Iterable[TurnActLabel],
    target_acts: Iterable[str] = ("DECISION", "COMMITMENT", "RETRACTION", "UNRESOLVED"),
) -> dict[str, ActIndex]:
    """Group labels by act and build a per-act index. Ignores target_acts that
    have zero members."""
    by_act: dict[str, list[int]] = {a: [] for a in target_acts}
    for lab in labels:
        if lab.label in by_act:
            by_act[lab.label].append(lab.parent_index)
    return {a: ActIndex(base, parents) for a, parents in by_act.items()}


# ---------------------------------------------------------------------------
# Query-to-act routing
# ---------------------------------------------------------------------------

# Routing rules: each pattern (compiled case-insensitively) maps to a set of
# acts to enable.
_ROUTE_RULES: list[tuple[re.Pattern[str], set[str]]] = [
    # Unfinished / commitment-style questions
    (re.compile(r"\b(complete[ds]?|never|didn'?t|did not|haven'?t|promise|follow[- ]?up|unfinished|pending|still need|yet to)\b", re.I),
     {"COMMITMENT", "UNRESOLVED"}),
    # Decision-like
    (re.compile(r"\b(decide[ds]?|decision|choose|chose|chosen|pick(ed)?|resolv(e|ed|ing)|settle[ds]?|pick[- ]?ed)\b", re.I),
     {"DECISION"}),
    # Correction / retraction
    (re.compile(r"\b(correct(ed)?|wrong|mistak(e|en)|updat(e|ed|ing)|actually|revers(e|ed)|chang(e|ed)|switch(ed)?|instead)\b", re.I),
     {"RETRACTION"}),
    # Contradiction / inconsistency
    (re.compile(r"\b(contradict\w*|inconsisten\w*|mismatch\w*|conflict\w*|disagree\w*)\b", re.I),
     {"RETRACTION", "DECISION"}),
    # Explicit question-hunting: "still need to", "open question", "unanswered"
    (re.compile(r"\b(open question|unanswered|unresolved|TBD|pending|waiting)\b", re.I),
     {"UNRESOLVED"}),
]


def route_query_keywords(question: str) -> set[str]:
    """Keyword-based mapping from question text -> set of acts to search.

    Returns an empty set if no rule fires (means: fall back to pure v2f).
    """
    out: set[str] = set()
    for pat, acts in _ROUTE_RULES:
        if pat.search(question):
            out |= acts
    return out


# ---------------------------------------------------------------------------
# Combining per-act retrievals into one ranked list
# ---------------------------------------------------------------------------
def combine_act_hits(
    act_hits: dict[str, list[tuple[int, float, Segment]]],
    top_m: int,
) -> list[tuple[int, float, Segment]]:
    """Union hits from several acts, taking max score per parent, then take
    top_m by score."""
    best: dict[int, tuple[float, Segment]] = {}
    for hits in act_hits.values():
        for parent_idx, sc, seg in hits:
            cur = best.get(parent_idx)
            if cur is None or sc > cur[0]:
                best[parent_idx] = (sc, seg)
    out = [(pi, sc, seg) for pi, (sc, seg) in best.items()]
    out.sort(key=lambda x: -x[1])
    return out[:top_m]


# ---------------------------------------------------------------------------
# Summary utilities
# ---------------------------------------------------------------------------
def act_distribution(labels: Iterable[TurnActLabel]) -> dict[str, int]:
    dist: dict[str, int] = {a: 0 for a in ACT_LABELS}
    dist["UNKNOWN"] = 0
    for lab in labels:
        dist[lab.label if lab.label in dist else "UNKNOWN"] += 1
    return dist
