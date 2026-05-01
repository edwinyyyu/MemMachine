"""Query clause decomposition: mechanical split of multi-part queries.

Motivation
----------
Some LoCoMo/synthetic queries are multi-part. Example (synthetic):
  "What is Bob allergic to? Please include any updates or corrections
   mentioned later in the conversation."
These contain multiple semantic targets. A single retrieval probe conflates
them; per-clause retrieval might surface distinct gold content per sub-intent.

Different from LLM decomposition (context_tree_v2 failed because an LLM
couldn't meaningfully decompose without losing intent). Mechanical
decomposition is cheap and preserves literal clause tokens.

Pipeline
--------
split_query_into_clauses(q, max_clauses)
  Split on:
    - sentence boundaries: "? " "! " and ". " (period + space, not decimals)
    - semicolons: "; "
    - sentence-level conjunctions: "... and ...", " or ", " as well as ",
      " including ", " plus " (only when the resulting pieces each have >= 4
      non-stopword tokens)
    - dangling "Please ..." / "Include ..." / "Also ..." / "organized by" /
      "with costs" trailing clauses
  Filter out clauses < 4 words or that are mostly stopwords.
  If only 1 clause results, fall back to raw query.

Variants (classes below)
------------------------
- ClauseCosineN2      — split into up to 2 clauses, cosine retrieval each
- ClauseCosineN3      — up to 3 clauses
- ClauseV2fN2         — v2f per clause (expensive)
- ClausePlusV2f       — v2f on full query AND cosine on each clause;
                        merge stacked with v2f first.

Aggregation
-----------
Segments are deduped by `index` (parent turn). Rank by max score across
clauses. For cosine-only variants we cap total returned to K=50.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import (
    V2F_PROMPT,
    BestshotBase,
    BestshotResult,
    _format_segments,
    _parse_cues,
)
from openai import OpenAI

# ---------------------------------------------------------------------------
# Dedicated caches
# ---------------------------------------------------------------------------
_CLAUSE_EMB_FILE = CACHE_DIR / "clause_embedding_cache.json"
_CLAUSE_LLM_FILE = CACHE_DIR / "clause_llm_cache.json"

# Warm-start: read shared caches but write only to dedicated files.
# IMPORTANT: merge order matches `antipara_cue_gen._SHARED_LLM_READ` so that
# our v2f LLM calls hit the same cached responses as `MetaV2fDedicated` — this
# keeps the clause-vs-meta comparison apples-to-apples on LoCoMo (where no
# queries actually split).
_SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "fewshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "clause_embedding_cache.json",
)
_SHARED_LLM_READ = (
    "llm_cache.json",
    "arch_llm_cache.json",
    "agent_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "synth_test_llm_cache.json",
    "bestshot_llm_cache.json",
    "fewshot_llm_cache.json",
    "antipara_llm_cache.json",
    "clause_llm_cache.json",
)


class ClauseEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in _SHARED_EMB_READ:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        self.cache_file = _CLAUSE_EMB_FILE
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        import os

        tmp = self.cache_file.parent / (self.cache_file.name + f".tmp.{os.getpid()}")
        try:
            with open(tmp, "w") as f:
                json.dump(existing, f)
            os.replace(tmp, self.cache_file)
        except FileNotFoundError:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return
        self._new_entries = {}


class ClauseLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in _SHARED_LLM_READ:
            p = self.cache_dir / name
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for k, v in data.items():
                if v:
                    self._cache[k] = v
        self.cache_file = _CLAUSE_LLM_FILE
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(self._new_entries)
        import os

        tmp = self.cache_file.parent / (self.cache_file.name + f".tmp.{os.getpid()}")
        try:
            with open(tmp, "w") as f:
                json.dump(existing, f)
            os.replace(tmp, self.cache_file)
        except FileNotFoundError:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Mechanical splitter
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    ["a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did", "will", "would", "can", "could", "should", "may", "might", "must", "shall", "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs", "this", "that", "these", "those", "what", "who", "whom", "whose", "which", "how", "why", "when", "where", "if", "then", "else", "not", "no", "yes", "as", "so", "than", "too", "very", "just", "also", "still", "please", "include", "any", "all", "every", "each", "every", "other", "all", "any", "some", "most", "more", "less"]
)

# Conjunction tokens to split on at sentence level. Ordered by specificity.
_CONJ_PATTERNS = [
    r"\s+as well as\s+",
    r"\s+including\s+",
    r"\s+along with\s+",
    r"\s+plus\s+",
    r"\s+and also\s+",
    r"\s+and\s+",
    r"\s+or\s+",
]


def _word_count(text: str) -> int:
    return len([t for t in re.findall(r"[A-Za-z0-9']+", text) if t])


def _content_word_count(text: str) -> int:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return sum(1 for w in words if w not in _STOPWORDS and len(w) > 1)


def _trivial(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if _word_count(t) < 4:
        return True
    if _content_word_count(t) < 2:
        return True
    return False


_ABBREV = frozenset(
    [
        "dr",
        "mr",
        "mrs",
        "ms",
        "st",
        "sr",
        "jr",
        "mt",
        "prof",
        "hon",
        "rev",
        "col",
        "gen",
        "sgt",
        "capt",
        "lt",
        "cpl",
        "vs",
        "etc",
        "inc",
        "ltd",
        "co",
        "corp",
        "no",
        "ave",
        "blvd",
        "rd",
        "p",  # p.m., p.s.
        "a",  # a.m.
        "e",  # e.g.
        "i",  # i.e.
        "u",
    ]
)


def _split_sentences(q: str) -> list[str]:
    """Split on sentence terminators followed by whitespace.

    Rules:
      - "? " and "! " always split.
      - ". " splits only if the preceding token is NOT a known abbreviation,
        AND the following char is uppercase. This avoids "Dr. Seuss".
    """
    q = re.sub(r"\s+", " ", q).strip()

    # Always split on "? " or "! "
    pieces: list[str] = re.split(r"(?<=[?!])\s+", q)

    # Now further split on ". " with abbreviation protection
    out: list[str] = []
    for piece in pieces:
        # Scan positions of ". <Upper>"
        cursor = 0
        parts: list[str] = []
        for m in re.finditer(r"\.\s+(?=[A-Z])", piece):
            end = m.start()  # position of the period
            # Get the token preceding the period
            prev = piece[:end]
            tok_m = re.search(r"([A-Za-z]+)$", prev)
            tok = tok_m.group(1).lower() if tok_m else ""
            if tok in _ABBREV:
                continue  # don't split here
            # Accept split
            parts.append(piece[cursor : end + 1])  # include the period
            cursor = m.end()
        parts.append(piece[cursor:])
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    return out


def _split_semicolon(parts: list[str]) -> list[str]:
    out: list[str] = []
    for p in parts:
        if ";" in p:
            for sub in p.split(";"):
                sub = sub.strip()
                if sub:
                    out.append(sub)
        else:
            out.append(p)
    return out


def _split_conjunction(parts: list[str]) -> list[str]:
    """Split on top-level conjunctions. Only split when BOTH sides have
    enough content tokens. This prevents breaking noun-phrases like
    "salt and pepper" or "his birthday and what are his interests" where
    the second side IS meaningful.
    """
    out: list[str] = []
    for p in parts:
        segments = [p]
        for pat in _CONJ_PATTERNS:
            new_segments: list[str] = []
            for seg in segments:
                # Split at first matching conjunction only — avoid fragmenting.
                m = re.search(pat, seg, re.IGNORECASE)
                if not m:
                    new_segments.append(seg)
                    continue
                left = seg[: m.start()].strip(" ,.;:")
                right = seg[m.end() :].strip(" ,.;:")
                if (
                    _word_count(left) >= 4
                    and _word_count(right) >= 4
                    and _content_word_count(left) >= 2
                    and _content_word_count(right) >= 2
                ):
                    new_segments.append(left)
                    new_segments.append(right)
                else:
                    new_segments.append(seg)
            segments = new_segments
        out.extend(segments)
    return out


def _split_list_commas(parts: list[str]) -> list[str]:
    """If a clause has multiple commas and no sentence conjunction
    structure, optionally split on commas. Conservative — only fire when
    the clause looks like a list enumeration (3+ commas). We do NOT split
    otherwise to avoid destroying phrases."""
    out: list[str] = []
    for p in parts:
        commas = p.count(",")
        if commas < 3:
            out.append(p)
            continue
        pieces = [x.strip(" ,.;:") for x in p.split(",")]
        pieces = [x for x in pieces if _word_count(x) >= 3]
        if len(pieces) >= 3 and all(_content_word_count(x) >= 2 for x in pieces):
            out.extend(pieces)
        else:
            out.append(p)
    return out


def split_query_into_clauses(query: str, max_clauses: int = 3) -> list[str]:
    """Mechanically split a query into clauses. Returns a list of
    non-trivial clauses. If fewer than 2 clauses result, returns
    [query] unchanged."""
    q = query.strip()
    if not q:
        return [query]

    # Stage 1: sentence split
    parts = _split_sentences(q)
    # Stage 2: semicolon
    parts = _split_semicolon(parts)
    # Stage 3: conjunctions (once per part)
    parts = _split_conjunction(parts)
    # Stage 4: list-comma split
    parts = _split_list_commas(parts)

    # Filter trivial
    clauses = [p.strip(" ,.;:") for p in parts]
    clauses = [c for c in clauses if not _trivial(c)]

    # Dedupe (preserve order)
    seen: set[str] = set()
    dedup: list[str] = []
    for c in clauses:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)

    if len(dedup) < 2:
        return [query]

    # Cap to max_clauses, preserving the first one (usually most salient).
    return dedup[:max_clauses]


# ---------------------------------------------------------------------------
# Base arch
# ---------------------------------------------------------------------------
class _ClauseBase(BestshotBase):
    """Base class using dedicated caches. Subclasses implement retrieve()."""

    arch_name: str = "clause_base"
    max_clauses: int = 2
    # Per-clause top-M for cosine retrieval. We use K=20 as our smallest
    # budget, K=50 as the largest; the eval reads top-K from what we return
    # and backfills cosine. For union variants, retrieve top 25-30 per clause
    # to keep ranking sensible after dedup.
    per_clause_top_m: int = 30

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = ClauseEmbeddingCache()
        self.llm_cache = ClauseLLMCache()

    def _cosine_per_clause(
        self, clauses: Iterable[str], conversation_id: str, top_m: int
    ) -> tuple[list[Segment], list[float], dict]:
        """For each clause, retrieve top-m. Union, dedupe by index, rank by
        max score across clauses. Returns (segments_ordered, scores, meta)."""
        per_parent_score: dict[int, float] = {}
        per_parent_clause: dict[int, str] = {}
        for c in clauses:
            emb = self.embed_text(c)
            res = self.store.search(emb, top_k=top_m, conversation_id=conversation_id)
            for seg, sc in zip(res.segments, res.scores):
                cur = per_parent_score.get(seg.index)
                if cur is None or sc > cur:
                    per_parent_score[seg.index] = sc
                    per_parent_clause[seg.index] = c
        # Rank by max score desc
        ranked = sorted(per_parent_score.items(), key=lambda t: t[1], reverse=True)
        segments = [self.store.segments[i] for i, _ in ranked]
        scores = [s for _, s in ranked]
        meta = {"best_clause_per_index": per_parent_clause}
        return segments, scores, meta

    def _run_v2f(
        self, query_text: str, conversation_id: str
    ) -> tuple[list[Segment], dict]:
        """Run v2f on the given query_text. Stacked: hop0 top-10, then
        cue1 top-10, cue2 top-10 (excluding overlaps)."""
        query_emb = self.embed_text(query_text)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments: list[Segment] = list(hop0.segments)
        exclude: set[int] = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(all_segments)
        )
        prompt = V2F_PROMPT.format(question=query_text, context_section=context_section)
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

        for cue in cues:
            cue_emb = self.embed_text(cue)
            res = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in res.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
        return all_segments, {"output": output, "cues": cues}


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
class ClauseCosineN2(_ClauseBase):
    """Split into up to 2 clauses, cosine per-clause, union + rank by max."""

    arch_name = "clause_cosine_n2"
    max_clauses = 2

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        clauses = split_query_into_clauses(question, max_clauses=self.max_clauses)

        if len(clauses) <= 1:
            # No split — fallback to raw cosine top-50
            query_emb = self.embed_text(question)
            res = self.store.search(
                query_emb, top_k=50, conversation_id=conversation_id
            )
            return BestshotResult(
                segments=list(res.segments),
                metadata={
                    "name": self.arch_name,
                    "clauses": clauses,
                    "n_clauses": 1,
                    "split": False,
                },
            )

        segments, scores, meta = self._cosine_per_clause(
            clauses, conversation_id, top_m=self.per_clause_top_m
        )
        return BestshotResult(
            segments=segments[:50],
            metadata={
                "name": self.arch_name,
                "clauses": clauses,
                "n_clauses": len(clauses),
                "split": True,
                "top_scores": [round(s, 4) for s in scores[:10]],
            },
        )


class ClauseCosineN3(ClauseCosineN2):
    arch_name = "clause_cosine_n3"
    max_clauses = 3


class ClauseV2fN2(_ClauseBase):
    """V2f per clause (expensive: ~1 LLM call per clause). Stacked merge:
    clause-1 v2f first, then clause-2 v2f items novel to clause-1."""

    arch_name = "clause_v2f_n2"
    max_clauses = 2

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        clauses = split_query_into_clauses(question, max_clauses=self.max_clauses)
        if len(clauses) <= 1:
            segs, meta = self._run_v2f(question, conversation_id)
            return BestshotResult(
                segments=segs,
                metadata={
                    "name": self.arch_name,
                    "clauses": clauses,
                    "n_clauses": 1,
                    "split": False,
                    "v2f_cues": meta.get("cues", []),
                },
            )

        # Run v2f per clause, stacked merge
        merged: list[Segment] = []
        seen: set[int] = set()
        per_clause_cues: list[list[str]] = []
        for c in clauses:
            segs, meta = self._run_v2f(c, conversation_id)
            per_clause_cues.append(meta.get("cues", []))
            for s in segs:
                if s.index not in seen:
                    merged.append(s)
                    seen.add(s.index)
        return BestshotResult(
            segments=merged[:50],
            metadata={
                "name": self.arch_name,
                "clauses": clauses,
                "n_clauses": len(clauses),
                "split": True,
                "v2f_cues_per_clause": per_clause_cues,
            },
        )


class ClausePlusV2f(_ClauseBase):
    """V2f on full query AND cosine on each clause; merge with v2f first,
    then novel clause-cosine hits by max-score order. Only adds per-clause
    retrievals; zero extra LLM calls beyond v2f baseline.
    """

    arch_name = "clause_plus_v2f"
    max_clauses = 2

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        v2f_segs, v2f_meta = self._run_v2f(question, conversation_id)
        v2f_indices = {s.index for s in v2f_segs}

        clauses = split_query_into_clauses(question, max_clauses=self.max_clauses)
        if len(clauses) <= 1:
            # No split — v2f only
            return BestshotResult(
                segments=v2f_segs,
                metadata={
                    "name": self.arch_name,
                    "clauses": clauses,
                    "n_clauses": 1,
                    "split": False,
                    "v2f_cues": v2f_meta.get("cues", []),
                },
            )

        # Cosine per-clause retrieval
        clause_segs, clause_scores, meta = self._cosine_per_clause(
            clauses, conversation_id, top_m=self.per_clause_top_m
        )
        # Append clause hits novel to v2f, in max-score order
        merged = list(v2f_segs)
        for seg in clause_segs:
            if seg.index not in v2f_indices:
                merged.append(seg)
                v2f_indices.add(seg.index)
        return BestshotResult(
            segments=merged[:50],
            metadata={
                "name": self.arch_name,
                "clauses": clauses,
                "n_clauses": len(clauses),
                "split": True,
                "v2f_cues": v2f_meta.get("cues", []),
                "top_clause_scores": [round(s, 4) for s in clause_scores[:10]],
                "n_v2f_segments": len(v2f_segs),
                "n_clause_segments": len(clause_segs),
                "n_clause_novel": len(merged) - len(v2f_segs),
            },
        )


ARCH_CLASSES: dict[str, type] = {
    "clause_cosine_n2": ClauseCosineN2,
    "clause_cosine_n3": ClauseCosineN3,
    "clause_v2f_n2": ClauseV2fN2,
    "clause_plus_v2f": ClausePlusV2f,
}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "What is Bob allergic to? Please include any updates or corrections "
        "mentioned later in the conversation.",
        "I need to buy a birthday gift for Bob. When is his birthday and what "
        "are his interests?",
        "What is the current status of Project Phoenix? Include any milestones "
        "reached and upcoming work.",
        "What did Caroline research?",
        "List ALL dietary restrictions and food preferences for every guest at "
        "the Saturday dinner party, including any updates or corrections.",
        "When did Caroline meet up with her friends, family, and mentors?",
        "What are all of the user's current medications, including dosages and "
        "what they're for? Include any recent changes.",
    ]
    for q in test_queries:
        cs = split_query_into_clauses(q, max_clauses=3)
        print(f"Q: {q}")
        for c in cs:
            print(f"  - {c}")
        print()
