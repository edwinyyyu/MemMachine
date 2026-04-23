"""Entity-mention exact-match index — non-cosine retrieval signal.

Motivation
----------
v2f relies on fuzzy cosine similarity. Gold turns that share a specific entity
(proper noun, ID, monetary figure) with the query can be missed when the rest
of the turn's wording differs from the question. This module builds an
inverted index {entity_token -> turn_indices} at ingest time and, at query
time, boosts cosine scores of turns that share an exact entity with the query.

This is NOT BM25. It is entity-specific — the index only contains named
entities (proper nouns, acronyms, IDs, numbers with units), not all tokens.

Pipeline
--------
Ingest-time (once per conversation, cached):
  - For each turn, extract entities using either a regex extractor (free) or
    an LLM extractor (cached). Build an inverted index per conversation.

Query-time (per question):
  - Extract entities from the query using the same extractor.
  - Build candidate turn set = union over query entities of index lookup.
  - For every turn in the conversation, compute cosine(question, turn).
  - final_score(turn) = cosine(turn) + beta * n_matches(turn)
    where n_matches is the number of *distinct* query entities mentioned in
    the turn (we cap by just 1 if we want indicator semantics — chosen via
    config, default: indicator, following plan).
  - Rank turns by final_score and return the list ordered by score (then
    fair-backfill with cosine supplies the rest).

Variants
--------
  entity_regex_b0.1      — regex extractor, beta=0.1, indicator bonus
  entity_regex_b0.2      — stronger bonus (beta=0.2)
  entity_llm_b0.1        — LLM extractor, beta=0.1
  entity_regex_plus_v2f  — regex + beta=0.1 + merge with v2f cue retrievals

Caches
------
Dedicated entity_mention_*_cache.json files to avoid corrupting other agents.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import (
    MODEL,
    BestshotBase,
    BestshotResult,
    V2F_PROMPT,
    _format_segments,
    _parse_cues,
)

# ---------------------------------------------------------------------------
# Dedicated caches — writes go here; reads union many shared caches.
# ---------------------------------------------------------------------------

_ENTITY_EMB_FILE = CACHE_DIR / "entity_mention_embedding_cache.json"
_ENTITY_LLM_FILE = CACHE_DIR / "entity_mention_llm_cache.json"
_TURN_ENTITIES_FILE = (
    Path(__file__).resolve().parent / "results" / "turn_entities.json"
)

# Best-effort list of shared caches to warm from. Writes stay in dedicated
# file so we don't corrupt other agents.
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
    "inv_query_embedding_cache.json",
    "anchor_embedding_cache.json",
    "alias_embedding_cache.json",
    "entity_mention_embedding_cache.json",
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
    "inv_query_llm_cache.json",
    "anchor_llm_cache.json",
    "alias_llm_cache.json",
    "entity_mention_llm_cache.json",
)


class EntityMentionEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _ENTITY_EMB_FILE
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
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


class EntityMentionLLMCache(LLMCache):
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
        self.cache_file = _ENTITY_LLM_FILE
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
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Regex-based entity extractor
# ---------------------------------------------------------------------------

# Words that commonly start sentences but aren't entities. Also include common
# conversational filler that, while capitalized mid-sentence (e.g. "I", "I'm"),
# is not an entity.
_STOPWORDS_COMMON = {
    # Function/content words that tend to appear capitalized
    "i", "i'm", "i've", "i'll", "i'd", "a", "an", "the", "this", "that",
    "these", "those", "it", "its", "it's", "he", "she", "they", "we", "you",
    "your", "yours", "my", "mine", "our", "ours", "me", "us", "them", "him",
    "her", "his", "hers", "their", "theirs", "who", "what", "when", "where",
    "why", "how", "which", "whose", "whom",
    # Common sentence starters
    "yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure", "well", "hmm",
    "oh", "ah", "um", "uh", "hey", "hi", "hello", "thanks", "thank",
    "wow", "cool", "nice", "great", "awesome", "amazing", "sweet",
    "lol", "haha", "lmao", "omg", "whoa",
    "what's", "that's", "there's", "here's", "who's", "where's",
    "how's", "when's", "let's", "you've", "you're", "you'd", "you'll",
    "we've", "we're", "we'd", "we'll", "they've", "they're", "they'd",
    "they'll", "gonna", "wanna", "gotta", "kinda", "sorta",
    "couldn't", "wouldn't", "shouldn't", "didn't", "doesn't", "don't",
    "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
    "sorry", "please", "maybe", "perhaps",
    # Modal/auxiliary
    "is", "are", "was", "were", "be", "been", "being", "am", "have", "has",
    "had", "do", "does", "did", "will", "would", "should", "could", "can",
    "may", "might", "must", "shall", "ought",
    # Conjunctions/prepositions
    "and", "or", "but", "if", "so", "for", "with", "about", "of", "to", "in",
    "on", "at", "by", "from", "as", "than", "then", "also", "too", "just",
    "very", "really", "quite", "actually", "still", "already", "yet", "even",
    # Common day words
    "today", "tomorrow", "yesterday", "now", "later", "soon", "recently",
    "sometimes", "always", "never", "often", "usually",
    # Common words in conversation
    "let", "let's", "get", "got", "going", "went", "come", "came", "see",
    "saw", "know", "knew", "think", "thought", "want", "wanted", "need",
    "needed", "like", "liked", "love", "loved", "feel", "felt", "make",
    "made", "take", "took", "give", "gave", "find", "found", "tell", "told",
    "say", "said", "says", "ask", "asked", "do", "does", "did", "done",
    # Short fillers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "here", "there", "everywhere", "anywhere", "somewhere",
    # Common nouns (not proper)
    "people", "person", "friend", "family", "work", "home", "day", "night",
    "morning", "afternoon", "evening", "week", "month", "year",
    "time", "thing", "things", "way", "ways", "life", "world",
    # Sentence-initial verbs/nouns that are commonly capitalized but not
    # entities. Keep this tight — false negatives here are fine (we
    # conservatively miss an entity); false positives (wrongly boosting
    # a turn that doesn't actually share an entity with the query) are
    # worse.
    "version", "cost", "cost.", "price", "amount", "type", "kind",
    "part", "number", "size", "level", "state", "status", "result",
    "great", "good", "bad", "nice", "awesome", "amazing", "cool",
    "happy", "sad", "glad", "right", "wrong", "true", "false",
    "yes.", "no.", "done", "first", "second", "third", "last", "next",
    "something", "anything", "nothing", "someone", "anyone", "everyone",
    "everything", "nowhere", "everybody", "anybody", "nobody",
}

# Days / months are proper-nouny but too generic to count as entities.
_STOPWORDS_GENERIC = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
}

_STOPWORDS = _STOPWORDS_COMMON | _STOPWORDS_GENERIC

# Acronyms (2+ caps optionally with -digits). Purposefully NOT matching IDs
# with lowercase hyphens here — rare in LoCoMo/synthetic.
_RE_ACRONYM = re.compile(r"\b[A-Z]{2,}(?:-\d+)?\b")

# Proper-noun sequences: runs of capitalized tokens, e.g., "Project Phoenix",
# "New York", "San Francisco Giants". Allows apostrophes and hyphens inside
# tokens. Does NOT drop sentence-initial matches — we handle that via stopwords
# plus the multi-word requirement for PP.
_RE_PROPER = re.compile(
    r"\b[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)*\b"
)

# Currency amounts: $12, $12.5, $12k, $12.5M, $12,000
_RE_CURRENCY = re.compile(r"\$\d+(?:[,.]?\d+)*(?:[kKmMbB])?")

# Percentages: 12%, 12.5%
_RE_PERCENT = re.compile(r"\b\d+(?:\.\d+)?%")

# Numbers with k/K/m/M suffix or standalone big: 1500, 200k, 3.2M
_RE_NUM_SUFFIX = re.compile(r"\b\d+(?:\.\d+)?[kKmMbB]\b")

# Short numeric dates / version numbers: 10/15, 10-15, v2.1, 3.1.4, 2023
_RE_DATE = re.compile(r"\b\d{1,4}[/\-]\d{1,4}(?:[/\-]\d{1,4})?\b")
_RE_VERSION = re.compile(r"\bv\d+(?:\.\d+){0,3}\b", re.IGNORECASE)

# Alphanumeric IDs: JIRA-4521, ABC-123, PR#42
_RE_ID = re.compile(r"\b[A-Z][A-Za-z]{1,}-\d{2,}\b")
_RE_HASH_ID = re.compile(r"\b[A-Za-z]+#\d+\b")

# Emails, phone numbers
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_RE_PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

# Four-digit year standalone: 2023, 1999
_RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# Roman-like ordinals/numerics after "Chapter", etc. — skip for simplicity.


def _normalize_entity(ent: str) -> str:
    """Canonicalize an entity string for indexing/matching.

    - Trim whitespace
    - Strip surrounding punctuation (.?!;:,"')
    - Lowercase (for case-insensitive matching)
    - Normalize internal whitespace
    """
    ent = ent.strip().strip(".?!;:,\"'()[]{}<>")
    ent = re.sub(r"\s+", " ", ent).lower()
    return ent


def extract_entities_regex(text: str) -> list[str]:
    """Regex-based entity extraction. Returns a de-duped list of entity
    strings (not normalized — callers normalize on insert/lookup).

    Strategy: gather high-confidence patterns (IDs, currency, versions,
    years) + capitalized proper-noun phrases. Drop proper-noun matches that
    overlap with already-captured spans, and drop single-word sentence-
    initial matches when the token is a common stopword."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    consumed_spans: list[tuple[int, int]] = []

    def _span_overlaps(span: tuple[int, int]) -> bool:
        for s, e in consumed_spans:
            if not (span[1] <= s or span[0] >= e):
                return True
        return False

    def _add(x: str, span: tuple[int, int] | None = None) -> None:
        x = x.strip()
        if not x:
            return
        norm = _normalize_entity(x)
        if not norm or len(norm) < 2:
            return
        if norm in seen:
            # Still record consumed span (for PROPER overlap avoidance)
            if span is not None:
                consumed_spans.append(span)
            return
        seen.add(norm)
        found.append(x)
        if span is not None:
            consumed_spans.append(span)

    # Order matters: capture longer/more-specific patterns first so shorter
    # generic matches don't consume them.
    for m in _RE_EMAIL.finditer(text):
        _add(m.group(0), (m.start(), m.end()))
    for m in _RE_PHONE.finditer(text):
        _add(m.group(0), (m.start(), m.end()))
    for m in _RE_CURRENCY.finditer(text):
        _add(m.group(0), (m.start(), m.end()))
    for m in _RE_PERCENT.finditer(text):
        _add(m.group(0), (m.start(), m.end()))
    for m in _RE_ID.finditer(text):
        _add(m.group(0), (m.start(), m.end()))
    for m in _RE_HASH_ID.finditer(text):
        _add(m.group(0), (m.start(), m.end()))
    for m in _RE_VERSION.finditer(text):
        _add(m.group(0), (m.start(), m.end()))
    for m in _RE_DATE.finditer(text):
        _add(m.group(0), (m.start(), m.end()))

    # Suffixed nums: $12k was already captured as currency; skip if overlap.
    for m in _RE_NUM_SUFFIX.finditer(text):
        sp = (m.start(), m.end())
        if _span_overlaps(sp):
            continue
        _add(m.group(0), sp)

    for m in _RE_YEAR.finditer(text):
        sp = (m.start(), m.end())
        if _span_overlaps(sp):
            continue
        _add(m.group(0), sp)

    # Acronyms (2+ caps with optional -digits). Filter overlaps (so 'JIRA'
    # inside 'JIRA-4521' already captured by _RE_ID is skipped).
    _IRL_STOP_ACRONYMS = {
        "ok", "lol", "omg", "lmao", "wtf", "btw", "idk", "imo", "imho",
        "tbh", "fyi", "asap", "rn", "ur", "u", "yo", "oh",
    }
    for m in _RE_ACRONYM.finditer(text):
        sp = (m.start(), m.end())
        if _span_overlaps(sp):
            continue
        s = m.group(0)
        if _normalize_entity(s) in _IRL_STOP_ACRONYMS:
            continue
        _add(s, sp)

    # Proper-noun phrases.
    for m in _RE_PROPER.finditer(text):
        sp = (m.start(), m.end())
        # Skip if this proper-noun match overlaps a previously captured span
        # (e.g. "The JIRA-" overlapping "JIRA-4521").
        if _span_overlaps(sp):
            continue
        phrase = m.group(0)
        norm = _normalize_entity(phrase)
        if not norm:
            continue
        tokens = norm.split()
        orig_tokens = phrase.split()
        # Trim leading/trailing stopword tokens from a multi-word phrase
        # (e.g., "Hey Mel" -> "Mel", "Great Sarah" -> "Sarah").
        while len(tokens) > 1 and tokens[0] in _STOPWORDS:
            tokens = tokens[1:]
            orig_tokens = orig_tokens[1:]
        while len(tokens) > 1 and tokens[-1] in _STOPWORDS:
            tokens = tokens[:-1]
            orig_tokens = orig_tokens[:-1]
        if not tokens:
            continue
        if len(tokens) == 1 and tokens[0] in _STOPWORDS:
            continue
        # Drop multi-word phrases where ALL tokens are stopwords
        if all(t in _STOPWORDS for t in tokens):
            continue
        trimmed = " ".join(orig_tokens)
        _add(trimmed, sp)

    return found


# ---------------------------------------------------------------------------
# LLM-based entity extractor (one call per turn)
# ---------------------------------------------------------------------------

ENTITY_LLM_PROMPT = """\
Extract SUBSTANTIVE named entities from this conversation turn. Include:
- Proper nouns that are SUBJECTS or OBJECTS of the statement (people being \
discussed, places being described, projects being worked on, products, \
organizations, specific events)
- Specific IDs (ticket numbers, file names, version numbers)
- Specific numbers (monetary amounts, percentages, versions, counts)
- Specific dates (actual dates, not generic "today"/"yesterday")

Do NOT include:
- Names used purely as greetings or addresses ("Hey Sarah!", "Bye, Bob" — \
the name is vocative, not about the person)
- Generic words ("friend", "work", "trip") unless used as a specific \
identifying name

Turn: {text}

Output ONE entity per line. If no entities, output exactly: NONE"""


def _parse_llm_entities(response: str) -> list[str]:
    if not response:
        return []
    lines = [ln.strip() for ln in response.strip().split("\n")]
    if len(lines) == 1 and lines[0].strip().upper() == "NONE":
        return []
    out: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        if not ln:
            continue
        if ln.strip().upper() == "NONE":
            continue
        # Strip list markers like "- ", "* ", "1. "
        ln = re.sub(r"^[-*\u2022]\s+", "", ln)
        ln = re.sub(r"^\d+[.)]\s+", "", ln)
        ln = ln.strip()
        if not ln:
            continue
        norm = _normalize_entity(ln)
        if not norm or len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        out.append(ln)
    return out


# ---------------------------------------------------------------------------
# Per-turn entity extractor with caching
# ---------------------------------------------------------------------------


@dataclass
class TurnEntityRecord:
    conversation_id: str
    turn_id: int
    text: str
    entities: list[str] = field(default_factory=list)


class TurnEntityExtractor:
    """Extract and cache per-turn entities for a conversation. Results are
    persisted in results/turn_entities.json keyed by
    (extractor_name, conversation_id, turn_id).

    The persisted structure:
      {
        "regex": {conv_id: {str(turn_id): [ent, ...]}},
        "llm":   {conv_id: {str(turn_id): [ent, ...]}}
      }
    """

    def __init__(
        self,
        extractor: str = "regex",  # "regex" or "llm"
        client: OpenAI | None = None,
    ) -> None:
        self.extractor = extractor
        self.client = client
        if extractor == "llm" and client is None:
            # Shorter per-call timeout so a stuck request doesn't hang the
            # whole thread pool. Retry up to 3x automatically.
            self.client = OpenAI(timeout=20.0, max_retries=3)
        self.llm_cache = EntityMentionLLMCache()

        self._store: dict[str, dict[str, dict[str, list[str]]]] = {
            "regex": {},
            "llm": {},
        }
        if _TURN_ENTITIES_FILE.exists():
            try:
                with open(_TURN_ENTITIES_FILE) as f:
                    data = json.load(f)
                for k in ("regex", "llm"):
                    if k in data:
                        self._store[k] = data[k]
            except (json.JSONDecodeError, OSError):
                pass

    def _get_cached(self, conv_id: str, turn_id: int) -> list[str] | None:
        convd = self._store[self.extractor].get(conv_id)
        if convd is None:
            return None
        return convd.get(str(turn_id))

    def _set_cached(
        self, conv_id: str, turn_id: int, entities: list[str]
    ) -> None:
        self._store[self.extractor].setdefault(conv_id, {})[str(turn_id)] = (
            entities
        )

    def extract_turn(
        self, conversation_id: str, turn_id: int, text: str
    ) -> list[str]:
        cached = self._get_cached(conversation_id, turn_id)
        if cached is not None:
            return cached

        if self.extractor == "regex":
            ents = extract_entities_regex(text)
        elif self.extractor == "llm":
            prompt = ENTITY_LLM_PROMPT.format(text=text)
            cached_resp = self.llm_cache.get(MODEL, prompt)
            if cached_resp is not None:
                response = cached_resp
            else:
                response = ""
                last_exc: Exception | None = None
                for attempt in range(3):
                    try:
                        completion = self.client.chat.completions.create(
                            model=MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=400,
                        )
                        response = completion.choices[0].message.content or ""
                        break
                    except Exception as e:
                        last_exc = e
                        time.sleep(1.5 * (attempt + 1))
                if not response and last_exc is not None:
                    print(
                        f"    [entity LLM] failed for turn {turn_id}: "
                        f"{last_exc}",
                        flush=True,
                    )
                self.llm_cache.put(MODEL, prompt, response)
            ents = _parse_llm_entities(response)
        else:
            raise ValueError(f"Unknown extractor: {self.extractor}")

        self._set_cached(conversation_id, turn_id, ents)
        return ents

    def extract_for_store(
        self,
        store: SegmentStore,
        conversation_ids: list[str] | None = None,
        max_workers: int = 8,
    ) -> dict[str, dict[int, list[str]]]:
        all_cids = sorted({s.conversation_id for s in store.segments})
        if conversation_ids is None:
            conversation_ids = all_cids

        # Collect segments to process (those missing from cache)
        pending: list[Segment] = []
        for seg in store.segments:
            if seg.conversation_id not in conversation_ids:
                continue
            if self._get_cached(seg.conversation_id, seg.turn_id) is None:
                pending.append(seg)

        if pending and self.extractor == "llm":
            print(
                f"  [entity] LLM-extracting entities for {len(pending)} "
                f"pending turns across {len(conversation_ids)} conversations",
                flush=True,
            )
            # Use a thread pool for LLM calls
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        self.extract_turn,
                        seg.conversation_id,
                        seg.turn_id,
                        seg.text,
                    ): seg
                    for seg in pending
                }
                done_n = 0
                for fut in as_completed(futures):
                    seg = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(
                            f"    [entity LLM] error on {seg.conversation_id}"
                            f"/{seg.turn_id}: {e}",
                            flush=True,
                        )
                    done_n += 1
                    if done_n % 25 == 0:
                        print(
                            f"    [entity LLM] {done_n}/{len(pending)}",
                            flush=True,
                        )
                        # Periodic save
                        self.save()
        elif pending:
            # Regex is fast enough to do inline.
            for seg in pending:
                try:
                    self.extract_turn(
                        seg.conversation_id, seg.turn_id, seg.text
                    )
                except Exception as e:
                    print(
                        f"    [entity regex] error on {seg.conversation_id}"
                        f"/{seg.turn_id}: {e}",
                        flush=True,
                    )

        self.save()

        out: dict[str, dict[int, list[str]]] = {}
        for cid in conversation_ids:
            out[cid] = {}
            convd = self._store[self.extractor].get(cid, {})
            for tid_str, ents in convd.items():
                try:
                    out[cid][int(tid_str)] = ents
                except ValueError:
                    pass
        return out

    def save(self) -> None:
        self.llm_cache.save()
        _TURN_ENTITIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TURN_ENTITIES_FILE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._store, f, indent=2, default=str)
        tmp.replace(_TURN_ENTITIES_FILE)


# ---------------------------------------------------------------------------
# Inverted index
# ---------------------------------------------------------------------------


@dataclass
class InvertedIndex:
    """entity(norm) -> set of segment indices (global store indices)."""

    by_entity: dict[str, set[int]] = field(default_factory=dict)
    # Optional: also store per-segment entities for diagnostics
    by_segment_index: dict[int, list[str]] = field(default_factory=dict)

    def add(self, entity_norm: str, seg_index: int) -> None:
        if not entity_norm:
            return
        bucket = self.by_entity.setdefault(entity_norm, set())
        bucket.add(seg_index)

    def query(self, entities_norm: list[str]) -> dict[int, int]:
        """Return {seg_index: match_count} for segments that contain any of
        the query entities. match_count is the number of *distinct* query
        entities a segment matches."""
        counts: dict[int, int] = {}
        seen_per_q: list[set[int]] = []
        for e in entities_norm:
            hits = self.by_entity.get(e)
            if not hits:
                continue
            seen_per_q.append(hits)
        # Count how many query entities each segment matches.
        for hits in seen_per_q:
            for idx in hits:
                counts[idx] = counts.get(idx, 0) + 1
        return counts


def build_conversation_index(
    store: SegmentStore,
    conv_id: str,
    extractor: TurnEntityExtractor,
) -> InvertedIndex:
    """Build a per-conversation inverted index using the cached entities.

    Also normalizes entity strings for matching.
    """
    idx = InvertedIndex()
    for seg in store.segments:
        if seg.conversation_id != conv_id:
            continue
        ents = extractor.extract_turn(conv_id, seg.turn_id, seg.text)
        norm_ents: list[str] = []
        for e in ents:
            n = _normalize_entity(e)
            if n:
                idx.add(n, seg.index)
                norm_ents.append(n)
        idx.by_segment_index[seg.index] = norm_ents
    return idx


# ---------------------------------------------------------------------------
# Query-time entity extraction (SAME extractor as ingest)
# ---------------------------------------------------------------------------


def extract_query_entities(
    query: str, extractor: TurnEntityExtractor
) -> list[str]:
    """Extract entities from the query using the same extractor used at
    ingest time. For the LLM extractor we cache the query-level LLM call
    the same way as turns (separate cache entry because the prompt is the
    same template)."""
    # Use a synthetic turn_id to cache; but we can simply skip the store
    # cache for queries and go straight to extractor.extract_turn with a
    # placeholder conv_id/turn_id — but that pollutes per-turn store. Use
    # a dedicated path: for regex just run extract_entities_regex; for LLM
    # reuse the prompt-level LLM cache directly.
    if extractor.extractor == "regex":
        return extract_entities_regex(query)
    # LLM path: call the LLM (or fetch from prompt cache) but don't pollute
    # turn_entities store.
    prompt = ENTITY_LLM_PROMPT.format(text=query)
    cached = extractor.llm_cache.get(MODEL, prompt)
    if cached is not None:
        return _parse_llm_entities(cached)
    # Fallback live call
    response = ""
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            completion = extractor.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=400,
            )
            response = completion.choices[0].message.content or ""
            break
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * (attempt + 1))
    if not response and last_exc is not None:
        print(f"    [entity LLM query] failed: {last_exc}", flush=True)
    extractor.llm_cache.put(MODEL, prompt, response)
    return _parse_llm_entities(response)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class _EntityMentionBase(BestshotBase):
    """Base: per-conversation inverted index + cosine score + bonus merge."""

    arch_name: str = "entity_mention_base"
    extractor_kind: str = "regex"  # "regex" or "llm"
    beta: float = 0.1
    run_v2f: bool = False  # merge with v2f cue retrievals

    # Class-level singletons keyed by store id so repeated instantiations
    # reuse the extractor + inverted indices.
    _extractor_cache: dict[tuple[int, str], TurnEntityExtractor] = {}
    _index_cache: dict[tuple[int, str, str], InvertedIndex] = {}

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None
    ) -> None:
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = EntityMentionEmbeddingCache()
        self.llm_cache = EntityMentionLLMCache()

        ext_key = (id(store), self.extractor_kind)
        extractor = self._extractor_cache.get(ext_key)
        if extractor is None:
            extractor = TurnEntityExtractor(
                extractor=self.extractor_kind, client=self.client
            )
            # Note: we defer extraction until retrieve() is called for a
            # conv, so we don't spend LLM budget on convs not being
            # queried. Regex is fast enough to just do eagerly.
            if self.extractor_kind == "regex":
                extractor.extract_for_store(store)
            self._extractor_cache[ext_key] = extractor
        self.extractor = extractor
        # Track which convs we've extracted for (LLM path)
        self._extracted_convs: set[str] = set()

    def save_caches(self) -> None:
        super().save_caches()
        # Also save extractor's caches (query-entity extractions for LLM path,
        # plus turn_entities.json).
        self.extractor.save()

    def _get_index(self, conv_id: str) -> InvertedIndex:
        ikey = (id(self.store), self.extractor_kind, conv_id)
        idx = self._index_cache.get(ikey)
        if idx is None:
            # For LLM extractor, extract on demand per conv
            if (
                self.extractor_kind == "llm"
                and conv_id not in self._extracted_convs
            ):
                self.extractor.extract_for_store(
                    self.store, conversation_ids=[conv_id]
                )
                self._extracted_convs.add(conv_id)
            idx = build_conversation_index(
                self.store, conv_id, self.extractor
            )
            self._index_cache[ikey] = idx
        return idx

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,
                )
                text = response.choices[0].message.content or ""
                self.llm_cache.put(model, prompt, text)
                self.llm_calls += 1
                return text
            except Exception as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        print(f"    LLM call failed after 3 attempts: {last_exc}", flush=True)
        self.llm_cache.put(model, prompt, "")
        self.llm_calls += 1
        return ""

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.embeddings.create(
                    model=EMBED_MODEL, input=[text]
                )
                embedding = np.array(
                    response.data[0].embedding, dtype=np.float32
                )
                self.embedding_cache.put(text, embedding)
                self.embed_calls += 1
                return embedding
            except Exception as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        print(f"    Embed failed after 3 attempts: {last_exc}", flush=True)
        self.embed_calls += 1
        return np.zeros(1536, dtype=np.float32)

    def retrieve(
        self, question: str, conversation_id: str
    ) -> BestshotResult:
        index = self._get_index(conversation_id)

        # 1. Extract entities from query (same extractor).
        query_ents_raw = extract_query_entities(question, self.extractor)
        query_ents_norm: list[str] = []
        seen_norm: set[str] = set()
        for e in query_ents_raw:
            n = _normalize_entity(e)
            if n and n not in seen_norm:
                seen_norm.add(n)
                query_ents_norm.append(n)

        # 2. Candidate-bonus lookup.
        bonus_counts = index.query(query_ents_norm)

        # 3. Compute cosine for ALL segments in this conversation and combine
        #    with bonus. We rank all segments (cosine + beta * indicator) then
        #    return ordered by score. Fair-backfill with cosine supplies
        #    leftover positions.
        query_emb = self.embed_text(question)
        q_norm = query_emb / max(float(np.linalg.norm(query_emb)), 1e-10)
        conv_mask = self.store.conversation_ids == conversation_id
        # Compute cosine over the whole corpus then mask
        sims = self.store.normalized_embeddings @ q_norm
        sims_conv = np.where(conv_mask, sims, -1e9)

        # Build score_map for conversation's segments only
        conv_indices = np.where(conv_mask)[0].tolist()
        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}
        for idx_i in conv_indices:
            cos = float(sims_conv[idx_i])
            n_match = bonus_counts.get(idx_i, 0)
            # Indicator bonus (1 if any match, 0 otherwise) — per the plan's
            # formula: final_score = cosine + beta * I(turn mentions query ent)
            bonus = self.beta if n_match > 0 else 0.0
            score = cos + bonus
            score_map[idx_i] = score
            seg_map[idx_i] = self.store.segments[idx_i]

        # 4. Optional: merge with v2f cue retrievals (use max-score merge).
        v2f_cues: list[str] = []
        v2f_outcomes: list[dict] = []
        if self.run_v2f:
            # Run v2f cue gen on the ORIGINAL question with the top-10 cosine
            # primer (same pattern as alias_expansion).
            primer_result = self.store.search(
                query_emb, top_k=10, conversation_id=conversation_id
            )
            primer_segs = list(primer_result.segments)
            context_section = (
                "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
                + _format_segments(primer_segs)
            )
            prompt = V2F_PROMPT.format(
                question=question, context_section=context_section
            )
            output = self.llm_call(prompt)
            cues = _parse_cues(output)[:2]
            v2f_cues = cues
            for cue in cues:
                if not cue:
                    continue
                cue_emb = self.embed_text(cue)
                cue_norm = cue_emb / max(
                    float(np.linalg.norm(cue_emb)), 1e-10
                )
                cue_sims = self.store.normalized_embeddings @ cue_norm
                cue_sims_conv = np.where(conv_mask, cue_sims, -1e9)
                # Take top-10 for this cue
                order = np.argsort(cue_sims_conv)[::-1][:10]
                rids: list[int] = []
                for i in order:
                    s = float(cue_sims_conv[i])
                    if s <= -1e8:
                        continue
                    # Apply entity bonus too (the turn's entity match against
                    # query entities) to stay consistent.
                    n_match = bonus_counts.get(int(i), 0)
                    bonus = self.beta if n_match > 0 else 0.0
                    combined = s + bonus
                    if int(i) not in score_map or combined > score_map[int(i)]:
                        score_map[int(i)] = combined
                    if int(i) not in seg_map:
                        seg_map[int(i)] = self.store.segments[int(i)]
                    rids.append(int(i))
                v2f_outcomes.append(
                    {
                        "cue": cue,
                        "retrieved_turn_ids": [
                            self.store.segments[i].turn_id for i in rids
                        ],
                    }
                )

        # 5. Rank.
        ranked = sorted(
            score_map.keys(), key=lambda i: score_map[i], reverse=True
        )
        all_segments = [seg_map[i] for i in ranked]

        # Metadata
        # Which segments got bonus?
        boosted_turn_ids = sorted(
            {self.store.segments[i].turn_id for i in bonus_counts.keys()}
        )
        metadata = {
            "name": self.arch_name,
            "extractor": self.extractor_kind,
            "beta": self.beta,
            "query_entities": query_ents_raw,
            "query_entities_norm": query_ents_norm,
            "num_boosted_turns": len(bonus_counts),
            "boosted_turn_ids": boosted_turn_ids[:50],
            "v2f_cues": v2f_cues,
            "v2f_outcomes": v2f_outcomes,
            "num_turns_in_index": sum(
                1 for _ in index.by_segment_index
            ),
            "index_num_entities": len(index.by_entity),
        }
        return BestshotResult(segments=all_segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class EntityRegexB005(_EntityMentionBase):
    arch_name = "entity_regex_b0.05"
    extractor_kind = "regex"
    beta = 0.05


class EntityRegexB01(_EntityMentionBase):
    arch_name = "entity_regex_b0.1"
    extractor_kind = "regex"
    beta = 0.1


class EntityRegexB02(_EntityMentionBase):
    arch_name = "entity_regex_b0.2"
    extractor_kind = "regex"
    beta = 0.2


class EntityLLMB01(_EntityMentionBase):
    arch_name = "entity_llm_b0.1"
    extractor_kind = "llm"
    beta = 0.1


class EntityRegexPlusV2f(_EntityMentionBase):
    arch_name = "entity_regex_plus_v2f"
    extractor_kind = "regex"
    beta = 0.1
    run_v2f = True


ARCH_CLASSES: dict[str, type[_EntityMentionBase]] = {
    "entity_regex_b0.05": EntityRegexB005,
    "entity_regex_b0.1": EntityRegexB01,
    "entity_regex_b0.2": EntityRegexB02,
    "entity_llm_b0.1": EntityLLMB01,
    "entity_regex_plus_v2f": EntityRegexPlusV2f,
}
