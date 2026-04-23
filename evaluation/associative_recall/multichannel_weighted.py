"""LLM-weighted multi-channel retrieval architecture.

"LLM as conductor": instead of running all retrieval channels in parallel with
fixed weights (which causes cannibalization — good cosine hits get displaced by
irrelevant channel outputs), the LLM is given one chance per query to decide
WHICH channels should engage and how strongly. Irrelevant channels get
weight=0 and are skipped; relevant channels get scored and their normalized
scores are linearly combined.

Channels (each produces ranked (segment, normalized_score) candidates):
  1. cosine_baseline : raw query cosine retrieval (general-purpose)
  2. v2f_cosine      : v2f cue-generated cosine retrieval (open/complex queries)
  3. speaker_filter  : role-filtered cosine when query names a participant
  4. alias_context   : alias-expanded cosine probes for entities with aliases
  5. critical_info   : ingest-time critical-fact alt-key retrieval
  6. temporal_tokens : boost turns containing date/time/sequence tokens
  7. entity_exact_match : boost turns with proper-noun tokens matching query

Routing (one LLM call per query, gpt-5-mini):
  - Outputs JSON weights in [0,1] per channel plus a reasoning sentence.
  - Channels with weight=0 are skipped (cost savings).
  - Each active channel's top-K results are normalized to [0,1] by dividing
    by its top score. Scores are fused:
        final_score(seg) = Σ weight_c × normalized_score_c(seg)
  - Ranked, truncated to K.

Variants
--------
  multich_llm_weighted : per-query LLM weights (full architecture)
  multich_uniform      : uniform weight=1 across all channels (control —
                         tests if LLM routing is the value vs just running
                         many channels)
  multich_binary       : LLM chooses binary {0,1} per channel (simpler)

All channels reuse persisted ingest artifacts where possible:
  - conversation_speakers.json / conversation_two_speakers.json (speaker IDs)
  - conversation_alias_groups.json (alias groups)
  - critical-info alt-key classification via CriticalInfoGenerator (cache-only)

Writes go ONLY to multich_*_cache.json. Reads from shared caches for
embeddings and LLM responses (v2f, alias) where available.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import V2F_PROMPT, _format_segments, _parse_cues
from alias_expansion import (
    AliasExtractor,
    build_expanded_queries,
    find_alias_matches,
)
from speaker_attributed import extract_name_mentions

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"

# Dedicated caches — writes go ONLY here.
_MULTICH_EMB_FILE = CACHE_DIR / "multich_embedding_cache.json"
_MULTICH_LLM_FILE = CACHE_DIR / "multich_llm_cache.json"
_MULTICH_CRIT_FILE = CACHE_DIR / "multich_crit_cache.json"

# Read-only shared caches (warm-start). Order matters: last wins on key clash.
_SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "alias_embedding_cache.json",
    "speaker_embedding_cache.json",
    "two_speaker_embedding_cache.json",
    "multich_embedding_cache.json",
)
_SHARED_LLM_READ = (
    "llm_cache.json",
    "arch_llm_cache.json",
    "tree_llm_cache.json",
    "frontier_llm_cache.json",
    "meta_llm_cache.json",
    "optim_llm_cache.json",
    "bestshot_llm_cache.json",
    "alias_llm_cache.json",
    "speaker_llm_cache.json",
    "two_speaker_llm_cache.json",
    "antipara_llm_cache.json",
    "multich_llm_cache.json",
)


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
class MultichEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _MULTICH_EMB_FILE
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
        tmp = self.cache_file.parent / (
            self.cache_file.name + f".tmp.{os.getpid()}"
        )
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


class MultichLLMCache(LLMCache):
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
        self.cache_file = _MULTICH_LLM_FILE
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
        tmp = self.cache_file.parent / (
            self.cache_file.name + f".tmp.{os.getpid()}"
        )
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Channel catalog and LLM routing prompt
# ---------------------------------------------------------------------------
CHANNEL_NAMES = (
    "cosine_baseline",
    "v2f_cosine",
    "speaker_filter",
    "alias_context",
    "critical_info",
    "temporal_tokens",
    "entity_exact_match",
)

CHANNEL_DESCRIPTIONS = {
    "cosine_baseline": (
        "raw cosine; general-purpose, works for most queries"
    ),
    "v2f_cosine": (
        "LLM-imagined cue cosine; best for open queries with complex intent"
    ),
    "speaker_filter": (
        "filter/boost turns spoken by a named person in the query"
    ),
    "alias_context": (
        "boost when query mentions an entity with known aliases"
    ),
    "critical_info": (
        "boost turns containing facts of enduring importance (dates, "
        "preferences, commitments)"
    ),
    "temporal_tokens": (
        "boost turns with dates/time/sequence words; use for temporal queries"
    ),
    "entity_exact_match": (
        "boost turns exact-matching proper nouns in query"
    ),
}

ROUTING_PROMPT = """\
You are routing a retrieval query across multiple channels. Given the query, \
output weights for each channel (values 0.0-1.0). Channels will retrieve in \
parallel; candidate turns are merged by score = Σ weight_i × \
normalized_score_i(turn). Set weight to 0.0 if a channel is irrelevant for \
this query.

Channels:
- cosine_baseline: raw cosine; general-purpose, works for most queries
- v2f_cosine: LLM-imagined cue cosine; best for open queries with complex \
intent
- speaker_filter: filter/boost turns spoken by a named person in the query
- alias_context: boost when query mentions an entity with known aliases
- critical_info: boost turns containing facts of enduring importance \
(dates, preferences, commitments)
- temporal_tokens: boost turns with dates/time/sequence words; use for \
temporal queries
- entity_exact_match: boost turns exact-matching proper nouns in query

Query: {query}

Output JSON: {{"cosine_baseline": 0.x, "v2f_cosine": 0.x, "speaker_filter": \
0.x, "alias_context": 0.x, "critical_info": 0.x, "temporal_tokens": 0.x, \
"entity_exact_match": 0.x, "reasoning": "..."}}

Include brief reasoning in a "reasoning" field (one sentence). Weights sum \
doesn't need to be 1.0; think of them as engagement strengths. Output ONLY \
the JSON object, no prose before or after."""


def parse_weights(raw: str) -> tuple[dict[str, float], str]:
    """Parse routing JSON. Returns (weights, reasoning). Fallback: uniform
    over {cosine_baseline, v2f_cosine}."""
    default = {ch: 0.0 for ch in CHANNEL_NAMES}
    default["cosine_baseline"] = 1.0
    default["v2f_cosine"] = 1.0
    fallback_reason = "parse_failed_uniform_fallback"

    if not raw:
        return default, fallback_reason

    # Strip code fences and surrounding prose.
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return default, fallback_reason

    if not isinstance(obj, dict):
        return default, fallback_reason

    weights: dict[str, float] = {}
    for ch in CHANNEL_NAMES:
        v = obj.get(ch, 0.0)
        try:
            w = float(v)
        except (TypeError, ValueError):
            w = 0.0
        weights[ch] = max(0.0, min(1.0, w))

    reasoning = str(obj.get("reasoning", "")).strip()[:300]

    # Safety: if LLM set all weights to 0, fall back to uniform over two.
    if sum(weights.values()) < 1e-6:
        weights = default
        reasoning = reasoning + " [all-zero -> fallback]"

    return weights, reasoning


# ---------------------------------------------------------------------------
# Channel implementations
# ---------------------------------------------------------------------------
_RE_DATE_WORDS = re.compile(
    r"\b("
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|"
    r"today|tomorrow|yesterday|tonight|week|month|year|weekend|weekday|"
    r"morning|afternoon|evening|night|noon|midnight|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"last|next|previous|before|after|recent|recently|later|earlier|"
    r"once|twice|"
    r"spring|summer|fall|autumn|winter"
    r")\b",
    re.IGNORECASE,
)
_RE_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\b")
_RE_DATE_DIGIT = re.compile(
    r"\b(?:\d{4}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}(?:st|nd|rd|th))\b"
)

_NAME_STOPWORDS = {
    "I", "You", "Me", "My", "We", "Us", "Our", "Your", "They", "Them",
    "He", "She", "It", "This", "That", "These", "Those", "There", "Here",
    "What", "When", "Where", "Who", "Why", "How", "Which", "Whom", "Whose",
    "The", "A", "An", "Is", "Are", "Was", "Were", "Be", "Been", "Being",
    "Have", "Has", "Had", "Do", "Does", "Did", "Will", "Would", "Should",
    "Could", "Can", "May", "Might", "Must", "Shall",
    "Yes", "No", "Not", "Never", "Always", "Maybe",
    "And", "Or", "But", "If", "So", "Then", "Else", "Because", "Since",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}

_RE_CAPWORD = re.compile(r"\b([A-Z][a-zA-Z\-']+)\b")


def turn_has_temporal_tokens(text: str) -> bool:
    """Cheap regex check for temporal-ish content."""
    if _RE_DATE_WORDS.search(text):
        return True
    if _RE_TIME.search(text):
        return True
    if _RE_DATE_DIGIT.search(text):
        return True
    return False


def extract_query_entities(query: str) -> set[str]:
    """Capitalized tokens in the query that are likely proper nouns."""
    tokens = _RE_CAPWORD.findall(query)
    out: set[str] = set()
    for tok in tokens:
        if tok in _NAME_STOPWORDS:
            continue
        # Exclude sentence-first-word if it's a common wh-word.
        if len(tok) <= 1:
            continue
        out.add(tok)
    return out


# ---------------------------------------------------------------------------
# Ingest-side precomputed artifacts
# ---------------------------------------------------------------------------
_CONV_SPEAKERS_FILE = (
    Path(__file__).resolve().parent / "results" / "conversation_speakers.json"
)
_CONV_TWO_SPEAKERS_FILE = (
    Path(__file__).resolve().parent
    / "results"
    / "conversation_two_speakers.json"
)


def load_speaker_map() -> dict[str, dict[str, str]]:
    """Load the two-speaker map. Returns cid -> {user, assistant} dict.

    Warm-starts from conversation_two_speakers.json; if only single-speaker
    file exists, back-fills user name only.
    """
    out: dict[str, dict[str, str]] = {}
    if _CONV_TWO_SPEAKERS_FILE.exists():
        try:
            with open(_CONV_TWO_SPEAKERS_FILE) as f:
                data = json.load(f)
            raw = data.get("speakers", {}) or {}
            for cid, pair in raw.items():
                if isinstance(pair, dict):
                    out[cid] = {
                        "user": (pair.get("user") or "UNKNOWN").strip()
                        or "UNKNOWN",
                        "assistant": (
                            pair.get("assistant") or "UNKNOWN"
                        ).strip()
                        or "UNKNOWN",
                    }
        except (json.JSONDecodeError, OSError):
            pass
    if _CONV_SPEAKERS_FILE.exists():
        try:
            with open(_CONV_SPEAKERS_FILE) as f:
                data = json.load(f)
            one = data.get("speakers", {}) or {}
            for cid, name in one.items():
                if cid not in out:
                    out[cid] = {"user": name or "UNKNOWN", "assistant": "UNKNOWN"}
        except (json.JSONDecodeError, OSError):
            pass
    return out


# ---------------------------------------------------------------------------
# Critical-info support (cache-only: classify on demand, read-heavy)
# ---------------------------------------------------------------------------
class _CriticalClassifier:
    """Minimal critical-info classifier that ONLY reads the multich cache.

    It does NOT call the LLM. If a turn's cache key is absent, the turn is
    treated as non-critical. This lets us run with zero new critical-info
    cost, piggy-backing on whatever critical_info decisions are already in
    the shared LLM caches via prior runs — but critically, if NONE are
    present for our datasets, the channel simply produces empty output and
    gets weight=0 naturally (nothing to reward).

    For fair comparison with ens_all_plus_crit we'd need the full ingest
    pipeline; however, our study is about LLM-weighted routing over
    channels we CAN cheaply produce. This keeps the cost budget tight.
    """

    def __init__(self, store: SegmentStore, llm_cache: MultichLLMCache):
        self.store = store
        self.llm_cache = llm_cache
        self._critical_flags: dict[int, bool] = {}
        self._alt_keys: dict[int, list[str]] = {}

    def lookup(self, seg_index: int) -> tuple[bool, list[str]]:
        """Return (is_critical, alt_keys) for a segment. Uses cached
        critical_info_store prompts if available."""
        if seg_index in self._critical_flags:
            return self._critical_flags[seg_index], self._alt_keys.get(
                seg_index, []
            )
        seg = self.store.segments[seg_index]
        # Build the same cache key as critical_info_store v3
        from critical_info_store import build_prompt, parse_response
        prompt = build_prompt("v3", seg.role, seg.text)
        ck = f"[critical_info_store/v3]\n" + prompt
        raw = self.llm_cache.get(MODEL, ck)
        if raw is None:
            self._critical_flags[seg_index] = False
            self._alt_keys[seg_index] = []
            return False, []
        critical, alts = parse_response(raw)
        self._critical_flags[seg_index] = critical
        self._alt_keys[seg_index] = alts
        return critical, alts

    def build_critical_set_for_conv(
        self, conversation_id: str
    ) -> list[tuple[int, list[str]]]:
        """Walk all segments in the conversation and return list of
        (seg_index, alt_keys) for critical ones."""
        out: list[tuple[int, list[str]]] = []
        for i, seg in enumerate(self.store.segments):
            if seg.conversation_id != conversation_id:
                continue
            critical, alts = self.lookup(i)
            if critical and alts:
                out.append((i, alts))
        return out


# ---------------------------------------------------------------------------
# Main architecture
# ---------------------------------------------------------------------------
@dataclass
class ChannelResult:
    name: str
    candidates: list[tuple[int, float]]  # (seg_index, raw_score)
    # normalized_score can be derived in merge step
    executed: bool = True


@dataclass
class MultichResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


class MultichannelWeighted:
    """LLM-weighted multi-channel retrieval.

    Mode:
      - "llm"     : per-query LLM routing
      - "uniform" : weight=1.0 for every channel
      - "binary"  : LLM routing, but weights thresholded to {0,1} at 0.5
    """

    arch_name = "multich_llm_weighted"

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        mode: str = "llm",
        per_channel_top_k: int = 50,
    ):
        self.store = store
        self.client = client or OpenAI(timeout=60.0, max_retries=3)
        self.mode = mode
        self.per_channel_top_k = per_channel_top_k
        self.embedding_cache = MultichEmbeddingCache()
        self.llm_cache = MultichLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

        # Ingest artifacts
        self.speaker_map = load_speaker_map()
        self.alias_extractor = AliasExtractor(client=self.client)
        # Do not force extraction; reuse whatever's persisted in
        # conversation_alias_groups.json. For conversations without groups,
        # alias_context channel will naturally yield empty output.
        self.crit_classifier = _CriticalClassifier(store, self.llm_cache)

        # Per-store role masks for speaker channel
        self.role_masks = {
            "user": np.array(
                [s.role == "user" for s in store.segments], dtype=bool
            ),
            "assistant": np.array(
                [s.role == "assistant" for s in store.segments], dtype=bool
            ),
        }

        # Per-store precomputed temporal mask
        self.temporal_mask = np.array(
            [turn_has_temporal_tokens(s.text) for s in store.segments],
            dtype=bool,
        )

        # Cache of critical-items per conv (built lazily)
        self._crit_conv_cache: dict[str, list[tuple[int, list[str]]]] = {}

        # For arch name in variant mode
        if mode == "uniform":
            self.arch_name = "multich_uniform"
        elif mode == "binary":
            self.arch_name = "multich_binary"
        else:
            self.arch_name = "multich_llm_weighted"

    # --- cache helpers ---
    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.client.embeddings.create(
            model=EMBED_MODEL, input=[text]
        )
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, emb)
        self.embed_calls += 1
        return emb

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        last_exc = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,
                )
                text = resp.choices[0].message.content or ""
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

    def save_caches(self) -> None:
        try:
            self.embedding_cache.save()
        except Exception as e:
            print(f"  (warn) embedding_cache.save failed: {e}", flush=True)
        try:
            self.llm_cache.save()
        except Exception as e:
            print(f"  (warn) llm_cache.save failed: {e}", flush=True)

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    # --- routing ---
    def get_weights(self, query: str) -> tuple[dict[str, float], str, str]:
        """Returns (weights, reasoning, raw_response)."""
        if self.mode == "uniform":
            w = {ch: 1.0 for ch in CHANNEL_NAMES}
            return w, "uniform_mode", ""
        prompt = ROUTING_PROMPT.format(query=query)
        raw = self.llm_call(prompt)
        weights, reasoning = parse_weights(raw)
        if self.mode == "binary":
            weights = {
                k: (1.0 if v >= 0.5 else 0.0) for k, v in weights.items()
            }
            # Ensure at least one channel fires.
            if sum(weights.values()) < 1e-6:
                weights["cosine_baseline"] = 1.0
                weights["v2f_cosine"] = 1.0
        return weights, reasoning, raw

    # --- channels ---
    def _cosine_search_in_conv(
        self,
        query_emb: np.ndarray,
        conversation_id: str,
        mask: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Generic cosine search filtered by conversation + optional mask.

        Returns list of (seg_index, score) descending. Score is cosine
        similarity in [-1, 1] but typically [0, 1]."""
        k = top_k or self.per_channel_top_k
        q = query_emb.astype(np.float32)
        qn = max(float(np.linalg.norm(q)), 1e-10)
        q = q / qn
        sims = self.store.normalized_embeddings @ q
        conv_mask = self.store.conversation_ids == conversation_id
        combined = conv_mask.copy()
        if mask is not None:
            combined = combined & mask
        sims = np.where(combined, sims, -1.0)
        order = np.argsort(sims)[::-1][: max(k, 1)]
        out: list[tuple[int, float]] = []
        for i in order:
            if sims[i] <= -0.5:
                break
            out.append((int(i), float(sims[i])))
        return out

    def ch_cosine_baseline(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> ChannelResult:
        cands = self._cosine_search_in_conv(query_emb, conversation_id)
        return ChannelResult("cosine_baseline", cands)

    def ch_v2f_cosine(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> ChannelResult:
        # Primer top-10 from raw cosine
        primer = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        primer_segs = list(primer.segments)
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(primer_segs)
        )
        prompt = V2F_PROMPT.format(
            question=query, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]
        # For each cue, retrieve top-K by cosine; merge by max score per
        # parent_index.
        agg: dict[int, float] = {}
        for cue in cues:
            cue_emb = self.embed_text(cue)
            res = self._cosine_search_in_conv(
                cue_emb, conversation_id, top_k=self.per_channel_top_k
            )
            for idx, sc in res:
                if idx not in agg or sc > agg[idx]:
                    agg[idx] = sc
        ordered = sorted(agg.items(), key=lambda x: -x[1])[
            : self.per_channel_top_k
        ]
        return ChannelResult("v2f_cosine", ordered)

    def ch_speaker_filter(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> ChannelResult:
        """Role-filtered cosine when query names one (and only one) of the
        two speakers."""
        pair = self.speaker_map.get(conversation_id, {})
        user_name = (pair.get("user") or "UNKNOWN").lower()
        asst_name = (pair.get("assistant") or "UNKNOWN").lower()
        mentions = {t.lower() for t in extract_name_mentions(query)}
        hit_user = user_name != "unknown" and user_name in mentions
        hit_asst = asst_name != "unknown" and asst_name in mentions
        if hit_user and not hit_asst:
            mask = self.role_masks["user"]
        elif hit_asst and not hit_user:
            mask = self.role_masks["assistant"]
        else:
            return ChannelResult("speaker_filter", [], executed=False)
        cands = self._cosine_search_in_conv(
            query_emb, conversation_id, mask=mask
        )
        return ChannelResult("speaker_filter", cands)

    def ch_alias_context(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> ChannelResult:
        groups = self.alias_extractor.get_groups(conversation_id)
        if not groups:
            return ChannelResult("alias_context", [], executed=False)
        matches = find_alias_matches(query, groups)
        if not matches:
            return ChannelResult("alias_context", [], executed=False)
        variants, _ = build_expanded_queries(query, groups)
        # Sibling probes (the aliases alone)
        probes: list[str] = [v for v in variants if v != query]
        for _matched, siblings in matches:
            for sib in siblings[:4]:
                if sib not in probes:
                    probes.append(sib)
        # Score map: max cosine across variants/probes
        agg: dict[int, float] = {}
        for text in probes[:8]:
            emb = self.embed_text(text)
            res = self._cosine_search_in_conv(
                emb, conversation_id, top_k=self.per_channel_top_k
            )
            for idx, sc in res:
                if idx not in agg or sc > agg[idx]:
                    agg[idx] = sc
        if not agg:
            return ChannelResult("alias_context", [], executed=False)
        ordered = sorted(agg.items(), key=lambda x: -x[1])[
            : self.per_channel_top_k
        ]
        return ChannelResult("alias_context", ordered)

    def ch_critical_info(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> ChannelResult:
        if conversation_id not in self._crit_conv_cache:
            self._crit_conv_cache[conversation_id] = (
                self.crit_classifier.build_critical_set_for_conv(
                    conversation_id
                )
            )
        crit_items = self._crit_conv_cache[conversation_id]
        if not crit_items:
            return ChannelResult("critical_info", [], executed=False)
        # Embed alt-keys and compute max cosine against query.
        qn = max(float(np.linalg.norm(query_emb)), 1e-10)
        q = query_emb.astype(np.float32) / qn
        per_parent: dict[int, float] = {}
        for parent_idx, alts in crit_items:
            best = -1.0
            for alt in alts:
                alt_emb = self.embed_text(alt)
                an = max(float(np.linalg.norm(alt_emb)), 1e-10)
                alt_n = alt_emb / an
                sim = float(alt_n @ q)
                if sim > best:
                    best = sim
            if best > 0.0:
                per_parent[parent_idx] = best
        ordered = sorted(per_parent.items(), key=lambda x: -x[1])[
            : self.per_channel_top_k
        ]
        return ChannelResult("critical_info", ordered)

    def ch_temporal_tokens(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> ChannelResult:
        """Cosine retrieval restricted to turns with temporal tokens."""
        cands = self._cosine_search_in_conv(
            query_emb, conversation_id, mask=self.temporal_mask
        )
        return ChannelResult("temporal_tokens", cands)

    def ch_entity_exact_match(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> ChannelResult:
        """Score turns by how many of the query's proper-noun tokens appear
        in the turn text. Scored as hits_per_turn / max_hits_in_query."""
        entities = extract_query_entities(query)
        if not entities:
            return ChannelResult("entity_exact_match", [], executed=False)
        conv_mask = self.store.conversation_ids == conversation_id
        # Score per turn in conv
        ents_lower = [e.lower() for e in entities]
        scores: list[tuple[int, float]] = []
        for i, seg in enumerate(self.store.segments):
            if not conv_mask[i]:
                continue
            tl = seg.text.lower()
            hits = 0
            for e in ents_lower:
                # word-boundary
                if re.search(
                    r"\b" + re.escape(e) + r"\b", tl
                ):
                    hits += 1
            if hits > 0:
                scores.append((i, float(hits) / float(len(ents_lower))))
        scores.sort(key=lambda x: -x[1])
        return ChannelResult(
            "entity_exact_match", scores[: self.per_channel_top_k]
        )

    # --- merge ---
    @staticmethod
    def _normalize(cands: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if not cands:
            return cands
        top = max(sc for _, sc in cands)
        if top <= 0:
            return [(i, 0.0) for i, _ in cands]
        return [(i, sc / top) for i, sc in cands]

    def retrieve(
        self, question: str, conversation_id: str
    ) -> MultichResult:
        weights, reasoning, raw_weights = self.get_weights(question)
        query_emb = self.embed_text(question)

        channel_funcs = {
            "cosine_baseline": self.ch_cosine_baseline,
            "v2f_cosine": self.ch_v2f_cosine,
            "speaker_filter": self.ch_speaker_filter,
            "alias_context": self.ch_alias_context,
            "critical_info": self.ch_critical_info,
            "temporal_tokens": self.ch_temporal_tokens,
            "entity_exact_match": self.ch_entity_exact_match,
        }

        executed: dict[str, ChannelResult] = {}
        for ch in CHANNEL_NAMES:
            if weights[ch] <= 0.0:
                continue
            fn = channel_funcs[ch]
            res = fn(question, query_emb, conversation_id)
            executed[ch] = res

        # Merge
        final_scores: dict[int, float] = {}
        channel_used: dict[str, int] = {}
        for ch, res in executed.items():
            if not res.candidates:
                channel_used[ch] = 0
                continue
            normalized = self._normalize(res.candidates)
            w = weights[ch]
            for idx, nsc in normalized:
                final_scores[idx] = final_scores.get(idx, 0.0) + w * nsc
            channel_used[ch] = len(normalized)

        # Always fall back to cosine_baseline if nothing found (e.g.
        # LLM chose channels that all returned empty).
        if not final_scores:
            fallback = self.ch_cosine_baseline(
                question, query_emb, conversation_id
            )
            normalized = self._normalize(fallback.candidates)
            for idx, nsc in normalized:
                final_scores[idx] = nsc
            channel_used["cosine_baseline_fallback"] = len(normalized)

        ranked = sorted(final_scores.items(), key=lambda x: -x[1])
        segments = [self.store.segments[i] for i, _ in ranked]

        metadata: dict = {
            "name": self.arch_name,
            "mode": self.mode,
            "weights": weights,
            "reasoning": reasoning,
            "channels_executed": list(executed.keys()),
            "channel_candidate_counts": channel_used,
            "num_channels_executed": len(executed),
            "raw_weights_response": raw_weights[:500],
            "final_score_top5": [
                (self.store.segments[i].turn_id, float(sc))
                for i, sc in ranked[:5]
            ],
        }
        return MultichResult(segments=segments, metadata=metadata)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------
def build_variant(
    name: str, store: SegmentStore, client: OpenAI | None = None
) -> MultichannelWeighted:
    if name == "multich_llm_weighted":
        return MultichannelWeighted(store, client=client, mode="llm")
    if name == "multich_uniform":
        return MultichannelWeighted(store, client=client, mode="uniform")
    if name == "multich_binary":
        return MultichannelWeighted(store, client=client, mode="binary")
    raise KeyError(name)


VARIANTS = (
    "multich_llm_weighted",
    "multich_uniform",
    "multich_binary",
)
