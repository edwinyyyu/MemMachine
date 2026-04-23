"""Gated overlay v2: extend gated_overlay with intent_parser's unique signals.

This architecture unifies two session wins into a single overlay:

- gated_overlay.py (v2f-primary + confidence-gated displacement from 6 channels
  already) delivered +3.3pp on LoCoMo K=50 with 1W/29T/0L vs meta_v2f.
- intent_parser.IntentParserArch (full plan + stacked signals) delivered
  +1.67pp on LoCoMo K=50 via four UNIQUE signals that aren't in gated_overlay:
  preference_markers (first-person self-statements), list_aggregation
  (expand retrieval pool), negation_markers, and answer_form_date
  (date-specific boost).

v2 keeps the gated overlay's displacement mechanism and prompt, but:
  1. Adds three new supplement channels:
       preference_markers  - boost turns matching first-person self-statements
       negation_markers    - boost turns with "never/not/didn't" patterns
       date_answer_boost   - boost turns with calendar tokens (builds on
                             temporal_tokens but tighter: dates only)
  2. Adds a K-expansion signal (list_aggregation) that is NOT a supplement
     channel. When the LLM's list_aggregation confidence >= threshold, we
     retrieve 1.5x K segments from v2f and fuse before truncating to K.
     The extra slots are filled from v2f's backfill tail, so the "expand"
     path keeps v2f's primary-picks and merely enlarges the supplement pool.
  3. Extends the routing LLM prompt with descriptions of all 9 gated signals
     (6 prior + 3 new channels + list_aggregation K-expander). One routing
     call per query outputs confidences for all 9.

Variants
--------
  gated_v2_all                : all 6 prior channels + 3 new channels + list
                                K-expansion. (Primary variant.)
  gated_v2_intent_only        : only the 3 new channels + list_aggregation;
                                isolates whether the new signals help
                                ALONE relative to intent_parser_full's stacked
                                addition (tests gated integration pattern
                                generalization vs linear fusion).
  gated_v2_minus_critical     : all v2 channels EXCEPT critical_info; tests
                                whether critical_info was redundant given the
                                new preference/negation/date channels on
                                LoCoMo.

Caches
------
Dedicated (writes-only):
  gatedv2_embedding_cache.json
  gatedv2_llm_cache.json
Reads-only (warm-start) from existing shared caches including gated_*
and intent_*.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

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
from multichannel_weighted import (
    _CriticalClassifier,
    extract_query_entities,
    load_speaker_map,
    turn_has_temporal_tokens,
)
# Reuse intent_parser's lexical detectors so the signal coverage is an exact
# match to what earned the +1.67pp in that experiment.
from intent_parser import (
    has_date_tokens,
    has_first_person_preference,
    has_negation_markers,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"

_GATEDV2_EMB_FILE = CACHE_DIR / "gatedv2_embedding_cache.json"
_GATEDV2_LLM_FILE = CACHE_DIR / "gatedv2_llm_cache.json"

# Warm-start from every overlapping cache we have.
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
    "gated_embedding_cache.json",
    "intent_embedding_cache.json",
    "gatedv2_embedding_cache.json",
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
    "gated_llm_cache.json",
    "intent_llm_cache.json",
    "gatedv2_llm_cache.json",
)


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
class GatedV2EmbeddingCache(EmbeddingCache):
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
        self.cache_file = _GATEDV2_EMB_FILE
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


class GatedV2LLMCache(LLMCache):
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
        self.cache_file = _GATEDV2_LLM_FILE
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
# Channel catalog (6 prior + 3 new); list_aggregation is a K-expander.
# ---------------------------------------------------------------------------
# These names match gated_overlay.py's supplement channels so evaluation
# tooling carries over.
PRIOR_CHANNELS = (
    "speaker_filter",
    "alias_context",
    "critical_info",
    "temporal_tokens",
    "entity_exact_match",
)

NEW_CHANNELS = (
    "preference_markers",
    "negation_markers",
    "date_answer_boost",
)

SUPPLEMENT_NAMES_V2 = PRIOR_CHANNELS + NEW_CHANNELS

# list_aggregation is NOT a candidate pool channel; it is a K-expander. We
# still ask the LLM for its confidence and include it in the routing prompt.
K_EXPANDER_NAME = "list_aggregation"
ALL_ROUTED_NAMES = SUPPLEMENT_NAMES_V2 + (K_EXPANDER_NAME,)

SUPPLEMENT_DESCRIPTIONS_V2 = {
    "speaker_filter": (
        "boost turns spoken by a specific named person; confidence high only "
        "if query names a person by first name"
    ),
    "alias_context": (
        "substitute entity aliases; confidence high if query mentions an "
        "entity with known aliases (e.g. 'Dr. Smith' / 'John Smith')"
    ),
    "critical_info": (
        "high if query seeks an enduring fact (medication, deadline, "
        "commitment, preference)"
    ),
    "temporal_tokens": (
        "high if query has temporal constraint (when, after, during, by, "
        "specific date)"
    ),
    "entity_exact_match": (
        "high if query has distinctive proper noun (not common names); "
        "irrelevant for generic queries"
    ),
    "preference_markers": (
        "high if query asks 'what do I like / prefer / always / never / "
        "hate / usually'; boosts first-person self-statements"
    ),
    "negation_markers": (
        "high if query asks about what was NOT done / refused / avoided; "
        "boosts turns containing never/not/didn't/decline/refuse"
    ),
    "date_answer_boost": (
        "high if query asks for a specific calendar date or a 'when "
        "exactly' / 'how long ago' answer; boosts turns containing month "
        "names or numeric dates (tighter than temporal_tokens)"
    ),
    "list_aggregation": (
        "high if query asks for 'all / every / total / list of / overall'; "
        "NOT a candidate pool - instead EXPANDS K by 50% before "
        "truncating to increase coverage of list-style golds"
    ),
}


# ---------------------------------------------------------------------------
# Extended routing prompt. One call per query produces confidences for all
# 6+3 supplements plus the list_aggregation K-expander.
# ---------------------------------------------------------------------------
ROUTING_PROMPT_V2 = """\
You are deciding which retrieval supplement channels to engage for this \
query. The primary channel is v2f (LLM-imagined cue cosine; always active). \
Supplements can OPTIONALLY replace v2f's weakest candidates if they are \
high-confidence for this specific query. One signal (list_aggregation) is \
NOT a candidate channel - it expands K by 50% before truncation.

For each channel, output CONFIDENCE:
- 1.0 = "this channel will definitely find content v2f might miss"
- 0.5 = "might help"
- 0.0 = "irrelevant to this query"

Only channels with confidence >= threshold will be engaged. Be STRICT - \
running irrelevant channels harms retrieval by displacing v2f's strong picks.

Channels:
- speaker_filter: boost turns spoken by a specific named person; confidence \
high only if query names a person by first name
- alias_context: substitute entity aliases; confidence high if query \
mentions an entity with known aliases
- critical_info: high if query seeks an enduring fact (medication, deadline, \
commitment, preference)
- temporal_tokens: high if query has any temporal constraint (when, after, \
during, by)
- entity_exact_match: high if query has distinctive proper noun (not common \
names)
- preference_markers: high if query asks "what do I like/prefer/always/\
never/hate"; boosts first-person self-statements like "I prefer X"
- negation_markers: high if query asks about what was NOT done/refused/\
avoided; boosts turns containing never/not/didn't/refuse/decline
- date_answer_boost: high if query asks for a specific calendar date or \
"when exactly / how long ago" answer; tighter than temporal_tokens (dates \
only, not generic time words)
- list_aggregation: high if query asks for "all/every/list of/total/\
overall"; this EXPANDS K by 50% rather than adding candidates

Query: {query}

Output JSON: {{"speaker_filter": 0.x, "alias_context": 0.x, \
"critical_info": 0.x, "temporal_tokens": 0.x, "entity_exact_match": 0.x, \
"preference_markers": 0.x, "negation_markers": 0.x, \
"date_answer_boost": 0.x, "list_aggregation": 0.x, \
"reasoning": "brief"}}

Output ONLY the JSON object, no prose before or after."""


def parse_confidences_v2(raw: str) -> tuple[dict[str, float], str]:
    """Parse routing JSON. Returns (confidences_for_all_routed, reasoning).

    Fallback: all zeros (no channels engaged, no K expansion).
    """
    default = {ch: 0.0 for ch in ALL_ROUTED_NAMES}
    fallback_reason = "parse_failed_no_supplements"

    if not raw:
        return default, fallback_reason

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

    confs: dict[str, float] = {}
    for ch in ALL_ROUTED_NAMES:
        v = obj.get(ch, 0.0)
        try:
            c = float(v)
        except (TypeError, ValueError):
            c = 0.0
        confs[ch] = max(0.0, min(1.0, c))

    reasoning = str(obj.get("reasoning", "")).strip()[:300]
    return confs, reasoning


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class GatedV2Result:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main architecture
# ---------------------------------------------------------------------------
class GatedOverlayV2:
    """Confidence-gated conditional channel overlay (v2).

    Adds 3 new supplement channels (preference/negation/date) and a K-
    expansion signal (list_aggregation). Same displacement mechanism as
    gated_overlay.py otherwise: v2f is primary; supplements replace v2f's
    lowest-ranked tail slots when their confidence clears the threshold.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        threshold: float = 0.7,
        strict_min: float | None = None,
        allowed_channels: tuple[str, ...] | None = None,
        allow_list_expander: bool = True,
        per_channel_top_m: int = 3,
        per_channel_retrieval_k: int = 20,
        k_expansion_factor: float = 1.5,
        name: str = "gated_v2_all",
    ):
        self.store = store
        self.client = client or OpenAI(timeout=60.0, max_retries=3)
        self.threshold = threshold
        self.strict_min = strict_min
        self.allowed_channels = (
            allowed_channels
            if allowed_channels is not None
            else SUPPLEMENT_NAMES_V2
        )
        self.allow_list_expander = allow_list_expander
        self.per_channel_top_m = per_channel_top_m
        self.per_channel_retrieval_k = per_channel_retrieval_k
        self.k_expansion_factor = k_expansion_factor
        self.arch_name = name

        self.embedding_cache = GatedV2EmbeddingCache()
        self.llm_cache = GatedV2LLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

        # Ingest artifacts
        self.speaker_map = load_speaker_map()
        self.alias_extractor = AliasExtractor(client=self.client)
        self.crit_classifier = _CriticalClassifier(store, self.llm_cache)

        # Per-store masks
        self.role_masks = {
            "user": np.array(
                [s.role == "user" for s in store.segments], dtype=bool
            ),
            "assistant": np.array(
                [s.role == "assistant" for s in store.segments], dtype=bool
            ),
        }
        self.temporal_mask = np.array(
            [turn_has_temporal_tokens(s.text) for s in store.segments],
            dtype=bool,
        )
        self.date_mask = np.array(
            [has_date_tokens(s.text) for s in store.segments], dtype=bool
        )
        self.negation_mask = np.array(
            [has_negation_markers(s.text) for s in store.segments], dtype=bool
        )
        self.preference_mask = np.array(
            [has_first_person_preference(s.text) for s in store.segments],
            dtype=bool,
        )

        # Cache of critical-items per conv (lazy build).
        self._crit_conv_cache: dict[str, list[tuple[int, list[str]]]] = {}

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
        print(f"    LLM call failed: {last_exc}", flush=True)
        self.llm_cache.put(model, prompt, "")
        self.llm_calls += 1
        return ""

    def save_caches(self) -> None:
        try:
            self.embedding_cache.save()
        except Exception as e:
            print(f"  (warn) embedding_cache.save: {e}", flush=True)
        try:
            self.llm_cache.save()
        except Exception as e:
            print(f"  (warn) llm_cache.save: {e}", flush=True)

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    # --- routing ---
    def get_confidences(
        self, query: str
    ) -> tuple[dict[str, float], str, str]:
        prompt = ROUTING_PROMPT_V2.format(query=query)
        raw = self.llm_call(prompt)
        confs, reasoning = parse_confidences_v2(raw)
        return confs, reasoning, raw

    # --- primary channel: v2f ---
    def run_v2f(
        self,
        query: str,
        query_emb: np.ndarray,
        conversation_id: str,
        K: int,
    ) -> tuple[list[Segment], list[str]]:
        """Replicates best_shot.MetaV2f's stacked ordering up to K segments.

        Uses V2f prompt on top-10 raw primer, parses 2 cues, retrieves
        top-10 per cue, then backfills with raw cosine up to K.
        """
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(hop0.segments)
        exclude: set[int] = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )
        prompt = V2F_PROMPT.format(
            question=query, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        if len(all_segments) < K:
            backfill = self.store.search(
                query_emb, top_k=K, conversation_id=conversation_id,
            )
            for seg in backfill.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    if len(all_segments) >= K:
                        break

        return all_segments[:K], cues

    # --- supplement channels (shared cosine helper) ---
    def _cosine_search_in_conv(
        self,
        query_emb: np.ndarray,
        conversation_id: str,
        mask: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        k = top_k or self.per_channel_retrieval_k
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

    # --- prior channels ---
    def ch_speaker_filter(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> list[int]:
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
            return []
        cands = self._cosine_search_in_conv(
            query_emb, conversation_id, mask=mask
        )
        return [i for i, _ in cands]

    def ch_alias_context(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> list[int]:
        groups = self.alias_extractor.get_groups(conversation_id)
        if not groups:
            return []
        matches = find_alias_matches(query, groups)
        if not matches:
            return []
        variants, _ = build_expanded_queries(query, groups)
        probes: list[str] = [v for v in variants if v != query]
        for _matched, siblings in matches:
            for sib in siblings[:4]:
                if sib not in probes:
                    probes.append(sib)
        agg: dict[int, float] = {}
        for text in probes[:8]:
            emb = self.embed_text(text)
            res = self._cosine_search_in_conv(emb, conversation_id)
            for idx, sc in res:
                if idx not in agg or sc > agg[idx]:
                    agg[idx] = sc
        if not agg:
            return []
        ordered = sorted(agg.items(), key=lambda x: -x[1])
        return [i for i, _ in ordered]

    def ch_critical_info(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> list[int]:
        if conversation_id not in self._crit_conv_cache:
            self._crit_conv_cache[conversation_id] = (
                self.crit_classifier.build_critical_set_for_conv(
                    conversation_id
                )
            )
        crit_items = self._crit_conv_cache[conversation_id]
        if not crit_items:
            return []
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
        ordered = sorted(per_parent.items(), key=lambda x: -x[1])
        return [i for i, _ in ordered]

    def ch_temporal_tokens(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> list[int]:
        cands = self._cosine_search_in_conv(
            query_emb, conversation_id, mask=self.temporal_mask
        )
        return [i for i, _ in cands]

    def ch_entity_exact_match(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> list[int]:
        entities = extract_query_entities(query)
        if not entities:
            return []
        conv_mask = self.store.conversation_ids == conversation_id
        ents_lower = [e.lower() for e in entities]
        scores: list[tuple[int, float]] = []
        for i, seg in enumerate(self.store.segments):
            if not conv_mask[i]:
                continue
            tl = seg.text.lower()
            hits = 0
            for e in ents_lower:
                if re.search(r"\b" + re.escape(e) + r"\b", tl):
                    hits += 1
            if hits > 0:
                scores.append((i, float(hits) / float(len(ents_lower))))
        scores.sort(key=lambda x: -x[1])
        return [i for i, _ in scores]

    # --- new channels ---
    def ch_preference_markers(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> list[int]:
        """Cosine over the query, restricted to turns matching the first-
        person-preference regex. Mask is precomputed over the store.
        """
        cands = self._cosine_search_in_conv(
            query_emb, conversation_id, mask=self.preference_mask
        )
        return [i for i, _ in cands]

    def ch_negation_markers(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> list[int]:
        """Cosine over the query, restricted to turns matching the
        negation regex.
        """
        cands = self._cosine_search_in_conv(
            query_emb, conversation_id, mask=self.negation_mask
        )
        return [i for i, _ in cands]

    def ch_date_answer_boost(
        self, query: str, query_emb: np.ndarray, conversation_id: str
    ) -> list[int]:
        """Cosine over the query, restricted to turns containing explicit
        date tokens (month names, numeric dates). Tighter than temporal_
        tokens because it excludes generic time words.
        """
        cands = self._cosine_search_in_conv(
            query_emb, conversation_id, mask=self.date_mask
        )
        return [i for i, _ in cands]

    def _run_supplement(
        self,
        name: str,
        query: str,
        query_emb: np.ndarray,
        conversation_id: str,
    ) -> list[int]:
        fn = {
            "speaker_filter": self.ch_speaker_filter,
            "alias_context": self.ch_alias_context,
            "critical_info": self.ch_critical_info,
            "temporal_tokens": self.ch_temporal_tokens,
            "entity_exact_match": self.ch_entity_exact_match,
            "preference_markers": self.ch_preference_markers,
            "negation_markers": self.ch_negation_markers,
            "date_answer_boost": self.ch_date_answer_boost,
        }[name]
        return fn(query, query_emb, conversation_id)

    # --- overlay assembly ---
    def _overlay(
        self,
        v2f_segments: list[Segment],
        supplement_candidates: dict[str, list[int]],
        channel_m_effective: dict[str, int],
        K: int,
    ) -> tuple[list[Segment], dict]:
        """Replace v2f's weakest tail slots with interleaved supplement
        candidates. Preserves v2f's strongest picks at the top.
        """
        overlay_info = {
            "v2f_size": len(v2f_segments),
            "displacements": {},
            "channels_contributing": [],
        }

        if not supplement_candidates:
            return v2f_segments[:K], overlay_info

        total_displace = min(
            sum(channel_m_effective.values()), max(K - 1, 0)
        )
        if total_displace <= 0:
            return v2f_segments[:K], overlay_info

        v2f_ids = [s.index for s in v2f_segments[:K]]
        v2f_id_set = set(v2f_ids)

        picked: list[tuple[str, int]] = []
        channel_iters: dict[str, list[int]] = {}
        channel_picked_count: dict[str, int] = {}
        for ch, cands in supplement_candidates.items():
            channel_iters[ch] = [c for c in cands if c not in v2f_id_set]
            channel_picked_count[ch] = 0

        used_ids: set[int] = set()
        order_active = list(supplement_candidates.keys())
        while len(picked) < total_displace and order_active:
            new_active: list[str] = []
            for ch in order_active:
                cap = channel_m_effective.get(ch, 0)
                if channel_picked_count[ch] >= cap:
                    continue
                cands = channel_iters[ch]
                chose = None
                for c in cands:
                    if c in used_ids:
                        continue
                    chose = c
                    break
                if chose is None:
                    continue
                channel_iters[ch] = [
                    x for x in channel_iters[ch] if x != chose
                ]
                used_ids.add(chose)
                picked.append((ch, chose))
                channel_picked_count[ch] += 1
                if len(picked) >= total_displace:
                    break
                if channel_picked_count[ch] < cap and channel_iters[ch]:
                    new_active.append(ch)
            order_active = new_active if new_active else []

        overlay_info["displacements"] = dict(channel_picked_count)
        overlay_info["channels_contributing"] = [
            ch for ch, n in channel_picked_count.items() if n > 0
        ]

        if not picked:
            return v2f_segments[:K], overlay_info

        keep = K - len(picked)
        keep = max(keep, 1)
        picked = picked[: K - keep]

        final_ids = v2f_ids[:keep] + [seg_idx for _, seg_idx in picked]
        final_segs = [self.store.segments[i] for i in final_ids]
        return final_segs, overlay_info

    def retrieve(
        self, question: str, conversation_id: str, K: int = 50
    ) -> GatedV2Result:
        confs, reasoning, raw = self.get_confidences(question)
        query_emb = self.embed_text(question)

        # K expansion: if list_aggregation fires, temporarily retrieve with
        # an expanded effective K so supplement channels have more slack and
        # more gold-rich v2f tail to pull from. We still return K segments.
        list_conf = confs.get(K_EXPANDER_NAME, 0.0)
        list_expander_on = (
            self.allow_list_expander
            and list_conf >= self.threshold
        )
        if list_expander_on:
            k_effective = int(math.ceil(K * self.k_expansion_factor))
        else:
            k_effective = K

        # Primary: v2f ordering, computed up to k_effective so the overlay
        # has a larger pool to draw from when list_aggregation fires. The
        # FINAL output is still K segments.
        v2f_segs, v2f_cues = self.run_v2f(
            question, query_emb, conversation_id, k_effective
        )

        # Determine firing supplements.
        firing: list[str] = []
        m_effective: dict[str, int] = {}
        for ch in SUPPLEMENT_NAMES_V2:
            if ch not in self.allowed_channels:
                continue
            c = confs.get(ch, 0.0)
            if c < self.threshold:
                continue
            if self.strict_min is not None and c < self.strict_min:
                continue
            firing.append(ch)
            m_effective[ch] = max(
                1, int(math.ceil(self.per_channel_top_m * c))
            )

        # Run firing supplements.
        supplement_cands: dict[str, list[int]] = {}
        for ch in firing:
            ids = self._run_supplement(
                ch, question, query_emb, conversation_id
            )
            if ids:
                supplement_cands[ch] = ids

        # Overlay at k_effective (so v2f tail that would have been kept
        # becomes the replaceable tail). Then truncate to K.
        final_segs, overlay_info = self._overlay(
            v2f_segs, supplement_cands, m_effective, k_effective
        )
        final_segs = final_segs[:K]

        metadata = {
            "name": self.arch_name,
            "threshold": self.threshold,
            "strict_min": self.strict_min,
            "allowed_channels": list(self.allowed_channels),
            "allow_list_expander": self.allow_list_expander,
            "confidences": confs,
            "reasoning": reasoning,
            "raw_routing_response": raw[:500],
            "firing_channels": firing,
            "m_effective": m_effective,
            "list_expander_on": list_expander_on,
            "k_effective": k_effective,
            "v2f_cues": v2f_cues,
            "overlay": overlay_info,
            "num_firing": len(firing),
        }
        return GatedV2Result(segments=final_segs, metadata=metadata)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
VARIANTS_V2 = (
    "gated_v2_all",
    "gated_v2_intent_only",
    "gated_v2_minus_critical",
)


def build_variant_v2(
    name: str, store: SegmentStore, client: OpenAI | None = None
) -> GatedOverlayV2:
    if name == "gated_v2_all":
        return GatedOverlayV2(
            store,
            client=client,
            threshold=0.7,
            allowed_channels=SUPPLEMENT_NAMES_V2,
            allow_list_expander=True,
            name=name,
        )
    if name == "gated_v2_intent_only":
        return GatedOverlayV2(
            store,
            client=client,
            threshold=0.7,
            allowed_channels=NEW_CHANNELS,
            allow_list_expander=True,
            name=name,
        )
    if name == "gated_v2_minus_critical":
        without_crit = tuple(
            ch for ch in SUPPLEMENT_NAMES_V2 if ch != "critical_info"
        )
        return GatedOverlayV2(
            store,
            client=client,
            threshold=0.7,
            allowed_channels=without_crit,
            allow_list_expander=True,
            name=name,
        )
    raise KeyError(name)
