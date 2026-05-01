"""Confidence-gated conditional channel overlay retrieval.

Pattern
-------
The multichannel_weighted study showed that linear fusion of channel scores
DILUTES v2f's positional strength on LoCoMo (-5pp at K=50) even with sensible
LLM-chosen weights. The underlying issue: when we add a channel's normalized
score to v2f's normalized score, strong v2f picks get knocked out of top-K by
the other channels' outputs even when those outputs aren't actually better.

This architecture uses v2f_cosine as the PRIMARY channel (always runs), and
treats the other 6 channels as SUPPLEMENT CANDIDATES. For each query the LLM
emits a CONFIDENCE score in [0, 1] for each supplement. Supplements fire only
if confidence >= threshold; firing supplements REPLACE v2f's weakest top-K
slots with their own top candidates, preserving v2f's strongest picks.

This generalizes critical_info_store's `always_top_M` pattern across many
channels with LLM-driven per-query gating.

Channels (same 7 as multichannel_weighted):
  1. cosine_baseline : raw query cosine retrieval (NOT used as supplement —
     we keep it for completeness but v2f_cosine dominates)
  2. v2f_cosine      : PRIMARY (always active)
  3. speaker_filter  : SUPPLEMENT
  4. alias_context   : SUPPLEMENT
  5. critical_info   : SUPPLEMENT
  6. temporal_tokens : SUPPLEMENT
  7. entity_exact_match : SUPPLEMENT

One LLM call per query produces CONFIDENCE values for 5 supplements
(cosine_baseline is excluded from the routing — v2f is strictly stronger on
our datasets, so the supplement-role there is degenerate).

Overlay assembly:
  - Base ordering = v2f_cosine's top-K.
  - For each supplement channel c with confidence[c] >= threshold,
    retrieve top-M candidates (M=3 default). Weight by confidence:
        M_effective = ceil(M * confidence[c])
  - Compute total displacement = sum over firing channels of M_effective,
    clipped to K-1 (always keep at least one v2f pick).
  - Displace v2f's LAST `displacement` slots with an interleaving of the
    supplement candidates (round-robin by channel, preserving each
    channel's internal ordering). Supplement items that duplicate v2f
    items higher in the list are skipped.

Variants
--------
  gated_threshold_0.7        : confidence >= 0.7 required
  gated_threshold_0.5        : more permissive (more channels engage)
  gated_replace_strict_0.85  : confidence >= 0.7 to fire, but only items
                               from channels with confidence >= 0.85 are
                               used for actual replacement
  gated_critical_only        : only `critical_info` can fire (replicates
                               the original pattern)

Caches
------
Dedicated:
  gated_embedding_cache.json
  gated_llm_cache.json
  gated_crit_cache.json

Reads shared caches (warm-start).
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
from alias_expansion import (
    AliasExtractor,
    build_expanded_queries,
    find_alias_matches,
)
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import V2F_PROMPT, _format_segments, _parse_cues
from dotenv import load_dotenv
from multichannel_weighted import (
    _CriticalClassifier,
    extract_query_entities,
    load_speaker_map,
    turn_has_temporal_tokens,
)
from openai import OpenAI
from speaker_attributed import extract_name_mentions

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"

_GATED_EMB_FILE = CACHE_DIR / "gated_embedding_cache.json"
_GATED_LLM_FILE = CACHE_DIR / "gated_llm_cache.json"

# Reuse existing caches for warm-start.
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
)


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
class GatedEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _GATED_EMB_FILE
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
        tmp = self.cache_file.parent / (self.cache_file.name + f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


class GatedLLMCache(LLMCache):
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
        self.cache_file = _GATED_LLM_FILE
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
        tmp = self.cache_file.parent / (self.cache_file.name + f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, self.cache_file)
        self._new_entries = {}


# ---------------------------------------------------------------------------
# Supplement channels (v2f is primary, cosine_baseline not used as supplement)
# ---------------------------------------------------------------------------
SUPPLEMENT_NAMES = (
    "speaker_filter",
    "alias_context",
    "critical_info",
    "temporal_tokens",
    "entity_exact_match",
)

SUPPLEMENT_DESCRIPTIONS = {
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
        "high if query has temporal constraint (when, after, during, by, specific date)"
    ),
    "entity_exact_match": (
        "high if query has distinctive proper noun (not common names); "
        "irrelevant for generic queries"
    ),
}

ROUTING_PROMPT = """\
You are deciding which retrieval supplement channels to engage for this \
query. The primary channel is v2f (LLM-imagined cue cosine; always active). \
Supplements can OPTIONALLY replace v2f's weakest candidates if they are \
high-confidence for this specific query.

For each supplement channel, output CONFIDENCE:
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
- temporal_tokens: high if query has temporal constraint (when, after, \
during, by)
- entity_exact_match: high if query has distinctive proper noun (not common \
names)

Query: {query}

Output JSON: {{"speaker_filter": 0.x, "alias_context": 0.x, \
"critical_info": 0.x, "temporal_tokens": 0.x, "entity_exact_match": 0.x, \
"reasoning": "brief"}}

Output ONLY the JSON object, no prose before or after."""


def parse_confidences(raw: str) -> tuple[dict[str, float], str]:
    """Parse routing JSON. Returns (confidences, reasoning). Fallback:
    no supplements engaged."""
    default = dict.fromkeys(SUPPLEMENT_NAMES, 0.0)
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
            text = text[start : end + 1]

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return default, fallback_reason

    if not isinstance(obj, dict):
        return default, fallback_reason

    confs: dict[str, float] = {}
    for ch in SUPPLEMENT_NAMES:
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
class GatedResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main architecture
# ---------------------------------------------------------------------------
class GatedOverlay:
    """Confidence-gated conditional channel overlay.

    Mode:
      threshold  : confidence >= threshold needed to fire
      strict     : confidence >= strict_min to use items for replacement
                   (even if fired at a lower threshold)
      critical_only : only critical_info can fire
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        threshold: float = 0.7,
        strict_min: float | None = None,
        allowed_channels: tuple[str, ...] | None = None,
        per_channel_top_m: int = 3,
        per_channel_retrieval_k: int = 20,
        name: str = "gated_overlay",
    ):
        self.store = store
        self.client = client or OpenAI(timeout=60.0, max_retries=3)
        self.threshold = threshold
        self.strict_min = strict_min
        self.allowed_channels = (
            allowed_channels or SUPPLEMENT_NAMES
        )
        self.per_channel_top_m = per_channel_top_m
        self.per_channel_retrieval_k = per_channel_retrieval_k
        self.arch_name = name

        self.embedding_cache = GatedEmbeddingCache()
        self.llm_cache = GatedLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

        # Ingest artifacts
        self.speaker_map = load_speaker_map()
        self.alias_extractor = AliasExtractor(client=self.client)
        self.crit_classifier = _CriticalClassifier(store, self.llm_cache)

        # Per-store role masks for speaker channel
        self.role_masks = {
            "user": np.array([s.role == "user" for s in store.segments], dtype=bool),
            "assistant": np.array(
                [s.role == "assistant" for s in store.segments], dtype=bool
            ),
        }

        # Per-store precomputed temporal mask
        self.temporal_mask = np.array(
            [turn_has_temporal_tokens(s.text) for s in store.segments],
            dtype=bool,
        )

        # Cache of critical-items per conv
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
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
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
    def get_confidences(self, query: str) -> tuple[dict[str, float], str, str]:
        prompt = ROUTING_PROMPT.format(query=query)
        raw = self.llm_call(prompt)
        confs, reasoning = parse_confidences(raw)
        return confs, reasoning, raw

    # --- primary channel: v2f ---
    def run_v2f(
        self, query: str, query_emb: np.ndarray, conversation_id: str, K: int
    ) -> list[Segment]:
        """Replicates best_shot.MetaV2f's stacked ordering up to K
        segments.

        Uses V2f prompt on top-10 raw primer, parses 2 cues, retrieves
        top-10 per cue, then backfills with raw cosine up to K."""
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments: list[Segment] = list(hop0.segments)
        exclude: set[int] = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(all_segments)
        )
        prompt = V2F_PROMPT.format(question=query, context_section=context_section)
        output = self.llm_call(prompt)
        cues = _parse_cues(output)[:2]

        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        # Backfill with raw cosine up to K (same behavior as fair_backfill)
        if len(all_segments) < K:
            backfill = self.store.search(
                query_emb,
                top_k=K,
                conversation_id=conversation_id,
            )
            for seg in backfill.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    if len(all_segments) >= K:
                        break

        return all_segments[:K], cues

    # --- supplement channels ---
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
        cands = self._cosine_search_in_conv(query_emb, conversation_id, mask=mask)
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
                self.crit_classifier.build_critical_set_for_conv(conversation_id)
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

        Returns (final_segments, overlay_info).
        """
        overlay_info = {
            "v2f_size": len(v2f_segments),
            "displacements": {},
            "channels_contributing": [],
        }

        if not supplement_candidates:
            return v2f_segments[:K], overlay_info

        # Total slots to displace
        total_displace = min(sum(channel_m_effective.values()), max(K - 1, 0))
        if total_displace <= 0:
            return v2f_segments[:K], overlay_info

        # v2f ids in order
        v2f_ids = [s.index for s in v2f_segments[:K]]
        v2f_id_set = set(v2f_ids)

        # Collect supplement candidates not already in v2f's top-K,
        # preserving channel order. Interleave via round-robin across
        # firing channels, up to each channel's M_effective.
        picked: list[tuple[str, int]] = []  # (channel, seg_index)
        channel_iters: dict[str, list[int]] = {}
        channel_picked_count: dict[str, int] = {}
        for ch, cands in supplement_candidates.items():
            channel_iters[ch] = [c for c in cands if c not in v2f_id_set]
            channel_picked_count[ch] = 0

        used_ids: set[int] = set()
        # Round-robin pick
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
                # Remove it from iter & mark used
                channel_iters[ch] = [x for x in channel_iters[ch] if x != chose]
                used_ids.add(chose)
                picked.append((ch, chose))
                channel_picked_count[ch] += 1
                if len(picked) >= total_displace:
                    break
                if channel_picked_count[ch] < cap and channel_iters[ch]:
                    new_active.append(ch)
            order_active = new_active or []

        overlay_info["displacements"] = dict(channel_picked_count)
        overlay_info["channels_contributing"] = [
            ch for ch, n in channel_picked_count.items() if n > 0
        ]

        if not picked:
            return v2f_segments[:K], overlay_info

        # Keep v2f's top (K - len(picked)) items, append picked
        keep = K - len(picked)
        keep = max(keep, 1)  # always keep at least one v2f item
        # Adjust picked if keep logic reduces slots
        picked = picked[: K - keep]

        final_ids = v2f_ids[:keep] + [seg_idx for _, seg_idx in picked]
        final_segs = [self.store.segments[i] for i in final_ids]
        return final_segs, overlay_info

    def retrieve(self, question: str, conversation_id: str, K: int = 50) -> GatedResult:
        confs, reasoning, raw = self.get_confidences(question)
        query_emb = self.embed_text(question)

        # Primary: v2f
        v2f_segs, v2f_cues = self.run_v2f(question, query_emb, conversation_id, K)

        # Determine which supplements fire.
        firing: list[str] = []
        m_effective: dict[str, int] = {}
        for ch in SUPPLEMENT_NAMES:
            if ch not in self.allowed_channels:
                continue
            c = confs.get(ch, 0.0)
            if c < self.threshold:
                continue
            if self.strict_min is not None and c < self.strict_min:
                # fired but ineligible for replacement
                continue
            firing.append(ch)
            m_effective[ch] = max(1, int(math.ceil(self.per_channel_top_m * c)))

        # Run firing supplements
        supplement_cands: dict[str, list[int]] = {}
        for ch in firing:
            ids = self._run_supplement(ch, question, query_emb, conversation_id)
            if ids:
                supplement_cands[ch] = ids

        # Assemble overlay
        final_segs, overlay_info = self._overlay(
            v2f_segs, supplement_cands, m_effective, K
        )

        metadata = {
            "name": self.arch_name,
            "threshold": self.threshold,
            "strict_min": self.strict_min,
            "allowed_channels": list(self.allowed_channels),
            "confidences": confs,
            "reasoning": reasoning,
            "raw_routing_response": raw[:500],
            "firing_channels": firing,
            "m_effective": m_effective,
            "v2f_cues": v2f_cues,
            "overlay": overlay_info,
            "num_firing": len(firing),
        }
        return GatedResult(segments=final_segs, metadata=metadata)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
VARIANTS = (
    "gated_threshold_0.7",
    "gated_threshold_0.5",
    "gated_replace_strict_0.85",
    "gated_critical_only",
)


def build_variant(
    name: str, store: SegmentStore, client: OpenAI | None = None
) -> GatedOverlay:
    if name == "gated_threshold_0.7":
        return GatedOverlay(store, client=client, threshold=0.7, name=name)
    if name == "gated_threshold_0.5":
        return GatedOverlay(store, client=client, threshold=0.5, name=name)
    if name == "gated_replace_strict_0.85":
        return GatedOverlay(
            store,
            client=client,
            threshold=0.7,
            strict_min=0.85,
            name=name,
        )
    if name == "gated_critical_only":
        return GatedOverlay(
            store,
            client=client,
            threshold=0.7,
            allowed_channels=("critical_info",),
            name=name,
        )
    raise KeyError(name)
