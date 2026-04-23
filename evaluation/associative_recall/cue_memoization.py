"""Cue memoization — retrieve-and-reuse past successful v2f cues as probes.

Motivation
----------
Few-shot cue generation (see `fewshot_cue_gen.py`) asked the LLM to IMITATE
exemplar cues' style. It often drifted — the LLM fabricated corpus-specific
vocabulary it had no way of knowing. MMR pushed cues away from their mode and
lost relevance; spreading activation / anchor derived probes from v2f-reachable
content so the probe space never left v2f's basin.

Cue memoization skips the LLM adaptation step entirely. For each new question:
  1. Embed the query.
  2. Find the top-M most embedding-similar PAST questions from the exemplar
     bank (leave-one-out: exclude any exemplar whose conversation_id matches
     the query's).
  3. Collect their stored cues verbatim.
  4. Use those cues DIRECTLY as embedding probes (embed each, top-K per probe).
  5. Merge with the query-cosine top-K by max-score.

The probes are sourced from an INDEPENDENT distribution (past successful runs
across different questions/conversations), so their retrievals are orthogonal
to v2f's content-imagination basin.

Variants
--------
  memoize_m2          — 2 nearest exemplars (probes = union of their cues).
  memoize_m3          — 3 nearest exemplars.
  memoize_filtered    — only keep exemplars with cosine(new_q, exemplar_q) >
                         threshold (0.5). If nothing passes, fall back to v2f.
  memoize_plus_v2f    — run v2f AND memoized probes; union via max-cosine merge.

Caches
------
Dedicated `memoize_embedding_cache.json` / `memoize_llm_cache.json` — reads
from shared caches for hits, writes only to dedicated files.

Zero new LLM calls expected: cue probes come from the pre-built bank, and v2f
prompts (for the +v2f variant) are fully covered by existing caches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from associative_recall import (
    CACHE_DIR,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from best_shot import (
    BestshotBase,
    BestshotResult,
    V2F_PROMPT,
    _format_segments,
    _parse_cues,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BANK_PATH = RESULTS_DIR / "fewshot_exemplar_bank.json"


# ---------------------------------------------------------------------------
# Dedicated caches (read-shared, write-local) — isolate from sibling agents.
# ---------------------------------------------------------------------------

_MEMOIZE_EMB_FILE = CACHE_DIR / "memoize_embedding_cache.json"
_MEMOIZE_LLM_FILE = CACHE_DIR / "memoize_llm_cache.json"

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
    "anchor_embedding_cache.json",
    "alias_embedding_cache.json",
    "mmr_embedding_cache.json",
    "memoize_embedding_cache.json",
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
    "anchor_llm_cache.json",
    "alias_llm_cache.json",
    "mmr_llm_cache.json",
    "memoize_llm_cache.json",
)


class MemoizeEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _MEMOIZE_EMB_FILE
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


class MemoizeLLMCache(LLMCache):
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
        self.cache_file = _MEMOIZE_LLM_FILE
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
# Exemplar bank loader
# ---------------------------------------------------------------------------


def load_exemplar_bank() -> list[dict]:
    """Load exemplar bank with pre-computed question embeddings (normalized)."""
    if not BANK_PATH.exists():
        raise FileNotFoundError(
            f"Exemplar bank missing: {BANK_PATH}. "
            "Run build_exemplar_bank.py first."
        )
    with open(BANK_PATH) as f:
        data = json.load(f)
    exemplars = data["exemplars"]
    for ex in exemplars:
        emb = np.array(ex["question_embedding"], dtype=np.float32)
        n = np.linalg.norm(emb)
        ex["_embedding_norm"] = emb / max(n, 1e-10)
    return exemplars


def select_nearest_exemplars(
    query_emb: np.ndarray,
    exemplars: list[dict],
    k: int,
    exclude_conv_id: str,
    threshold: float | None = None,
) -> list[tuple[dict, float]]:
    """Return top-k (exemplar, cosine_sim) pairs, leaving out the query's conv.

    If `threshold` is set, only exemplars with sim > threshold are kept; k is
    still respected as an upper bound.
    """
    candidates = [e for e in exemplars if e["conversation_id"] != exclude_conv_id]
    if not candidates:
        return []
    q_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
    sims = np.array([float(q_norm @ e["_embedding_norm"]) for e in candidates])
    order = np.argsort(sims)[::-1]
    picked: list[tuple[dict, float]] = []
    for idx in order:
        if len(picked) >= k:
            break
        s = float(sims[idx])
        if threshold is not None and s < threshold:
            break
        picked.append((candidates[idx], s))
    return picked


# ---------------------------------------------------------------------------
# Base architecture
# ---------------------------------------------------------------------------


class _MemoizationBase(BestshotBase):
    """Retrieve hop0 (cosine top-10), then use past cues as extra probes.

    Uses score-based merging: maintains `score_map[index] = max cosine` across
    all probes and sorts segments by max score. This lets `memoize_plus_v2f`
    combine v2f cues and memoized cues symmetrically.
    """

    arch_name: str = "memoize_base"
    m_exemplars: int = 2
    per_probe_top_k: int = 10
    sim_threshold: float | None = None  # if set, filter exemplars by similarity
    include_v2f: bool = False
    v2f_fallback_on_empty: bool = False  # memoize_filtered: fall back to v2f

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        exemplars: list[dict] | None = None,
    ):
        super().__init__(store, client)
        # Swap in dedicated caches to isolate from sibling agents
        self.embedding_cache = MemoizeEmbeddingCache()
        self.llm_cache = MemoizeLLMCache()
        self.exemplars = (
            exemplars if exemplars is not None else load_exemplar_bank()
        )

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        hop0_segments = list(hop0.segments)
        hop0_scores = list(hop0.scores)

        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}
        for seg, sc in zip(hop0_segments, hop0_scores):
            score_map[seg.index] = sc
            seg_map[seg.index] = seg

        # Select exemplars (leave-one-out on conversation_id).
        selected = select_nearest_exemplars(
            query_emb,
            self.exemplars,
            k=self.m_exemplars,
            exclude_conv_id=conversation_id,
            threshold=self.sim_threshold,
        )

        # Gather memoized cues (deduped, order preserved by exemplar rank).
        memoized_cues: list[str] = []
        memoized_cue_sources: list[dict] = []
        seen_cue_texts: set[str] = set()
        for ex, sim in selected:
            for cue in ex.get("cues", []):
                key = cue.strip().lower()
                if not key or key in seen_cue_texts:
                    continue
                seen_cue_texts.add(key)
                memoized_cues.append(cue)
                memoized_cue_sources.append(
                    {
                        "cue": cue,
                        "source_question": ex["question"],
                        "source_conv_id": ex["conversation_id"],
                        "source_dataset": ex["dataset"],
                        "source_category": ex["category"],
                        "source_sim": round(sim, 4),
                    }
                )

        # Retrieve per memoized cue
        memoized_probe_outcomes: list[dict] = []
        for src in memoized_cue_sources:
            cue = src["cue"]
            cue_emb = self.embed_text(cue)
            res = self.store.search(
                cue_emb,
                top_k=self.per_probe_top_k,
                conversation_id=conversation_id,
            )
            retrieved_ids = []
            for seg, sc in zip(res.segments, res.scores):
                retrieved_ids.append(seg.index)
                if seg.index not in score_map or sc > score_map[seg.index]:
                    score_map[seg.index] = sc
                if seg.index not in seg_map:
                    seg_map[seg.index] = seg
            memoized_probe_outcomes.append(
                {
                    **src,
                    "retrieved_turn_ids": [
                        seg_map[idx].turn_id for idx in retrieved_ids
                    ],
                }
            )

        # Optional v2f (for memoize_plus_v2f OR fallback when filter is empty)
        v2f_cues: list[str] = []
        v2f_outcomes: list[dict] = []
        ran_v2f = False
        should_run_v2f = self.include_v2f or (
            self.v2f_fallback_on_empty and not memoized_cues
        )

        if should_run_v2f:
            context_section = (
                "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
                + _format_segments(hop0_segments)
            )
            prompt = V2F_PROMPT.format(
                question=question, context_section=context_section
            )
            output = self.llm_call(prompt)
            v2f_cues = _parse_cues(output)[:2]
            ran_v2f = True
            for cue in v2f_cues:
                if not cue.strip():
                    continue
                cue_emb = self.embed_text(cue)
                res = self.store.search(
                    cue_emb,
                    top_k=self.per_probe_top_k,
                    conversation_id=conversation_id,
                )
                retrieved_ids = []
                for seg, sc in zip(res.segments, res.scores):
                    retrieved_ids.append(seg.index)
                    if seg.index not in score_map or sc > score_map[seg.index]:
                        score_map[seg.index] = sc
                    if seg.index not in seg_map:
                        seg_map[seg.index] = seg
                v2f_outcomes.append(
                    {
                        "cue": cue,
                        "retrieved_turn_ids": [
                            seg_map[idx].turn_id for idx in retrieved_ids
                        ],
                    }
                )

        # Rank by max score
        ranked_indices = sorted(
            score_map.keys(), key=lambda i: score_map[i], reverse=True
        )
        all_segments = [seg_map[i] for i in ranked_indices]

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": self.arch_name,
                "hop0_size": len(hop0_segments),
                "nearest_exemplars": [
                    {
                        "question": ex["question"],
                        "conversation_id": ex["conversation_id"],
                        "dataset": ex["dataset"],
                        "category": ex["category"],
                        "sim": round(sim, 4),
                    }
                    for ex, sim in selected
                ],
                "memoized_cues": memoized_cues,
                "memoized_probe_outcomes": memoized_probe_outcomes,
                "v2f_cues": v2f_cues,
                "v2f_outcomes": v2f_outcomes,
                "ran_v2f": ran_v2f,
                "sim_threshold": self.sim_threshold,
                "m_exemplars": self.m_exemplars,
                "num_probes": (
                    1 + len(memoized_cues) + len(v2f_cues)
                ),
                "nearest_exemplar_sim": (
                    round(float(selected[0][1]), 4) if selected else None
                ),
            },
        )


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


class MemoizeM2(_MemoizationBase):
    """Top-2 nearest exemplars, all their cues used as probes. Zero LLM."""

    arch_name = "memoize_m2"
    m_exemplars = 2
    per_probe_top_k = 10
    sim_threshold = None
    include_v2f = False
    v2f_fallback_on_empty = False


class MemoizeM3(_MemoizationBase):
    """Top-3 nearest exemplars (more cues → broader probe coverage). Zero LLM."""

    arch_name = "memoize_m3"
    m_exemplars = 3
    per_probe_top_k = 10
    sim_threshold = None
    include_v2f = False
    v2f_fallback_on_empty = False


class MemoizeFiltered(_MemoizationBase):
    """Only use exemplars with similarity > 0.5 to the new query.

    If no exemplars pass the threshold, fall back to v2f (1 LLM call).
    """

    arch_name = "memoize_filtered"
    m_exemplars = 3  # upper bound when many pass threshold
    per_probe_top_k = 10
    sim_threshold = 0.5
    include_v2f = False
    v2f_fallback_on_empty = True


class MemoizePlusV2f(_MemoizationBase):
    """Union: top-2 memoized exemplars' cues + v2f cues merged by max-cosine."""

    arch_name = "memoize_plus_v2f"
    m_exemplars = 2
    per_probe_top_k = 10
    sim_threshold = None
    include_v2f = True
    v2f_fallback_on_empty = False


ARCH_CLASSES: dict[str, type] = {
    "memoize_m2": MemoizeM2,
    "memoize_m3": MemoizeM3,
    "memoize_filtered": MemoizeFiltered,
    "memoize_plus_v2f": MemoizePlusV2f,
}
