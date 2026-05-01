"""Iterative Query Refinement — Hopfield-style attractor variant.

For each question q at budget K:
  1. q_0 = normalize(embed(question))
  2. Retrieve top-10 with q_0, take their stored embeddings {e_1..e_10}
  3. centroid c_0 = normalize(mean(e_i))
  4. q_1 = normalize((1 - beta) * q_0 + beta * c_0)
  5. Retrieve top-K with q_1
  6. (Optional) repeat 2-5 for T=2 iterations

Zero LLM calls in the pure variants; one LLM call in iqr_plus_v2f.

Variants:
  iqr_beta_0.2_t1       — mild pull, 1 iteration
  iqr_beta_0.4_t1       — moderate pull
  iqr_beta_0.6_t1       — strong pull
  iqr_beta_0.4_t2       — moderate pull, 2 iterations (convergence test)
  iqr_beta_0.4_filtered — moderate pull, centroid over cos > median
  iqr_plus_v2f          — produce q_refined (beta 0.4, t1), then run v2f cue
                           generation using q_refined as initial probe
"""

from __future__ import annotations

import json

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
# Dedicated caches — read many shared caches for hits, write only to own files
# ---------------------------------------------------------------------------

_IQR_EMB_FILE = CACHE_DIR / "iqr_embedding_cache.json"
_IQR_LLM_FILE = CACHE_DIR / "iqr_llm_cache.json"

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
    "iqr_embedding_cache.json",
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
    "iqr_llm_cache.json",
)


class IqrEmbeddingCache(EmbeddingCache):
    """Reads shared embedding caches (best-effort), writes to dedicated file."""

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
        self.cache_file = _IQR_EMB_FILE
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


class IqrLLMCache(LLMCache):
    """Reads shared LLM caches (best-effort), writes to dedicated file."""

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
        self.cache_file = _IQR_LLM_FILE
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
# Helpers
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v
    return v / n


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _IQRBase(BestshotBase):
    """Iterative query refinement base class.

    Mechanism (zero LLM calls):
      q_t -> cosine top-N -> centroid of retrieved embeddings c_t
      q_{t+1} = normalize((1 - beta) * q_t + beta * c_t)
    After T iterations, retrieve top-K with q_T. We always keep q_0's top-10
    as the merged pool so no gold is lost to refinement drift (the final
    ranking is by max score across {q_0, q_T}).
    """

    arch_name: str = "iqr"
    beta: float = 0.4
    num_iterations: int = 1
    centroid_top_n: int = 10  # number of retrieved turns to average
    filter_by_median: bool = False  # filter centroid input by median-cosine
    max_return: int = 100  # max segments to return (eval uses K up to 50)

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        super().__init__(store, client)
        self.embedding_cache = IqrEmbeddingCache()
        self.llm_cache = IqrLLMCache()

    def _centroid(
        self,
        segments: list[Segment],
        scores: list[float],
    ) -> np.ndarray | None:
        if not segments:
            return None

        # Gather normalized stored embeddings (store keeps normalized versions)
        vecs = np.stack([self.store.normalized_embeddings[s.index] for s in segments])
        if self.filter_by_median and len(scores) >= 2:
            median = float(np.median(scores))
            mask = np.array([sc >= median for sc in scores])
            if mask.sum() == 0:
                filtered = vecs
            else:
                filtered = vecs[mask]
        else:
            filtered = vecs
        mean_vec = filtered.mean(axis=0)
        return _normalize(mean_vec)

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        # Step 1: initial query embedding
        q_raw = self.embed_text(question)
        q_0 = _normalize(q_raw)

        # Hop 0: top-N for centroid + max-score merge
        hop0 = self.store.search(
            q_0, top_k=self.centroid_top_n, conversation_id=conversation_id
        )
        if not hop0.segments:
            return BestshotResult(
                segments=[],
                metadata={
                    "name": self.arch_name,
                    "beta": self.beta,
                    "num_iterations": self.num_iterations,
                    "filter_by_median": self.filter_by_median,
                    "hop0_empty": True,
                },
            )

        # Track max score per index across all probe rounds (q_0, q_1, ...)
        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}
        for seg, sc in zip(hop0.segments, hop0.scores):
            score_map[seg.index] = sc
            seg_map[seg.index] = seg

        # Iterative refinement
        q_t = q_0
        centroids: list[list[float]] = []  # for diagnostics
        q_history: list[list[float]] = [q_0.tolist()]
        hop_segments = hop0.segments
        hop_scores = hop0.scores

        for t in range(self.num_iterations):
            c_t = self._centroid(hop_segments, hop_scores)
            if c_t is None:
                break
            centroids.append(c_t.tolist())
            q_next = _normalize((1 - self.beta) * q_t + self.beta * c_t)
            q_history.append(q_next.tolist())
            # Retrieve with refined query
            res = self.store.search(
                q_next,
                top_k=max(self.centroid_top_n, self.max_return),
                conversation_id=conversation_id,
            )
            for seg, sc in zip(res.segments, res.scores):
                if seg.index not in score_map or sc > score_map[seg.index]:
                    score_map[seg.index] = sc
                if seg.index not in seg_map:
                    seg_map[seg.index] = seg
            # Next iteration uses top-N of refined retrieval
            hop_segments = res.segments[: self.centroid_top_n]
            hop_scores = res.scores[: self.centroid_top_n]
            q_t = q_next

        # Rank by max score across all probe rounds
        ranked = sorted(score_map.keys(), key=lambda idx: score_map[idx], reverse=True)
        all_segments = [seg_map[idx] for idx in ranked][: self.max_return]

        # Build q_refined for diagnostics
        q_refined = np.array(q_history[-1], dtype=np.float32)

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": self.arch_name,
                "beta": self.beta,
                "num_iterations": self.num_iterations,
                "filter_by_median": self.filter_by_median,
                "q_0": q_0.tolist(),
                "q_refined": q_refined.tolist(),
                "centroids": centroids,
                "hop0_turn_ids": [s.turn_id for s in hop0.segments],
                "num_probes": 1 + len(centroids),
                "hop0_empty": False,
            },
        )


# ---------------------------------------------------------------------------
# Pure IQR variants (zero LLM calls)
# ---------------------------------------------------------------------------


class IqrBeta02T1(_IQRBase):
    arch_name = "iqr_beta_0.2_t1"
    beta = 0.2
    num_iterations = 1


class IqrBeta04T1(_IQRBase):
    arch_name = "iqr_beta_0.4_t1"
    beta = 0.4
    num_iterations = 1


class IqrBeta06T1(_IQRBase):
    arch_name = "iqr_beta_0.6_t1"
    beta = 0.6
    num_iterations = 1


class IqrBeta04T2(_IQRBase):
    arch_name = "iqr_beta_0.4_t2"
    beta = 0.4
    num_iterations = 2


class IqrBeta04Filtered(_IQRBase):
    arch_name = "iqr_beta_0.4_filtered"
    beta = 0.4
    num_iterations = 1
    filter_by_median = True


# ---------------------------------------------------------------------------
# iqr_plus_v2f — use refined query as initial probe, then v2f cue generation
# ---------------------------------------------------------------------------


class IqrPlusV2f(BestshotBase):
    """Produce q_refined via beta=0.4, t=1 IQR; then run v2f cue generation.

    Hop0 context for v2f is the top-10 retrieved by q_refined (not q_0), so
    the LLM sees the same segments it would in a normal v2f run but anchored
    at the refined probe. Final merge: max score across {q_0, q_refined, 2 v2f cues}.
    """

    arch_name = "iqr_plus_v2f"
    beta = 0.4
    num_iterations = 1
    centroid_top_n = 10

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        super().__init__(store, client)
        self.embedding_cache = IqrEmbeddingCache()
        self.llm_cache = IqrLLMCache()

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        q_raw = self.embed_text(question)
        q_0 = _normalize(q_raw)

        hop0 = self.store.search(
            q_0, top_k=self.centroid_top_n, conversation_id=conversation_id
        )
        if not hop0.segments:
            return BestshotResult(
                segments=[],
                metadata={
                    "name": self.arch_name,
                    "hop0_empty": True,
                    "num_probes": 1,
                },
            )

        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}
        for seg, sc in zip(hop0.segments, hop0.scores):
            score_map[seg.index] = sc
            seg_map[seg.index] = seg

        # Centroid of hop0 embeddings
        vecs = np.stack(
            [self.store.normalized_embeddings[s.index] for s in hop0.segments]
        )
        c_0 = _normalize(vecs.mean(axis=0))
        q_1 = _normalize((1 - self.beta) * q_0 + self.beta * c_0)

        # Retrieve with refined query
        hop1 = self.store.search(
            q_1, top_k=self.centroid_top_n, conversation_id=conversation_id
        )
        for seg, sc in zip(hop1.segments, hop1.scores):
            if seg.index not in score_map or sc > score_map[seg.index]:
                score_map[seg.index] = sc
            if seg.index not in seg_map:
                seg_map[seg.index] = seg

        # v2f cue generation using hop1 segments as context (anchored at q_refined)
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(hop1.segments)
        )
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        v2f_cues = _parse_cues(v2f_output)[:2]

        for cue in v2f_cues:
            cue_emb = self.embed_text(cue)
            res = self.store.search(cue_emb, top_k=10, conversation_id=conversation_id)
            for seg, sc in zip(res.segments, res.scores):
                if seg.index not in score_map or sc > score_map[seg.index]:
                    score_map[seg.index] = sc
                if seg.index not in seg_map:
                    seg_map[seg.index] = seg

        ranked = sorted(score_map.keys(), key=lambda idx: score_map[idx], reverse=True)
        all_segments = [seg_map[idx] for idx in ranked]

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": self.arch_name,
                "beta": self.beta,
                "num_iterations": self.num_iterations,
                "q_0": q_0.tolist(),
                "q_refined": q_1.tolist(),
                "centroids": [c_0.tolist()],
                "v2f_cues": v2f_cues,
                "hop0_turn_ids": [s.turn_id for s in hop0.segments],
                "num_probes": 2 + len(v2f_cues),
                "hop0_empty": False,
            },
        )


ARCH_CLASSES: dict[str, type] = {
    "iqr_beta_0.2_t1": IqrBeta02T1,
    "iqr_beta_0.4_t1": IqrBeta04T1,
    "iqr_beta_0.6_t1": IqrBeta06T1,
    "iqr_beta_0.4_t2": IqrBeta04T2,
    "iqr_beta_0.4_filtered": IqrBeta04Filtered,
    "iqr_plus_v2f": IqrPlusV2f,
}
