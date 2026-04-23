"""Contrastive retrieval probe.

Per-cue attribution showed the dominant loser archetype is "interrogative
paraphrase" — cues staying near the query embedding and far from the gold
answer. These retrieve turns that ALSO paraphrase the question: high cosine
to query, semantically wrong.

This module tests: generate a DISTRACTOR probe (text that looks similar to
the question but does NOT contain the answer), then score candidate turns
by:

    score(t) = cosine(answer_emb, t) - alpha * cosine(distractor_emb, t)

Turns matching both probes (paraphrase-style matches) get penalized; turns
matching only the answer-probe get surfaced.

Two variants per alpha:
  contrast_only — raw query_emb as answer-probe, contrastive scoring over
                  the full conversation, pick top-K.
  contrast_v2f  — v2f retrieval (with its two cues) produces a candidate
                  pool; we re-rank it by the contrastive score against
                  (query_emb, distractor_emb) and return in re-ranked order.
                  Dedicated distractor probe; answer side is the query
                  embedding so the v2f cue merging stays intact.

All writes go to contrast_* dedicated caches; reads pull from shared caches
to maximize hit rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
    BestshotBase,
    BestshotResult,
    V2F_PROMPT,
    _format_segments,
    _parse_cues,
)

MODEL = "gpt-5-mini"

# ---------------------------------------------------------------------------
# Dedicated caches
# ---------------------------------------------------------------------------

_CONTRAST_EMB_FILE = CACHE_DIR / "contrast_embedding_cache.json"
_CONTRAST_LLM_FILE = CACHE_DIR / "contrast_llm_cache.json"

_SHARED_EMB_READ = (
    "embedding_cache.json",
    "arch_embedding_cache.json",
    "agent_embedding_cache.json",
    "frontier_embedding_cache.json",
    "meta_embedding_cache.json",
    "optim_embedding_cache.json",
    "synth_test_embedding_cache.json",
    "bestshot_embedding_cache.json",
    "antipara_embedding_cache.json",
    "contrast_embedding_cache.json",
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
    "antipara_llm_cache.json",
    "contrast_llm_cache.json",
)


class ContrastEmbeddingCache(EmbeddingCache):
    """Reads shared caches, writes only to contrast_embedding_cache.json."""

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
        self.cache_file = _CONTRAST_EMB_FILE
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


class ContrastLLMCache(LLMCache):
    """Reads shared caches, writes only to contrast_llm_cache.json."""

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
        self.cache_file = _CONTRAST_LLM_FILE
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
# Distractor probe generation
# ---------------------------------------------------------------------------

DISTRACTOR_PROMPT = """\
You are generating a DISTRACTOR probe for retrieval. Given a question, \
write text that LOOKS similar to the question (shares its vocabulary and \
theme) but does NOT contain the answer. Examples of distractors: rephrased \
versions of the question, statements where the topic is mentioned but not \
resolved, tangential discussions, greetings or meta-commentary about the \
topic.

Question: {question}

Generate 1 distractor text, 1-2 sentences:
DISTRACTOR: <text>"""


def _parse_distractor(response: str) -> str:
    """Pull the DISTRACTOR line from the LLM response."""
    for line in response.strip().split("\n"):
        s = line.strip()
        if s.upper().startswith("DISTRACTOR:"):
            return s.split(":", 1)[1].strip()
    # Fallback: first non-empty line that isn't a label.
    for line in response.strip().split("\n"):
        s = line.strip()
        if s and not s.endswith(":"):
            return s
    return ""


# ---------------------------------------------------------------------------
# Shared base — adds distractor generation + contrastive scoring
# ---------------------------------------------------------------------------


@dataclass
class ContrastiveMeta:
    answer_probe: str
    distractor_probe: str
    cues: list[str] = field(default_factory=list)
    alpha: float = 0.0
    # Sample of (turn_id, cos_answer, cos_distractor, final_score) for
    # top-N candidates, for qualitative inspection.
    sample_scores: list[dict] = field(default_factory=list)


class _ContrastiveBase(BestshotBase):
    """Shares distractor generation + contrastive scoring utilities."""

    def __init__(
        self,
        store: SegmentStore,
        alpha: float,
        client: OpenAI | None = None,
    ):
        super().__init__(store, client)
        self.embedding_cache = ContrastEmbeddingCache()
        self.llm_cache = ContrastLLMCache()
        self.alpha = alpha

    def _distractor_text(self, question: str) -> str:
        prompt = DISTRACTOR_PROMPT.format(question=question)
        output = self.llm_call(prompt)
        return _parse_distractor(output)

    def _unit(self, emb: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(emb))
        if n < 1e-10:
            return emb
        return emb / n

    def _contrastive_score_indices(
        self,
        answer_emb: np.ndarray,
        distractor_emb: np.ndarray,
        conversation_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (indices, scores, cos_answer, cos_distractor) for all
        segments in the given conversation, sorted by contrastive score
        descending. Convenience for contrast_only mode.

        Returns arrays of length N (#segments in conversation), where
        indices is the segment-store index.
        """
        # SegmentStore stores normalized_embeddings so we can do dot products
        # directly against a normalized probe.
        a = self._unit(answer_emb)
        d = self._unit(distractor_emb)

        sim_a = self.store.normalized_embeddings @ a
        sim_d = self.store.normalized_embeddings @ d

        # Conversation mask
        mask = self.store.conversation_ids == conversation_id
        score = sim_a - self.alpha * sim_d
        # Exclude out-of-conversation segments by pushing very low.
        score_masked = np.where(mask, score, -1e9)

        order = np.argsort(score_masked)[::-1]
        order = order[score_masked[order] > -1e8]
        return order, score_masked[order], sim_a[order], sim_d[order]


# ---------------------------------------------------------------------------
# contrast_only — use raw query embedding as answer-probe, re-rank whole conv
# ---------------------------------------------------------------------------


class ContrastiveOnly(_ContrastiveBase):
    """Uses query embedding as answer-probe; re-ranks the full conversation
    by the contrastive score, returns top-K (up to max eval budget)."""

    arch_name_tmpl = "contrast_only_a{alpha:g}"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        distractor = self._distractor_text(question)
        if not distractor:
            distractor_emb = np.zeros_like(query_emb)
        else:
            distractor_emb = self.embed_text(distractor)

        order, scores, sim_a, sim_d = self._contrastive_score_indices(
            query_emb, distractor_emb, conversation_id
        )

        # Return the top 50 (the eval caps at max BUDGETS = 50; we give the
        # full ordered list so fair_backfill has plenty). Trim to 100 for
        # safety — fair_backfill only looks at top-K anyway.
        top_indices = order[:100]
        segments = [self.store.segments[int(i)] for i in top_indices]

        # Gather sample scores for the top 20 for qualitative diagnostics.
        sample = []
        for rank, i in enumerate(top_indices[:20]):
            seg = self.store.segments[int(i)]
            sample.append(
                {
                    "rank": rank,
                    "turn_id": seg.turn_id,
                    "role": seg.role,
                    "text_preview": seg.text[:160],
                    "cos_answer": float(sim_a[rank]),
                    "cos_distractor": float(sim_d[rank]),
                    "score": float(scores[rank]),
                }
            )

        return BestshotResult(
            segments=segments,
            metadata={
                "name": self.arch_name_tmpl.format(alpha=self.alpha),
                "alpha": self.alpha,
                "answer_probe": question,
                "distractor_probe": distractor,
                "cues": [],
                "sample_scores": sample,
            },
        )


# ---------------------------------------------------------------------------
# contrast_v2f — v2f retrieval + contrastive re-scoring of candidate pool
# ---------------------------------------------------------------------------


class ContrastiveV2F(_ContrastiveBase):
    """V2f's hop0-top-10 + 2 cues top-10 retrieval builds a candidate pool;
    we re-rank the pool by contrastive score (query vs distractor) and
    return segments in that re-ranked order.

    Rationale: the answer-probe is the query embedding (same as what v2f
    uses for hop0); the v2f cues still bring in associative content, but
    the ordering of the pool is contrastive."""

    arch_name_tmpl = "contrast_v2f_a{alpha:g}"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)

        # --- v2f retrieval (identical logic to MetaV2f in best_shot.py) ---
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        pool: list[Segment] = list(hop0.segments)
        exclude = {s.index for s in pool}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(pool)
        )
        prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
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
                    pool.append(seg)
                    exclude.add(seg.index)

        # --- contrastive re-scoring of pool ---
        distractor = self._distractor_text(question)
        if not distractor:
            distractor_emb = np.zeros_like(query_emb)
        else:
            distractor_emb = self.embed_text(distractor)

        a = self._unit(query_emb)
        d = self._unit(distractor_emb)

        scored: list[tuple[float, float, float, Segment]] = []
        for seg in pool:
            e = self.store.normalized_embeddings[seg.index]
            ca = float(np.dot(e, a))
            cd = float(np.dot(e, d))
            score = ca - self.alpha * cd
            scored.append((score, ca, cd, seg))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [s[3] for s in scored]

        sample = []
        for rank, (score, ca, cd, seg) in enumerate(scored[:20]):
            sample.append(
                {
                    "rank": rank,
                    "turn_id": seg.turn_id,
                    "role": seg.role,
                    "text_preview": seg.text[:160],
                    "cos_answer": ca,
                    "cos_distractor": cd,
                    "score": score,
                }
            )

        return BestshotResult(
            segments=reranked,
            metadata={
                "name": self.arch_name_tmpl.format(alpha=self.alpha),
                "alpha": self.alpha,
                "answer_probe": question,
                "distractor_probe": distractor,
                "cues": cues,
                "output": output,
                "sample_scores": sample,
                "pool_size": len(pool),
            },
        )


# ---------------------------------------------------------------------------
# Baselines — v2f reference and a pure cosine baseline (hop0 only, no LLM)
# ---------------------------------------------------------------------------


class V2FReference(_ContrastiveBase):
    """V2f baseline using contrast_* caches so results align with the
    contrastive arms. alpha is ignored; kept to share the cache plumbing."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        super().__init__(store, alpha=0.0, client=client)

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments = list(hop0.segments)
        exclude = {s.index for s in all_segments}

        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(all_segments)
        )
        prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
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

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": "v2f",
                "output": output,
                "cues": cues,
            },
        )


class CosineBaseline(_ContrastiveBase):
    """Pure cosine top-K baseline (no LLM, no cues). alpha ignored."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        super().__init__(store, alpha=0.0, client=client)

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        query_emb = self.embed_text(question)
        # 100 > max(BUDGETS) = 50 with margin
        result = self.store.search(
            query_emb, top_k=100, conversation_id=conversation_id
        )
        return BestshotResult(
            segments=list(result.segments),
            metadata={"name": "cosine_baseline", "cues": []},
        )
