"""Inverse query generation — orthogonal multi-probe retrieval.

For each question q at budget K:
  1. Initial cosine top-10 from raw query embedding.
  2. LLM generates 1-2 "what question would this turn answer?" per retrieved
     turn (batched per question for main variant, batched for top-3 variant).
  3. Embed each generated question; retrieve top-5 per generated question.
  4. Merge by max score across probes (raw query + inverse queries); truncate K.

Variants:
  inverse_query       — batch 10 initial turns in ONE LLM call
  inverse_query_top3  — batch 3 initial turns in ONE LLM call (cheaper)
  inverse_query_v2f   — v2f cues + inverse queries on v2f's hop0; sum-cosine union

Motivation:
  LoCoMo gold sits at +0.14 cosine off the query (kNN-adjacency only 36%).
  Inverse query generation starts from retrieved content and works backward —
  the generated question is anchored in actual corpus text, not user phrasing.
  Orthogonal to v2f which imagines chat content from query forward.

Dedicated caches (inv_query_*_cache.json) avoid concurrent-agent corruption.
"""

from __future__ import annotations

import json
import re

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

_INV_EMB_FILE = CACHE_DIR / "inv_query_embedding_cache.json"
_INV_LLM_FILE = CACHE_DIR / "inv_query_llm_cache.json"

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
)


class InvQueryEmbeddingCache(EmbeddingCache):
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
        self.cache_file = _INV_EMB_FILE
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


class InvQueryLLMCache(LLMCache):
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
        self.cache_file = _INV_LLM_FILE
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
# Prompts
# ---------------------------------------------------------------------------

# Batched inverse-question generation. One LLM call per question for up to 10
# retrieved turns. Asks 1-2 questions per turn using specific vocabulary.
INVERSE_BATCH_PROMPT = """\
You will see a list of conversation turns that a retrieval system pulled \
from a long chat history. For EACH turn, generate 1-2 questions that \
someone might later ask which this turn would answer. Use specific \
vocabulary from the turn. Be concrete — no generic or meta questions.

Do NOT write questions that could be answered by most turns. Each question \
must be specific to THIS turn's content (names, facts, numbers, decisions).

TURNS:
{turns_block}

Respond in EXACTLY this format, one block per turn, blank line between:
TURN {{i}}:
Q1: <specific question>
Q2: <specific question>

Use turn index numbers as shown. Nothing else."""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_TURN_HEADER_RE = re.compile(r"^\s*TURN\s+(\d+)\s*:", re.IGNORECASE)
_Q_LINE_RE = re.compile(r"^\s*Q\d+\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _parse_inverse_batch(response: str) -> dict[int, list[str]]:
    """Parse batched inverse-question response.

    Returns mapping turn_index (0-based) -> list of generated questions.
    """
    out: dict[int, list[str]] = {}
    current: int | None = None
    for line in response.splitlines():
        line = line.rstrip()
        m = _TURN_HEADER_RE.match(line)
        if m:
            try:
                current = int(m.group(1))
            except ValueError:
                current = None
            if current is not None:
                out.setdefault(current, [])
            continue
        if current is None:
            continue
        m2 = _Q_LINE_RE.match(line)
        if m2:
            q = m2.group(1).strip().strip('"').strip("'")
            # Drop trivial / very short questions
            if q and len(q) > 5:
                out[current].append(q)
    return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _InverseQueryBase(BestshotBase):
    """Shared retrieval body for inverse-query variants."""

    arch_name = "inverse_query"
    n_initial = 10  # how many top-k retrieved turns to generate inverse queries for
    per_probe_top_k = 5  # top-k per inverse question

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        super().__init__(store, client)
        # Override with dedicated caches
        self.embedding_cache = InvQueryEmbeddingCache()
        self.llm_cache = InvQueryLLMCache()

    def _make_turns_block(self, segments: list[Segment], max_chars: int = 400) -> str:
        lines = []
        for i, seg in enumerate(segments):
            txt = seg.text.replace("\n", " ").strip()[:max_chars]
            lines.append(f"TURN {i} [turn_id={seg.turn_id}, {seg.role}]: {txt}")
        return "\n".join(lines)

    def generate_inverse_queries(
        self, initial_segments: list[Segment]
    ) -> dict[int, list[str]]:
        """One batched LLM call returning {turn_index -> [questions]}."""
        if not initial_segments:
            return {}
        turns_block = self._make_turns_block(initial_segments)
        prompt = INVERSE_BATCH_PROMPT.format(turns_block=turns_block)
        output = self.llm_call(prompt)
        return _parse_inverse_batch(output)

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        """Standard inverse-query pipeline: raw query top-10, generate inverse
        queries on top-N of those (N = self.n_initial), retrieve top-5 per
        inverse query, merge by max cosine across all probes."""
        # Step 1: raw query top-10
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        hop0_segments = list(hop0.segments)
        hop0_scores = list(hop0.scores)

        if not hop0_segments:
            return BestshotResult(
                segments=[],
                metadata={
                    "name": self.arch_name,
                    "inverse_queries": [],
                    "num_probes": 1,
                    "hop0_empty": True,
                },
            )

        # Score map: index -> max cosine across probes
        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}
        for seg, sc in zip(hop0_segments, hop0_scores):
            score_map[seg.index] = sc
            seg_map[seg.index] = seg

        # Step 2: generate inverse queries from top-N of hop0
        top_for_inv = hop0_segments[: self.n_initial]
        inv_by_turn = self.generate_inverse_queries(top_for_inv)

        # Flatten generated questions with attribution
        inverse_queries: list[dict] = []
        for i, seg in enumerate(top_for_inv):
            qs = inv_by_turn.get(i, [])
            for q in qs[:2]:  # cap 2 per turn
                inverse_queries.append(
                    {
                        "source_turn_id": seg.turn_id,
                        "source_index": seg.index,
                        "question": q,
                    }
                )

        # Step 3: embed + retrieve top-5 per inverse query
        probe_outcomes: list[dict] = []
        for iq in inverse_queries:
            iq_emb = self.embed_text(iq["question"])
            res = self.store.search(
                iq_emb,
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
            probe_outcomes.append(
                {
                    "source_turn_id": iq["source_turn_id"],
                    "question": iq["question"],
                    "retrieved_turn_ids": [
                        seg_map[idx].turn_id for idx in retrieved_ids
                    ],
                }
            )

        # Step 4: rank by max score
        ranked_indices = sorted(
            score_map.keys(), key=lambda idx: score_map[idx], reverse=True
        )
        all_segments = [seg_map[idx] for idx in ranked_indices]

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": self.arch_name,
                "inverse_queries": [iq["question"] for iq in inverse_queries],
                "probe_outcomes": probe_outcomes,
                "num_probes": 1 + len(inverse_queries),
                "hop0_empty": False,
                "n_initial": self.n_initial,
            },
        )


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


class InverseQuery(_InverseQueryBase):
    """Main variant: inverse queries over all top-10 retrieved turns."""

    arch_name = "inverse_query"
    n_initial = 10
    per_probe_top_k = 5


class InverseQueryTop3(_InverseQueryBase):
    """Cheaper variant: inverse queries over only top-3 retrieved turns."""

    arch_name = "inverse_query_top3"
    n_initial = 3
    per_probe_top_k = 5


class InverseQueryV2f(_InverseQueryBase):
    """Union of v2f cues AND inverse queries from v2f's hop0.

    Both probe sets draw from the same hop0. Merge by max cosine across
    ALL probes (raw query + v2f cues + inverse queries).
    """

    arch_name = "inverse_query_v2f"
    n_initial = 10  # inverse queries on hop0 top-10
    per_probe_top_k = 5

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        # Step 1: raw query top-10
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        hop0_segments = list(hop0.segments)
        hop0_scores = list(hop0.scores)

        if not hop0_segments:
            return BestshotResult(
                segments=[],
                metadata={
                    "name": self.arch_name,
                    "v2f_cues": [],
                    "inverse_queries": [],
                    "num_probes": 1,
                    "hop0_empty": True,
                },
            )

        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}
        for seg, sc in zip(hop0_segments, hop0_scores):
            score_map[seg.index] = sc
            seg_map[seg.index] = seg

        # Step 2a: v2f cues
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(hop0_segments)
        )
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        v2f_cues = _parse_cues(v2f_output)[:2]

        v2f_outcomes: list[dict] = []
        for cue in v2f_cues:
            cue_emb = self.embed_text(cue)
            res = self.store.search(cue_emb, top_k=10, conversation_id=conversation_id)
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

        # Step 2b: inverse queries over hop0 top-10 (batched)
        top_for_inv = hop0_segments[: self.n_initial]
        inv_by_turn = self.generate_inverse_queries(top_for_inv)

        inverse_queries: list[dict] = []
        for i, seg in enumerate(top_for_inv):
            qs = inv_by_turn.get(i, [])
            for q in qs[:2]:
                inverse_queries.append(
                    {
                        "source_turn_id": seg.turn_id,
                        "source_index": seg.index,
                        "question": q,
                    }
                )

        probe_outcomes: list[dict] = []
        for iq in inverse_queries:
            iq_emb = self.embed_text(iq["question"])
            res = self.store.search(
                iq_emb,
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
            probe_outcomes.append(
                {
                    "source_turn_id": iq["source_turn_id"],
                    "question": iq["question"],
                    "retrieved_turn_ids": [
                        seg_map[idx].turn_id for idx in retrieved_ids
                    ],
                }
            )

        ranked_indices = sorted(
            score_map.keys(), key=lambda idx: score_map[idx], reverse=True
        )
        all_segments = [seg_map[idx] for idx in ranked_indices]

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": self.arch_name,
                "v2f_cues": v2f_cues,
                "v2f_outcomes": v2f_outcomes,
                "inverse_queries": [iq["question"] for iq in inverse_queries],
                "probe_outcomes": probe_outcomes,
                "num_probes": 1 + len(v2f_cues) + len(inverse_queries),
                "hop0_empty": False,
                "n_initial": self.n_initial,
            },
        )


# Map for use by eval driver
ARCH_CLASSES: dict[str, type] = {
    "inverse_query": InverseQuery,
    "inverse_query_top3": InverseQueryTop3,
    "inverse_query_v2f": InverseQueryV2f,
}
