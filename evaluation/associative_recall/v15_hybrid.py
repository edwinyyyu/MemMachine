"""Test the 'v15 first, then specialist' hybrid pattern across specialists.

Pattern observation (from chain_retrieval.py hybrid_v15_term):
  - Running v15 first (verbatim cosine + 2 assessment cues) then appending
    a specialist's output retained v15's single_hop / direct-retrieval wins
    while adding specialist coverage on the harder subcategories.

Hypothesis: This pattern generalizes. Fixing v15's phase-1 budget while
allocating remaining slots to a specialist (CoT, memory-index, or both)
should preserve v15's wins AND add specialist gains on hard categories.

Three architectures tested:
  hybrid_v15_cot     : phase-1 v15 (top-10 + 1 cue x 5 = 15 segs),
                       phase-2 CoT (1 cue x 5 = 5 segs). K=20.
                       2 LLM calls total.
  hybrid_v15_memidx  : phase-1 v15 (top-10 + 1 cue x 5 = 15 segs),
                       phase-2 memory_index cue (1 cue x 5 = 5 segs). K=20.
                       2 LLM calls total (+ 1 prebuilt index).
  hybrid_v15_dual    : phase-1 v15 (top-20 + 2 cues x 10 = 40 segs),
                       phase-2 CoT (1 cue x 5 = 5 segs),
                       phase-3 memory_index (1 cue x 5 = 5 segs). K=50.
                       3 LLM calls total (+ 1 prebuilt index).

All evaluations use FAIR K-budget backfill (no reranking).

Usage:
    uv run python v15_hybrid.py [--arch NAME] [--dataset NAME] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    EmbeddingCache,
    LLMCache,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from memory_index import MEMINDEX_INDEX_FILE, MemoryIndex
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_FILE_EMB = CACHE_DIR / "v15_hybrid_embedding_cache.json"
CACHE_FILE_LLM = CACHE_DIR / "v15_hybrid_llm_cache.json"
BUDGETS_K20 = [20]
BUDGETS_K50 = [50]


# ---------------------------------------------------------------------------
# Caches -- read from all prior caches, write to v15_hybrid_* files.
# ---------------------------------------------------------------------------
class HybridEmbeddingCache(EmbeddingCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for p in sorted(self.cache_dir.glob("*embedding_cache.json")):
            try:
                with open(p) as f:
                    self._cache.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        self.cache_file = CACHE_FILE_EMB
        self._new: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new = {}


class HybridLLMCache(LLMCache):
    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for p in sorted(self.cache_dir.glob("*llm_cache.json")):
            try:
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
            except (json.JSONDecodeError, OSError):
                pass
        self.cache_file = CACHE_FILE_LLM
        self._new: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new[key] = response

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        existing.update(self._new)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment], max_items: int = 14, max_chars: int = 260
) -> str:
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs
    )


def _parse_cues(text: str, key: str = "CUE:") -> list[str]:
    out: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith(key.upper()):
            val = line[len(key) :].strip()
            if val:
                out.append(val)
    return out


def load_memory_indices() -> dict[str, MemoryIndex]:
    if not MEMINDEX_INDEX_FILE.exists():
        return {}
    with open(MEMINDEX_INDEX_FILE) as f:
        raw = json.load(f)
    return {cid: MemoryIndex.from_dict(d) for cid, d in raw.items()}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Phase-1 v15 prompt: verbatim from chain_retrieval.py V15_PROMPT but asks for
# only 1 cue (K=20 budget) or 2 cues (K=50 budget).
V15_PHASE1_PROMPT_1CUE = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 1 search cue based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
Nothing else."""


V15_PHASE1_PROMPT_2CUE = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# Phase-2 CoT prompt (from chain_retrieval/cot_universal.py, 1 cue variant).
COT_PHASE2_PROMPT = """\
You are performing semantic retrieval over a conversation history. Cues will \
be embedded and matched via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({num_segs} segments, chronological):
{all_segs}

ALREADY SEARCHED FOR (do NOT repeat):
{explored}

Think step by step:
1. What specific terminology appears in the retrieved segments (names, tools, \
symptoms, decisions, tickets, numbers)?
2. What RELATED terminology might be used elsewhere? (aliases, codenames, \
abbreviations, informal references like "the bird", "that thing", ...)
3. If this is a CHAIN (A -> B -> C where each link has different vocabulary), \
what is the NEXT link to search for?
4. If this topic has ALTERNATIVE NAMES, what are they? Include every alias \
you can justify from the retrieved text or reasonable guesses.

Then generate {num_cues} search cue(s) that EXTEND the retrieval in the most \
promising directions. A cue may be:
  - a short alias/name phrase (1-5 words) that might appear inline
  - a 1-2 sentence plausible conversation snippet targeting the next link

Prefer DIVERSE cues (cover multiple aliases and/or multiple chain links). \
Do not rephrase the question.

Format:
REASON: <1-2 sentences: what's current link or what aliases you identified>
CUE: <text>
(up to {num_cues} cues)
Nothing else."""


# Phase-2 memory-index prompt (adapted from memory_index.V15_WITH_INDEX_PROMPT,
# 1-cue variant; also sees phase-1 retrieved context).
MEMINDEX_PHASE2_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{memindex_section}

RETRIEVED SO FAR:
{context_section}

The CONVERSATION INDEX above tells you what's actually in this conversation's \
memory. Use it to:
1. Identify WHICH topic/entity/decision is relevant to the question.
2. Generate a cue using vocabulary tied to those specific topics/entities, \
targeting what the retrieved segments still MISS.
3. If the question asks for something NOT in the index, still generate your \
best-guess cue but note it in the assessment.

First, briefly assess: which index items match the question? What's still \
missing from what's been retrieved?

Then generate {num_cues} search cue(s) grounded in the index's vocabulary. \
Use specific words that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentences; reference index items by name>
CUE: <text>
(up to {num_cues} cues)
Nothing else."""


# ---------------------------------------------------------------------------
# Base hybrid class
# ---------------------------------------------------------------------------
@dataclass
class HybridResult:
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class HybridBase:
    """Shared infrastructure: caches, counters, embeddings, LLM calls."""

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
    ) -> None:
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = HybridEmbeddingCache()
        self.llm_cache = HybridLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(text)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, embedding)
        self.embed_calls += 1
        return embedding

    def llm_call(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(model, prompt, text)
        self.llm_calls += 1
        return text

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HybridV15CoT (K=20): v15 phase (1 cue) + CoT phase (1 cue)
# ---------------------------------------------------------------------------
class HybridV15CoT(HybridBase):
    """Phase 1: question top-10 + 1 v15 cue x 5 = 15 segs.
    Phase 2: 1 CoT cue x 5 = 5 segs. Total ~20 segs, 2 LLM calls.
    """

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []

        # Phase 1a: question embedding, top 10
        q_emb = self.embed_text(question)
        r0 = self.store.search(q_emb, top_k=10, conversation_id=conversation_id)
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        # Phase 1b: v15 prompt -> 1 cue -> top 5
        context = _format_segments(all_segs, max_items=12, max_chars=250)
        v15_resp = self.llm_call(
            V15_PHASE1_PROMPT_1CUE.format(question=question, context=context)
        )
        v15_cues = _parse_cues(v15_resp)[:1]
        for cue in v15_cues:
            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=5,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        # Phase 2: CoT step, 1 cue -> top 5
        cot_prompt = COT_PHASE2_PROMPT.format(
            question=question,
            num_segs=len(all_segs),
            all_segs=_format_segments(all_segs, max_items=14),
            explored=(
                "\n".join(f"- {c}" for c in explored) if explored else "(none yet)"
            ),
            num_cues=1,
        )
        cot_resp = self.llm_call(cot_prompt)
        cot_cues = _parse_cues(cot_resp)[:1]
        for cue in cot_cues:
            if cue in explored:
                continue
            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=5,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return HybridResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "hybrid_v15_cot",
                "v15_cues": v15_cues,
                "cot_cues": cot_cues,
                "total_segments": len(all_segs),
            },
        )


# ---------------------------------------------------------------------------
# HybridV15MemIdx (K=20): v15 phase (1 cue) + memory_index phase (1 cue)
# ---------------------------------------------------------------------------
class HybridV15MemIdx(HybridBase):
    """Phase 1: question top-10 + 1 v15 cue x 5 = 15 segs.
    Phase 2: 1 memory-index cue x 5 = 5 segs. Total ~20 segs, 2 LLM calls
    (+ 1 precomputed / cached index per conversation).
    """

    def __init__(
        self,
        store: SegmentStore,
        indices: dict[str, MemoryIndex],
        client: OpenAI | None = None,
    ) -> None:
        super().__init__(store, client)
        self.indices = indices

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []

        # Phase 1a
        q_emb = self.embed_text(question)
        r0 = self.store.search(q_emb, top_k=10, conversation_id=conversation_id)
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        # Phase 1b: v15 cue
        context = _format_segments(all_segs, max_items=12, max_chars=250)
        v15_resp = self.llm_call(
            V15_PHASE1_PROMPT_1CUE.format(question=question, context=context)
        )
        v15_cues = _parse_cues(v15_resp)[:1]
        for cue in v15_cues:
            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=5,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        # Phase 2: memory-index cue
        idx = self.indices.get(conversation_id)
        if idx is None:
            memindex_section = (
                "CONVERSATION INDEX (what's actually in memory):\n"
                "(unavailable for this conversation)"
            )
        else:
            memindex_section = idx.format_for_prompt()

        midx_prompt = MEMINDEX_PHASE2_PROMPT.format(
            question=question,
            memindex_section=memindex_section,
            context_section=_format_segments(all_segs, max_items=14),
            num_cues=1,
        )
        midx_resp = self.llm_call(midx_prompt)
        midx_cues = _parse_cues(midx_resp)[:1]
        for cue in midx_cues:
            if cue in explored:
                continue
            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=5,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return HybridResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "hybrid_v15_memidx",
                "v15_cues": v15_cues,
                "memidx_cues": midx_cues,
                "total_segments": len(all_segs),
            },
        )


# ---------------------------------------------------------------------------
# HybridV15Dual (K=50): v15 phase (2 cues) + CoT (1 cue) + memory_index (1 cue)
# ---------------------------------------------------------------------------
class HybridV15Dual(HybridBase):
    """Phase 1: question top-20 + 2 v15 cues x 10 = 40 segs.
    Phase 2: 1 CoT cue x 5 = 5 segs.
    Phase 3: 1 memory-index cue x 5 = 5 segs. Total ~50, 3 LLM calls.
    """

    def __init__(
        self,
        store: SegmentStore,
        indices: dict[str, MemoryIndex],
        client: OpenAI | None = None,
    ) -> None:
        super().__init__(store, client)
        self.indices = indices

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        exclude: set[int] = set()
        all_segs: list[Segment] = []
        explored: list[str] = []

        # Phase 1a: top-20 on question
        q_emb = self.embed_text(question)
        r0 = self.store.search(q_emb, top_k=20, conversation_id=conversation_id)
        for s in r0.segments:
            if s.index not in exclude:
                all_segs.append(s)
                exclude.add(s.index)

        # Phase 1b: v15 prompt -> 2 cues -> top 10 each
        context = _format_segments(all_segs, max_items=14, max_chars=250)
        v15_resp = self.llm_call(
            V15_PHASE1_PROMPT_2CUE.format(question=question, context=context)
        )
        v15_cues = _parse_cues(v15_resp)[:2]
        for cue in v15_cues:
            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        # Phase 2: CoT cue x top 5
        cot_prompt = COT_PHASE2_PROMPT.format(
            question=question,
            num_segs=len(all_segs),
            all_segs=_format_segments(all_segs, max_items=14),
            explored=(
                "\n".join(f"- {c}" for c in explored) if explored else "(none yet)"
            ),
            num_cues=1,
        )
        cot_resp = self.llm_call(cot_prompt)
        cot_cues = _parse_cues(cot_resp)[:1]
        for cue in cot_cues:
            if cue in explored:
                continue
            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=5,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        # Phase 3: memory-index cue x top 5
        idx = self.indices.get(conversation_id)
        if idx is None:
            memindex_section = (
                "CONVERSATION INDEX (what's actually in memory):\n"
                "(unavailable for this conversation)"
            )
        else:
            memindex_section = idx.format_for_prompt()
        midx_prompt = MEMINDEX_PHASE2_PROMPT.format(
            question=question,
            memindex_section=memindex_section,
            context_section=_format_segments(all_segs, max_items=14),
            num_cues=1,
        )
        midx_resp = self.llm_call(midx_prompt)
        midx_cues = _parse_cues(midx_resp)[:1]
        for cue in midx_cues:
            if cue in explored:
                continue
            explored.append(cue)
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=5,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for s in result.segments:
                if s.index not in exclude:
                    all_segs.append(s)
                    exclude.add(s.index)

        return HybridResult(
            segments=all_segs,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "hybrid_v15_dual",
                "v15_cues": v15_cues,
                "cot_cues": cot_cues,
                "memidx_cues": midx_cues,
                "total_segments": len(all_segs),
            },
        )


# ---------------------------------------------------------------------------
# Datasets (same as cot_universal.py / memory_index.py)
# ---------------------------------------------------------------------------
DATASETS = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "limit": 30,
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": lambda q: True,
        "limit": None,
    },
    "puzzle_16q": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "filter": lambda q: True,
        "limit": None,
    },
    "advanced_23q": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "filter": lambda q: True,
        "limit": None,
    },
}


def load_dataset(key: str) -> tuple[list[dict], SegmentStore]:
    meta = DATASETS[key]
    with open(DATA_DIR / meta["questions"]) as f:
        qs = json.load(f)
    qs = [q for q in qs if meta["filter"](q)]
    if meta["limit"] is not None:
        qs = qs[: meta["limit"]]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=meta["npz"])
    return qs, store


# ---------------------------------------------------------------------------
# Fair K-budget evaluation
# ---------------------------------------------------------------------------
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: HybridBase,
    question: dict,
    budgets: list[int],
    verbose: bool = False,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Dedup preserving order
    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for s in result.segments:
        if s.index not in seen:
            arch_segments.append(s)
            seen.add(s.index)

    # Baseline cosine top-max(budgets) on question
    q_emb = arch.embed_text(q_text)
    max_b = max(budgets)
    baseline = arch.store.search(q_emb, top_k=max_b, conversation_id=conv_id)

    # FAIR backfill: arch + baseline residual, in order, dedup
    arch_idx = {s.index for s in arch_segments}
    backfilled = list(arch_segments) + [
        s for s in baseline.segments if s.index not in arch_idx
    ]

    arch_recalls: dict[str, float] = {}
    baseline_recalls: dict[str, float] = {}
    for K in budgets:
        a_ids = {s.turn_id for s in backfilled[:K]}
        b_ids = {s.turn_id for s in baseline.segments[:K]}
        arch_recalls[f"r@{K}"] = compute_recall(a_ids, source_ids)
        baseline_recalls[f"r@{K}"] = compute_recall(b_ids, source_ids)

    row = {
        "conversation_id": conv_id,
        "category": question["category"],
        "question_index": question.get("question_index"),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "arch_pool_size": len(arch_segments),
        "baseline_recalls": baseline_recalls,
        "arch_recalls": arch_recalls,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "metadata": result.metadata,
    }
    if verbose:
        K0 = budgets[0]
        print(
            f"    pool={len(arch_segments)} "
            f"r@{K0}: base={baseline_recalls[f'r@{K0}']:.3f} "
            f"arch={arch_recalls[f'r@{K0}']:.3f}  "
            f"emb={arch.embed_calls} llm={arch.llm_calls}",
            flush=True,
        )
    return row


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------
ARCH_SPECS = {
    "hybrid_v15_cot": {
        "cls": HybridV15CoT,
        "budgets": BUDGETS_K20,
        "use_index": False,
    },
    "hybrid_v15_memidx": {
        "cls": HybridV15MemIdx,
        "budgets": BUDGETS_K20,
        "use_index": True,
    },
    "hybrid_v15_dual": {
        "cls": HybridV15Dual,
        "budgets": BUDGETS_K50,
        "use_index": True,
    },
}


def build_arch(
    arch_name: str, store: SegmentStore, indices: dict[str, MemoryIndex]
) -> HybridBase:
    spec = ARCH_SPECS[arch_name]
    if spec["use_index"]:
        return spec["cls"](store, indices)
    return spec["cls"](store)


# ---------------------------------------------------------------------------
# Baseline loaders for comparison (budget_v15_tight_*, budget_v2f_tight_*,
# budget_baseline_*, cot_chain_of_thought_*)
# ---------------------------------------------------------------------------
def load_budget_recalls(arch_name: str, K: int, dataset_key: str) -> dict[tuple, float]:
    """Load recall per question from budget_<name>_<K>_<dataset>.json.

    Returns {(conversation_id, question_index): recall}.
    """
    path = RESULTS_DIR / f"budget_{arch_name}_{K}_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        payload = json.load(f)
    out: dict[tuple, float] = {}
    for r in payload.get("results", []):
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = r["recall"]
    return out


def load_cot_recalls(K: int, dataset_key: str) -> dict[tuple, float]:
    path = RESULTS_DIR / f"cot_chain_of_thought_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    out: dict[tuple, float] = {}
    for r in rows:
        key = (r["conversation_id"], r.get("question_index"))
        out[key] = r["cot_recalls"][f"r@{K}"]
    return out


# ---------------------------------------------------------------------------
# Per-category comparison table
# ---------------------------------------------------------------------------
def compare_per_category(
    arch_rows: list[dict],
    arch_name: str,
    dataset_key: str,
    K: int,
) -> list[dict]:
    """Build per-category rows comparing arch vs baseline/v15/v2f/CoT at K.

    Uses per-question recalls to keep means comparable across architectures.
    """
    # Load per-question recalls for comparators
    baseline_recalls = load_budget_recalls("baseline", K, dataset_key)
    v15_recalls = load_budget_recalls("v15_tight", K, dataset_key)
    v2f_recalls = load_budget_recalls("v2f_tight", K, dataset_key)
    cot_recalls = load_cot_recalls(K, dataset_key)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in arch_rows:
        by_cat[r["category"]].append(r)

    out: list[dict] = []
    for cat, rows in sorted(by_cat.items()):
        n = len(rows)
        if n == 0:
            continue
        arch_vals = [r["arch_recalls"][f"r@{K}"] for r in rows]
        arch_mean = sum(arch_vals) / n

        b_vals: list[float] = []
        v15_vals: list[float] = []
        v2f_vals: list[float] = []
        cot_vals: list[float] = []
        for r in rows:
            key = (r["conversation_id"], r["question_index"])
            if key in baseline_recalls:
                b_vals.append(baseline_recalls[key])
            if key in v15_recalls:
                v15_vals.append(v15_recalls[key])
            if key in v2f_recalls:
                v2f_vals.append(v2f_recalls[key])
            if key in cot_recalls:
                cot_vals.append(cot_recalls[key])

        def _mean(xs: list[float]) -> float | None:
            if not xs:
                return None
            return sum(xs) / len(xs)

        b = _mean(b_vals)
        v15 = _mean(v15_vals)
        v2f = _mean(v2f_vals)
        cot = _mean(cot_vals)

        out.append(
            {
                "arch_name": arch_name,
                "dataset": dataset_key,
                "category": cat,
                "K": K,
                "n": n,
                "baseline": b,
                "v15": v15,
                "v2f": v2f,
                "cot": cot,
                "arch": arch_mean,
                "vs_baseline": (arch_mean - b) if b is not None else None,
                "vs_v15": (arch_mean - v15) if v15 is not None else None,
                "vs_v2f": (arch_mean - v2f) if v2f is not None else None,
                "vs_cot": (arch_mean - cot) if cot is not None else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_arch_on_dataset(
    arch_name: str,
    dataset_key: str,
    indices: dict[str, MemoryIndex],
    force: bool = False,
    verbose: bool = False,
) -> list[dict]:
    spec = ARCH_SPECS[arch_name]
    result_file = RESULTS_DIR / f"v15_hybrid_{arch_name}_{dataset_key}.json"
    if result_file.exists() and not force:
        with open(result_file) as f:
            return json.load(f)

    qs, store = load_dataset(dataset_key)
    arch = build_arch(arch_name, store, indices)
    print(
        f"\n>>> {arch_name} on {dataset_key}: {len(qs)} questions, "
        f"{len(store.segments)} segments"
    )
    rows: list[dict] = []
    for i, q in enumerate(qs):
        q_short = q["question"][:60].replace("\n", " ")
        print(
            f"  [{i + 1}/{len(qs)}] {q['category']}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_one(arch, q, spec["budgets"], verbose=verbose)
            rows.append(row)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback

            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  Saved -> {result_file}")
    return rows


def fmt_cell(val: float | None, plus_sign: bool = False) -> str:
    if val is None:
        return "    —"
    s = f"{val:+.3f}" if plus_sign else f"{val:.3f}"
    return f"{s:>6s}"


def print_per_category_table(rows: list[dict], K: int) -> None:
    sel = [r for r in rows if r["K"] == K]
    if not sel:
        return
    archs = sorted({r["arch_name"] for r in sel})
    print(f"\n{'=' * 130}")
    print(f"PER-CATEGORY RESULTS at K={K} (recall, fair backfill)")
    print(f"{'=' * 130}")
    for arch in archs:
        print(f"\n--- {arch} ---")
        hdr = (
            f"{'Dataset':<14s} {'Category':<26s} {'n':>3s} "
            f"{'Base':>7s} {'v15':>7s} {'v2f':>7s} {'CoT':>7s} {'Arch':>7s}  "
            f"{'vs base':>8s} {'vs v15':>7s} {'vs v2f':>7s} {'vs CoT':>7s}"
        )
        print(hdr)
        print("-" * len(hdr))
        arch_rows = [r for r in sel if r["arch_name"] == arch]
        last_ds = None
        for r in arch_rows:
            if last_ds is not None and r["dataset"] != last_ds:
                print()
            last_ds = r["dataset"]
            print(
                f"{r['dataset']:<14s} {r['category']:<26s} {r['n']:>3d} "
                f"{fmt_cell(r['baseline'])} {fmt_cell(r['v15'])} "
                f"{fmt_cell(r['v2f'])} {fmt_cell(r['cot'])} "
                f"{fmt_cell(r['arch'])}  "
                f"{fmt_cell(r['vs_baseline'], True)} "
                f"{fmt_cell(r['vs_v15'], True)} "
                f"{fmt_cell(r['vs_v2f'], True)} "
                f"{fmt_cell(r['vs_cot'], True)}"
            )


def print_overall_by_dataset(rows: list[dict], K: int) -> None:
    sel = [r for r in rows if r["K"] == K]
    if not sel:
        return
    archs = sorted({r["arch_name"] for r in sel})
    print(f"\n{'-' * 130}")
    print(f"OVERALL per DATASET at K={K} (weighted by category n)")
    print(f"{'-' * 130}")
    hdr = (
        f"{'Arch':<20s} {'Dataset':<14s} {'n':>3s} "
        f"{'Base':>7s} {'v15':>7s} {'v2f':>7s} {'CoT':>7s} {'Arch':>7s}  "
        f"{'vs base':>8s} {'vs v15':>7s} {'vs v2f':>7s} {'vs CoT':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for arch in archs:
        by_ds: dict[str, list[dict]] = defaultdict(list)
        for r in sel:
            if r["arch_name"] == arch:
                by_ds[r["dataset"]].append(r)
        for ds, rs in by_ds.items():
            total_n = sum(r["n"] for r in rs)

            def _weighted(key: str) -> float | None:
                vals = [(r[key], r["n"]) for r in rs if r.get(key) is not None]
                if not vals:
                    return None
                tot = sum(n for _, n in vals)
                return sum(v * n for v, n in vals) / tot

            b = _weighted("baseline")
            v15 = _weighted("v15")
            v2f = _weighted("v2f")
            cot = _weighted("cot")
            am = _weighted("arch")
            vb = (am - b) if (am is not None and b is not None) else None
            v15d = (am - v15) if (am is not None and v15 is not None) else None
            v2fd = (am - v2f) if (am is not None and v2f is not None) else None
            cotd = (am - cot) if (am is not None and cot is not None) else None
            print(
                f"{arch:<20s} {ds:<14s} {total_n:>3d} "
                f"{fmt_cell(b)} {fmt_cell(v15)} {fmt_cell(v2f)} {fmt_cell(cot)} "
                f"{fmt_cell(am)}  "
                f"{fmt_cell(vb, True)} "
                f"{fmt_cell(v15d, True)} "
                f"{fmt_cell(v2fd, True)} "
                f"{fmt_cell(cotd, True)}"
            )


def print_cot_regression_table(all_rows: list[dict]) -> None:
    """Focused view: did hybrid_v15_cot eliminate CoT's regressions?"""
    print(f"\n{'=' * 110}")
    print("HYBRID_V15_COT vs CoT — regression elimination check (K=20)")
    print(f"{'=' * 110}")
    sel = [r for r in all_rows if r["arch_name"] == "hybrid_v15_cot" and r["K"] == 20]
    hdr = (
        f"{'Dataset':<14s} {'Category':<28s} {'n':>3s} "
        f"{'v15':>7s} {'CoT':>7s} {'Hybrid':>7s}  "
        f"{'CoT-v15':>8s} {'Hyb-v15':>8s} {'Hyb-CoT':>8s}  verdict"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sel:
        v15 = r["v15"]
        cot = r["cot"]
        h = r["arch"]
        cot_v15 = (cot - v15) if (cot is not None and v15 is not None) else None
        h_v15 = (h - v15) if (h is not None and v15 is not None) else None
        h_cot = (h - cot) if (h is not None and cot is not None) else None
        # Verdict: did hybrid rescue a CoT regression?
        verdict = ""
        if cot_v15 is not None and cot_v15 <= -0.02:
            # CoT regressed. Did hybrid recover?
            if h_v15 is not None and h_v15 >= -0.01:
                verdict = "RESCUED"
            elif h_v15 is not None and h_v15 > cot_v15 + 0.02:
                verdict = "partial"
            else:
                verdict = "still hurt"
        elif cot_v15 is not None and cot_v15 >= 0.02:
            if h_v15 is not None and h_v15 >= cot_v15 - 0.01:
                verdict = "PRESERVED"
            elif h_v15 is not None and h_v15 >= 0.01:
                verdict = "partial"
            else:
                verdict = "lost"
        else:
            verdict = "~neutral"
        print(
            f"{r['dataset']:<14s} {r['category']:<28s} {r['n']:>3d} "
            f"{fmt_cell(v15)} {fmt_cell(cot)} {fmt_cell(h)}  "
            f"{fmt_cell(cot_v15, True)} "
            f"{fmt_cell(h_v15, True)} "
            f"{fmt_cell(h_cot, True)}  {verdict}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=list(ARCH_SPECS.keys()),
        help="Run a single architecture (default: all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASETS.keys()),
        help="Run a single dataset (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rerun even if result file exists"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load memory indices (cached by memory_index.py).
    indices = load_memory_indices()
    print(f"Loaded {len(indices)} memory indices from cache.")

    archs = [args.arch] if args.arch else list(ARCH_SPECS.keys())
    ds_keys = [args.dataset] if args.dataset else list(DATASETS.keys())

    all_comparisons: list[dict] = []
    for arch_name in archs:
        spec = ARCH_SPECS[arch_name]
        for ds in ds_keys:
            rows = run_arch_on_dataset(
                arch_name, ds, indices, force=args.force, verbose=args.verbose
            )
            for K in spec["budgets"]:
                all_comparisons.extend(compare_per_category(rows, arch_name, ds, K))

    out_path = RESULTS_DIR / "v15_hybrid_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_comparisons, f, indent=2, default=str)
    print(f"\nSaved summary -> {out_path}")

    for K in (20, 50):
        print_per_category_table(all_comparisons, K)
        print_overall_by_dataset(all_comparisons, K)

    print_cot_regression_table(all_comparisons)


if __name__ == "__main__":
    main()
