"""MMR (Maximum Marginal Relevance) cue-diversity selection.

For each question q at budget K:
  1. Single LLM call with modified v2f prompt asking for 8 candidate cues.
  2. Embed all 8 candidates.
  3. Apply MMR greedy selection to pick K_cues (3 or 4):
       score(cue) = lambda * cos(cue, query) - (1-lambda) * max_sel cos(cue, sel)
     Start with the cue with highest cos(cue, query); iteratively add the cue
     maximizing the MMR score against the already-selected set.
  4. Retrieve top-10 per selected cue. Merge via sum_cosine — segments ranked
     by sum of cosine scores across the probes that hit them (plus the raw
     query as one probe).

Motivation:
  v2f generates 2-3 cues that may probe the same region of embedding space.
  The substrate insight (dispersion beats convergence) suggests multi-probe
  wins come from covering distinct regions. MMR explicitly enforces diversity
  among selected cues.

Variants
  mmr_lam0.5_k3  — balanced relevance/diversity, 3 cues
  mmr_lam0.3_k3  — diversity-heavy, 3 cues
  mmr_lam0.7_k3  — relevance-heavy, 3 cues
  mmr_lam0.5_k4  — balanced, 4 cues
  v2f_3cues      — baseline: v2f asking for 3 cues (first 3, no MMR)

Dedicated caches (mmr_*_cache.json) avoid concurrent-agent corruption.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
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
    MODEL,
    BestshotBase,
    BestshotResult,
    _format_segments,
    _parse_cues,
)


# ---------------------------------------------------------------------------
# Dedicated caches
# ---------------------------------------------------------------------------

_MMR_EMB_FILE = CACHE_DIR / "mmr_embedding_cache.json"
_MMR_LLM_FILE = CACHE_DIR / "mmr_llm_cache.json"

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
    "mmr_embedding_cache.json",
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
    "mmr_llm_cache.json",
)


class MMREmbeddingCache(EmbeddingCache):
    """Reads shared embedding caches, writes to dedicated mmr file."""

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
        self.cache_file = _MMR_EMB_FILE
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


class MMRLLMCache(LLMCache):
    """Reads shared LLM caches, writes to dedicated mmr file."""

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
        self.cache_file = _MMR_LLM_FILE
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
# Prompts — modified v2f asking for N candidate cues
# ---------------------------------------------------------------------------

# 8-candidate prompt (for MMR selection). Keeps v2f skeleton.
V2F_MMR_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate exactly 8 search cues based on your assessment. Each cue \
should probe a DIFFERENT aspect or angle of the question — varied vocabulary, \
varied framing, different sub-topics the answer might touch. Do not paraphrase \
the same idea 8 times. Use specific vocabulary that would appear in the \
target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
CUE: <text>
CUE: <text>
CUE: <text>
CUE: <text>
CUE: <text>
CUE: <text>
Nothing else."""


# 3-cue v2f baseline — same as v2f but asks for 3 instead of 2.
V2F_3CUES_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 3 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
CUE: <text>
Nothing else."""


# ---------------------------------------------------------------------------
# MMR selection
# ---------------------------------------------------------------------------

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_select(
    cue_embs: list[np.ndarray],
    query_emb: np.ndarray,
    k: int,
    lam: float,
) -> list[int]:
    """Greedy MMR selection. Returns indices into cue_embs of selected cues.

    Start with the cue maximizing cos(cue, query). Then iteratively add the
    cue maximizing:
        lam * cos(cue, query) - (1 - lam) * max_sel cos(cue, sel)
    """
    n = len(cue_embs)
    if n == 0:
        return []
    k = min(k, n)

    rel = [_cos(ce, query_emb) for ce in cue_embs]
    selected: list[int] = []
    remaining: set[int] = set(range(n))

    # First pick: highest relevance
    first = max(remaining, key=lambda i: rel[i])
    selected.append(first)
    remaining.discard(first)

    while len(selected) < k and remaining:
        best_idx = None
        best_score = -float("inf")
        for i in remaining:
            max_sim_to_selected = max(
                _cos(cue_embs[i], cue_embs[j]) for j in selected
            )
            score = lam * rel[i] - (1.0 - lam) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def mean_pairwise_cosine(embs: list[np.ndarray]) -> float:
    """Mean pairwise cosine across a list of embeddings. Returns 0 for <2."""
    n = len(embs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _cos(embs[i], embs[j])
            count += 1
    return total / count if count else 0.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _MMRBase(BestshotBase):
    """Base for MMR cue selection variants."""

    arch_name: str = "mmr_base"
    lam: float = 0.5
    k_cues: int = 3
    n_candidates: int = 8
    per_probe_top_k: int = 10
    prompt_template: str = V2F_MMR_PROMPT
    mmr_select_enabled: bool = True  # if False, just use the first k_cues

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        if client is None:
            client = OpenAI(timeout=60.0, max_retries=3)
        super().__init__(store, client)
        self.embedding_cache = MMREmbeddingCache()
        self.llm_cache = MMRLLMCache()

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
                    max_completion_tokens=3000,
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

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        # Hop 0: embed question, top-10 cosine as context
        query_emb = self.embed_text(question)
        hop0 = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        hop0_segments = list(hop0.segments)
        hop0_scores = list(hop0.scores)

        # Score map starts from raw-query cosine hits (acts as a probe too)
        score_map: dict[int, float] = {}
        seg_map: dict[int, Segment] = {}
        for seg, sc in zip(hop0_segments, hop0_scores):
            score_map[seg.index] = sc
            seg_map[seg.index] = seg

        # Generate candidates
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(hop0_segments)
        )
        prompt = self.prompt_template.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        candidate_cues = _parse_cues(output)

        # Truncate/pad to expected count
        candidate_cues = [c for c in candidate_cues if c.strip()][
            : self.n_candidates
        ]

        # Embed candidates in parallel
        if candidate_cues:
            with ThreadPoolExecutor(
                max_workers=max(1, len(candidate_cues))
            ) as pool:
                cand_embs = list(pool.map(self.embed_text, candidate_cues))
        else:
            cand_embs = []

        # Select
        if self.mmr_select_enabled and cand_embs:
            selected_idx = mmr_select(
                cand_embs, query_emb, k=self.k_cues, lam=self.lam
            )
        else:
            # Baseline: just first k_cues
            selected_idx = list(range(min(self.k_cues, len(cand_embs))))

        selected_cues = [candidate_cues[i] for i in selected_idx]
        selected_embs = [cand_embs[i] for i in selected_idx]

        # Diversity metric: mean pairwise cosine among selected cues + among
        # all candidates (for reference).
        diversity_selected = mean_pairwise_cosine(selected_embs)
        diversity_all = mean_pairwise_cosine(cand_embs)

        # Retrieve per selected cue; merge via sum_cosine
        probe_outcomes: list[dict] = []
        for cue, cue_emb in zip(selected_cues, selected_embs):
            res = self.store.search(
                cue_emb,
                top_k=self.per_probe_top_k,
                conversation_id=conversation_id,
            )
            retrieved_ids = []
            for seg, sc in zip(res.segments, res.scores):
                retrieved_ids.append(seg.index)
                # sum_cosine: accumulate across probes
                if seg.index not in seg_map:
                    seg_map[seg.index] = seg
                score_map[seg.index] = score_map.get(seg.index, 0.0) + sc
            probe_outcomes.append(
                {
                    "cue": cue,
                    "retrieved_turn_ids": [
                        seg_map[idx].turn_id for idx in retrieved_ids
                    ],
                }
            )

        # Rank by sum_cosine
        ranked_indices = sorted(
            score_map.keys(), key=lambda idx: score_map[idx], reverse=True
        )
        all_segments = [seg_map[idx] for idx in ranked_indices]

        return BestshotResult(
            segments=all_segments,
            metadata={
                "name": self.arch_name,
                "output": output,
                "candidate_cues": candidate_cues,
                "selected_cues": selected_cues,
                "selected_indices": selected_idx,
                "diversity_selected_pairwise_cos": round(
                    diversity_selected, 4
                ),
                "diversity_all_candidates_pairwise_cos": round(
                    diversity_all, 4
                ),
                "n_candidates": len(candidate_cues),
                "n_selected": len(selected_cues),
                "probe_outcomes": probe_outcomes,
                "lam": self.lam,
                "k_cues": self.k_cues,
                "hop0_empty": len(hop0_segments) == 0,
            },
        )


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


class MMRLam05K3(_MMRBase):
    """Balanced relevance/diversity, 3 cues."""

    arch_name = "mmr_lam0.5_k3"
    lam = 0.5
    k_cues = 3


class MMRLam03K3(_MMRBase):
    """Diversity-heavy, 3 cues."""

    arch_name = "mmr_lam0.3_k3"
    lam = 0.3
    k_cues = 3


class MMRLam07K3(_MMRBase):
    """Relevance-heavy, 3 cues (close to raw top-3)."""

    arch_name = "mmr_lam0.7_k3"
    lam = 0.7
    k_cues = 3


class MMRLam05K4(_MMRBase):
    """Balanced, 4 cues."""

    arch_name = "mmr_lam0.5_k4"
    lam = 0.5
    k_cues = 4


class V2f3Cues(_MMRBase):
    """Baseline: v2f asking for 3 cues, no MMR (first 3)."""

    arch_name = "v2f_3cues"
    lam = 0.5  # unused
    k_cues = 3
    n_candidates = 3
    prompt_template = V2F_3CUES_PROMPT
    mmr_select_enabled = False


ARCH_CLASSES: dict[str, type] = {
    "mmr_lam0.5_k3": MMRLam05K3,
    "mmr_lam0.3_k3": MMRLam03K3,
    "mmr_lam0.7_k3": MMRLam07K3,
    "mmr_lam0.5_k4": MMRLam05K4,
    "v2f_3cues": V2f3Cues,
}
