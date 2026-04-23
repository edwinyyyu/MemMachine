"""Query rewriting vs cue generation: head-to-head experiment.

Tests whether REWRITING the original question (alternative phrasings that
preserve question structure) beats our current approach of generating
separate "cues" (disposable vocabulary bundles).

Variants tested (each with K=20 and K=50 budgets):
  1. cosine_only            — pure top-K (no LLM)
  2. v15_control_split      — question top-a + 2 cues top-b (K split 3-ways)
  3. v2f_v2_split           — v2f without anti-question (K split 3-ways)
  4. query_rewrite_3        — original + 2 rewrites (K split 3-ways, no-exclude)
  5. hyde                   — 1 hypothetical answer, 1 search, top-K
  6. decomposition_4        — 4 semantic primitives (K split 4-ways, no-exclude)
  7. question_plus_noise    — original + 2 noise-perturbed embeddings

Budget semantics: total retrieved segments = K. Split evenly across searches;
extras go to earlier searches. Duplicates allowed under no-exclude rewrites
(they may inflate the count above K-unique but we truncate the per-search
results to fit the budget exactly after dedup).

Datasets: LoCoMo (30q), Synthetic (19q), Puzzle (16q), Advanced (23q).
Model: gpt-5-mini. Embeddings: text-embedding-3-small.

Usage:
    uv run python query_rewriting.py
    uv run python query_rewriting.py --variant hyde --dataset locomo_30q
    uv run python query_rewriting.py --list
"""

import argparse
import json
import sys
import time
from collections import defaultdict
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

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50]

# Noise parameters for the adversarial control variant
NOISE_SIGMA = 0.15
NOISE_SEEDS = [1, 2]


DATASETS = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_questions": 30,
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": None,
        "max_questions": None,
    },
    "puzzle_16q": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "filter": None,
        "max_questions": None,
    },
    "advanced_23q": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "filter": None,
        "max_questions": None,
    },
}


# ---------------------------------------------------------------------------
# Caches specific to this experiment
# ---------------------------------------------------------------------------
class QueryRewriteEmbeddingCache(EmbeddingCache):
    """Shares reads with the main embedding caches, writes a dedicated file."""

    SHARED_READS = (
        "embedding_cache.json",
        "arch_embedding_cache.json",
        "bestshot_embedding_cache.json",
        "meta_embedding_cache.json",
        "optim_embedding_cache.json",
        "synth_test_embedding_cache.json",
        "query_rewrite_embedding_cache.json",
    )

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in self.SHARED_READS:
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "query_rewrite_embedding_cache.json"
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
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries.clear()


class QueryRewriteLLMCache(LLMCache):
    """Dedicated LLM cache for query-rewriting LLM calls."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "query_rewrite_llm_cache.json"
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)
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
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
        self._new_entries.clear()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
QUERY_REWRITE_PROMPT = """\
You are rewriting a user's question into alternative phrasings for \
semantic retrieval. Each rewrite will be embedded and matched via cosine \
similarity against stored conversation turns.

Original question: {question}

Produce 2 alternative phrasings of this question. Each rewrite MUST:
- Still be a complete question or direct request (preserve question structure)
- Preserve the original intent exactly — same information is being asked for
- Use DIFFERENT vocabulary from the original where possible (synonyms, \
related terms, different framing) so the embeddings cover different regions
- Sound like something a person would actually ask

Do NOT write keyword bundles, tags, or fragments. These are full questions.
Do NOT add information that was not in the original question.

Format (exactly 2 lines, nothing else):
REWRITE: <question 1>
REWRITE: <question 2>"""


HYDE_PROMPT = """\
You are helping retrieve information from a past conversation. Your output \
will be embedded and matched via cosine similarity against stored \
conversation turns.

Question: {question}

Write ONE short hypothetical answer to this question, as if you were \
recalling what was actually said in the conversation. Include plausible \
specific details (names, numbers, dates, tools, phrases). Write it in the \
natural voice of a chat message, 1-3 sentences.

Do NOT hedge ("it might be..."). Write it as if you know the answer.
Do NOT mention that this is hypothetical.
Do NOT write a question — write a direct statement/answer.

Format (one line, nothing else):
ANSWER: <hypothetical answer>"""


DECOMPOSITION_PROMPT = """\
You are decomposing a question into its semantic components for retrieval. \
Each component will be embedded separately and used as a search query.

Question: {question}

Break the question into 4 semantic primitives. A primitive is a short \
phrase (1-5 words) capturing ONE concept from the question: a named entity, \
a category, a specific type, or a closely-related concept. Together the \
primitives should cover the question's meaning, but each primitive is its \
own focused probe.

Examples:
- "What dietary restrictions does Bob have?" → "Bob", \
"dietary restrictions", "allergies", "food preferences"
- "When did Alice start learning piano?" → "Alice", "piano lessons", \
"learning music", "music teacher"

Do NOT write full questions or full sentences.
Do NOT add content not implied by the question.

Format (exactly 4 lines, nothing else):
COMPONENT: <primitive 1>
COMPONENT: <primitive 2>
COMPONENT: <primitive 3>
COMPONENT: <primitive 4>"""


V15_CONTROL_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

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


V2F_V2_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""


# ---------------------------------------------------------------------------
# Base infrastructure
# ---------------------------------------------------------------------------
@dataclass
class VariantResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


class VariantBase:
    """Base class: embed/LLM with caching and call counters."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = QueryRewriteEmbeddingCache()
        self.llm_cache = QueryRewriteLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

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
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1500,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(model, prompt, text)
        self.llm_calls += 1
        return text

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0


def split_budget(K: int, n_searches: int) -> list[int]:
    """Split budget K across n_searches.

    Returns a list of per-search top_k values summing to K, with earlier
    searches getting the extra when K doesn't divide evenly.
    """
    if n_searches <= 0:
        return []
    base = K // n_searches
    extra = K % n_searches
    return [base + (1 if i < extra else 0) for i in range(n_searches)]


def _parse_lines(response: str, prefix: str) -> list[str]:
    """Parse lines starting with `prefix:` (case-insensitive)."""
    out = []
    p = prefix.upper() + ":"
    for line in response.strip().splitlines():
        line = line.strip()
        if line.upper().startswith(p):
            val = line[len(p):].strip()
            if val:
                out.append(val)
    return out


def merge_search_results(
    per_search_segments: list[list[Segment]],
    K: int,
) -> list[Segment]:
    """Merge results from multiple searches into an ordered, deduped list.

    Interleaves (round-robin) across searches to preserve diversity from each.
    Drops duplicate segment indices, keeping the first occurrence. Truncates
    to exactly K.
    """
    seen: set[int] = set()
    merged: list[Segment] = []
    max_len = max((len(r) for r in per_search_segments), default=0)
    for i in range(max_len):
        for results in per_search_segments:
            if i >= len(results):
                continue
            seg = results[i]
            if seg.index in seen:
                continue
            merged.append(seg)
            seen.add(seg.index)
            if len(merged) >= K:
                return merged
    return merged


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
class CosineOnlyVariant(VariantBase):
    """Pure cosine top-K using the original question embedding."""

    name = "cosine_only"

    def retrieve(
        self, question: str, conversation_id: str, K: int,
    ) -> VariantResult:
        q_emb = self.embed_text(question)
        result = self.store.search(
            q_emb, top_k=K, conversation_id=conversation_id
        )
        return VariantResult(
            segments=list(result.segments),
            metadata={"name": self.name, "searches": 1, "split": [K]},
        )


class V15ControlSplitVariant(VariantBase):
    """v15 prompt: question + 2 cues, K split across the 3 searches."""

    name = "v15_control_split"

    def retrieve(
        self, question: str, conversation_id: str, K: int,
    ) -> VariantResult:
        q_emb = self.embed_text(question)
        # Hop 0 context (use a generous top retrieval for the context section,
        # not counted toward budget — it's a temporary view for the LLM).
        context_probe = self.store.search(
            q_emb, top_k=10, conversation_id=conversation_id
        )
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(list(context_probe.segments))
        )
        prompt = V15_CONTROL_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_lines(output, "CUE")[:2]

        # 3 searches: question + up to 2 cues
        queries = [question] + cues
        # If fewer than 2 cues parsed, fall back to question-only splits
        queries = queries[:3]
        while len(queries) < 3:
            queries.append(question)  # dummy fallback (shouldn't trigger often)

        splits = split_budget(K, 3)
        per_search: list[list[Segment]] = []
        exclude: set[int] = set()
        for q, k in zip(queries, splits):
            emb = self.embed_text(q)
            res = self.store.search(
                emb, top_k=k, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            segs = list(res.segments)
            per_search.append(segs)
            for s in segs:
                exclude.add(s.index)

        merged = merge_search_results(per_search, K)
        return VariantResult(
            segments=merged,
            metadata={
                "name": self.name,
                "searches": 3,
                "split": splits,
                "cues": cues,
                "output": output,
            },
        )


class V2fV2SplitVariant(VariantBase):
    """v2f_v2 (v2f without anti-question): question + 2 cues, K split 3-ways."""

    name = "v2f_v2_split"

    def retrieve(
        self, question: str, conversation_id: str, K: int,
    ) -> VariantResult:
        q_emb = self.embed_text(question)
        context_probe = self.store.search(
            q_emb, top_k=10, conversation_id=conversation_id
        )
        context_section = (
            "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n"
            + _format_segments(list(context_probe.segments))
        )
        prompt = V2F_V2_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = _parse_lines(output, "CUE")[:2]

        queries = [question] + cues
        queries = queries[:3]
        while len(queries) < 3:
            queries.append(question)

        splits = split_budget(K, 3)
        per_search: list[list[Segment]] = []
        exclude: set[int] = set()
        for q, k in zip(queries, splits):
            emb = self.embed_text(q)
            res = self.store.search(
                emb, top_k=k, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            segs = list(res.segments)
            per_search.append(segs)
            for s in segs:
                exclude.add(s.index)

        merged = merge_search_results(per_search, K)
        return VariantResult(
            segments=merged,
            metadata={
                "name": self.name,
                "searches": 3,
                "split": splits,
                "cues": cues,
                "output": output,
            },
        )


class QueryRewrite3Variant(VariantBase):
    """Original question + 2 rewrites. 3 searches, NO exclusion between them.

    Interleaved merge with dedup produces exactly K segments.
    """

    name = "query_rewrite_3"

    def retrieve(
        self, question: str, conversation_id: str, K: int,
    ) -> VariantResult:
        prompt = QUERY_REWRITE_PROMPT.format(question=question)
        output = self.llm_call(prompt)
        rewrites = _parse_lines(output, "REWRITE")[:2]
        queries = [question] + rewrites
        queries = queries[:3]
        while len(queries) < 3:
            queries.append(question)

        splits = split_budget(K, 3)
        # Oversample each search a bit to ensure post-dedup we can fill K.
        # The merge step truncates to K anyway.
        per_search: list[list[Segment]] = []
        for q, k in zip(queries, splits):
            emb = self.embed_text(q)
            # Oversample by 2x up to K to cushion duplicates during merge.
            oversample_k = min(K, max(k * 2, k))
            res = self.store.search(
                emb, top_k=oversample_k, conversation_id=conversation_id,
            )
            # But we only "allocate" k slots per search in the interleaving.
            segs = list(res.segments)[:k]
            per_search.append(segs)

        merged = merge_search_results(per_search, K)
        # If dedup dropped us below K, backfill from oversampled pool.
        if len(merged) < K:
            extra_pool = []
            for q, k in zip(queries, splits):
                emb = self.embed_text(q)
                oversample_k = min(K, max(k * 2, k))
                res = self.store.search(
                    emb, top_k=oversample_k, conversation_id=conversation_id,
                )
                extra_pool.extend(res.segments)
            seen = {s.index for s in merged}
            for s in extra_pool:
                if len(merged) >= K:
                    break
                if s.index not in seen:
                    merged.append(s)
                    seen.add(s.index)

        return VariantResult(
            segments=merged,
            metadata={
                "name": self.name,
                "searches": 3,
                "split": splits,
                "rewrites": rewrites,
                "output": output,
            },
        )


class HydeVariant(VariantBase):
    """Classic HyDE: one hypothetical answer, one search, top-K."""

    name = "hyde"

    def retrieve(
        self, question: str, conversation_id: str, K: int,
    ) -> VariantResult:
        prompt = HYDE_PROMPT.format(question=question)
        output = self.llm_call(prompt)
        answers = _parse_lines(output, "ANSWER")
        hyde_text = answers[0] if answers else question

        emb = self.embed_text(hyde_text)
        res = self.store.search(
            emb, top_k=K, conversation_id=conversation_id,
        )
        return VariantResult(
            segments=list(res.segments),
            metadata={
                "name": self.name,
                "searches": 1,
                "split": [K],
                "hyde_answer": hyde_text,
                "output": output,
            },
        )


class Decomposition4Variant(VariantBase):
    """4 semantic primitives. K split 4-ways, no exclusion between primitives."""

    name = "decomposition_4"

    def retrieve(
        self, question: str, conversation_id: str, K: int,
    ) -> VariantResult:
        prompt = DECOMPOSITION_PROMPT.format(question=question)
        output = self.llm_call(prompt)
        components = _parse_lines(output, "COMPONENT")[:4]
        while len(components) < 4:
            # Fallback: pad with the question itself if decomposition fails.
            components.append(question)

        splits = split_budget(K, 4)
        per_search: list[list[Segment]] = []
        for c, k in zip(components, splits):
            emb = self.embed_text(c)
            oversample_k = min(K, max(k * 2, k))
            res = self.store.search(
                emb, top_k=oversample_k, conversation_id=conversation_id,
            )
            segs = list(res.segments)[:k]
            per_search.append(segs)

        merged = merge_search_results(per_search, K)
        if len(merged) < K:
            # Backfill from all components' oversampled pools.
            extra_pool = []
            for c, k in zip(components, splits):
                emb = self.embed_text(c)
                oversample_k = min(K, max(k * 2, k))
                res = self.store.search(
                    emb, top_k=oversample_k, conversation_id=conversation_id,
                )
                extra_pool.extend(res.segments)
            seen = {s.index for s in merged}
            for s in extra_pool:
                if len(merged) >= K:
                    break
                if s.index not in seen:
                    merged.append(s)
                    seen.add(s.index)

        return VariantResult(
            segments=merged,
            metadata={
                "name": self.name,
                "searches": 4,
                "split": splits,
                "components": components,
                "output": output,
            },
        )


class QuestionPlusNoiseVariant(VariantBase):
    """Adversarial control: original embedding + 2 Gaussian-perturbed copies.

    Uses NO LLM. Tests whether cue CONTENT matters or just the fact that we
    search multiple different embedding regions.
    """

    name = "question_plus_noise"

    def retrieve(
        self, question: str, conversation_id: str, K: int,
    ) -> VariantResult:
        q_emb = self.embed_text(question)
        # Deterministic noise using fixed seeds per question so results are
        # reproducible and cacheable at the segment level.
        embs = [q_emb]
        for seed in NOISE_SEEDS:
            # Seed per question for determinism across runs.
            local_seed = (seed * 1_000_003) ^ (hash(question) & 0xFFFFFFFF)
            rng = np.random.default_rng(local_seed)
            noise = rng.normal(
                loc=0.0, scale=NOISE_SIGMA, size=q_emb.shape
            ).astype(np.float32)
            perturbed = q_emb + noise
            # The store normalizes inside search; still, we keep the perturbed
            # vector as-is (search divides by its norm).
            embs.append(perturbed)

        splits = split_budget(K, 3)
        per_search: list[list[Segment]] = []
        for emb, k in zip(embs, splits):
            oversample_k = min(K, max(k * 2, k))
            res = self.store.search(
                emb, top_k=oversample_k, conversation_id=conversation_id,
            )
            segs = list(res.segments)[:k]
            per_search.append(segs)

        merged = merge_search_results(per_search, K)
        if len(merged) < K:
            extra_pool = []
            for emb, k in zip(embs, splits):
                oversample_k = min(K, max(k * 2, k))
                res = self.store.search(
                    emb, top_k=oversample_k, conversation_id=conversation_id,
                )
                extra_pool.extend(res.segments)
            seen = {s.index for s in merged}
            for s in extra_pool:
                if len(merged) >= K:
                    break
                if s.index not in seen:
                    merged.append(s)
                    seen.add(s.index)

        return VariantResult(
            segments=merged,
            metadata={
                "name": self.name,
                "searches": 3,
                "split": splits,
                "sigma": NOISE_SIGMA,
                "seeds": NOISE_SEEDS,
            },
        )


VARIANTS = {
    CosineOnlyVariant.name: CosineOnlyVariant,
    V15ControlSplitVariant.name: V15ControlSplitVariant,
    V2fV2SplitVariant.name: V2fV2SplitVariant,
    QueryRewrite3Variant.name: QueryRewrite3Variant,
    HydeVariant.name: HydeVariant,
    Decomposition4Variant.name: Decomposition4Variant,
    QuestionPlusNoiseVariant.name: QuestionPlusNoiseVariant,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment], max_items: int = 10, max_chars: int = 250,
) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    return "\n".join(
        f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}"
        for seg in sorted_segs
    )


def load_dataset(ds_name: str) -> tuple[SegmentStore, list[dict]]:
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        questions = json.load(f)
    if cfg["filter"]:
        questions = [q for q in questions if cfg["filter"](q)]
    if cfg["max_questions"]:
        questions = questions[: cfg["max_questions"]]
    return store, questions


def compute_recall(retrieved_turn_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_turn_ids & source_ids) / len(source_ids)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_question(
    variant: VariantBase, question: dict,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    row: dict = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "num_source_turns": len(source_ids),
        "per_budget": {},
    }

    per_budget_meta: dict[int, dict] = {}

    for K in BUDGETS:
        variant.reset_counters()
        t0 = time.time()
        result = variant.retrieve(q_text, conv_id, K)
        elapsed = time.time() - t0

        retrieved_ids = {s.turn_id for s in result.segments[:K]}
        recall = compute_recall(retrieved_ids, source_ids)

        row["per_budget"][f"r@{K}"] = round(recall, 4)
        row["per_budget"][f"n_retrieved@{K}"] = len(result.segments)
        row["per_budget"][f"embed_calls@{K}"] = variant.embed_calls
        row["per_budget"][f"llm_calls@{K}"] = variant.llm_calls
        row["per_budget"][f"time_s@{K}"] = round(elapsed, 2)

        # Keep only the lightweight metadata (strings), not segments.
        meta_copy = {
            k: v for k, v in result.metadata.items()
            if k not in ("segments",)
        }
        per_budget_meta[K] = meta_copy

    row["metadata"] = per_budget_meta
    return row


def summarize(
    results: list[dict], variant_name: str, dataset: str,
) -> dict:
    n = len(results)
    summary: dict = {"variant": variant_name, "dataset": dataset, "n": n}
    if n == 0:
        return summary
    for K in BUDGETS:
        vals = [r["per_budget"][f"r@{K}"] for r in results]
        summary[f"r@{K}"] = round(sum(vals) / n, 4)
        summary[f"avg_embed@{K}"] = round(
            sum(r["per_budget"][f"embed_calls@{K}"] for r in results) / n, 2
        )
        summary[f"avg_llm@{K}"] = round(
            sum(r["per_budget"][f"llm_calls@{K}"] for r in results) / n, 2
        )
        summary[f"avg_time_s@{K}"] = round(
            sum(r["per_budget"][f"time_s@{K}"] for r in results) / n, 2
        )
    return summary


def summarize_by_category(results: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out: dict[str, dict] = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry: dict = {"n": n}
        for K in BUDGETS:
            vals = [r["per_budget"][f"r@{K}"] for r in rs]
            entry[f"r@{K}"] = round(sum(vals) / n, 4)
        out[cat] = entry
    return out


def compare_vs_baseline(
    results: list[dict], baseline_results: list[dict],
) -> dict:
    """Paired comparison against a baseline variant (same questions)."""
    # Index baseline by (conversation_id, question_index)
    base_by_key = {
        (r["conversation_id"], r["question_index"]): r
        for r in baseline_results
    }
    out: dict = {}
    for K in BUDGETS:
        wins = losses = ties = 0
        deltas = []
        for r in results:
            key = (r["conversation_id"], r["question_index"])
            b = base_by_key.get(key)
            if b is None:
                continue
            a_val = r["per_budget"][f"r@{K}"]
            b_val = b["per_budget"][f"r@{K}"]
            deltas.append(a_val - b_val)
            if a_val > b_val + 0.001:
                wins += 1
            elif b_val > a_val + 0.001:
                losses += 1
            else:
                ties += 1
        n = len(deltas)
        out[f"delta_r@{K}"] = (
            round(sum(deltas) / n, 4) if n else 0.0
        )
        out[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
    return out


def run_variant(
    variant_name: str,
    dataset: str,
    store: SegmentStore,
    questions: list[dict],
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 70}")
    print(f"{variant_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}")

    cls = VARIANTS[variant_name]
    variant = cls(store)

    results: list[dict] = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i+1}/{len(questions)}] "
            f"{q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(variant, q)
            results.append(row)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            variant.save_caches()

    variant.save_caches()
    summary = summarize(results, variant_name, dataset)
    by_cat = summarize_by_category(results)

    print(f"\n--- {variant_name} on {dataset} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: {summary[f'r@{K}']:.4f} "
            f"(LLM={summary[f'avg_llm@{K}']:.2f}, "
            f"embed={summary[f'avg_embed@{K}']:.2f})"
        )
    print("  Per-category:")
    for cat, c in by_cat.items():
        print(
            f"    {cat:24s} (n={c['n']:2d}): "
            f"r@20={c['r@20']:.3f} r@50={c['r@50']:.3f}"
        )
    return results, summary, by_cat


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Run a single variant (default: all)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Run a single dataset (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List variants and datasets",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing per-(variant, dataset) results",
    )
    args = parser.parse_args()

    if args.list:
        print("Variants:")
        for v in VARIANTS:
            print(f"  {v}")
        print("\nDatasets:")
        for d in DATASETS:
            print(f"  {d}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variant_names = [args.variant] if args.variant else list(VARIANTS)
    dataset_names = [args.dataset] if args.dataset else list(DATASETS)

    for v in variant_names:
        if v not in VARIANTS:
            raise SystemExit(
                f"Unknown variant: {v}. Available: {list(VARIANTS)}"
            )
    for d in dataset_names:
        if d not in DATASETS:
            raise SystemExit(
                f"Unknown dataset: {d}. Available: {list(DATASETS)}"
            )

    # Per-dataset loaded store (reuse across variants).
    store_cache: dict[str, tuple[SegmentStore, list[dict]]] = {}
    for d in dataset_names:
        store, questions = load_dataset(d)
        store_cache[d] = (store, questions)
        print(
            f"Loaded {d}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

    all_results: dict[str, dict[str, list[dict]]] = {}
    all_summaries: dict[str, dict[str, dict]] = {}
    all_by_cat: dict[str, dict[str, dict]] = {}

    for d in dataset_names:
        store, questions = store_cache[d]
        for v in variant_names:
            out_path = RESULTS_DIR / f"query_rewrite_{v}_{d}.json"
            if out_path.exists() and not args.force:
                print(f"Skipping {v} on {d} (exists). Use --force to rerun.")
                with open(out_path) as f:
                    saved = json.load(f)
                results = saved["results"]
                summary = saved["summary"]
                by_cat = saved.get("category_breakdown", {})
            else:
                results, summary, by_cat = run_variant(
                    v, d, store, questions,
                )
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "variant": v,
                            "dataset": d,
                            "summary": summary,
                            "category_breakdown": by_cat,
                            "results": results,
                        },
                        f,
                        indent=2,
                        default=str,
                    )
                print(f"  Saved: {out_path}")

            all_results.setdefault(v, {})[d] = results
            all_summaries.setdefault(v, {})[d] = summary
            all_by_cat.setdefault(v, {})[d] = by_cat

    # Paired comparisons vs v15_control_split (the canonical cue baseline).
    if "v15_control_split" in all_results:
        comparisons: dict[str, dict[str, dict]] = {}
        for v in variant_names:
            if v == "v15_control_split":
                continue
            comparisons[v] = {}
            for d in dataset_names:
                if (
                    v not in all_results
                    or d not in all_results[v]
                    or d not in all_results.get("v15_control_split", {})
                ):
                    continue
                comp = compare_vs_baseline(
                    all_results[v][d],
                    all_results["v15_control_split"][d],
                )
                comparisons[v][d] = comp

        comp_path = RESULTS_DIR / "query_rewrite_vs_v15_control_split.json"
        with open(comp_path, "w") as f:
            json.dump(comparisons, f, indent=2, default=str)
        print(f"\nSaved paired comparisons: {comp_path}")

    # Aggregated summary file
    summary_path = RESULTS_DIR / "query_rewrite_all_summaries.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "summaries": all_summaries,
                "category_breakdowns": all_by_cat,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved summary: {summary_path}")

    # Final comparison table
    print("\n" + "=" * 100)
    print("QUERY REWRITING vs CUE GENERATION — FINAL TABLE")
    print("=" * 100)
    header = (
        f"{'Variant':<24s} {'Dataset':<14s} "
        f"{'r@20':>8s} {'r@50':>8s} "
        f"{'LLM@20':>8s} {'emb@20':>8s}"
    )
    print(header)
    print("-" * len(header))
    for v in variant_names:
        for d in dataset_names:
            if v not in all_summaries or d not in all_summaries[v]:
                continue
            s = all_summaries[v][d]
            print(
                f"{v:<24s} {d:<14s} "
                f"{s.get('r@20', 0):>8.4f} {s.get('r@50', 0):>8.4f} "
                f"{s.get('avg_llm@20', 0):>8.2f} "
                f"{s.get('avg_embed@20', 0):>8.2f}"
            )


if __name__ == "__main__":
    main()
