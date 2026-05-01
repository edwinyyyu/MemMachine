"""Circular batch relevance filtering as a retrieval post-processor.

Strategy:
  1. Run v2f-style retrieval with aggressive expansion:
     - Initial: cosine top-20 on the raw question
     - Cues: 1 LLM call generates 4 cues, each retrieves top-15
     - Total pool: up to ~80 segments, deduplicated, ordered by retrieval
  2. Apply circular batch filter:
     - Partition the pool into overlapping batches (each segment appears in
       exactly 2 batches in the default variant)
     - Each batch goes to a parallel LLM call that marks "useless" segments
     - Consensus: segment removed only if BOTH batches agree it's useless
  3. Take top-K of survivors (ordered by retrieval order from step 1)

Variants:
  A. batch_filter_consensus : batch=20, segment in 2 batches, votes >= 2 removes
  B. batch_filter_strict    : batch=40, votes >= num_batches removes (all agree)
  C. batch_filter_single    : batch=20, segment in 1 batch (no overlap, no consensus)

Cache: cache/circ_filter_llm_cache.json
Results: results/circ_filter_<variant>_<dataset>.json (saved incrementally)

Usage:
    uv run python circular_filter.py [--variant A|B|C|all] [--dataset <name>] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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
from openai import AsyncOpenAI, OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50]

# Over-retrieval configuration
INITIAL_TOP_K = 20
NUM_CUES = 4
CUE_TOP_K = 15
TOTAL_POOL_TARGET = 80  # informational: 20 + 4*15 = 80 before dedup


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
@dataclass
class VariantConfig:
    name: str
    batch_size: int
    # How votes translate to removal. 'consensus_2' means votes >= 2 (i.e.
    # segment is in exactly 2 batches and both flagged it). 'all_agree' means
    # vote count == number_of_batches_containing_segment. 'single' means
    # vote count >= 1 (only one batch per segment, no overlap).
    vote_rule: str  # "consensus_2" | "all_agree" | "single"
    # Number of batches each segment should appear in (1 for variant C)
    segment_appearances: int  # 1 or 2


VARIANTS: dict[str, VariantConfig] = {
    "A": VariantConfig(
        name="v2f_plus_batch_filter",
        batch_size=20,
        vote_rule="consensus_2",
        segment_appearances=2,
    ),
    "B": VariantConfig(
        name="v2f_batch_filter_strict",
        batch_size=40,
        vote_rule="all_agree",
        segment_appearances=2,
    ),
    "C": VariantConfig(
        name="v2f_batch_filter_single",
        batch_size=20,
        vote_rule="single",
        segment_appearances=1,
    ),
}


# ---------------------------------------------------------------------------
# Datasets (mirrors fair_backfill_eval.py)
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
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


# ---------------------------------------------------------------------------
# Caches: read existing caches for warm-up, write to dedicated files
# ---------------------------------------------------------------------------
class CircFilterLLMCache(LLMCache):
    """Dedicated cache for circular filter runs, warm-loads existing LLM caches.

    The cache stores raw response text keyed by (model, prompt).
    """

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        # Warm-load bestshot/fulleval caches so v2f cue-gen calls get hits.
        for name in (
            "llm_cache.json",
            "bestshot_llm_cache.json",
            "fulleval_llm_cache.json",
            "circ_filter_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "circ_filter_llm_cache.json"
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


class CircFilterEmbeddingCache(EmbeddingCache):
    """Warm-loads existing embedding caches so cue embeddings get hits."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "bestshot_embedding_cache.json",
            "fulleval_embedding_cache.json",
            "circ_filter_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "circ_filter_embedding_cache.json"
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


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
V2F_CUE_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate {num_cues} search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns. Each cue \
should target a DIFFERENT aspect of the question.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
CUE: <text>
CUE: <text>
Nothing else."""


FILTER_SYSTEM_PROMPT = (
    "You are a relevance judge. Given a query and a list of indexed "
    "conversation turns, identify which turns are NOT useful for answering "
    "the query.\n\n"
    "A turn is useful if:\n"
    "1. It is directly relevant to the query, OR\n"
    "2. It provides important context that makes another useful turn "
    "easier to understand (e.g., a question that precedes a relevant answer, "
    "or a setup message that gives meaning to a later statement).\n\n"
    "Return the indexes of useless turns — those that are neither directly "
    "relevant nor contextually supporting any relevant turn. If all turns "
    "are useful, return an empty list.\n\n"
    "Respond with JSON matching this schema:\n"
    '{"useless_indexes": [<int>, ...]}\n'
    "Only include indexes that appear in the batch. Return valid JSON only, "
    "no prose."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_pool_for_cue(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    if not segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = [f"[Turn {s.turn_id}, {s.role}]: {s.text[:max_chars]}" for s in sorted_segs]
    return "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + "\n".join(lines)


def _parse_cues(response: str) -> list[str]:
    cues: list[str] = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


def _indexed_batch_string(batch: list[Segment]) -> str:
    """Render a batch as indexed text for the filter LLM."""
    lines = []
    for i, seg in enumerate(batch):
        text = seg.text.replace("\n", " ")
        if len(text) > 350:
            text = text[:350] + "..."
        lines.append(f"[{i}] (Turn {seg.turn_id}, {seg.role}): {text}")
    return "\n".join(lines)


def build_circular_batches(
    pool_size: int,
    batch_size: int,
    segment_appearances: int = 2,
) -> list[list[int]]:
    """Build overlapping circular batches over pool indexes.

    For segment_appearances == 2:
      Partition [0, pool_size) into B balanced segments (where B is chosen so
      each batch = two adjacent segments has ~batch_size items). Batch k is
      segment k union segment (k+1) mod B. Each pool index appears in exactly
      2 batches.

    For segment_appearances == 1:
      Partition [0, pool_size) into ceil(pool_size / batch_size) contiguous
      batches of ~batch_size items. Each pool index appears in exactly 1 batch.
    """
    if pool_size <= 0:
        return []

    if segment_appearances == 1:
        num_batches = max(1, math.ceil(pool_size / batch_size))
        seg_start = [(k * pool_size) // num_batches for k in range(num_batches + 1)]
        batches = []
        for k in range(num_batches):
            idxs = list(range(seg_start[k], seg_start[k + 1]))
            if idxs:
                batches.append(idxs)
        return batches

    # segment_appearances == 2: overlapping circular
    if pool_size <= batch_size:
        # Degenerate: single batch covers everything; duplicate it so each
        # segment appears in 2 batches and the consensus rule still applies.
        base = list(range(pool_size))
        return [base, list(base)]

    num_batches = max(2, math.ceil(2 * pool_size / batch_size))
    num_segments = num_batches  # each batch = union of 2 adjacent segments
    seg_start = [(k * pool_size) // num_segments for k in range(num_segments + 1)]
    batches: list[list[int]] = []
    for k in range(num_batches):
        next_k = (k + 1) % num_segments
        idxs = list(range(seg_start[k], seg_start[k + 1]))
        idxs += list(range(seg_start[next_k], seg_start[next_k + 1]))
        idxs.sort()
        if idxs:
            batches.append(idxs)
    return batches


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class CircularFilterEngine:
    def __init__(
        self,
        store: SegmentStore,
        variant: VariantConfig,
        sync_client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ):
        self.store = store
        self.variant = variant
        self.sync_client = sync_client or OpenAI(timeout=60.0)
        self.async_client = async_client or AsyncOpenAI(timeout=60.0)
        self.embedding_cache = CircFilterEmbeddingCache()
        self.llm_cache = CircFilterLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    # ---- embedding ----
    def embed_text(self, text: str) -> np.ndarray:
        t = text.strip()
        if not t:
            return np.zeros(1536, dtype=np.float32)
        cached = self.embedding_cache.get(t)
        if cached is not None:
            self.embed_calls += 1
            return cached
        response = self.sync_client.embeddings.create(model=EMBED_MODEL, input=[t])
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(t, emb)
        self.embed_calls += 1
        return emb

    # ---- sync LLM for cue generation ----
    def _llm_cue(self, prompt: str, model: str = MODEL) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        response = self.sync_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(model, prompt, text)
        self.llm_calls += 1
        return text

    # ---- async LLM for filter calls (batches run in parallel) ----
    async def _llm_filter_async(self, user_prompt: str, model: str = MODEL) -> str:
        # Cache key is (model, system + user) -- include system for determinism.
        full_prompt = FILTER_SYSTEM_PROMPT + "\n\n" + user_prompt
        cached = self.llm_cache.get(model, full_prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2000,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(model, full_prompt, text)
        self.llm_calls += 1
        return text

    # ---- over-retrieval (v2f-style with wider fan-out) ----
    def over_retrieve(
        self,
        question: str,
        conversation_id: str,
    ) -> tuple[list[Segment], list[str]]:
        """Initial top-20 + 4 cues × top-15. Returns (pool, cues)."""
        # Initial retrieval on raw question
        q_emb = self.embed_text(question)
        hop0 = self.store.search(
            q_emb, top_k=INITIAL_TOP_K, conversation_id=conversation_id
        )
        pool: list[Segment] = list(hop0.segments)
        exclude: set[int] = {s.index for s in pool}

        # One LLM call to generate 4 cues
        context_section = _format_pool_for_cue(pool)
        prompt = V2F_CUE_PROMPT.format(
            question=question,
            context_section=context_section,
            num_cues=NUM_CUES,
        )
        output = self._llm_cue(prompt)
        cues = _parse_cues(output)[:NUM_CUES]

        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=CUE_TOP_K,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    pool.append(seg)
                    exclude.add(seg.index)

        return pool, cues

    # ---- filter step ----
    async def filter_pool(
        self,
        question: str,
        pool: list[Segment],
    ) -> tuple[set[int], dict]:
        """Return (indexes_to_remove_from_pool, filter_stats)."""
        n = len(pool)
        variant = self.variant
        batches = build_circular_batches(
            n, variant.batch_size, variant.segment_appearances
        )
        if not batches:
            return set(), {
                "num_batches": 0,
                "useless_votes": {},
                "removed_count": 0,
                "batch_sizes": [],
            }

        # For each batch, build the prompt and call the LLM in parallel.
        async def run_batch(batch_idx: int, pool_indexes: list[int]) -> set[int]:
            batch_segments = [pool[i] for i in pool_indexes]
            indexed = _indexed_batch_string(batch_segments)
            user_prompt = f"Query: {question}\n\nTurns:\n{indexed}"
            try:
                text = await self._llm_filter_async(user_prompt)
            except Exception as e:
                print(f"    filter batch {batch_idx} error: {e}", flush=True)
                return set()
            # Parse JSON response
            try:
                data = json.loads(text)
                raw = data.get("useless_indexes", [])
            except Exception:
                return set()
            out: set[int] = set()
            for x in raw:
                try:
                    i = int(x)
                except Exception:
                    continue
                if 0 <= i < len(batch_segments):
                    out.add(pool_indexes[i])
            return out

        results = await asyncio.gather(
            *[run_batch(bi, b) for bi, b in enumerate(batches)]
        )

        # Tally votes per pool index.
        votes: dict[int, int] = defaultdict(int)
        appearances: dict[int, int] = defaultdict(int)
        for batch_indexes, flagged in zip(batches, results):
            for idx in batch_indexes:
                appearances[idx] += 1
            for idx in flagged:
                votes[idx] += 1

        # Apply vote rule
        remove: set[int] = set()
        if variant.vote_rule == "consensus_2":
            remove = {i for i, v in votes.items() if v >= 2}
        elif variant.vote_rule == "all_agree":
            remove = {
                i
                for i, v in votes.items()
                if appearances[i] > 0 and v >= appearances[i]
            }
        elif variant.vote_rule == "single":
            remove = {i for i, v in votes.items() if v >= 1}
        else:
            raise ValueError(f"unknown vote_rule: {variant.vote_rule}")

        stats = {
            "num_batches": len(batches),
            "batch_sizes": [len(b) for b in batches],
            "pool_size": n,
            "total_flags": sum(len(r) for r in results),
            "flagged_unique": len(votes),
            "removed_count": len(remove),
            "survivors": n - len(remove),
            "vote_rule": variant.vote_rule,
            "batch_size_cfg": variant.batch_size,
            "segment_appearances": variant.segment_appearances,
        }
        return remove, stats

    # ---- full pipeline ----
    async def retrieve(
        self,
        question: str,
        conversation_id: str,
    ) -> tuple[list[Segment], list[Segment], list[str], dict]:
        """Return (survivors_in_retrieval_order, full_pool, cues, filter_stats)."""
        pool, cues = self.over_retrieve(question, conversation_id)
        remove, stats = await self.filter_pool(question, pool)
        survivors = [s for i, s in enumerate(pool) if i not in remove]
        return survivors, pool, cues, stats


# ---------------------------------------------------------------------------
# Evaluation (fair backfill)
# ---------------------------------------------------------------------------
def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    # dedupe arch
    seen: set[int] = set()
    uniq: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            uniq.append(s)
            seen.add(s.index)
    arch_at_K = uniq[:budget]
    arch_idx = {s.index for s in arch_at_K}
    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_idx]
        needed = budget - len(arch_at_K)
        arch_at_K = arch_at_K + backfill[:needed]
    arch_at_K = arch_at_K[:budget]
    base_at_K = cosine_segments[:budget]

    arch_ids = {s.turn_id for s in arch_at_K}
    base_ids = {s.turn_id for s in base_at_K}
    return (
        compute_recall(base_ids, source_ids),
        compute_recall(arch_ids, source_ids),
    )


async def evaluate_one(
    engine: CircularFilterEngine,
    question: dict,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    engine.reset_counters()
    t0 = time.time()
    survivors, pool, cues, fstats = await engine.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Cosine top-max(BUDGETS) for baseline + backfill
    q_emb = engine.embed_text(q_text)
    max_K = max(BUDGETS)
    cosine_result = engine.store.search(q_emb, top_k=max_K, conversation_id=conv_id)
    cosine_segments = list(cosine_result.segments)

    row = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "pool_size": len(pool),
        "survivor_count": len(survivors),
        "total_arch_retrieved": len(survivors),
        "embed_calls": engine.embed_calls,
        "llm_calls": engine.llm_calls,
        "time_s": round(elapsed, 2),
        "cues": cues,
        "filter_stats": fstats,
        "fair_backfill": {},
        # Also track recall on the unfiltered pool for reference
        "pool_fair_backfill": {},
    }

    for K in BUDGETS:
        b, a = fair_backfill(survivors, cosine_segments, source_ids, K)
        row["fair_backfill"][f"baseline_r@{K}"] = round(b, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a - b, 4)

        pb, pa = fair_backfill(pool, cosine_segments, source_ids, K)
        row["pool_fair_backfill"][f"arch_r@{K}"] = round(pa, 4)
        row["pool_fair_backfill"][f"delta_r@{K}"] = round(pa - b, 4)

    return row


def summarize(results: list[dict], variant_name: str, dataset: str) -> dict:
    n = len(results)
    if n == 0:
        return {"variant": variant_name, "dataset": dataset, "n": 0}
    s: dict = {"variant": variant_name, "dataset": dataset, "n": n}
    for K in BUDGETS:
        b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in results]
        a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in results]
        p_vals = [r["pool_fair_backfill"][f"arch_r@{K}"] for r in results]
        bm, am, pm = sum(b_vals) / n, sum(a_vals) / n, sum(p_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses
        s[f"baseline_r@{K}"] = round(bm, 4)
        s[f"arch_r@{K}"] = round(am, 4)
        s[f"pool_r@{K}"] = round(pm, 4)
        s[f"delta_r@{K}"] = round(am - bm, 4)
        s[f"pool_delta_r@{K}"] = round(pm - bm, 4)
        s[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"

    s["avg_pool_size"] = round(sum(r["pool_size"] for r in results) / n, 1)
    s["avg_survivors"] = round(sum(r["survivor_count"] for r in results) / n, 1)
    s["avg_removed"] = round(
        sum(r["filter_stats"].get("removed_count", 0) for r in results) / n, 1
    )
    s["avg_num_batches"] = round(
        sum(r["filter_stats"].get("num_batches", 0) for r in results) / n, 2
    )
    s["avg_total_flags"] = round(
        sum(r["filter_stats"].get("total_flags", 0) for r in results) / n, 1
    )
    s["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 2)
    s["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 2)
    s["avg_time_s"] = round(sum(r["time_s"] for r in results) / n, 2)
    # Filter aggressiveness
    total_poolsz = sum(r["pool_size"] for r in results)
    total_removed = sum(r["filter_stats"].get("removed_count", 0) for r in results)
    s["pct_removed"] = (
        round(100 * total_removed / total_poolsz, 1) if total_poolsz else 0.0
    )
    return s


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out: dict[str, dict] = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry: dict = {"n": n}
        for K in BUDGETS:
            bvs = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            avs = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            pvs = [r["pool_fair_backfill"][f"arch_r@{K}"] for r in rs]
            bm, am, pm = sum(bvs) / n, sum(avs) / n, sum(pvs) / n
            wins = sum(1 for b, a in zip(bvs, avs) if a > b + 0.001)
            losses = sum(1 for b, a in zip(bvs, avs) if b > a + 0.001)
            ties = n - wins - losses
            entry[f"baseline_r@{K}"] = round(bm, 4)
            entry[f"arch_r@{K}"] = round(am, 4)
            entry[f"pool_r@{K}"] = round(pm, 4)
            entry[f"delta_r@{K}"] = round(am - bm, 4)
            entry[f"pool_delta_r@{K}"] = round(pm - bm, 4)
            entry[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
        out[cat] = entry
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def run_variant_dataset(
    variant_key: str,
    dataset: str,
    force: bool = False,
) -> None:
    variant = VARIANTS[variant_key]
    tag = f"{variant.name}_{dataset}"
    out_path = RESULTS_DIR / f"circ_filter_{tag}.json"

    # Resume from existing file if present (unless --force)
    existing_results: list[dict] = []
    existing_keys: set[tuple[str, int]] = set()
    if out_path.exists() and not force:
        try:
            with open(out_path) as f:
                saved = json.load(f)
            existing_results = saved.get("results", [])
            for r in existing_results:
                existing_keys.add((r["conversation_id"], r["question_index"]))
            print(
                f"\nResuming {tag}: {len(existing_results)} already done",
                flush=True,
            )
        except Exception as e:
            print(f"  Could not load {out_path}: {e}")
            existing_results = []
            existing_keys = set()

    store, questions = load_dataset(dataset)
    print(
        f"\n{'=' * 70}\nvariant={variant_key} ({variant.name}) | "
        f"dataset={dataset} | {len(questions)} Qs | "
        f"pool={len(store.segments)} segs\n{'=' * 70}",
        flush=True,
    )

    engine = CircularFilterEngine(store, variant)
    results: list[dict] = list(existing_results)

    for i, q in enumerate(questions):
        key = (q["conversation_id"], q.get("question_index", -1))
        if key in existing_keys:
            continue
        q_short = q["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = await evaluate_one(engine, q)
            results.append(row)
            b20 = row["fair_backfill"]["baseline_r@20"]
            a20 = row["fair_backfill"]["arch_r@20"]
            p20 = row["pool_fair_backfill"]["arch_r@20"]
            print(
                f"    r@20: base={b20:.3f} arch={a20:.3f} (pool={p20:.3f}) "
                f"pool={row['pool_size']} surv={row['survivor_count']} "
                f"batches={row['filter_stats'].get('num_batches', 0)} "
                f"llm={row['llm_calls']} t={row['time_s']:.1f}s",
                flush=True,
            )
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()
            continue

        # Incremental save
        summary = summarize(results, variant.name, dataset)
        by_cat = summarize_by_category(results)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "variant": variant.name,
                    "variant_key": variant_key,
                    "variant_config": {
                        "batch_size": variant.batch_size,
                        "vote_rule": variant.vote_rule,
                        "segment_appearances": variant.segment_appearances,
                    },
                    "dataset": dataset,
                    "summary": summary,
                    "category_breakdown": by_cat,
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )
        if (len(results) % 3) == 0:
            engine.save_caches()

    engine.save_caches()

    # Final summary print
    summary = summarize(results, variant.name, dataset)
    by_cat = summarize_by_category(results)
    print(f"\n--- {variant.name} | {dataset} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: base={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"pool={summary[f'pool_r@{K}']:.3f} "
            f"d={summary[f'delta_r@{K}']:+.3f} "
            f"pool_d={summary[f'pool_delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(
        f"  avg pool={summary['avg_pool_size']:.1f} "
        f"survivors={summary['avg_survivors']:.1f} "
        f"removed={summary['avg_removed']:.1f} "
        f"(pct={summary['pct_removed']:.1f}%) "
        f"batches={summary['avg_num_batches']:.2f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"embed={summary['avg_embed_calls']:.1f}"
    )
    print("  Per-category:")
    for cat, c in by_cat.items():
        print(
            f"    {cat:26s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} (pool_d={c['pool_delta_r@20']:+.3f}) "
            f"r@50 d={c['delta_r@50']:+.3f} (pool_d={c['pool_delta_r@50']:+.3f})"
        )

    print(f"  Saved: {out_path}")


async def main_async(variants: list[str], datasets: list[str], force: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries: dict = {}

    for ds in datasets:
        for v in variants:
            await run_variant_dataset(v, ds, force=force)

            # Load the written result to gather summary
            tag = f"{VARIANTS[v].name}_{ds}"
            out_path = RESULTS_DIR / f"circ_filter_{tag}.json"
            if out_path.exists():
                with open(out_path) as f:
                    data = json.load(f)
                all_summaries.setdefault(VARIANTS[v].name, {})[ds] = {
                    "summary": data.get("summary", {}),
                    "category_breakdown": data.get("category_breakdown", {}),
                }

    summary_path = RESULTS_DIR / "circ_filter_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    # Final comparison table
    print("\n" + "=" * 110)
    print("CIRCULAR FILTER SUMMARY")
    print("=" * 110)
    header = (
        f"{'Variant':<28s} {'Dataset':<14s} "
        f"{'base@20':>8s} {'arch@20':>8s} {'pool@20':>8s} {'d@20':>7s} {'W/T/L@20':>10s} "
        f"{'base@50':>8s} {'arch@50':>8s} {'pool@50':>8s} {'d@50':>7s}"
    )
    print(header)
    print("-" * len(header))
    for vname, per_ds in all_summaries.items():
        for ds, entry in per_ds.items():
            s = entry["summary"]
            if not s or s.get("n", 0) == 0:
                continue
            print(
                f"{vname:<28s} {ds:<14s} "
                f"{s['baseline_r@20']:>8.3f} {s['arch_r@20']:>8.3f} "
                f"{s['pool_r@20']:>8.3f} {s['delta_r@20']:>+7.3f} "
                f"{s['W/T/L_r@20']:>10s} "
                f"{s['baseline_r@50']:>8.3f} {s['arch_r@50']:>8.3f} "
                f"{s['pool_r@50']:>8.3f} {s['delta_r@50']:>+7.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="all",
        help="A, B, C, or all (comma-separated also supported)",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset name or 'all' (comma-separated also supported)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results (default: resume)",
    )
    args = parser.parse_args()

    # Parse variants
    if args.variant == "all":
        variants = ["A", "B", "C"]
    else:
        variants = [v.strip() for v in args.variant.split(",") if v.strip()]
        for v in variants:
            if v not in VARIANTS:
                print(f"Unknown variant: {v}. Known: {list(VARIANTS)}")
                sys.exit(1)

    # Parse datasets
    if args.dataset == "all":
        datasets = list(DATASETS.keys())
    else:
        datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
        for d in datasets:
            if d not in DATASETS:
                print(f"Unknown dataset: {d}. Known: {list(DATASETS)}")
                sys.exit(1)

    asyncio.run(main_async(variants, datasets, force=args.force))


if __name__ == "__main__":
    main()
