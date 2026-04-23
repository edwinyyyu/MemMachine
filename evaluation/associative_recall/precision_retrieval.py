"""Precision retrieval experiments: reranker fairness + precision cue generation.

Investigation 1 - Reranker Fairness:
  A. cosine_top60_rerank: cosine top-60 -> LLM rerank -> top-20
  B. v2f_30_rerank: v2f (30 segments) -> LLM rerank -> top-20
  C. v2f_gencheck_50_rerank: v2f + gencheck (50 segments) -> LLM rerank -> top-20

Investigation 2 - Precision Cue Generation:
  A. backfill: baseline top-20 fixed + cue-found fill 21-50
  B. embedding_filter: skip cues too similar to question (cosine > threshold)
  C. cue_dedup_metric: track % overlap between cue-found and baseline
  D. what_baseline_missed: prompt that emphasizes finding DIFFERENT vocabulary

Usage:
    uv run python precision_retrieval.py [--verbose] [--investigation 1|2|both]
"""

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
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
    RetrievalResult,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = [20, 50]


def get_cache_paths(investigation: int = 0):
    """Return cache file paths, optionally scoped by investigation."""
    suffix = f"_inv{investigation}" if investigation else ""
    return (
        CACHE_DIR / f"precision_llm_cache{suffix}.json",
        CACHE_DIR / f"precision_embedding_cache{suffix}.json",
    )


# ---------------------------------------------------------------------------
# Cache classes
# ---------------------------------------------------------------------------
class PrecisionEmbeddingCache(EmbeddingCache):
    def __init__(self, investigation: int = 0):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        # Load main embedding cache + our own + other precision caches
        cache_names = [
            "embedding_cache.json",
            "pipeline_embedding_cache.json",
            "precision_embedding_cache.json",
            "precision_embedding_cache_inv1.json",
            "precision_embedding_cache_inv2.json",
        ]
        for name in cache_names:
            p = self.cache_dir / name
            if p.exists():
                print(f"    Loading embedding cache: {name}...", flush=True)
                with open(p) as f:
                    self._cache.update(json.load(f))
        _, emb_path = get_cache_paths(investigation)
        self.cache_file = emb_path
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


class PrecisionLLMCache(LLMCache):
    def __init__(self, investigation: int = 0):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        # Load key LLM caches + other precision caches
        cache_names = [
            "llm_cache.json",
            "bestshot_llm_cache.json",
            "optim_llm_cache.json",
            "fulleval_llm_cache.json",
            "pipeline_llm_cache.json",
            "rerank_llm_cache.json",
            "precision_llm_cache.json",
            "precision_llm_cache_inv1.json",
            "precision_llm_cache_inv2.json",
        ]
        for name in cache_names:
            p = self.cache_dir / name
            if p.exists():
                print(f"    Loading LLM cache: {name}...", flush=True)
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        llm_path, _ = get_cache_paths(investigation)
        self.cache_file = llm_path
        self._new_entries: dict[str, str] = {}

    def put(self, model: str, prompt: str, response: str) -> None:
        key = self._key(model, prompt)
        self._cache[key] = response
        self._new_entries[key] = response

    def save(self) -> None:
        if not self._new_entries:
            return
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# V2f prompt (from full_pipeline.py)
V2F_PROMPT = """\
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

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

# Gen-Check gap assessment (from full_pipeline.py)
GAP_ASSESSMENT_PROMPT = """\
You are reviewing retrieval results for a task. Given the question/task and \
the conversation segments retrieved so far, assess whether anything \
important is missing.

QUESTION/TASK: {question}

RETRIEVED SEGMENTS:
{formatted_segments}

Think critically:
1. Given what I've found, is there anything important for this task that \
I HAVEN'T retrieved?
2. What assumptions am I making that should be checked?
3. Are there implicit requirements (e.g., dietary restrictions, scheduling \
conflicts, prerequisites) that the question doesn't explicitly ask about \
but would be important?
4. If this is a proactive task (planning, drafting, preparing), what \
background information might I need that isn't directly mentioned in \
the question?

If there are genuine gaps, generate 1-2 targeted search cues. Each cue \
should sound like conversation content (not a search command).

If the retrieval looks comprehensive, respond with DONE.

Format:
ASSESSMENT: <what's missing or what assumptions need checking>
GAP: <text mimicking conversation content>
GAP: <text mimicking conversation content>
(or)
ASSESSMENT: <retrieval looks complete because...>
DONE"""

# Investigation 2D: "What baseline missed" prompt
WHAT_BASELINE_MISSED_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

IMPORTANT CONTEXT: The baseline search already found content matching the \
question's vocabulary. The segments above were retrieved by direct cosine \
similarity to the question. Your job is to find content the baseline \
CANNOT find -- content that uses DIFFERENT vocabulary but is still relevant.

Do NOT generate cues that paraphrase the question. The baseline already \
covers that semantic region. Instead:
1. Think about what RELATED topics would be discussed NEAR the answer
2. Think about what PREREQUISITE information would appear EARLIER in the \
conversation
3. Think about what CONSEQUENCES or FOLLOW-UP discussion would appear LATER
4. Use vocabulary that is DIFFERENT from the question but would appear in \
relevant conversation turns

Generate 2 search cues that target content the baseline MISSED.

Format:
ASSESSMENT: <what the baseline likely found vs what it missed>
CUE: <text using different vocabulary from the question>
CUE: <text using different vocabulary from the question>
Nothing else."""

# Listwise reranking prompt
LISTWISE_RERANK_PROMPT = """\
You are a relevance judge for a memory retrieval system. A user asked a \
question about a past conversation, and a retrieval system found the segments \
below. Your job: select the segments most relevant to ANSWERING the question.

QUESTION: {question}

SEGMENTS (numbered for reference):
{segments_text}

Select the {top_k} most relevant segments for answering this question. \
A segment is relevant if it contains information that directly helps answer \
the question -- names, dates, events, opinions, decisions, or context \
mentioned in the question.

Rank them from MOST relevant to LEAST relevant. Output ONLY the segment \
numbers, one per line, most relevant first. Output exactly {top_k} numbers.

Format:
RANK: <number>
RANK: <number>
...
Nothing else."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_segments(segments: list[Segment], max_items: int = 12,
                    max_chars: int = 250) -> str:
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def build_context_section(
    all_segments: list[Segment],
    previous_cues: list[str] | None = None,
) -> str:
    if not all_segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    context = format_segments(all_segments)
    context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context
    if previous_cues:
        context_section += (
            "\n\nPREVIOUS CUES ALREADY TRIED (do NOT repeat or paraphrase):\n"
            + "\n".join(f"- {c}" for c in previous_cues)
        )
    return context_section


def parse_cues(response: str) -> list[str]:
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


def parse_gaps(response: str) -> list[str]:
    gaps = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("GAP:"):
            gap = line[4:].strip()
            if gap:
                gaps.append(gap)
    return gaps


def parse_rank_response(response: str, num_segments: int) -> list[int]:
    indices = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("RANK:"):
            try:
                num = int(line[5:].strip())
                if 1 <= num <= num_segments:
                    indices.append(num - 1)
            except ValueError:
                continue
        else:
            try:
                num = int(line.strip().rstrip("."))
                if 1 <= num <= num_segments:
                    indices.append(num - 1)
            except (ValueError, IndexError):
                continue
    return indices


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 0.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class PrecisionEngine:
    """Engine for precision retrieval experiments."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 investigation: int = 0):
        self.store = store
        self.client = client or OpenAI(timeout=120.0)
        self.embedding_cache = PrecisionEmbeddingCache(investigation)
        self.llm_cache = PrecisionLLMCache(investigation)
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
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, embedding)
        self.embed_calls += 1
        return embedding

    def llm_call(self, prompt: str, model: str = MODEL,
                 max_tokens: int = 2000) -> str:
        cached = self.llm_cache.get(model, prompt)
        if cached is not None:
            self.llm_calls += 1
            return cached
        last_err = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                )
                text = response.choices[0].message.content or ""
                self.llm_cache.put(model, prompt, text)
                self.llm_calls += 1
                return text
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise last_err  # type: ignore[misc]

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def _retrieve(
        self,
        query_emb: np.ndarray,
        conversation_id: str,
        top_k: int = 10,
        exclude_indices: set[int] | None = None,
    ) -> list[Segment]:
        result = self.store.search(
            query_emb, top_k=top_k,
            conversation_id=conversation_id,
            exclude_indices=exclude_indices,
        )
        return list(result.segments)

    def _llm_rerank(
        self,
        question: str,
        segments: list[Segment],
        top_k: int = 20,
    ) -> list[Segment]:
        """Rerank segments using LLM listwise selection."""
        if not segments or len(segments) <= top_k:
            return segments

        effective_k = min(top_k, len(segments))
        batch_size = 40

        if len(segments) > batch_size:
            batches = []
            for i in range(0, len(segments), batch_size):
                batches.append(segments[i:i + batch_size])

            candidates = []
            for batch in batches:
                batch_top = self._rerank_single_batch(
                    question, batch, min(top_k, len(batch))
                )
                candidates.extend(batch_top)

            if len(candidates) > top_k:
                return self._rerank_single_batch(question, candidates, top_k)
            return candidates

        return self._rerank_single_batch(question, segments, effective_k)

    def _rerank_single_batch(
        self,
        question: str,
        segments: list[Segment],
        top_k: int,
    ) -> list[Segment]:
        lines = []
        for i, seg in enumerate(segments, 1):
            text = seg.text[:300]
            lines.append(f"[{i}] Turn {seg.turn_id}, {seg.role}: {text}")
        segments_text = "\n".join(lines)

        prompt = LISTWISE_RERANK_PROMPT.format(
            question=question,
            segments_text=segments_text,
            top_k=min(top_k, len(segments)),
        )

        response = self.llm_call(prompt, max_tokens=4000)
        ranked_indices = parse_rank_response(response, len(segments))

        reranked = []
        seen: set[int] = set()
        for idx in ranked_indices[:top_k]:
            if idx < len(segments) and idx not in seen:
                reranked.append(segments[idx])
                seen.add(idx)

        for i, seg in enumerate(segments):
            if i not in seen:
                reranked.append(seg)

        return reranked

    # ==================================================================
    # INVESTIGATION 1: Reranker Fairness
    # ==================================================================

    def cosine_top60_rerank(
        self, question: str, conversation_id: str,
    ) -> dict:
        """Cosine top-60 -> LLM rerank -> top-20.

        This is the baseline with reranking: retrieve 60 segments by cosine
        similarity, then use the LLM to select the best 20.
        0 LLM calls for retrieval, 1-2 LLM calls for reranking.
        """
        query_emb = self.embed_text(question)
        all_segments = self._retrieve(query_emb, conversation_id, top_k=60)
        reranked = self._llm_rerank(question, all_segments, top_k=20)
        # Return the reranked list but keep the full pool for r@50
        # For r@50, use reranked (top-20) + remaining from pool
        remaining = [s for s in all_segments if s not in reranked[:20]]
        final = reranked[:20] + remaining
        return {
            "segments": final,
            "metadata": {
                "name": "cosine_top60_rerank",
                "pool_size": len(all_segments),
            },
        }

    def v2f_30_rerank(
        self, question: str, conversation_id: str,
    ) -> dict:
        """V2f retrieval (~30 segments) -> LLM rerank -> top-20."""
        all_segments: list[Segment] = []
        exclude: set[int] = set()

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        # Stage 1: cosine top-10
        query_emb = self.embed_text(question)
        initial = self._retrieve(query_emb, conversation_id, top_k=10)
        add_segments(initial)

        # Stage 2: V2f cue generation
        context_section = build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = parse_cues(v2f_output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            new_segs = self._retrieve(cue_emb, conversation_id, top_k=10,
                                      exclude_indices=exclude)
            add_segments(new_segs)

        # Rerank the full pool -> top-20
        pool_size = len(all_segments)
        reranked = self._llm_rerank(question, all_segments, top_k=20)
        remaining = [s for s in all_segments
                     if s.index not in {r.index for r in reranked[:20]}]
        final = reranked[:20] + remaining

        return {
            "segments": final,
            "metadata": {
                "name": "v2f_30_rerank",
                "pool_size": pool_size,
                "cues": cues[:2],
            },
        }

    def v2f_gencheck_50_rerank(
        self, question: str, conversation_id: str,
    ) -> dict:
        """V2f + gencheck (~50 segments) -> LLM rerank -> top-20."""
        all_segments: list[Segment] = []
        exclude: set[int] = set()

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        # Stage 1: cosine top-10
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        initial = self._retrieve(query_emb, conversation_id, top_k=10)
        add_segments(initial)

        # Stage 2: V2f
        context_section = build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = parse_cues(v2f_output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            new_segs = self._retrieve(cue_emb, conversation_id, top_k=10,
                                      exclude_indices=exclude)
            add_segments(new_segs)

        # Stage 3: Gen-Check gap assessment
        formatted = format_segments(all_segments, max_items=16, max_chars=300)
        gap_prompt = GAP_ASSESSMENT_PROMPT.format(
            question=question, formatted_segments=formatted
        )
        gap_output = self.llm_call(gap_prompt)
        gaps = parse_gaps(gap_output)
        done = "DONE" in gap_output.upper().split("\n")[-1] if gap_output else True

        if not done and gaps:
            for gap in gaps[:2]:
                gap_emb = self.embed_text(gap)
                new_segs = self._retrieve(gap_emb, conversation_id, top_k=10,
                                          exclude_indices=exclude)
                add_segments(new_segs)

        # Negative-space probe (0 LLM cost)
        if all_segments:
            found_indices = [s.index for s in all_segments]
            found_embs = self.store.normalized_embeddings[found_indices]
            centroid = found_embs.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-10)
            alpha = 0.3
            residual = query_norm + alpha * (query_norm - centroid)
            residual /= max(np.linalg.norm(residual), 1e-10)
            ns_segs = self._retrieve(residual, conversation_id, top_k=10,
                                     exclude_indices=exclude)
            add_segments(ns_segs)
            self.embed_calls += 1

        # Rerank the full pool -> top-20
        pool_size = len(all_segments)
        reranked = self._llm_rerank(question, all_segments, top_k=20)
        remaining = [s for s in all_segments
                     if s.index not in {r.index for r in reranked[:20]}]
        final = reranked[:20] + remaining

        return {
            "segments": final,
            "metadata": {
                "name": "v2f_gencheck_50_rerank",
                "pool_size": pool_size,
                "cues": cues[:2],
                "gaps": gaps[:2],
                "gap_done": done,
            },
        }

    # ==================================================================
    # INVESTIGATION 2: Precision Cue Generation
    # ==================================================================

    def backfill(
        self, question: str, conversation_id: str,
    ) -> dict:
        """Backfill mode: baseline top-20 stays fixed, cue-found fills 21-50.

        At r@20: identical to baseline (can't hurt).
        At r@50: baseline + unique cue finds.
        """
        query_emb = self.embed_text(question)

        # Baseline top-50 (positions 1-50 of cosine search)
        baseline_50 = self._retrieve(query_emb, conversation_id, top_k=50)
        baseline_20 = baseline_50[:20]
        baseline_20_indices = {s.index for s in baseline_20}

        # V2f cue generation using baseline top-10 as context
        context_section = build_context_section(baseline_50[:10])
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = parse_cues(v2f_output)

        # Retrieve with cues, but DON'T exclude baseline segments yet
        cue_found: list[Segment] = []
        cue_found_indices: set[int] = set()
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            cue_segs = self._retrieve(cue_emb, conversation_id, top_k=10)
            for seg in cue_segs:
                if seg.index not in baseline_20_indices and seg.index not in cue_found_indices:
                    cue_found.append(seg)
                    cue_found_indices.add(seg.index)

        # Compute overlap metric for diagnostics
        cue_raw_all: list[Segment] = []
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            cue_segs = self._retrieve(cue_emb, conversation_id, top_k=10)
            cue_raw_all.extend(cue_segs)
        total_cue_retrieved = len(cue_raw_all)
        overlap_count = sum(1 for s in cue_raw_all if s.index in baseline_20_indices)
        overlap_pct = overlap_count / max(total_cue_retrieved, 1)

        # Build final list: baseline top-20 + cue-found (up to 30 more)
        final = list(baseline_20) + cue_found[:30]

        return {
            "segments": final,
            "metadata": {
                "name": "backfill",
                "cues": cues[:2],
                "cue_unique_found": len(cue_found),
                "cue_overlap_pct": round(overlap_pct, 3),
                "total_cue_retrieved": total_cue_retrieved,
            },
        }

    def embedding_filter(
        self, question: str, conversation_id: str,
        similarity_threshold: float = 0.7,
    ) -> dict:
        """Embedding distance filter: skip cues too similar to the question.

        Only use cues whose embedding is far enough from the question embedding.
        If cosine(cue, question) > threshold, skip -- it's a paraphrase.
        """
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)

        all_segments: list[Segment] = []
        exclude: set[int] = set()

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        # Stage 1: cosine top-10
        initial = self._retrieve(query_emb, conversation_id, top_k=10)
        add_segments(initial)

        # Stage 2: V2f
        context_section = build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = parse_cues(v2f_output)

        cue_details = []
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            cue_norm = cue_emb / max(np.linalg.norm(cue_emb), 1e-10)
            similarity = float(np.dot(query_norm, cue_norm))
            used = similarity <= similarity_threshold
            cue_details.append({
                "cue": cue,
                "similarity_to_question": round(similarity, 4),
                "used": used,
            })
            if used:
                new_segs = self._retrieve(cue_emb, conversation_id, top_k=10,
                                          exclude_indices=exclude)
                add_segments(new_segs)

        return {
            "segments": all_segments,
            "metadata": {
                "name": "embedding_filter",
                "cue_details": cue_details,
                "cues_used": sum(1 for d in cue_details if d["used"]),
                "cues_skipped": sum(1 for d in cue_details if not d["used"]),
            },
        }

    def cue_dedup_metric(
        self, question: str, conversation_id: str,
    ) -> dict:
        """Standard V2f with cue-found deduplication metric.

        Same as v2f retrieval, but tracks the overlap between cue-found
        segments and baseline top-K for diagnostic purposes.
        """
        query_emb = self.embed_text(question)

        # Baseline top-20
        baseline_result = self.store.search(
            query_emb, top_k=20, conversation_id=conversation_id
        )
        baseline_indices = {s.index for s in baseline_result.segments}
        baseline_turn_ids = {s.turn_id for s in baseline_result.segments}

        all_segments: list[Segment] = []
        exclude: set[int] = set()

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        # Stage 1: cosine top-10
        initial = self._retrieve(query_emb, conversation_id, top_k=10)
        add_segments(initial)

        # Stage 2: V2f
        context_section = build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = parse_cues(v2f_output)

        per_cue_overlap = []
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            # Retrieve WITHOUT exclusion to measure raw overlap
            raw_result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id
            )
            raw_indices = {s.index for s in raw_result.segments}
            overlap = len(raw_indices & baseline_indices)
            per_cue_overlap.append({
                "cue": cue,
                "raw_retrieved": len(raw_result.segments),
                "overlap_with_baseline_top20": overlap,
                "overlap_pct": round(overlap / max(len(raw_result.segments), 1), 3),
                "unique_new": len(raw_indices - baseline_indices),
            })

            # Now retrieve with exclusion for actual segment list
            new_segs = self._retrieve(cue_emb, conversation_id, top_k=10,
                                      exclude_indices=exclude)
            add_segments(new_segs)

        # Aggregate overlap
        total_raw = sum(d["raw_retrieved"] for d in per_cue_overlap)
        total_overlap = sum(d["overlap_with_baseline_top20"] for d in per_cue_overlap)
        total_unique = sum(d["unique_new"] for d in per_cue_overlap)

        return {
            "segments": all_segments,
            "metadata": {
                "name": "cue_dedup_metric",
                "per_cue_overlap": per_cue_overlap,
                "aggregate_overlap_pct": round(total_overlap / max(total_raw, 1), 3),
                "aggregate_unique_new": total_unique,
                "cues": [d["cue"] for d in per_cue_overlap],
            },
        }

    def what_baseline_missed(
        self, question: str, conversation_id: str,
    ) -> dict:
        """'What baseline missed' prompting: tell the LLM to find DIFFERENT
        vocabulary from the question."""
        query_emb = self.embed_text(question)

        all_segments: list[Segment] = []
        exclude: set[int] = set()

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        # Stage 1: cosine top-10
        initial = self._retrieve(query_emb, conversation_id, top_k=10)
        add_segments(initial)

        # Stage 2: "What baseline missed" prompt instead of V2f
        context_section = build_context_section(all_segments)
        prompt = WHAT_BASELINE_MISSED_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            new_segs = self._retrieve(cue_emb, conversation_id, top_k=10,
                                      exclude_indices=exclude)
            add_segments(new_segs)

        return {
            "segments": all_segments,
            "metadata": {
                "name": "what_baseline_missed",
                "cues": cues[:2],
            },
        }

    def backfill_what_missed(
        self, question: str, conversation_id: str,
    ) -> dict:
        """Combination: 'what baseline missed' prompt + backfill mode.

        Baseline top-20 stays fixed. Uses the 'what baseline missed' prompt
        to generate cues, then fills positions 21-50 with unique cue-found
        segments.
        """
        query_emb = self.embed_text(question)

        # Baseline top-50
        baseline_50 = self._retrieve(query_emb, conversation_id, top_k=50)
        baseline_20 = baseline_50[:20]
        baseline_20_indices = {s.index for s in baseline_20}

        # 'What baseline missed' cue generation
        context_section = build_context_section(baseline_50[:10])
        prompt = WHAT_BASELINE_MISSED_PROMPT.format(
            question=question, context_section=context_section
        )
        output = self.llm_call(prompt)
        cues = parse_cues(output)

        cue_found: list[Segment] = []
        cue_found_indices: set[int] = set()
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            cue_segs = self._retrieve(cue_emb, conversation_id, top_k=10)
            for seg in cue_segs:
                if seg.index not in baseline_20_indices and seg.index not in cue_found_indices:
                    cue_found.append(seg)
                    cue_found_indices.add(seg.index)

        final = list(baseline_20) + cue_found[:30]

        return {
            "segments": final,
            "metadata": {
                "name": "backfill_what_missed",
                "cues": cues[:2],
                "cue_unique_found": len(cue_found),
            },
        }

    def v2f_standard(
        self, question: str, conversation_id: str,
    ) -> dict:
        """Standard V2f (no reranking) for comparison baseline."""
        query_emb = self.embed_text(question)

        all_segments: list[Segment] = []
        exclude: set[int] = set()

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        initial = self._retrieve(query_emb, conversation_id, top_k=10)
        add_segments(initial)

        context_section = build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = parse_cues(v2f_output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            new_segs = self._retrieve(cue_emb, conversation_id, top_k=10,
                                      exclude_indices=exclude)
            add_segments(new_segs)

        return {
            "segments": all_segments,
            "metadata": {
                "name": "v2f_standard",
                "cues": cues[:2],
            },
        }


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    name: str
    npz: str
    questions_file: str
    max_questions: int | None = None
    benchmark_filter: str | None = None


DATASETS = [
    DatasetConfig(
        name="locomo_30q",
        npz="segments_extended.npz",
        questions_file="questions_extended.json",
        max_questions=30,
        benchmark_filter="locomo",
    ),
    DatasetConfig(
        name="synthetic_19q",
        npz="segments_synthetic.npz",
        questions_file="questions_synthetic.json",
    ),
    DatasetConfig(
        name="puzzle_16q",
        npz="segments_puzzle.npz",
        questions_file="questions_puzzle.json",
    ),
    DatasetConfig(
        name="advanced_23q",
        npz="segments_advanced.npz",
        questions_file="questions_advanced.json",
    ),
]


def load_questions(json_path: Path, benchmark_filter: str | None = None,
                   max_questions: int | None = None) -> list[dict]:
    with open(json_path) as f:
        questions = json.load(f)
    if benchmark_filter:
        questions = [q for q in questions if q.get("benchmark") == benchmark_filter]
    if max_questions:
        questions = questions[:max_questions]
    return questions


# ---------------------------------------------------------------------------
# Investigation 1: Reranker Fairness
# ---------------------------------------------------------------------------
RERANKER_APPROACHES = [
    "cosine_top60_rerank",
    "v2f_30_rerank",
    "v2f_gencheck_50_rerank",
]

# ---------------------------------------------------------------------------
# Investigation 2: Precision Cue Generation
# ---------------------------------------------------------------------------
PRECISION_APPROACHES = [
    "v2f_standard",       # baseline v2f for comparison
    "backfill",
    "embedding_filter",
    "cue_dedup_metric",
    "what_baseline_missed",
    "backfill_what_missed",
]


def run_investigation(
    investigation: int,
    verbose: bool = False,
) -> dict:
    """Run one investigation across all datasets."""

    if investigation == 1:
        approaches = RERANKER_APPROACHES
        label = "RERANKER FAIRNESS"
    else:
        approaches = PRECISION_APPROACHES
        label = "PRECISION CUE GENERATION"

    print(f"\n{'='*70}", flush=True)
    print(f"INVESTIGATION {investigation}: {label}", flush=True)
    print(f"{'='*70}", flush=True)

    all_dataset_results: dict[str, dict[str, list[dict]]] = {}

    for ds in DATASETS:
        print(f"\n{'='*60}", flush=True)
        print(f"DATASET: {ds.name}", flush=True)
        print(f"{'='*60}", flush=True)

        store = SegmentStore(DATA_DIR, ds.npz)
        questions = load_questions(
            DATA_DIR / ds.questions_file,
            benchmark_filter=ds.benchmark_filter,
            max_questions=ds.max_questions,
        )
        print(f"  Loaded {len(questions)} questions", flush=True)

        client = OpenAI(timeout=120.0)
        print(f"  Creating engine (loading caches)...", flush=True)
        engine = PrecisionEngine(store, client, investigation=investigation)
        print(f"  Engine ready. LLM cache: {len(engine.llm_cache._cache)} entries, "
              f"Embedding cache: {len(engine.embedding_cache._cache)} entries", flush=True)

        dataset_results: dict[str, list[dict]] = {a: [] for a in approaches}

        for qi, q in enumerate(questions):
            conv_id = q["conversation_id"]
            question_text = q["question"]
            source_ids = set(q["source_chat_ids"])
            category = q.get("category", "unknown")
            q_idx = q.get("question_index", qi)

            # Cosine baseline
            query_emb = engine.embed_text(question_text)
            baseline_20 = store.search(query_emb, top_k=20, conversation_id=conv_id)
            baseline_50 = store.search(query_emb, top_k=50, conversation_id=conv_id)
            baseline_recalls = {
                "r@20": compute_recall({s.turn_id for s in baseline_20.segments}, source_ids),
                "r@50": compute_recall({s.turn_id for s in baseline_50.segments}, source_ids),
            }

            for approach in approaches:
                engine.reset_counters()
                t0 = time.time()

                method = getattr(engine, approach)
                result = method(question_text, conv_id)

                elapsed = time.time() - t0
                segments = result["segments"]
                meta = result["metadata"]

                arch_recalls = {}
                for budget in BUDGETS:
                    retrieved_at_k = {s.turn_id for s in segments[:budget]}
                    arch_recalls[f"r@{budget}"] = compute_recall(retrieved_at_k, source_ids)

                result_entry = {
                    "conversation_id": conv_id,
                    "category": category,
                    "question_index": q_idx,
                    "question": question_text,
                    "source_chat_ids": sorted(source_ids),
                    "num_source_turns": len(source_ids),
                    "baseline_recalls": baseline_recalls,
                    "arch_recalls": arch_recalls,
                    "total_retrieved": len(segments),
                    "embed_calls": engine.embed_calls,
                    "llm_calls": engine.llm_calls,
                    "time_s": round(elapsed, 2),
                    "metadata": meta,
                }
                dataset_results[approach].append(result_entry)

                delta_20 = arch_recalls["r@20"] - baseline_recalls["r@20"]
                marker = "W" if delta_20 > 0.001 else ("L" if delta_20 < -0.001 else "T")

                if verbose:
                    print(f"  Q{q_idx:2d} [{category:25s}] {approach:25s}: "
                          f"r@20={arch_recalls['r@20']:.1%} "
                          f"r@50={arch_recalls['r@50']:.1%} "
                          f"(base r@20={baseline_recalls['r@20']:.1%} "
                          f"r@50={baseline_recalls['r@50']:.1%} "
                          f"delta20={delta_20:+.1%} {marker}) "
                          f"LLM={engine.llm_calls} emb={engine.embed_calls}",
                          flush=True)

            if not verbose:
                print(f"  Q{qi+1}/{len(questions)} done", flush=True)

            engine.save_caches()

        # Print summary
        print(f"\n--- {ds.name} Summary ---")
        for approach in approaches:
            results = dataset_results[approach]
            if not results:
                continue

            r20s = [r["arch_recalls"]["r@20"] for r in results]
            r50s = [r["arch_recalls"]["r@50"] for r in results]
            b20s = [r["baseline_recalls"]["r@20"] for r in results]
            b50s = [r["baseline_recalls"]["r@50"] for r in results]

            avg_r20 = sum(r20s) / len(r20s)
            avg_r50 = sum(r50s) / len(r50s)
            avg_b20 = sum(b20s) / len(b20s)
            avg_b50 = sum(b50s) / len(b50s)

            wins_20 = sum(1 for a, b in zip(r20s, b20s) if a > b + 0.001)
            ties_20 = sum(1 for a, b in zip(r20s, b20s) if abs(a - b) <= 0.001)
            losses_20 = len(r20s) - wins_20 - ties_20
            wins_50 = sum(1 for a, b in zip(r50s, b50s) if a > b + 0.001)
            ties_50 = sum(1 for a, b in zip(r50s, b50s) if abs(a - b) <= 0.001)
            losses_50 = len(r50s) - wins_50 - ties_50

            total_llm = sum(r["llm_calls"] for r in results)
            total_emb = sum(r["embed_calls"] for r in results)

            print(f"\n  {approach}:")
            print(f"    r@20: {avg_r20:.1%} (baseline {avg_b20:.1%}, "
                  f"delta {avg_r20 - avg_b20:+.1%}) "
                  f"W/T/L={wins_20}/{ties_20}/{losses_20}")
            print(f"    r@50: {avg_r50:.1%} (baseline {avg_b50:.1%}, "
                  f"delta {avg_r50 - avg_b50:+.1%}) "
                  f"W/T/L={wins_50}/{ties_50}/{losses_50}")
            print(f"    LLM calls: {total_llm} ({total_llm/len(results):.1f}/q), "
                  f"Embed calls: {total_emb} ({total_emb/len(results):.1f}/q)")

            # Cost efficiency: delta r@20 per LLM call
            delta_r20 = avg_r20 - avg_b20
            avg_llm = total_llm / len(results) if results else 0
            if avg_llm > 0:
                efficiency = delta_r20 / avg_llm
                print(f"    Cost efficiency: {delta_r20:+.1%} / {avg_llm:.1f} LLM = "
                      f"{efficiency*100:+.2f}pp per LLM call")

            # For cue_dedup_metric, print aggregate overlap
            if approach == "cue_dedup_metric":
                overlap_pcts = [
                    r["metadata"].get("aggregate_overlap_pct", 0)
                    for r in results
                ]
                avg_overlap = sum(overlap_pcts) / max(len(overlap_pcts), 1)
                unique_news = [
                    r["metadata"].get("aggregate_unique_new", 0)
                    for r in results
                ]
                avg_unique = sum(unique_news) / max(len(unique_news), 1)
                print(f"    Cue overlap with baseline: {avg_overlap:.1%} "
                      f"(avg unique new: {avg_unique:.1f})")

            # For embedding_filter, print skip rate
            if approach == "embedding_filter":
                skipped = [r["metadata"].get("cues_skipped", 0) for r in results]
                total_cues = [
                    r["metadata"].get("cues_used", 0) + r["metadata"].get("cues_skipped", 0)
                    for r in results
                ]
                total_skipped = sum(skipped)
                total_total = sum(total_cues)
                print(f"    Cue skip rate: {total_skipped}/{total_total} "
                      f"({total_skipped/max(total_total,1):.1%})")

            # For backfill, print overlap
            if approach in ("backfill", "backfill_what_missed"):
                unique_found = [r["metadata"].get("cue_unique_found", 0) for r in results]
                avg_unique = sum(unique_found) / max(len(unique_found), 1)
                print(f"    Avg unique cue-found segments: {avg_unique:.1f}")
                if approach == "backfill":
                    overlap_pcts = [r["metadata"].get("cue_overlap_pct", 0) for r in results]
                    avg_overlap = sum(overlap_pcts) / max(len(overlap_pcts), 1)
                    print(f"    Avg cue-found overlap with baseline: {avg_overlap:.1%}")

        # Save results
        inv_prefix = "reranker" if investigation == 1 else "precision"
        for approach in approaches:
            results = dataset_results[approach]
            outpath = RESULTS_DIR / f"precision_{inv_prefix}_{ds.name}_{approach}.json"
            with open(outpath, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved: {outpath.name}")

        all_dataset_results[ds.name] = dataset_results

    # Cross-dataset summary
    print_cross_dataset_summary(all_dataset_results, approaches, investigation)

    return all_dataset_results


def print_cross_dataset_summary(
    all_dataset_results: dict[str, dict[str, list[dict]]],
    approaches: list[str],
    investigation: int,
):
    """Print a comparison table across all datasets."""
    label = "RERANKER FAIRNESS" if investigation == 1 else "PRECISION CUE GENERATION"
    print(f"\n{'='*70}")
    print(f"CROSS-DATASET COMPARISON: {label}")
    print(f"{'='*70}")

    ds_names = list(all_dataset_results.keys())

    for metric_key in ["r@20", "r@50"]:
        print(f"\n  {metric_key}:")
        header = f"  {'Approach':28s}"
        for ds in ds_names:
            header += f"  {ds:>16s}"
        print(header)
        print("  " + "-" * (28 + 18 * len(ds_names)))

        # Baseline row
        row = f"  {'cosine_baseline':28s}"
        for ds in ds_names:
            results = list(all_dataset_results[ds].values())[0]
            avg = sum(r["baseline_recalls"][metric_key] for r in results) / len(results)
            row += f"  {avg:>16.1%}"
        print(row)

        # Approach rows
        for approach in approaches:
            row = f"  {approach:28s}"
            for ds in ds_names:
                results = all_dataset_results[ds].get(approach, [])
                if not results:
                    row += f"  {'N/A':>16s}"
                    continue
                avg = sum(r["arch_recalls"][metric_key] for r in results) / len(results)
                base = sum(r["baseline_recalls"][metric_key] for r in results) / len(results)
                delta = avg - base
                row += f"  {avg:.1%}({delta:+.1%})"
                # Pad to 16 chars
                cell = f"{avg:.1%}({delta:+.1%})"
                row_padding = 16 - len(cell)
                if row_padding > 0:
                    row = row[:-len(cell)] + " " * row_padding + cell
            print(row)

    # LLM calls per question
    print(f"\n  LLM calls/question:")
    header = f"  {'Approach':28s}"
    for ds in ds_names:
        header += f"  {ds:>16s}"
    print(header)
    print("  " + "-" * (28 + 18 * len(ds_names)))

    for approach in approaches:
        row = f"  {approach:28s}"
        for ds in ds_names:
            results = all_dataset_results[ds].get(approach, [])
            if not results:
                row += f"  {'N/A':>16s}"
                continue
            avg_llm = sum(r["llm_calls"] for r in results) / len(results)
            cell = f"{avg_llm:.1f}"
            row += f"  {cell:>16s}"
        print(row)

    # Cost efficiency: delta r@20 per LLM call
    print(f"\n  Cost efficiency (delta r@20 per LLM call, in pp):")
    header = f"  {'Approach':28s}"
    for ds in ds_names:
        header += f"  {ds:>16s}"
    print(header)
    print("  " + "-" * (28 + 18 * len(ds_names)))

    for approach in approaches:
        row = f"  {approach:28s}"
        for ds in ds_names:
            results = all_dataset_results[ds].get(approach, [])
            if not results:
                row += f"  {'N/A':>16s}"
                continue
            avg_r20 = sum(r["arch_recalls"]["r@20"] for r in results) / len(results)
            avg_b20 = sum(r["baseline_recalls"]["r@20"] for r in results) / len(results)
            avg_llm = sum(r["llm_calls"] for r in results) / len(results)
            delta = avg_r20 - avg_b20
            if avg_llm > 0:
                eff = (delta * 100) / avg_llm
                cell = f"{eff:+.2f}pp/call"
            else:
                cell = "N/A"
            row += f"  {cell:>16s}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    investigation_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == "--investigation" and i + 1 < len(sys.argv):
            investigation_arg = sys.argv[i + 1]

    investigations = []
    if investigation_arg == "1":
        investigations = [1]
    elif investigation_arg == "2":
        investigations = [2]
    else:
        investigations = [1, 2]

    all_results = {}
    for inv in investigations:
        all_results[inv] = run_investigation(inv, verbose=verbose)

    # Final combined summary if both ran
    if len(investigations) == 2:
        print(f"\n{'='*70}")
        print("FINAL COMBINED ANALYSIS")
        print(f"{'='*70}")

        for ds_name in DATASETS:
            ds = ds_name.name
            print(f"\n  {ds}:")
            # Get all approaches from both investigations
            inv1_results = all_results.get(1, {}).get(ds, {})
            inv2_results = all_results.get(2, {}).get(ds, {})

            all_approaches = {}
            all_approaches.update(inv1_results)
            all_approaches.update(inv2_results)

            if not all_approaches:
                continue

            # Get baseline from any approach
            sample = list(all_approaches.values())[0]
            avg_b20 = sum(r["baseline_recalls"]["r@20"] for r in sample) / len(sample)
            avg_b50 = sum(r["baseline_recalls"]["r@50"] for r in sample) / len(sample)
            print(f"    cosine baseline: r@20={avg_b20:.1%}, r@50={avg_b50:.1%}")

            for approach, results in sorted(all_approaches.items()):
                avg_r20 = sum(r["arch_recalls"]["r@20"] for r in results) / len(results)
                avg_r50 = sum(r["arch_recalls"]["r@50"] for r in results) / len(results)
                avg_llm = sum(r["llm_calls"] for r in results) / len(results)
                d20 = avg_r20 - avg_b20
                d50 = avg_r50 - avg_b50
                print(f"    {approach:28s}: r@20={avg_r20:.1%}({d20:+.1%}) "
                      f"r@50={avg_r50:.1%}({d50:+.1%}) "
                      f"LLM={avg_llm:.1f}/q")


if __name__ == "__main__":
    main()
