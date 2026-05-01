"""Full unified retrieval pipeline combining all winning components.

Components:
  1. V2f cue generation (self-monitoring + completeness hint + anti-question)
  2. Gen-Check gap assessment (skeptical implicit-need discovery)
  3. Constraint-type cues (item-type expansion for collection questions)
  4. LLM reranking (listwise selection from large pools)
  5. Negative-space probe (residual vector, zero LLM cost)

Pipelines:
  - full_pipeline: all 5 components
  - lite_pipeline: stages 1-2-3-5 only (skip constraint + reranking)

Usage:
    uv run python full_pipeline.py [--verbose]
"""

import json
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
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE_LLM = CACHE_DIR / "pipeline_llm_cache.json"
CACHE_FILE_EMB = CACHE_DIR / "pipeline_embedding_cache.json"
BUDGETS = [20, 50]


# ---------------------------------------------------------------------------
# Cache classes — reads from all existing caches, writes to pipeline-specific
# ---------------------------------------------------------------------------
class PipelineEmbeddingCache(EmbeddingCache):
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in (
            "embedding_cache.json",
            "arch_embedding_cache.json",
            "agent_embedding_cache.json",
            "frontier_embedding_cache.json",
            "meta_embedding_cache.json",
            "optim_embedding_cache.json",
            "synth_test_embedding_cache.json",
            "bestshot_embedding_cache.json",
            "task_exec_embedding_cache.json",
            "general_embedding_cache.json",
            "adaptive_embedding_cache.json",
            "fulleval_embedding_cache.json",
            "constraint_embedding_cache.json",
            "hybrid_embedding_cache.json",
            "resume_embedding_cache.json",
            "pipeline_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = CACHE_FILE_EMB
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


class PipelineLLMCache(LLMCache):
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in (
            "llm_cache.json",
            "arch_llm_cache.json",
            "agent_llm_cache.json",
            "tree_llm_cache.json",
            "frontier_llm_cache.json",
            "meta_llm_cache.json",
            "optim_llm_cache.json",
            "synth_test_llm_cache.json",
            "bestshot_llm_cache.json",
            "task_exec_llm_cache.json",
            "general_llm_cache.json",
            "adaptive_llm_cache.json",
            "fulleval_llm_cache.json",
            "constraint_llm_cache.json",
            "hybrid_llm_cache.json",
            "rerank_llm_cache.json",
            "resume_llm_cache.json",
            "pipeline_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = CACHE_FILE_LLM
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

# Stage 2: V2f cue generation (proven best for r@20)
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

# Stage 3: Gen-Check gap assessment (skeptical, discovers implicit needs)
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

# Stage 4: Constraint-type expansion
CONSTRAINT_TYPE_PROMPT = """\
You are retrieving information from a conversation. The question asks about \
ALL items, constraints, or requirements that were discussed.

Question: {question}

RETRIEVED SO FAR:
{context}

In conversations, items of the same general category come in many TYPES. \
For each type that might exist but ISN'T covered in the retrieved content, \
generate a search cue. The cue should sound like something someone would \
actually say in conversation when mentioning that type of item.

Focus on item types NOT YET found. Generate cues for missing types.

Format:
ASSESSMENT: <which types are covered vs missing>
CUE: <text mimicking conversation content about a missing type>
CUE: <text>
CUE: <text>
Nothing else."""

# Stage 6: LLM reranking (listwise selection)
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
def format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
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
    """Parse RANK: lines into 0-indexed segment indices."""
    indices = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("RANK:"):
            try:
                num = int(line[5:].strip())
                if 1 <= num <= num_segments:
                    indices.append(num - 1)  # 0-indexed
            except ValueError:
                continue
    return indices


COLLECTION_KEYWORDS = {"all", "every", "list", "complete", "each", "total", "how many"}


def implies_collection(question: str) -> bool:
    """Check if question implies a collection/completeness need."""
    words = set(question.lower().split())
    return bool(words & COLLECTION_KEYWORDS)


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 0.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


# ---------------------------------------------------------------------------
# Pipeline engine
# ---------------------------------------------------------------------------
class PipelineEngine:
    """Unified retrieval pipeline combining all winning components."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=120.0)
        self.embedding_cache = PipelineEmbeddingCache()
        self.llm_cache = PipelineLLMCache()
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

    def llm_call(self, prompt: str, model: str = MODEL, max_tokens: int = 2000) -> str:
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
                time.sleep(2**attempt)
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
            query_emb,
            top_k=top_k,
            conversation_id=conversation_id,
            exclude_indices=exclude_indices,
        )
        return list(result.segments)

    # ------------------------------------------------------------------
    # Full pipeline: all 5 stages + reranking
    # ------------------------------------------------------------------
    def full_pipeline(
        self,
        question: str,
        conversation_id: str,
    ) -> dict:
        """Full pipeline: V2f + GenCheck + constraint-type + neg-space + rerank."""
        all_segments: list[Segment] = []
        exclude: set[int] = set()
        all_cues: list[str] = []
        metadata: dict = {"name": "full_pipeline"}

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        # Stage 1: Initial retrieval (1 embed call)
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        initial = self._retrieve(query_emb, conversation_id, top_k=10)
        add_segments(initial)

        # Stage 2: V2f cue generation (1 LLM + 2 embed calls)
        context_section = build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = parse_cues(v2f_output)
        all_cues.extend(cues[:2])

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            new_segs = self._retrieve(
                cue_emb, conversation_id, top_k=10, exclude_indices=exclude
            )
            add_segments(new_segs)

        # Stage 3: Gen-Check gap assessment (1 LLM + 1-2 embed calls)
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
                new_segs = self._retrieve(
                    gap_emb, conversation_id, top_k=10, exclude_indices=exclude
                )
                add_segments(new_segs)
                all_cues.append(gap)

        metadata["v2f_cues"] = cues[:2]
        metadata["gaps"] = gaps[:2]
        metadata["gap_done"] = done

        # Stage 4: Constraint-type expansion (conditional)
        constraint_cues: list[str] = []
        if implies_collection(question):
            context = format_segments(all_segments, max_items=14, max_chars=250)
            constraint_prompt = CONSTRAINT_TYPE_PROMPT.format(
                question=question, context=context
            )
            constraint_output = self.llm_call(constraint_prompt)
            constraint_cues = parse_cues(constraint_output)
            for cue in constraint_cues[:3]:
                cue_emb = self.embed_text(cue)
                new_segs = self._retrieve(
                    cue_emb, conversation_id, top_k=10, exclude_indices=exclude
                )
                add_segments(new_segs)
                all_cues.append(cue)

        metadata["constraint_cues"] = constraint_cues[:3]
        metadata["collection_triggered"] = implies_collection(question)

        # Stage 5: Negative-space probe (0 LLM, 1 embed call)
        if all_segments:
            found_indices = [s.index for s in all_segments]
            found_embs = self.store.normalized_embeddings[found_indices]
            centroid = found_embs.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-10)

            alpha = 0.3
            residual = query_norm + alpha * (query_norm - centroid)
            residual /= max(np.linalg.norm(residual), 1e-10)

            # Negative-space uses the residual vector directly (no embed call)
            ns_segs = self._retrieve(
                residual, conversation_id, top_k=10, exclude_indices=exclude
            )
            ns_count = add_segments(ns_segs)
            metadata["ns_new_count"] = ns_count
            # Count as 1 embed call even though it's computed, for budget tracking
            self.embed_calls += 1

        # Stage 6: LLM reranking (only if pool > 20)
        metadata["pool_size_before_rerank"] = len(all_segments)
        if len(all_segments) > 20:
            reranked = self._llm_rerank(question, all_segments, top_k=20)
            metadata["reranked"] = True
            all_segments = reranked
        else:
            metadata["reranked"] = False

        metadata["cues"] = all_cues
        return {
            "segments": all_segments,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Lite pipeline: V2f + GenCheck + neg-space (no constraint, no rerank)
    # ------------------------------------------------------------------
    def lite_pipeline(
        self,
        question: str,
        conversation_id: str,
    ) -> dict:
        """Lite pipeline: V2f + GenCheck + neg-space. 2 LLM + ~5 embed."""
        all_segments: list[Segment] = []
        exclude: set[int] = set()
        all_cues: list[str] = []
        metadata: dict = {"name": "lite_pipeline"}

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        # Stage 1: Initial retrieval
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        initial = self._retrieve(query_emb, conversation_id, top_k=10)
        add_segments(initial)

        # Stage 2: V2f cue generation
        context_section = build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = parse_cues(v2f_output)
        all_cues.extend(cues[:2])

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            new_segs = self._retrieve(
                cue_emb, conversation_id, top_k=10, exclude_indices=exclude
            )
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
                new_segs = self._retrieve(
                    gap_emb, conversation_id, top_k=10, exclude_indices=exclude
                )
                add_segments(new_segs)
                all_cues.append(gap)

        metadata["v2f_cues"] = cues[:2]
        metadata["gaps"] = gaps[:2]
        metadata["gap_done"] = done

        # Stage 5: Negative-space probe
        if all_segments:
            found_indices = [s.index for s in all_segments]
            found_embs = self.store.normalized_embeddings[found_indices]
            centroid = found_embs.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-10)

            alpha = 0.3
            residual = query_norm + alpha * (query_norm - centroid)
            residual /= max(np.linalg.norm(residual), 1e-10)

            ns_segs = self._retrieve(
                residual, conversation_id, top_k=10, exclude_indices=exclude
            )
            ns_count = add_segments(ns_segs)
            metadata["ns_new_count"] = ns_count
            self.embed_calls += 1

        metadata["cues"] = all_cues
        return {
            "segments": all_segments,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # V2f-only baseline for comparison
    # ------------------------------------------------------------------
    def v2f_only(
        self,
        question: str,
        conversation_id: str,
    ) -> dict:
        """V2f only: 1 LLM + 3 embed calls."""
        all_segments: list[Segment] = []
        exclude: set[int] = set()
        metadata: dict = {"name": "v2f_only"}

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        query_emb = self.embed_text(question)
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
            new_segs = self._retrieve(
                cue_emb, conversation_id, top_k=10, exclude_indices=exclude
            )
            add_segments(new_segs)

        metadata["cues"] = cues[:2]
        return {
            "segments": all_segments,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # v15 control baseline for comparison
    # ------------------------------------------------------------------
    def v15_control(
        self,
        question: str,
        conversation_id: str,
    ) -> dict:
        """v15 control: same as v2f but with original v15 prompt."""
        V15_PROMPT = """\
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

        all_segments: list[Segment] = []
        exclude: set[int] = set()
        metadata: dict = {"name": "v15_control"}

        def add_segments(new_segs: list[Segment]) -> int:
            count = 0
            for seg in new_segs:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)
                    count += 1
            return count

        query_emb = self.embed_text(question)
        initial = self._retrieve(query_emb, conversation_id, top_k=10)
        add_segments(initial)

        context_section = build_context_section(all_segments)
        prompt = V15_PROMPT.format(question=question, context_section=context_section)
        output = self.llm_call(prompt)
        cues = parse_cues(output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            new_segs = self._retrieve(
                cue_emb, conversation_id, top_k=10, exclude_indices=exclude
            )
            add_segments(new_segs)

        metadata["cues"] = cues[:2]
        return {
            "segments": all_segments,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # LLM reranker (listwise)
    # ------------------------------------------------------------------
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

        # Format segments for prompt
        lines = []
        for i, seg in enumerate(segments, 1):
            text = seg.text[:300]
            lines.append(f"[{i}] Turn {seg.turn_id}, {seg.role}: {text}")
        segments_text = "\n".join(lines)

        # If too many segments for one batch, split into batches of 40
        batch_size = 40
        if len(segments) > batch_size:
            # Multi-batch: rerank each batch, then merge-rerank
            batches = []
            for i in range(0, len(segments), batch_size):
                batches.append(segments[i : i + batch_size])

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

        # Append remaining in original order
        for i, seg in enumerate(segments):
            if i not in seen:
                reranked.append(seg)

        return reranked


# ---------------------------------------------------------------------------
# Evaluation harness
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

PIPELINES = ["full_pipeline", "lite_pipeline", "v2f_only", "v15_control"]


def load_questions(
    json_path: Path,
    benchmark_filter: str | None = None,
    max_questions: int | None = None,
) -> list[dict]:
    with open(json_path) as f:
        questions = json.load(f)
    if benchmark_filter:
        questions = [q for q in questions if q.get("benchmark") == benchmark_filter]
    if max_questions:
        questions = questions[:max_questions]
    return questions


def evaluate_dataset(
    ds: DatasetConfig,
    verbose: bool = False,
) -> dict:
    """Run all 4 pipelines on one dataset, return results."""
    print(f"\n{'=' * 70}")
    print(f"DATASET: {ds.name}")
    print(f"{'=' * 70}")

    store = SegmentStore(DATA_DIR, ds.npz)
    questions = load_questions(
        DATA_DIR / ds.questions_file,
        benchmark_filter=ds.benchmark_filter,
        max_questions=ds.max_questions,
    )
    print(f"  Loaded {len(questions)} questions")

    client = OpenAI(timeout=120.0)
    engine = PipelineEngine(store, client)

    all_results: dict[str, list[dict]] = {p: [] for p in PIPELINES}

    for qi, q in enumerate(questions):
        conv_id = q["conversation_id"]
        question_text = q["question"]
        source_ids = set(q["source_chat_ids"])
        category = q.get("category", "unknown")
        q_idx = q.get("question_index", qi)

        # Cosine baseline (r@20, r@50)
        query_emb = engine.embed_text(question_text)
        baseline_20 = store.search(query_emb, top_k=20, conversation_id=conv_id)
        baseline_50 = store.search(query_emb, top_k=50, conversation_id=conv_id)
        baseline_recalls = {
            "r@20": compute_recall(
                {s.turn_id for s in baseline_20.segments}, source_ids
            ),
            "r@50": compute_recall(
                {s.turn_id for s in baseline_50.segments}, source_ids
            ),
        }

        for pipeline_name in PIPELINES:
            engine.reset_counters()
            t0 = time.time()

            method = getattr(engine, pipeline_name)
            result = method(question_text, conv_id)

            elapsed = time.time() - t0
            segments = result["segments"]
            meta = result["metadata"]

            # Compute recalls at budgets
            retrieved_ids_all = {s.turn_id for s in segments}
            arch_recalls = {}
            for budget in BUDGETS:
                retrieved_at_k = {s.turn_id for s in segments[:budget]}
                arch_recalls[f"r@{budget}"] = compute_recall(retrieved_at_k, source_ids)
            arch_recalls["r@actual"] = compute_recall(retrieved_ids_all, source_ids)

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
            all_results[pipeline_name].append(result_entry)

            # W/T/L vs baseline at r@20
            delta_20 = arch_recalls["r@20"] - baseline_recalls["r@20"]
            marker = "W" if delta_20 > 0.001 else ("L" if delta_20 < -0.001 else "T")

            if verbose:
                print(
                    f"  Q{q_idx:2d} [{category:25s}] {pipeline_name:15s}: "
                    f"r@20={arch_recalls['r@20']:.1%} "
                    f"(baseline={baseline_recalls['r@20']:.1%}, "
                    f"delta={delta_20:+.1%} {marker}) "
                    f"pool={len(segments)} "
                    f"LLM={engine.llm_calls} emb={engine.embed_calls}"
                )

        engine.save_caches()

    # Print summary for this dataset
    print(f"\n--- {ds.name} Summary ---")
    for pipeline_name in PIPELINES:
        results = all_results[pipeline_name]
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

        # W/T/L
        wins_20, ties_20, losses_20 = 0, 0, 0
        wins_50, ties_50, losses_50 = 0, 0, 0
        for r in results:
            d20 = r["arch_recalls"]["r@20"] - r["baseline_recalls"]["r@20"]
            d50 = r["arch_recalls"]["r@50"] - r["baseline_recalls"]["r@50"]
            if d20 > 0.001:
                wins_20 += 1
            elif d20 < -0.001:
                losses_20 += 1
            else:
                ties_20 += 1
            if d50 > 0.001:
                wins_50 += 1
            elif d50 < -0.001:
                losses_50 += 1
            else:
                ties_50 += 1

        total_llm = sum(r["llm_calls"] for r in results)
        total_emb = sum(r["embed_calls"] for r in results)

        print(f"\n  {pipeline_name}:")
        print(
            f"    r@20: {avg_r20:.1%} (baseline {avg_b20:.1%}, "
            f"delta {avg_r20 - avg_b20:+.1%}) "
            f"W/T/L={wins_20}/{ties_20}/{losses_20}"
        )
        print(
            f"    r@50: {avg_r50:.1%} (baseline {avg_b50:.1%}, "
            f"delta {avg_r50 - avg_b50:+.1%}) "
            f"W/T/L={wins_50}/{ties_50}/{losses_50}"
        )
        print(
            f"    LLM calls: {total_llm} ({total_llm / len(results):.1f}/q), "
            f"Embed calls: {total_emb} ({total_emb / len(results):.1f}/q)"
        )

        # Per-category breakdown at r@20
        cat_results: dict[str, list[float]] = defaultdict(list)
        cat_baselines: dict[str, list[float]] = defaultdict(list)
        for r in results:
            cat_results[r["category"]].append(r["arch_recalls"]["r@20"])
            cat_baselines[r["category"]].append(r["baseline_recalls"]["r@20"])
        print("    Per-category r@20:")
        for cat in sorted(cat_results.keys()):
            cat_avg = sum(cat_results[cat]) / len(cat_results[cat])
            cat_base = sum(cat_baselines[cat]) / len(cat_baselines[cat])
            print(
                f"      {cat:30s}: {cat_avg:.1%} (base {cat_base:.1%}, "
                f"delta {cat_avg - cat_base:+.1%}, n={len(cat_results[cat])})"
            )

    # Save results
    for pipeline_name in PIPELINES:
        results = all_results[pipeline_name]
        # Strip non-serializable metadata
        for r in results:
            # Ensure metadata is JSON-serializable
            meta = r.get("metadata", {})
            for k, v in list(meta.items()):
                if isinstance(v, (np.integer, np.floating)):
                    meta[k] = int(v) if isinstance(v, np.integer) else float(v)

        outpath = RESULTS_DIR / f"pipeline_{ds.name}_{pipeline_name}.json"
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved: {outpath.name}")

    return all_results


# ---------------------------------------------------------------------------
# Cross-dataset comparison
# ---------------------------------------------------------------------------
def print_cross_dataset_summary(all_dataset_results: dict[str, dict[str, list[dict]]]):
    """Print a comparison table across all datasets."""
    print(f"\n{'=' * 70}")
    print("CROSS-DATASET COMPARISON")
    print(f"{'=' * 70}")

    # Header
    ds_names = list(all_dataset_results.keys())
    print(f"\n{'Pipeline':20s}", end="")
    for ds in ds_names:
        print(f"  {ds:>18s}", end="")
    print()
    print("-" * (20 + 20 * len(ds_names)))

    for metric_name, metric_key in [("r@20", "r@20"), ("r@50", "r@50")]:
        print(f"\n  {metric_name}:")
        print(f"  {'':18s}", end="")
        for ds in ds_names:
            print(f"  {'arch':>8s} {'delta':>8s}", end="")
        print()

        # Baseline row
        print(f"  {'cosine_baseline':18s}", end="")
        for ds in ds_names:
            results = list(all_dataset_results[ds].values())[0]  # any pipeline
            avg = sum(r["baseline_recalls"][metric_key] for r in results) / len(results)
            print(f"  {avg:8.1%} {'--':>8s}", end="")
        print()

        # Pipeline rows
        for pipeline_name in PIPELINES:
            print(f"  {pipeline_name:18s}", end="")
            for ds in ds_names:
                results = all_dataset_results[ds].get(pipeline_name, [])
                if not results:
                    print(f"  {'N/A':>8s} {'N/A':>8s}", end="")
                    continue
                avg = sum(r["arch_recalls"][metric_key] for r in results) / len(results)
                base = sum(r["baseline_recalls"][metric_key] for r in results) / len(
                    results
                )
                delta = avg - base
                print(f"  {avg:8.1%} {delta:+8.1%}", end="")
            print()

    # W/T/L summary
    print("\n  W/T/L at r@20:")
    for pipeline_name in PIPELINES:
        print(f"  {pipeline_name:18s}", end="")
        for ds in ds_names:
            results = all_dataset_results[ds].get(pipeline_name, [])
            if not results:
                print(f"  {'N/A':>18s}", end="")
                continue
            w, t, l = 0, 0, 0
            for r in results:
                d = r["arch_recalls"]["r@20"] - r["baseline_recalls"]["r@20"]
                if d > 0.001:
                    w += 1
                elif d < -0.001:
                    l += 1
                else:
                    t += 1
            print(f"  {w:2d}W/{t:2d}T/{l:2d}L       ", end="")
        print()

    # Head-to-head: full vs lite
    print("\n  Full vs Lite (r@20):")
    for ds in ds_names:
        full_results = all_dataset_results[ds].get("full_pipeline", [])
        lite_results = all_dataset_results[ds].get("lite_pipeline", [])
        if not full_results or not lite_results:
            continue
        w, t, l = 0, 0, 0
        for fr, lr in zip(full_results, lite_results):
            fd = fr["arch_recalls"]["r@20"]
            ld = lr["arch_recalls"]["r@20"]
            diff = fd - ld
            if diff > 0.001:
                w += 1
            elif diff < -0.001:
                l += 1
            else:
                t += 1
        avg_full = sum(r["arch_recalls"]["r@20"] for r in full_results) / len(
            full_results
        )
        avg_lite = sum(r["arch_recalls"]["r@20"] for r in lite_results) / len(
            lite_results
        )
        print(
            f"    {ds}: full={avg_full:.1%} lite={avg_lite:.1%} "
            f"full_wins={w} ties={t} lite_wins={l}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    all_dataset_results: dict[str, dict[str, list[dict]]] = {}

    for ds in DATASETS:
        try:
            results = evaluate_dataset(ds, verbose=verbose)
            all_dataset_results[ds.name] = results
        except Exception as e:
            print(f"\nERROR on {ds.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if len(all_dataset_results) > 1:
        print_cross_dataset_summary(all_dataset_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
