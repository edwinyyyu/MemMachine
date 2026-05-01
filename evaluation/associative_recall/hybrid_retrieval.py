"""Hybrid retrieval architectures and daydream exploration.

Part 1: Hybrid architecture combining v2f cue generation with Gen-Check
        gap assessment. Flow:
          1. Initial retrieval with raw query (1 embed call)
          2. V2f-style cue generation (1 LLM call) -> retrieve cues (2 embed)
          3. Gen-Check assessment: what gaps remain? (1 LLM call)
          4. If gaps, retrieve for each gap (1-2 embed calls)
        Total: ~2 LLM calls, 5-6 embed calls.

Part 2: Daydream exploration step (add-on to v2f):
  a) LLM daydream: "What loosely related topic might also be relevant?"
     (1 short LLM call + 1 embed call)
  b) Negative-space daydream: subtract centroid of found segments from query,
     retrieve with residual vector. Zero LLM cost.

Usage:
    uv run python hybrid_retrieval.py --arch <name> [--verbose]
    uv run python hybrid_retrieval.py --all [--verbose]
    uv run python hybrid_retrieval.py --list
"""

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
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50, 100]


# ---------------------------------------------------------------------------
# Cache classes -- hybrid-specific cache files
# ---------------------------------------------------------------------------
class HybridEmbeddingCache(EmbeddingCache):
    """Reads all existing caches, writes to hybrid_embedding_cache.json."""

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
            "general_embedding_cache.json",
            "synth_test_embedding_cache.json",
            "task_exec_embedding_cache.json",
            "bestshot_embedding_cache.json",
            "hybrid_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "hybrid_embedding_cache.json"
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


class HybridLLMCache(LLMCache):
    """Reads all existing caches, writes to hybrid_llm_cache.json."""

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
            "general_llm_cache.json",
            "synth_test_llm_cache.json",
            "task_exec_llm_cache.json",
            "bestshot_llm_cache.json",
            "hybrid_llm_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if v:
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "hybrid_llm_cache.json"
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
# Base class
# ---------------------------------------------------------------------------
class HybridBase:
    """Base class with embedding/LLM utilities and counters."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=120.0)
        self.embedding_cache = HybridEmbeddingCache()
        self.llm_cache = HybridLLMCache()
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
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class HybridResult:
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


def _format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    """Format segments chronologically."""
    if not segments:
        return "(no content retrieved yet)"
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


def _parse_cues(response: str) -> list[str]:
    """Parse CUE: lines from LLM response."""
    cues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CUE:"):
            cue = line[4:].strip()
            if cue:
                cues.append(cue)
    return cues


def _parse_gaps(response: str) -> list[str]:
    """Parse GAP: lines from LLM response."""
    gaps = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("GAP:"):
            gap = line[4:].strip()
            if gap:
                gaps.append(gap)
    return gaps


def _build_context_section(
    all_segments: list[Segment],
    new_segments: list[Segment] | None = None,
    previous_cues: list[str] | None = None,
) -> str:
    """Build context section matching v15 format."""
    if not all_segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    context = _format_segments(all_segments)
    context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + context
    if new_segments:
        latest_lines = []
        for seg in sorted(new_segments, key=lambda s: s.turn_id)[:6]:
            latest_lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:200]}")
        context_section += "\n\nMOST RECENTLY FOUND (last hop):\n" + "\n".join(
            latest_lines
        )
    if previous_cues:
        context_section += (
            "\n\nPREVIOUS CUES ALREADY TRIED (do NOT repeat or paraphrase):\n"
            + "\n".join(f"- {c}" for c in previous_cues)
        )
    return context_section


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# V2f cue generation prompt (proven best for r@20)
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

# Gen-Check gap assessment prompt (adapted from task_execution.py skeptical style)
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

# Daydream prompt (tangential exploration, short)
DAYDREAM_PROMPT = """\
Question: {question}

Already retrieved topics: {topic_summary}

What loosely related topic might also be relevant to this question that \
hasn't been searched for? Think of a tangential connection -- something \
not obviously related but potentially useful. Write one short search cue \
(1-2 sentences) that sounds like conversation content about that topic.

CUE: """


# ===========================================================================
# Architecture 1: V2f baseline (reference)
# ===========================================================================
class V2fBaseline(HybridBase):
    """Standard v2f: initial retrieval + 1 LLM call for cues."""

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        # Hop 0: embed question, retrieve top-10
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Single LLM call for cues
        context_section = _build_context_section(all_segments)
        prompt = V2F_PROMPT.format(question=question, context_section=context_section)
        output = self.llm_call(prompt)
        cues = _parse_cues(output)

        # Retrieve with cues
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    exclude_indices.add(seg.index)

        return HybridResult(
            segments=all_segments,
            metadata={"name": "v2f_baseline", "cues": cues[:2]},
        )


# ===========================================================================
# Architecture 2: Hybrid (v2f + Gen-Check gap assessment)
# ===========================================================================
class HybridV2fGenCheck(HybridBase):
    """Combines v2f cue generation with Gen-Check gap assessment.

    Flow:
      1. Initial retrieval with raw query (1 embed)
      2. V2f cue generation (1 LLM) -> retrieve 2 cues (2 embed)
      3. Gen-Check assessment of gaps (1 LLM)
      4. If gaps found, retrieve for each (1-2 embed)
    Total: 2 LLM calls, 5-6 embed calls.
    """

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        all_cues: list[str] = []
        all_gaps: list[str] = []

        # Step 1: Initial retrieval
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Step 2: V2f cue generation (1 LLM call)
        context_section = _build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = _parse_cues(v2f_output)
        all_cues.extend(cues[:2])

        # Retrieve for v2f cues
        v2f_new_segments: list[Segment] = []
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    v2f_new_segments.append(seg)
                    exclude_indices.add(seg.index)

        # Step 3: Gen-Check gap assessment (1 LLM call)
        formatted = _format_segments(all_segments, max_items=16, max_chars=300)
        gap_prompt = GAP_ASSESSMENT_PROMPT.format(
            question=question, formatted_segments=formatted
        )
        gap_output = self.llm_call(gap_prompt)
        gaps = _parse_gaps(gap_output)
        all_gaps.extend(gaps[:2])

        # Step 4: Retrieve for gaps (if any)
        gap_new_segments: list[Segment] = []
        done = "DONE" in gap_output.upper().split("\n")[-1] if gap_output else True
        if not done and gaps:
            for gap in gaps[:2]:
                gap_emb = self.embed_text(gap)
                result = self.store.search(
                    gap_emb,
                    top_k=10,
                    conversation_id=conversation_id,
                    exclude_indices=exclude_indices,
                )
                for seg in result.segments:
                    if seg.index not in exclude_indices:
                        all_segments.append(seg)
                        gap_new_segments.append(seg)
                        exclude_indices.add(seg.index)

        return HybridResult(
            segments=all_segments,
            metadata={
                "name": "hybrid_v2f_gencheck",
                "v2f_cues": all_cues,
                "v2f_output": v2f_output[:500],
                "gap_assessment": gap_output[:500],
                "gaps": all_gaps,
                "gap_done": done,
                "v2f_new_count": len(v2f_new_segments),
                "gap_new_count": len(gap_new_segments),
                "cues": all_cues + all_gaps,
            },
        )


# ===========================================================================
# Architecture 3: V2f + LLM Daydream
# ===========================================================================
class V2fWithDaydream(HybridBase):
    """V2f baseline + one tangential exploration step via LLM.

    After v2f retrieval, ask the LLM (short call, ~100 tokens):
    "What loosely related topic might also be relevant?"
    Use response as one additional retrieval query.
    """

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        # Step 1-2: Standard v2f
        query_emb = self.embed_text(question)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        context_section = _build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = _parse_cues(v2f_output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    exclude_indices.add(seg.index)

        # Step 3: Daydream -- tangential exploration (1 short LLM call)
        # Summarize retrieved topics for the daydream prompt
        topic_summary = "; ".join(
            seg.text[:80] for seg in sorted(all_segments[:8], key=lambda s: s.turn_id)
        )
        daydream_prompt = DAYDREAM_PROMPT.format(
            question=question, topic_summary=topic_summary
        )
        daydream_output = self.llm_call(daydream_prompt, max_tokens=150)

        # Parse the daydream cue -- take everything after "CUE:" if present,
        # otherwise use the whole response
        daydream_cue = daydream_output.strip()
        if "CUE:" in daydream_cue:
            daydream_cue = daydream_cue.split("CUE:", 1)[1].strip()
        # Take first line only
        daydream_cue = daydream_cue.split("\n")[0].strip()

        # Step 4: Retrieve with daydream cue
        daydream_new: list[Segment] = []
        if daydream_cue:
            dd_emb = self.embed_text(daydream_cue)
            result = self.store.search(
                dd_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    daydream_new.append(seg)
                    exclude_indices.add(seg.index)

        return HybridResult(
            segments=all_segments,
            metadata={
                "name": "v2f_daydream_llm",
                "v2f_cues": cues[:2],
                "daydream_cue": daydream_cue,
                "daydream_new_count": len(daydream_new),
                "cues": cues[:2] + [daydream_cue] if daydream_cue else cues[:2],
            },
        )


# ===========================================================================
# Architecture 4: V2f + Negative-Space Daydream (zero LLM cost)
# ===========================================================================
class V2fWithNegativeSpace(HybridBase):
    """V2f baseline + negative-space exploration (zero extra LLM cost).

    After v2f retrieval, compute the residual vector:
      residual = query - centroid(found segments)
    This points toward unexplored territory. Retrieve with the residual.
    """

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        # Step 1-2: Standard v2f
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)

        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        context_section = _build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = _parse_cues(v2f_output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    exclude_indices.add(seg.index)

        # Step 3: Negative-space daydream
        # Compute centroid of all found segments
        found_indices = [s.index for s in all_segments]
        found_embs = self.store.normalized_embeddings[found_indices]
        centroid = found_embs.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-10)

        # Residual: push query away from what we already found
        # residual = query + alpha * (query - centroid)
        alpha = 0.3
        residual = query_norm + alpha * (query_norm - centroid)
        residual /= max(np.linalg.norm(residual), 1e-10)

        # Retrieve with residual vector
        ns_new: list[Segment] = []
        result = self.store.search(
            residual,
            top_k=10,
            conversation_id=conversation_id,
            exclude_indices=exclude_indices,
        )
        for seg in result.segments:
            if seg.index not in exclude_indices:
                all_segments.append(seg)
                ns_new.append(seg)
                exclude_indices.add(seg.index)

        return HybridResult(
            segments=all_segments,
            metadata={
                "name": "v2f_daydream_negspace",
                "v2f_cues": cues[:2],
                "negative_space_alpha": alpha,
                "ns_new_count": len(ns_new),
                "cues": cues[:2],
            },
        )


# ===========================================================================
# Architecture 5: Full Hybrid (v2f + Gen-Check + Negative-Space Daydream)
# ===========================================================================
class FullHybrid(HybridBase):
    """Combines v2f + Gen-Check + negative-space daydream.

    Flow:
      1. Initial retrieval (1 embed)
      2. V2f cue generation (1 LLM) -> retrieve (2 embed)
      3. Gen-Check assessment (1 LLM) -> retrieve gaps (0-2 embed)
      4. Negative-space daydream -> retrieve (0 LLM, 0 embed for vector math)
    """

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        all_cues: list[str] = []
        all_gaps: list[str] = []

        # Step 1: Initial retrieval
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        # Step 2: V2f cue generation
        context_section = _build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = _parse_cues(v2f_output)
        all_cues.extend(cues[:2])

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    exclude_indices.add(seg.index)

        # Step 3: Gen-Check gap assessment
        formatted = _format_segments(all_segments, max_items=16, max_chars=300)
        gap_prompt = GAP_ASSESSMENT_PROMPT.format(
            question=question, formatted_segments=formatted
        )
        gap_output = self.llm_call(gap_prompt)
        gaps = _parse_gaps(gap_output)
        all_gaps.extend(gaps[:2])

        done = "DONE" in gap_output.upper().split("\n")[-1] if gap_output else True
        if not done and gaps:
            for gap in gaps[:2]:
                gap_emb = self.embed_text(gap)
                result = self.store.search(
                    gap_emb,
                    top_k=10,
                    conversation_id=conversation_id,
                    exclude_indices=exclude_indices,
                )
                for seg in result.segments:
                    if seg.index not in exclude_indices:
                        all_segments.append(seg)
                        exclude_indices.add(seg.index)

        # Step 4: Negative-space daydream
        found_indices = [s.index for s in all_segments]
        found_embs = self.store.normalized_embeddings[found_indices]
        centroid = found_embs.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-10)

        alpha = 0.3
        residual = query_norm + alpha * (query_norm - centroid)
        residual /= max(np.linalg.norm(residual), 1e-10)

        ns_new: list[Segment] = []
        result = self.store.search(
            residual,
            top_k=10,
            conversation_id=conversation_id,
            exclude_indices=exclude_indices,
        )
        for seg in result.segments:
            if seg.index not in exclude_indices:
                all_segments.append(seg)
                ns_new.append(seg)
                exclude_indices.add(seg.index)

        return HybridResult(
            segments=all_segments,
            metadata={
                "name": "full_hybrid",
                "v2f_cues": all_cues,
                "gaps": all_gaps,
                "gap_done": done,
                "ns_new_count": len(ns_new),
                "cues": all_cues + all_gaps,
            },
        )


# ===========================================================================
# Architecture 6: V2f + LLM Daydream + Negative-Space Daydream
# ===========================================================================
class V2fWithBothDaydreams(HybridBase):
    """V2f + both LLM and negative-space daydreams."""

    def retrieve(self, question: str, conversation_id: str) -> HybridResult:
        # Step 1-2: Standard v2f
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)

        hop0 = self.store.search(query_emb, top_k=10, conversation_id=conversation_id)
        all_segments = list(hop0.segments)
        exclude_indices = {s.index for s in all_segments}

        context_section = _build_context_section(all_segments)
        v2f_prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        v2f_output = self.llm_call(v2f_prompt)
        cues = _parse_cues(v2f_output)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    exclude_indices.add(seg.index)

        # Step 3a: LLM daydream
        topic_summary = "; ".join(
            seg.text[:80] for seg in sorted(all_segments[:8], key=lambda s: s.turn_id)
        )
        daydream_prompt = DAYDREAM_PROMPT.format(
            question=question, topic_summary=topic_summary
        )
        daydream_output = self.llm_call(daydream_prompt, max_tokens=150)

        daydream_cue = daydream_output.strip()
        if "CUE:" in daydream_cue:
            daydream_cue = daydream_cue.split("CUE:", 1)[1].strip()
        daydream_cue = daydream_cue.split("\n")[0].strip()

        if daydream_cue:
            dd_emb = self.embed_text(daydream_cue)
            result = self.store.search(
                dd_emb,
                top_k=10,
                conversation_id=conversation_id,
                exclude_indices=exclude_indices,
            )
            for seg in result.segments:
                if seg.index not in exclude_indices:
                    all_segments.append(seg)
                    exclude_indices.add(seg.index)

        # Step 3b: Negative-space daydream
        found_indices = [s.index for s in all_segments]
        found_embs = self.store.normalized_embeddings[found_indices]
        centroid = found_embs.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-10)

        alpha = 0.3
        residual = query_norm + alpha * (query_norm - centroid)
        residual /= max(np.linalg.norm(residual), 1e-10)

        ns_new: list[Segment] = []
        result = self.store.search(
            residual,
            top_k=10,
            conversation_id=conversation_id,
            exclude_indices=exclude_indices,
        )
        for seg in result.segments:
            if seg.index not in exclude_indices:
                all_segments.append(seg)
                ns_new.append(seg)
                exclude_indices.add(seg.index)

        return HybridResult(
            segments=all_segments,
            metadata={
                "name": "v2f_both_daydreams",
                "v2f_cues": cues[:2],
                "daydream_cue": daydream_cue,
                "ns_new_count": len(ns_new),
                "cues": cues[:2] + ([daydream_cue] if daydream_cue else []),
            },
        )


# ===========================================================================
# Evaluation harness
# ===========================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: HybridBase,
    question: dict,
    verbose: bool = False,
) -> dict:
    """Evaluate a single architecture on a single question."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Deduplicate preserving order
    seen: set[int] = set()
    deduped: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            deduped.append(seg)
            seen.add(seg.index)
    arch_segments = deduped
    total_retrieved = len(arch_segments)

    # Baseline: cosine top-N at same budget
    query_emb = arch.embed_text(q_text)
    max_budget = max(BUDGETS + [total_retrieved])
    baseline_result = arch.store.search(
        query_emb, top_k=max_budget, conversation_id=conv_id
    )

    baseline_recalls: dict[str, float] = {}
    arch_recalls: dict[str, float] = {}
    for budget in BUDGETS:
        baseline_ids = {s.turn_id for s in baseline_result.segments[:budget]}
        baseline_recalls[f"r@{budget}"] = compute_recall(baseline_ids, source_ids)

        arch_ids = {s.turn_id for s in arch_segments[:budget]}
        arch_recalls[f"r@{budget}"] = compute_recall(arch_ids, source_ids)

    # Also at actual retrieval size
    baseline_ids_actual = {
        s.turn_id for s in baseline_result.segments[:total_retrieved]
    }
    arch_ids_actual = {s.turn_id for s in arch_segments}
    baseline_recalls["r@actual"] = compute_recall(baseline_ids_actual, source_ids)
    arch_recalls["r@actual"] = compute_recall(arch_ids_actual, source_ids)

    row = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question["question_index"],
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "baseline_recalls": baseline_recalls,
        "arch_recalls": arch_recalls,
        "total_retrieved": total_retrieved,
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "metadata": result.metadata,
    }

    if verbose:
        print(f"  Source: {sorted(source_ids)} ({len(source_ids)} turns)")
        print(
            f"  Retrieved: {total_retrieved}, Embed: {arch.embed_calls}, "
            f"LLM: {arch.llm_calls}, Time: {elapsed:.1f}s"
        )
        for budget in BUDGETS:
            b = baseline_recalls[f"r@{budget}"]
            a = arch_recalls[f"r@{budget}"]
            delta = a - b
            marker = "W" if delta > 0.001 else ("L" if delta < -0.001 else "T")
            print(
                f"  @{budget:3d}: baseline={b:.3f} arch={a:.3f} "
                f"delta={delta:+.3f} [{marker}]"
            )
        cues = result.metadata.get("cues", [])
        for cue in cues[:4]:
            print(f"    Cue: {cue[:120]}")
        gaps = result.metadata.get("gaps", [])
        for gap in gaps[:4]:
            print(f"    Gap: {gap[:120]}")
        dd = result.metadata.get("daydream_cue", "")
        if dd:
            print(f"    Daydream: {dd[:120]}")

    return row


def summarize(results: list[dict], variant_name: str, benchmark: str) -> dict:
    """Compute summary statistics."""
    n = len(results)
    if n == 0:
        return {}

    summary: dict = {"variant": variant_name, "benchmark": benchmark, "n": n}

    for label in [f"r@{b}" for b in BUDGETS] + ["r@actual"]:
        b_vals = [r["baseline_recalls"][label] for r in results]
        a_vals = [r["arch_recalls"][label] for r in results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n

        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses

        summary[f"baseline_{label}"] = round(b_mean, 4)
        summary[f"arch_{label}"] = round(a_mean, 4)
        summary[f"delta_{label}"] = round(a_mean - b_mean, 4)
        summary[f"W/T/L_{label}"] = f"{wins}/{ties}/{losses}"

    summary["avg_total_retrieved"] = round(
        sum(r["total_retrieved"] for r in results) / n, 1
    )
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    summary["avg_time_s"] = round(sum(r["time_s"] for r in results) / n, 2)

    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    """Per-category breakdown at r@20 and r@50."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    cat_summaries = {}
    for cat, cat_results in sorted(by_cat.items()):
        n = len(cat_results)
        entry: dict = {"n": n}
        for budget in [20, 50]:
            label = f"r@{budget}"
            b_vals = [r["baseline_recalls"][label] for r in cat_results]
            a_vals = [r["arch_recalls"][label] for r in cat_results]
            b_mean = sum(b_vals) / n
            a_mean = sum(a_vals) / n
            wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
            losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
            entry[f"baseline_{label}"] = round(b_mean, 4)
            entry[f"arch_{label}"] = round(a_mean, 4)
            entry[f"delta_{label}"] = round(a_mean - b_mean, 4)
            entry[f"W/T/L_{label}"] = f"{wins}/{n - wins - losses}/{losses}"
        cat_summaries[cat] = entry
    return cat_summaries


def run_variant(
    variant_name: str,
    arch: HybridBase,
    questions: list[dict],
    benchmark_label: str,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Run one variant, return (results, summary)."""
    print(f"\n{'=' * 70}")
    print(
        f"VARIANT: {variant_name} | BENCHMARK: {benchmark_label} | "
        f"{len(questions)} questions"
    )
    print(f"{'=' * 70}")

    results = []
    for i, question in enumerate(questions):
        q_short = question["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {question.get('category', '?')}: "
            f"{q_short}...",
            flush=True,
        )
        try:
            result = evaluate_one(arch, question, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, variant_name, benchmark_label)

    # Print compact summary
    print(f"\n--- {variant_name} on {benchmark_label} ---")
    for budget in BUDGETS:
        lbl = f"r@{budget}"
        print(
            f"  {lbl}: baseline={summary.get(f'baseline_{lbl}', 0):.3f} "
            f"arch={summary.get(f'arch_{lbl}', 0):.3f} "
            f"delta={summary.get(f'delta_{lbl}', 0):+.3f} "
            f"[{summary.get(f'W/T/L_{lbl}', '?')}]"
        )
    print(
        f"  Avg retrieved: {summary.get('avg_total_retrieved', 0):.0f}, "
        f"Embed: {summary.get('avg_embed_calls', 0):.1f}, "
        f"LLM: {summary.get('avg_llm_calls', 0):.1f}"
    )

    # Category breakdown
    cat_breakdown = summarize_by_category(results)
    if cat_breakdown:
        print("\n  Per-category r@20 / r@50:")
        for cat, vals in sorted(cat_breakdown.items()):
            print(
                f"    {cat:20s} ({vals['n']}q): "
                f"r@20 {vals['arch_r@20']:.3f} ({vals['delta_r@20']:+.3f}) "
                f"r@50 {vals['arch_r@50']:.3f} ({vals['delta_r@50']:+.3f})"
            )

    return results, summary


# ===========================================================================
# Architecture registry
# ===========================================================================
ARCHITECTURES = {
    "v2f_baseline": V2fBaseline,
    "hybrid_v2f_gencheck": HybridV2fGenCheck,
    "v2f_daydream_llm": V2fWithDaydream,
    "v2f_daydream_negspace": V2fWithNegativeSpace,
    "v2f_both_daydreams": V2fWithBothDaydreams,
    "full_hybrid": FullHybrid,
}


# ===========================================================================
# Dataset configs
# ===========================================================================
DATASETS = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions_file": "questions_extended.json",
        "label": "LoCoMo 30q",
        "max_questions": 30,
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions_file": "questions_synthetic.json",
        "label": "Synthetic 19q",
        "max_questions": None,
    },
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid retrieval experiments")
    parser.add_argument("--arch", type=str, help="Architecture name")
    parser.add_argument(
        "--dataset",
        type=str,
        default="locomo_30q",
        choices=list(DATASETS.keys()),
        help="Dataset to use",
    )
    parser.add_argument("--all", action="store_true", help="Run all architectures")
    parser.add_argument(
        "--all-datasets", action="store_true", help="Run on all datasets"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available architectures"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing results"
    )
    args = parser.parse_args()

    if args.list:
        print("Available architectures:")
        for name in ARCHITECTURES:
            print(f"  {name}")
        print("\nAvailable datasets:")
        for name in DATASETS:
            print(f"  {name}")
        return

    # Determine which architectures to run
    arch_names = list(ARCHITECTURES.keys()) if args.all else [args.arch]
    if not args.all and not args.arch:
        parser.error("Specify --arch <name> or --all")

    # Determine which datasets to run
    dataset_names = list(DATASETS.keys()) if args.all_datasets else [args.dataset]

    all_summaries = []

    for ds_name in dataset_names:
        ds = DATASETS[ds_name]
        # Load data
        store = SegmentStore(DATA_DIR, ds["npz"])

        questions_file = DATA_DIR / ds["questions_file"]
        with open(questions_file) as f:
            questions = json.load(f)
        if ds["max_questions"]:
            questions = questions[: ds["max_questions"]]

        for arch_name in arch_names:
            if arch_name not in ARCHITECTURES:
                print(f"Unknown architecture: {arch_name}")
                continue

            result_file = RESULTS_DIR / f"hybrid_{arch_name}_{ds_name}.json"
            if result_file.exists() and not args.force:
                print(
                    f"Skipping {arch_name} on {ds_name} (exists). "
                    f"Use --force to overwrite."
                )
                # Load existing for summary
                with open(result_file) as f:
                    existing = json.load(f)
                if "summary" in existing:
                    all_summaries.append(existing["summary"])
                continue

            arch = ARCHITECTURES[arch_name](store)
            results, summary = run_variant(
                arch_name,
                arch,
                questions,
                ds["label"],
                verbose=args.verbose,
            )
            all_summaries.append(summary)

            # Category breakdown
            cat_breakdown = summarize_by_category(results)

            # Save
            output = {
                "summary": summary,
                "category_breakdown": cat_breakdown,
                "results": results,
            }
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(output, f, indent=2, default=str)
            print(f"  Saved to {result_file}")

    # Final comparison table
    if len(all_summaries) > 1:
        print(f"\n{'=' * 80}")
        print("COMPARISON TABLE")
        print(f"{'=' * 80}")
        header = (
            f"{'Variant':30s} {'Dataset':15s} "
            f"{'r@20':>8s} {'d@20':>8s} {'r@50':>8s} {'d@50':>8s} "
            f"{'LLM':>4s} {'Emb':>4s}"
        )
        print(header)
        print("-" * len(header))
        for s in all_summaries:
            print(
                f"{s.get('variant', '?'):30s} {s.get('benchmark', '?'):15s} "
                f"{s.get('arch_r@20', 0):8.3f} {s.get('delta_r@20', 0):+8.3f} "
                f"{s.get('arch_r@50', 0):8.3f} {s.get('delta_r@50', 0):+8.3f} "
                f"{s.get('avg_llm_calls', 0):4.1f} "
                f"{s.get('avg_embed_calls', 0):4.1f}"
            )


if __name__ == "__main__":
    main()
