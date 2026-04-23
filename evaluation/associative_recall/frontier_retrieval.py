"""A*-like frontier-based retrieval for conversation memory.

Instead of decomposing a question upfront, we use a priority queue of "gaps"
discovered through retrieve-then-reflect loops.

Variants:
  V1: Simple frontier (FIFO, 1 reflect, 2 gaps) — baseline gap framing
  V2: Iterative frontier with re-reflection (max 4 reflects)
  V3: Priority frontier (V2 + model-assigned priorities)
  V4: Hybrid v15 + frontier (proven v15 first, then gap discovery)
  V5: Retrieval-grounded decomposition (linear chain, purely reactive)
"""

import hashlib
import heapq
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
    RetrievalResult,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50, 100]


# ---------------------------------------------------------------------------
# Cache classes — frontier-specific, reads from all existing caches
# ---------------------------------------------------------------------------
class FrontierEmbeddingCache(EmbeddingCache):
    """Reads all existing caches, writes to frontier-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        for name in ("embedding_cache.json", "arch_embedding_cache.json",
                     "agent_embedding_cache.json", "frontier_embedding_cache.json"):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "frontier_embedding_cache.json"
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


class FrontierLLMCache(LLMCache):
    """Reads all existing caches, writes to frontier-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in ("llm_cache.json", "arch_llm_cache.json",
                     "agent_llm_cache.json", "tree_llm_cache.json",
                     "frontier_llm_cache.json"):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                # Skip empty responses (poisoned cache entries)
                for k, v in data.items():
                    if v:  # only import non-empty responses
                        self._cache[k] = v
        self.cache_file = self.cache_dir / "frontier_llm_cache.json"
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
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Gap:
    """A gap in retrieval — something still missing."""
    query: str
    priority: float = 0.0  # higher = explore first
    source: str = ""  # which reflect call generated this

    def __lt__(self, other: "Gap"):
        # For heapq (min-heap), negate priority so highest goes first
        return self.priority > other.priority


@dataclass
class FrontierResult:
    """Result from a frontier retrieval."""
    segments: list[Segment]
    embed_calls: int = 0
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class FrontierBase:
    """Base class with embedding/LLM utilities."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = FrontierEmbeddingCache()
        self.llm_cache = FrontierLLMCache()
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

    def save_caches(self) -> None:
        self.embedding_cache.save()
        self.llm_cache.save()

    def reset_counters(self) -> None:
        self.embed_calls = 0
        self.llm_calls = 0

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: format segments for LLM context
# ---------------------------------------------------------------------------
def _format_segments(segments: list[Segment], max_items: int = 12,
                     max_chars: int = 250) -> str:
    """Format segments chronologically for LLM context.

    Matches agent_architectures._format_segments exactly (12 items, 250 chars)
    to ensure cache compatibility with v15_control.
    """
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared v15-style retrieval logic
# ---------------------------------------------------------------------------
def _build_v15_prompt(question: str, context: str) -> str:
    """Build the v15 self-monitoring prompt. Shared across V1, V4, etc.

    IMPORTANT: This must match agent_architectures.V15Control's prompt EXACTLY
    (including whitespace) to hit the same LLM cache entries.
    """
    return f"""\
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


def _v15_retrieve(
    base: FrontierBase,
    question: str,
    conversation_id: str,
) -> tuple[list[Segment], set[int], list[str]]:
    """Execute v15-style retrieval: question top-10 + 2 cues top-10 each.

    Returns (all_segments, exclude_set, cues).
    """
    exclude: set[int] = set()

    # Hop 0: question embedding
    query_emb = base.embed_text(question)
    result = base.store.search(
        query_emb, top_k=10, conversation_id=conversation_id
    )
    all_segments: list[Segment] = list(result.segments)
    for s in all_segments:
        exclude.add(s.index)

    # Hop 1: v15 self-monitoring prompt with 2 forced cues
    context = _format_segments(all_segments)
    prompt = _build_v15_prompt(question, context)
    response = base.llm_call(prompt)
    cues = _parse_cues(response)

    for cue in cues[:2]:
        cue_emb = base.embed_text(cue)
        result = base.store.search(
            cue_emb, top_k=10, conversation_id=conversation_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

    return all_segments, exclude, cues[:2]


# ===================================================================
# V1: Simple frontier (baseline gap framing)
# ===================================================================
class SimpleFrontier(FrontierBase):
    """V1: 1 reflect call generates 2 gaps after initial retrieval.

    Explore each gap once. No priority ordering (FIFO).
    This is basically v15 but with "gap" framing instead of "cue" framing.
    """

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        all_segments, exclude, cues = _v15_retrieve(
            self, question, conversation_id
        )

        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "simple_frontier",
                "gaps": cues,
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# V2: Iterative frontier with re-reflection
# ===================================================================
class IterativeFrontier(FrontierBase):
    """V2: After exploring each gap, reflect again.

    New gaps can be added based on what the new retrieval found.
    Stop when reflection says "no more gaps" or budget exhausted.
    Max 4 reflect calls total (cost control).
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 max_reflects: int = 4, gaps_per_reflect: int = 2,
                 segment_budget: int = 80):
        super().__init__(store, client)
        self.max_reflects = max_reflects
        self.gaps_per_reflect = gaps_per_reflect
        self.segment_budget = segment_budget

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []
        frontier: list[Gap] = []
        reflect_log: list[dict] = []

        # Phase 0: initial probe
        query_emb = self.embed_text(question)
        result = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments.extend(result.segments)
        for s in result.segments:
            exclude.add(s.index)

        for reflect_i in range(self.max_reflects):
            if len(all_segments) >= self.segment_budget:
                break

            context = _format_segments(all_segments)
            explored_text = ""
            if reflect_log:
                explored = []
                for entry in reflect_log:
                    for g in entry.get("gaps", []):
                        explored.append(f"- {g}")
                explored_text = (
                    "\n\nALREADY SEARCHED FOR (do NOT repeat these):\n"
                    + "\n".join(explored)
                )

            prompt = f"""\
You are performing iterative retrieval over a conversation history to answer \
a question. Your cues will be embedded and matched via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({len(all_segments)} segments):
{context}{explored_text}

First, assess: How well do the retrieved segments cover the question? \
What specific content is still missing?

If content is still missing, generate 1-2 search cues targeting the gaps. \
Use specific vocabulary that would appear in the target conversation turns.

If the retrieval looks complete, respond with DONE.

Format:
ASSESSMENT: <1-2 sentence evaluation of what's found vs missing>
CUE: <text>
(or)
ASSESSMENT: <evaluation>
DONE"""

            response = self.llm_call(prompt)

            # Parse
            assessment = ""
            gaps = []
            done = False
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("ASSESSMENT:"):
                    assessment = line[11:].strip()
                elif line.startswith("CUE:"):
                    cue = line[4:].strip()
                    if cue:
                        gaps.append(Gap(query=cue, source=f"reflect_{reflect_i}"))
                elif line.strip().upper() == "DONE":
                    done = True

            reflect_log.append({
                "reflect": reflect_i,
                "assessment": assessment,
                "gaps": [g.query for g in gaps],
                "done": done,
            })

            if done or not gaps:
                break

            # Explore gaps from this reflect
            for gap in gaps[:self.gaps_per_reflect]:
                if len(all_segments) >= self.segment_budget:
                    break
                gap_emb = self.embed_text(gap.query)
                result = self.store.search(
                    gap_emb, top_k=10, conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                for seg in result.segments:
                    if seg.index not in exclude:
                        all_segments.append(seg)
                        exclude.add(seg.index)

        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "iterative_frontier",
                "reflect_log": reflect_log,
                "total_segments": len(all_segments),
                "num_reflects": len(reflect_log),
            },
        )


# ===================================================================
# V3: Priority frontier
# ===================================================================
class PriorityFrontier(FrontierBase):
    """V3: Same as V2 but reflect assigns priority scores.

    Higher priority gaps explored first.
    Tests whether the model can usefully prioritize.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 max_reflects: int = 4, segment_budget: int = 80):
        super().__init__(store, client)
        self.max_reflects = max_reflects
        self.segment_budget = segment_budget

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []
        priority_queue: list[Gap] = []  # min-heap, Gap.__lt__ handles priority
        explored_gaps: list[str] = []
        reflect_log: list[dict] = []

        # Phase 0: initial probe
        query_emb = self.embed_text(question)
        result = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments.extend(result.segments)
        for s in result.segments:
            exclude.add(s.index)

        # Initial reflect to seed the frontier
        new_gaps = self._reflect(
            question, all_segments, explored_gaps, reflect_log, 0
        )
        for g in new_gaps:
            heapq.heappush(priority_queue, g)

        # Iterative exploration
        reflect_count = 1
        while (priority_queue
               and len(all_segments) < self.segment_budget
               and reflect_count < self.max_reflects):
            # Pop best gap
            gap = heapq.heappop(priority_queue)
            explored_gaps.append(gap.query)

            gap_emb = self.embed_text(gap.query)
            result = self.store.search(
                gap_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

            # Re-reflect
            new_gaps = self._reflect(
                question, all_segments, explored_gaps, reflect_log, reflect_count
            )
            reflect_count += 1
            for g in new_gaps:
                heapq.heappush(priority_queue, g)

        # Drain remaining frontier if budget allows
        while priority_queue and len(all_segments) < self.segment_budget:
            gap = heapq.heappop(priority_queue)
            explored_gaps.append(gap.query)
            gap_emb = self.embed_text(gap.query)
            result = self.store.search(
                gap_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "priority_frontier",
                "reflect_log": reflect_log,
                "total_segments": len(all_segments),
                "num_reflects": len(reflect_log),
                "explored_gaps": explored_gaps,
            },
        )

    def _reflect(
        self,
        question: str,
        all_segments: list[Segment],
        explored_gaps: list[str],
        reflect_log: list[dict],
        reflect_i: int,
    ) -> list[Gap]:
        context = _format_segments(all_segments)
        explored_text = ""
        if explored_gaps:
            explored_text = (
                "\n\nALREADY SEARCHED FOR (do NOT repeat):\n"
                + "\n".join(f"- {g}" for g in explored_gaps)
            )

        prompt = f"""\
You are performing iterative retrieval over a conversation history. \
Your cues will be embedded and matched via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({len(all_segments)} segments):
{context}{explored_text}

Assess what's been found and what's missing. Then generate 1-3 search cues, \
each with a priority score (1-10, where 10 = most likely to find critical \
missing content).

If retrieval looks complete, respond with DONE.

Format:
ASSESSMENT: <what's found, what's missing>
GAP [priority]: <search text>
GAP [priority]: <search text>
(or)
ASSESSMENT: <evaluation>
DONE"""

        response = self.llm_call(prompt)

        assessment = ""
        gaps = []
        done = False
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("ASSESSMENT:"):
                assessment = line[11:].strip()
            elif line.upper().startswith("GAP"):
                # Parse "GAP [7]: search text" or "GAP [7] search text"
                rest = line[3:].strip()
                priority = 5.0  # default
                query = rest
                if rest.startswith("["):
                    bracket_end = rest.find("]")
                    if bracket_end > 0:
                        try:
                            priority = float(rest[1:bracket_end])
                        except ValueError:
                            pass
                        query = rest[bracket_end + 1:].strip()
                        if query.startswith(":"):
                            query = query[1:].strip()
                if query:
                    gaps.append(Gap(query=query, priority=priority,
                                    source=f"reflect_{reflect_i}"))
            elif line.strip().upper() == "DONE":
                done = True

        reflect_log.append({
            "reflect": reflect_i,
            "assessment": assessment,
            "gaps": [(g.query, g.priority) for g in gaps],
            "done": done,
        })

        if done:
            return []
        return gaps


# ===================================================================
# V4: Hybrid v15 + frontier
# ===================================================================
class HybridV15Frontier(FrontierBase):
    """V4: V1 (proven v15) for top-20 positions, then frontier for 21+.

    Phase 1: Run V1 (SimpleFrontier) — produces ~30 segments in v15 order
    Phase 2: Reflect on what V1 found, generate frontier gaps
    Phase 3: Explore 1-2 frontier gaps (results go AFTER v15 results)

    The key: V1's 30 segments fill positions 1-20 with optimal ordering.
    Frontier results only affect r@50+ positions, so r@20 is preserved.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 max_frontier_gaps: int = 2):
        super().__init__(store, client)
        self.max_frontier_gaps = max_frontier_gaps

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        # === Phase 1: exact v15 retrieval (shared code with V1) ===
        all_segments, exclude, v15_cues = _v15_retrieve(
            self, question, conversation_id
        )
        v15_segment_count = len(all_segments)

        # === Phase 2: frontier reflection ===
        context = _format_segments(all_segments, max_items=20)
        searched_text = "\n".join(f"- {c}" for c in v15_cues)
        frontier_prompt = (
            "You are performing gap analysis on retrieval results for a "
            "conversation question. Your cues will be embedded and matched "
            "via cosine similarity.\n\n"
            f"Question: {question}\n\n"
            "ALREADY SEARCHED WITH:\n"
            "- Original question (embedding)\n"
            f"{searched_text}\n\n"
            f"RETRIEVED SO FAR ({len(all_segments)} segments):\n"
            f"{context}\n\n"
            "Assess: what specific content is STILL MISSING to fully answer "
            "this question? Generate 1-2 gap queries targeting the missing "
            "content. Use specific vocabulary from the conversation domain.\n\n"
            "If retrieval looks complete, respond with DONE.\n\n"
            "Format:\n"
            "ASSESSMENT: <what's covered, what's missing>\n"
            "GAP: <search text for missing content>\n"
            "(or)\n"
            "ASSESSMENT: <evaluation>\n"
            "DONE"
        )

        response = self.llm_call(frontier_prompt)
        frontier_gaps: list[Gap] = []
        assessment = ""
        done = False
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("ASSESSMENT:"):
                assessment = line[11:].strip()
            elif line.upper().startswith("GAP:"):
                query = line[4:].strip()
                if query:
                    frontier_gaps.append(Gap(query=query, source="frontier"))
            elif line.strip().upper() == "DONE":
                done = True

        # === Phase 3: explore frontier gaps ===
        explored_gaps: list[str] = []
        if not done:
            for gap in frontier_gaps[:self.max_frontier_gaps]:
                gap_emb = self.embed_text(gap.query)
                result = self.store.search(
                    gap_emb, top_k=10, conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                for seg in result.segments:
                    if seg.index not in exclude:
                        all_segments.append(seg)
                        exclude.add(seg.index)
                explored_gaps.append(gap.query)

        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "hybrid_v15_frontier",
                "v15_cues": v15_cues,
                "frontier_assessment": assessment,
                "frontier_gaps": explored_gaps,
                "frontier_done": done,
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# V4b: Hybrid v15 + frontier with 3 gaps
# ===================================================================
class HybridV15Frontier3(HybridV15Frontier):
    """V4b: Same as V4 but allows up to 3 frontier gaps."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        super().__init__(store, client, max_frontier_gaps=3)


# ===================================================================
# V4c: Hybrid v15 + frontier + 2nd reflect
# ===================================================================
class HybridV15FrontierDeep(FrontierBase):
    """V4c: V4 + a second reflect round after exploring frontier gaps.

    Phase 1: v15 retrieval (positions 1-30)
    Phase 2: reflect + explore 2 gaps (positions 31-50)
    Phase 3: reflect again on everything + explore 1-2 more gaps (51+)
    """

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        # Phase 1: v15
        all_segments, exclude, v15_cues = _v15_retrieve(
            self, question, conversation_id
        )

        all_explored: list[str] = list(v15_cues)
        reflect_log: list[dict] = []

        # Phase 2-3: two reflect rounds
        for round_i in range(2):
            context = _format_segments(all_segments, max_items=20)
            searched = "\n".join(f"- {c}" for c in all_explored)
            prompt = (
                "You are performing gap analysis on retrieval results for a "
                "conversation question. Your cues will be embedded and matched "
                "via cosine similarity.\n\n"
                f"Question: {question}\n\n"
                "ALREADY SEARCHED WITH:\n"
                "- Original question (embedding)\n"
                f"{searched}\n\n"
                f"RETRIEVED SO FAR ({len(all_segments)} segments):\n"
                f"{context}\n\n"
                "Assess: what specific content is STILL MISSING to fully answer "
                "this question? Generate 1-2 gap queries targeting the missing "
                "content. Use specific vocabulary from the conversation domain.\n\n"
                "If retrieval looks complete, respond with DONE.\n\n"
                "Format:\n"
                "ASSESSMENT: <what's covered, what's missing>\n"
                "GAP: <search text for missing content>\n"
                "(or)\n"
                "ASSESSMENT: <evaluation>\n"
                "DONE"
            )
            response = self.llm_call(prompt)
            gaps: list[Gap] = []
            assessment = ""
            done = False
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("ASSESSMENT:"):
                    assessment = line[11:].strip()
                elif line.upper().startswith("GAP:"):
                    query = line[4:].strip()
                    if query:
                        gaps.append(Gap(query=query, source=f"reflect_{round_i}"))
                elif line.strip().upper() == "DONE":
                    done = True

            reflect_log.append({
                "round": round_i,
                "assessment": assessment,
                "gaps": [g.query for g in gaps],
                "done": done,
            })

            if done or not gaps:
                break

            for gap in gaps[:2]:
                gap_emb = self.embed_text(gap.query)
                result = self.store.search(
                    gap_emb, top_k=10, conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                for seg in result.segments:
                    if seg.index not in exclude:
                        all_segments.append(seg)
                        exclude.add(seg.index)
                all_explored.append(gap.query)

        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "hybrid_v15_frontier_deep",
                "v15_cues": v15_cues,
                "reflect_log": reflect_log,
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# V6: V15 + backfill-only (control for whether frontier adds value)
# ===================================================================
class V15BackfillOnly(FrontierBase):
    """V6: Just V1 (v15) with no frontier. Identical to V1.

    This serves as a control: V1's r@50 performance with backfill
    should be compared against V4's r@50 to measure frontier's value.
    Identical to V1 — exists just for labeling clarity in the summary.
    """

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        all_segments, exclude, cues = _v15_retrieve(
            self, question, conversation_id
        )
        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "v15_backfill_only",
                "cues": cues,
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# V7: Hybrid v15 + constrained frontier (no boolean/meta queries)
# ===================================================================
class HybridV15FrontierConstrained(FrontierBase):
    """V7: V4 with a much stricter frontier prompt.

    The frontier gaps in V4 often degrade to Boolean-style queries
    ("X OR Y OR Z") or meta-instructions, which are terrible for
    embedding search. This variant uses the same v15-style prompt
    format (proven to work) for the frontier phase too.
    """

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        # Phase 1: v15
        all_segments, exclude, v15_cues = _v15_retrieve(
            self, question, conversation_id
        )

        # Phase 2: second v15-style reflect (same prompt structure that works)
        context = _format_segments(all_segments)
        prev_cues = "\n".join(f"- {c}" for c in v15_cues)
        prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

PREVIOUS CUES ALREADY TRIED (do NOT repeat these):
{prev_cues}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

CRITICAL: Each cue must be 1-2 sentences of plausible conversation content. \
Do NOT use "OR" patterns, boolean syntax, or meta-instructions like \
"search for X". Write text that would actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

        response = self.llm_call(prompt)
        frontier_cues = _parse_cues(response)

        for cue in frontier_cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "hybrid_v15_frontier_constrained",
                "v15_cues": v15_cues,
                "frontier_cues": frontier_cues[:2],
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# V8: Double v15 (2 rounds of v15, no frontier framing)
# ===================================================================
class DoubleV15(FrontierBase):
    """V8: Two consecutive v15-style rounds.

    Round 1: Standard v15 (question -> top-10, 2 cues -> top-10 each)
    Round 2: Same v15 prompt but with MORE context (all 30 segments) and
             explicit instruction to not repeat previous cues.

    Tests the simplest possible "more retrieval" approach: just ask v15
    again with the accumulated context.
    """

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        # Round 1: standard v15
        all_segments, exclude, cues_r1 = _v15_retrieve(
            self, question, conversation_id
        )

        # Round 2: v15 again with accumulated context + don't repeat
        context = _format_segments(all_segments)
        prev_cues = "\n".join(f"- {c}" for c in cues_r1)
        prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

PREVIOUS CUES ALREADY TRIED (do NOT repeat these):
{prev_cues}

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

        response = self.llm_call(prompt)
        cues_r2 = _parse_cues(response)

        for cue in cues_r2[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "double_v15",
                "cues_r1": cues_r1,
                "cues_r2": cues_r2[:2],
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# V5: Retrieval-grounded decomposition
# ===================================================================
class RetrievalGroundedDecomp(FrontierBase):
    """V5: Linear chain — purely reactive to what's found.

    retrieve -> reflect -> generate 1 gap -> retrieve -> reflect -> ...
    No branching. Tests whether depth alone is enough.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 max_rounds: int = 4, segment_budget: int = 80):
        super().__init__(store, client)
        self.max_rounds = max_rounds
        self.segment_budget = segment_budget

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        exclude: set[int] = set()
        all_segments: list[Segment] = []
        round_log: list[dict] = []

        # Initial probe
        query_emb = self.embed_text(question)
        result = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments.extend(result.segments)
        for s in result.segments:
            exclude.add(s.index)

        for round_i in range(self.max_rounds):
            if len(all_segments) >= self.segment_budget:
                break

            context = _format_segments(all_segments)
            prev_text = ""
            if round_log:
                prev_items = [entry["gap"] for entry in round_log if entry.get("gap")]
                if prev_items:
                    prev_text = (
                        "\n\nPREVIOUS SEARCHES (do NOT repeat):\n"
                        + "\n".join(f"- {p}" for p in prev_items)
                    )

            prompt = f"""\
You are performing iterative retrieval over a conversation to answer a \
question. Your cue will be embedded and matched via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({len(all_segments)} segments):
{context}{prev_text}

Given what you've found, what is the single most important piece of missing \
content? Generate exactly 1 search cue targeting it. Use specific vocabulary \
that would appear in the actual conversation.

If the retrieval is complete, respond with DONE.

Format:
ASSESSMENT: <what's found, what's the biggest gap>
CUE: <search text>
(or)
ASSESSMENT: <evaluation>
DONE"""

            response = self.llm_call(prompt)

            assessment = ""
            gap_query = ""
            done = False
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("ASSESSMENT:"):
                    assessment = line[11:].strip()
                elif line.startswith("CUE:"):
                    gap_query = line[4:].strip()
                elif line.strip().upper() == "DONE":
                    done = True

            round_log.append({
                "round": round_i,
                "assessment": assessment,
                "gap": gap_query,
                "done": done,
            })

            if done or not gap_query:
                break

            gap_emb = self.embed_text(gap_query)
            result = self.store.search(
                gap_emb, top_k=10, conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    all_segments.append(seg)
                    exclude.add(seg.index)

        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "retrieval_grounded_decomp",
                "round_log": round_log,
                "total_segments": len(all_segments),
                "num_rounds": len(round_log),
            },
        )


# ===================================================================
# Evaluation helpers
# ===================================================================
def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate_one(
    arch: FrontierBase,
    question: dict,
    backfill: bool = True,
    verbose: bool = False,
) -> dict:
    """Evaluate a single frontier architecture on one question."""
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    # Deduplicate preserving order
    seen = set()
    deduped: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            deduped.append(seg)
            seen.add(seg.index)
    arch_segments = deduped
    total_retrieved = len(arch_segments)

    # Baseline: cosine top-N
    query_emb = arch.embed_text(q_text)
    max_budget = max(BUDGETS + [total_retrieved])
    baseline_result = arch.store.search(
        query_emb, top_k=max_budget, conversation_id=conv_id
    )

    # Backfill: extend arch pool with baseline results beyond arch pool
    if backfill:
        arch_indices = {seg.index for seg in arch_segments}
        backfill_segs = [
            seg for seg in baseline_result.segments
            if seg.index not in arch_indices
        ]
        arch_with_backfill = list(arch_segments) + backfill_segs
    else:
        arch_with_backfill = arch_segments

    # Compute recalls at fixed budgets
    baseline_recalls = {}
    arch_recalls = {}
    for budget in BUDGETS:
        baseline_ids = {s.turn_id for s in baseline_result.segments[:budget]}
        baseline_recalls[f"r@{budget}"] = compute_recall(baseline_ids, source_ids)

        arch_ids = {s.turn_id for s in arch_with_backfill[:budget]}
        arch_recalls[f"r@{budget}"] = compute_recall(arch_ids, source_ids)

    # Also at actual retrieval count
    baseline_ids_actual = {s.turn_id for s in baseline_result.segments[:total_retrieved]}
    arch_ids_actual = {s.turn_id for s in arch_segments}
    baseline_recalls["r@actual"] = compute_recall(baseline_ids_actual, source_ids)
    arch_recalls["r@actual"] = compute_recall(arch_ids_actual, source_ids)

    result_dict = {
        "conversation_id": conv_id,
        "category": question["category"],
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
        print(f"  Retrieved: {total_retrieved}, Embed: {arch.embed_calls}, "
              f"LLM: {arch.llm_calls}, Time: {elapsed:.1f}s")
        for budget in BUDGETS:
            b = baseline_recalls[f"r@{budget}"]
            a = arch_recalls[f"r@{budget}"]
            delta = a - b
            marker = "W" if delta > 0.001 else ("L" if delta < -0.001 else "T")
            print(f"  @{budget:3d}: baseline={b:.3f} arch={a:.3f} "
                  f"delta={delta:+.3f} [{marker}]")

    return result_dict


def summarize(results: list[dict], arch_name: str, benchmark: str) -> dict:
    """Compute summary statistics."""
    n = len(results)
    if n == 0:
        return {}

    summary = {"arch": arch_name, "benchmark": benchmark, "n": n}

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
        sum(r["total_retrieved"] for r in results) / n, 1)
    summary["avg_embed_calls"] = round(
        sum(r["embed_calls"] for r in results) / n, 1)
    summary["avg_llm_calls"] = round(
        sum(r["llm_calls"] for r in results) / n, 1)
    summary["avg_time_s"] = round(
        sum(r["time_s"] for r in results) / n, 2)

    return summary


def print_summary(summary: dict) -> None:
    """Print a compact summary table row."""
    if not summary:
        return
    print(f"\n--- {summary['arch']} on {summary['benchmark']} ---")
    for budget in BUDGETS:
        lbl = f"r@{budget}"
        print(f"  {lbl}: baseline={summary.get(f'baseline_{lbl}', 0):.3f} "
              f"arch={summary.get(f'arch_{lbl}', 0):.3f} "
              f"delta={summary.get(f'delta_{lbl}', 0):+.3f} "
              f"W/T/L={summary.get(f'W/T/L_{lbl}', '?')}")
    print(f"  Avg retrieved: {summary.get('avg_total_retrieved', 0):.0f}, "
          f"Embed: {summary.get('avg_embed_calls', 0):.1f}, "
          f"LLM: {summary.get('avg_llm_calls', 0):.1f}, "
          f"Time: {summary.get('avg_time_s', 0):.1f}s")


# ===================================================================
# Registry
# ===================================================================
# ===================================================================
# V9: Triple v15 (3 rounds)
# ===================================================================
class TripleV15(FrontierBase):
    """V9: Three consecutive v15-style rounds.

    Tests whether a third round adds value at r@100 without hurting r@20/50.
    """

    def retrieve(self, question: str, conversation_id: str) -> FrontierResult:
        # Round 1: standard v15
        all_segments, exclude, cues_r1 = _v15_retrieve(
            self, question, conversation_id
        )
        all_cues = list(cues_r1)

        # Rounds 2-3
        for round_i in range(2):
            context = _format_segments(all_segments)
            prev_cues = "\n".join(f"- {c}" for c in all_cues)
            prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

PREVIOUS CUES ALREADY TRIED (do NOT repeat these):
{prev_cues}

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

            response = self.llm_call(prompt)
            new_cues = _parse_cues(response)

            for cue in new_cues[:2]:
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb, top_k=10, conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                for seg in result.segments:
                    if seg.index not in exclude:
                        all_segments.append(seg)
                        exclude.add(seg.index)
                all_cues.append(cue)

        return FrontierResult(
            segments=all_segments,
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "name": "triple_v15",
                "all_cues": all_cues,
                "total_segments": len(all_segments),
            },
        )


FRONTIER_ARCHITECTURES = {
    "simple_frontier": SimpleFrontier,
    "iterative_frontier": IterativeFrontier,
    "priority_frontier": PriorityFrontier,
    "hybrid_v15_frontier": HybridV15Frontier,
    "hybrid_v15_frontier_3gap": HybridV15Frontier3,
    "hybrid_v15_frontier_deep": HybridV15FrontierDeep,
    "hybrid_v15_constrained": HybridV15FrontierConstrained,
    "double_v15": DoubleV15,
    "triple_v15": TripleV15,
    "retrieval_grounded_decomp": RetrievalGroundedDecomp,
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default=None,
                        help="Run specific architecture (default: all)")
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-backfill", action="store_true",
                        help="Disable cosine backfill")
    args = parser.parse_args()

    # Load data
    with open(DATA_DIR / "questions_extended.json") as f:
        all_questions = json.load(f)

    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    print(f"Loaded {len(store.segments)} segments")

    locomo_qs = [q for q in all_questions
                 if q.get("benchmark") == "locomo"][:args.num_questions]
    print(f"LoCoMo: {len(locomo_qs)} questions")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which architectures to run
    if args.arch:
        arch_names = [args.arch]
    else:
        arch_names = list(FRONTIER_ARCHITECTURES.keys())

    all_summaries = []

    for arch_name in arch_names:
        if arch_name not in FRONTIER_ARCHITECTURES:
            print(f"Unknown architecture: {arch_name}")
            continue

        results_file = RESULTS_DIR / f"frontier_{arch_name}_locomo_30q.json"

        if results_file.exists() and not args.force:
            print(f"\nSkipping {arch_name} (exists, use --force)")
            with open(results_file) as f:
                existing = json.load(f)
            summary = summarize(existing, arch_name, "locomo_30q")
            all_summaries.append(summary)
            print_summary(summary)
            continue

        arch_cls = FRONTIER_ARCHITECTURES[arch_name]
        arch = arch_cls(store)

        print(f"\n{'='*70}")
        print(f"ARCH: {arch_name} | LoCoMo | {len(locomo_qs)} questions")
        print(f"{'='*70}")

        results = []
        for i, question in enumerate(locomo_qs):
            q_short = question["question"][:55]
            print(f"  [{i+1}/{len(locomo_qs)}] {question['category']}: "
                  f"{q_short}...", flush=True)
            try:
                result_dict = evaluate_one(
                    arch, question,
                    backfill=not args.no_backfill,
                    verbose=args.verbose,
                )
                results.append(result_dict)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
            sys.stdout.flush()
            if (i + 1) % 5 == 0:
                arch.save_caches()

        arch.save_caches()
        summary = summarize(results, arch_name, "locomo_30q")
        all_summaries.append(summary)
        print_summary(summary)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {results_file}")

    # Grand summary table
    print(f"\n{'='*120}")
    print("FRONTIER RETRIEVAL — GRAND SUMMARY TABLE")
    print(f"{'='*120}")
    print(f"{'Architecture':<30s} {'B-r@20':>8s} {'A-r@20':>8s} "
          f"{'Delta':>8s} {'W/T/L':>10s} {'B-r@50':>8s} {'A-r@50':>8s} "
          f"{'D-r@50':>8s} {'#Ret':>6s} {'Emb':>5s} "
          f"{'LLM':>5s} {'Time':>6s}")
    print("-" * 120)
    for s in all_summaries:
        if not s:
            continue
        print(f"{s['arch']:<30s} "
              f"{s.get('baseline_r@20', 0):>8.3f} "
              f"{s.get('arch_r@20', 0):>8.3f} "
              f"{s.get('delta_r@20', 0):>+8.3f} "
              f"{s.get('W/T/L_r@20', '?'):>10s} "
              f"{s.get('baseline_r@50', 0):>8.3f} "
              f"{s.get('arch_r@50', 0):>8.3f} "
              f"{s.get('delta_r@50', 0):>+8.3f} "
              f"{s.get('avg_total_retrieved', 0):>6.0f} "
              f"{s.get('avg_embed_calls', 0):>5.1f} "
              f"{s.get('avg_llm_calls', 0):>5.0f} "
              f"{s.get('avg_time_s', 0):>6.1f}")
    print("-" * 120)
    print("Reference v15 (1 hop, 2 cues, backfill): LoCoMo delta r@20 = +0.339")

    # Save all summaries
    summary_file = RESULTS_DIR / "frontier_all_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summaries to {summary_file}")


if __name__ == "__main__":
    main()
