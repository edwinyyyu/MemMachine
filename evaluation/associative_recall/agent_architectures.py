"""Agent-centric retrieval architectures with real model agency.

Each architecture gives the LLM control over the retrieval process:
how many cues to generate, when to stop, what strategy to use.
"""

import json
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
AGENT_CACHE_DIR = CACHE_DIR


@dataclass
class AgentArchResult:
    """Result from an agent architecture's retrieval."""

    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cache classes that isolate from the main experiment caches
# ---------------------------------------------------------------------------
class AgentEmbeddingCache(EmbeddingCache):
    """Reads main + arch caches, writes to agent-specific file."""

    def __init__(self):
        self.cache_dir = AGENT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        # Read from all existing caches
        for name in (
            "embedding_cache.json",
            "arch_embedding_cache.json",
            "agent_embedding_cache.json",
        ):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "agent_embedding_cache.json"
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


class AgentLLMCache(LLMCache):
    """Reads main + arch caches, writes to agent-specific file."""

    def __init__(self):
        self.cache_dir = AGENT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        for name in ("llm_cache.json", "arch_llm_cache.json", "agent_llm_cache.json"):
            p = self.cache_dir / name
            if p.exists():
                with open(p) as f:
                    self._cache.update(json.load(f))
        self.cache_file = self.cache_dir / "agent_llm_cache.json"
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
class AgentBase:
    """Base class with embedding/LLM utilities and counters."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = AgentEmbeddingCache()
        self.llm_cache = AgentLLMCache()
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

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: format segments for display in prompts
# ---------------------------------------------------------------------------
def _format_segments(
    segments: list[Segment], max_items: int = 12, max_chars: int = 250
) -> str:
    """Format segments chronologically for LLM context."""
    sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:max_items]
    lines = []
    for seg in sorted_segs:
        lines.append(f"[Turn {seg.turn_id}, {seg.role}]: {seg.text[:max_chars]}")
    return "\n".join(lines)


# ===================================================================
# Architecture 1: Agentic Loop with Model Agency
# ===================================================================
class AgenticLoop(AgentBase):
    """The model controls cue count, strategy, and stopping.

    Each iteration, the model sees what's been found and decides:
    - SEARCH: generate 1-4 cues (model chooses how many)
    - STOP: enough context found

    Key differences from v15:
    - Variable cue count per hop (not fixed at 2)
    - Model-initiated stopping (not fixed hops)
    - Explicit assessment drives action choice
    - Max 4 LLM calls (orient + up to 3 search iterations)
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_iterations: int = 3,
        per_cue_k: int = 10,
    ):
        super().__init__(store, client)
        self.max_iterations = max_iterations
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        # Hop 0: initial retrieval with question embedding
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )

        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}
        previous_cues: list[str] = []
        iteration_log: list[dict] = []

        for iteration in range(self.max_iterations):
            context = _format_segments(all_segments)

            prev_cue_text = ""
            if previous_cues:
                prev_cue_text = (
                    "\n\nPREVIOUS CUES ALREADY TRIED (do NOT repeat):\n"
                    + "\n".join(f"- {c}" for c in previous_cues)
                )

            prompt = f"""\
You are performing iterative retrieval over a conversation history to answer \
a question. Your cues will be embedded and matched via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({len(all_segments)} segments):
{context}{prev_cue_text}

INSTRUCTIONS:
1. ASSESS: How well do the retrieved segments cover the question? What's missing?
2. DECIDE: Do you need more search, or is this sufficient?
3. ACT: Either STOP or generate cues.

If you generate cues:
- Choose how many (1-4) based on how much is missing
- Each cue should be 1-2 sentences of plausible conversation content
- Use specific vocabulary that would appear in the target turns
- Each cue must target DIFFERENT missing content

Format:
ASSESSMENT: <1-2 sentences>
ACTION: SEARCH or STOP
CUE: <text>
(repeat CUE lines as needed, 1-4 cues)

If stopping:
ASSESSMENT: <1-2 sentences>
ACTION: STOP"""

            response = self.llm_call(prompt)

            # Parse response
            action = "STOP"
            cues = []
            assessment = ""
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("ASSESSMENT:"):
                    assessment = line[11:].strip()
                elif line.upper().startswith("ACTION:"):
                    action_text = line[7:].strip().upper()
                    if "SEARCH" in action_text:
                        action = "SEARCH"
                    else:
                        action = "STOP"
                elif line.startswith("CUE:"):
                    cue = line[4:].strip()
                    if cue:
                        cues.append(cue)

            iteration_log.append(
                {
                    "iteration": iteration,
                    "assessment": assessment,
                    "action": action,
                    "num_cues": len(cues),
                    "cues": cues,
                }
            )

            if action == "STOP" or not cues:
                break

            # Execute search for each cue
            for cue in cues[:4]:  # Hard cap at 4
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=seen_indices,
                )
                for seg in result.segments:
                    if seg.index not in seen_indices:
                        all_segments.append(seg)
                        seen_indices.add(seg.index)
                previous_cues.append(cue)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "agentic_loop",
                "iterations": len(iteration_log),
                "iteration_log": iteration_log,
                "total_cues": len(previous_cues),
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 2: Context Bootstrapping (Orient-Focus-Refine)
# ===================================================================
class ContextBootstrapping(AgentBase):
    """Three-phase retrieval: Orient, Focus, Refine.

    Phase 1 (ORIENT): Broad initial retrieval + ask model what topics
    the memory store contains relevant to the question.
    Phase 2 (FOCUS): Generate targeted cues based on orientation.
    Phase 3 (REFINE): Based on focused results, fill gaps.

    Tests whether knowing the memory's content first improves cue quality.
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        # Phase 0: Initial broad retrieval
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        # Phase 1: ORIENT — understand what's in the memory
        context = _format_segments(all_segments)
        orient_prompt = f"""\
You are exploring a conversation memory store to answer a question. Here's \
what an initial broad search found.

Question: {question}

INITIAL RETRIEVAL (top 10 by similarity):
{context}

ORIENT: Based on these excerpts, what topics does this conversation cover? \
What kind of content is in this memory store? What aspects of the question \
might be answerable from this conversation?

Then generate 2 search cues targeting the MOST PROMISING areas — content \
that would directly help answer the question based on what you've learned \
about this conversation's topics.

Format:
ORIENTATION: <2-3 sentences about what this conversation contains>
CUE: <targeted search text>
CUE: <targeted search text>
Nothing else."""

        orient_response = self.llm_call(orient_prompt)
        orient_cues = []
        orientation = ""
        for line in orient_response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("ORIENTATION:"):
                orientation = line[12:].strip()
            elif line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    orient_cues.append(cue)

        # Phase 2: FOCUS — execute oriented cues
        for cue in orient_cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        # Phase 3: REFINE — look at what we have and fill gaps
        context = _format_segments(all_segments)
        refine_prompt = f"""\
You are refining a search over a conversation memory to answer a question.

Question: {question}

CONTEXT (orientation): {orientation}

ALL RETRIEVED SO FAR ({len(all_segments)} segments):
{context}

PREVIOUS CUES: {", ".join(orient_cues[:2])}

What's still MISSING? Generate 2 cues targeting gaps in the retrieved \
content. Use specific vocabulary from the conversation.

Format:
ASSESSMENT: <what's covered, what's missing>
CUE: <text>
CUE: <text>
Nothing else."""

        refine_response = self.llm_call(refine_prompt)
        refine_cues = []
        for line in refine_response.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    refine_cues.append(cue)

        for cue in refine_cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "context_bootstrapping",
                "orientation": orientation,
                "orient_cues": orient_cues[:2],
                "refine_cues": refine_cues[:2],
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 3: Hypothesis-Driven Search
# ===================================================================
class HypothesisDriven(AgentBase):
    """Form a hypothesis, search to test it, evaluate, revise.

    1. Form initial hypothesis about what the answer looks like
    2. Generate cues to find supporting/contradicting evidence
    3. Evaluate: does evidence support or contradict the hypothesis?
    4. If needed, revise hypothesis and search again

    Naturally handles contradictions, knowledge updates, and synthesis.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        max_revisions: int = 2,
        per_cue_k: int = 10,
    ):
        super().__init__(store, client)
        self.max_revisions = max_revisions
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        # Hop 0: initial retrieval
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}
        previous_cues: list[str] = []

        hypothesis_log: list[dict] = []

        for revision in range(self.max_revisions + 1):
            context = _format_segments(all_segments)

            prev_cue_text = ""
            if previous_cues:
                prev_cue_text = "\n\nPREVIOUS CUES TRIED:\n" + "\n".join(
                    f"- {c}" for c in previous_cues
                )

            if revision == 0:
                # Initial hypothesis formation
                prompt = f"""\
You are searching a conversation history to answer a question. Your approach: \
form a HYPOTHESIS about the answer, then search for evidence.

Question: {question}

INITIAL RETRIEVAL (top 10):
{context}

STEP 1: Based on these excerpts, form a HYPOTHESIS about what the answer \
to this question is. Be specific — include concrete details you'd expect.

STEP 2: Generate 2 search cues to find EVIDENCE for or against your \
hypothesis. Target content that would CONFIRM or CONTRADICT your hypothesis. \
Each cue should be plausible conversation content with specific vocabulary.

Format:
HYPOTHESIS: <your specific hypothesis about the answer>
EVIDENCE_NEEDED: <what would confirm or contradict this>
CUE: <search text targeting evidence>
CUE: <search text targeting evidence>
Nothing else."""
            else:
                # Revision based on evidence
                prompt = f"""\
You are testing a hypothesis about a conversation to answer a question.

Question: {question}

CURRENT HYPOTHESIS: {hypothesis_log[-1].get("hypothesis", "none")}

ALL EVIDENCE FOUND ({len(all_segments)} segments):
{context}{prev_cue_text}

EVALUATE: Does the evidence SUPPORT, CONTRADICT, or PARTIALLY SUPPORT \
your hypothesis? What specific details confirm or challenge it?

Then either:
- REVISE your hypothesis based on the evidence and generate 2 new cues
- CONFIRM the hypothesis is well-supported and STOP

Format:
EVALUATION: <SUPPORTS/CONTRADICTS/PARTIAL — specific reasoning>
REVISED_HYPOTHESIS: <updated hypothesis, or CONFIRMED if stopping>
CUE: <search text> (omit if CONFIRMED)
CUE: <search text> (omit if CONFIRMED)
Nothing else."""

            response = self.llm_call(prompt)

            # Parse
            hypothesis = ""
            evaluation = ""
            cues = []
            confirmed = False
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("HYPOTHESIS:"):
                    hypothesis = line[11:].strip()
                elif line.upper().startswith("REVISED_HYPOTHESIS:"):
                    hypothesis = line[19:].strip()
                    if "CONFIRMED" in hypothesis.upper():
                        confirmed = True
                elif line.upper().startswith("EVALUATION:"):
                    evaluation = line[11:].strip()
                elif line.startswith("CUE:"):
                    cue = line[4:].strip()
                    if cue:
                        cues.append(cue)

            hypothesis_log.append(
                {
                    "revision": revision,
                    "hypothesis": hypothesis,
                    "evaluation": evaluation,
                    "num_cues": len(cues),
                    "confirmed": confirmed,
                }
            )

            if confirmed or not cues:
                break

            # Search for evidence
            for cue in cues[:2]:
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=seen_indices,
                )
                for seg in result.segments:
                    if seg.index not in seen_indices:
                        all_segments.append(seg)
                        seen_indices.add(seg.index)
                previous_cues.append(cue)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "hypothesis_driven",
                "revisions": len(hypothesis_log),
                "hypothesis_log": hypothesis_log,
                "total_cues": len(previous_cues),
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 4: Working Memory Buffer
# ===================================================================
class WorkingMemoryBuffer(AgentBase):
    """Fixed-size buffer with selective attention and eviction.

    After each retrieval hop, the model sees new segments + current buffer
    and decides what to KEEP (max 8 slots) and what to EVICT.
    Cues are generated from the curated buffer, not the raw retrieval dump.

    Tests whether explicit curation of context improves cue quality.
    """

    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        buffer_size: int = 8,
        max_hops: int = 2,
        per_cue_k: int = 10,
    ):
        super().__init__(store, client)
        self.buffer_size = buffer_size
        self.max_hops = max_hops
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        # Hop 0: initial retrieval
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )

        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}
        # Buffer starts with top results
        buffer: list[Segment] = list(initial.segments[: self.buffer_size])
        previous_cues: list[str] = []
        hop_log: list[dict] = []

        for hop in range(self.max_hops):
            # Show buffer to model, ask for curation + cues
            buffer_text = _format_segments(buffer, max_items=self.buffer_size)

            # Show new segments not in buffer (candidates for swap)
            new_candidates = [s for s in all_segments if s not in buffer]
            new_text = ""
            if new_candidates:
                recent = sorted(new_candidates, key=lambda s: s.turn_id)[-6:]
                new_text = (
                    "\n\nNEW SEGMENTS (not in buffer, available for swap):\n"
                    + _format_segments(recent, max_items=6)
                )

            prev_cue_text = ""
            if previous_cues:
                prev_cue_text = "\n\nPREVIOUS CUES:\n" + "\n".join(
                    f"- {c}" for c in previous_cues
                )

            prompt = f"""\
You are managing a working memory buffer to answer a question from a \
conversation history. Your buffer holds the {self.buffer_size} most \
important segments for answering the question.

Question: {question}

CURRENT BUFFER ({len(buffer)} slots, max {self.buffer_size}):
{buffer_text}{new_text}{prev_cue_text}

INSTRUCTIONS:
1. ASSESS what the buffer covers and what's missing for the question.
2. If any buffer slots contain irrelevant content, EVICT them (list turn IDs).
3. Generate 2 search cues targeting MISSING information. Use specific \
vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <what's in buffer, what's missing>
EVICT: <comma-separated turn IDs to remove, or NONE>
CUE: <search text>
CUE: <search text>
Nothing else."""

            response = self.llm_call(prompt)

            # Parse
            evict_ids: set[int] = set()
            cues = []
            assessment = ""
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("ASSESSMENT:"):
                    assessment = line[11:].strip()
                elif line.upper().startswith("EVICT:"):
                    evict_text = line[6:].strip()
                    if evict_text.upper() != "NONE":
                        for part in evict_text.split(","):
                            part = part.strip()
                            try:
                                evict_ids.add(int(part))
                            except ValueError:
                                pass
                elif line.startswith("CUE:"):
                    cue = line[4:].strip()
                    if cue:
                        cues.append(cue)

            # Execute eviction
            if evict_ids:
                buffer = [s for s in buffer if s.turn_id not in evict_ids]

            hop_log.append(
                {
                    "hop": hop,
                    "assessment": assessment,
                    "evicted": sorted(evict_ids),
                    "buffer_size_after_evict": len(buffer),
                    "num_cues": len(cues),
                }
            )

            if not cues:
                break

            # Execute search
            new_this_hop: list[Segment] = []
            for cue in cues[:2]:
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=seen_indices,
                )
                for seg in result.segments:
                    if seg.index not in seen_indices:
                        all_segments.append(seg)
                        new_this_hop.append(seg)
                        seen_indices.add(seg.index)
                previous_cues.append(cue)

            # Add new segments to buffer (up to buffer_size)
            for seg in new_this_hop:
                if len(buffer) < self.buffer_size:
                    buffer.append(seg)

        # Return: buffer segments first (prioritized), then everything else
        buffer_indices = {s.index for s in buffer}
        non_buffer = [s for s in all_segments if s.index not in buffer_indices]
        ordered = list(buffer) + non_buffer

        return AgentArchResult(
            segments=ordered,
            metadata={
                "name": "working_memory_buffer",
                "buffer_size": self.buffer_size,
                "hops": len(hop_log),
                "hop_log": hop_log,
                "final_buffer_size": len(buffer),
                "total_cues": len(previous_cues),
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 5: Adaptive Strategy
# ===================================================================
class AdaptiveStrategy(AgentBase):
    """Model classifies the question and adapts its retrieval strategy.

    Step 1: Classify question type and choose strategy
    Step 2: Execute strategy-specific retrieval
    Step 3: Model decides if done or needs refinement

    The model can choose between:
    - DEEP_DRILL: few precise cues for factual lookups
    - BROAD_SWEEP: many diverse cues for synthesis/summarization
    - TEMPORAL_SCAN: cues targeting different time periods
    - EVIDENCE_HUNT: hypothesis-driven for contradictions
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        # Hop 0: initial retrieval
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        # Step 1: Classify and plan
        context = _format_segments(all_segments)
        classify_prompt = f"""\
You are planning a retrieval strategy to answer a question from a \
conversation history. Cues will be embedded and matched via cosine similarity.

Question: {question}

INITIAL RETRIEVAL (top 10):
{context}

CLASSIFY this question and choose a strategy:
- DEEP_DRILL: For factual lookups needing 1-2 precise cues
- BROAD_SWEEP: For synthesis/summarization needing 3-4 diverse cues
- TEMPORAL_SCAN: For event ordering/temporal questions needing cues \
targeting different time periods

Then generate cues appropriate to your chosen strategy. Use specific \
vocabulary from the conversation.

Format:
STRATEGY: <DEEP_DRILL or BROAD_SWEEP or TEMPORAL_SCAN>
REASONING: <why this strategy>
CUE: <text>
(repeat as appropriate: 1-2 for DEEP_DRILL, 3-4 for BROAD_SWEEP, \
2-3 for TEMPORAL_SCAN)
Nothing else."""

        response = self.llm_call(classify_prompt)

        strategy = "BROAD_SWEEP"
        reasoning = ""
        cues = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("STRATEGY:"):
                s = line[9:].strip().upper()
                if "DEEP" in s:
                    strategy = "DEEP_DRILL"
                elif "TEMPORAL" in s:
                    strategy = "TEMPORAL_SCAN"
                else:
                    strategy = "BROAD_SWEEP"
            elif line.upper().startswith("REASONING:"):
                reasoning = line[10:].strip()
            elif line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)

        # Cap cues based on strategy
        max_cues = {"DEEP_DRILL": 2, "BROAD_SWEEP": 4, "TEMPORAL_SCAN": 3}
        cues = cues[: max_cues.get(strategy, 3)]

        # Execute search
        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        # Step 2: Refine — one more round based on what was found
        context = _format_segments(all_segments)
        refine_prompt = f"""\
You are refining a {strategy} search over a conversation.

Question: {question}

ALL RETRIEVED ({len(all_segments)} segments):
{context}

PREVIOUS CUES: {", ".join(cues)}

Should you search more or stop? If searching, generate 1-2 targeted cues \
for the most critical gaps. Use specific vocabulary.

Format:
ACTION: SEARCH or STOP
CUE: <text> (if searching)
Nothing else."""

        refine_response = self.llm_call(refine_prompt)
        refine_action = "STOP"
        refine_cues = []
        for line in refine_response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("ACTION:"):
                if "SEARCH" in line.upper():
                    refine_action = "SEARCH"
            elif line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    refine_cues.append(cue)

        if refine_action == "SEARCH" and refine_cues:
            for cue in refine_cues[:2]:
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=seen_indices,
                )
                for seg in result.segments:
                    if seg.index not in seen_indices:
                        all_segments.append(seg)
                        seen_indices.add(seg.index)
                cues.append(cue)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "adaptive_strategy",
                "strategy": strategy,
                "reasoning": reasoning,
                "refined": refine_action == "SEARCH",
                "total_cues": len(cues),
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 6: v15 Faithful Reproduction (control)
# ===================================================================
class V15Control(AgentBase):
    """Faithful reproduction of v15 (self-monitoring, fixed 2 cues, 1 hop).

    This is the control to verify our evaluation framework matches
    the reference results. Uses the exact v15 prompt.
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        # Hop 0: initial retrieval
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        # Hop 1: v15 self-monitoring prompt with 2 cues
        context = _format_segments(all_segments)
        prompt = f"""\
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

        response = self.llm_call(prompt)
        cues = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "v15_control",
                "cues": cues[:2],
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 7: Focused Agentic (learns from v15 strengths)
# ===================================================================
class FocusedAgentic(AgentBase):
    """Agentic loop constrained to be focused like v15 but adaptive.

    Key insight from research: v15 with 2 cues beats v15 with 3 cues.
    More is NOT better. So this architecture:
    - Defaults to 2 cues (v15's sweet spot)
    - But allows the model to generate just 1 if confident
    - Allows a second hop ONLY if the model's assessment indicates poor coverage
    - Self-monitoring drives decisions, not fixed structure

    Difference from agentic_loop: tighter constraints, default-2 bias,
    explicit quality gate for second hop.
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        # Hop 0: initial retrieval
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}
        all_cues: list[str] = []
        hop_log: list[dict] = []

        for hop in range(2):  # Max 2 hops of cue generation
            context = _format_segments(all_segments)

            prev_cue_text = ""
            if all_cues:
                prev_cue_text = "\n\nPREVIOUS CUES (do NOT repeat):\n" + "\n".join(
                    f"- {c}" for c in all_cues
                )

            prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED SO FAR ({len(all_segments)} segments):
{context}{prev_cue_text}

First, assess the search:
- COVERAGE: What fraction of the question can be answered from retrieved content? (LOW/MEDIUM/HIGH)
- GAPS: What specific information is still missing?

Then generate search cues. Default to 2 cues, but:
- If coverage is HIGH and only a small gap remains, generate just 1 precise cue
- If coverage is LOW and the question needs diverse content, generate 2 cues
- NEVER generate more than 2 cues

Use specific vocabulary that would appear in the target turns.

Format:
COVERAGE: <LOW/MEDIUM/HIGH>
GAPS: <what's missing>
CUE: <text>
CUE: <text> (optional if coverage is HIGH)
Nothing else."""

            response = self.llm_call(prompt)

            coverage = "LOW"
            gaps = ""
            cues = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("COVERAGE:"):
                    cov_text = line[9:].strip().upper()
                    if "HIGH" in cov_text:
                        coverage = "HIGH"
                    elif "MEDIUM" in cov_text:
                        coverage = "MEDIUM"
                    else:
                        coverage = "LOW"
                elif line.upper().startswith("GAPS:"):
                    gaps = line[5:].strip()
                elif line.startswith("CUE:"):
                    cue = line[4:].strip()
                    if cue:
                        cues.append(cue)

            cues = cues[:2]  # Hard cap

            hop_log.append(
                {
                    "hop": hop,
                    "coverage": coverage,
                    "gaps": gaps,
                    "num_cues": len(cues),
                }
            )

            if not cues:
                break

            # Execute search
            for cue in cues:
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=seen_indices,
                )
                for seg in result.segments:
                    if seg.index not in seen_indices:
                        all_segments.append(seg)
                        seen_indices.add(seg.index)
                all_cues.append(cue)

            # Quality gate: only do second hop if coverage was LOW
            if hop == 0 and coverage != "LOW":
                break

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "focused_agentic",
                "hops": len(hop_log),
                "hop_log": hop_log,
                "total_cues": len(all_cues),
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 8: v15 with Stop (minimal agency)
# ===================================================================
class V15WithStop(AgentBase):
    """v15 prompt but the model can choose to generate 0 cues (STOP).

    If the assessment says the retrieval already covers the question,
    the model can output STOP instead of cues. Otherwise identical to v15.

    Tests: does giving a stop option reduce noise without losing wins?
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        context = _format_segments(all_segments)
        prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then EITHER:
- Generate 2 search cues if important content is still missing
- Output STOP if the retrieved excerpts already cover the question well

Use specific vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
OR:
ASSESSMENT: <1-2 sentence self-evaluation>
STOP
Nothing else."""

        response = self.llm_call(prompt)
        cues = []
        stopped = False
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper() == "STOP":
                stopped = True
            elif line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)

        if not stopped:
            for cue in cues[:2]:
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=seen_indices,
                )
                for seg in result.segments:
                    if seg.index not in seen_indices:
                        all_segments.append(seg)
                        seen_indices.add(seg.index)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "v15_with_stop",
                "stopped": stopped,
                "cues": cues[:2] if not stopped else [],
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 9: v15 + Conditional Second Hop
# ===================================================================
class V15ConditionalHop2(AgentBase):
    """v15 hop 1 always fires. Hop 2 fires only if model says gaps remain.

    After hop 1, show all accumulated context and ask: are there still
    critical gaps? If yes, generate 2 more cues. If no, stop.

    Tests: does a conditional second hop improve without diluting?
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        # Hop 1: exact v15 prompt
        context = _format_segments(all_segments)
        hop1_prompt = f"""\
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

        response1 = self.llm_call(hop1_prompt)
        hop1_cues = []
        for line in response1.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    hop1_cues.append(cue)

        for cue in hop1_cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        # Hop 2: conditional — ask if more is needed
        context2 = _format_segments(all_segments)
        hop2_prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context2}

PREVIOUS CUES ALREADY TRIED (do NOT repeat or paraphrase):
{chr(10).join(f"- {c}" for c in hop1_cues[:2])}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then EITHER:
- Generate 2 search cues if critical content is still missing
- Output STOP if the question is well-covered

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
OR:
ASSESSMENT: <1-2 sentence self-evaluation>
STOP
Nothing else."""

        response2 = self.llm_call(hop2_prompt)
        hop2_cues = []
        hop2_stopped = False
        for line in response2.strip().split("\n"):
            line = line.strip()
            if line.upper() == "STOP":
                hop2_stopped = True
            elif line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    hop2_cues.append(cue)

        if not hop2_stopped and hop2_cues:
            for cue in hop2_cues[:2]:
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=seen_indices,
                )
                for seg in result.segments:
                    if seg.index not in seen_indices:
                        all_segments.append(seg)
                        seen_indices.add(seg.index)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "v15_conditional_hop2",
                "hop1_cues": hop1_cues[:2],
                "hop2_stopped": hop2_stopped,
                "hop2_cues": hop2_cues[:2] if not hop2_stopped else [],
                "total_cues": len(hop1_cues[:2])
                + (len(hop2_cues[:2]) if not hop2_stopped else 0),
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 10: v15 with Variable Cue Count
# ===================================================================
class V15VariableCues(AgentBase):
    """v15 prompt but allows 1-3 cues instead of exactly 2.

    Model decides how many cues based on its assessment. Everything else
    identical to v15. Tests whether variable cue count helps.
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        context = _format_segments(all_segments)
        prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

Then generate 1-3 search cues based on your assessment:
- 1 cue if just one specific piece is missing
- 2 cues for moderate gaps (most common)
- 3 cues only if multiple distinct topics are still missing

Use specific vocabulary that would appear in the target conversation turns.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
(1-3 CUE lines)
Nothing else."""

        response = self.llm_call(prompt)
        cues = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)

        cues = cues[:3]  # Hard cap

        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "v15_variable_cues",
                "num_cues": len(cues),
                "cues": cues,
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 11: Orient-then-v15
# ===================================================================
class OrientThenV15(AgentBase):
    """Orientation phase followed by v15's exact prompt.

    Phase 1: Look at initial retrieval and summarize what the conversation
    is about (1 LLM call). This summary is prepended to the v15 prompt.
    Phase 2: v15 exact prompt but with orientation context.

    Tests: does knowing what's in the memory store improve cue quality?
    Uses 2 LLM calls total (orient + cue generation).
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        # Phase 1: Orient
        context = _format_segments(all_segments)
        orient_prompt = f"""\
Summarize in 2-3 sentences what topics this conversation covers, based on \
these excerpts. Focus on specific subjects, names, activities, and themes.

{context}

Summary:"""

        orientation = self.llm_call(orient_prompt)

        # Phase 2: v15 with orientation
        v15_prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

CONVERSATION CONTEXT: {orientation.strip()}

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

        response = self.llm_call(v15_prompt)
        cues = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "orient_then_v15",
                "orientation": orientation.strip()[:200],
                "cues": cues[:2],
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 12: Dual-Perspective Cues
# ===================================================================
class DualPerspective(AgentBase):
    """Two independent LLM calls each produce 1 cue from different perspectives.

    Call 1: "What would the USER have said?" perspective
    Call 2: "What would the ASSISTANT have said?" perspective

    v15 generates 2 cues from one call — they may be semantically close.
    Two independent calls with different prompts might produce more diverse cues.
    Same budget: 2 LLM calls, 2 cues, ~30 segments.
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}
        context = _format_segments(all_segments)

        # Perspective 1: User-side
        user_prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cue will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}

Think about what the USER (human) would have typed when discussing this \
topic. Users ask short questions, make requests, share experiences, and \
react informally. Generate a cue that sounds like a USER message related \
to the missing information.

Briefly assess what's missing, then generate exactly 1 cue.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text that sounds like a user message>
Nothing else."""

        # Perspective 2: Assistant-side
        assistant_prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cue will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context}

Think about what the ASSISTANT (AI) would have said when discussing this \
topic. Assistants give detailed explanations, provide specific information, \
and use technical vocabulary. Generate a cue that sounds like an ASSISTANT \
response containing the missing information.

Briefly assess what's missing, then generate exactly 1 cue.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text that sounds like an assistant response>
Nothing else."""

        all_cues = []
        for prompt in [user_prompt, assistant_prompt]:
            response = self.llm_call(prompt)
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("CUE:"):
                    cue = line[4:].strip()
                    if cue:
                        all_cues.append(cue)
                        break  # Only take first cue from each prompt

        for cue in all_cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "dual_perspective",
                "cues": all_cues[:2],
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 13: v15 + Reranking hop
# ===================================================================
class V15Rerank(AgentBase):
    """v15 retrieval followed by LLM-based reranking of top results.

    Hop 1: Standard v15 (10 initial + 2 cues * 10 each = ~30 segments).
    Hop 2: Show model the top ~20 segments and ask it to rank them by
    relevance. The model's ranking replaces the embedding-based ordering.

    Tests: does LLM reranking improve r@20 by putting the right segments
    in the top 20?
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        # Hop 1: v15 cue generation
        context = _format_segments(all_segments)
        cue_prompt = f"""\
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

        response = self.llm_call(cue_prompt)
        cues = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)

        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        # Hop 2: Rerank top segments
        # Show the model up to 25 segments and ask it to pick the most relevant
        candidates = sorted(all_segments, key=lambda s: s.turn_id)[:25]
        cand_text = "\n".join(
            f"[{i}] Turn {s.turn_id}: {s.text[:200]}" for i, s in enumerate(candidates)
        )

        rerank_prompt = f"""\
Given this question about a conversation, rank the following segments by \
relevance. List the MOST relevant segment numbers first.

Question: {question}

Segments:
{cand_text}

List the segment numbers in order of relevance (most relevant first). \
Include ALL segments. Just list numbers separated by commas.

RANKING:"""

        rerank_response = self.llm_call(rerank_prompt)

        # Parse ranking
        ranked_indices: list[int] = []
        seen_ranked: set[int] = set()
        for part in rerank_response.strip().split(","):
            part = part.strip().strip("[]").strip()
            try:
                idx = int(part)
                if 0 <= idx < len(candidates) and idx not in seen_ranked:
                    ranked_indices.append(idx)
                    seen_ranked.add(idx)
            except ValueError:
                pass

        # Build reranked list
        reranked: list[Segment] = [candidates[i] for i in ranked_indices]
        # Add any candidates not in ranking
        for i, seg in enumerate(candidates):
            if i not in seen_ranked:
                reranked.append(seg)
        # Add remaining segments not in candidates
        cand_indices = {s.index for s in candidates}
        remaining = [s for s in all_segments if s.index not in cand_indices]
        reranked.extend(remaining)

        return AgentArchResult(
            segments=reranked,
            metadata={
                "name": "v15_rerank",
                "cues": cues[:2],
                "reranked": len(ranked_indices),
                "total_segments": len(all_segments),
            },
        )


# ===================================================================
# Architecture 14: v15 Conditional Hop2 + Mandatory Two Cues
# ===================================================================
class V15ConditionalForced2(AgentBase):
    """Like v15_conditional_hop2, but the second hop always generates 2 cues
    (no STOP option in hop 2). However, hop 2 only fires if hop 1's
    assessment indicates the coverage is incomplete.

    The difference from plain v15 2-hop: the second hop is conditional.
    The difference from v15_conditional_hop2: no STOP option in hop 2.
    """

    def __init__(
        self, store: SegmentStore, client: OpenAI | None = None, per_cue_k: int = 10
    ):
        super().__init__(store, client)
        self.per_cue_k = per_cue_k

    def retrieve(self, question: str, conversation_id: str) -> AgentArchResult:
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=10, conversation_id=conversation_id
        )
        all_segments: list[Segment] = list(initial.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        # Hop 1: exact v15
        context = _format_segments(all_segments)
        hop1_prompt = f"""\
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

        response1 = self.llm_call(hop1_prompt)
        hop1_cues = []
        hop1_assessment = ""
        for line in response1.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("ASSESSMENT:"):
                hop1_assessment = line[11:].strip()
            elif line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    hop1_cues.append(cue)

        for cue in hop1_cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.per_cue_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        # Decide if hop 2 is needed based on assessment keywords
        # Look for signals of incomplete coverage
        assessment_lower = hop1_assessment.lower()
        needs_hop2 = any(
            kw in assessment_lower
            for kw in [
                "missing",
                "not found",
                "no mention",
                "doesn't cover",
                "hasn't been",
                "need to find",
                "pivot",
                "different",
                "incomplete",
                "lacking",
                "gap",
                "absent",
                "still need",
                "not yet",
                "limited",
                "insufficient",
            ]
        )

        hop2_cues: list[str] = []
        if needs_hop2:
            context2 = _format_segments(all_segments)
            hop2_prompt = f"""\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{context2}

PREVIOUS CUES ALREADY TRIED (do NOT repeat or paraphrase):
{chr(10).join(f"- {c}" for c in hop1_cues[:2])}

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

            response2 = self.llm_call(hop2_prompt)
            for line in response2.strip().split("\n"):
                line = line.strip()
                if line.startswith("CUE:"):
                    cue = line[4:].strip()
                    if cue:
                        hop2_cues.append(cue)

            for cue in hop2_cues[:2]:
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.per_cue_k,
                    conversation_id=conversation_id,
                    exclude_indices=seen_indices,
                )
                for seg in result.segments:
                    if seg.index not in seen_indices:
                        all_segments.append(seg)
                        seen_indices.add(seg.index)

        return AgentArchResult(
            segments=all_segments,
            metadata={
                "name": "v15_conditional_forced2",
                "hop1_assessment": hop1_assessment,
                "hop1_cues": hop1_cues[:2],
                "needs_hop2": needs_hop2,
                "hop2_cues": hop2_cues[:2],
                "total_cues": len(hop1_cues[:2]) + len(hop2_cues[:2]),
                "total_segments": len(all_segments),
            },
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
AGENT_ARCHITECTURES: dict[str, type[AgentBase]] = {
    "agentic_loop": AgenticLoop,
    "context_bootstrapping": ContextBootstrapping,
    "hypothesis_driven": HypothesisDriven,
    "working_memory_buffer": WorkingMemoryBuffer,
    "adaptive_strategy": AdaptiveStrategy,
    "v15_control": V15Control,
    "focused_agentic": FocusedAgentic,
    "v15_with_stop": V15WithStop,
    "v15_conditional_hop2": V15ConditionalHop2,
    "v15_variable_cues": V15VariableCues,
    "orient_then_v15": OrientThenV15,
    "dual_perspective": DualPerspective,
    "v15_rerank": V15Rerank,
    "v15_conditional_forced2": V15ConditionalForced2,
}
