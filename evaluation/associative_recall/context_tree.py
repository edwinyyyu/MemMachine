"""DFS context tree for reasoning-integrated retrieval.

Architectures:
  A. decompose_then_retrieve: structured decomposition → v15-style retrieval per subtask
  B. iterative_deepen: single-thread DFS where each level retrieves and the model
     decides whether to go deeper or pop, with forced retrieval at every level
  C. flat_multi_cue: baseline — v15 with sub-question cues instead of free-form cues
     (controls for whether decomposition helps vs just having more cues)
  D. rerank_pool: run decompose_then_retrieve, then re-rank the full pool by cosine
     similarity to the original question (fixes ordering)
  E. interleaved: retrieve first, show results to decomposer, decompose based on gaps,
     retrieve per sub-question, assess per branch
  F. retrieve_then_decompose: v15-style first hop, then decompose based on findings,
     retrieve per gap
  G. branch_assess: decompose_then_retrieve with per-branch assessment loop
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from associative_recall import (
    CACHE_DIR,
    EMBED_MODEL,
    AssociativeRecallEngine,
    EmbeddingCache,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"


@dataclass
class TreeResult:
    all_segments: list[Segment]
    total_retrieved: int
    embed_calls: int
    llm_calls: int
    metadata: dict = field(default_factory=dict)


class TreeCache:
    """Separate LLM cache for context tree experiments."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "tree_llm_cache.json"
        self._cache: dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)
        self._dirty = False

    def _key(self, model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(self._key(model, prompt))

    def put(self, model: str, prompt: str, response: str) -> None:
        self._cache[self._key(model, prompt)] = response
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        tmp.replace(self.cache_file)
        self._dirty = False


class TreeEngine:
    def __init__(
        self,
        store: SegmentStore,
        client: OpenAI | None = None,
        model: str = MODEL,
        top_k: int = 10,
        neighbor_radius: int = 1,
    ):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.model = model
        self.top_k = top_k
        self.neighbor_radius = neighbor_radius
        self.embedding_cache = EmbeddingCache()
        self.llm_cache = TreeCache()
        self.embed_calls = 0
        self.llm_calls = 0
        self._total_embed_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache.put(text, embedding)
        self.embed_calls += 1
        self._total_embed_calls += 1
        return embedding

    def _retrieve(
        self,
        query: str,
        conversation_id: str,
        exclude_indices: set[int],
    ) -> list[Segment]:
        query_emb = self.embed_text(query)
        result = self.store.search(
            query_emb,
            top_k=self.top_k,
            conversation_id=conversation_id,
            exclude_indices=exclude_indices,
        )
        segments = list(result.segments)
        exclude_indices.update(s.index for s in segments)

        if self.neighbor_radius > 0:
            neighbor_segs = []
            for seg in segments:
                neighbors = self.store.get_neighbors(
                    seg,
                    radius=self.neighbor_radius,
                    exclude_indices=exclude_indices,
                )
                for n in neighbors:
                    neighbor_segs.append(n)
                    exclude_indices.add(n.index)
            segments.extend(neighbor_segs)

        return segments

    def _llm_call(self, prompt: str) -> str:
        cached = self.llm_cache.get(self.model, prompt)
        if cached is not None:
            return cached
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=800,
        )
        text = response.choices[0].message.content or ""
        self.llm_cache.put(self.model, prompt, text)
        self.llm_calls += 1
        return text

    def _format_segments(self, segments: list[Segment], limit: int = 10) -> str:
        if not segments:
            return "(none)"
        sorted_segs = sorted(segments, key=lambda s: s.turn_id)[:limit]
        lines = []
        for s in sorted_segs:
            lines.append(f"[Turn {s.turn_id}, {s.role}]: {s.text[:200]}")
        return "\n".join(lines)

    def save_caches(self) -> None:
        if self._total_embed_calls > 0:
            try:
                self.embedding_cache.save()
            except OSError:
                pass  # tolerate transient file system errors
        self.llm_cache.save()

    # ------------------------------------------------------------------
    # Architecture A: decompose_then_retrieve
    #
    # Phase 1: decompose question into 2-3 sub-questions (1 LLM call)
    # Phase 2: for each sub-question, retrieve with the sub-question as query,
    #          then generate 1 v15-style follow-up cue, then retrieve again
    # Phase 3: collect all segments
    # ------------------------------------------------------------------

    def decompose_then_retrieve(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        exclude = set()
        all_segments = []

        # Phase 1: decompose
        sub_questions = self._decompose(question)
        if verbose:
            print(f"  Decomposed into {len(sub_questions)} sub-questions:")
            for sq in sub_questions:
                print(f"    - {sq[:80]}")

        # Phase 2: retrieve per sub-question
        branch_summaries = []
        for i, sq in enumerate(sub_questions):
            # Initial retrieval with sub-question
            segs = self._retrieve(sq, conversation_id, exclude)
            all_segments.extend(segs)
            if verbose:
                print(f"  Branch {i}: +{len(segs)} from '{sq[:60]}...'")

            # v15-style follow-up: assess what we found, generate 1 cue
            cue = self._generate_followup_cue(question, sq, segs)
            if cue:
                more_segs = self._retrieve(cue, conversation_id, exclude)
                all_segments.extend(more_segs)
                if verbose:
                    print(f"    Follow-up cue: '{cue[:60]}...' → +{len(more_segs)}")

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "decompose_then_retrieve",
                "sub_questions": sub_questions,
                "elapsed": round(elapsed, 2),
            },
        )

    def _decompose(self, question: str) -> list[str]:
        prompt = DECOMPOSE_PROMPT.format(question=question)
        text = self._llm_call(prompt)
        sub_questions = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("SUB:"):
                sq = line[4:].strip()
                if sq:
                    sub_questions.append(sq)
        # Fallback: if parsing fails, use the original question
        if not sub_questions:
            sub_questions = [question]
        return sub_questions[:4]  # Cap at 4

    def _generate_followup_cue(
        self,
        question: str,
        sub_question: str,
        retrieved: list[Segment],
    ) -> str | None:
        context = self._format_segments(retrieved)
        prompt = FOLLOWUP_CUE_PROMPT.format(
            question=question,
            sub_question=sub_question,
            retrieved_context=context,
        )
        text = self._llm_call(prompt)
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    return cue
        return None

    # ------------------------------------------------------------------
    # Architecture B: iterative_deepen
    #
    # DFS with forced retrieval at every level. At each node:
    #   1. Retrieve using node context (automatic)
    #   2. Model reviews what was found
    #   3. Model either PUSHes deeper (with a narrower sub-question) or POPs
    # Max depth 3, single thread. The model's assessment at each level
    # determines whether to go deeper — like v15's self-monitoring but
    # applied to tree structure.
    # ------------------------------------------------------------------

    def iterative_deepen(
        self,
        question: str,
        conversation_id: str,
        max_depth: int = 3,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        exclude = set()
        all_segments = []
        step_log = []

        # Stack: list of (description, depth, retrieved_at_this_level)
        # Start with the question itself at depth 0
        stack = [(question, 0)]
        context_path = []  # descriptions from root to current

        while stack:
            description, depth = stack.pop()
            context_path = context_path[:depth]
            context_path.append(description)

            # Forced retrieval at this level
            # Query = the current description (narrowed by depth)
            query = description
            segs = self._retrieve(query, conversation_id, exclude)
            all_segments.extend(segs)

            if verbose:
                indent = "  " * depth
                print(
                    f"  {indent}[d{depth}] '{description[:60]}...' → +{len(segs)} segs"
                )

            step_log.append(
                {
                    "depth": depth,
                    "description": description[:100],
                    "num_retrieved": len(segs),
                }
            )

            if depth >= max_depth:
                # Can't go deeper — done with this branch
                continue

            # Ask the model: go deeper or done?
            decision = self._deepen_or_pop(
                question,
                context_path,
                segs,
                all_segments,
                depth,
                max_depth,
            )

            if decision["action"] == "PUSH" and decision.get("sub_questions"):
                # Push sub-questions onto stack (reverse order for DFS)
                for sq in reversed(decision["sub_questions"]):
                    stack.append((sq, depth + 1))
                    if verbose:
                        indent = "  " * (depth + 1)
                        print(f"  {indent}→ queued: '{sq[:60]}...'")

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "iterative_deepen",
                "step_log": step_log,
                "elapsed": round(elapsed, 2),
            },
        )

    def _deepen_or_pop(
        self,
        question: str,
        context_path: list[str],
        current_segs: list[Segment],
        all_segs: list[Segment],
        depth: int,
        max_depth: int,
    ) -> dict:
        context = self._format_segments(current_segs, limit=8)
        all_context = self._format_segments(all_segs, limit=6)
        path_str = " → ".join(context_path)

        prompt = DEEPEN_PROMPT.format(
            question=question,
            context_path=path_str,
            current_retrieved=context,
            all_retrieved=all_context,
            total_retrieved=len(all_segs),
            depth=depth,
            max_depth=max_depth,
        )
        text = self._llm_call(prompt)

        # Parse response
        if "ACTION: POP" in text.upper() or "ACTION:POP" in text.upper():
            return {"action": "POP"}

        sub_questions = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("SUB:"):
                sq = line[4:].strip()
                if sq:
                    sub_questions.append(sq)

        if sub_questions:
            return {"action": "PUSH", "sub_questions": sub_questions[:3]}

        # Default: POP (don't go deeper if we can't parse)
        return {"action": "POP"}

    # ------------------------------------------------------------------
    # Architecture C: flat_multi_cue (control)
    #
    # Same as decompose_then_retrieve but WITHOUT the tree structure.
    # Decompose into sub-questions, then use them ALL as cues for a single
    # flat retrieval (like v15 with question-derived cues instead of
    # free-form cues). This isolates whether decomposition helps the cue
    # quality vs whether the tree structure helps the retrieval.
    # ------------------------------------------------------------------

    def flat_multi_cue(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        exclude = set()
        all_segments = []

        # Initial retrieval with original question
        segs = self._retrieve(question, conversation_id, exclude)
        all_segments.extend(segs)
        if verbose:
            print(f"  Initial: +{len(segs)} from question")

        # Decompose into sub-questions (same as arch A)
        sub_questions = self._decompose(question)
        if verbose:
            print(f"  Decomposed into {len(sub_questions)} cues")

        # Use each sub-question as a cue (flat, no tree)
        for i, sq in enumerate(sub_questions):
            segs = self._retrieve(sq, conversation_id, exclude)
            all_segments.extend(segs)
            if verbose:
                print(f"  Cue {i}: '{sq[:60]}...' → +{len(segs)}")

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "flat_multi_cue",
                "sub_questions": sub_questions,
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture D: rerank_pool
    #
    # Run decompose_then_retrieve to collect a large pool of segments,
    # then re-rank the entire pool by cosine similarity to the ORIGINAL
    # question. This tests whether the +30.6pp at r@100 can be lifted
    # to r@20 by fixing the ordering.
    # ------------------------------------------------------------------

    def rerank_pool(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        # Run decompose_then_retrieve to collect the pool
        inner = self.decompose_then_retrieve(question, conversation_id, verbose=verbose)
        pool = inner.all_segments
        if not pool:
            return inner

        # Re-rank entire pool by cosine similarity to original question
        q_emb = self.embed_text(question)
        q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)

        # Get embeddings for each segment from the store
        scored = []
        for seg in pool:
            seg_emb = self.store.normalized_embeddings[seg.index]
            score = float(np.dot(q_norm, seg_emb))
            scored.append((score, seg))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [seg for _, seg in scored]

        elapsed = inner.metadata.get("elapsed", 0)
        return TreeResult(
            all_segments=reranked,
            total_retrieved=len(reranked),
            embed_calls=inner.embed_calls,
            llm_calls=inner.llm_calls,
            metadata={
                "architecture": "rerank_pool",
                "inner_architecture": "decompose_then_retrieve",
                "sub_questions": inner.metadata.get("sub_questions", []),
                "elapsed": elapsed,
            },
        )

    # ------------------------------------------------------------------
    # Architecture E: interleaved
    #
    # Phase 1: Retrieve with original question (see what's there)
    # Phase 2: Show retrieved to decomposer — decompose based on GAPS
    # Phase 3: Retrieve per sub-question
    # Phase 4: Per-branch assessment — generate one more cue if needed
    # ------------------------------------------------------------------

    def interleaved(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        exclude = set()
        all_segments = []

        # Phase 1: Initial retrieval with original question
        initial_segs = self._retrieve(question, conversation_id, exclude)
        all_segments.extend(initial_segs)
        if verbose:
            print(f"  Phase 1: +{len(initial_segs)} from original question")

        # Phase 2: Grounded decomposition — show what we found, ask for gaps
        sub_questions = self._grounded_decompose(question, initial_segs)
        if verbose:
            print(
                f"  Phase 2: decomposed into {len(sub_questions)} gap-based sub-questions:"
            )
            for sq in sub_questions:
                print(f"    - {sq[:80]}")

        # Phase 3: Retrieve per sub-question + Phase 4: per-branch assessment
        for i, sq in enumerate(sub_questions):
            branch_segs = self._retrieve(sq, conversation_id, exclude)
            all_segments.extend(branch_segs)
            if verbose:
                print(f"  Branch {i}: +{len(branch_segs)} from '{sq[:60]}...'")

            # Per-branch assessment: is this branch done?
            followup_cue = self._branch_assess(question, sq, branch_segs)
            if followup_cue:
                more_segs = self._retrieve(followup_cue, conversation_id, exclude)
                all_segments.extend(more_segs)
                if verbose:
                    print(
                        f"    Assessment cue: '{followup_cue[:60]}...' → +{len(more_segs)}"
                    )

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "interleaved",
                "sub_questions": sub_questions,
                "initial_retrieved": len(initial_segs),
                "elapsed": round(elapsed, 2),
            },
        )

    def _grounded_decompose(
        self, question: str, found_segments: list[Segment]
    ) -> list[str]:
        context = self._format_segments(found_segments, limit=10)
        prompt = GROUNDED_DECOMPOSE_PROMPT.format(
            question=question,
            retrieved_context=context,
        )
        text = self._llm_call(prompt)
        sub_questions = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("GAP:"):
                sq = line[4:].strip()
                if sq:
                    sub_questions.append(sq)
        if not sub_questions:
            sub_questions = [question]
        return sub_questions[:4]

    def _branch_assess(
        self,
        question: str,
        sub_question: str,
        retrieved: list[Segment],
    ) -> str | None:
        context = self._format_segments(retrieved, limit=8)
        prompt = BRANCH_ASSESS_PROMPT.format(
            question=question,
            sub_question=sub_question,
            retrieved_context=context,
        )
        text = self._llm_call(prompt)
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    return cue
        return None

    # ------------------------------------------------------------------
    # Architecture F: retrieve_then_decompose (v15 hybrid)
    #
    # Phase 1: v15-style — question retrieval, then self-monitoring cue
    # Phase 2: Decompose based on what was found, identify gaps
    # Phase 3: Retrieve per gap
    # ------------------------------------------------------------------

    def retrieve_then_decompose(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        exclude = set()
        all_segments = []

        # Phase 1a: Initial retrieval with original question
        initial_segs = self._retrieve(question, conversation_id, exclude)
        all_segments.extend(initial_segs)
        if verbose:
            print(f"  Phase 1a: +{len(initial_segs)} from question")

        # Phase 1b: v15-style self-monitoring cue
        v15_cue = self._v15_style_cue(question, initial_segs)
        if v15_cue:
            v15_segs = self._retrieve(v15_cue, conversation_id, exclude)
            all_segments.extend(v15_segs)
            if verbose:
                print(f"  Phase 1b: v15 cue '{v15_cue[:60]}...' → +{len(v15_segs)}")
        else:
            v15_segs = []

        # Phase 2: Grounded decomposition based on everything found so far
        found_so_far = initial_segs + v15_segs
        sub_questions = self._grounded_decompose(question, found_so_far)
        if verbose:
            print(f"  Phase 2: {len(sub_questions)} gap sub-questions:")
            for sq in sub_questions:
                print(f"    - {sq[:80]}")

        # Phase 3: Retrieve per gap
        for i, sq in enumerate(sub_questions):
            gap_segs = self._retrieve(sq, conversation_id, exclude)
            all_segments.extend(gap_segs)
            if verbose:
                print(f"  Gap {i}: +{len(gap_segs)} from '{sq[:60]}...'")

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "retrieve_then_decompose",
                "v15_cue": v15_cue,
                "sub_questions": sub_questions,
                "elapsed": round(elapsed, 2),
            },
        )

    def _v15_style_cue(self, question: str, retrieved: list[Segment]) -> str | None:
        context = self._format_segments(retrieved, limit=10)
        prompt = V15_STYLE_CUE_PROMPT.format(
            question=question,
            retrieved_context=context,
        )
        text = self._llm_call(prompt)
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    return cue
        return None

    # ------------------------------------------------------------------
    # Architecture G: branch_assess (decompose + per-branch assessment)
    #
    # Same as decompose_then_retrieve but with v15-style self-monitoring
    # applied PER BRANCH: after each branch retrieves, show the model
    # what it found and ask if it needs one more cue.
    # ------------------------------------------------------------------

    def branch_assess(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        exclude = set()
        all_segments = []

        sub_questions = self._decompose(question)
        if verbose:
            print(f"  Decomposed into {len(sub_questions)} sub-questions:")
            for sq in sub_questions:
                print(f"    - {sq[:80]}")

        for i, sq in enumerate(sub_questions):
            # Initial retrieval
            segs = self._retrieve(sq, conversation_id, exclude)
            all_segments.extend(segs)
            if verbose:
                print(f"  Branch {i}: +{len(segs)} from '{sq[:60]}...'")

            # v15-style follow-up cue
            cue = self._generate_followup_cue(question, sq, segs)
            if cue:
                more_segs = self._retrieve(cue, conversation_id, exclude)
                all_segments.extend(more_segs)
                if verbose:
                    print(f"    Follow-up: '{cue[:60]}...' → +{len(more_segs)}")

            # Per-branch assessment: is the branch done?
            branch_total = segs + (more_segs if cue else [])
            assess_cue = self._branch_assess(question, sq, branch_total)
            if assess_cue:
                assess_segs = self._retrieve(assess_cue, conversation_id, exclude)
                all_segments.extend(assess_segs)
                if verbose:
                    print(
                        f"    Assessment: '{assess_cue[:60]}...' → +{len(assess_segs)}"
                    )

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "branch_assess",
                "sub_questions": sub_questions,
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture E2: interleaved_prioritized
    #
    # Same as interleaved but keeps initial retrieval segments at the
    # front of the result list, appending gap-filling segments after.
    # Tests whether the losses at r@20 are caused by gap-filling
    # segments displacing good first-hop results.
    # ------------------------------------------------------------------

    def interleaved_prioritized(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        exclude = set()

        # Phase 1: Initial retrieval with original question (these get priority)
        initial_segs = self._retrieve(question, conversation_id, exclude)
        if verbose:
            print(f"  Phase 1: +{len(initial_segs)} from original question (priority)")

        # Phase 2: Grounded decomposition
        sub_questions = self._grounded_decompose(question, initial_segs)
        if verbose:
            print(f"  Phase 2: {len(sub_questions)} gap sub-questions")

        # Phase 3+4: Retrieve per sub-question + assessment
        gap_segments = []
        for i, sq in enumerate(sub_questions):
            branch_segs = self._retrieve(sq, conversation_id, exclude)
            gap_segments.extend(branch_segs)
            if verbose:
                print(f"  Branch {i}: +{len(branch_segs)} from '{sq[:60]}...'")

            followup_cue = self._branch_assess(question, sq, branch_segs)
            if followup_cue:
                more_segs = self._retrieve(followup_cue, conversation_id, exclude)
                gap_segments.extend(more_segs)
                if verbose:
                    print(
                        f"    Assessment cue: '{followup_cue[:60]}...' → +{len(more_segs)}"
                    )

        # Priority ordering: initial segments first, then gap-filling
        all_segments = initial_segs + gap_segments

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "interleaved_prioritized",
                "sub_questions": sub_questions,
                "initial_count": len(initial_segs),
                "gap_count": len(gap_segments),
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture F2v: v15_then_gaps
    #
    # v15-style 2-hop first (question + 1 self-monitoring cue), then
    # grounded decomposition for gaps, with initial 2 hops prioritized.
    # ------------------------------------------------------------------

    def v15_then_gaps(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        exclude = set()

        # Phase 1a: Initial retrieval (v15 hop 0)
        initial_segs = self._retrieve(question, conversation_id, exclude)
        if verbose:
            print(f"  Hop 0: +{len(initial_segs)} from question")

        # Phase 1b: v15-style self-monitoring cue (hop 1)
        v15_cue = self._v15_style_cue(question, initial_segs)
        hop1_segs = []
        if v15_cue:
            hop1_segs = self._retrieve(v15_cue, conversation_id, exclude)
            if verbose:
                print(f"  Hop 1: v15 cue '{v15_cue[:60]}...' → +{len(hop1_segs)}")

        # Priority pool = hop 0 + hop 1
        priority_segs = initial_segs + hop1_segs

        # Phase 2: Grounded decomposition for gaps
        sub_questions = self._grounded_decompose(question, priority_segs)
        if verbose:
            print(f"  Gaps: {len(sub_questions)} sub-questions")

        # Phase 3: Gap retrieval
        gap_segs = []
        for i, sq in enumerate(sub_questions):
            branch = self._retrieve(sq, conversation_id, exclude)
            gap_segs.extend(branch)
            if verbose:
                print(f"  Gap {i}: +{len(branch)} from '{sq[:60]}...'")

        # Priority ordering: v15 hops first, then gaps
        all_segments = priority_segs + gap_segs

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "v15_then_gaps",
                "v15_cue": v15_cue,
                "sub_questions": sub_questions,
                "priority_count": len(priority_segs),
                "gap_count": len(gap_segs),
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture H: v15_plus_tree
    #
    # Faithful v15 reproduction (hop 0 + hop 1 with 2 self-monitoring
    # cues) as the priority block, then grounded decomposition for gaps.
    #
    # Key differences from v15_then_gaps:
    # - Uses v15's EXACT retrieval pattern (no neighbor expansion at hop 0)
    # - 2 cues at hop 1 (matching v15's num_cues=2)
    # - Neighbor expansion only at hop 1
    # - Gap-filling appended after the v15 block
    # ------------------------------------------------------------------

    def v15_plus_tree(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        exclude = set()

        # ---- V15 HOP 0: question → top-10, no neighbors ----
        q_emb = self.embed_text(question)
        hop0_result = self.store.search(
            q_emb,
            top_k=self.top_k,
            conversation_id=conversation_id,
        )
        hop0_segs = list(hop0_result.segments)
        exclude.update(s.index for s in hop0_segs)
        if verbose:
            print(f"  Hop 0: +{len(hop0_segs)} from question (no neighbors)")

        # ---- V15 HOP 1: self-monitoring → 2 cues, each top-10 + neighbors ----
        context = self._format_segments(hop0_segs, limit=12)
        prompt = V15_STYLE_2CUE_PROMPT.format(
            question=question,
            retrieved_context=context,
        )
        text = self._llm_call(prompt)
        cues = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)
        cues = cues[:2]  # v15 uses exactly 2 cues

        hop1_segs = []
        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.top_k,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    hop1_segs.append(seg)
                    exclude.add(seg.index)

        # Neighbor expansion for hop 1 segments
        if self.neighbor_radius > 0:
            neighbors = []
            for seg in hop1_segs:
                nbrs = self.store.get_neighbors(
                    seg,
                    radius=self.neighbor_radius,
                    exclude_indices=exclude,
                )
                for n in nbrs:
                    neighbors.append(n)
                    exclude.add(n.index)
            hop1_segs.extend(neighbors)

        if verbose:
            print(f"  Hop 1: {len(cues)} cues → +{len(hop1_segs)} (with neighbors)")
            for c in cues:
                print(f"    CUE: {c[:70]}")

        # Priority block = hop 0 + hop 1 (v15-style)
        v15_block = hop0_segs + hop1_segs

        # ---- GAP FILLING: grounded decomposition ----
        sub_questions = self._grounded_decompose(question, v15_block)
        if verbose:
            print(f"  Gaps: {len(sub_questions)} sub-questions")

        gap_segs = []
        for i, sq in enumerate(sub_questions):
            branch = self._retrieve(sq, conversation_id, exclude)
            gap_segs.extend(branch)
            if verbose:
                print(f"  Gap {i}: +{len(branch)} from '{sq[:60]}...'")

        all_segments = v15_block + gap_segs

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "v15_plus_tree",
                "cues": cues,
                "sub_questions": sub_questions,
                "v15_count": len(v15_block),
                "gap_count": len(gap_segs),
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture I: actual_v15_plus_gaps
    #
    # Uses the ACTUAL v15 AssociativeRecallEngine for the first 2 hops
    # (exactly matching the +33.9pp result), then grounded decomposition
    # for gap-filling. The v15 segments get priority ordering.
    # ------------------------------------------------------------------

    def actual_v15_plus_gaps(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()

        # Run actual v15 engine
        v15_engine = AssociativeRecallEngine(
            store=self.store,
            client=self.client,
            cue_model=self.model,
            prompt_version="v15",
            max_hops=1,
            top_k_per_hop=10,
            num_cues=2,
            neighbor_radius=1,
        )
        v15_result = v15_engine.associative_retrieve(
            question,
            conversation_id,
            top_k_initial=10,
        )
        v15_segs = v15_result.all_retrieved_segments
        v15_engine.save_caches()

        # Count LLM/embed calls from v15 (approximate: 1 LLM for cue gen, embeds for cues)
        self.llm_calls += 1  # cue generation
        # embed calls: question + 2 cues = up to 3, but some cached
        exclude = {s.index for s in v15_segs}

        if verbose:
            print(f"  V15 block: {len(v15_segs)} segments from actual v15 engine")

        # Gap-filling via grounded decomposition
        sub_questions = self._grounded_decompose(question, v15_segs)
        if verbose:
            print(f"  Gaps: {len(sub_questions)} sub-questions")

        gap_segs = []
        for i, sq in enumerate(sub_questions):
            branch = self._retrieve(sq, conversation_id, exclude)
            gap_segs.extend(branch)
            if verbose:
                print(f"  Gap {i}: +{len(branch)} from '{sq[:60]}...'")

        all_segments = v15_segs + gap_segs

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "actual_v15_plus_gaps",
                "v15_count": len(v15_segs),
                "gap_count": len(gap_segs),
                "sub_questions": sub_questions,
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture J: actual_v15_plus_gaps_reranked
    #
    # Same as actual_v15_plus_gaps but re-rank the v15 block internally
    # by cosine to original question (so the v15 results are in the best
    # order) and then append gap segments.
    # ------------------------------------------------------------------

    def actual_v15_plus_gaps_reranked(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        inner = self.actual_v15_plus_gaps(question, conversation_id, verbose=verbose)
        v15_count = inner.metadata.get("v15_count", 0)

        if v15_count == 0 or not inner.all_segments:
            return inner

        v15_segs = inner.all_segments[:v15_count]
        gap_segs = inner.all_segments[v15_count:]

        # Re-rank v15 block by cosine to original question
        q_emb = self.embed_text(question)
        q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)

        scored = []
        for seg in v15_segs:
            seg_emb = self.store.normalized_embeddings[seg.index]
            score = float(np.dot(q_norm, seg_emb))
            scored.append((score, seg))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked_v15 = [seg for _, seg in scored]

        all_segments = reranked_v15 + gap_segs

        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=inner.embed_calls,
            llm_calls=inner.llm_calls,
            metadata={
                **inner.metadata,
                "architecture": "actual_v15_plus_gaps_reranked",
            },
        )

    # ------------------------------------------------------------------
    # Architecture K: actual_v15_control
    #
    # Just the actual v15 engine with no gap-filling. This is the
    # control to verify we can reproduce the +33.9pp baseline.
    # ------------------------------------------------------------------

    def actual_v15_control(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()

        v15_engine = AssociativeRecallEngine(
            store=self.store,
            client=self.client,
            cue_model=self.model,
            prompt_version="v15",
            max_hops=1,
            top_k_per_hop=10,
            num_cues=2,
            neighbor_radius=1,
        )
        v15_result = v15_engine.associative_retrieve(
            question,
            conversation_id,
            top_k_initial=10,
        )
        v15_segs = v15_result.all_retrieved_segments
        v15_engine.save_caches()

        self.llm_calls += 1

        if verbose:
            print(f"  V15 control: {len(v15_segs)} segments")

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=v15_segs,
            total_retrieved=len(v15_segs),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "actual_v15_control",
                "v15_count": len(v15_segs),
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture L: v15_targeted_second_hop
    #
    # Run actual v15 (hop 0 + hop 1 with 2 cues), then a targeted
    # second hop: show ALL accumulated v15 segments and ask for 2 more
    # cues specifically targeting MISSING information. The second hop
    # uses the same v15-style prompt but with richer context.
    # ------------------------------------------------------------------

    def v15_targeted_second_hop(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()

        # Run actual v15 (hop 0 + hop 1)
        v15_engine = AssociativeRecallEngine(
            store=self.store,
            client=self.client,
            cue_model=self.model,
            prompt_version="v15",
            max_hops=1,
            top_k_per_hop=10,
            num_cues=2,
            neighbor_radius=1,
        )
        v15_result = v15_engine.associative_retrieve(
            question,
            conversation_id,
            top_k_initial=10,
        )
        v15_segs = v15_result.all_retrieved_segments
        v15_engine.save_caches()
        exclude = {s.index for s in v15_segs}

        if verbose:
            print(f"  V15: {len(v15_segs)} segments")

        # Targeted second hop: show all v15 segments, ask for cues
        # about what's MISSING. Use a focused prompt.
        context = self._format_segments(v15_segs, limit=16)
        prompt = TARGETED_SECOND_HOP_PROMPT.format(
            question=question,
            retrieved_context=context,
            num_retrieved=len(v15_segs),
        )
        text = self._llm_call(prompt)
        cues = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)
        cues = cues[:2]

        hop2_segs = []
        for cue in cues:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb,
                top_k=self.top_k,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            for seg in result.segments:
                if seg.index not in exclude:
                    hop2_segs.append(seg)
                    exclude.add(seg.index)

        # Neighbor expansion for hop 2
        if self.neighbor_radius > 0:
            neighbors = []
            for seg in hop2_segs:
                nbrs = self.store.get_neighbors(
                    seg,
                    radius=self.neighbor_radius,
                    exclude_indices=exclude,
                )
                for n in nbrs:
                    neighbors.append(n)
                    exclude.add(n.index)
            hop2_segs.extend(neighbors)

        if verbose:
            print(f"  Hop 2: {len(cues)} cues → +{len(hop2_segs)} segments")
            for c in cues:
                print(f"    CUE: {c[:70]}")

        all_segments = v15_segs + hop2_segs

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "v15_targeted_second_hop",
                "v15_count": len(v15_segs),
                "hop2_cues": cues,
                "hop2_count": len(hop2_segs),
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture M: v15_multi_sub
    #
    # Run v15 for the original question, then decompose into 2-3
    # sub-questions and run a SEPARATE v15-style cue generation for
    # each sub-question. This tests whether sub-question-specific cues
    # find different segments than the original question cues.
    # ------------------------------------------------------------------

    def v15_multi_sub(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()

        # Run actual v15 for original question
        v15_engine = AssociativeRecallEngine(
            store=self.store,
            client=self.client,
            cue_model=self.model,
            prompt_version="v15",
            max_hops=1,
            top_k_per_hop=10,
            num_cues=2,
            neighbor_radius=1,
        )
        v15_result = v15_engine.associative_retrieve(
            question,
            conversation_id,
            top_k_initial=10,
        )
        v15_segs = v15_result.all_retrieved_segments
        v15_engine.save_caches()
        exclude = {s.index for s in v15_segs}

        if verbose:
            print(f"  V15 main: {len(v15_segs)} segments")

        # Decompose into sub-questions
        sub_questions = self._decompose(question)
        if verbose:
            print(f"  Sub-questions: {len(sub_questions)}")

        # For each sub-question, generate 1 focused cue and retrieve
        sub_segs = []
        for i, sq in enumerate(sub_questions):
            # Retrieve with sub-question directly
            sq_segs = self._retrieve(sq, conversation_id, exclude)
            sub_segs.extend(sq_segs)
            if verbose:
                print(f"  Sub {i}: +{len(sq_segs)} from '{sq[:50]}...'")

        all_segments = v15_segs + sub_segs

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "v15_multi_sub",
                "v15_count": len(v15_segs),
                "sub_questions": sub_questions,
                "sub_count": len(sub_segs),
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture N: v15_maxsim_rerank
    #
    # Run v15 + gap-filling, then rerank the ENTIRE pool by max-sim:
    # for each segment, its score = max cosine similarity across ALL
    # queries used (original question, v15 cues, sub-questions).
    # This should surface segments that are highly similar to ANY
    # query, not just the original question.
    # ------------------------------------------------------------------

    def v15_maxsim_rerank(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()

        # Run actual v15
        v15_engine = AssociativeRecallEngine(
            store=self.store,
            client=self.client,
            cue_model=self.model,
            prompt_version="v15",
            max_hops=1,
            top_k_per_hop=10,
            num_cues=2,
            neighbor_radius=1,
        )
        v15_result = v15_engine.associative_retrieve(
            question,
            conversation_id,
            top_k_initial=10,
        )
        v15_segs = v15_result.all_retrieved_segments
        v15_engine.save_caches()
        exclude = {s.index for s in v15_segs}

        # Collect v15 cues
        all_queries = [question]
        for hop in v15_result.hops:
            all_queries.extend(hop.cues)

        # Run gap-filling
        sub_questions = self._decompose(question)
        all_queries.extend(sub_questions)

        gap_segs = []
        for sq in sub_questions:
            branch = self._retrieve(sq, conversation_id, exclude)
            gap_segs.extend(branch)

        pool = v15_segs + gap_segs

        # Compute query embeddings
        query_embs = []
        for q in all_queries:
            emb = self.embed_text(q)
            emb_norm = emb / max(np.linalg.norm(emb), 1e-10)
            query_embs.append(emb_norm)

        # Max-sim rerank: score = max cosine similarity to any query
        scored = []
        for seg in pool:
            seg_emb = self.store.normalized_embeddings[seg.index]
            max_sim = max(float(np.dot(q_emb, seg_emb)) for q_emb in query_embs)
            scored.append((max_sim, seg))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [seg for _, seg in scored]

        if verbose:
            print(f"  V15: {len(v15_segs)} segs, Gaps: {len(gap_segs)} segs")
            print(f"  Queries used for rerank: {len(all_queries)}")
            print(f"  Total pool: {len(pool)}")

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=reranked,
            total_retrieved=len(reranked),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "v15_maxsim_rerank",
                "v15_count": len(v15_segs),
                "gap_count": len(gap_segs),
                "num_queries": len(all_queries),
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture O: v15_rrf_rerank
    #
    # Reciprocal Rank Fusion: for each query, rank all pool segments by
    # cosine to that query. Then combine ranks using RRF formula:
    # score(seg) = sum(1/(k+rank_q(seg))) across all queries.
    # This gives high scores to segments that rank well for MULTIPLE
    # queries, even if they're not top for the original question.
    # ------------------------------------------------------------------

    def v15_rrf_rerank(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()
        RRF_K = 60  # standard RRF constant

        # Run actual v15
        v15_engine = AssociativeRecallEngine(
            store=self.store,
            client=self.client,
            cue_model=self.model,
            prompt_version="v15",
            max_hops=1,
            top_k_per_hop=10,
            num_cues=2,
            neighbor_radius=1,
        )
        v15_result = v15_engine.associative_retrieve(
            question,
            conversation_id,
            top_k_initial=10,
        )
        v15_segs = v15_result.all_retrieved_segments
        v15_engine.save_caches()
        exclude = {s.index for s in v15_segs}

        # Collect all queries
        all_queries = [question]
        for hop in v15_result.hops:
            all_queries.extend(hop.cues)

        # Decompose for more queries
        sub_questions = self._decompose(question)
        all_queries.extend(sub_questions)

        gap_segs = []
        for sq in sub_questions:
            branch = self._retrieve(sq, conversation_id, exclude)
            gap_segs.extend(branch)

        pool = v15_segs + gap_segs

        # Compute query embeddings
        query_embs = []
        for q in all_queries:
            emb = self.embed_text(q)
            emb_norm = emb / max(np.linalg.norm(emb), 1e-10)
            query_embs.append(emb_norm)

        # For each query, rank all segments
        rrf_scores: dict[int, float] = {seg.index: 0.0 for seg in pool}
        for q_emb in query_embs:
            # Score all pool segments by this query
            seg_scores = []
            for seg in pool:
                seg_emb = self.store.normalized_embeddings[seg.index]
                sim = float(np.dot(q_emb, seg_emb))
                seg_scores.append((sim, seg.index))
            seg_scores.sort(key=lambda x: x[0], reverse=True)
            # RRF score
            for rank, (_, idx) in enumerate(seg_scores):
                rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

        # Sort by RRF score
        # Deduplicate pool (in case of overlaps)
        seen = set()
        unique_pool = []
        for seg in pool:
            if seg.index not in seen:
                unique_pool.append(seg)
                seen.add(seg.index)
        unique_pool.sort(key=lambda s: rrf_scores[s.index], reverse=True)

        if verbose:
            print(
                f"  V15: {len(v15_segs)}, Gaps: {len(gap_segs)}, Pool: {len(unique_pool)}"
            )
            print(f"  Queries: {len(all_queries)}")

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=unique_pool,
            total_retrieved=len(unique_pool),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "v15_rrf_rerank",
                "v15_count": len(v15_segs),
                "gap_count": len(gap_segs),
                "num_queries": len(all_queries),
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture P: v15_boosted_tail
    #
    # Use v15's exact ordering for positions 1-15, then replace
    # positions 16-20 with the highest-cosine gap-filling segments.
    # This preserves v15's wins (which are mostly in top-15) while
    # using the tail to boost gap-filling discoveries.
    # ------------------------------------------------------------------

    def v15_boosted_tail(
        self,
        question: str,
        conversation_id: str,
        boost_start: int = 15,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()

        # Run actual v15
        v15_engine = AssociativeRecallEngine(
            store=self.store,
            client=self.client,
            cue_model=self.model,
            prompt_version="v15",
            max_hops=1,
            top_k_per_hop=10,
            num_cues=2,
            neighbor_radius=1,
        )
        v15_result = v15_engine.associative_retrieve(
            question,
            conversation_id,
            top_k_initial=10,
        )
        v15_segs = v15_result.all_retrieved_segments
        v15_engine.save_caches()
        exclude = {s.index for s in v15_segs}

        # Gap-filling via sub-questions
        sub_questions = self._decompose(question)
        gap_segs = []
        for sq in sub_questions:
            branch = self._retrieve(sq, conversation_id, exclude)
            gap_segs.extend(branch)

        if verbose:
            print(f"  V15: {len(v15_segs)} segs, Gaps: {len(gap_segs)} segs")

        # Build output:
        # - Positions 1 to boost_start: v15 ordering
        # - Positions boost_start+1 to 20: best gap segments by cosine
        # - Remaining: v15 tail + remaining gap segs
        v15_head = v15_segs[:boost_start]
        v15_tail = v15_segs[boost_start:]

        # Rank gap segments by max-sim to all queries
        all_queries = [question]
        for hop in v15_result.hops:
            all_queries.extend(hop.cues)
        all_queries.extend(sub_questions)

        query_embs = []
        for q in all_queries:
            emb = self.embed_text(q)
            emb_norm = emb / max(np.linalg.norm(emb), 1e-10)
            query_embs.append(emb_norm)

        scored_gaps = []
        for seg in gap_segs:
            seg_emb = self.store.normalized_embeddings[seg.index]
            max_sim = max(float(np.dot(q_emb, seg_emb)) for q_emb in query_embs)
            scored_gaps.append((max_sim, seg))
        scored_gaps.sort(key=lambda x: x[0], reverse=True)

        # Take top gap segments to fill the boost window
        boost_count = 20 - boost_start
        boost_segs = [seg for _, seg in scored_gaps[:boost_count]]
        remaining_gaps = [seg for _, seg in scored_gaps[boost_count:]]

        all_segments = v15_head + boost_segs + v15_tail + remaining_gaps

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "v15_boosted_tail",
                "boost_start": boost_start,
                "v15_count": len(v15_segs),
                "gap_count": len(gap_segs),
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture Q: v15_branch_cues
    #
    # Phase 1: Run v15 normally (question → top-10, then 2 cues → top-10 each)
    # Phase 2: Decompose into 2 sub-questions
    # Phase 3: For each sub-question, generate 1 FOCUSED cue based on
    #          what v15 already found for that branch. Retrieve per cue.
    # This gives sub-question-specific cues rather than generic follow-ups.
    # ------------------------------------------------------------------

    def v15_branch_cues(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        self.embed_calls = 0
        self.llm_calls = 0
        t0 = time.time()

        # Phase 1: Run actual v15
        v15_engine = AssociativeRecallEngine(
            store=self.store,
            client=self.client,
            cue_model=self.model,
            prompt_version="v15",
            max_hops=1,
            top_k_per_hop=10,
            num_cues=2,
            neighbor_radius=1,
        )
        v15_result = v15_engine.associative_retrieve(
            question,
            conversation_id,
            top_k_initial=10,
        )
        v15_segs = v15_result.all_retrieved_segments
        v15_engine.save_caches()
        exclude = {s.index for s in v15_segs}

        if verbose:
            print(f"  V15: {len(v15_segs)} segments")

        # Phase 2: Decompose
        sub_questions = self._decompose(question)
        if verbose:
            print(f"  Sub-questions: {len(sub_questions)}")

        # Phase 3: For each sub-question, generate a focused cue
        # based on what v15 found that's relevant to this sub-question
        branch_segs = []
        for i, sq in enumerate(sub_questions):
            # Find v15 segments most relevant to this sub-question
            sq_emb = self.embed_text(sq)
            sq_norm = sq_emb / max(np.linalg.norm(sq_emb), 1e-10)
            scored = []
            for seg in v15_segs:
                seg_emb = self.store.normalized_embeddings[seg.index]
                score = float(np.dot(sq_norm, seg_emb))
                scored.append((score, seg))
            scored.sort(key=lambda x: x[0], reverse=True)
            relevant = [seg for _, seg in scored[:8]]

            # Generate a focused follow-up cue for this sub-question
            cue = self._generate_followup_cue(question, sq, relevant)
            if cue:
                # Retrieve with this focused cue
                cue_emb = self.embed_text(cue)
                result = self.store.search(
                    cue_emb,
                    top_k=self.top_k,
                    conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                for seg in result.segments:
                    if seg.index not in exclude:
                        branch_segs.append(seg)
                        exclude.add(seg.index)

                # Neighbor expansion
                if self.neighbor_radius > 0:
                    for seg in list(branch_segs[-len(result.segments) :]):
                        nbrs = self.store.get_neighbors(
                            seg,
                            radius=self.neighbor_radius,
                            exclude_indices=exclude,
                        )
                        for n in nbrs:
                            branch_segs.append(n)
                            exclude.add(n.index)

                if verbose:
                    print(f"  Branch {i}: cue '{cue[:60]}...' → +{len(branch_segs)}")

        all_segments = v15_segs + branch_segs

        elapsed = time.time() - t0
        return TreeResult(
            all_segments=all_segments,
            total_retrieved=len(all_segments),
            embed_calls=self.embed_calls,
            llm_calls=self.llm_calls,
            metadata={
                "architecture": "v15_branch_cues",
                "v15_count": len(v15_segs),
                "branch_count": len(branch_segs),
                "sub_questions": sub_questions,
                "elapsed": round(elapsed, 2),
            },
        )

    # ------------------------------------------------------------------
    # Architecture D2: rerank_interleaved
    #
    # Run interleaved, then re-rank by cosine similarity to original question.
    # Combines the gap-aware retrieval of interleaved with ordering fix.
    # ------------------------------------------------------------------

    def rerank_interleaved(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        inner = self.interleaved(question, conversation_id, verbose=verbose)
        pool = inner.all_segments
        if not pool:
            return inner

        q_emb = self.embed_text(question)
        q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)

        scored = []
        for seg in pool:
            seg_emb = self.store.normalized_embeddings[seg.index]
            score = float(np.dot(q_norm, seg_emb))
            scored.append((score, seg))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [seg for _, seg in scored]

        return TreeResult(
            all_segments=reranked,
            total_retrieved=len(reranked),
            embed_calls=inner.embed_calls,
            llm_calls=inner.llm_calls,
            metadata={
                "architecture": "rerank_interleaved",
                "inner_architecture": "interleaved",
                "sub_questions": inner.metadata.get("sub_questions", []),
                "elapsed": inner.metadata.get("elapsed", 0),
            },
        )

    # ------------------------------------------------------------------
    # Architecture F2: rerank_retrieve_then_decompose
    #
    # Run retrieve_then_decompose, then re-rank by cosine to original question.
    # ------------------------------------------------------------------

    def rerank_retrieve_then_decompose(
        self,
        question: str,
        conversation_id: str,
        verbose: bool = False,
    ) -> TreeResult:
        inner = self.retrieve_then_decompose(question, conversation_id, verbose=verbose)
        pool = inner.all_segments
        if not pool:
            return inner

        q_emb = self.embed_text(question)
        q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)

        scored = []
        for seg in pool:
            seg_emb = self.store.normalized_embeddings[seg.index]
            score = float(np.dot(q_norm, seg_emb))
            scored.append((score, seg))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [seg for _, seg in scored]

        return TreeResult(
            all_segments=reranked,
            total_retrieved=len(reranked),
            embed_calls=inner.embed_calls,
            llm_calls=inner.llm_calls,
            metadata={
                "architecture": "rerank_retrieve_then_decompose",
                "inner_architecture": "retrieve_then_decompose",
                "sub_questions": inner.metadata.get("sub_questions", []),
                "elapsed": inner.metadata.get("elapsed", 0),
            },
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
A user asked this question about a past conversation:

QUESTION: {question}

Break this into 2-3 specific sub-questions that each target a DIFFERENT \
aspect or piece of information needed to answer. Each sub-question should \
be specific enough to retrieve relevant conversation segments via embedding \
similarity search.

Write each sub-question as a natural sentence (the kind of thing someone \
would actually say in a conversation). Do NOT write boolean queries or \
meta-instructions.

Format — exactly 2-3 lines:
SUB: <natural language sub-question>
SUB: <natural language sub-question>
Nothing else."""

FOLLOWUP_CUE_PROMPT = """\
You are searching a conversation history to answer a question. You retrieved \
some segments for a specific sub-question. Assess what's missing and generate \
ONE follow-up search cue.

ORIGINAL QUESTION: {question}
SUB-QUESTION: {sub_question}

RETRIEVED:
{retrieved_context}

Briefly assess: what aspect of the sub-question is NOT covered by the \
retrieved content? Generate one search cue — a short (1-2 sentence) natural \
language text using specific vocabulary that would APPEAR in the missing \
conversation content.

Format:
ASSESSMENT: <what's missing>
CUE: <search text with specific vocabulary>"""

DEEPEN_PROMPT = """\
You are navigating a reasoning tree to answer a question about a past \
conversation. At each level, segments are retrieved automatically. You \
decide: should you decompose further (PUSH) or is this branch done (POP)?

QUESTION: {question}
PATH: {context_path}
DEPTH: {depth}/{max_depth}

RETRIEVED AT THIS LEVEL ({total_retrieved} total so far):
{current_retrieved}

ALL RETRIEVED SO FAR:
{all_retrieved}

If the retrieved content covers this branch well → POP (done).
If important aspects are still missing and you can identify specific \
sub-topics to search → PUSH with 1-2 specific sub-questions.

Each sub-question must be a natural sentence with specific vocabulary \
(not boolean queries, not meta-instructions).

Respond:
ACTION: PUSH or POP
SUB: <natural language sub-question>  (only if PUSH, 1-2 lines)"""


GROUNDED_DECOMPOSE_PROMPT = """\
A user asked this question about a past conversation:

QUESTION: {question}

Here is what an initial search found — these are conversation segments \
that are already retrieved:

RETRIEVED:
{retrieved_context}

Based on what HAS been found, identify what is still MISSING to answer \
the question. Generate 2-3 focused search queries targeting the GAPS — \
aspects of the question NOT covered by the retrieved content.

Each query should be a short natural language phrase (70-110 characters) \
using specific vocabulary that would actually appear in the missing \
conversation content. Do NOT repeat topics already covered above.

Format — exactly 2-3 lines:
GAP: <search query targeting missing content>
GAP: <search query targeting missing content>
Nothing else."""


BRANCH_ASSESS_PROMPT = """\
You searched a conversation history for a specific sub-question and found \
the segments below. Assess: is this branch COMPLETE or is there still a \
specific piece of missing content you can target with ONE more search?

ORIGINAL QUESTION: {question}
SUB-QUESTION: {sub_question}

RETRIEVED:
{retrieved_context}

If the retrieved content adequately covers the sub-question, respond with \
just "DONE" on a line by itself.

If a specific piece is still missing, generate ONE short search cue \
(70-110 characters) using vocabulary that would appear in the missing \
conversation content.

Format (pick one):
DONE
or
CUE: <short search text with specific vocabulary>"""


TARGETED_SECOND_HOP_PROMPT = """\
You are searching a conversation history to answer a question. Two rounds \
of retrieval have already been done, finding {num_retrieved} segments total. \
Assess what's still MISSING and generate 2 more search cues.

QUESTION: {question}

ALL RETRIEVED SO FAR ({num_retrieved} segments):
{retrieved_context}

CRITICAL: You've already searched twice. Think carefully:
1. What specific aspect of the question is NOT yet covered?
2. What concrete vocabulary would the MISSING content use?
3. Are there related topics that appear NEAR the answer in conversation?

Generate 2 cues targeting genuinely MISSING content. Each cue should be \
a short (70-110 character) phrase using vocabulary from the conversation, \
NOT abstract descriptions.

Format:
ASSESSMENT: <what specific content is still missing>
CUE: <text with specific vocabulary>
CUE: <text with specific vocabulary>
Nothing else."""


V15_STYLE_2CUE_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

RETRIEVED CONVERSATION EXCERPTS SO FAR:
{retrieved_context}

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


V15_STYLE_CUE_PROMPT = """\
You are searching a conversation history to answer a question. You did an \
initial search and found some segments. Assess what's missing and generate \
ONE follow-up search cue.

QUESTION: {question}

RETRIEVED SO FAR:
{retrieved_context}

Briefly assess: how is this search going? What content is still missing?

Then generate one search cue — a short (70-110 character) natural language \
text using specific vocabulary that would APPEAR in the missing conversation \
content. Focus on keywords and concrete terms, not abstract descriptions.

Format:
ASSESSMENT: <1-2 sentence evaluation>
CUE: <search text with specific vocabulary>"""


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


def compute_recall(retrieved_turn_ids: set[int], source_turn_ids: set[int]) -> float:
    if not source_turn_ids:
        return 1.0
    return len(retrieved_turn_ids & source_turn_ids) / len(source_turn_ids)


def evaluate(
    engine: TreeEngine,
    questions: list[dict],
    architecture: str,
    budgets: tuple[int, ...] = (20, 50, 100),
    verbose: bool = False,
) -> list[dict]:
    results = []

    for i, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])

        # Run architecture
        engine.embed_calls = 0
        engine.llm_calls = 0
        arch_dispatch = {
            "decompose_then_retrieve": engine.decompose_then_retrieve,
            "iterative_deepen": engine.iterative_deepen,
            "flat_multi_cue": engine.flat_multi_cue,
            "rerank_pool": engine.rerank_pool,
            "interleaved": engine.interleaved,
            "retrieve_then_decompose": engine.retrieve_then_decompose,
            "branch_assess": engine.branch_assess,
            "rerank_interleaved": engine.rerank_interleaved,
            "rerank_retrieve_then_decompose": engine.rerank_retrieve_then_decompose,
            "interleaved_prioritized": engine.interleaved_prioritized,
            "v15_then_gaps": engine.v15_then_gaps,
            "v15_plus_tree": engine.v15_plus_tree,
            "actual_v15_plus_gaps": engine.actual_v15_plus_gaps,
            "actual_v15_plus_gaps_reranked": engine.actual_v15_plus_gaps_reranked,
            "actual_v15_control": engine.actual_v15_control,
            "v15_targeted_second_hop": engine.v15_targeted_second_hop,
            "v15_multi_sub": engine.v15_multi_sub,
            "v15_maxsim_rerank": engine.v15_maxsim_rerank,
            "v15_rrf_rerank": engine.v15_rrf_rerank,
            "v15_boosted_tail": engine.v15_boosted_tail,
            "v15_branch_cues": engine.v15_branch_cues,
        }
        if architecture not in arch_dispatch:
            raise ValueError(f"Unknown architecture: {architecture}")
        tree_result = arch_dispatch[architecture](q_text, conv_id, verbose=verbose)

        tree_segments = tree_result.all_segments

        # Baseline
        max_budget = max(budgets + (len(tree_segments),))
        q_emb = engine.embed_text(q_text)
        baseline = engine.store.search(q_emb, top_k=max_budget, conversation_id=conv_id)

        baseline_recalls = {}
        tree_recalls = {}
        for b in budgets:
            b_ids = {s.turn_id for s in baseline.segments[:b]}
            t_ids = {s.turn_id for s in tree_segments[:b]}
            baseline_recalls[f"r@{b}"] = compute_recall(b_ids, source_ids)
            tree_recalls[f"r@{b}"] = compute_recall(t_ids, source_ids)

        b_ids_actual = {s.turn_id for s in baseline.segments[: len(tree_segments)]}
        t_ids_actual = {s.turn_id for s in tree_segments}
        baseline_recalls["r@actual"] = compute_recall(b_ids_actual, source_ids)
        tree_recalls["r@actual"] = compute_recall(t_ids_actual, source_ids)

        result = {
            "conversation_id": conv_id,
            "category": q["category"],
            "question_index": q["question_index"],
            "question": q_text,
            "source_chat_ids": sorted(source_ids),
            "baseline_recalls": baseline_recalls,
            "arch_recalls": tree_recalls,
            "total_arch_retrieved": tree_result.total_retrieved,
            "embed_calls": tree_result.embed_calls,
            "llm_calls": tree_result.llm_calls,
            "metadata": tree_result.metadata,
        }
        results.append(result)

        delta20 = tree_recalls.get("r@20", 0) - baseline_recalls.get("r@20", 0)
        marker = "+" if delta20 > 0.001 else ("-" if delta20 < -0.001 else "=")
        print(
            f"[{i + 1}/{len(questions)}] {marker} "
            f"B={baseline_recalls.get('r@20', 0):.3f} "
            f"T={tree_recalls.get('r@20', 0):.3f} "
            f"d={delta20:+.3f} "
            f"segs={tree_result.total_retrieved} "
            f"llm={tree_result.llm_calls} "
            f"emb={tree_result.embed_calls} "
            f"| {q['category']}: {q_text[:50]}..."
        )

        engine.save_caches()

    return results


def summarize(results: list[dict], label: str) -> dict:
    n = len(results)
    if n == 0:
        return {}

    budgets = [k for k in results[0]["baseline_recalls"] if k.startswith("r@")]

    print(f"\n{'=' * 70}")
    print(f"{label} ({n} questions)")
    print(f"{'=' * 70}")

    summary = {"label": label, "n": n}
    for b in budgets:
        b_vals = [r["baseline_recalls"][b] for r in results]
        t_vals = [r["arch_recalls"][b] for r in results]
        deltas = [t - bv for t, bv in zip(t_vals, b_vals)]
        w = sum(1 for d in deltas if d > 0.001)
        t = sum(1 for d in deltas if abs(d) <= 0.001)
        l = sum(1 for d in deltas if d < -0.001)
        avg_b = sum(b_vals) / n
        avg_t = sum(t_vals) / n
        avg_d = sum(deltas) / n
        print(
            f"  {b:>10s}: B={avg_b:.3f} T={avg_t:.3f} Δ={avg_d:+.3f} W/T/L={w}/{t}/{l}"
        )
        summary[f"baseline_{b}"] = avg_b
        summary[f"tree_{b}"] = avg_t
        summary[f"delta_{b}"] = avg_d
        summary[f"wtl_{b}"] = f"{w}/{t}/{l}"

    avg_ret = sum(r["total_arch_retrieved"] for r in results) / n
    avg_llm = sum(r["llm_calls"] for r in results) / n
    avg_emb = sum(r["embed_calls"] for r in results) / n
    print(
        f"\n  Avg retrieved: {avg_ret:.1f}, LLM calls: {avg_llm:.1f}, Embed calls: {avg_emb:.1f}"
    )
    summary["avg_retrieved"] = avg_ret
    summary["avg_llm_calls"] = avg_llm
    summary["avg_embed_calls"] = avg_emb

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    all_archs = [
        "decompose_then_retrieve",
        "iterative_deepen",
        "flat_multi_cue",
        "rerank_pool",
        "interleaved",
        "retrieve_then_decompose",
        "branch_assess",
        "rerank_interleaved",
        "rerank_retrieve_then_decompose",
        "interleaved_prioritized",
        "v15_then_gaps",
        "v15_plus_tree",
        "actual_v15_plus_gaps",
        "actual_v15_plus_gaps_reranked",
        "actual_v15_control",
        "v15_targeted_second_hop",
        "v15_multi_sub",
        "v15_maxsim_rerank",
        "v15_rrf_rerank",
        "v15_boosted_tail",
        "v15_branch_cues",
    ]
    parser.add_argument("--arch", default="decompose_then_retrieve", choices=all_archs)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--data-suffix", default="_extended")
    args = parser.parse_args()

    store = SegmentStore(npz_name=f"segments{args.data_suffix}.npz")
    questions_path = (
        Path(__file__).resolve().parent / "data" / f"questions{args.data_suffix}.json"
    )
    with open(questions_path) as f:
        questions = json.load(f)

    if args.benchmark:
        questions = [q for q in questions if q.get("benchmark") == args.benchmark]
    if args.category:
        questions = [q for q in questions if args.category in q["category"]]
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"Loaded {len(questions)} questions, {len(store.segments)} segments")

    engine = TreeEngine(
        store=store,
        top_k=args.top_k,
        neighbor_radius=args.neighbor_radius,
    )

    results = evaluate(engine, questions, args.arch, verbose=args.verbose)

    label = f"tree_{args.arch}_k{args.top_k}_nr{args.neighbor_radius}"
    if args.benchmark:
        label += f"_{args.benchmark}"
    summary = summarize(results, label)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{label}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_file}")

    engine.save_caches()
