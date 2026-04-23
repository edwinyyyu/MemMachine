"""Alternative retrieval architectures for associative recall.

Each architecture implements a different retrieval flow, returning a list of
Segment objects. All architectures use the same SegmentStore for embedding
lookups but differ in their retrieval strategy.
"""

import hashlib
import json
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


@dataclass
class ArchResult:
    """Result from any architecture's retrieval."""
    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


class ArchEmbeddingCache(EmbeddingCache):
    """Embedding cache that reads from main cache but writes to arch-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Read from main cache
        main_cache_file = self.cache_dir / "embedding_cache.json"
        self._cache: dict[str, list[float]] = {}
        if main_cache_file.exists():
            with open(main_cache_file) as f:
                self._cache = json.load(f)
        # Also load arch-specific cache
        self.cache_file = self.cache_dir / "arch_embedding_cache.json"
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                arch_cache = json.load(f)
                self._cache.update(arch_cache)
        # Track new entries separately for saving
        self._new_entries: dict[str, list[float]] = {}

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding.tolist()
        self._new_entries[key] = embedding.tolist()

    def save(self) -> None:
        """Only save new entries to the arch-specific cache."""
        if not self._new_entries:
            return
        # Load existing arch cache, merge, save
        existing = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                existing = json.load(f)
        existing.update(self._new_entries)
        tmp_file = self.cache_file.with_suffix(".json.tmp")
        with open(tmp_file, "w") as f:
            json.dump(existing, f)
        tmp_file.replace(self.cache_file)


class ArchLLMCache(LLMCache):
    """LLM cache that reads from main cache but writes to arch-specific file."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Read from main cache
        main_cache_file = self.cache_dir / "llm_cache.json"
        self._cache: dict[str, str] = {}
        if main_cache_file.exists():
            with open(main_cache_file) as f:
                self._cache = json.load(f)
        # Also load arch-specific cache
        self.cache_file = self.cache_dir / "arch_llm_cache.json"
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                arch_cache = json.load(f)
                self._cache.update(arch_cache)
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
        tmp_file = self.cache_file.with_suffix(".json.tmp")
        with open(tmp_file, "w") as f:
            json.dump(existing, f)
        tmp_file.replace(self.cache_file)


class BaseArchitecture:
    """Base class providing shared embedding/LLM utilities."""

    def __init__(self, store: SegmentStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI(timeout=60.0)
        self.embedding_cache = ArchEmbeddingCache()
        self.llm_cache = ArchLLMCache()
        self.embed_calls = 0
        self.llm_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            # Return zero vector for empty text
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

    def llm_call(self, model: str, prompt: str) -> str:
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

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Architecture 1: Segment-as-Query (no LLM)
# ---------------------------------------------------------------------------
class SegmentAsQuery(BaseArchitecture):
    """Walk through embedding space using retrieved segments as queries.

    Take top-1 result, use its text as a new query, repeat for N hops.
    Collects all unique segments found along the walk.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 initial_top_k: int = 5, walk_hops: int = 4,
                 walk_top_k: int = 5):
        super().__init__(store, client)
        self.initial_top_k = initial_top_k
        self.walk_hops = walk_hops
        self.walk_top_k = walk_top_k

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        # Initial retrieval with the question
        query_emb = self.embed_text(question)
        initial_result = self.store.search(
            query_emb, top_k=self.initial_top_k, conversation_id=conversation_id
        )

        all_segments: list[Segment] = list(initial_result.segments)
        seen_indices: set[int] = {s.index for s in all_segments}

        # For each of the top results, walk from it
        walk_seeds = initial_result.segments[:3]  # Walk from top-3
        for seed in walk_seeds:
            current_seg = seed
            for hop in range(self.walk_hops):
                # Use the segment text as query
                seg_emb = self.embed_text(current_seg.text)
                result = self.store.search(
                    seg_emb, top_k=self.walk_top_k,
                    conversation_id=conversation_id,
                    exclude_indices=seen_indices,
                )
                if not result.segments:
                    break
                for seg in result.segments:
                    if seg.index not in seen_indices:
                        all_segments.append(seg)
                        seen_indices.add(seg.index)
                # Next hop: use the top-1 new result
                current_seg = result.segments[0]

        return ArchResult(
            segments=all_segments,
            metadata={
                "name": "segment_as_query",
                "initial_top_k": self.initial_top_k,
                "walk_hops": self.walk_hops,
            },
        )


# ---------------------------------------------------------------------------
# Architecture 2: Cluster-then-Diversify (no LLM)
# ---------------------------------------------------------------------------
class ClusterDiversify(BaseArchitecture):
    """Retrieve large initial set, cluster, then diversify.

    Retrieve top-100 by cosine. K-means cluster the embeddings.
    Select representatives from each cluster, prioritizing diversity.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 initial_top_k: int = 100, n_clusters: int = 8):
        super().__init__(store, client)
        self.initial_top_k = initial_top_k
        self.n_clusters = n_clusters

    def _kmeans(self, embeddings: np.ndarray, k: int,
                max_iter: int = 50) -> np.ndarray:
        """Simple k-means clustering. Returns cluster assignments."""
        n = len(embeddings)
        if n <= k:
            return np.arange(n)
        # Initialize with k-means++
        indices = [np.random.randint(n)]
        for _ in range(1, k):
            # Distance to nearest existing center
            centers = embeddings[indices]
            dists = np.min(
                1 - embeddings @ centers.T, axis=1
            )  # cosine distance
            dists[indices] = 0
            probs = dists / (dists.sum() + 1e-10)
            idx = np.random.choice(n, p=probs)
            indices.append(idx)

        centers = embeddings[indices].copy()
        assignments = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign
            sims = embeddings @ centers.T
            new_assignments = np.argmax(sims, axis=1)
            if np.array_equal(assignments, new_assignments):
                break
            assignments = new_assignments
            # Update centers
            for c in range(k):
                mask = assignments == c
                if mask.any():
                    centers[c] = embeddings[mask].mean(axis=0)
                    centers[c] /= max(np.linalg.norm(centers[c]), 1e-10)

        return assignments

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        query_emb = self.embed_text(question)
        initial_result = self.store.search(
            query_emb, top_k=self.initial_top_k, conversation_id=conversation_id
        )

        if len(initial_result.segments) <= 20:
            return ArchResult(
                segments=initial_result.segments,
                metadata={"name": "cluster_diversify", "note": "too few segments"},
            )

        # Get embeddings for initial results
        seg_indices = [s.index for s in initial_result.segments]
        seg_embeddings = self.store.normalized_embeddings[seg_indices]

        # Cluster
        k = min(self.n_clusters, len(seg_indices))
        assignments = self._kmeans(seg_embeddings, k)

        # From each cluster, pick the segment closest to query
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        diversified: list[Segment] = []
        cluster_reps: dict[int, list[tuple[float, Segment]]] = {}

        for i, seg in enumerate(initial_result.segments):
            c = assignments[i]
            score = float(np.dot(seg_embeddings[i], query_norm))
            if c not in cluster_reps:
                cluster_reps[c] = []
            cluster_reps[c].append((score, seg))

        # Sort each cluster by score, then interleave (round-robin)
        for c in cluster_reps:
            cluster_reps[c].sort(key=lambda x: x[0], reverse=True)

        # Round-robin selection to ensure diversity
        seen = set()
        round_num = 0
        while len(diversified) < len(initial_result.segments):
            added_this_round = False
            for c in sorted(cluster_reps.keys()):
                if round_num < len(cluster_reps[c]):
                    seg = cluster_reps[c][round_num][1]
                    if seg.index not in seen:
                        diversified.append(seg)
                        seen.add(seg.index)
                        added_this_round = True
            if not added_this_round:
                break
            round_num += 1

        return ArchResult(
            segments=diversified,
            metadata={
                "name": "cluster_diversify",
                "n_clusters": k,
                "cluster_sizes": {
                    str(c): len(segs) for c, segs in cluster_reps.items()
                },
            },
        )


# ---------------------------------------------------------------------------
# Architecture 3: Multi-Query Parallel Fusion (LLM for upfront cues only)
# ---------------------------------------------------------------------------
class MultiQueryFusion(BaseArchitecture):
    """Generate all cues upfront, retrieve for each, fuse with RRF.

    No iterative refinement: generate diverse queries from the question alone,
    retrieve in parallel, combine with Reciprocal Rank Fusion.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 num_queries: int = 5, per_query_k: int = 20,
                 model: str = "gpt-5-mini"):
        super().__init__(store, client)
        self.num_queries = num_queries
        self.per_query_k = per_query_k
        self.model = model

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        # Generate all queries upfront (1 LLM call)
        prompt = f"""\
Generate {self.num_queries} different search queries to find all conversation \
content relevant to answering this question. Each query should target a \
DIFFERENT aspect or sub-topic. The queries will be embedded and matched \
against conversation turns via cosine similarity.

Question: {question}

Write queries that sound like conversation content, not meta-instructions. \
Use specific vocabulary, tool names, actions, and details.

Format: one query per line, prefixed with "Q: ". Nothing else."""

        response = self.llm_call(self.model, prompt)
        queries = [question]  # Always include original
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("Q:"):
                q = line[2:].strip()
                if q:
                    queries.append(q)

        # Retrieve for each query
        all_results: dict[int, list[int]] = {}  # index -> list of ranks
        for qi, query in enumerate(queries):
            q_emb = self.embed_text(query)
            result = self.store.search(
                q_emb, top_k=self.per_query_k, conversation_id=conversation_id
            )
            for rank, seg in enumerate(result.segments):
                if seg.index not in all_results:
                    all_results[seg.index] = []
                all_results[seg.index].append(rank)

        # Reciprocal Rank Fusion
        K = 60  # RRF constant
        rrf_scores: list[tuple[float, int]] = []
        for idx, ranks in all_results.items():
            score = sum(1.0 / (K + r) for r in ranks)
            rrf_scores.append((score, idx))
        rrf_scores.sort(reverse=True)

        segments = [self.store.segments[idx] for _, idx in rrf_scores]

        return ArchResult(
            segments=segments,
            metadata={
                "name": "multi_query_fusion",
                "num_queries": len(queries),
                "total_unique": len(segments),
            },
        )


# ---------------------------------------------------------------------------
# Architecture 4: Retrieve-Summarize-Retrieve
# ---------------------------------------------------------------------------
class RetrieveSummarizeRetrieve(BaseArchitecture):
    """Hop 0: retrieve. Summarize findings. Use summary as next query.

    The summary creates a "centroid" in embedding space surrounded by
    related content.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 initial_top_k: int = 10, summary_hops: int = 2,
                 per_hop_k: int = 15, model: str = "gpt-5-mini"):
        super().__init__(store, client)
        self.initial_top_k = initial_top_k
        self.summary_hops = summary_hops
        self.per_hop_k = per_hop_k
        self.model = model

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        query_emb = self.embed_text(question)
        initial_result = self.store.search(
            query_emb, top_k=self.initial_top_k, conversation_id=conversation_id
        )

        all_segments = list(initial_result.segments)
        seen_indices = {s.index for s in all_segments}

        for hop in range(self.summary_hops):
            # Summarize what's been found
            context = "\n".join(
                f"[Turn {s.turn_id}]: {s.text[:200]}"
                for s in sorted(all_segments[-15:], key=lambda x: x.turn_id)
            )
            prompt = f"""\
Summarize the conversation content found so far in relation to this question.
Write a 2-3 sentence summary that captures the GIST of what was discussed.
Use specific vocabulary from the excerpts.

Question: {question}

Found so far:
{context}

Summary (2-3 sentences, specific vocabulary):"""

            summary = self.llm_call(self.model, prompt)
            # Embed the summary and use as query
            summary_emb = self.embed_text(summary.strip())
            result = self.store.search(
                summary_emb, top_k=self.per_hop_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        return ArchResult(
            segments=all_segments,
            metadata={
                "name": "retrieve_summarize_retrieve",
                "hops": self.summary_hops,
                "total": len(all_segments),
            },
        )


# ---------------------------------------------------------------------------
# Architecture 5: Agent with Working Set Management
# ---------------------------------------------------------------------------
class AgentWorkingSet(BaseArchitecture):
    """LLM decides what to search, keep, and drop. Working set management.

    The model sees the current working set and decides actions:
    SEARCH(query), KEEP(ids), DROP(ids), STOP.
    Limited to ~5 tool calls total.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 max_tool_calls: int = 5, per_search_k: int = 10,
                 model: str = "gpt-5-mini"):
        super().__init__(store, client)
        self.max_tool_calls = max_tool_calls
        self.per_search_k = per_search_k
        self.model = model

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        # Initial retrieval
        query_emb = self.embed_text(question)
        initial = self.store.search(
            query_emb, top_k=self.per_search_k, conversation_id=conversation_id
        )

        working_set: dict[int, Segment] = {}
        all_ever_seen: dict[int, Segment] = {}
        excluded_indices: set[int] = set()

        for seg in initial.segments:
            working_set[seg.index] = seg
            all_ever_seen[seg.index] = seg
            excluded_indices.add(seg.index)

        for step in range(self.max_tool_calls):
            # Build working set description
            ws_lines = []
            for seg in sorted(working_set.values(), key=lambda s: s.turn_id):
                ws_lines.append(f"[ID:{seg.index} Turn:{seg.turn_id}]: {seg.text[:200]}")
            ws_text = "\n".join(ws_lines) if ws_lines else "(empty)"

            prompt = f"""\
You are searching a conversation to answer: {question}

CURRENT WORKING SET ({len(working_set)} segments):
{ws_text}

You have {self.max_tool_calls - step} actions remaining. Choose ONE action:

SEARCH: <query text> — search for more segments using this query
DROP: <comma-separated IDs> — remove irrelevant segments from working set
STOP — done searching, return current working set

Choose the action that will best help answer the question. If the working \
set already covers the question well, STOP. If segments are clearly \
irrelevant, DROP them. Otherwise, SEARCH for missing information.

Output exactly ONE action line. Nothing else."""

            response = self.llm_call(self.model, prompt)
            action_line = response.strip().split("\n")[0].strip()

            if action_line.startswith("STOP"):
                break
            elif action_line.startswith("SEARCH:"):
                query = action_line[7:].strip()
                if query:
                    q_emb = self.embed_text(query)
                    result = self.store.search(
                        q_emb, top_k=self.per_search_k,
                        conversation_id=conversation_id,
                        exclude_indices=excluded_indices,
                    )
                    for seg in result.segments:
                        working_set[seg.index] = seg
                        all_ever_seen[seg.index] = seg
                        excluded_indices.add(seg.index)
            elif action_line.startswith("DROP:"):
                ids_text = action_line[5:].strip()
                for part in ids_text.split(","):
                    part = part.strip()
                    try:
                        idx = int(part)
                        if idx in working_set:
                            del working_set[idx]
                    except ValueError:
                        pass

        # Return all_ever_seen (not just working set) for fair comparison
        # but with working set segments first (prioritized)
        ws_segments = sorted(working_set.values(), key=lambda s: s.turn_id)
        dropped = [s for idx, s in all_ever_seen.items() if idx not in working_set]
        all_segments = ws_segments + dropped

        return ArchResult(
            segments=all_segments,
            metadata={
                "name": "agent_working_set",
                "working_set_size": len(working_set),
                "total_ever_seen": len(all_ever_seen),
                "steps": step + 1,
            },
        )


# ---------------------------------------------------------------------------
# Architecture 6: Hybrid Baseline + Gap-Fill
# ---------------------------------------------------------------------------
class HybridGapFill(BaseArchitecture):
    """Use baseline top-20 as core, then one LLM call to gap-fill.

    Most practical: keep what works (cosine top-k), add a small associative
    layer for gap-filling.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 baseline_k: int = 20, gap_fill_k: int = 10,
                 model: str = "gpt-5-mini"):
        super().__init__(store, client)
        self.baseline_k = baseline_k
        self.gap_fill_k = gap_fill_k
        self.model = model

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        # Step 1: Baseline top-k
        query_emb = self.embed_text(question)
        baseline = self.store.search(
            query_emb, top_k=self.baseline_k, conversation_id=conversation_id
        )

        all_segments = list(baseline.segments)
        seen_indices = {s.index for s in all_segments}

        # Step 2: Show baseline results to LLM, ask what's missing
        context = "\n".join(
            f"[Turn {s.turn_id}]: {s.text[:200]}"
            for s in sorted(all_segments, key=lambda s: s.turn_id)
        )

        prompt = f"""\
I'm trying to answer this question about a conversation:
{question}

Here are the top-{self.baseline_k} most similar conversation turns I found:
{context}

What specific information is MISSING from these results that would be needed \
to answer the question? Generate 2 search queries targeting the missing \
content. Each query should sound like conversation content (not meta-instructions).

Format:
CUE: <search text>
CUE: <search text>
Nothing else."""

        response = self.llm_call(self.model, prompt)
        cues = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("CUE:"):
                cue = line[4:].strip()
                if cue:
                    cues.append(cue)

        # Step 3: Retrieve for gap-fill cues
        for cue in cues[:2]:
            cue_emb = self.embed_text(cue)
            result = self.store.search(
                cue_emb, top_k=self.gap_fill_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        return ArchResult(
            segments=all_segments,
            metadata={
                "name": "hybrid_gap_fill",
                "baseline_k": self.baseline_k,
                "gap_cues": cues[:2],
                "total": len(all_segments),
            },
        )


# ---------------------------------------------------------------------------
# Architecture 7a: Embedding Centroid Walk (no LLM)
# ---------------------------------------------------------------------------
class CentroidWalk(BaseArchitecture):
    """Iteratively shift a query centroid toward found content.

    Start with the question embedding. Retrieve top-k. Compute the centroid
    of found relevant-looking segments (high cosine). Shift the query toward
    that centroid. Repeat. The centroid "drifts" toward dense regions of
    related content.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 initial_top_k: int = 10, hops: int = 3,
                 per_hop_k: int = 10, drift_alpha: float = 0.3):
        super().__init__(store, client)
        self.initial_top_k = initial_top_k
        self.hops = hops
        self.per_hop_k = per_hop_k
        self.drift_alpha = drift_alpha  # How much to drift toward centroid

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)

        initial = self.store.search(
            query_emb, top_k=self.initial_top_k, conversation_id=conversation_id
        )

        all_segments = list(initial.segments)
        seen_indices = {s.index for s in all_segments}
        current_query = query_norm.copy()

        for hop in range(self.hops):
            # Compute centroid of all found segments
            found_embs = self.store.normalized_embeddings[
                [s.index for s in all_segments]
            ]
            centroid = found_embs.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-10)

            # Drift query toward centroid
            current_query = (1 - self.drift_alpha) * current_query + self.drift_alpha * centroid
            current_query /= max(np.linalg.norm(current_query), 1e-10)

            result = self.store.search(
                current_query, top_k=self.per_hop_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        return ArchResult(
            segments=all_segments,
            metadata={
                "name": "centroid_walk",
                "hops": self.hops,
                "drift_alpha": self.drift_alpha,
                "total": len(all_segments),
            },
        )


# ---------------------------------------------------------------------------
# Architecture 7b: Negative-Space Retrieval (no LLM)
# ---------------------------------------------------------------------------
class NegativeSpace(BaseArchitecture):
    """Find content that's DIFFERENT from what's already been found.

    Retrieve initial top-k. Compute the anti-centroid (negate the centroid
    of found content relative to the query). Search using:
    query + alpha * (query - centroid_of_found).

    This pushes the query AWAY from the already-found cluster, toward
    content that's related to the question but in a different embedding
    region.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 initial_top_k: int = 15, hops: int = 2,
                 per_hop_k: int = 15, push_alpha: float = 0.3):
        super().__init__(store, client)
        self.initial_top_k = initial_top_k
        self.hops = hops
        self.per_hop_k = per_hop_k
        self.push_alpha = push_alpha

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)

        initial = self.store.search(
            query_emb, top_k=self.initial_top_k, conversation_id=conversation_id
        )

        all_segments = list(initial.segments)
        seen_indices = {s.index for s in all_segments}

        for hop in range(self.hops):
            # Centroid of found
            found_embs = self.store.normalized_embeddings[
                [s.index for s in all_segments]
            ]
            centroid = found_embs.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-10)

            # Push query away from centroid (explore new territory)
            push_direction = query_norm - centroid
            pushed_query = query_norm + self.push_alpha * push_direction
            pushed_query /= max(np.linalg.norm(pushed_query), 1e-10)

            result = self.store.search(
                pushed_query, top_k=self.per_hop_k,
                conversation_id=conversation_id,
                exclude_indices=seen_indices,
            )
            for seg in result.segments:
                if seg.index not in seen_indices:
                    all_segments.append(seg)
                    seen_indices.add(seg.index)

        return ArchResult(
            segments=all_segments,
            metadata={
                "name": "negative_space",
                "hops": self.hops,
                "push_alpha": self.push_alpha,
                "total": len(all_segments),
            },
        )


# ---------------------------------------------------------------------------
# Architecture 7c: MMR-based Diversified Retrieval (no LLM)
# ---------------------------------------------------------------------------
class MMRDiversified(BaseArchitecture):
    """Maximal Marginal Relevance — balance relevance with diversity.

    Standard MMR: at each step, select the segment that maximizes:
    lambda * sim(query, seg) - (1-lambda) * max_sim(seg, already_selected)

    This naturally diversifies by penalizing similarity to already-selected
    segments.
    """

    def __init__(self, store: SegmentStore, client: OpenAI | None = None,
                 total_k: int = 60, lambda_param: float = 0.7,
                 candidate_pool: int = 150):
        super().__init__(store, client)
        self.total_k = total_k
        self.lambda_param = lambda_param
        self.candidate_pool = candidate_pool

    def retrieve(self, question: str, conversation_id: str) -> ArchResult:
        query_emb = self.embed_text(question)
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)

        # Get large candidate pool
        candidates = self.store.search(
            query_emb, top_k=self.candidate_pool, conversation_id=conversation_id
        )
        if not candidates.segments:
            return ArchResult(segments=[], metadata={"name": "mmr_diversified"})

        cand_indices = [s.index for s in candidates.segments]
        cand_embs = self.store.normalized_embeddings[cand_indices]
        query_sims = cand_embs @ query_norm  # Shape: (n_cands,)

        selected: list[int] = []  # indices into candidates list
        selected_embs: list[np.ndarray] = []

        for _ in range(min(self.total_k, len(candidates.segments))):
            if not selected:
                # Pick the most relevant
                best = int(np.argmax(query_sims))
                selected.append(best)
                selected_embs.append(cand_embs[best])
                continue

            # Compute MMR score for each remaining candidate
            remaining = [i for i in range(len(cand_indices)) if i not in set(selected)]
            if not remaining:
                break

            best_score = -float("inf")
            best_idx = remaining[0]
            sel_embs_arr = np.array(selected_embs)

            for i in remaining:
                relevance = float(query_sims[i])
                # Max similarity to any selected segment
                diversity_penalty = float(np.max(cand_embs[i] @ sel_embs_arr.T))
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * diversity_penalty
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(best_idx)
            selected_embs.append(cand_embs[best_idx])

        segments = [candidates.segments[i] for i in selected]
        return ArchResult(
            segments=segments,
            metadata={
                "name": "mmr_diversified",
                "lambda": self.lambda_param,
                "total": len(segments),
            },
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ARCHITECTURES: dict[str, type[BaseArchitecture]] = {
    "segment_as_query": SegmentAsQuery,
    "cluster_diversify": ClusterDiversify,
    "multi_query_fusion": MultiQueryFusion,
    "retrieve_summarize_retrieve": RetrieveSummarizeRetrieve,
    "agent_working_set": AgentWorkingSet,
    "hybrid_gap_fill": HybridGapFill,
    "centroid_walk": CentroidWalk,
    "negative_space": NegativeSpace,
    "mmr_diversified": MMRDiversified,
}
