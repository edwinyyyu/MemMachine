"""Clustering logic for attribute-memory ingestion.

:class:`ClusterManager` groups events (by embedding cosine similarity
and optional time-gap eligibility) into clusters that are flushed to
the LLM as coherent batches.  :class:`ClusterSplitterProtocol` lets a
reranker-driven strategy sub-divide clusters before flush — a
:class:`NoOpClusterSplitter` default is provided.

Data models (``ClusterState``, ``ClusterInfo``, etc.) live in
:mod:`.data_types`; this module is pure logic.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

import numpy as np

from memmachine_server.common.reranker import Reranker
from memmachine_server.semantic_memory.attribute_memory.data_types import (
    ClusterAssignment,
    ClusterInfo,
    ClusterParams,
    ClusterSplitParams,
    ClusterSplitRecord,
    ClusterState,
    Content,
    ContinuitySignals,
    Event,
    MessageContext,
    Text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# ClusterManager
# ---------------------------------------------------------------------- #


class ClusterManager:
    """Assigns events to clusters and updates state in place."""

    def __init__(self, params: ClusterParams) -> None:
        """Bind the assignment parameters."""
        self._params = params

    def assign(
        self,
        *,
        event_id: UUID,
        embedding: Sequence[float],
        timestamp: datetime,
        state: ClusterState,
    ) -> ClusterAssignment:
        """Assign a single event to a cluster, mutating ``state``.

        If the event is already known (in ``state.event_to_cluster``)
        returns its existing assignment without touching the state.
        Otherwise picks the most-similar eligible cluster (if any
        passes the similarity threshold) or creates a new one.
        """
        if event_id in state.event_to_cluster:
            return ClusterAssignment(
                cluster_id=state.event_to_cluster[event_id],
                similarity=None,
                created_new=False,
            )

        selected_id, similarity = self._select_cluster(
            embedding=embedding,
            timestamp=timestamp,
            state=state,
        )

        if selected_id is None or similarity < self._params.similarity_threshold:
            cluster_id = self._create_cluster(
                embedding=embedding,
                timestamp=timestamp,
                state=state,
            )
            state.event_to_cluster[event_id] = cluster_id
            return ClusterAssignment(
                cluster_id=cluster_id,
                similarity=None,
                created_new=True,
            )

        self._update_cluster(
            cluster_id=selected_id,
            embedding=embedding,
            timestamp=timestamp,
            state=state,
        )
        state.event_to_cluster[event_id] = selected_id
        return ClusterAssignment(
            cluster_id=selected_id,
            similarity=similarity,
            created_new=False,
        )

    def _select_cluster(
        self,
        *,
        embedding: Sequence[float],
        timestamp: datetime,
        state: ClusterState,
    ) -> tuple[str | None, float]:
        best_id: str | None = None
        best_similarity = -1.0
        for cluster_id, info in state.clusters.items():
            if not self._cluster_is_eligible(info, timestamp):
                continue
            similarity = _cosine_similarity(info.centroid, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = cluster_id
        return best_id, best_similarity

    def _cluster_is_eligible(self, info: ClusterInfo, timestamp: datetime) -> bool:
        max_gap = self._params.max_time_gap
        if max_gap is None:
            return True
        gap = timestamp - info.last_ts
        if gap.total_seconds() < 0:
            gap = info.last_ts - timestamp
        return gap <= max_gap

    def _create_cluster(
        self,
        *,
        embedding: Sequence[float],
        timestamp: datetime,
        state: ClusterState,
    ) -> str:
        cluster_id = f"{self._params.id_prefix}{state.next_cluster_id}"
        state.next_cluster_id += 1
        state.clusters[cluster_id] = ClusterInfo(
            centroid=[float(x) for x in embedding],
            count=1,
            last_ts=timestamp,
        )
        return cluster_id

    def _update_cluster(
        self,
        *,
        cluster_id: str,
        embedding: Sequence[float],
        timestamp: datetime,
        state: ClusterState,
    ) -> None:
        info = state.clusters[cluster_id]
        centroid = np.asarray(info.centroid, dtype=float)
        new_vec = np.asarray(embedding, dtype=float)
        if centroid.shape != new_vec.shape:
            raise ValueError("Embedding dimension mismatch")
        new_count = info.count + 1
        updated = (centroid * info.count + new_vec) / new_count
        info.centroid = [float(x) for x in updated.tolist()]
        info.count = new_count
        info.last_ts = timestamp


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors; returns 0 if either is zero."""
    a_vec = np.asarray(a, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    if a_vec.shape != b_vec.shape:
        raise ValueError("Embedding dimension mismatch")
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


# ---------------------------------------------------------------------- #
# Cluster splitters
# ---------------------------------------------------------------------- #


@runtime_checkable
class ClusterSplitterProtocol(Protocol):
    """Contract for the split phase in the ingestion pipeline."""

    async def maybe_split_clusters(
        self,
        *,
        cluster_events: Sequence[tuple[str, Sequence[Event]]],
        cluster_embeddings: Mapping[UUID, Sequence[float]],
        state: ClusterState,
        reranker: Reranker | None,
    ) -> tuple[list[tuple[str, list[Event]]], ClusterState]:
        """Return possibly-split clusters and the (mutated) state."""
        ...


class NoOpClusterSplitter:
    """A splitter that never splits — the default, for callers without a reranker."""

    async def maybe_split_clusters(
        self,
        *,
        cluster_events: Sequence[tuple[str, Sequence[Event]]],
        cluster_embeddings: Mapping[UUID, Sequence[float]],  # noqa: ARG002
        state: ClusterState,
        reranker: Reranker | None,  # noqa: ARG002
    ) -> tuple[list[tuple[str, list[Event]]], ClusterState]:
        return [(cid, list(events)) for cid, events in cluster_events], state


class SplitGate:
    """Decides whether a cluster is a candidate for reranker scoring."""

    def __init__(self, params: ClusterSplitParams) -> None:
        """Bind the split parameters."""
        self._params = params

    def is_candidate(
        self,
        events: Sequence[Event],
        signals: ContinuitySignals,
    ) -> bool:
        """True only when min-size and at least one discontinuity signal holds."""
        if len(events) < self._params.min_cluster_size:
            return False
        if signals.min_adjacent_similarity < self._params.low_similarity_threshold:
            return True
        if (
            self._params.time_gap_seconds is not None
            and signals.max_time_gap_seconds > self._params.time_gap_seconds
        ):
            return True
        sims = np.asarray(signals.adjacent_similarities)
        if len(sims) >= 2:
            std = float(sims.std())
            if std > 0.0:
                mean = float(sims.mean())
                z_scores = (mean - sims) / std
                if float(z_scores.max()) >= self._params.cohesion_drop_zscore:
                    return True
        return False


class RerankerClusterSplitter:
    """Production splitter: gates on heuristics, scores with reranker, splits.

    Flow per ready cluster:
      1. Compute continuity signals (adjacent-similarity + time-gaps).
      2. Consult :class:`SplitGate` — skip if the cluster is coherent.
      3. Replay a previously recorded split if the input hash matches
         (idempotency across retries).
      4. Ask the reranker to score adjacent pairs, pick split points,
         apply the split, and record the decision in ``state``.
    """

    def __init__(self, params: ClusterSplitParams) -> None:
        """Bind the split parameters and internal gate."""
        self._params = params
        self._gate = SplitGate(params)

    async def maybe_split_clusters(
        self,
        *,
        cluster_events: Sequence[tuple[str, Sequence[Event]]],
        cluster_embeddings: Mapping[UUID, Sequence[float]],
        state: ClusterState,
        reranker: Reranker | None,
    ) -> tuple[list[tuple[str, list[Event]]], ClusterState]:
        result: list[tuple[str, list[Event]]] = []
        for cluster_id, events in cluster_events:
            pieces = await self._maybe_split_cluster(
                cluster_id=cluster_id,
                events=events,
                cluster_embeddings=cluster_embeddings,
                state=state,
                reranker=reranker,
            )
            result.extend(pieces)
        return result, state

    async def _maybe_split_cluster(
        self,
        *,
        cluster_id: str,
        events: Sequence[Event],
        cluster_embeddings: Mapping[UUID, Sequence[float]],
        state: ClusterState,
        reranker: Reranker | None,
    ) -> list[tuple[str, list[Event]]]:
        ordered_embeddings, event_uuids = _collect_embeddings(
            events, cluster_embeddings
        )
        current_hash = _input_hash(event_uuids)

        replayed = self._replay_if_unchanged(
            cluster_id, events, ordered_embeddings, current_hash, state
        )
        if replayed is not None:
            return replayed

        signals = _compute_signals(events, ordered_embeddings)
        if not self._gate.is_candidate(events, signals):
            return self._keep_cluster_intact(state, cluster_id, events, current_hash)

        if reranker is None:
            logger.warning(
                "Cluster %s is a split candidate but no reranker is configured",
                cluster_id,
            )
            return self._keep_cluster_intact(state, cluster_id, events, current_hash)

        logger.info(
            "Cluster %s is a split candidate (size=%d), scoring with reranker",
            cluster_id,
            len(events),
        )
        window_size = min(len(events), self._params.max_messages_in_prompt)
        window = events[:window_size]
        scores = await self._get_reranker_scores(window, reranker, cluster_id)
        if not scores:
            return self._keep_cluster_intact(state, cluster_id, events, current_hash)

        time_gaps = signals.time_gaps_seconds[: max(len(window) - 1, 0)]
        split_indices = self._select_split_indices(scores, time_gaps)
        valid_indices = validate_split_indices(split_indices, window_size)
        if not valid_indices:
            logger.info("Reranker decided no split for cluster %s", cluster_id)
            return self._keep_cluster_intact(state, cluster_id, events, current_hash)

        pieces, resulting_ids = apply_cluster_split(
            cluster_id, events, ordered_embeddings, valid_indices, state
        )
        _record_split(state, cluster_id, resulting_ids, current_hash)
        return pieces

    def _replay_if_unchanged(
        self,
        cluster_id: str,
        events: Sequence[Event],
        ordered_embeddings: Sequence[Sequence[float]],
        current_hash: str,
        state: ClusterState,
    ) -> list[tuple[str, list[Event]]] | None:
        record = state.split_records.get(cluster_id)
        if record is None:
            return None
        if record.input_hash != current_hash:
            state.split_records.pop(cluster_id, None)
            return None
        if record.resulting_cluster_ids:
            return self._replay_split(
                cluster_id, events, ordered_embeddings, record, state
            )
        return [(cluster_id, list(events))]

    @staticmethod
    def _keep_cluster_intact(
        state: ClusterState,
        cluster_id: str,
        events: Sequence[Event],
        current_hash: str,
    ) -> list[tuple[str, list[Event]]]:
        _record_split(state, cluster_id, [], current_hash)
        return [(cluster_id, list(events))]

    async def _get_reranker_scores(
        self,
        window: Sequence[Event],
        reranker: Reranker,
        cluster_id: str,
    ) -> list[float] | None:
        try:
            return await _adjacent_reranker_scores(window, reranker)
        except Exception:
            logger.exception(
                "Reranker split failed for cluster %s, keeping intact", cluster_id
            )
            if self._params.debug_fail_loudly:
                raise
            return None

    def _select_split_indices(
        self,
        scores: Sequence[float],
        time_gaps_seconds: Sequence[float],
    ) -> list[int]:
        normalized = _normalize_scores(scores)
        split_indices: set[int] = set()

        for i, score in enumerate(normalized):
            if score <= self._params.low_similarity_threshold:
                split_indices.add(i + 1)

        if self._params.time_gap_seconds is not None:
            for i, gap in enumerate(time_gaps_seconds):
                if gap > self._params.time_gap_seconds:
                    split_indices.add(i + 1)

        sims = np.asarray(normalized)
        if len(sims) >= 2:
            std = float(sims.std())
            if std > 0.0:
                mean = float(sims.mean())
                z_scores = (mean - sims) / std
                for i, z in enumerate(z_scores.tolist()):
                    if float(z) >= self._params.cohesion_drop_zscore:
                        split_indices.add(i + 1)

        return sorted(split_indices)

    def _replay_split(
        self,
        cluster_id: str,
        events: Sequence[Event],
        embeddings: Sequence[Sequence[float]],
        record: ClusterSplitRecord,
        state: ClusterState,
    ) -> list[tuple[str, list[Event]]]:
        """Re-apply a recorded split without rescoring."""
        resulting_ids = record.resulting_cluster_ids
        n_segments = len(resulting_ids)

        boundaries: list[int] = [0]
        search_from = 0
        for seg_id in resulting_ids[1:]:
            for i in range(search_from, len(events)):
                event = events[i]
                if state.event_to_cluster.get(event.uuid) == seg_id:
                    boundaries.append(i)
                    search_from = i
                    break
        boundaries.append(len(events))

        if len(boundaries) != n_segments + 1:
            return [(cluster_id, list(events))]

        pieces: list[tuple[str, list[Event]]] = []
        for seg_idx in range(n_segments):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1]
            seg_events = events[start:end]
            seg_embeddings = embeddings[start:end]
            seg_id = resulting_ids[seg_idx]

            if seg_events:
                arr = np.asarray(seg_embeddings, dtype=float)
                centroid = [float(x) for x in arr.mean(axis=0).tolist()]
                state.clusters[seg_id] = ClusterInfo(
                    centroid=centroid,
                    count=len(seg_events),
                    last_ts=max(e.timestamp for e in seg_events),
                )
            pieces.append((seg_id, list(seg_events)))

        return pieces


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #


def segment_cluster_id(original_id: str, segment_index: int) -> str:
    """Deterministic child-cluster id from an original + segment index."""
    raw = f"{original_id}:{segment_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def validate_split_indices(indices: Sequence[int], n: int) -> list[int]:
    """Validate split indices.  Must be sorted, unique, and in (0, n)."""
    try:
        as_ints = [int(x) for x in indices]
    except (TypeError, ValueError):
        logger.warning("Invalid split indices (non-integer): %s", indices)
        return []
    valid = sorted({i for i in as_ints if 0 < i < n})
    if len(valid) != len(as_ints):
        logger.warning("Split indices adjusted (original=%s, valid=%s)", as_ints, valid)
    return valid


def apply_cluster_split(
    cluster_id: str,
    events: Sequence[Event],
    embeddings: Sequence[Sequence[float]],
    split_indices: Sequence[int],
    state: ClusterState,
) -> tuple[list[tuple[str, list[Event]]], list[str]]:
    """Partition events at ``split_indices`` and update ``state``.

    The first segment keeps the original ``cluster_id``; subsequent
    segments get deterministic child ids.  Returns
    ``(pieces, resulting_ids)``.
    """
    boundaries = [0, *split_indices, len(events)]
    pieces: list[tuple[str, list[Event]]] = []
    resulting_ids: list[str] = []

    for seg_idx, (start, end) in enumerate(itertools.pairwise(boundaries)):
        seg_events = events[start:end]
        seg_embeddings = embeddings[start:end]
        seg_id = cluster_id if seg_idx == 0 else segment_cluster_id(cluster_id, seg_idx)

        arr = np.asarray(seg_embeddings, dtype=float)
        centroid = [float(x) for x in arr.mean(axis=0).tolist()]
        state.clusters[seg_id] = ClusterInfo(
            centroid=centroid,
            count=len(seg_events),
            last_ts=max(e.timestamp for e in seg_events),
        )
        for event in seg_events:
            state.event_to_cluster[event.uuid] = seg_id

        pieces.append((seg_id, list(seg_events)))
        resulting_ids.append(seg_id)

    logger.info(
        "Split cluster %s into %d segments: %s",
        cluster_id,
        len(pieces),
        resulting_ids,
    )
    return pieces, resulting_ids


def _collect_embeddings(
    events: Sequence[Event],
    cluster_embeddings: Mapping[UUID, Sequence[float]],
) -> tuple[list[Sequence[float]], list[UUID]]:
    ordered_embeddings = [cluster_embeddings[e.uuid] for e in events]
    event_uuids = [e.uuid for e in events]
    return ordered_embeddings, event_uuids


def _compute_signals(
    events: Sequence[Event],
    embeddings: Sequence[Sequence[float]],
) -> ContinuitySignals:
    adj_sims = _adjacent_similarities(embeddings)
    time_gaps: list[float] = []
    for i in range(len(events) - 1):
        gap = abs((events[i + 1].timestamp - events[i].timestamp).total_seconds())
        time_gaps.append(gap)
    return ContinuitySignals(
        adjacent_similarities=adj_sims,
        time_gaps_seconds=time_gaps,
        min_adjacent_similarity=min(adj_sims) if adj_sims else 1.0,
        max_time_gap_seconds=max(time_gaps) if time_gaps else 0.0,
    )


def _adjacent_similarities(
    embeddings: Sequence[Sequence[float]],
) -> list[float]:
    """Cosine similarities between consecutive embeddings."""
    result: list[float] = []
    for i in range(len(embeddings) - 1):
        a = np.asarray(embeddings[i], dtype=float)
        b = np.asarray(embeddings[i + 1], dtype=float)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        result.append(float(np.dot(a, b) / denom) if denom != 0.0 else 0.0)
    return result


async def _adjacent_reranker_scores(
    events: Sequence[Event],
    reranker: Reranker,
) -> list[float]:
    scores: list[float] = []
    for i in range(len(events) - 1):
        query = _format_reranker_message(events[i])
        candidate = _format_reranker_message(events[i + 1])
        pair_scores = await reranker.score(query, [candidate])
        scores.append(float(pair_scores[0]) if pair_scores else 0.0)
    return scores


def _format_reranker_message(event: Event) -> str:
    ts = event.timestamp.isoformat()
    body = event.body
    source = ""
    text = ""
    if isinstance(body, Content):
        if isinstance(body.context, MessageContext):
            source = body.context.source
        text = "\n".join(item.text for item in body.items if isinstance(item, Text))
    return f"[{ts}] {source}: {text}" if source else f"[{ts}] {text}"


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    min_score = float(min(scores))
    max_score = float(max(scores))
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(float(s) - min_score) / (max_score - min_score) for s in scores]


def _input_hash(event_uuids: Sequence[UUID]) -> str:
    raw = ",".join(str(u) for u in event_uuids)
    return hashlib.sha256(raw.encode()).hexdigest()


def _record_split(
    state: ClusterState,
    original_cluster_id: str,
    resulting_cluster_ids: Sequence[str],
    input_hash: str,
) -> None:
    state.split_records[original_cluster_id] = ClusterSplitRecord(
        original_cluster_id=original_cluster_id,
        resulting_cluster_ids=list(resulting_cluster_ids),
        input_hash=input_hash,
    )
