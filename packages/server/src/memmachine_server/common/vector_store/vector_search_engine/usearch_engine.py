"""USearch HNSW implementation of VectorSearchEngine."""

import asyncio
from collections.abc import Container, Iterable, Mapping, Sequence
from typing import ClassVar, override

import numpy as np
from usearch.index import Index, MetricKind

from memmachine_server.common.data_types import SimilarityMetric

from .vector_search_engine import SearchMatch, SearchResult, VectorSearchEngine


class USearchVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by USearch HNSW.

    Scores are pure metric values:
    - Cosine: cosine similarity (USearch returns ``1 - cos_sim``,
      converted back to ``cos_sim``).
    - Dot: inner product (USearch returns ``1 - dot``, converted back).
    - Euclidean: L2 squared distance (passed through as-is).
    """

    _METRIC_MAP: ClassVar[dict[SimilarityMetric, MetricKind]] = {
        SimilarityMetric.COSINE: MetricKind.Cos,
        SimilarityMetric.EUCLIDEAN: MetricKind.L2sq,
        SimilarityMetric.DOT: MetricKind.IP,
    }

    def __init__(self, *, ndim: int, metric: SimilarityMetric) -> None:
        """Initialize with vector dimensions and similarity metric."""
        usearch_metric = self._METRIC_MAP.get(metric)
        if usearch_metric is None:
            supported = ", ".join(m.value for m in self._METRIC_MAP)
            raise ValueError(
                f"USearch does not support {metric.value!r}. Supported: {supported}"
            )
        self._index = Index(ndim=ndim, metric=usearch_metric, dtype="f32")
        self._metric = metric
        self._lock = asyncio.Lock()

    def _distance_to_score(self, distance: float) -> float:
        """Convert a USearch distance to a pure metric score."""
        if self._metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT):
            return 1.0 - distance
        return distance

    def _sync_add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        keys_array = np.array(list(vectors.keys()), dtype=np.int64)
        vectors_array = np.array(list(vectors.values()), dtype=np.float32)
        self._index.add(keys_array, vectors_array)

    @override
    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        async with self._lock:
            await asyncio.to_thread(self._sync_add, vectors)

    def _sync_search_one(
        self,
        vector: Sequence[float],
        k: int,
        allowed_keys: Container[int] | None,
    ) -> SearchResult:
        query = np.array(vector, dtype=np.float32)
        if allowed_keys is None:
            results = self._index.search(query, k)
        else:
            # USearch has no native filter — full scan + post-filter.
            results = self._index.search(query, self._index.size)
        matches: list[SearchMatch] = []
        for key, dist in zip(results.keys, results.distances, strict=True):
            int_key = int(key)
            if allowed_keys is not None and int_key not in allowed_keys:
                continue
            matches.append(
                SearchMatch(key=int_key, score=self._distance_to_score(float(dist)))
            )
            if len(matches) >= k:
                break
        return SearchResult(matches=matches)

    def _sync_search(
        self,
        vectors: Sequence[Sequence[float]],
        k: int,
        allowed_keys: Container[int] | None,
    ) -> list[SearchResult]:
        return [self._sync_search_one(v, k, allowed_keys) for v in vectors]

    @override
    async def search(
        self,
        vectors: Iterable[Sequence[float]],
        k: int,
        *,
        allowed_keys: Container[int] | None = None,
    ) -> list[SearchResult]:
        vectors_list = list(vectors)
        if self._index.size == 0 or not vectors_list:
            return [SearchResult(matches=[]) for _ in vectors_list]
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_search, vectors_list, k, allowed_keys
            )

    def _sync_remove(self, keys: Iterable[int]) -> None:
        index = self._index
        for key in keys:
            # USearch silently ignores remove on nonexistent keys.
            index.remove(int(key))

    @override
    async def remove(self, keys: Iterable[int]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_remove, keys)

    @override
    async def save(self, path: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._index.save, path)

    @override
    async def load(self, path: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._index.load, path)
