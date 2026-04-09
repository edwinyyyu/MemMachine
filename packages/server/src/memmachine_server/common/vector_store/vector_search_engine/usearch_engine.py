"""USearch HNSW implementation of VectorSearchEngine."""

import asyncio
import math
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
    - Euclidean: Euclidean distance [0, inf) (converted from L2 squared).
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
        match self._metric:
            case SimilarityMetric.COSINE | SimilarityMetric.DOT:
                return 1.0 - distance
            case SimilarityMetric.EUCLIDEAN:
                return math.sqrt(max(0.0, distance))
            case _:
                raise NotImplementedError(self._metric)

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

    _OVERFETCH_BASE: ClassVar[int] = 4

    def _sync_search_one(
        self,
        vector: Sequence[float],
        k: int,
        allowed_keys: Container[int] | None,
    ) -> SearchResult:
        query = np.array(vector, dtype=np.float32)
        if allowed_keys is None:
            results = self._index.search(query, k)
            return SearchResult(
                matches=[
                    SearchMatch(
                        key=int(key), score=self._distance_to_score(float(dist))
                    )
                    for key, dist in zip(results.keys, results.distances, strict=True)
                ]
            )

        # Geometric overfetch: 4x → 16x → 64x → ... → full scan.
        index_size = self._index.size
        factor = self._OVERFETCH_BASE
        while True:
            fetch_k = min(k * factor, index_size)
            results = self._index.search(query, fetch_k)
            matches: list[SearchMatch] = []
            for key, dist in zip(results.keys, results.distances, strict=True):
                int_key = int(key)
                if int_key not in allowed_keys:
                    continue
                matches.append(
                    SearchMatch(key=int_key, score=self._distance_to_score(float(dist)))
                )
                if len(matches) >= k:
                    break
            if len(matches) >= k or fetch_k >= index_size:
                return SearchResult(matches=matches)
            factor *= self._OVERFETCH_BASE

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

    @override
    async def get_vectors(self, keys: Iterable[int]) -> dict[int, list[float]]:
        keys_list = list(keys)
        if not keys_list:
            return {}
        keys_array = np.array(keys_list, dtype=np.int64)
        vectors = await asyncio.to_thread(self._index.get, keys_array)
        result: dict[int, list[float]] = {}
        for i, key in enumerate(keys_list):
            vec = vectors[i] if vectors is not None else None
            if vec is not None:
                result[key] = list(vec)
        return result

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
