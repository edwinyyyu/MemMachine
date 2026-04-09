"""hnswlib HNSW implementation of VectorSearchEngine."""

import asyncio
import contextlib
from collections.abc import Container, Iterable, Mapping, Sequence
from typing import ClassVar, override

import hnswlib  # ty: ignore[unresolved-import]  # C extension, no py.typed
import numpy as np

from memmachine_server.common.data_types import SimilarityMetric

from .vector_search_engine import SearchMatch, SearchResult, VectorSearchEngine


class HnswlibVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by hnswlib HNSW.

    Supports caller-provided integer keys, tombstone-based deletion,
    and in-place upserts via ``mark_deleted`` + ``add_items``.

    Scores are pure metric values:
    - Cosine: cosine similarity (hnswlib returns ``1 - cos_sim``
      as distance, converted back to ``cos_sim``).
    - Dot: inner product (hnswlib ``ip`` space returns ``1 - IP``
      as distance, converted back to ``IP``).
    - Euclidean: L2 squared distance (passed through as-is).
    """

    _SPACE_MAP: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "cosine",
        SimilarityMetric.EUCLIDEAN: "l2",
        SimilarityMetric.DOT: "ip",
    }

    _DEFAULT_M: ClassVar[int] = 16
    _DEFAULT_EF_CONSTRUCTION: ClassVar[int] = 200
    _DEFAULT_EF_SEARCH: ClassVar[int] = 200
    _DEFAULT_INITIAL_CAPACITY: ClassVar[int] = 1024
    _RESIZE_FACTOR: ClassVar[float] = 2.0

    def __init__(
        self,
        *,
        ndim: int,
        metric: SimilarityMetric,
        m: int = _DEFAULT_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        ef_search: int = _DEFAULT_EF_SEARCH,
        initial_capacity: int = _DEFAULT_INITIAL_CAPACITY,
    ) -> None:
        """Initialize with vector dimensions and similarity metric."""
        space = self._SPACE_MAP.get(metric)
        if space is None:
            supported = ", ".join(m.value for m in self._SPACE_MAP)
            raise ValueError(
                f"hnswlib does not support {metric.value!r}. Supported: {supported}"
            )
        self._ndim = ndim
        self._metric = metric
        self._space = space
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._index = hnswlib.Index(space=space, dim=ndim)
        self._index.init_index(
            max_elements=initial_capacity, ef_construction=ef_construction, M=m
        )
        self._index.set_ef(ef_search)
        self._live_ids: set[int] = set()
        self._lock = asyncio.Lock()

    def _distance_to_score(self, distance: float) -> float:
        """Convert an hnswlib distance to a pure metric score."""
        if self._metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT):
            return 1.0 - distance
        return distance

    def _ensure_capacity(self, needed: int) -> None:
        """Grow the index if there isn't room for ``needed`` more elements."""
        current_max = self._index.get_max_elements()
        current_count = self._index.element_count
        if current_count + needed <= current_max:
            return
        new_max = max(
            current_max + needed,
            int(current_max * HnswlibVectorSearchEngine._RESIZE_FACTOR),
        )
        self._index.resize_index(new_max)

    def _sync_add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        keys_array = np.array(list(vectors.keys()), dtype=np.int64)
        vectors_array = np.array(list(vectors.values()), dtype=np.float32)
        self._ensure_capacity(len(keys_array))
        self._index.add_items(vectors_array, keys_array)
        self._live_ids.update(int(k) for k in keys_array)

    @override
    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        async with self._lock:
            await asyncio.to_thread(self._sync_add, vectors)

    def _sync_search(
        self,
        vectors: Sequence[Sequence[float]],
        k: int,
        allowed_keys: Container[int] | None,
    ) -> list[SearchResult]:
        query = np.array(vectors, dtype=np.float32)
        effective_k = min(k, len(self._live_ids))
        try:
            if allowed_keys is None:
                all_labels, all_distances = self._index.knn_query(
                    query, k=effective_k, num_threads=1
                )
            else:
                all_labels, all_distances = self._index.knn_query(
                    query,
                    k=effective_k,
                    num_threads=1,
                    filter=lambda idx: idx in allowed_keys,
                )
        except RuntimeError:
            return [SearchResult(matches=[]) for _ in vectors]
        results: list[SearchResult] = []
        for labels, distances in zip(all_labels, all_distances, strict=True):
            matches = [
                SearchMatch(key=int(label), score=self._distance_to_score(float(dist)))
                for label, dist in zip(labels, distances, strict=True)
                if int(label) >= 0
            ]
            results.append(SearchResult(matches=matches))
        return results

    @override
    async def search(
        self,
        vectors: Iterable[Sequence[float]],
        k: int,
        *,
        allowed_keys: Container[int] | None = None,
    ) -> list[SearchResult]:
        vectors_list = list(vectors)
        if not self._live_ids or not vectors_list:
            return [SearchResult(matches=[]) for _ in vectors_list]
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_search, vectors_list, k, allowed_keys
            )

    def _sync_remove(self, keys: Iterable[int]) -> None:
        for key in keys:
            with contextlib.suppress(RuntimeError):
                self._index.mark_deleted(key)
            self._live_ids.discard(key)

    @override
    async def remove(self, keys: Iterable[int]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_remove, keys)

    @override
    async def save(self, path: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._index.save_index, path)

    @override
    async def load(self, path: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load, path)

    def _load(self, path: str) -> None:
        self._index = hnswlib.Index(space=self._space, dim=self._ndim)
        self._index.load_index(path)
        self._index.set_ef(self._ef_search)
        self._live_ids = set()
        for label in self._index.get_ids_list():
            try:
                self._index.get_items([label])
                self._live_ids.add(int(label))
            except RuntimeError:
                pass
