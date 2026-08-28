"""USearch HNSW implementation of VectorSearchEngine."""

import asyncio
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import ClassVar, override

import numpy as np
import numpy.typing as npt
from usearch.index import Index, MetricKind

from memmachine_core.common.rw_locks import AsyncRWLock

from .index_persistence import atomic_index_write, clear_stale_index_temp
from .scoring import cosine_similarities, top_k_matches
from .vector_search_engine import SearchMatch, SearchResult, VectorSearchEngine


class USearchVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by USearch HNSW."""

    _METRIC_KIND: ClassVar[MetricKind] = MetricKind.Cos

    _DEFAULT_M: ClassVar[int] = 16
    _DEFAULT_EF_CONSTRUCTION: ClassVar[int] = 128
    _DEFAULT_EF_SEARCH: ClassVar[int] = 128

    def __init__(
        self,
        *,
        num_dimensions: int,
        m: int = _DEFAULT_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        ef_search: int = _DEFAULT_EF_SEARCH,
    ) -> None:
        """Initialize."""
        self._index = Index(
            ndim=num_dimensions,
            metric=self._METRIC_KIND,
            dtype="f32",
            connectivity=m,
            expansion_add=ef_construction,
            expansion_search=ef_search,
        )

        self._lock = AsyncRWLock()

    @staticmethod
    def _distance_to_cosine_similarity(distance: float) -> float:
        """Convert a USearch cosine distance to a cosine similarity."""
        return 1.0 - distance

    @override
    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_add, vectors)

    def _sync_add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        keys_array = np.array(list(vectors.keys()), dtype=np.int64)
        vectors_array = np.array(list(vectors.values()), dtype=np.float32)
        self._index.add(keys_array, vectors_array)

    @override
    async def search(
        self,
        vectors: Iterable[Sequence[float]],
        *,
        limit: int,
        allowlist: Collection[int] | None = None,
    ) -> list[SearchResult]:
        vectors = list(vectors)
        if (
            self._index.size == 0
            or not vectors
            or (allowlist is not None and not allowlist)
        ):
            return [SearchResult(matches=[]) for _ in vectors]

        async with self._lock.read_lock():
            return await asyncio.to_thread(self._sync_search, vectors, limit, allowlist)

    def _sync_search(
        self,
        vectors: Sequence[Sequence[float]],
        limit: int,
        allowlist: Collection[int] | None,
    ) -> list[SearchResult]:
        if allowlist is not None:
            return self._sync_search_allowlist(vectors, limit, allowlist)

        query = np.array(vectors, dtype=np.float32)
        fetch_limit = min(limit, self._index.size)

        results = self._index.search(query, fetch_limit)
        all_keys = np.atleast_2d(results.keys)
        all_distances = np.atleast_2d(results.distances)

        search_results: list[SearchResult] = []
        for keys, distances in zip(all_keys, all_distances, strict=True):
            matches = [
                SearchMatch(
                    key=int(key),
                    cosine_similarity=self._distance_to_cosine_similarity(float(dist)),
                )
                for key, dist in zip(keys, distances, strict=True)
                if int(key) >= 0
            ]
            search_results.append(SearchResult(matches=matches))
        return search_results

    def _sync_search_allowlist(
        self,
        vectors: Sequence[Sequence[float]],
        limit: int,
        allowlist: Collection[int],
    ) -> list[SearchResult]:
        """Exact: gather the allowed vectors and score them directly."""
        present_keys, matrix = self._sync_gather_vectors(allowlist)
        if not present_keys:
            return [SearchResult(matches=[]) for _ in vectors]

        return [
            SearchResult(
                matches=top_k_matches(query_vector, present_keys, matrix, limit)
            )
            for query_vector in vectors
        ]

    @override
    async def get_cosine_similarities(
        self,
        query_vector: Sequence[float],
        keys: Iterable[int],
    ) -> dict[int, float]:
        async with self._lock.read_lock():
            present_keys, matrix = await asyncio.to_thread(
                self._sync_gather_vectors, keys
            )
        if not present_keys:
            return {}
        similarities = cosine_similarities(query_vector, matrix)
        return {
            key: float(similarity)
            for key, similarity in zip(present_keys, similarities, strict=True)
        }

    def _sync_gather_vectors(
        self, keys: Iterable[int]
    ) -> tuple[list[int], npt.NDArray[np.float32]]:
        """Gather stored vectors by key as a float32 matrix; missing keys drop."""
        keys = list(dict.fromkeys(int(key) for key in keys))
        empty = np.empty((0, self._index.ndim), dtype=np.float32)
        if not keys:
            return [], empty

        gathered = self._index.get(np.array(keys, dtype=np.int64))
        if gathered is None:
            return [], empty
        if isinstance(gathered, np.ndarray):
            return keys, np.asarray(gathered, dtype=np.float32).reshape(len(keys), -1)

        present_keys: list[int] = []
        rows: list[np.ndarray] = []
        for key, row in zip(keys, gathered, strict=True):
            if row is not None:
                present_keys.append(key)
                rows.append(np.asarray(row, dtype=np.float32))
        if not present_keys:
            return [], empty
        return present_keys, np.vstack(rows)

    @override
    async def remove(self, keys: Iterable[int]) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_remove, keys)

    def _sync_remove(self, keys: Iterable[int]) -> None:
        index = self._index
        for key in keys:
            index.remove(int(key))

    @override
    async def save(self, path: str) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_save, path)

    def _sync_save(self, path: str) -> None:
        with atomic_index_write(path) as temp_path:
            self._index.save(temp_path)

    @override
    async def load(self, path: str) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_load, path)

    def _sync_load(self, path: str) -> None:
        clear_stale_index_temp(path)
        self._index.load(path)
