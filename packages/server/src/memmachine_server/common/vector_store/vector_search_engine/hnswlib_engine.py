"""hnswlib HNSW implementation of VectorSearchEngine."""

import asyncio
import contextlib
import math
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from typing import ClassVar, override

import hnswlib  # ty: ignore[unresolved-import]  # C extension, no py.typed
import numpy as np

from memmachine_server.common.data_types import SimilarityMetric

from .vector_search_engine import SearchMatch, SearchResult, VectorSearchEngine


class HnswlibVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by hnswlib HNSW."""

    _SPACE_MAP: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "cosine",
        SimilarityMetric.EUCLIDEAN: "l2",
        SimilarityMetric.DOT: "ip",
    }

    _DEFAULT_M: ClassVar[int] = 16
    _DEFAULT_EF_CONSTRUCTION: ClassVar[int] = 128
    _DEFAULT_EF_SEARCH: ClassVar[int] = 128

    _DEFAULT_INITIAL_CAPACITY: ClassVar[int] = 1024
    _RESIZE_FACTOR: ClassVar[float] = 2.0

    def __init__(
        self,
        *,
        num_dimensions: int,
        similarity_metric: SimilarityMetric,
        m: int = _DEFAULT_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        ef_search: int = _DEFAULT_EF_SEARCH,
        initial_capacity: int = _DEFAULT_INITIAL_CAPACITY,
    ) -> None:
        """Initialize."""
        hnswlib_space = self._SPACE_MAP.get(similarity_metric)
        if hnswlib_space is None:
            supported = ", ".join(
                similarity_metric.value for similarity_metric in self._SPACE_MAP
            )
            raise ValueError(
                f"hnswlib does not support {similarity_metric.value!r}. Supported: {supported}"
            )

        self._num_dimensions = num_dimensions
        self._similarity_metric = similarity_metric

        self._space = hnswlib_space
        self._ef_search = ef_search

        self._index = hnswlib.Index(space=hnswlib_space, dim=num_dimensions)
        self._index.init_index(
            max_elements=initial_capacity, ef_construction=ef_construction, M=m
        )
        self._index.set_ef(ef_search)

        self._lock = asyncio.Lock()

    def _distance_to_score(self, distance: float) -> float:
        """Convert an hnswlib distance to a pure metric score."""
        match self._similarity_metric:
            case SimilarityMetric.COSINE | SimilarityMetric.DOT:
                return 1.0 - distance
            case SimilarityMetric.EUCLIDEAN:
                return math.sqrt(max(0.0, distance))
            case _:
                raise NotImplementedError(self._similarity_metric)

    def _ensure_capacity(self, needed: int) -> None:
        """Grow the index if there isn't room for `needed` more elements."""
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

    @override
    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        async with self._lock:
            await asyncio.to_thread(self._sync_add, vectors)

    @override
    async def search(
        self,
        vectors: Iterable[Sequence[float]],
        *,
        limit: int | None = None,
        allowed_keys: Container[int] | None = None,
    ) -> list[SearchResult]:
        vectors = list(vectors)
        if self._index.element_count == 0 or not vectors:
            return [SearchResult(matches=[]) for _ in vectors]

        async with self._lock:
            return await asyncio.to_thread(
                self._sync_search, vectors, limit, allowed_keys
            )

    def _sync_search(
        self,
        vectors: Iterable[Sequence[float]],
        limit: int | None,
        allowed_keys: Container[int] | None,
    ) -> list[SearchResult]:
        vectors = list(vectors)
        if not vectors:
            return []

        query = np.array(vectors, dtype=np.float32)

        effective_limit = (
            min(limit, self._index.element_count)
            if limit is not None
            else self._index.element_count
        )
        if effective_limit <= 0:
            return [SearchResult(matches=[]) for _ in vectors]

        filter_fn = (
            (lambda idx: idx in allowed_keys) if allowed_keys is not None else None
        )
        return self._binary_search_query(query, effective_limit, filter_fn)

    def _binary_search_query(
        self,
        query: np.ndarray,
        k: int,
        filter_fn: Callable[[int], bool] | None,
    ) -> list[SearchResult]:
        # Fast path: requested k works for the full batch.
        result = self._try_knn_query(query, k, filter_fn)
        if result is not None:
            return self._build_search_results_from_knn(result)

        # Binary search with a single probe to find max fillable k.
        max_k = self._find_max_k(query[0:1], k, filter_fn)
        if max_k == 0:
            return [SearchResult(matches=[]) for _ in query]

        # Get candidate keys from the probe (these are all passing vectors).
        probe_result = self._try_knn_query(query[0:1], max_k, filter_fn)
        if probe_result is None:
            return [SearchResult(matches=[]) for _ in query]

        candidate_keys = [int(label) for label in probe_result[0][0] if label >= 0]
        if not candidate_keys:
            return [SearchResult(matches=[]) for _ in query]

        return self._build_search_results_from_candidates(query, candidate_keys)

    def _try_knn_query(
        self,
        query: np.ndarray,
        k: int,
        filter_fn: Callable[[int], bool] | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Run knn_query, returning None if hnswlib throws."""
        try:
            kwargs: dict = {"k": k, "num_threads": 1}
            if filter_fn is not None:
                kwargs["filter"] = filter_fn
            return self._index.knn_query(query, **kwargs)
        except RuntimeError:
            return None

    def _build_search_results_from_knn(
        self,
        result: tuple[np.ndarray, np.ndarray],
    ) -> list[SearchResult]:
        all_labels, all_distances = result
        results: list[SearchResult] = []
        for labels, distances in zip(all_labels, all_distances, strict=True):
            matches = [
                SearchMatch(key=int(label), score=self._distance_to_score(float(dist)))
                for label, dist in zip(labels, distances, strict=True)
                if int(label) >= 0
            ]
            results.append(SearchResult(matches=matches))
        return results

    def _find_max_k(
        self, probe: np.ndarray, k: int, filter_fn: Callable[[int], bool] | None
    ) -> int:
        """Binary search for the max k that hnswlib can fill, using a single probe."""
        low, high = 1, k
        max_fillable = 0
        while low <= high:
            mid = (low + high) // 2
            if self._try_knn_query(probe, mid, filter_fn) is not None:
                max_fillable = mid
                low = mid + 1
            else:
                high = mid - 1

        return max_fillable

    def _build_search_results_from_candidates(
        self, query_vectors: np.ndarray, candidate_keys: list[int]
    ) -> list[SearchResult]:
        candidate_vectors = np.array(
            self._index.get_items(candidate_keys), dtype=np.float32
        )

        # Precompute normalized candidate vectors once for cosine.
        normalized_candidates: np.ndarray | None = None
        if self._similarity_metric == SimilarityMetric.COSINE:
            candidate_norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
            candidate_norms = np.maximum(candidate_norms, 1e-10)
            normalized_candidates = candidate_vectors / candidate_norms

        results: list[SearchResult] = []
        for query_vector in query_vectors:
            match self._similarity_metric:
                case SimilarityMetric.COSINE:
                    assert normalized_candidates is not None
                    query_norm = max(float(np.linalg.norm(query_vector)), 1e-10)
                    normalized_query = query_vector / query_norm
                    distances = 1.0 - normalized_candidates @ normalized_query
                case SimilarityMetric.DOT:
                    distances = 1.0 - candidate_vectors @ query_vector
                case SimilarityMetric.EUCLIDEAN:
                    difference = candidate_vectors - query_vector
                    distances = np.einsum("ij,ij->i", difference, difference)
                case _:
                    raise NotImplementedError(self._similarity_metric)

            sorted_indices = np.argsort(distances)
            matches = [
                SearchMatch(
                    key=candidate_keys[int(index)],
                    score=self._distance_to_score(float(distances[index])),
                )
                for index in sorted_indices
            ]
            results.append(SearchResult(matches=matches))
        return results

    @override
    async def get_vectors(self, keys: Iterable[int]) -> dict[int, list[float]]:
        return await asyncio.to_thread(self._sync_get_vectors, keys)

    def _sync_get_vectors(self, keys: Iterable[int]) -> dict[int, list[float]]:
        keys = list(keys)
        if not keys:
            return {}

        try:
            vectors = self._index.get_items(keys)
            return {
                key: list(vector) for key, vector in zip(keys, vectors, strict=True)
            }
        except RuntimeError:
            # Fallback: a deleted key was requested. Fetch one by one.
            result: dict[int, list[float]] = {}
            for key in keys:
                with contextlib.suppress(RuntimeError):
                    result[key] = list(self._index.get_items([key])[0])
            return result

    @override
    async def remove(self, keys: Iterable[int]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_remove, keys)

    def _sync_remove(self, keys: Iterable[int]) -> None:
        for key in keys:
            with contextlib.suppress(RuntimeError):
                self._index.mark_deleted(key)

    @override
    async def save(self, path: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._index.save_index, path)

    def _load(self, path: str) -> None:
        self._index = hnswlib.Index(space=self._space, dim=self._num_dimensions)
        self._index.load_index(path)
        self._index.set_ef(self._ef_search)

    @override
    async def load(self, path: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load, path)
