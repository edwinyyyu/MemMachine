"""FAISS HNSW implementation of VectorSearchEngine."""

from collections.abc import Iterable, Sequence
from typing import ClassVar

import faiss
import numpy as np

from memmachine_server.common.data_types import SimilarityMetric

from .vector_search_engine import KeyFilter, SearchResult, VectorSearchEngine


class FAISSVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by FAISS IndexHNSWFlat.

    Uses ``IndexIDMap`` to support caller-provided integer keys.
    Filtering uses ``PyCallbackIDSelector`` which is fast in FAISS.

    FAISS HNSW does not support ``remove_ids``, so deletions are
    handled via tombstones: deleted keys are tracked in a set and
    excluded automatically from all search results.

    Scores are pure metric values:
    - Cosine: cosine similarity (FAISS uses inner product on normalized
      vectors; score = -distance since FAISS IP returns negative similarity).
    - Dot: inner product (score = -distance).
    - Euclidean: L2 distance (score = distance, passed through as-is).
    """

    _DEFAULT_M: ClassVar[int] = 16
    _DEFAULT_EF_CONSTRUCTION: ClassVar[int] = 200
    _DEFAULT_EF_SEARCH: ClassVar[int] = 50

    def __init__(
        self,
        *,
        ndim: int,
        metric: SimilarityMetric,
        m: int = _DEFAULT_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        ef_search: int = _DEFAULT_EF_SEARCH,
    ) -> None:
        """Initialize with vector dimensions and similarity metric."""
        if metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT):
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == SimilarityMetric.EUCLIDEAN:
            faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError(
                f"FAISS does not support {metric.value!r}. "
                f"Supported: cosine, dot, euclidean"
            )

        hnsw_index = faiss.IndexHNSWFlat(ndim, m, faiss_metric)
        hnsw_index.hnsw.efConstruction = ef_construction
        hnsw_index.hnsw.efSearch = ef_search
        self._index = faiss.IndexIDMap(hnsw_index)
        self._hnsw_index = hnsw_index
        self._metric = metric
        self._ndim = ndim
        self._live_ids: set[int] = set()
        self._deleted_ids: set[int] = set()

    @staticmethod
    def _to_scores(distances: np.ndarray) -> list[float]:
        """Convert FAISS distances to pure metric scores.

        FAISS METRIC_INNER_PRODUCT returns the actual inner product
        (higher = more similar).  METRIC_L2 returns squared L2 distance.
        Both are already the pure metric values.
        """
        return [float(d) for d in distances]

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors in-place for cosine similarity."""
        if self._metric == SimilarityMetric.COSINE:
            faiss.normalize_L2(vectors)
        return vectors

    def _tombstone_filter(self) -> faiss.IDSelector | None:
        """Build an IDSelector that excludes tombstoned keys, or None."""
        if not self._deleted_ids:
            return None
        return faiss.PyCallbackIDSelector(lambda key: key not in self._deleted_ids)

    def _search_params(
        self, key_filter: KeyFilter | None
    ) -> faiss.SearchParameters | None:
        """Build FAISS SearchParameters combining tombstones and key_filter."""
        deleted = self._deleted_ids

        if not deleted and key_filter is None:
            return None

        if key_filter is None:
            selector = faiss.PyCallbackIDSelector(lambda key: key not in deleted)
        elif not deleted:
            selector = faiss.PyCallbackIDSelector(lambda key: key in key_filter)
        else:
            selector = faiss.PyCallbackIDSelector(
                lambda key: key not in deleted and key in key_filter
            )

        params = faiss.SearchParametersHNSW()
        params.sel = selector
        return params

    def add(self, keys: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        if not keys:
            return
        keys_array = np.array(keys, dtype=np.int64)
        vectors_array = self._normalize(np.array(vectors, dtype=np.float32))

        # FAISS HNSW does not support remove_ids, so upserts create
        # duplicate entries.  Search results are deduplicated by key,
        # keeping the best (most recent) score.
        # Clear any tombstones for re-added keys.
        for key in keys_array:
            self._deleted_ids.discard(int(key))

        self._index.add_with_ids(vectors_array, keys_array)  # ty: ignore[missing-argument]  # SWIG stub has C signature
        self._live_ids.update(int(k) for k in keys_array)

    def _faiss_search(
        self,
        query: np.ndarray,
        k: int,
        params: faiss.SearchParameters | None,
    ) -> SearchResult:
        """Run FAISS search and deduplicate results by key."""
        # Request extra results to account for duplicates from upserts.
        fetch_k = min(k * 2, self._index.ntotal) if self._index.ntotal > 0 else k
        if params is not None:
            distances, ids = self._index.search(query, fetch_k, params=params)  # ty: ignore[missing-argument]  # SWIG stub has C signature
        else:
            distances, ids = self._index.search(query, fetch_k)  # ty: ignore[missing-argument]  # SWIG stub has C signature

        # Deduplicate: first occurrence of each key has the best score.
        seen: set[int] = set()
        result_keys: list[int] = []
        result_scores: list[float] = []
        for raw_id, dist in zip(ids[0], distances[0], strict=True):
            if raw_id < 0:
                continue
            key = int(raw_id)
            if key in seen:
                continue
            seen.add(key)
            result_keys.append(key)
            result_scores.append(float(dist))
            if len(result_keys) >= k:
                break
        return SearchResult(keys=result_keys, scores=result_scores)

    def _raw_search(self, vector: Sequence[float], k: int) -> SearchResult:
        query = np.array([vector], dtype=np.float32)
        self._normalize(query)
        return self._faiss_search(query, k, self._search_params(key_filter=None))

    def search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        key_filter: KeyFilter | None = None,
    ) -> SearchResult:
        """Search with optional native filtering via PyCallbackIDSelector."""
        query = np.array([vector], dtype=np.float32)
        self._normalize(query)
        return self._faiss_search(query, k, self._search_params(key_filter))

    def remove(self, keys: Iterable[int]) -> None:
        for key in keys:
            if key in self._live_ids:
                self._deleted_ids.add(key)
                self._live_ids.discard(key)

    def __len__(self) -> int:
        """Return number of live vectors in the index."""
        return len(self._live_ids)

    def __contains__(self, key: int) -> bool:
        """Return whether the key is live in the index."""
        return key in self._live_ids

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        loaded = faiss.read_index(path)
        self._index = loaded
        # Reconstruct the live ID set from the loaded index
        self._live_ids = set()
        self._deleted_ids = set()
        if loaded.ntotal > 0:
            id_map = faiss.downcast_index(loaded)
            ids = faiss.vector_to_array(id_map.id_map)
            self._live_ids = {int(i) for i in ids}
