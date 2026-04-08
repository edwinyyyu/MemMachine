"""hnswlib HNSW implementation of VectorSearchEngine."""

from collections.abc import Iterable, Sequence
from typing import ClassVar

import hnswlib  # ty: ignore[unresolved-import]  # C extension, no py.typed
import numpy as np

from memmachine_server.common.data_types import SimilarityMetric

from .vector_search_engine import KeyFilter, SearchResult, VectorSearchEngine


class HnswlibVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by hnswlib HNSW.

    Supports caller-provided integer keys, tombstone-based deletion,
    and in-place upserts via ``mark_deleted`` + ``add_items``.

    Filtered search uses hnswlib's native callback filter, which
    evaluates a predicate per candidate during graph traversal.

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

    def add(self, keys: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        if not keys:
            return
        keys_array = np.array(keys, dtype=np.int64)
        vectors_array = np.array(vectors, dtype=np.float32)

        # Upsert: mark_deleted + re-add for existing keys.
        for key in keys_array:
            k = int(key)
            if k in self._live_ids:
                self._index.mark_deleted(k)
                self._live_ids.discard(k)

        self._ensure_capacity(len(keys_array))
        self._index.add_items(vectors_array, keys_array)
        self._live_ids.update(int(k) for k in keys_array)

    def _raw_search(self, vector: Sequence[float], k: int) -> SearchResult:
        if not self._live_ids:
            return SearchResult(keys=[], scores=[])
        query = np.array([vector], dtype=np.float32)
        effective_k = min(k, len(self._live_ids))
        labels, distances = self._index.knn_query(query, k=effective_k, num_threads=1)
        return SearchResult(
            keys=[int(i) for i in labels[0]],
            scores=[self._distance_to_score(float(d)) for d in distances[0]],
        )

    def search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        key_filter: KeyFilter | None = None,
    ) -> SearchResult:
        """Search with optional native callback filtering."""
        if not self._live_ids:
            return SearchResult(keys=[], scores=[])
        query = np.array([vector], dtype=np.float32)
        effective_k = min(k, len(self._live_ids))
        try:
            if key_filter is None:
                labels, distances = self._index.knn_query(
                    query, k=effective_k, num_threads=1
                )
            else:
                labels, distances = self._index.knn_query(
                    query,
                    k=effective_k,
                    num_threads=1,
                    filter=lambda idx: idx in key_filter,
                )
        except RuntimeError:
            # hnswlib throws when filter excludes all candidates.
            return SearchResult(keys=[], scores=[])
        return SearchResult(
            keys=[int(i) for i in labels[0]],
            scores=[self._distance_to_score(float(d)) for d in distances[0]],
        )

    def remove(self, keys: Iterable[int]) -> None:
        for key in keys:
            if key in self._live_ids:
                self._index.mark_deleted(key)
                self._live_ids.discard(key)

    def __len__(self) -> int:
        """Return number of live vectors in the index."""
        return len(self._live_ids)

    def __contains__(self, key: int) -> bool:
        """Return whether the key is live in the index."""
        return key in self._live_ids

    def save(self, path: str) -> None:
        self._index.save_index(path)

    def load(self, path: str) -> None:
        self._index = hnswlib.Index(space=self._space, dim=self._ndim)
        self._index.load_index(path)
        self._index.set_ef(self._ef_search)
        # Reconstruct live IDs — get_ids_list includes tombstoned IDs,
        # so filter by attempting get_items (throws for deleted).
        self._live_ids = set()
        for label in self._index.get_ids_list():
            try:
                self._index.get_items([label])
                self._live_ids.add(int(label))
            except RuntimeError:
                pass
