"""USearch HNSW implementation of VectorSearchEngine."""

from collections.abc import Iterable, Sequence
from typing import ClassVar

import numpy as np
from usearch.index import Index, MetricKind

from memmachine_server.common.data_types import SimilarityMetric

from .vector_search_engine import SearchResult, VectorSearchEngine


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

    def _distance_to_score(self, distance: float) -> float:
        """Convert a USearch distance to a pure metric score."""
        if self._metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT):
            # USearch returns 1 - similarity for both Cos and IP metrics.
            return 1.0 - distance
        # Euclidean: L2sq distance is already the pure metric value.
        return distance

    def add(self, keys: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        if not keys:
            return
        index = self._index
        keys_array = np.array(keys, dtype=np.int64)
        vectors_array = np.array(vectors, dtype=np.float32)
        for key in keys_array:
            if index.count(int(key)) > 0:
                index.remove(int(key))
        index.add(keys_array, vectors_array)

    def _raw_search(self, vector: Sequence[float], k: int) -> SearchResult:
        query = np.array(vector, dtype=np.float32)
        results = self._index.search(query, k)
        return SearchResult(
            keys=[int(key) for key in results.keys],
            scores=[self._distance_to_score(float(d)) for d in results.distances],
        )

    def remove(self, keys: Iterable[int]) -> None:
        index = self._index
        for key in keys:
            if index.count(int(key)) > 0:
                index.remove(int(key))

    def __len__(self) -> int:
        """Return number of vectors in the index."""
        return self._index.size

    def __contains__(self, key: int) -> bool:
        """Return whether the key exists in the index."""
        return bool(self._index.count(int(key)) > 0)

    def save(self, path: str) -> None:
        self._index.save(path)

    def load(self, path: str) -> None:
        self._index.load(path)
