"""FAISS HNSW implementation of VectorSearchEngine."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

import faiss
import numpy as np

from memmachine_server.common.data_types import SimilarityMetric

from .vector_search_engine import KeyFilter, SearchResult, VectorSearchEngine


class FAISSVectorSearchEngine(VectorSearchEngine):
    """Vector search engine backed by FAISS IndexHNSWFlat.

    Uses ``IndexIDMap`` with auto-incrementing internal labels to avoid
    duplicate FAISS IDs.  A ``key → label`` mapping translates between
    caller-provided keys and FAISS-internal labels.

    FAISS HNSW does not support ``remove_ids``, so deletions and upserts
    are handled via tombstones: the old label is marked deleted and a
    fresh label is assigned.

    Filtering uses ``PyCallbackIDSelector`` which evaluates a predicate
    per candidate during HNSW graph traversal.

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
        self._next_label: int = 0
        self._key_to_label: dict[int, int] = {}
        self._label_to_key: dict[int, int] = {}
        self._deleted_labels: set[int] = set()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors in-place for cosine similarity."""
        if self._metric == SimilarityMetric.COSINE:
            faiss.normalize_L2(vectors)
        return vectors

    def _search_params(
        self, key_filter: KeyFilter | None
    ) -> faiss.SearchParameters | None:
        """Build FAISS SearchParameters combining tombstones and key_filter."""
        deleted = self._deleted_labels
        label_to_key = self._label_to_key

        if not deleted and key_filter is None:
            return None

        if key_filter is None:
            selector = faiss.PyCallbackIDSelector(lambda label: label not in deleted)
        elif not deleted:
            selector = faiss.PyCallbackIDSelector(
                lambda label: label_to_key.get(label) in key_filter
            )
        else:
            selector = faiss.PyCallbackIDSelector(
                lambda label: (
                    label not in deleted and label_to_key.get(label) in key_filter
                )
            )

        params = faiss.SearchParametersHNSW()
        params.sel = selector
        return params

    def add(self, keys: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        if not keys:
            return
        vectors_array = self._normalize(np.array(vectors, dtype=np.float32))

        labels: list[int] = []
        for key in keys:
            # Tombstone old label on upsert
            if key in self._key_to_label:
                old_label = self._key_to_label[key]
                self._deleted_labels.add(old_label)
                del self._label_to_key[old_label]

            label = self._next_label
            self._next_label += 1
            self._key_to_label[key] = label
            self._label_to_key[label] = key
            labels.append(label)

        labels_array = np.array(labels, dtype=np.int64)
        self._index.add_with_ids(vectors_array, labels_array)  # ty: ignore[missing-argument]  # SWIG stub has C signature

    def _faiss_search(
        self,
        query: np.ndarray,
        k: int,
        params: faiss.SearchParameters | None,
    ) -> SearchResult:
        """Run FAISS search and map internal labels to external keys."""
        if self._index.ntotal == 0:
            return SearchResult(keys=[], scores=[])

        fetch_k = min(k, self._index.ntotal)
        if params is not None:
            distances, ids = self._index.search(query, fetch_k, params=params)  # ty: ignore[missing-argument]  # SWIG stub has C signature
        else:
            distances, ids = self._index.search(query, fetch_k)  # ty: ignore[missing-argument]  # SWIG stub has C signature

        label_to_key = self._label_to_key
        deleted = self._deleted_labels
        result_keys: list[int] = []
        result_scores: list[float] = []
        for raw_id, dist in zip(ids[0], distances[0], strict=True):
            if raw_id < 0:
                continue
            label = int(raw_id)
            if label in deleted:
                continue
            key = label_to_key.get(label)
            if key is None:
                continue
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
            if key in self._key_to_label:
                label = self._key_to_label.pop(key)
                self._deleted_labels.add(label)
                self._label_to_key.pop(label, None)

    def __len__(self) -> int:
        """Return number of live vectors in the index."""
        return len(self._key_to_label)

    def __contains__(self, key: int) -> bool:
        """Return whether the key is live in the index."""
        return key in self._key_to_label

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)
        meta = {
            "next_label": self._next_label,
            "key_to_label": {str(k): v for k, v in self._key_to_label.items()},
            "deleted_labels": sorted(self._deleted_labels),
        }
        with Path(f"{path}.meta").open("w") as f:
            json.dump(meta, f)

    def load(self, path: str) -> None:
        self._index = faiss.read_index(path)
        with Path(f"{path}.meta").open() as f:
            meta = json.load(f)
        self._next_label = meta["next_label"]
        self._key_to_label = {int(k): v for k, v in meta["key_to_label"].items()}
        self._label_to_key = {v: int(k) for k, v in meta["key_to_label"].items()}
        self._deleted_labels = set(meta["deleted_labels"])
