"""Tests for FAISSVectorSearchEngine."""

import math
from pathlib import Path

import pytest

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.vector_store.vector_search_engine import KeyFilter
from memmachine_server.common.vector_store.vector_search_engine.faiss_engine import (
    FAISSVectorSearchEngine,
)

NDIM = 3


def _normalize(v: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in v))
    return [x / magnitude for x in v]


class _SetFilter(KeyFilter):
    """Simple set-backed KeyFilter for testing."""

    def __init__(self, allowed: set[int]) -> None:
        self._allowed = allowed

    def __contains__(self, key: object) -> bool:
        return key in self._allowed


# -- Construction --


class TestConstruction:
    def test_supported_metrics(self):
        for metric in (
            SimilarityMetric.COSINE,
            SimilarityMetric.EUCLIDEAN,
            SimilarityMetric.DOT,
        ):
            engine = FAISSVectorSearchEngine(ndim=NDIM, metric=metric)
            assert len(engine) == 0

    def test_unsupported_metric_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.MANHATTAN)


# -- Add / len / contains --


class TestAdd:
    def test_add_single(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1], [[1.0, 0.0, 0.0]])
        assert len(engine) == 1
        assert 1 in engine
        assert 99 not in engine

    def test_add_batch(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([10, 20, 30], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert len(engine) == 3
        assert 10 in engine
        assert 20 in engine
        assert 30 in engine

    def test_add_replaces_existing_key(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1], [[1.0, 0.0, 0.0]])
        engine.add([1], [[0.0, 1.0, 0.0]])
        assert len(engine) == 1
        result = engine.search(_normalize([0, 1, 0]), k=1)
        assert result.keys[0] == 1
        assert result.scores[0] == pytest.approx(1.0, abs=0.01)

    def test_add_empty(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([], [])
        assert len(engine) == 0


# -- Remove --


class TestRemove:
    def test_remove_existing(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1, 2], [[1, 0, 0], [0, 1, 0]])
        engine.remove([1])
        assert len(engine) == 1
        assert 1 not in engine
        assert 2 in engine

    def test_remove_missing_is_ignored(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1], [[1, 0, 0]])
        engine.remove([99, 100])
        assert len(engine) == 1

    def test_remove_all(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1, 2, 3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine.remove([1, 2, 3])
        assert len(engine) == 0

    def test_remove_empty_iterable(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1], [[1, 0, 0]])
        engine.remove([])
        assert len(engine) == 1


# -- Search: Cosine --


class TestSearchCosine:
    def test_basic_knn(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add(
            [1, 2, 3],
            [_normalize([1, 0, 0]), _normalize([0, 1, 0]), _normalize([1, 1, 0])],
        )
        result = engine.search(_normalize([1, 0, 0]), k=3)
        assert len(result.keys) == 3
        assert result.keys[0] == 1
        assert result.scores[0] == pytest.approx(1.0, abs=0.01)
        assert result.scores[1] == pytest.approx(1.0 / math.sqrt(2), abs=0.01)

    def test_scores_are_cosine_similarity(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        v1 = _normalize([1, 0, 0])
        v2 = _normalize([0, 1, 0])
        engine.add([1, 2], [v1, v2])
        result = engine.search(v1, k=2)
        assert result.scores[0] == pytest.approx(1.0, abs=0.01)
        assert result.scores[1] == pytest.approx(0.0, abs=0.01)

    def test_scores_ordered_best_first(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add(
            [1, 2, 3],
            [_normalize([1, 0, 0]), _normalize([0, 1, 0]), _normalize([1, 1, 0])],
        )
        result = engine.search(_normalize([1, 0, 0]), k=3)
        for i in range(len(result.scores) - 1):
            assert result.scores[i] >= result.scores[i + 1]

    def test_k_larger_than_index(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1], [[1, 0, 0]])
        result = engine.search([1, 0, 0], k=10)
        assert len(result.keys) == 1

    def test_search_empty_index(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        result = engine.search([1, 0, 0], k=5)
        assert len(result.keys) == 0


# -- Search: Euclidean --


class TestSearchEuclidean:
    def test_scores_are_l2_distance(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.EUCLIDEAN)
        engine.add([1, 2], [[0, 0, 0], [3, 4, 0]])
        result = engine.search([0, 0, 0], k=2)
        assert result.keys[0] == 1
        assert result.scores[0] == pytest.approx(0.0, abs=0.01)
        assert result.scores[1] == pytest.approx(25.0, abs=0.01)

    def test_scores_ordered_best_first(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.EUCLIDEAN)
        engine.add([1, 2, 3], [[0, 0, 0], [1, 0, 0], [3, 4, 0]])
        result = engine.search([0, 0, 0], k=3)
        for i in range(len(result.scores) - 1):
            assert result.scores[i] <= result.scores[i + 1]


# -- Search: Dot product --


class TestSearchDot:
    def test_scores_are_inner_product(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.DOT)
        v1 = _normalize([1, 0, 0])
        v2 = _normalize([0, 1, 0])
        engine.add([1, 2], [v1, v2])
        result = engine.search(v1, k=2)
        assert result.keys[0] == 1
        assert result.scores[0] == pytest.approx(1.0, abs=0.01)
        assert result.scores[1] == pytest.approx(0.0, abs=0.01)

    def test_scores_ordered_best_first(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.DOT)
        v1 = _normalize([1, 0, 0])
        v2 = _normalize([1, 1, 0])
        v3 = _normalize([0, 1, 0])
        engine.add([1, 2, 3], [v1, v2, v3])
        result = engine.search(v1, k=3)
        for i in range(len(result.scores) - 1):
            assert result.scores[i] >= result.scores[i + 1]


# -- Filtered search --


class TestFilteredSearch:
    def test_filter_excludes_best_match(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add(
            [1, 2, 3],
            [_normalize([1, 0, 0]), _normalize([0, 1, 0]), _normalize([1, 1, 0])],
        )
        # Exclude key 1 (best match for [1,0,0])
        result = engine.search(
            _normalize([1, 0, 0]), k=2, key_filter=_SetFilter({2, 3})
        )
        assert 1 not in result.keys
        assert len(result.keys) == 2

    def test_filter_all_excluded(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1, 2], [_normalize([1, 0, 0]), _normalize([0, 1, 0])])
        result = engine.search(_normalize([1, 0, 0]), k=2, key_filter=_SetFilter(set()))
        assert len(result.keys) == 0

    def test_filter_with_single_allowed(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add(
            [1, 2, 3],
            [_normalize([1, 0, 0]), _normalize([0, 1, 0]), _normalize([0, 0, 1])],
        )
        result = engine.search(_normalize([1, 0, 0]), k=3, key_filter=_SetFilter({2}))
        assert result.keys == [2]

    def test_filter_none_is_unfiltered(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1, 2], [_normalize([1, 0, 0]), _normalize([0, 1, 0])])
        result = engine.search(_normalize([1, 0, 0]), k=2, key_filter=None)
        assert len(result.keys) == 2


# -- Persistence --


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([1, 2], [_normalize([1, 0, 0]), _normalize([0, 1, 0])])

        path = str(tmp_path / "test.faiss")
        engine.save(path)

        engine2 = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine2.load(path)

        assert len(engine2) == 2
        assert 1 in engine2
        assert 2 in engine2

        result = engine2.search(_normalize([1, 0, 0]), k=2)
        assert result.keys[0] == 1
        assert result.scores[0] == pytest.approx(1.0, abs=0.01)


# -- SearchResult types --


class TestSearchResultTypes:
    def test_keys_are_ints(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([42], [[1, 0, 0]])
        result = engine.search([1, 0, 0], k=1)
        assert isinstance(result.keys[0], int)

    def test_scores_are_floats(self):
        engine = FAISSVectorSearchEngine(ndim=NDIM, metric=SimilarityMetric.COSINE)
        engine.add([42], [[1, 0, 0]])
        result = engine.search([1, 0, 0], k=1)
        assert isinstance(result.scores[0], float)
