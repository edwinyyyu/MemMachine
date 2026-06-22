"""Tests for TurboVecFreshVectorSearchEngine.

The Rust crate's test suite covers the FreshIndex storage model in depth
(WAL durability, crash cleanup, incremental maintenance, oracle parity);
these tests cover the engine wiring: the directory-shaped save/load
contract, knob passthrough, and the recall levers through the engine API.
"""

import math
from pathlib import Path

import pytest

pytest.importorskip("turbovec")

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.vector_store.vector_search_engine.turbovec_fresh_engine import (
    TurboVecFreshVectorSearchEngine,
)

NDIM = 8

QUANT_ABS = 0.05


def _normalize(v: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in v))
    return [x / magnitude for x in v]


def _one_hot(index: int, value: float = 1.0) -> list[float]:
    vector = [0.0] * NDIM
    vector[index] = value
    return vector


async def _search_one(engine, vector, limit=10, **kwargs):
    results = await engine.search([vector], limit=limit, **kwargs)
    return results[0]


def _make_engine(metric=SimilarityMetric.COSINE, **kwargs):
    return TurboVecFreshVectorSearchEngine(
        num_dimensions=NDIM, similarity_metric=metric, **kwargs
    )


class TestConstruction:
    def test_supported_metrics(self):
        for metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT):
            _make_engine(metric)

    def test_unsupported_metric_raises(self):
        with pytest.raises(NotImplementedError, match="does not support"):
            _make_engine(SimilarityMetric.EUCLIDEAN)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_add_search_remove_before_save(self):
        engine = _make_engine()
        await engine.add(
            {
                1: _normalize(_one_hot(0)),
                2: _normalize(_one_hot(1)),
                3: _normalize([1, 1, 0, 0, 0, 0, 0, 0]),
            }
        )
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=3)
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)

        await engine.remove([1])
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=3)
        assert 1 not in {m.key for m in result.matches}

    @pytest.mark.asyncio
    async def test_allowed_keys_restricts_results(self):
        engine = _make_engine()
        await engine.add(
            {
                1: _normalize(_one_hot(0)),
                2: _normalize(_one_hot(1)),
                3: _normalize(_one_hot(2)),
            }
        )
        result = await _search_one(
            engine, _normalize(_one_hot(0)), limit=3, allowed_keys={2, 3}
        )
        assert {m.key for m in result.matches} == {2, 3}

    @pytest.mark.asyncio
    async def test_get_vectors_raises_without_store_vectors(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        with pytest.raises(NotImplementedError, match="store_vectors=True"):
            await engine.get_vectors([1])


class TestPersistence:
    @pytest.mark.asyncio
    async def test_incremental_saves_and_load(self, tmp_path: Path):
        engine = _make_engine()
        path = str(tmp_path / "index.idx")  # becomes a directory
        await engine.add({1: _normalize(_one_hot(0)), 2: _normalize(_one_hot(1))})
        await engine.save(path)
        await engine.add({3: _normalize(_one_hot(2))})
        await engine.remove([1])
        await engine.save(path)

        live = await _search_one(engine, _normalize(_one_hot(0)), limit=5)
        assert {m.key for m in live.matches} == {2, 3}

        engine2 = _make_engine()
        await engine2.load(path)
        loaded = await _search_one(engine2, _normalize(_one_hot(0)), limit=5)
        assert {m.key for m in loaded.matches} == {2, 3}
        assert [m.score for m in loaded.matches] == [m.score for m in live.matches]

    @pytest.mark.asyncio
    async def test_unsaved_mutations_survive_via_wal(self, tmp_path: Path):
        path = str(tmp_path / "index.idx")
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        await engine.save(path)
        # Mutations after the save, never flushed.
        await engine.add({2: _normalize(_one_hot(1))})
        await engine.remove([1])
        del engine  # "crash"

        engine2 = _make_engine()
        await engine2.load(path)
        result = await _search_one(engine2, _normalize(_one_hot(1)), limit=5)
        assert {m.key for m in result.matches} == {2}

    @pytest.mark.asyncio
    async def test_repeated_upsert_cycles(self, tmp_path: Path):
        path = str(tmp_path / "index.idx")
        engine = _make_engine()
        for index in range(4):
            await engine.remove([7])
            await engine.add({7: _normalize(_one_hot(index))})
            await engine.save(path)
            engine = _make_engine()
            await engine.load(path)
        result = await _search_one(engine, _normalize(_one_hot(3)), limit=1)
        assert result.matches[0].key == 7


class TestPartitionedAndLevers:
    @pytest.mark.asyncio
    async def test_partitioned_lifecycle_with_levers(self, tmp_path: Path):
        engine = _make_engine(
            target_partition_size=8,
            store_vectors=True,
            nprobe=4,
            probe_epsilon=0.2,
        )
        await engine.add(
            {
                key: _normalize(_one_hot(key % NDIM, value=1.0 + key / 100))
                for key in range(64)
            }
        )
        path = str(tmp_path / "part.idx")
        await engine.save(path)
        assert engine._index.nlist > 1

        # Rescoring is on (store_vectors default): self-match is exact.
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=5)
        assert result.matches[0].score == pytest.approx(1.0, abs=1e-5)

        got = await engine.get_vectors([1, 999])
        assert set(got) == {1}
        assert got[1] == pytest.approx(_normalize(_one_hot(1, value=1.01)))

        engine2 = _make_engine()
        await engine2.load(path)
        assert engine2._index.store_vectors
        assert engine2._index.nlist == engine._index.nlist
        loaded = await _search_one(engine2, _normalize(_one_hot(0)), limit=5)
        assert loaded.matches[0].score == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_incremental_growth_keeps_serving(self, tmp_path: Path):
        # Cross the clustering threshold over several engine saves.
        engine = _make_engine(target_partition_size=8)
        path = str(tmp_path / "grow.idx")
        for generation in range(4):
            await engine.add(
                {
                    100 * generation + key: _normalize(
                        _one_hot(key % NDIM, value=1.0 + key / 50)
                    )
                    for key in range(16)
                }
            )
            await engine.save(path)
        result = await _search_one(engine, _normalize(_one_hot(2)), limit=10)
        assert result.matches
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)
