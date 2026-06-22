"""Tests for TurboVecDiskVectorSearchEngine.

The disk engine shares the quantized-search contract with the in-RAM
turbovec engine (whose test suite covers it in depth); these tests focus on
what differs: the delta/tombstone lifecycle around ``save``/``load``,
compaction, and that a freshly-loaded mmap-backed index serves the same
results as the live one that wrote it.
"""

import math
from pathlib import Path

import pytest

pytest.importorskip("turbovec")

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.vector_store.vector_search_engine.turbovec_disk_engine import (
    TurboVecDiskVectorSearchEngine,
)

NDIM = 8

# Tolerance for quantized scores (a self-match lands ~1.0, e.g. 1.0003).
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
    return TurboVecDiskVectorSearchEngine(
        num_dimensions=NDIM, similarity_metric=metric, **kwargs
    )


class TestConstruction:
    def test_supported_metrics(self):
        for metric in (SimilarityMetric.COSINE, SimilarityMetric.DOT):
            _make_engine(metric)

    def test_unsupported_metric_raises(self):
        for metric in (SimilarityMetric.EUCLIDEAN, SimilarityMetric.MANHATTAN):
            with pytest.raises(NotImplementedError, match="does not support"):
                _make_engine(metric)

    def test_invalid_bit_width_raises(self):
        with pytest.raises(ValueError, match="bit_width"):
            _make_engine(bit_width=5)


class TestInRamLifecycle:
    """Before any save, the engine behaves like the in-RAM engine."""

    @pytest.mark.asyncio
    async def test_add_search_remove(self):
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
        assert result.matches[1].key == 3

        await engine.remove([1])
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=3)
        assert 1 not in {m.key for m in result.matches}

    @pytest.mark.asyncio
    async def test_remove_then_add_replaces_key(self):
        # The store's upsert path is remove-then-re-add of the same key.
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        await engine.remove([1])
        await engine.add({1: _normalize(_one_hot(1))})
        result = await _search_one(engine, _normalize(_one_hot(1)), limit=1)
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)

    @pytest.mark.asyncio
    async def test_search_empty_index(self):
        engine = _make_engine()
        result = await _search_one(engine, _normalize(_one_hot(0)), limit=5)
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_get_vectors_raises_without_store_vectors(self):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        with pytest.raises(NotImplementedError, match="store_vectors=True"):
            await engine.get_vectors([1])

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


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_then_load_serves_same_results(self, tmp_path: Path):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0)), 2: _normalize(_one_hot(1))})

        path = str(tmp_path / "test.tvdm")
        await engine.save(path)

        live = await _search_one(engine, _normalize(_one_hot(0)), limit=2)

        engine2 = _make_engine()
        await engine2.load(path)
        loaded = await _search_one(engine2, _normalize(_one_hot(0)), limit=2)

        assert [m.key for m in loaded.matches] == [m.key for m in live.matches]
        assert [m.score for m in loaded.matches] == [m.score for m in live.matches]

    @pytest.mark.asyncio
    async def test_load_replaces_existing_index(self, tmp_path: Path):
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0))})
        path = str(tmp_path / "test.tvdm")
        await engine.save(path)

        engine2 = _make_engine()
        await engine2.add({2: _normalize(_one_hot(1))})
        await engine2.load(path)

        result = await _search_one(engine2, _normalize(_one_hot(0)), limit=5)
        assert {m.key for m in result.matches} == {1}

    @pytest.mark.asyncio
    async def test_mutations_after_save_compact_on_next_save(self, tmp_path: Path):
        # Adds land in the delta and removals tombstone the base; both must
        # be visible immediately and survive the next save/load cycle.
        engine = _make_engine()
        await engine.add({1: _normalize(_one_hot(0)), 2: _normalize(_one_hot(1))})
        path = str(tmp_path / "test.tvdm")
        await engine.save(path)

        await engine.add({3: _normalize(_one_hot(2))})
        await engine.remove([1])

        result = await _search_one(engine, _normalize(_one_hot(0)), limit=5)
        assert {m.key for m in result.matches} == {2, 3}

        await engine.save(path)
        engine2 = _make_engine()
        await engine2.load(path)
        result = await _search_one(engine2, _normalize(_one_hot(0)), limit=5)
        assert {m.key for m in result.matches} == {2, 3}

    @pytest.mark.asyncio
    async def test_repeated_save_load_upsert_cycles(self, tmp_path: Path):
        # Exercise several base/delta/tombstone generations of one key.
        engine = _make_engine()
        path = str(tmp_path / "test.tvdm")
        for index in range(4):
            await engine.remove([7])
            await engine.add({7: _normalize(_one_hot(index))})
            await engine.save(path)
            engine = _make_engine()
            await engine.load(path)
        result = await _search_one(engine, _normalize(_one_hot(3)), limit=1)
        assert result.matches[0].key == 7
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)


class TestPartitioned:
    @pytest.mark.asyncio
    async def test_partitioned_engine_lifecycle(self, tmp_path: Path):
        # Enough vectors to form several partitions at a small target;
        # nearest-neighbor queries must keep working across save/load and
        # upsert cycles.
        engine = _make_engine(target_partition_size=8)
        await engine.add(
            {
                key: _normalize(_one_hot(key % NDIM, value=1.0 + key / 100))
                for key in range(64)
            }
        )
        path = str(tmp_path / "part.tvdm")
        await engine.save(path)
        assert engine._index.nlist > 1

        result = await _search_one(engine, _normalize(_one_hot(0)), limit=5)
        assert result.matches, "partitioned search returned nothing"

        engine2 = _make_engine(target_partition_size=8)
        await engine2.load(path)
        assert engine2._index.nlist == engine._index.nlist
        loaded = await _search_one(engine2, _normalize(_one_hot(0)), limit=5)
        assert [m.key for m in loaded.matches] == [m.key for m in result.matches]

        # Upsert + remove against the partitioned base. Keys 1, 9, 17, ...
        # share the identical direction (value normalizes away), so assert
        # membership among the ties rather than an arbitrary tie-break rank.
        await engine2.remove([1])
        await engine2.add({1: _normalize(_one_hot(1))})
        result = await _search_one(engine2, _normalize(_one_hot(1)), limit=10)
        assert 1 in {m.key for m in result.matches}
        assert result.matches[0].score == pytest.approx(1.0, abs=QUANT_ABS)
        await engine2.save(path)
        result = await _search_one(engine2, _normalize(_one_hot(1)), limit=10)
        assert 1 in {m.key for m in result.matches}


class TestRecallLevers:
    """store_vectors / rescoring, adaptive probing, and replication knobs."""

    @pytest.mark.asyncio
    async def test_get_vectors_roundtrip_and_missing_keys_omitted(
        self, tmp_path: Path
    ):
        engine = _make_engine(store_vectors=True)
        vector_one = _normalize([1, 2, 3, 4, 5, 6, 7, 8])
        vector_two = _normalize(_one_hot(1))
        await engine.add({1: vector_one, 2: vector_two})

        got = await engine.get_vectors([1, 2, 99])
        assert set(got) == {1, 2}
        assert got[1] == pytest.approx(vector_one)

        # From the mmap-backed base after a save, and through a fresh load
        # (store_vectors follows the file).
        path = str(tmp_path / "sv.tvdm")
        await engine.save(path)
        assert (await engine.get_vectors([2]))[2] == pytest.approx(vector_two)

        engine2 = _make_engine()
        await engine2.load(path)
        assert (await engine2.get_vectors([1]))[1] == pytest.approx(vector_one)
        assert await engine2.get_vectors([]) == {}

    @pytest.mark.asyncio
    async def test_rescored_scores_are_exact_not_quantized(self, tmp_path: Path):
        engine = _make_engine(store_vectors=True)
        vector = _normalize([1, 2, 3, 4, 5, 6, 7, 8])
        await engine.add({1: vector, 2: _normalize(_one_hot(1))})
        path = str(tmp_path / "exact.tvdm")
        await engine.save(path)

        # With rescoring on (the store_vectors default), a self-match scores
        # the exact inner product 1.0 — not the quantized estimate, which is
        # only within QUANT_ABS.
        result = await _search_one(engine, vector, limit=1)
        assert result.matches[0].key == 1
        assert result.matches[0].score == pytest.approx(1.0, abs=1e-5)

        # rescore_k=0 turns rescoring off and quantized scores return.
        engine_off = _make_engine(store_vectors=True, rescore_k=0)
        await engine_off.load(path)
        result = await _search_one(engine_off, vector, limit=1)
        assert result.matches[0].key == 1
        assert result.matches[0].score != pytest.approx(1.0, abs=1e-7)

    @pytest.mark.asyncio
    async def test_routing_and_replication_knobs_persist(self, tmp_path: Path):
        engine = _make_engine(
            target_partition_size=8,
            store_vectors=True,
            replica_epsilon=0.5,
            nprobe=4,
            probe_epsilon=0.2,
        )
        await engine.add(
            {
                key: _normalize(_one_hot(key % NDIM, value=1.0 + key / 100))
                for key in range(64)
            }
        )
        path = str(tmp_path / "knobs.tvdm")
        await engine.save(path)
        assert engine._index.nlist > 1

        result = await _search_one(engine, _normalize(_one_hot(0)), limit=5)
        assert result.matches
        assert result.matches[0].score == pytest.approx(1.0, abs=1e-5)

        # A default-configured engine follows the file's settings on load.
        engine2 = _make_engine()
        await engine2.load(path)
        assert engine2._index.store_vectors
        assert engine2._index.replica_epsilon == pytest.approx(0.5)
        assert engine2._index.target_partition_size == 8
        loaded = await _search_one(engine2, _normalize(_one_hot(0)), limit=5)
        assert loaded.matches[0].score == pytest.approx(1.0, abs=1e-5)
