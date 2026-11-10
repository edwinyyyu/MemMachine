"""The benchmark's own vector store (rbench/vectorstore.py): it ranks as the legacy store
(MemMachine's SQLiteVectorStore with ExactSearchEngine) up to float16 near-tie swaps, searching
at a width equals a store built from vectors of that width, and vectors are stored once."""

import asyncio
import hashlib
import sqlite3
import uuid

import numpy as np
import pytest

from rbench.memory import Memory
from rbench.paths import PREPARED
from rbench.vectorstore import ContentVectorStore, ensure_content, vector_hash
from tests.test_memory import FakeEmbedder, _events

from memmachine_server.common.data_types import SimilarityMetric  # noqa: I001
from memmachine_server.common.vector_store.data_types import Record, VectorStoreSchema

QUERIES = ("bought apples", "price was too high", "my sister said", "ok? Then we left.", "the store yesterday")
FLOAT16_TIE = 2e-3  # the legacy index rounds unit vectors to float16 (about 3 decimal digits)


def _schema(dims: int) -> VectorStoreSchema:
    return VectorStoreSchema(vector_dimensions=dims, similarity_metric=SimilarityMetric.COSINE, indexed_properties={})


def _ranked(root, backend: str, events) -> list[list[tuple[tuple, float]]]:
    """Every query's whole ranking: (event, segment index, offset) of each seed, and its score."""

    async def go():
        async with Memory("fake", "synthetic", "seg80", root=root, backend=FakeEmbedder(), vector_store=backend) as m:
            await m.ingest("h", events)
            partition = await (await m.event_memory()).open_or_create_partition("h")
            out = []
            for q in QUERIES:
                result = await partition.query(q, vector_search_limit=10_000, expand_context=0)
                out.append([((s.event_uuid, s.index, s.offset), c.score)
                            for c in result.scored_segment_contexts for s in c.segments if s.uuid == c.seed_segment_uuid])
            return out

    return asyncio.run(go())


def test_ranks_as_the_legacy_store_up_to_float16_near_ties(tmp_path):
    events = _events(80, seed=3)
    legacy, ours = _ranked(tmp_path / "sqlite", "sqlite", events), _ranked(tmp_path / "content", "content", events)
    swaps = []
    for old, new in zip(legacy, ours, strict=True):
        assert {k for k, _ in old} == {k for k, _ in new}  # every segment, both ways
        score = dict(new)
        for (k, s) in old:
            assert abs(s - score[k]) < FLOAT16_TIE
        order = [k for k, _ in old]
        for i in range(len(order)):  # pairs the legacy order puts the other way round
            for j in range(i + 1, len(order)):
                if score[order[i]] < score[order[j]]:
                    swaps.append(score[order[j]] - score[order[i]])
    assert all(gap < FLOAT16_TIE for gap in swaps), swaps  # only near ties swap
    print(f"{len(swaps)} near-tie swaps over {sum(len(r) for r in ours)} ranked seeds; largest gap "
          f"{max(swaps, default=0):.2e}")


def test_searching_at_a_width_equals_a_store_of_that_width(tmp_path):
    """The user's rule (2026-11-10): keep 4,096 and truncate at search if it works. A store
    searched at w ranks and scores as one built from the vectors truncated to w and
    re-normalized, queried with the truncated query."""
    rng = np.random.default_rng(0)
    full, queries, w = rng.normal(size=(400, 64)).astype(np.float32), rng.normal(size=(6, 64)), 16
    cut = full[:, :w] / np.linalg.norm(full[:, :w], axis=1, keepdims=True)
    ids = [uuid.UUID(int=i + 1) for i in range(len(full))]

    async def search(name, dims, vectors, width, qs):
        store = ContentVectorStore(tmp_path / name, _schema(dims), tmp_path / f"{name}.sqlite", width=width)
        await store.startup()
        partition = await store.open_or_create_partition("p")
        await partition.upsert(records=[Record(uuid=u, vector=v.tolist()) for u, v in zip(ids, vectors, strict=True)])
        results = await partition.query(query_vectors=[q.tolist() for q in qs], limit=60)
        await store.shutdown()
        return [[(m.record_uuid, m.score) for m in r.matches] for r in results]

    truncated = asyncio.run(search("a", 64, full, w, queries))
    built = asyncio.run(search("b", w, cut.astype(np.float32), None, queries[:, :w]))
    for a, b in zip(truncated, built, strict=True):
        assert [u for u, _ in a] == [u for u, _ in b]
        assert np.allclose([s for _, s in a], [s for _, s in b], atol=1e-6)
    whole = asyncio.run(search("c", 64, full, None, queries))
    assert [u for u, _ in whole[0]] != [u for u, _ in truncated[0]]  # the width does change the search
    with pytest.raises(ValueError):
        ContentVectorStore(tmp_path / "d", _schema(64), tmp_path / "d.sqlite", width=65)


def test_vectors_are_stored_once(tmp_path):
    """A vector already in the permanent copy (cached under its text) adds no row; a new one is
    added once however many records share it; deleting a partition keeps the vectors."""
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(5, 8)).astype(np.float32)
    content = tmp_path / "copy.sqlite"
    db = sqlite3.connect(content)
    ensure_content(db)
    with db:  # three of them cached under their texts, as EmbeddingCache.put writes them
        db.executemany("INSERT INTO vectors (key, v, vhash) VALUES (?, ?, ?)",
                       [(hashlib.sha1(f"text {i}".encode()).hexdigest(), v.tobytes(), vector_hash(v.tobytes()))
                        for i, v in enumerate(vectors[:3])])
    db.close()

    def rows() -> list[str]:
        db = sqlite3.connect(content)
        try:
            return [k for (k,) in db.execute("SELECT key FROM vectors")]
        finally:
            db.close()

    async def go():
        store = ContentVectorStore(tmp_path / "s", _schema(8), content)
        await store.startup()
        p = await store.open_or_create_partition("p")
        await p.upsert(records=[Record(uuid=uuid.UUID(int=i + 1), vector=v.tolist()) for i, v in enumerate(vectors[:3])])
        assert len(rows()) == 3  # all three were in the copy
        shared = [Record(uuid=uuid.UUID(int=10 + i), vector=vectors[3 + i % 2].tolist()) for i in range(6)]
        await p.upsert(records=shared)
        assert len(rows()) == 5 and sum(k.startswith("v:") for k in rows()) == 2  # two new vectors, once each
        assert len((await p.query(query_vectors=[vectors[3].tolist()], limit=100))[0].matches) == 9
        await store.delete_partition("p")
        assert len(rows()) == 5  # paid for: kept
        q = await store.open_or_create_partition("p")
        assert (await q.query(query_vectors=[vectors[0].tolist()], limit=10))[0].matches == []
        await store.shutdown()

    asyncio.run(go())


def test_a_missing_vector_fails_loudly_and_unused_methods_raise(tmp_path):
    async def go():
        store = ContentVectorStore(tmp_path / "s", _schema(4), tmp_path / "c.sqlite")
        await store.startup()
        p = await store.open_or_create_partition("p")
        await p.upsert(records=[Record(uuid=uuid.UUID(int=1), vector=[1.0, 0, 0, 0])])
        with pytest.raises(NotImplementedError):
            await p.query(query_vectors=[[1.0, 0, 0, 0]], limit=1, score_threshold=0.5)
        with pytest.raises(NotImplementedError):
            await p.delete(record_uuids=[uuid.UUID(int=1)])
        for call in (store.create_partition("x"), store.open_partition("x"), store.purge_deleted_partitions()):
            with pytest.raises(NotImplementedError):
                await call
        with pytest.raises(ValueError):
            await p.upsert(records=[Record(uuid=uuid.UUID(int=2), vector=[1.0, 0, 0])])  # wrong dimensions
        with store.db:
            store.db.execute("DELETE FROM content.vectors")
        with pytest.raises(RuntimeError, match="missing"):
            await (await store.open_or_create_partition("p")).query(query_vectors=[[1.0, 0, 0, 0]], limit=1)
        await store.shutdown()

    asyncio.run(go())


def test_a_legacy_root_keeps_the_legacy_store(tmp_path):
    events = _events(20)
    root = tmp_path / "old"
    first = _ranked(root, "sqlite", events)
    m = Memory("fake", "synthetic", "seg80", root=root, backend=FakeEmbedder())
    assert m.vector_store == "sqlite" and (root / "vectors.sqlite").exists() and not (root / "records.sqlite").exists()
    assert _ranked(root, None, events) == first  # reopened as it was written
    assert Memory("fake", "synthetic", "seg80", root=tmp_path / "new", backend=FakeEmbedder()).vector_store == "content"
    with pytest.raises(ValueError):
        Memory("fake", "synthetic", "seg80", root=root, backend=FakeEmbedder(), width=16)


def _fake_api(server_truncates: bool, noise: float = 0.0):
    """A stand-in for DeepInfra's embeddings: a fixed vector per text; asked for w dimensions,
    the leading w (server_truncates) or another vector's (not a truncation)."""
    calls = {"n": 0}

    def raw(spec, texts, dimensions):
        calls["n"] += 1
        rng = np.random.default_rng(calls["n"])
        out = []
        for t in texts:
            v = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).normal(size=4096)
            if dimensions < 4096 and not server_truncates:
                v = np.roll(v, 1)
            v = v[:dimensions] + noise * rng.normal(size=dimensions)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out), {"prompt_tokens": 10 * len(texts)}

    return raw


@pytest.mark.skipif(not (PREPARED / "longmemeval_m" / "queries.jsonl").exists(), reason="needs LongMemEval-M prepared")
@pytest.mark.parametrize(("server_truncates", "noise", "passes"), [(True, 0.0, True), (True, 1e-6, True),
                                                                   (False, 0.0, False)])
def test_the_truncation_check_tells_truncation_apart(monkeypatch, server_truncates, noise, passes):
    from rbench import preflight

    monkeypatch.setattr(preflight, "_raw", _fake_api(server_truncates, noise))
    out = preflight.truncation_check(probes=4, partitions=1, haystack=200)
    assert out["pass"] is passes, out
    assert out["top 10 at 1024"]["queries"] >= 1
