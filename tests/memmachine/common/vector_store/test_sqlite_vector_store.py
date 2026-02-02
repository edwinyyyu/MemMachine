"""Tests for SQLiteVectorStore."""

import math
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.errors import ResourceNotFoundError
from memmachine.common.filter.filter_parser import And, Comparison, Or
from memmachine.common.vector_store.data_types import Record
from memmachine.common.vector_store.sqlite_vector_store import (
    SQLiteVectorStore,
    SQLiteVectorStoreParams,
)


def _norm(vec: list[float]) -> list[float]:
    """L2-normalise a vector (so cosine distance is meaningful)."""
    mag = math.sqrt(sum(x * x for x in vec))
    return [x / mag for x in vec]


@pytest.fixture
async def store():
    s = SQLiteVectorStore(SQLiteVectorStoreParams())
    await s.startup()
    yield s
    await s.shutdown()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_shutdown():
    s = SQLiteVectorStore(SQLiteVectorStoreParams())
    await s.startup()
    await s.shutdown()


@pytest.mark.asyncio
async def test_ensure_db_before_startup():
    s = SQLiteVectorStore(SQLiteVectorStoreParams())
    with pytest.raises(RuntimeError, match="not started"):
        s.ensure_db()


# ---------------------------------------------------------------------------
# Collection CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_get_collection(store):
    await store.create_collection("things", vector_dimensions=4)
    coll = await store.get_collection("things")
    assert coll is not None


@pytest.mark.asyncio
async def test_get_missing_collection_raises(store):
    with pytest.raises(ResourceNotFoundError):
        await store.get_collection("nonexistent")


@pytest.mark.asyncio
async def test_delete_collection(store):
    await store.create_collection("temp", vector_dimensions=4)
    await store.delete_collection("temp")
    with pytest.raises(ResourceNotFoundError):
        await store.get_collection("temp")


@pytest.mark.asyncio
async def test_delete_missing_collection_raises(store):
    with pytest.raises(ResourceNotFoundError):
        await store.delete_collection("nope")


@pytest.mark.asyncio
async def test_unsupported_metric_raises(store):
    with pytest.raises(ValueError, match="Unsupported"):
        await store.create_collection(
            "bad",
            vector_dimensions=4,
            similarity_metric=SimilarityMetric.DOT,
        )


@pytest.mark.asyncio
async def test_invalid_collection_name_raises(store):
    with pytest.raises(ValueError, match="Invalid collection name"):
        await store.create_collection("bad-name!", vector_dimensions=4)


# ---------------------------------------------------------------------------
# Upsert & Get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_and_get(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        properties_schema={"title": str},
    )
    coll = await store.get_collection("docs")

    uid = uuid4()
    vec = _norm([1.0, 0.0, 0.0])
    await coll.upsert(
        records=[Record(uuid=uid, vector=vec, properties={"title": "hello"})]
    )

    results = list(await coll.get(record_uuids=[uid]))
    assert len(results) == 1
    assert results[0].uuid == uid
    assert results[0].properties == {"title": "hello"}
    for a, b in zip(results[0].vector, vec, strict=True):
        assert abs(a - b) < 1e-5


@pytest.mark.asyncio
async def test_upsert_replaces_existing(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        properties_schema={"title": str},
    )
    coll = await store.get_collection("docs")

    uid = uuid4()
    vec1 = _norm([1.0, 0.0, 0.0])
    vec2 = _norm([0.0, 1.0, 0.0])

    await coll.upsert(
        records=[Record(uuid=uid, vector=vec1, properties={"title": "v1"})]
    )
    await coll.upsert(
        records=[Record(uuid=uid, vector=vec2, properties={"title": "v2"})]
    )

    results = list(await coll.get(record_uuids=[uid]))
    assert len(results) == 1
    assert results[0].properties == {"title": "v2"}
    for a, b in zip(results[0].vector, vec2, strict=True):
        assert abs(a - b) < 1e-5


@pytest.mark.asyncio
async def test_get_preserves_input_order(store):
    await store.create_collection("docs", vector_dimensions=3)
    coll = await store.get_collection("docs")

    ids = [uuid4() for _ in range(3)]
    for i, uid in enumerate(ids):
        vec = [0.0] * 3
        vec[i] = 1.0
        await coll.upsert(records=[Record(uuid=uid, vector=_norm(vec))])

    results = list(await coll.get(record_uuids=list(reversed(ids))))
    assert [r.uuid for r in results] == list(reversed(ids))


@pytest.mark.asyncio
async def test_get_missing_uuid_skipped(store):
    await store.create_collection("docs", vector_dimensions=3)
    coll = await store.get_collection("docs")

    uid = uuid4()
    await coll.upsert(records=[Record(uuid=uid, vector=_norm([1.0, 0.0, 0.0]))])
    results = list(await coll.get(record_uuids=[uid, uuid4()]))
    assert len(results) == 1


@pytest.mark.asyncio
async def test_get_without_vector(store):
    await store.create_collection(
        "docs", vector_dimensions=3, properties_schema={"title": str}
    )
    coll = await store.get_collection("docs")

    uid = uuid4()
    await coll.upsert(
        records=[
            Record(uuid=uid, vector=_norm([1.0, 0.0, 0.0]), properties={"title": "hi"})
        ]
    )
    results = list(await coll.get(record_uuids=[uid], return_vector=False))
    assert results[0].vector is None
    assert results[0].properties == {"title": "hi"}


@pytest.mark.asyncio
async def test_get_without_properties(store):
    await store.create_collection(
        "docs", vector_dimensions=3, properties_schema={"title": str}
    )
    coll = await store.get_collection("docs")

    uid = uuid4()
    await coll.upsert(
        records=[
            Record(uuid=uid, vector=_norm([1.0, 0.0, 0.0]), properties={"title": "hi"})
        ]
    )
    results = list(await coll.get(record_uuids=[uid], return_properties=False))
    assert results[0].vector is not None
    assert results[0].properties is None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_records(store):
    await store.create_collection("docs", vector_dimensions=3)
    coll = await store.get_collection("docs")

    uid = uuid4()
    await coll.upsert(records=[Record(uuid=uid, vector=_norm([1.0, 0.0, 0.0]))])
    await coll.delete(record_uuids=[uid])

    results = list(await coll.get(record_uuids=[uid]))
    assert len(results) == 0


@pytest.mark.asyncio
async def test_delete_nonexistent_uuid_no_error(store):
    await store.create_collection("docs", vector_dimensions=3)
    coll = await store.get_collection("docs")
    await coll.delete(record_uuids=[uuid4()])


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_basic(store):
    await store.create_collection("docs", vector_dimensions=3)
    coll = await store.get_collection("docs")

    vecs = [
        _norm([1.0, 0.0, 0.0]),
        _norm([0.9, 0.1, 0.0]),
        _norm([0.0, 0.0, 1.0]),
    ]
    for i, v in enumerate(vecs):
        await coll.upsert(records=[Record(uuid=UUID(int=i + 1), vector=v)])

    results = list(
        await coll.query(query_vector=_norm([1.0, 0.0, 0.0]), limit=2)
    )
    assert len(results) == 2
    # Most similar first
    assert results[0].record.uuid == UUID(int=1)
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_query_with_similarity_threshold(store):
    await store.create_collection("docs", vector_dimensions=3)
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(uuid=UUID(int=1), vector=_norm([1.0, 0.0, 0.0])),
            Record(uuid=UUID(int=2), vector=_norm([0.0, 0.0, 1.0])),
        ]
    )

    results = list(
        await coll.query(
            query_vector=_norm([1.0, 0.0, 0.0]),
            similarity_threshold=0.9,
        )
    )
    assert all(r.score >= 0.9 for r in results)
    assert any(r.record.uuid == UUID(int=1) for r in results)


@pytest.mark.asyncio
async def test_query_with_property_filter(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        properties_schema={"category": str},
    )
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(
                uuid=UUID(int=1),
                vector=_norm([1.0, 0.0, 0.0]),
                properties={"category": "A"},
            ),
            Record(
                uuid=UUID(int=2),
                vector=_norm([0.9, 0.1, 0.0]),
                properties={"category": "B"},
            ),
        ]
    )

    filt = Comparison(field="category", op="=", value="A")
    results = list(
        await coll.query(
            query_vector=_norm([1.0, 0.0, 0.0]),
            property_filter=filt,
        )
    )
    assert len(results) == 1
    assert results[0].record.properties["category"] == "A"


@pytest.mark.asyncio
async def test_query_with_and_filter(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        properties_schema={"category": str, "priority": int},
    )
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(
                uuid=UUID(int=1),
                vector=_norm([1.0, 0.0, 0.0]),
                properties={"category": "A", "priority": 1},
            ),
            Record(
                uuid=UUID(int=2),
                vector=_norm([0.9, 0.1, 0.0]),
                properties={"category": "A", "priority": 2},
            ),
            Record(
                uuid=UUID(int=3),
                vector=_norm([0.8, 0.2, 0.0]),
                properties={"category": "B", "priority": 1},
            ),
        ]
    )

    filt = And(
        left=Comparison(field="category", op="=", value="A"),
        right=Comparison(field="priority", op="=", value=1),
    )
    results = list(
        await coll.query(query_vector=_norm([1.0, 0.0, 0.0]), property_filter=filt)
    )
    assert len(results) == 1
    assert results[0].record.uuid == UUID(int=1)


@pytest.mark.asyncio
async def test_query_with_or_filter(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        properties_schema={"category": str},
    )
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(
                uuid=UUID(int=1),
                vector=_norm([1.0, 0.0, 0.0]),
                properties={"category": "A"},
            ),
            Record(
                uuid=UUID(int=2),
                vector=_norm([0.9, 0.1, 0.0]),
                properties={"category": "B"},
            ),
            Record(
                uuid=UUID(int=3),
                vector=_norm([0.0, 0.0, 1.0]),
                properties={"category": "C"},
            ),
        ]
    )

    filt = Or(
        left=Comparison(field="category", op="=", value="A"),
        right=Comparison(field="category", op="=", value="B"),
    )
    results = list(
        await coll.query(query_vector=_norm([1.0, 0.0, 0.0]), property_filter=filt)
    )
    uuids = {r.record.uuid for r in results}
    assert UUID(int=1) in uuids
    assert UUID(int=2) in uuids
    assert UUID(int=3) not in uuids


@pytest.mark.asyncio
async def test_query_with_in_filter(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        properties_schema={"category": str},
    )
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(
                uuid=UUID(int=1),
                vector=_norm([1.0, 0.0, 0.0]),
                properties={"category": "A"},
            ),
            Record(
                uuid=UUID(int=2),
                vector=_norm([0.9, 0.1, 0.0]),
                properties={"category": "B"},
            ),
        ]
    )

    filt = Comparison(field="category", op="in", value=["A", "B"])
    results = list(
        await coll.query(query_vector=_norm([1.0, 0.0, 0.0]), property_filter=filt)
    )
    assert len(results) == 2


@pytest.mark.asyncio
async def test_query_without_vector(store):
    await store.create_collection("docs", vector_dimensions=3)
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[Record(uuid=UUID(int=1), vector=_norm([1.0, 0.0, 0.0]))]
    )
    results = list(
        await coll.query(
            query_vector=_norm([1.0, 0.0, 0.0]),
            return_vector=False,
        )
    )
    assert results[0].record.vector is None


@pytest.mark.asyncio
async def test_query_without_properties(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        properties_schema={"title": str},
    )
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(
                uuid=UUID(int=1),
                vector=_norm([1.0, 0.0, 0.0]),
                properties={"title": "hi"},
            )
        ]
    )
    results = list(
        await coll.query(
            query_vector=_norm([1.0, 0.0, 0.0]),
            return_properties=False,
        )
    )
    assert results[0].record.properties is None


# ---------------------------------------------------------------------------
# Property types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bool_property(store):
    await store.create_collection(
        "docs", vector_dimensions=3, properties_schema={"active": bool}
    )
    coll = await store.get_collection("docs")

    uid = uuid4()
    await coll.upsert(
        records=[
            Record(
                uuid=uid, vector=_norm([1.0, 0.0, 0.0]), properties={"active": True}
            )
        ]
    )
    results = list(await coll.get(record_uuids=[uid]))
    assert results[0].properties["active"] is True


@pytest.mark.asyncio
async def test_datetime_property(store):
    await store.create_collection(
        "docs", vector_dimensions=3, properties_schema={"created": datetime}
    )
    coll = await store.get_collection("docs")

    uid = uuid4()
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    await coll.upsert(
        records=[
            Record(
                uuid=uid, vector=_norm([1.0, 0.0, 0.0]), properties={"created": now}
            )
        ]
    )
    results = list(await coll.get(record_uuids=[uid]))
    assert results[0].properties["created"] == now


@pytest.mark.asyncio
async def test_float_property(store):
    await store.create_collection(
        "docs", vector_dimensions=3, properties_schema={"score": float}
    )
    coll = await store.get_collection("docs")

    uid = uuid4()
    await coll.upsert(
        records=[
            Record(
                uuid=uid, vector=_norm([1.0, 0.0, 0.0]), properties={"score": 0.95}
            )
        ]
    )
    results = list(await coll.get(record_uuids=[uid]))
    assert abs(results[0].properties["score"] - 0.95) < 1e-5


# ---------------------------------------------------------------------------
# Handle persists across delete/recreate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_persists_across_delete_recreate(store):
    await store.create_collection("docs", vector_dimensions=3)
    coll = await store.get_collection("docs")

    uid = uuid4()
    await coll.upsert(records=[Record(uuid=uid, vector=_norm([1.0, 0.0, 0.0]))])

    await store.delete_collection("docs")
    await store.create_collection("docs", vector_dimensions=3)

    # Old handle should work on the recreated collection
    uid2 = uuid4()
    await coll.upsert(records=[Record(uuid=uid2, vector=_norm([0.0, 1.0, 0.0]))])
    results = list(await coll.get(record_uuids=[uid2]))
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Filter edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_null_filter(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        properties_schema={"note": str},
    )
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(
                uuid=UUID(int=1),
                vector=_norm([1.0, 0.0, 0.0]),
                properties={"note": "hello"},
            ),
            Record(
                uuid=UUID(int=2),
                vector=_norm([0.9, 0.1, 0.0]),
                properties={},
            ),
        ]
    )

    filt = Comparison(field="note", op="is_null", value=None)
    results = list(
        await coll.query(query_vector=_norm([1.0, 0.0, 0.0]), property_filter=filt)
    )
    assert len(results) == 1
    assert results[0].record.uuid == UUID(int=2)


@pytest.mark.asyncio
async def test_comparison_gt_filter(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        properties_schema={"priority": int},
    )
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(
                uuid=UUID(int=1),
                vector=_norm([1.0, 0.0, 0.0]),
                properties={"priority": 5},
            ),
            Record(
                uuid=UUID(int=2),
                vector=_norm([0.9, 0.1, 0.0]),
                properties={"priority": 1},
            ),
        ]
    )

    filt = Comparison(field="priority", op=">", value=3)
    results = list(
        await coll.query(query_vector=_norm([1.0, 0.0, 0.0]), property_filter=filt)
    )
    assert len(results) == 1
    assert results[0].record.uuid == UUID(int=1)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_euclidean_metric(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        similarity_metric=SimilarityMetric.EUCLIDEAN,
    )
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(uuid=UUID(int=1), vector=[1.0, 0.0, 0.0]),
            Record(uuid=UUID(int=2), vector=[0.0, 1.0, 0.0]),
            Record(uuid=UUID(int=3), vector=[0.0, 0.0, 1.0]),
        ]
    )

    results = list(await coll.query(query_vector=[1.0, 0.0, 0.0], limit=3))
    assert len(results) == 3
    # Exact match has highest similarity
    assert results[0].record.uuid == UUID(int=1)
    assert results[0].score > results[1].score
    # Similarity is 1/(1+distance), exact match → 1/(1+0) = 1.0
    assert abs(results[0].score - 1.0) < 1e-5


@pytest.mark.asyncio
async def test_manhattan_metric(store):
    await store.create_collection(
        "docs",
        vector_dimensions=3,
        similarity_metric=SimilarityMetric.MANHATTAN,
    )
    coll = await store.get_collection("docs")

    await coll.upsert(
        records=[
            Record(uuid=UUID(int=1), vector=[1.0, 0.0, 0.0]),
            Record(uuid=UUID(int=2), vector=[0.0, 1.0, 0.0]),
        ]
    )

    results = list(await coll.query(query_vector=[1.0, 0.0, 0.0], limit=2))
    assert len(results) == 2
    assert results[0].record.uuid == UUID(int=1)
    assert abs(results[0].score - 1.0) < 1e-5
    # L1 distance between [1,0,0] and [0,1,0] is 2.0 → similarity = 1/3
    assert abs(results[1].score - 1.0 / 3.0) < 1e-5
