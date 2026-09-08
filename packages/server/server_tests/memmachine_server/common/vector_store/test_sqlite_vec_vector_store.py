"""Tests for SQLiteVecVectorStore."""

import math
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    Not,
    Or,
)
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
)
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLiteVecVectorStore,
    SQLiteVecVectorStoreCollection,
    SQLiteVecVectorStoreParams,
)

pytestmark = pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason="sqlite3 built without extension loading support",
)

NAMESPACE = "test_namespace"
NAME = "test_name"
VECTOR_DIM = 3


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


_FETCH_LIMIT = 1000


async def _fetch_records(collection, record_uuids) -> list[Record]:
    """Read records back by UUID.

    The collection contract has no read-by-UUID: `query` is the only read, and
    with no score threshold it returns every record it holds, so a probe vector
    in any direction lists the collection for a test to pick out of.
    """
    record_uuids = list(record_uuids)
    if not record_uuids:
        return []
    dimensions = collection.config.vector_dimensions
    probe = [1.0] + [0.0] * (dimensions - 1)
    [result] = await collection.query(query_vectors=[probe], limit=_FETCH_LIMIT)
    by_uuid = {match.record.uuid: match.record for match in result.matches}
    return [by_uuid[uuid] for uuid in record_uuids if uuid in by_uuid]


def _make_record(
    *,
    uuid=None,
    vector: list[float] | None = None,
    properties: dict | None = None,
) -> Record:
    return Record(
        uuid=uuid or uuid4(),
        vector=vector,
        properties=properties,
    )


@pytest_asyncio.fixture
async def store(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    params = SQLiteVecVectorStoreParams(engine=engine)
    vector_store = SQLiteVecVectorStore(params)
    await vector_store.startup()
    yield vector_store
    await vector_store.shutdown()
    await engine.dispose()


@pytest_asyncio.fixture
async def collection(store):
    await store.create_collection(
        namespace=NAMESPACE,
        name=NAME,
        config=VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM,
            indexed_properties_schema={
                "name": str,
                "age": int,
                "score": float,
                "active": bool,
                "created_at": datetime,
            },
        ),
    )
    coll = await store.open_collection(namespace=NAMESPACE, name=NAME)
    assert coll is not None
    yield coll
    await store.delete_collection(namespace=NAMESPACE, name=NAME)


# ── Collection lifecycle ──


class TestCollectionLifecycle:
    @pytest.mark.asyncio
    async def test_create_open_delete(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="lifecycle",
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        coll = await store.open_collection(namespace=NAMESPACE, name="lifecycle")
        assert isinstance(coll, SQLiteVecVectorStoreCollection)
        await store.delete_collection(namespace=NAMESPACE, name="lifecycle")

    @pytest.mark.asyncio
    async def test_open_returns_correct_type(self, store, collection):
        coll = await store.open_collection(namespace=NAMESPACE, name=NAME)
        assert isinstance(coll, SQLiteVecVectorStoreCollection)

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, collection):
        with pytest.raises(VectorStoreCollectionAlreadyExistsError):
            await store.create_collection(
                namespace=NAMESPACE,
                name=NAME,
                config=VectorStoreCollectionConfig(
                    vector_dimensions=VECTOR_DIM,
                    indexed_properties_schema={
                        "name": str,
                        "age": int,
                        "score": float,
                        "active": bool,
                        "created_at": datetime,
                    },
                ),
            )

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, store):
        await store.delete_collection(namespace=NAMESPACE, name="nonexistent")

    @pytest.mark.asyncio
    async def test_open_or_create_creates_when_missing(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="new", config=config
        )
        assert isinstance(coll, SQLiteVecVectorStoreCollection)
        await store.delete_collection(namespace=NAMESPACE, name="new")

    @pytest.mark.asyncio
    async def test_open_or_create_opens_when_exists(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="existing", config=config
        )
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="existing", config=config
        )
        assert isinstance(coll, SQLiteVecVectorStoreCollection)
        await store.delete_collection(namespace=NAMESPACE, name="existing")

    @pytest.mark.asyncio
    async def test_open_or_create_raises_on_config_mismatch(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="mismatch",
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        with pytest.raises(VectorStoreCollectionConfigMismatchError):
            await store.open_or_create_collection(
                namespace=NAMESPACE,
                name="mismatch",
                config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM + 1),
            )
        await store.delete_collection(namespace=NAMESPACE, name="mismatch")

    @pytest.mark.asyncio
    async def test_open_nonexistent_returns_none(self, store):
        assert await store.open_collection(namespace=NAMESPACE, name="nope") is None

    @pytest.mark.asyncio
    async def test_invalid_namespace_raises(self, store):
        with pytest.raises(ValueError, match="Invalid namespace"):
            await store.create_collection(
                namespace="INVALID",
                name="test",
                config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
            )

    @pytest.mark.asyncio
    async def test_invalid_name_raises(self, store):
        with pytest.raises(ValueError, match="Invalid namespace"):
            await store.create_collection(
                namespace="valid",
                name="INVALID",
                config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
            )


# ── Upsert + Query ──


class TestUpsertAndQuery:
    @pytest.mark.asyncio
    async def test_upsert_and_query_basic(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        v3 = _normalize([1.0, 0.1, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        r3 = _make_record(vector=v3, properties={"name": "c"})

        await collection.upsert(records=[r1, r2, r3])

        query_results = await collection.query(query_vectors=[v1], limit=3)
        matches = query_results[0].matches

        assert len(matches) == 3
        assert matches[0].record.uuid == r1.uuid
        assert (
            matches[0].cosine_similarity
            >= matches[1].cosine_similarity
            >= matches[2].cosine_similarity
        )

    @pytest.mark.asyncio
    async def test_upsert_update(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=v1, properties={"name": "original"})
        await collection.upsert(records=[record])

        updated = Record(
            uuid=record.uuid,
            vector=_normalize([0.0, 1.0, 0.0]),
            properties={"name": "updated"},
        )
        await collection.upsert(records=[updated])

        results = await _fetch_records(collection, [record.uuid])
        properties = results[0].properties
        assert properties is not None
        assert properties["name"] == "updated"

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])

        query_results = await collection.query(
            query_vectors=[v1], limit=10, min_cosine_similarity=0.9
        )
        matches = query_results[0].matches

        assert len(matches) == 1
        assert matches[0].record.uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_with_limit(self, collection):
        vectors = [_normalize([1.0, float(index) * 0.01, 0.0]) for index in range(5)]
        records = [_make_record(vector=vector) for vector in vectors]
        await collection.upsert(records=records)

        query_results = await collection.query(query_vectors=[vectors[0]], limit=2)
        assert len(query_results[0].matches) == 2

    @pytest.mark.asyncio
    async def test_query_never_returns_vectors(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        query_results = await collection.query(query_vectors=[v1], limit=10)
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.vector is None
        assert matches[0].record.properties is not None

    @pytest.mark.asyncio
    async def test_query_return_properties_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            return_properties=False,
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.properties is None

    @pytest.mark.asyncio
    async def test_query_batch_multiple_vectors(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await collection.upsert(records=[r1, r2])

        all_results = await collection.query(query_vectors=[v1, v2], limit=1)

        assert len(all_results) == 2
        assert all_results[0].matches[0].record.uuid == r1.uuid
        assert all_results[1].matches[0].record.uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_query_empty_vectors(self, collection):
        all_results = await collection.query(query_vectors=[], limit=10)
        assert len(all_results) == 0

    @pytest.mark.asyncio
    async def test_upsert_empty_records(self, collection):
        await collection.upsert(records=[])


# ── Filters ──


class TestFilters:
    async def _setup(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        r1 = _make_record(
            vector=v1,
            properties={"name": "alice", "age": 30, "score": 9.5, "active": True},
        )
        r2 = _make_record(
            vector=v2,
            properties={"name": "bob", "age": 25, "score": 7.0, "active": False},
        )
        r3 = _make_record(
            vector=v3,
            properties={"name": "carol", "age": 35, "score": 8.0, "active": True},
        )
        await collection.upsert(records=[r1, r2, r3])
        return r1, r2, r3, v1

    async def _setup_floats(self, collection):
        vectors = [_normalize([1.0, float(index) * 0.01, 0.0]) for index in range(5)]
        scores = [-1.5, 0.0, 0.5, 1.5, 2.0]
        records = [
            _make_record(vector=vector, properties={"score": score})
            for vector, score in zip(vectors, scores, strict=True)
        ]
        await collection.upsert(records=records)
        return records, vectors[0]

    async def _setup_datetimes(self, collection):
        vectors = [_normalize([1.0, float(index) * 0.01, 0.0]) for index in range(5)]
        datetimes = [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 3, 15, tzinfo=UTC),
            datetime(2024, 6, 1, tzinfo=UTC),
            datetime(2024, 9, 1, tzinfo=UTC),
            datetime(2024, 12, 31, tzinfo=UTC),
        ]
        records = [
            _make_record(
                vector=vector,
                properties={"name": f"r{index}", "created_at": dt},
            )
            for index, (vector, dt) in enumerate(zip(vectors, datetimes, strict=True))
        ]
        await collection.upsert(records=records)
        return records, vectors[0], datetimes

    async def _query(self, collection, query_vector, field, op, value):
        all_results = await collection.query(
            query_vectors=[query_vector],
            limit=10,
            property_filter=Comparison(field=field, op=op, value=value),
        )
        return {match.record.uuid for match in all_results[0].matches}

    # ── String / int ──

    @pytest.mark.asyncio
    async def test_eq_str(self, collection):
        r1, _r2, _r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "name", "=", "alice")
        assert r1.uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_str(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "name", "!=", "alice")
        assert r1.uuid not in uuids
        assert r2.uuid in uuids
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_gt_int(self, collection):
        _r1, _r2, r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "age", ">", 30)
        assert len(uuids) == 1
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_gte_int(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "age", ">=", 30)
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(uuids) == 2

    @pytest.mark.asyncio
    async def test_lt_int(self, collection):
        _r1, r2, _r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "age", "<", 30)
        assert len(uuids) == 1
        assert r2.uuid in uuids

    @pytest.mark.asyncio
    async def test_lte_int(self, collection):
        r1, r2, _r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "age", "<=", 30)
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(uuids) == 2

    # ── Bool ──

    @pytest.mark.asyncio
    async def test_eq_bool(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "active", "=", True)
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert r2.uuid not in uuids

    @pytest.mark.asyncio
    async def test_ne_bool(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "active", "!=", True)
        assert r2.uuid in uuids
        assert r1.uuid not in uuids
        assert r3.uuid not in uuids

    # ── Float ──

    @pytest.mark.asyncio
    async def test_eq_float(self, collection):
        records, query_vector = await self._setup_floats(collection)
        uuids = await self._query(collection, query_vector, "score", "=", 0.5)
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_gt_float(self, collection):
        records, query_vector = await self._setup_floats(collection)
        uuids = await self._query(collection, query_vector, "score", ">", 0.5)
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids
        assert records[2].uuid not in uuids

    @pytest.mark.asyncio
    async def test_lt_float(self, collection):
        records, query_vector = await self._setup_floats(collection)
        uuids = await self._query(collection, query_vector, "score", "<", 0.5)
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid not in uuids

    # ── Datetime ──

    @pytest.mark.asyncio
    async def test_datetime_roundtrip(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "test", "created_at": dt})
        await collection.upsert(records=[r1])

        results = await _fetch_records(collection, [r1.uuid])
        properties = results[0].properties
        assert properties is not None
        assert properties["created_at"] == dt

    @pytest.mark.asyncio
    async def test_eq_datetime(self, collection):
        records, query_vector, datetimes = await self._setup_datetimes(collection)
        uuids = await self._query(
            collection, query_vector, "created_at", "=", datetimes[2]
        )
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_gt_datetime(self, collection):
        records, query_vector, datetimes = await self._setup_datetimes(collection)
        uuids = await self._query(
            collection, query_vector, "created_at", ">", datetimes[2]
        )
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids
        assert records[2].uuid not in uuids

    @pytest.mark.asyncio
    async def test_lt_datetime(self, collection):
        records, query_vector, datetimes = await self._setup_datetimes(collection)
        uuids = await self._query(
            collection, query_vector, "created_at", "<", datetimes[2]
        )
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid not in uuids

    @pytest.mark.asyncio
    async def test_eq_datetime_cross_timezone(self, collection):
        """Equality matches the same instant expressed in a different timezone."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt_other = datetime(2024, 6, 15, 18, 0, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": dt_utc})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": dt_other})
        await collection.upsert(records=[r1, r2])

        plus5 = timezone(timedelta(hours=5))
        dt_filter = datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)
        uuids = await self._query(collection, v1, "created_at", "=", dt_filter)
        assert r1.uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_datetime_timezone_roundtrip(self, collection):
        """Original timezone is preserved through storage."""
        v1 = _normalize([1.0, 0.0, 0.0])
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 6, 15, 7, 0, 0, tzinfo=est)
        r1 = _make_record(vector=v1, properties={"name": "tz", "created_at": dt})
        await collection.upsert(records=[r1])

        results = await _fetch_records(collection, [r1.uuid])
        properties = results[0].properties
        assert properties is not None
        got = properties["created_at"]
        assert isinstance(got, datetime)
        assert got == dt
        assert got.utcoffset() == timedelta(hours=-5)

    # ── In / And / Or / Not ──

    @pytest.mark.asyncio
    async def test_in(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=In(field="name", values=["alice", "carol"]),
        )
        uuids = {match.record.uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(uuids) == 2

    @pytest.mark.asyncio
    async def test_and(self, collection):
        _r1, _r2, r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=And(
                left=Comparison(field="active", op="=", value=True),
                right=Comparison(field="age", op=">", value=30),
            ),
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.uuid == r3.uuid

    @pytest.mark.asyncio
    async def test_or(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Or(
                left=Comparison(field="name", op="=", value="alice"),
                right=Comparison(field="name", op="=", value="carol"),
            ),
        )
        uuids = {match.record.uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(uuids) == 2

    @pytest.mark.asyncio
    async def test_not(self, collection):
        r1, r2, _r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Not(expr=Comparison(field="age", op=">", value=30)),
        )
        uuids = {match.record.uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(uuids) == 2


# ── Set properties ──


class TestSetProperties:
    @pytest.mark.asyncio
    async def test_replaces_properties_and_keeps_the_vector(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "a", "rank": 1})
        await collection.upsert(records=[r1])

        await collection.set_properties(record_properties={r1.uuid: {"name": "b"}})

        [fetched] = await _fetch_records(collection, [r1.uuid])
        assert fetched.properties == {"name": "b"}

        # The vector is untouched, so the record still answers its own query.
        results = await collection.query(query_vectors=[v1], limit=1)
        assert results[0].matches[0].record.uuid == r1.uuid
        assert results[0].matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_replaced_properties_are_filterable(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "a"})
        await collection.upsert(records=[r1])

        await collection.set_properties(record_properties={r1.uuid: {"name": "b"}})

        stale = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="name", op="=", value="a"),
        )
        assert stale[0].matches == []

        fresh = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="name", op="=", value="b"),
        )
        assert [match.record.uuid for match in fresh[0].matches] == [r1.uuid]

    @pytest.mark.asyncio
    async def test_ignores_uuids_the_collection_does_not_hold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "a"})
        await collection.upsert(records=[r1])

        await collection.set_properties(record_properties={uuid4(): {"name": "ghost"}})

        [fetched] = await _fetch_records(collection, [r1.uuid])
        assert fetched.properties == {"name": "a"}

    @pytest.mark.asyncio
    async def test_empty_mapping(self, collection):
        await collection.set_properties(record_properties={})


# ── Delete ──


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_records(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])
        await collection.delete(record_uuids=[r1.uuid])

        results = await _fetch_records(collection, [r1.uuid, r2.uuid])
        assert len(results) == 1
        assert results[0].uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, collection):
        await collection.delete(record_uuids=[])

    @pytest.mark.asyncio
    async def test_delete_nonexistent_uuid(self, collection):
        await collection.delete(record_uuids=[uuid4()])

    @pytest.mark.asyncio
    async def test_delete_removes_from_query(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await collection.upsert(records=[r1])
        await collection.delete(record_uuids=[r1.uuid])

        query_results = await collection.query(query_vectors=[v1], limit=10)
        assert len(query_results[0].matches) == 0


# ── Partition isolation ──


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_query_only_returns_own_collection(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_a", config=config
        )
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_b", config=config
        )
        coll_a = await store.open_collection(namespace=NAMESPACE, name="tenant_a")
        coll_b = await store.open_collection(namespace=NAMESPACE, name="tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results_a = await coll_a.query(query_vectors=[v1], limit=10)
        results_b = await coll_b.query(query_vectors=[v1], limit=10)

        uuids_a = {match.record.uuid for match in results_a[0].matches}
        uuids_b = {match.record.uuid for match in results_b[0].matches}
        assert uuids_a == {r1.uuid}
        assert uuids_b == {r2.uuid}

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")

    @pytest.mark.asyncio
    async def test_get_only_returns_own_collection(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_a", config=config
        )
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_b", config=config
        )
        coll_a = await store.open_collection(namespace=NAMESPACE, name="tenant_a")
        coll_b = await store.open_collection(namespace=NAMESPACE, name="tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results = await _fetch_records(coll_a, [r1.uuid, r2.uuid])
        assert len(results) == 1
        assert results[0].uuid == r1.uuid

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")

    @pytest.mark.asyncio
    async def test_delete_only_affects_own_collection(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_a", config=config
        )
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_b", config=config
        )
        coll_a = await store.open_collection(namespace=NAMESPACE, name="tenant_a")
        coll_b = await store.open_collection(namespace=NAMESPACE, name="tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        await coll_a.delete(record_uuids=[r2.uuid])

        results = await _fetch_records(coll_b, [r2.uuid])
        assert len(results) == 1
        assert results[0].uuid == r2.uuid

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace="namespace_a", name="coll", config=config
        )
        await store.create_collection(
            namespace="namespace_b", name="coll", config=config
        )
        coll_a = await store.open_collection(namespace="namespace_a", name="coll")
        coll_b = await store.open_collection(namespace="namespace_b", name="coll")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results_a = await coll_a.query(query_vectors=[v1], limit=10)
        assert {match.record.uuid for match in results_a[0].matches} == {r1.uuid}

        await store.delete_collection(namespace="namespace_a", name="coll")
        await store.delete_collection(namespace="namespace_b", name="coll")

    @pytest.mark.asyncio
    async def test_delete_collection_does_not_affect_sibling(self, store):
        """Deleting one collection doesn't break a sibling sharing tables."""
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        coll_a = await store.open_or_create_collection(
            namespace=NAMESPACE, name="sibling_a", config=config
        )
        coll_b = await store.open_or_create_collection(
            namespace=NAMESPACE, name="sibling_b", config=config
        )

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)
        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        await store.delete_collection(namespace=NAMESPACE, name="sibling_a")

        results = await coll_b.query(query_vectors=[v1], limit=10)
        assert len(results[0].matches) == 1
        assert results[0].matches[0].record.uuid == r2.uuid

        await store.delete_collection(namespace=NAMESPACE, name="sibling_b")


class TestNoProperties:
    @pytest.mark.asyncio
    async def test_collection_without_properties(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=2)
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="no_props", config=config
        )
        r1 = _make_record(vector=[1.0, 0.0])
        await coll.upsert(records=[r1])

        results = await coll.query(query_vectors=[[1.0, 0.0]], limit=1)
        assert len(results[0].matches) == 1

        await store.delete_collection(namespace=NAMESPACE, name="no_props")


# ── Input validation ──


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_upsert_rejects_none_vector(self, collection):
        record = _make_record(vector=None)
        with pytest.raises(ValueError, match="vector=None"):
            await collection.upsert(records=[record])

    @pytest.mark.asyncio
    async def test_upsert_none_properties_treated_as_empty(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=v1, properties=None)
        await collection.upsert(records=[record])
        fetched = await _fetch_records(collection, [record.uuid])
        assert len(fetched) == 1
        assert fetched[0].properties == {}


# ── Score semantics ──


class TestScoreSemantics:
    @pytest.mark.asyncio
    async def test_cosine_higher_is_better(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await collection.upsert(records=[r1, r2])

        results = await collection.query(query_vectors=[v1], limit=2)
        scores = [m.cosine_similarity for m in results[0].matches]
        assert scores[0] > scores[1]


# ── Upsert behavior ──


class TestUpsertBehavior:
    @pytest.mark.asyncio
    async def test_upsert_replaces_vector_and_properties(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        record_uuid = uuid4()

        await collection.upsert(
            records=[
                _make_record(uuid=record_uuid, vector=v1, properties={"name": "alice"})
            ]
        )
        await collection.upsert(
            records=[
                _make_record(uuid=record_uuid, vector=v2, properties={"name": "bob"})
            ]
        )

        fetched = await _fetch_records(collection, [record_uuid])
        assert len(fetched) == 1
        properties = fetched[0].properties
        assert properties is not None
        assert properties["name"] == "bob"

        results = await collection.query(query_vectors=[v2], limit=1)
        assert results[0].matches[0].record.uuid == record_uuid
        assert results[0].matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)


# ── Filter edge cases ──


class TestFilterEdgeCases:
    @pytest.mark.asyncio
    async def test_or_filter_does_not_match_all(self, collection):
        """Regression: OR filter with parenthesization must not bypass rowid constraint."""
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "alice"})
        r2 = _make_record(vector=v1, properties={"name": "bob"})
        r3 = _make_record(vector=v1, properties={"name": "carol"})
        await collection.upsert(records=[r1, r2, r3])

        results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Or(
                left=Comparison(field="name", op="=", value="alice"),
                right=Comparison(field="name", op="=", value="carol"),
            ),
        )
        uuids = {m.record.uuid for m in results[0].matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert r2.uuid not in uuids
