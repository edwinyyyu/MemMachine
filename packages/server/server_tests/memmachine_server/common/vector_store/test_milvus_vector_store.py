"""Tests for MilvusVectorStore."""

# ruff: noqa: E402

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import override
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from server_tests.memmachine_server.common.vector_store.collection_lifecycle_contract import (
    CollectionLifecycleContract,
)
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_collection import (
    evaluate_filter,
)

pymilvus = pytest.importorskip("pymilvus")
DataType = pymilvus.DataType
AsyncMilvusClient = pymilvus.AsyncMilvusClient

from pymilvus.client.types import ConsistencyLevel

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.metrics_factory import OperationTracker
from memmachine_server.common.properties_json import decode_properties
from memmachine_server.common.vector_store.collection_registry import (
    Registration,
)
from memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry import (
    SQLAlchemyVectorStoreCollectionRegistry,
    SQLAlchemyVectorStoreCollectionRegistryParams,
)
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
)
from memmachine_server.common.vector_store.milvus_vector_store import (
    MilvusVectorStore,
    MilvusVectorStoreCollection,
    MilvusVectorStoreParams,
)

NAMESPACE = "test_namespace"
NAME = "test_name"
VECTOR_DIM = 3
VECTOR_STORE_NAME = "milvus_test"
REQUEST_TIMEOUT_SECONDS = 30
MAX_VARCHAR_LENGTH = 1024
PURGE_BATCH_SIZE = 10000


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _make_record(
    *,
    uuid: UUID | None = None,
    vector: list[float],
    properties: dict | None = None,
) -> Record:
    return Record(
        uuid=uuid or uuid4(),
        vector=vector,
        properties=properties or {},
    )


async def _settle(collection: MilvusVectorStoreCollection) -> None:
    """Return once the store's reads reflect every write made so far.

    The store reads at Bounded, which may lag its writes. A Strong read
    returns only once the server has applied every earlier write, and with
    one replica the store's later reads start from that point.
    """
    await collection._client.query(
        collection_name=collection._native_collection_name,
        filter=f'partition_key == "{collection._incarnation}"',
        output_fields=["id"],
        limit=1,
        consistency_level="Strong",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


async def _stored(
    collection: MilvusVectorStoreCollection, record_uuids: list[UUID]
) -> dict[UUID, dict]:
    """The entities Milvus holds under these UUIDs, read past the store at Strong.

    The store's reads may lag its writes by its consistency level; a Strong
    read reflects every write that returned before it.
    """
    rows = await collection._client.get(
        collection_name=collection._native_collection_name,
        ids=[collection._primary_id(record_uuid) for record_uuid in record_uuids],
        output_fields=["*"],
        consistency_level="Strong",
    )
    return {UUID(row["record_uuid"]): row for row in rows}


async def _incarnation_uuids(
    client: AsyncMilvusClient, native_collection_name: str, incarnation: UUID
) -> list[UUID]:
    """The record UUIDs Milvus holds under an incarnation, once per entity, read past the store at Strong."""
    rows = await client.query(
        collection_name=native_collection_name,
        filter=f'partition_key == "{incarnation}"',
        output_fields=["record_uuid"],
        limit=16384,
        consistency_level="Strong",
    )
    return [UUID(row["record_uuid"]) for row in rows]


async def _drain(store: MilvusVectorStore) -> None:
    """Run purge rounds until none is due, failing on a purge that never ends."""

    async def drain() -> None:
        while await store.purge_deleted_collections():
            pass

    await asyncio.wait_for(drain(), 60)


# Properties of every type, declared and not, for the tests against a model.
_MODEL_DECLARED: dict[str, type[PropertyValue]] = {
    "d_bool": bool,
    "d_int": int,
    "d_float": float,
    "d_str": str,
    "d_datetime": datetime,
}
_MODEL_PROPERTY_TYPES: dict[str, type[PropertyValue]] = {
    **_MODEL_DECLARED,
    "u_bool": bool,
    "u_int": int,
    "u_float": float,
    "u_str": str,
    "u_datetime": datetime,
}
_MODEL_CONFIG = VectorStoreCollectionConfig(
    vector_dimensions=VECTOR_DIM, indexed_properties_schema=_MODEL_DECLARED
)
_MODEL_INTS = [-7, -1, 0, 1, 2, 3, 1 << 40]
_MODEL_FLOATS = [-2.5, -0.5, 0.0, 0.25, 1.0, 3.75]
_MODEL_STRINGS = [
    "",
    "alpha",
    "Alpha",
    "beta",
    "a b",
    'q"uote',
    "back\\slash",
    "it's",
    "\u00fcn\u00ef",
]
# Instants a microsecond and a second apart, each written at varied offsets.
_MODEL_INSTANTS = [
    datetime(1999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
    datetime(2024, 6, 14, 12, tzinfo=UTC),
    datetime(2024, 6, 15, 12, tzinfo=UTC),
    datetime(2024, 6, 15, 12, 0, 0, 1, tzinfo=UTC),
    datetime(2024, 6, 15, 12, 0, 1, tzinfo=UTC),
]
_MODEL_OFFSETS = [
    UTC,
    timezone(timedelta(hours=5, minutes=30)),
    timezone(timedelta(hours=-8)),
    timezone(timedelta(hours=14)),
    timezone(timedelta(hours=-12)),
]


def _model_value(rng: random.Random, value_type: type[PropertyValue]) -> PropertyValue:
    if value_type is bool:
        return rng.choice([False, True])
    if value_type is int:
        return rng.choice(_MODEL_INTS)
    if value_type is float:
        return rng.choice(_MODEL_FLOATS)
    if value_type is str:
        return rng.choice(_MODEL_STRINGS)
    return rng.choice(_MODEL_INSTANTS).astimezone(rng.choice(_MODEL_OFFSETS))


def _model_properties(rng: random.Random) -> dict[str, PropertyValue]:
    """Properties of every type, each missing a quarter of the time."""
    return {
        key: _model_value(rng, value_type)
        for key, value_type in _MODEL_PROPERTY_TYPES.items()
        if rng.random() < 0.75
    }


def _model_filter(rng: random.Random, depth: int = 3) -> FilterExpr:
    """A random filter tree whose values have their properties' types.

    It compares with `!=` only through Not(=): on a property with no value,
    the store's `!=` holds, as the complement of `=`, where the model's does
    not.
    """
    if depth > 0 and rng.random() < 0.6:
        match rng.choice(("and", "or", "not")):
            case "not":
                return Not(expr=_model_filter(rng, depth - 1))
            case "and":
                return And(
                    left=_model_filter(rng, depth - 1),
                    right=_model_filter(rng, depth - 1),
                )
            case _:
                return Or(
                    left=_model_filter(rng, depth - 1),
                    right=_model_filter(rng, depth - 1),
                )
    key = rng.choice(list(_MODEL_PROPERTY_TYPES))
    value_type = _MODEL_PROPERTY_TYPES[key]
    match rng.choice(("is_null", "comparison", "in")):
        case "is_null":
            return IsNull(field=key)
        case "in" if value_type is int:
            return In(field=key, values=rng.sample(_MODEL_INTS, k=rng.randint(1, 3)))
        case "in" if value_type is str:
            return In(field=key, values=rng.sample(_MODEL_STRINGS, k=rng.randint(1, 3)))
        case _:
            op = "=" if value_type is bool else rng.choice(("=", "<", "<=", ">", ">="))
            return Comparison(field=key, op=op, value=_model_value(rng, value_type))


async def _model_matches(
    collection: MilvusVectorStoreCollection, property_filter: FilterExpr | None
) -> set[UUID]:
    """The UUIDs of every record of the collection the filter selects."""
    [result] = await collection.query(
        query_vectors=[_normalize([1.0, 0.0, 0.0])],
        limit=1000,
        property_filter=property_filter,
    )
    return {match.record_uuid for match in result.matches}


@pytest_asyncio.fixture
async def server_milvus_client(milvus_container):
    client = AsyncMilvusClient(uri=milvus_container.get_connection_url())
    yield client
    await client.close()


@pytest.fixture(
    params=[pytest.param("server_milvus_client", marks=pytest.mark.integration)],
)
def milvus_client(request):
    return request.getfixturevalue(request.param)


async def _started_store(
    client: AsyncMilvusClient, registry_engine: AsyncEngine, **overrides: int
) -> MilvusVectorStore:
    """A started store with its own registry object over the registry database."""
    collection_registry = SQLAlchemyVectorStoreCollectionRegistry(
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=registry_engine,
            vector_store_name=VECTOR_STORE_NAME,
            # Tombstones come due at once, so a test can purge right after
            # deleting.
            tombstone_retention_seconds=0,
        )
    )
    await collection_registry.startup()
    settings = {
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "max_varchar_length": MAX_VARCHAR_LENGTH,
        "purge_batch_size": PURGE_BATCH_SIZE,
        **overrides,
    }
    vector_store = MilvusVectorStore(
        MilvusVectorStoreParams(
            client=client, collection_registry=collection_registry, **settings
        )
    )
    await vector_store.startup()
    return vector_store


@pytest.fixture
def registry_url(tmp_path) -> str:
    """The registry database every store of a test shares."""
    return f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"


@pytest_asyncio.fixture
async def store(milvus_client, registry_url):
    registry_engine = create_async_engine(registry_url)
    vector_store = await _started_store(milvus_client, registry_engine)
    yield vector_store
    await vector_store.shutdown()
    await registry_engine.dispose()


@pytest_asyncio.fixture
async def other_store(milvus_container, registry_url):
    """A store as another process runs it beside `store`: its own Milvus
    client and registry connections, over the same Milvus and registry
    database."""
    client = AsyncMilvusClient(uri=milvus_container.get_connection_url())
    registry_engine = create_async_engine(registry_url)
    vector_store = await _started_store(client, registry_engine)
    yield vector_store
    await vector_store.shutdown()
    await registry_engine.dispose()
    await client.close()


# The client requests the store makes; each must carry the store's timeout.
_CLIENT_REQUESTS = (
    "has_collection",
    "create_collection",
    "list_indexes",
    "create_index",
    "load_collection",
    "query",
    "upsert",
    "search",
    "delete",
)


@pytest.mark.asyncio
async def test_every_request_carries_the_timeout(store, monkeypatch):
    # Its own namespace: a native collection an earlier test left behind
    # would let create_collection skip its Milvus request.
    namespace = "timed_namespace"
    spies = {}
    for name in _CLIENT_REQUESTS:
        spies[name] = MagicMock(wraps=getattr(store._client, name))
        monkeypatch.setattr(store._client, name, spies[name])

    await store.create_collection(
        namespace=namespace,
        name="timed",
        config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
    )
    coll = await store.open_collection(namespace=namespace, name="timed")
    assert coll is not None
    record, kept = (
        _make_record(vector=_normalize([1.0, 0.0, 0.0])),
        _make_record(vector=_normalize([0.0, 1.0, 0.0])),
    )
    await coll.upsert(records=[record, kept])
    await _settle(coll)
    await coll.query(query_vectors=[record.vector], limit=1)
    await coll.delete(record_uuids=[record.uuid])
    await store.delete_collection(namespace=namespace, name="timed")
    # The purge finds the record the deletion left and reclaims it.
    while await store.purge_deleted_collections():
        pass

    assert {name for name, spy in spies.items() if spy.call_count} >= set(
        _CLIENT_REQUESTS
    )
    for name, spy in spies.items():
        for call in spy.call_args_list:
            assert call.kwargs.get("timeout") == REQUEST_TIMEOUT_SECONDS, (name, call)


@pytest_asyncio.fixture
async def collection(store):
    await store.create_collection(
        namespace=NAMESPACE,
        name=NAME,
        config=VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
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


class TestCollectionLifecycle:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing_step", ["create_index", "load_collection"])
    async def test_a_creation_that_failed_part_way_is_as_if_never_attempted(
        self, store, monkeypatch, failing_step
    ):
        """The next creation completes the native collection a failed one left behind."""
        namespace = f"partial_{failing_step}"
        config = VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM, indexed_properties_schema={"name": str}
        )

        async def refuse(*args, **kwargs):
            raise pymilvus.MilvusException(message=f"{failing_step} refused")

        with monkeypatch.context() as patch:
            patch.setattr(store._client, failing_step, refuse)
            with pytest.raises(pymilvus.MilvusException, match="refused"):
                await store.create_collection(
                    namespace=namespace, name="partial", config=config
                )
        assert await store.open_collection(namespace=namespace, name="partial") is None

        await store.create_collection(
            namespace=namespace, name="partial", config=config
        )
        coll = await store.open_collection(namespace=namespace, name="partial")
        assert coll is not None
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"name": "alice"}
        )
        await coll.upsert(records=[record])
        await _settle(coll)
        [result] = await coll.query(
            query_vectors=[record.vector],
            limit=1,
            property_filter=Comparison(field="name", op="=", value="alice"),
        )
        assert [match.record_uuid for match in result.matches] == [record.uuid]
        native = coll._native_collection_name
        assert set(await store._client.list_indexes(native)) == {"vector", "_p_name"}
        await store.delete_collection(namespace=namespace, name="partial")

    @pytest.mark.asyncio
    async def test_create_open_delete(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="lifecycle",
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        coll = await store.open_collection(namespace=NAMESPACE, name="lifecycle")
        assert isinstance(coll, MilvusVectorStoreCollection)
        await store.delete_collection(namespace=NAMESPACE, name="lifecycle")

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, collection):
        with pytest.raises(VectorStoreCollectionAlreadyExistsError):
            await store.create_collection(
                namespace=NAMESPACE,
                name=NAME,
                config=collection.config,
            )

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, store):
        await store.delete_collection(namespace=NAMESPACE, name="nonexistent")

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
    async def test_same_config_shares_native_collection(self, store):
        schema: dict[str, type[PropertyValue]] = {"name": str}
        config = VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
            indexed_properties_schema=schema,
        )
        await store.create_collection(namespace=NAMESPACE, name="coll_a", config=config)
        await store.create_collection(namespace=NAMESPACE, name="coll_b", config=config)

        coll_a = await store.open_collection(namespace=NAMESPACE, name="coll_a")
        coll_b = await store.open_collection(namespace=NAMESPACE, name="coll_b")
        assert coll_a is not None
        assert coll_b is not None
        assert coll_a._native_collection_name == coll_b._native_collection_name

        await store.delete_collection(namespace=NAMESPACE, name="coll_a")
        await store.delete_collection(namespace=NAMESPACE, name="coll_b")

    @pytest.mark.asyncio
    async def test_native_collection_schema(self, store):
        """Each declared property is a typed, nullable, indexed field, a
        datetime with a field for its offset; the collection isolates tenants."""
        await store.create_collection(
            namespace=NAMESPACE,
            name="schema",
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
        coll = await store.open_collection(namespace=NAMESPACE, name="schema")
        assert coll is not None
        native = coll._native_collection_name

        schema = await store._client.describe_collection(native)
        fields = {field["name"]: field for field in schema["fields"]}
        assert schema["auto_id"] is False
        assert schema["enable_dynamic_field"] is False
        assert schema["properties"]["partitionkey.isolation"] == "True"
        assert fields["id"]["is_primary"] is True
        assert fields["partition_key"]["is_partition_key"] is True
        assert fields["vector"]["type"] == DataType.FLOAT_VECTOR
        assert fields["vector"]["params"]["dim"] == VECTOR_DIM
        assert fields["properties"]["type"] == DataType.JSON
        expected = {
            "_p_name": DataType.VARCHAR,
            "_p_age": DataType.INT64,
            "_p_score": DataType.DOUBLE,
            "_p_active": DataType.BOOL,
            "_p_created_at": DataType.TIMESTAMPTZ,
            "_tz_created_at": DataType.INT32,
        }
        for field_name, data_type in expected.items():
            assert fields[field_name]["type"] == data_type
            assert fields[field_name]["nullable"] is True
        assert fields["_p_name"]["params"]["max_length"] == MAX_VARCHAR_LENGTH
        indexed = {
            (await store._client.describe_index(native, index_name))["field_name"]
            for index_name in await store._client.list_indexes(native)
        }
        assert indexed == {
            "vector",
            "_p_name",
            "_p_age",
            "_p_score",
            "_p_active",
            "_p_created_at",
        }

        await store.delete_collection(namespace=NAMESPACE, name="schema")

    @pytest.mark.asyncio
    async def test_unsupported_metric_raises(self, store):
        with pytest.raises(ValueError, match="Milvus only supports"):
            await store.create_collection(
                namespace=NAMESPACE,
                name="bad_metric",
                config=VectorStoreCollectionConfig(
                    vector_dimensions=VECTOR_DIM,
                    similarity_metric=SimilarityMetric.MANHATTAN,
                ),
            )


async def _require_usable(collection: MilvusVectorStoreCollection) -> None:
    """Write a record to the collection, and find it by a filtered query."""
    record = _make_record(
        vector=_normalize([1.0, 0.0, 0.0]), properties={"name": "usable"}
    )
    await collection.upsert(records=[record])
    await _settle(collection)
    [result] = await collection.query(
        query_vectors=[record.vector],
        limit=10,
        property_filter=Comparison(field="name", op="=", value="usable"),
    )
    assert [match.record_uuid for match in result.matches] == [record.uuid]


class TestConcurrentPreparation:
    @pytest.mark.asyncio
    async def test_a_creation_finding_the_native_collection_mid_creation_completes_it(
        self, store, other_store, monkeypatch
    ):
        """Two stores create collections of one namespace and configuration at
        once, and the second finds the native collection the first created but
        has not yet indexed or loaded. The second's creation completes it, so
        its collection is usable before the first creation goes on, and both
        collections end usable with every index."""
        namespace = "raced_preparation"
        config = VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM, indexed_properties_schema={"name": str}
        )
        first_created = asyncio.Event()
        second_done = asyncio.Event()
        create_native_collection = store._client.create_collection

        async def create_and_hold(*args, **kwargs):
            await create_native_collection(*args, **kwargs)
            first_created.set()
            await asyncio.wait_for(second_done.wait(), 60)

        monkeypatch.setattr(store._client, "create_collection", create_and_hold)

        async def create_second() -> None:
            try:
                await asyncio.wait_for(first_created.wait(), 60)
                await other_store.create_collection(
                    namespace=namespace, name="second", config=config
                )
                second = await other_store.open_collection(
                    namespace=namespace, name="second"
                )
                assert second is not None
                await _require_usable(second)
            finally:
                second_done.set()

        await asyncio.gather(
            store.create_collection(namespace=namespace, name="first", config=config),
            create_second(),
        )

        first = await store.open_collection(namespace=namespace, name="first")
        assert first is not None
        await _require_usable(first)
        assert set(await store._client.list_indexes(first._native_collection_name)) == {
            "vector",
            "_p_name",
        }


@pytest.mark.asyncio
async def test_a_creation_that_loses_the_native_collection_to_another_completes_it(
    registry_url,
):
    """A creation whose create request Milvus refuses as existing, because
    another creator made the native collection after this one checked, still
    indexes and loads the native collection, and the collection opens."""
    indexed: set[str] = set()
    loaded: list[str] = []

    async def list_indexes(collection_name: str, **kwargs) -> list[str]:
        return sorted(indexed)

    async def create_index(collection_name: str, index_params, **kwargs) -> None:
        indexed.update(index.field_name for index in index_params)

    async def load_collection(collection_name: str, **kwargs) -> None:
        loaded.append(collection_name)

    client = MagicMock(spec=AsyncMilvusClient)
    client.prepare_index_params = AsyncMilvusClient.prepare_index_params
    client.create_schema = AsyncMilvusClient.create_schema
    client.has_collection = AsyncMock(return_value=False)
    client.create_collection = AsyncMock(
        side_effect=pymilvus.MilvusException(message="collection already exists")
    )
    client.list_indexes = AsyncMock(side_effect=list_indexes)
    client.create_index = AsyncMock(side_effect=create_index)
    client.load_collection = AsyncMock(side_effect=load_collection)
    registry_engine = create_async_engine(registry_url)
    store = await _started_store(client, registry_engine)
    config = VectorStoreCollectionConfig(
        vector_dimensions=VECTOR_DIM, indexed_properties_schema={"name": str}
    )

    await store.create_collection(namespace=NAMESPACE, name=NAME, config=config)

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is not None
    assert indexed == {"vector", "_p_name"}
    assert loaded
    await registry_engine.dispose()


class TestUpsertAndQuery:
    @pytest.mark.asyncio
    async def test_a_query_may_ask_for_hundreds_of_results(self, collection):
        records = [
            _make_record(vector=_normalize([1.0, float(i), 0.0])) for i in range(3)
        ]
        await collection.upsert(records=records)
        await _settle(collection)

        [result] = await collection.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=200
        )

        assert {match.record_uuid for match in result.matches} == {
            record.uuid for record in records
        }

    @pytest.mark.asyncio
    async def test_upsert_and_query_basic(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        v3 = _normalize([1.0, 0.1, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        r3 = _make_record(vector=v3, properties={"name": "c"})

        await collection.upsert(records=[r1, r2, r3])
        await _settle(collection)

        query_results = await collection.query(query_vectors=[v1], limit=3)
        matches = query_results[0].matches

        assert len(matches) == 3
        assert matches[0].record_uuid == r1.uuid
        assert matches[1].record_uuid == r3.uuid
        assert matches[2].record_uuid == r2.uuid
        assert matches[0].score >= matches[1].score >= matches[2].score
        assert matches[0].score == pytest.approx(1.0)
        assert matches[2].score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await collection.upsert(records=[r1, r2])
        await _settle(collection)

        query_results = await collection.query(
            query_vectors=[v1], limit=10, score_threshold=0.9
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_batch_multiple_vectors(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await collection.upsert(records=[r1, r2])
        await _settle(collection)

        all_results = await collection.query(query_vectors=[v1, v2], limit=1)

        assert len(all_results) == 2
        assert all_results[0].matches[0].record_uuid == r1.uuid
        assert all_results[1].matches[0].record_uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_upsert_removes_stale_filter_fields(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=v1, properties={"name": "old"})
        await collection.upsert(records=[record])
        await _settle(collection)

        await collection.upsert(
            records=[Record(uuid=record.uuid, vector=v1, properties={})]
        )
        await _settle(collection)

        results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=IsNull(field="name"),
        )
        assert {match.record_uuid for match in results[0].matches} == {record.uuid}
        [old] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="name", op="=", value="old"),
        )
        assert old.matches == []

    @pytest.mark.asyncio
    async def test_upsert_failure_preserves_existing_record(
        self, collection, monkeypatch
    ):
        old_vector = _normalize([1.0, 0.0, 0.0])
        new_vector = _normalize([0.0, 1.0, 0.0])
        record = _make_record(vector=old_vector, properties={"name": "old"})
        await collection.upsert(records=[record])
        await _settle(collection)

        async def fail_upsert(*args, **kwargs):
            raise RuntimeError("upsert failed")

        monkeypatch.setattr(collection._client, "upsert", fail_upsert)

        with pytest.raises(RuntimeError, match="upsert failed"):
            await collection.upsert(
                records=[
                    Record(
                        uuid=record.uuid,
                        vector=new_vector,
                        properties={"name": "new"},
                    )
                ]
            )

        stored = (await _stored(collection, [record.uuid]))[record.uuid]
        assert list(stored["vector"]) == old_vector
        assert stored["_p_name"] == "old"


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
        await _settle(collection)
        return r1, r2, r3, v1

    async def _query(self, collection, query_vec, field, op, value):
        all_results = await collection.query(
            query_vectors=[query_vec],
            limit=10,
            property_filter=Comparison(field=field, op=op, value=value),
        )
        return {match.record_uuid for match in all_results[0].matches}

    @pytest.mark.asyncio
    async def test_scalar_filters(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)

        assert await self._query(collection, v1, "name", "=", "alice") == {r1.uuid}
        assert await self._query(collection, v1, "name", "!=", "alice") == {
            r2.uuid,
            r3.uuid,
        }
        assert await self._query(collection, v1, "age", ">", 30) == {r3.uuid}
        assert await self._query(collection, v1, "age", "<=", 30) == {
            r1.uuid,
            r2.uuid,
        }
        assert await self._query(collection, v1, "score", ">=", 8.0) == {
            r1.uuid,
            r3.uuid,
        }
        assert await self._query(collection, v1, "active", "=", True) == {
            r1.uuid,
            r3.uuid,
        }

    @pytest.mark.asyncio
    async def test_datetime_filters(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt_other = datetime(2024, 6, 15, 18, 0, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"created_at": dt_utc})
        r2 = _make_record(vector=v2, properties={"created_at": dt_other})
        await collection.upsert(records=[r1, r2])
        await _settle(collection)

        plus5 = timezone(timedelta(hours=5))
        dt_filter = datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)
        uuids = await self._query(collection, v1, "created_at", "=", dt_filter)
        assert uuids == {r1.uuid}

    @pytest.mark.asyncio
    async def test_is_null_and_not_null(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_missing = _make_record(vector=v2, properties={"age": 25})
        await collection.upsert(records=[r_has_value, r_missing])
        await _settle(collection)

        null_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=IsNull(field="name"),
        )
        assert {m.record_uuid for m in null_results[0].matches} == {r_missing.uuid}

        not_null_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Not(expr=IsNull(field="name")),
        )
        assert {m.record_uuid for m in not_null_results[0].matches} == {
            r_has_value.uuid
        }

    @pytest.mark.asyncio
    async def test_in_and_or_not(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)

        in_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=In(field="name", values=["alice", "carol"]),
        )
        assert {m.record_uuid for m in in_results[0].matches} == {r1.uuid, r3.uuid}

        and_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=And(
                left=Comparison(field="active", op="=", value=True),
                right=Comparison(field="age", op=">", value=30),
            ),
        )
        assert {m.record_uuid for m in and_results[0].matches} == {r3.uuid}

        or_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Or(
                left=Comparison(field="name", op="=", value="alice"),
                right=Comparison(field="name", op="=", value="bob"),
            ),
        )
        assert {m.record_uuid for m in or_results[0].matches} == {r1.uuid, r2.uuid}

    @pytest.mark.asyncio
    async def test_negation_is_the_complement_missing_values_included(self, collection):
        """A condition on a property with no value is false, so its negation,
        `!=` included, holds there."""
        r1, r2, r3, v1 = await self._setup(collection)
        bare = _make_record(vector=_normalize([1.0, 0.3, 0.0]), properties={})
        await collection.upsert(records=[bare])
        await _settle(collection)

        async def uuids(expr):
            [result] = await collection.query(
                query_vectors=[v1], limit=10, property_filter=expr
            )
            return {match.record_uuid for match in result.matches}

        assert await uuids(Comparison(field="name", op="!=", value="alice")) == {
            r2.uuid,
            r3.uuid,
            bare.uuid,
        }
        assert await uuids(
            Not(expr=Comparison(field="name", op="=", value="alice"))
        ) == {r2.uuid, r3.uuid, bare.uuid}
        assert await uuids(Not(expr=Comparison(field="age", op=">", value=30))) == {
            r1.uuid,
            r2.uuid,
            bare.uuid,
        }
        assert await uuids(Not(expr=In(field="name", values=["alice", "bob"]))) == {
            r3.uuid,
            bare.uuid,
        }
        assert await uuids(
            Not(
                expr=And(
                    left=Comparison(field="active", op="=", value=True),
                    right=Comparison(field="age", op=">", value=30),
                )
            )
        ) == {r1.uuid, r2.uuid, bare.uuid}
        assert await uuids(Not(expr=Not(expr=IsNull(field="name")))) == {bare.uuid}

    @pytest.mark.asyncio
    async def test_filters_on_undeclared_properties(self, collection):
        """A property the schema does not declare is stored and filtered too."""
        v1 = _normalize([1.0, 0.0, 0.0])
        red = _make_record(vector=v1, properties={"color": "red", "size": 3})
        blue = _make_record(
            vector=_normalize([1.0, 0.1, 0.0]), properties={"color": "blue"}
        )
        await collection.upsert(records=[red, blue])
        await _settle(collection)

        async def uuids(expr):
            [result] = await collection.query(
                query_vectors=[v1], limit=10, property_filter=expr
            )
            return {match.record_uuid for match in result.matches}

        assert await uuids(Comparison(field="color", op="=", value="red")) == {red.uuid}
        assert await uuids(Comparison(field="size", op=">=", value=3)) == {red.uuid}
        assert await uuids(In(field="color", values=["blue", "green"])) == {blue.uuid}
        assert await uuids(IsNull(field="size")) == {blue.uuid}
        assert await uuids(Comparison(field="size", op="!=", value=3)) == {blue.uuid}
        stored = await _stored(collection, [red.uuid])
        assert decode_properties(stored[red.uuid]["properties"]) == {
            "color": "red",
            "size": 3,
        }

    @pytest.mark.asyncio
    async def test_a_declared_datetime_is_stored_with_its_offset(self, collection):
        written = datetime(
            2024, 6, 15, 17, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))
        )
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"created_at": written}
        )
        await collection.upsert(records=[record])

        stored = (await _stored(collection, [record.uuid]))[record.uuid]
        assert datetime.fromisoformat(stored["_p_created_at"]) == written
        assert stored["_tz_created_at"] == 5 * 3600 + 30 * 60

    @pytest.mark.asyncio
    async def test_datetime_filters_compare_instants_across_offsets(self, collection):
        base = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        plus5 = timezone(timedelta(hours=5))
        records = [
            _make_record(
                vector=_normalize([1.0, 0.1 * index, 0.0]),
                properties={"created_at": instant},
            )
            for index, instant in enumerate(
                [
                    base,
                    (base + timedelta(microseconds=1)).astimezone(plus5),
                    base - timedelta(days=1),
                ]
            )
        ]
        await collection.upsert(records=records)
        await _settle(collection)
        v1 = _normalize([1.0, 0.0, 0.0])

        async def uuids(expr):
            [result] = await collection.query(
                query_vectors=[v1], limit=10, property_filter=expr
            )
            return {match.record_uuid for match in result.matches}

        same_instant = base.astimezone(plus5)
        assert await uuids(
            Comparison(field="created_at", op="=", value=same_instant)
        ) == {records[0].uuid}
        assert await uuids(Comparison(field="created_at", op=">", value=base)) == {
            records[1].uuid
        }
        assert await uuids(
            Comparison(field="created_at", op="<", value=same_instant)
        ) == {records[2].uuid}

    @pytest.mark.asyncio
    async def test_a_value_of_another_type_matches_nothing(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)

        [matched] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="age", op="=", value="thirty"),
        )
        assert matched.matches == []
        [complement] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="age", op="!=", value="thirty"),
        )
        assert {m.record_uuid for m in complement.matches} == {
            r1.uuid,
            r2.uuid,
            r3.uuid,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("key", "value"),
        [("age", "old"), ("age", 30.5), ("age", True), ("score", 3), ("score", "high")],
    )
    async def test_a_declared_property_of_another_type_is_refused(
        self, collection, key, value
    ):
        with pytest.raises(ValueError, match=f"{key!r} is declared"):
            await collection.upsert(
                records=[
                    _make_record(
                        vector=_normalize([1.0, 0.0, 0.0]), properties={key: value}
                    )
                ]
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "dimensions", [VECTOR_DIM - 1, VECTOR_DIM + 1], ids=["too_short", "too_long"]
    )
    async def test_a_vector_of_another_width_is_refused(self, collection, dimensions):
        vector = [1.0] * dimensions
        with pytest.raises(ValueError, match="dimensions"):
            await collection.upsert(records=[_make_record(vector=vector)])
        with pytest.raises(ValueError, match="dimensions"):
            await collection.query(query_vectors=[vector], limit=1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("coordinate", [math.nan, math.inf], ids=["nan", "inf"])
    async def test_a_query_vector_with_a_coordinate_that_is_not_finite_is_refused(
        self, collection, coordinate
    ):
        with pytest.raises(ValueError, match="not finite"):
            await collection.query(
                query_vectors=[[coordinate] + [1.0] * (VECTOR_DIM - 1)], limit=1
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
    async def test_a_score_threshold_that_is_not_finite_is_refused(
        self, collection, threshold
    ):
        with pytest.raises(ValueError, match="not finite"):
            await collection.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])],
                limit=1,
                score_threshold=threshold,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("limit", [0, -1])
    async def test_a_limit_that_is_not_positive_is_refused(self, collection, limit):
        with pytest.raises(ValueError, match="not positive"):
            await collection.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=limit
            )


class TestFilterModel:
    @pytest.mark.asyncio
    async def test_random_filters_agree_with_the_model(self, store):
        """Seeded filter trees over properties of every type, declared and not,
        datetimes at varied offsets and missing values included, select the
        records the in-memory evaluator selects."""
        rng = random.Random(1736)
        # Its own native collection, holding only the model's records.
        namespace = "filter_model"
        await store.create_collection(
            namespace=namespace, name="model", config=_MODEL_CONFIG
        )
        coll = await store.open_collection(namespace=namespace, name="model")
        assert coll is not None
        records = [
            Record(
                uuid=record_uuid,
                vector=_normalize([1.0, rng.random(), rng.random()]),
                properties=_model_properties(rng),
            )
            for record_uuid in sorted(uuid4() for _ in range(48))
        ]
        await coll.upsert(records=records)
        await _settle(coll)

        for _ in range(200):
            property_filter = _model_filter(rng)
            assert await _model_matches(coll, property_filter) == {
                record.uuid
                for record in records
                if evaluate_filter(property_filter, record.properties)
            }, property_filter

        await store.delete_collection(namespace=namespace, name="model")


@dataclass
class _ModelCollections:
    """Collections of one store, with the records a model says each holds."""

    store: MilvusVectorStore
    namespace: str
    handles: dict[str, MilvusVectorStoreCollection]
    records: dict[str, dict[UUID, Record]]
    dead_incarnations: list[UUID] = field(default_factory=list)

    async def upsert(self, name: str, batch: list[Record]) -> list[Record]:
        """Upsert a batch, and return the records it replaced."""
        await self.handles[name].upsert(records=batch)
        records = self.records[name]
        replaced = [records[record.uuid] for record in batch if record.uuid in records]
        records.update({record.uuid: record for record in batch})
        return replaced

    async def delete(self, name: str, record_uuids: list[UUID]) -> None:
        await self.handles[name].delete(record_uuids=record_uuids)
        for record_uuid in record_uuids:
            self.records[name].pop(record_uuid, None)

    async def recreate(self, name: str) -> None:
        self.dead_incarnations.append(self.handles[name]._incarnation)
        await self.store.delete_collection(namespace=self.namespace, name=name)
        await self.store.create_collection(
            namespace=self.namespace, name=name, config=_MODEL_CONFIG
        )
        handle = await self.store.open_collection(namespace=self.namespace, name=name)
        assert handle is not None
        self.handles[name] = handle
        self.records[name] = {}

    async def purge(self) -> None:
        """Drain the purge, and check that no deleted collection's record remains."""
        await _drain(self.store)
        for native in {
            handle._native_collection_name for handle in self.handles.values()
        }:
            for incarnation in self.dead_incarnations:
                assert (
                    await _incarnation_uuids(self.store._client, native, incarnation)
                    == []
                ), incarnation

    async def check(self, rng: random.Random) -> None:
        """Check what each collection holds, and what its settled queries select."""
        for name, handle in self.handles.items():
            records = self.records[name]
            stored = await _incarnation_uuids(
                self.store._client,
                handle._native_collection_name,
                handle._incarnation,
            )
            assert sorted(stored) == sorted(records), name
            await _settle(handle)
            assert await _model_matches(handle, None) == set(records), name
            for _ in range(3):
                property_filter = _model_filter(rng)
                assert await _model_matches(handle, property_filter) == {
                    record.uuid
                    for record in records.values()
                    if evaluate_filter(property_filter, record.properties)
                }, (name, property_filter)

    async def check_replaced(self, name: str, replaced: list[Record]) -> None:
        """Check that a replaced record's old values select it no more."""
        for old in replaced:
            current = self.records[name][old.uuid]
            for key, value in old.properties.items():
                old_value = Comparison(field=key, op="=", value=value)
                if not evaluate_filter(old_value, current.properties):
                    assert old.uuid not in await _model_matches(
                        self.handles[name], old_value
                    ), old_value


class TestOperationModel:
    @pytest.mark.asyncio
    async def test_a_random_sequence_of_operations_agrees_with_a_model(self, store):
        """Seeded upserts, new and replacing, deletes of present and absent
        records, deletion and recreation of a collection, and purge drains, on
        two collections sharing a native collection: after each step, each
        collection holds the model's records, its settled queries select what
        the model selects, and a replaced record's old values select nothing."""
        rng = random.Random(1736)
        namespace = "operation_model"
        names = ("alpha", "beta")
        # Few UUIDs, so each recurs in both collections and across their lives.
        pool = sorted(uuid4() for _ in range(12))
        handles = {}
        for name in names:
            await store.create_collection(
                namespace=namespace, name=name, config=_MODEL_CONFIG
            )
            handles[name] = await store.open_collection(namespace=namespace, name=name)
            assert handles[name] is not None
        model = _ModelCollections(
            store=store,
            namespace=namespace,
            handles=handles,
            records={name: {} for name in names},
        )

        for step in range(60):
            name = rng.choice(names)
            replaced: list[Record] = []
            action = rng.choices(
                ("upsert", "delete", "recreate", "purge"), weights=(8, 3, 1, 1)
            )[0]
            match action:
                case "upsert":
                    replaced = await model.upsert(
                        name,
                        [
                            Record(
                                uuid=record_uuid,
                                vector=_normalize([1.0, rng.random(), rng.random()]),
                                properties=_model_properties(rng),
                            )
                            for record_uuid in rng.sample(pool, k=rng.randint(1, 4))
                        ],
                    )
                case "delete":
                    await model.delete(name, rng.sample(pool, k=rng.randint(1, 4)))
                case "recreate":
                    await model.recreate(name)
                case _:
                    await model.purge()
            try:
                await model.check(rng)
                await model.check_replaced(name, replaced)
            except AssertionError as error:
                raise AssertionError(f"after step {step}, {action} {name}") from error

        for name in names:
            await store.delete_collection(namespace=namespace, name=name)


class TestScores:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            (SimilarityMetric.COSINE, 0.6),
            (SimilarityMetric.DOT, 1.2),
            (SimilarityMetric.EUCLIDEAN, math.sqrt(2.0 * 2.0 + 1.0 - 2.0 * 1.2)),
        ],
    )
    async def test_scores_are_the_metric_values(self, store, metric, expected):
        """Scores come from the server: cosine similarity, inner product, and
        Euclidean distance (Milvus returns it squared)."""
        name = f"scores_{metric.value}"
        await store.create_collection(
            namespace=NAMESPACE,
            name=name,
            config=VectorStoreCollectionConfig(
                vector_dimensions=VECTOR_DIM, similarity_metric=metric
            ),
        )
        collection = await store.open_collection(namespace=NAMESPACE, name=name)
        assert collection is not None
        record = _make_record(vector=[1.2, 1.6, 0.0])
        await collection.upsert(records=[record])
        await _settle(collection)

        [result] = await collection.query(query_vectors=[[1.0, 0.0, 0.0]], limit=1)
        assert result.matches[0].score == pytest.approx(expected, abs=1e-3)
        await store.delete_collection(namespace=NAMESPACE, name=name)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("metric", "vectors", "threshold", "expected_scores"),
        [
            # Distances 1, 2 and 5; the threshold lies between 2 and its square.
            (
                SimilarityMetric.EUCLIDEAN,
                [[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 0.0, 5.0]],
                3.0,
                [1.0, 2.0],
            ),
            (
                SimilarityMetric.DOT,
                [[3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                1.5,
                [3.0, 2.0],
            ),
            (
                SimilarityMetric.COSINE,
                [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 1.0, 0.0]],
                0.5,
                [1.0, 0.6],
            ),
        ],
        ids=["euclidean", "dot", "cosine"],
    )
    async def test_a_threshold_keeps_the_matches_within_it_best_first(
        self, store, metric, vectors, threshold, expected_scores
    ):
        """Of three records at increasing distance from the query, the two
        within the threshold match, best first, scored by the metric."""
        name = f"threshold_{metric.value}"
        await store.create_collection(
            namespace=NAMESPACE,
            name=name,
            config=VectorStoreCollectionConfig(
                vector_dimensions=VECTOR_DIM, similarity_metric=metric
            ),
        )
        collection = await store.open_collection(namespace=NAMESPACE, name=name)
        assert collection is not None
        nearest, middle, farthest = (_make_record(vector=vector) for vector in vectors)
        await collection.upsert(records=[farthest, nearest, middle])
        await _settle(collection)

        [result] = await collection.query(
            query_vectors=[[1.0, 0.0, 0.0]], limit=10, score_threshold=threshold
        )

        assert [match.record_uuid for match in result.matches] == [
            nearest.uuid,
            middle.uuid,
        ]
        assert [match.score for match in result.matches] == pytest.approx(
            expected_scores, abs=1e-3
        )
        await store.delete_collection(namespace=NAMESPACE, name=name)


# The client's reads; each takes an optional consistency level.
_CLIENT_READS = ("search", "query", "get", "hybrid_search")
# The levels at which a read reflects every write that returned at least
# common.gracefulTime before it, given by name or by value.
_BOUNDED_OR_STRONGER = {
    "Bounded",
    "Strong",
    ConsistencyLevel.Bounded,
    ConsistencyLevel.Strong,
}


class TestConsistency:
    @pytest.mark.asyncio
    async def test_the_store_reads_at_bounded_or_stronger(self, store, monkeypatch):
        """The native collection reads at Bounded, and no read of the store
        names a weaker level, so a read lags writes by at most
        common.gracefulTime: the purge and the tombstone retention rely on it."""
        # Its own namespace, so the store creates the native collection.
        namespace = "bounded_namespace"
        spies = {}
        for name in _CLIENT_READS:
            spies[name] = MagicMock(wraps=getattr(store._client, name))
            monkeypatch.setattr(store._client, name, spies[name])
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)

        await store.create_collection(
            namespace=namespace, name="bounded", config=config
        )
        coll = await store.open_collection(namespace=namespace, name="bounded")
        assert coll is not None
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await coll.upsert(records=[record])
        await _settle(coll)
        await coll.query(query_vectors=[record.vector], limit=1)
        await store.delete_collection(namespace=namespace, name="bounded")
        while await store.purge_deleted_collections():
            pass

        description = await store._client.describe_collection(
            coll._native_collection_name
        )
        assert description["consistency_level"] == ConsistencyLevel.Bounded
        calls = [
            (name, call) for name, spy in spies.items() for call in spy.call_args_list
        ]
        assert calls
        for name, call in calls:
            level = call.kwargs.get("consistency_level")
            assert level is None or level in _BOUNDED_OR_STRONGER, (name, call)


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_records(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await collection.upsert(records=[r1, r2])
        await _settle(collection)

        await collection.delete(record_uuids=[r1.uuid])

        assert set(await _stored(collection, [r1.uuid, r2.uuid])) == {r2.uuid}

    @pytest.mark.asyncio
    async def test_deleting_records_it_does_not_hold_succeeds(self, collection):
        """Milvus accepts the delete of a primary key it does not hold."""
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await collection.upsert(records=[record])
        await _settle(collection)

        await collection.delete(record_uuids=[record.uuid, uuid4(), uuid4()])

        assert await _stored(collection, [record.uuid]) == {}


@dataclass(frozen=True)
class _CurrentRegistration(Registration):
    """A registration whose collection is never deleted."""

    @override
    async def require_current(self) -> None:
        return None


@pytest.mark.asyncio
async def test_a_delete_milvus_does_not_accept_in_full_raises():
    """A delete Milvus accepts for fewer primary keys than the store sent raises."""
    client = MagicMock(spec=AsyncMilvusClient)
    client.delete = AsyncMock(return_value={"delete_count": 1})
    collection = MilvusVectorStoreCollection(
        client=client,
        native_collection_name="native",
        registration=_CurrentRegistration(
            namespace=NAMESPACE,
            name=NAME,
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
            incarnation=uuid4(),
        ),
        tracker=OperationTracker(None, prefix="test"),
        request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )

    with pytest.raises(pymilvus.MilvusException):
        await collection.delete(record_uuids=[uuid4(), uuid4()])


@pytest.mark.asyncio
async def test_a_purge_round_milvus_does_not_accept_in_full_raises(registry_url):
    """A purge round raises when Milvus accepts the delete of fewer primary
    keys than the round listed."""
    client = MagicMock(spec=AsyncMilvusClient)
    client.prepare_index_params = AsyncMilvusClient.prepare_index_params
    # The native collection exists with its index, and holds two entities of
    # the deleted collection.
    client.has_collection = AsyncMock(return_value=True)
    client.list_indexes = AsyncMock(return_value=["vector"])
    client.load_collection = AsyncMock(return_value=None)
    client.query = AsyncMock(return_value=[{"id": "listed_a"}, {"id": "listed_b"}])
    client.delete = AsyncMock(return_value={"delete_count": 1})
    registry_engine = create_async_engine(registry_url)
    store = await _started_store(client, registry_engine)
    config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
    await store.create_collection(namespace=NAMESPACE, name=NAME, config=config)
    await store.delete_collection(namespace=NAMESPACE, name=NAME)

    with pytest.raises(pymilvus.MilvusException):
        await store.purge_deleted_collections()
    await registry_engine.dispose()


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_same_uuid_can_exist_in_different_logical_collections(self, store):
        """The same UUID in two collections is two records, written and deleted apart."""
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

        record_uuid = uuid4()
        v1 = _normalize([1.0, 0.0, 0.0])
        await coll_a.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "a"})]
        )
        await _settle(coll_a)
        await coll_b.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "b"})]
        )
        await _settle(coll_b)

        stored_a = await _stored(coll_a, [record_uuid])
        stored_b = await _stored(coll_b, [record_uuid])

        assert decode_properties(stored_a[record_uuid]["properties"]) == {"name": "a"}
        assert decode_properties(stored_b[record_uuid]["properties"]) == {"name": "b"}

        await coll_a.delete(record_uuids=[record_uuid])
        await _settle(coll_b)

        assert await _stored(coll_a, [record_uuid]) == {}
        assert set(await _stored(coll_b, [record_uuid])) == {record_uuid}
        [kept] = await coll_b.query(query_vectors=[v1], limit=10)
        assert [match.record_uuid for match in kept.matches] == [record_uuid]

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")

    @pytest.mark.asyncio
    async def test_a_query_returns_only_its_own_collections_records(self, store):
        """Collections of one namespace and configuration answer with their own
        records alone, filtered or not, though the other's records match too."""
        config = VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM, indexed_properties_schema={"name": str}
        )
        vector = _normalize([1.0, 0.0, 0.0])
        own_records = {}
        for name in ("query_a", "query_b"):
            await store.create_collection(namespace=NAMESPACE, name=name, config=config)
            coll = await store.open_collection(namespace=NAMESPACE, name=name)
            assert coll is not None
            records = [
                _make_record(
                    vector=vector, properties={"name": "shared", "color": "red"}
                )
                for _ in range(2)
            ]
            await coll.upsert(records=records)
            own_records[name] = (coll, {record.uuid for record in records})

        for coll, own in own_records.values():
            await _settle(coll)
            for property_filter in (
                None,
                Comparison(field="name", op="=", value="shared"),
                Comparison(field="color", op="=", value="red"),
                Not(expr=IsNull(field="name")),
            ):
                [result] = await coll.query(
                    query_vectors=[vector], limit=10, property_filter=property_filter
                )
                assert {match.record_uuid for match in result.matches} == own, (
                    property_filter
                )

        for name in own_records:
            await store.delete_collection(namespace=NAMESPACE, name=name)


class TestPurgeBatches:
    @pytest.mark.asyncio
    async def test_a_purge_round_reclaims_at_most_one_batch(self, store):
        store = MilvusVectorStore(
            MilvusVectorStoreParams(
                client=store._client,
                collection_registry=store._collection_registry,
                request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                max_varchar_length=MAX_VARCHAR_LENGTH,
                purge_batch_size=2,
            )
        )
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="batched", config=config
        )
        collection = await store.open_collection(namespace=NAMESPACE, name="batched")
        assert collection is not None
        await collection.upsert(
            records=[
                _make_record(vector=_normalize([1.0, float(i), 0.0])) for i in range(5)
            ]
        )
        await _settle(collection)
        incarnation = collection._incarnation
        await store.delete_collection(namespace=NAMESPACE, name="batched")
        native = MilvusVectorStore._build_native_collection_name(NAMESPACE, config)

        async def left_of_the_incarnation() -> int:
            return len(
                await store._client.query(
                    collection_name=native,
                    filter=f'partition_key == "{incarnation}"',
                    output_fields=["id"],
                    limit=16384,
                    consistency_level="Strong",
                )
            )

        left_after_each_round = []
        while await store.purge_deleted_collections():
            await _settle(collection)
            left_after_each_round.append(await left_of_the_incarnation())
        # Batches of 2, then a round that finds nothing and removes the tombstone.
        assert left_after_each_round == [3, 1, 0, 0]


class TestPurge:
    @pytest.mark.asyncio
    async def test_a_purge_round_on_a_dropped_native_collection_retires_the_tombstone(
        self, store
    ):
        """A deleted collection whose native collection is gone holds nothing:
        its purge round raises nothing and removes the tombstone."""
        namespace = "dropped_native"
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=namespace, name="dropped", config=config
        )
        coll = await store.open_collection(namespace=namespace, name="dropped")
        assert coll is not None
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])
        await store.delete_collection(namespace=namespace, name="dropped")
        await store._client.drop_collection(coll._native_collection_name)

        await _drain(store)
        assert await store.purge_deleted_collections() is False

    @pytest.mark.asyncio
    async def test_purgers_on_two_stores_reclaim_the_deleted_collections_alone(
        self, store, other_store
    ):
        """Purgers on two stores sharing a registry, draining at once in small
        batches, reclaim every deleted collection's records and leave the live
        collection's."""
        namespace = "two_purgers"
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        handles = {}
        for name in ("live", "dead_0", "dead_1", "dead_2", "dead_3"):
            await store.create_collection(namespace=namespace, name=name, config=config)
            handle = await store.open_collection(namespace=namespace, name=name)
            assert handle is not None
            await handle.upsert(
                records=[
                    _make_record(vector=_normalize([1.0, 0.1 * index, 0.0]))
                    for index in range(7)
                ]
            )
            handles[name] = handle
        live = handles.pop("live")
        live_uuids = await _incarnation_uuids(
            store._client, live._native_collection_name, live._incarnation
        )
        await _settle(live)
        for name in handles:
            await store.delete_collection(namespace=namespace, name=name)
        purgers = [
            MilvusVectorStore(
                MilvusVectorStoreParams(
                    client=purging._client,
                    collection_registry=purging._collection_registry,
                    request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                    purge_batch_size=3,
                )
            )
            for purging in (store, other_store)
        ]

        await asyncio.gather(*(_drain(purger) for purger in purgers))

        for handle in handles.values():
            assert (
                await _incarnation_uuids(
                    store._client, handle._native_collection_name, handle._incarnation
                )
                == []
            )
        assert sorted(
            await _incarnation_uuids(
                store._client, live._native_collection_name, live._incarnation
            )
        ) == sorted(live_uuids)
        assert len(live_uuids) == 7
        assert [await purger.purge_deleted_collections() for purger in purgers] == [
            False,
            False,
        ]
        await store.delete_collection(namespace=namespace, name="live")


class TestLifecycleContract(CollectionLifecycleContract):
    """The collection lifecycle contract, against this store."""

    @staticmethod
    async def count_stored(store, namespace: str, config) -> int:
        native = MilvusVectorStore._build_native_collection_name(namespace, config)
        rows = await store._client.query(
            collection_name=native,
            filter='id != ""',
            output_fields=["id"],
            limit=16384,
            consistency_level="Strong",
        )
        return len(list(rows))

    settle = staticmethod(_settle)

    @staticmethod
    async def stored_uuids(collection) -> set[UUID]:
        rows = await collection._client.query(
            collection_name=collection._native_collection_name,
            filter=f'partition_key == "{collection._incarnation}"',
            output_fields=["record_uuid"],
            limit=16384,
            consistency_level="Strong",
        )
        return {UUID(row["record_uuid"]) for row in rows}
