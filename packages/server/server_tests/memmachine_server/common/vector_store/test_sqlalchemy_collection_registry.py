"""Tests for SQLAlchemyCollectionRegistry — SQLite (unit) and PostgreSQL (integration)."""

import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool, StaticPool

from memmachine_server.common.data_types import ConcurrencyScope
from memmachine_server.common.vector_store import VectorStoreCollectionConfig
from memmachine_server.common.vector_store.collection_registry import (
    CollectionAlreadyRegisteredError,
    CollectionRegistryEntry,
)
from memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry import (
    SQLAlchemyCollectionRegistry,
    SQLAlchemyCollectionRegistryParams,
)

REGISTRY_NAME = "test_registry"
NAMESPACE = "test_namespace"
NAME = "test_name"

ENTRY = CollectionRegistryEntry(
    config=VectorStoreCollectionConfig(vector_dimensions=3),
    native_collection_name="test_namespace__digest",
    partition_key="test_name#generation",
)
OTHER_ENTRY = CollectionRegistryEntry(
    config=VectorStoreCollectionConfig(vector_dimensions=4),
    native_collection_name="test_namespace__otherdigest",
    partition_key="test_name#othergeneration",
)


def _build_registry(engine, name=REGISTRY_NAME):
    return SQLAlchemyCollectionRegistry(
        SQLAlchemyCollectionRegistryParams(engine=engine, name=name)
    )


@pytest_asyncio.fixture
async def registry(sqlalchemy_engine):
    registry = _build_registry(sqlalchemy_engine)
    await registry.startup()
    yield registry
    async with sqlalchemy_engine.begin() as connection:
        await connection.run_sync(registry._table.metadata.drop_all)


@pytest.mark.asyncio
async def test_startup_is_idempotent(sqlalchemy_engine, registry):
    await registry.startup()

    other_instance = _build_registry(sqlalchemy_engine)
    await other_instance.startup()

    await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)
    assert await other_instance.get(namespace=NAMESPACE, name=NAME) == ENTRY


@pytest.mark.asyncio
async def test_register_and_get_round_trip(registry):
    await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)

    stored_entry = await registry.get(namespace=NAMESPACE, name=NAME)
    assert stored_entry == ENTRY
    assert isinstance(stored_entry, CollectionRegistryEntry)


@pytest.mark.asyncio
async def test_get_missing_returns_none(registry):
    assert await registry.get(namespace=NAMESPACE, name="missing") is None


@pytest.mark.asyncio
async def test_duplicate_register_raises(registry):
    await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)

    with pytest.raises(CollectionAlreadyRegisteredError) as exc_info:
        await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)
    assert exc_info.value.namespace == NAMESPACE
    assert exc_info.value.name == NAME


@pytest.mark.asyncio
async def test_duplicate_register_with_different_entry_raises(registry):
    await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)

    with pytest.raises(CollectionAlreadyRegisteredError):
        await registry.register(namespace=NAMESPACE, name=NAME, entry=OTHER_ENTRY)

    assert await registry.get(namespace=NAMESPACE, name=NAME) == ENTRY


@pytest.mark.asyncio
async def test_get_or_register_registers_when_missing(registry):
    stored_entry, registered = await registry.get_or_register(
        namespace=NAMESPACE, name=NAME, entry=ENTRY
    )

    assert registered
    assert stored_entry == ENTRY
    assert await registry.get(namespace=NAMESPACE, name=NAME) == ENTRY


@pytest.mark.asyncio
async def test_get_or_register_returns_stored_entry_unchanged(registry):
    await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)

    stored_entry, registered = await registry.get_or_register(
        namespace=NAMESPACE, name=NAME, entry=OTHER_ENTRY
    )

    assert not registered
    assert stored_entry == ENTRY
    assert await registry.get(namespace=NAMESPACE, name=NAME) == ENTRY


@pytest.mark.asyncio
async def test_deregister_is_idempotent(registry):
    await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)

    await registry.deregister(namespace=NAMESPACE, name=NAME)
    assert await registry.get(namespace=NAMESPACE, name=NAME) is None

    await registry.deregister(namespace=NAMESPACE, name=NAME)

    await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)
    assert await registry.get(namespace=NAMESPACE, name=NAME) == ENTRY


@pytest.mark.asyncio
async def test_registries_are_isolated(sqlalchemy_engine, registry):
    other_registry = _build_registry(sqlalchemy_engine, name="test_other")
    await other_registry.startup()

    await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)
    await other_registry.register(namespace=NAMESPACE, name=NAME, entry=OTHER_ENTRY)

    assert await registry.get(namespace=NAMESPACE, name=NAME) == ENTRY
    assert await other_registry.get(namespace=NAMESPACE, name=NAME) == OTHER_ENTRY

    await other_registry.deregister(namespace=NAMESPACE, name=NAME)
    assert await registry.get(namespace=NAMESPACE, name=NAME) == ENTRY
    assert await other_registry.get(namespace=NAMESPACE, name=NAME) is None

    async with sqlalchemy_engine.begin() as connection:
        await connection.run_sync(other_registry._table.metadata.drop_all)


@pytest.mark.asyncio
async def test_registry_table_name_is_derived_from_name(sqlalchemy_engine, registry):
    await registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY)

    async with sqlalchemy_engine.connect() as connection:
        table_names = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).get_table_names()
        )
    assert f"collection_registry_{REGISTRY_NAME}" in table_names


@pytest.mark.asyncio
async def test_collection_keys_are_unambiguous(registry):
    # ("a__b", "c") and ("a", "b__c") must be distinct collections,
    # so the storage key separator cannot be an identifier character.
    await registry.register(namespace="a__b", name="c", entry=ENTRY)
    await registry.register(namespace="a", name="b__c", entry=OTHER_ENTRY)

    assert await registry.get(namespace="a__b", name="c") == ENTRY
    assert await registry.get(namespace="a", name="b__c") == OTHER_ENTRY

    await registry.deregister(namespace="a__b", name="c")
    assert await registry.get(namespace="a", name="b__c") == OTHER_ENTRY


class TestValidation:
    @pytest.mark.asyncio
    async def test_invalid_namespace_raises(self, registry):
        with pytest.raises(ValueError, match="Namespace"):
            await registry.get(namespace="Bad-Namespace", name=NAME)

    @pytest.mark.asyncio
    async def test_invalid_name_raises(self, registry):
        with pytest.raises(ValueError, match="Name"):
            await registry.register(namespace=NAMESPACE, name="", entry=ENTRY)

    @pytest.mark.parametrize(
        "name",
        ["", "Uppercase", "hyphen-ated", "dotted.name", "x" * 33],
    )
    def test_invalid_registry_name_raises(self, sqlalchemy_engine, name):
        with pytest.raises(ValueError, match="Registry name"):
            _build_registry(sqlalchemy_engine, name=name)

    def test_static_pool_engine_raises(self):
        engine = create_async_engine(
            "sqlite+aiosqlite://",
            poolclass=StaticPool,
        )
        with pytest.raises(Exception, match="StaticPool"):
            _build_registry(engine)

    def test_ephemeral_sqlite_engine_raises(self):
        # NullPool to sidestep the StaticPool default for async SQLite memory DBs,
        # exercising the ephemeral check itself.
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=NullPool)
        with pytest.raises(ValueError, match="ephemeral"):
            _build_registry(engine)


class TestConcurrency:
    """Concurrency tests using a second registry instance over the same engine."""

    @pytest.mark.asyncio
    async def test_concurrent_register_same_collection(
        self, sqlalchemy_engine, registry
    ):
        other_instance = _build_registry(sqlalchemy_engine)

        results = await asyncio.gather(
            registry.register(namespace=NAMESPACE, name=NAME, entry=ENTRY),
            other_instance.register(namespace=NAMESPACE, name=NAME, entry=OTHER_ENTRY),
            return_exceptions=True,
        )

        errors = [result for result in results if isinstance(result, Exception)]
        assert len(errors) == 1
        assert isinstance(errors[0], CollectionAlreadyRegisteredError)
        assert await registry.get(namespace=NAMESPACE, name=NAME) is not None

    @pytest.mark.asyncio
    async def test_concurrent_get_or_register_mismatched_entries(
        self, sqlalchemy_engine, registry
    ):
        other_instance = _build_registry(sqlalchemy_engine)

        results = await asyncio.gather(
            registry.get_or_register(namespace=NAMESPACE, name=NAME, entry=ENTRY),
            other_instance.get_or_register(
                namespace=NAMESPACE, name=NAME, entry=OTHER_ENTRY
            ),
        )

        stored_entries = [stored_entry for stored_entry, _ in results]
        assert stored_entries[0] == stored_entries[1]
        assert stored_entries[0] in (ENTRY, OTHER_ENTRY)

        registered_flags = [registered for _, registered in results]
        assert sorted(registered_flags) == [False, True]

        assert await registry.get(namespace=NAMESPACE, name=NAME) == stored_entries[0]

    @pytest.mark.asyncio
    async def test_concurrent_register_distinct_collections(
        self, sqlalchemy_engine, registry
    ):
        other_instance = _build_registry(sqlalchemy_engine)

        await asyncio.gather(
            registry.register(namespace=NAMESPACE, name="test_name_a", entry=ENTRY),
            other_instance.register(
                namespace=NAMESPACE, name="test_name_b", entry=ENTRY
            ),
        )

        assert await registry.get(namespace=NAMESPACE, name="test_name_a") == ENTRY
        assert await registry.get(namespace=NAMESPACE, name="test_name_b") == ENTRY


def test_version_1_rows_stay_readable():
    """
    Guards the add-optional-only evolution rule: an entry row exactly as
    version 1 code wrote it must always validate under the current model.
    """
    version_1_row = {
        "config": {
            "vector_dimensions": 3,
            "similarity_metric": "cosine",
            "indexed_properties_schema": {"name": "str"},
        },
        "native_collection_name": "test_namespace__digest",
        "partition_key": "test_name#generation",
    }
    entry = CollectionRegistryEntry.model_validate(version_1_row)
    assert entry.config.vector_dimensions == 3
    assert entry.native_collection_name == "test_namespace__digest"
    assert entry.partition_key == "test_name#generation"


@pytest.mark.asyncio
async def test_concurrency_scope_matches_dialect(sqlalchemy_engine, registry):
    if sqlalchemy_engine.dialect.name == "postgresql":
        assert registry.concurrency_scope == ConcurrencyScope.CLUSTER
    else:
        assert registry.concurrency_scope == ConcurrencyScope.MACHINE
