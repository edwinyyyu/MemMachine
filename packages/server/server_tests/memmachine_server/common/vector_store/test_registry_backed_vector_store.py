"""Tests of RegistryBackedVectorStore's creation flow, on a store whose storage preparation the test controls."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import override
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.vector_store import (
    QueryResult,
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    registry_backed_vector_store,
)
from memmachine_server.common.vector_store.collection_registry import (
    RegisteredCollection,
)
from memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry import (
    SQLAlchemyVectorStoreCollectionRegistry,
    SQLAlchemyVectorStoreCollectionRegistryParams,
)
from memmachine_server.common.vector_store.registry_backed_vector_store import (
    RegistryBackedVectorStore,
    RegistryBackedVectorStoreCollection,
    RegistryBackedVectorStoreParams,
)

NAMESPACE = "ns"
NAME = "c"
CONFIG = VectorStoreCollectionConfig(vector_dimensions=3)

type Preparation = Callable[[str, VectorStoreCollectionConfig, UUID], Awaitable[None]]


class _Collection(RegistryBackedVectorStoreCollection):
    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        raise NotImplementedError

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        raise NotImplementedError

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        raise NotImplementedError


class _Store(RegistryBackedVectorStore[_Collection]):
    """A store whose storage preparation is the test's `prepare`."""

    def __init__(self, registry: SQLAlchemyVectorStoreCollectionRegistry) -> None:
        super().__init__(
            RegistryBackedVectorStoreParams(collection_registry=registry),
            metrics_prefix="test",
        )
        self.prepare: Preparation = _prepared
        self.purged: list[UUID] = []

    @override
    async def _prepare_storage(
        self, namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> None:
        await self.prepare(namespace, config, incarnation)

    @override
    def _build_collection_handle(
        self, namespace: str, name: str, registered: RegisteredCollection
    ) -> _Collection:
        return _Collection(
            namespace=namespace,
            name=name,
            incarnation=registered.incarnation,
            config=registered.config,
            tracker=self._tracker,
            get_registered_collection=self._collection_registry.get,
        )

    @override
    async def _purge_round(
        self, namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> bool:
        self.purged.append(incarnation)
        return False


async def _prepared(
    namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
) -> None:
    return None


@pytest_asyncio.fixture
async def store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    registry = SQLAlchemyVectorStoreCollectionRegistry(
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=engine, vector_store_name="store", tombstone_retention_seconds=0
        )
    )
    await registry.startup()
    yield _Store(registry)
    await engine.dispose()


async def _purged(store: _Store) -> list[UUID]:
    """Incarnations the purge rounds claim, until no tombstone is left."""
    while True:
        claimed = len(store.purged)
        await store.purge_deleted_collections()
        if len(store.purged) == claimed:
            return store.purged


@pytest.mark.asyncio
async def test_a_collection_is_invisible_while_its_storage_is_prepared(store):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(namespace, config, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(
        store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None
    with pytest.raises(VectorStoreCollectionAlreadyExistsError):
        await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)

    release.set()
    await creating
    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_open_or_create_waits_for_a_pending_collection(store, monkeypatch):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0.05
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(namespace, config, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(
        store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()
    store.prepare = _prepared
    registry = store._collection_registry
    register_collection = registry.register
    registrations = 0

    async def counted(namespace, name, config):
        nonlocal registrations
        registrations += 1
        return await register_collection(namespace, name, config)

    monkeypatch.setattr(registry, "register", counted)
    opening = asyncio.create_task(
        store.open_or_create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await asyncio.sleep(0.1)
    assert not opening.done()
    # It waits on the pending collection instead of trying to register.
    assert registrations == 0

    release.set()
    await creating
    opened = await opening

    created = await store._collection_registry.get(NAMESPACE, NAME)
    assert created is not None
    assert opened._incarnation == created.incarnation


@pytest.mark.asyncio
async def test_open_or_create_refuses_a_pending_collection_of_another_configuration(
    store,
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(namespace, config, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(
        store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()

    with pytest.raises(VectorStoreCollectionConfigMismatchError):
        await store.open_or_create_collection(
            namespace=NAMESPACE,
            name=NAME,
            config=VectorStoreCollectionConfig(vector_dimensions=4),
        )

    release.set()
    await creating


@pytest.mark.asyncio
@pytest.mark.parametrize("create", ["create_collection", "open_or_create_collection"])
async def test_a_failed_preparation_frees_the_name_and_queues_its_incarnation(
    store, create
):
    incarnations: list[UUID] = []

    async def refused(namespace, config, incarnation) -> None:
        incarnations.append(incarnation)
        raise RuntimeError("the backend refused")

    store.prepare = refused
    with pytest.raises(RuntimeError, match="refused"):
        await getattr(store, create)(namespace=NAMESPACE, name=NAME, config=CONFIG)

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None
    assert await _purged(store) == incarnations
    store.prepare = _prepared
    await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_a_failed_preparation_the_registry_cannot_undo_stays_pending_until_deleted(
    store, monkeypatch
):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
    )
    registry = store._collection_registry

    async def refused(namespace, config, incarnation) -> None:
        raise RuntimeError("the backend refused")

    async def unreachable(namespace, name, *, incarnation=None) -> None:
        raise ConnectionError("the registry is unreachable")

    store.prepare = refused
    unregister_collection = registry.unregister
    monkeypatch.setattr(registry, "unregister", unreachable)
    with pytest.raises(RuntimeError, match="refused"):
        await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    monkeypatch.setattr(registry, "unregister", unregister_collection)
    store.prepare = _prepared

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None
    with pytest.raises(VectorStoreCollectionAlreadyExistsError):
        await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    with pytest.raises(VectorStoreAttemptsExhaustedError, match="pending"):
        await store.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )

    await store.delete_collection(namespace=NAMESPACE, name=NAME)
    await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_a_collection_deleted_while_its_storage_is_prepared_is_not_marked_live(
    store,
):
    """The creation returns, as one the deletion followed, and the collection
    stays deleted."""
    incarnations: list[UUID] = []

    async def deleted_meanwhile(namespace, config, incarnation) -> None:
        incarnations.append(incarnation)
        await store.delete_collection(namespace=NAMESPACE, name=NAME)

    store.prepare = deleted_meanwhile
    await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None
    assert await _purged(store) == incarnations


@pytest.mark.asyncio
async def test_open_or_create_creates_again_after_a_deletion_during_preparation(
    store, monkeypatch
):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
    )
    incarnations: list[UUID] = []

    async def deleted_the_first_time(namespace, config, incarnation) -> None:
        incarnations.append(incarnation)
        if len(incarnations) == 1:
            await store.delete_collection(namespace=NAMESPACE, name=NAME)

    store.prepare = deleted_the_first_time
    opened = await store.open_or_create_collection(
        namespace=NAMESPACE, name=NAME, config=CONFIG
    )

    assert len(incarnations) == 2
    assert opened._incarnation == incarnations[1]
    assert await _purged(store) == incarnations[:1]
