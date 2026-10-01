"""Tests of RegistryBackedVectorStore's creation flow, on a store whose storage preparation the test controls."""

import asyncio
from collections.abc import Awaitable, Callable
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
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionPendingError,
    registry_backed_vector_store,
)
from memmachine_server.common.vector_store.partition_registry import (
    Registration,
    sqlalchemy_partition_registry,
)
from memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyVectorStorePartitionRegistry,
    SQLAlchemyVectorStorePartitionRegistryParams,
)
from memmachine_server.common.vector_store.registry_backed_vector_store import (
    RegistryBackedVectorStore,
    RegistryBackedVectorStoreParams,
    RegistryBackedVectorStorePartition,
)

NAMESPACE = "ns"
NAME = "c"
CONFIG = VectorStoreCollectionConfig(vector_dimensions=3)

type Preparation = Callable[[str, VectorStoreCollectionConfig, UUID], Awaitable[None]]


class _Partition(RegistryBackedVectorStorePartition):
    @override
    async def _upsert(self, records: list[Record]) -> None:
        raise NotImplementedError

    @override
    async def _query(
        self,
        query_vectors: list[list[float]],
        *,
        limit: int,
        score_threshold: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryResult]:
        raise NotImplementedError

    @override
    async def _delete(self, record_uuids: list[UUID]) -> None:
        raise NotImplementedError


class _Store(RegistryBackedVectorStore[_Partition]):
    """A store whose storage preparation is the test's `prepare`."""

    def __init__(self, registry: SQLAlchemyVectorStorePartitionRegistry) -> None:
        super().__init__(
            RegistryBackedVectorStoreParams(partition_registry=registry),
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
    def _build_partition_handle(self, registration: Registration) -> _Partition:
        return _Partition(registration=registration, tracker=self._tracker)

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
    registry = SQLAlchemyVectorStorePartitionRegistry(
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=engine, vector_store_name="store", tombstone_retention_seconds=0
        )
    )
    await registry.startup()
    yield _Store(registry)
    await engine.dispose()


async def _purged(store: _Store) -> list[UUID]:
    """Incarnations the purge rounds claim, calling the purge until it returns False."""
    while await store.purge_deleted_partitions():
        pass
    return store.purged


@pytest.mark.asyncio
async def test_a_collection_is_pending_while_its_storage_is_prepared(store):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(namespace, config, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(
        store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()

    with pytest.raises(VectorStorePartitionPendingError) as pending:
        await store.get_partition(namespace=NAMESPACE, name=NAME)
    with pytest.raises(VectorStorePartitionPendingError) as registered:
        await store._partition_registry.resolve(NAMESPACE, NAME)
    assert pending.value.registered_at == registered.value.registered_at
    assert pending.value.config == CONFIG
    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)

    release.set()
    await creating
    assert await store.get_partition(namespace=NAMESPACE, name=NAME) is not None


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
        store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()
    store.prepare = _prepared
    registry = store._partition_registry
    reserve = registry.reserve
    reservations = 0

    async def counted(namespace, name, config):
        nonlocal reservations
        reservations += 1
        return await reserve(namespace, name, config)

    monkeypatch.setattr(registry, "reserve", counted)
    opening = asyncio.create_task(
        store.open_or_create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await asyncio.sleep(0.1)
    assert not opening.done()
    # It waits on the pending collection instead of trying to reserve it.
    assert reservations == 0

    release.set()
    await creating
    opened = await opening

    created = await store._partition_registry.resolve(NAMESPACE, NAME)
    assert created is not None
    assert opened._incarnation == created.incarnation


@pytest.mark.asyncio
async def test_open_or_create_gives_up_from_the_race_it_last_lost(store, monkeypatch):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
    )
    lost = VectorStorePartitionAlreadyExistsError(NAMESPACE, NAME)

    async def taken(namespace, name, config):
        raise lost

    monkeypatch.setattr(store._partition_registry, "reserve", taken)

    with pytest.raises(
        VectorStoreAttemptsExhaustedError, match="no progress"
    ) as gave_up:
        await store.open_or_create_partition(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
    assert gave_up.value.__cause__ is lost


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
        store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()

    with pytest.raises(VectorStoreCollectionConfigMismatchError):
        await store.open_or_create_partition(
            namespace=NAMESPACE,
            name=NAME,
            config=VectorStoreCollectionConfig(vector_dimensions=4),
        )

    release.set()
    await creating


@pytest.mark.asyncio
@pytest.mark.parametrize("create", ["create_partition", "open_or_create_partition"])
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

    assert await store.get_partition(namespace=NAMESPACE, name=NAME) is None
    assert await _purged(store) == incarnations
    store.prepare = _prepared
    await store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    assert await store.get_partition(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_a_cancelled_preparation_frees_the_name_and_queues_its_incarnation(
    store,
):
    started = asyncio.Event()
    incarnations: list[UUID] = []

    async def hangs(namespace, config, incarnation) -> None:
        incarnations.append(incarnation)
        started.set()
        await asyncio.Event().wait()

    store.prepare = hangs
    creating = asyncio.create_task(
        store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    assert await store.get_partition(namespace=NAMESPACE, name=NAME) is None
    assert await _purged(store) == incarnations


@pytest.mark.asyncio
async def test_cancelling_the_reservation_after_a_cancelled_preparation_survives_another_cancellation(
    store, monkeypatch
):
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    reservation_type = sqlalchemy_partition_registry._SQLAlchemyReservation
    cancel = reservation_type.cancel

    async def hangs(namespace, config, incarnation) -> None:
        started.set()
        await asyncio.Event().wait()

    async def slow(reservation) -> None:
        cancelling.set()
        await release.wait()
        await cancel(reservation)

    store.prepare = hangs
    monkeypatch.setattr(reservation_type, "cancel", slow)
    creating = asyncio.create_task(
        store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(cancelling.wait(), 5)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    release.set()
    await asyncio.wait_for(asyncio.gather(*store._cancellations), 5)
    assert await store.get_partition(namespace=NAMESPACE, name=NAME) is None


@pytest.mark.asyncio
async def test_a_failed_preparation_the_registry_cannot_undo_stays_pending_until_deleted(
    store, monkeypatch
):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
    )

    async def refused(namespace, config, incarnation) -> None:
        raise RuntimeError("the backend refused")

    async def unreachable(reservation) -> None:
        raise ConnectionError("the registry is unreachable")

    store.prepare = refused
    with monkeypatch.context() as cancellation:
        cancellation.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "cancel",
            unreachable,
        )
        with pytest.raises(RuntimeError, match="refused"):
            await store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    store.prepare = _prepared

    with pytest.raises(VectorStorePartitionPendingError):
        await store.get_partition(namespace=NAMESPACE, name=NAME)
    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    with pytest.raises(VectorStorePartitionPendingError):
        await store.open_or_create_partition(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )

    await store.delete_partition(namespace=NAMESPACE, name=NAME)
    await store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    assert await store.get_partition(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_a_collection_deleted_while_its_storage_is_prepared_is_not_confirmed(
    store,
):
    """The creation raises, and the collection stays deleted."""
    incarnations: list[UUID] = []

    async def deleted_meanwhile(namespace, config, incarnation) -> None:
        incarnations.append(incarnation)
        await store.delete_partition(namespace=NAMESPACE, name=NAME)

    store.prepare = deleted_meanwhile
    with pytest.raises(VectorStorePartitionDeletedError):
        await store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)

    assert await store.get_partition(namespace=NAMESPACE, name=NAME) is None
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
            await store.delete_partition(namespace=NAMESPACE, name=NAME)

    store.prepare = deleted_the_first_time
    opened = await store.open_or_create_partition(
        namespace=NAMESPACE, name=NAME, config=CONFIG
    )

    assert len(incarnations) == 2
    assert opened._incarnation == incarnations[1]
    assert await _purged(store) == incarnations[:1]


@pytest.mark.asyncio
async def test_calling_the_purge_until_it_returns_false_drains_every_tombstone(
    store,
):
    """A round that finds nothing still ran, so the calls go on to the next
    due tombstone."""
    incarnations = []
    for name in ("a", "b"):
        await store.create_partition(namespace=NAMESPACE, name=name, config=CONFIG)
        live = await store._partition_registry.resolve(NAMESPACE, name)
        assert live is not None
        incarnations.append(live.incarnation)
        await store.delete_partition(namespace=NAMESPACE, name=name)

    assert sorted(await _purged(store)) == sorted(incarnations)


@pytest.mark.asyncio
@pytest.mark.parametrize("query_vectors", [[], [[1.0, 0.0, 0.0]]])
@pytest.mark.parametrize("limit", [0, -1])
async def test_a_query_limit_that_is_not_positive_is_refused(
    store, query_vectors, limit
):
    await store.create_partition(namespace=NAMESPACE, name=NAME, config=CONFIG)
    collection = await store.get_partition(namespace=NAMESPACE, name=NAME)
    assert collection is not None

    with pytest.raises(ValueError, match="not positive"):
        await collection.query(query_vectors=query_vectors, limit=limit)
