"""The SQLAlchemy collection registry: creation arbitrated by the database, deletion queued, purge claimed."""

import asyncio
import logging
from dataclasses import replace
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import (
    ColumnElement,
    delete,
    event,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry import (
    _MAX_FAILED_PURGE_ROUNDS,
    CollectionRow,
    PurgeQueueRow,
    SQLAlchemyVectorStoreCollectionRegistry,
    SQLAlchemyVectorStoreCollectionRegistryParams,
)
from memmachine_server.common.vector_store.data_types import (
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
)

CONFIG = VectorStoreCollectionConfig(
    vector_dimensions=3, indexed_properties_schema={"name": str}
)
OTHER_CONFIG = VectorStoreCollectionConfig(vector_dimensions=4)
RETENTION = timedelta(days=1)
RETENTION_SECONDS = int(RETENTION.total_seconds())
NAMESPACE = "ns"


@pytest.fixture
def vector_store_name() -> str:
    """A vector store name no other test (or earlier run on a shared server) used."""
    return f"test_{uuid4().hex[:12]}"


async def _registry(
    engine: AsyncEngine,
    vector_store_name: str,
    tombstone_retention_seconds: int = RETENTION_SECONDS,
) -> SQLAlchemyVectorStoreCollectionRegistry:
    registry = SQLAlchemyVectorStoreCollectionRegistry(
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=engine,
            vector_store_name=vector_store_name,
            tombstone_retention_seconds=tombstone_retention_seconds,
        )
    )
    await registry.startup()
    return registry


async def _queued(registry: SQLAlchemyVectorStoreCollectionRegistry) -> list[UUID]:
    """The registry's tombstones, oldest first."""
    async with registry._engine.connect() as connection:
        rows = await connection.execute(
            select(PurgeQueueRow.incarnation)
            .where(PurgeQueueRow.vector_store_name == registry._vector_store_name)
            .order_by(PurgeQueueRow.enqueued_at)
        )
        return list(rows.scalars())


def _database_time_ago(engine: AsyncEngine, delta: timedelta) -> ColumnElement:
    """The database clock's now less `delta`, written by the database in the form of its own stamps."""
    if engine.dialect.name == "sqlite":
        return func.datetime("now", f"-{int(delta.total_seconds())} seconds")
    return func.now() - delta


async def _age_deletion(
    registry: SQLAlchemyVectorStoreCollectionRegistry,
    incarnation: UUID,
    extra: timedelta = timedelta(0),
) -> None:
    """Move the tombstone's deletion back past the retention, by `extra` more, on the database clock."""
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == incarnation)
            .values(
                enqueued_at=_database_time_ago(
                    registry._engine, RETENTION + timedelta(seconds=1) + extra
                )
            )
        )


async def _blocked_or_done(engine: AsyncEngine, task: asyncio.Task) -> str:
    """Wait until `task` finishes ("done") or another backend waits on a lock ("blocked").

    Decided by the database's own state (pg_stat_activity); raises TimeoutError
    if neither happens within 30 seconds.
    """
    deadline = asyncio.get_running_loop().time() + 30
    while not task.done():
        async with engine.connect() as connection:
            blocked = (
                await connection.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE wait_event_type = 'Lock' "
                        "AND datname = current_database() "
                        "AND pid != pg_backend_pid()"
                    )
                )
            ).scalar_one()
        if blocked:
            return "blocked"
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("the task neither blocked on a lock nor finished")
        await asyncio.sleep(0.01)
    return "done"


async def _failed_rounds(
    registry: SQLAlchemyVectorStoreCollectionRegistry, incarnation: UUID
) -> int:
    """The tombstone's count of consecutive failed purge rounds."""
    async with registry._engine.connect() as connection:
        return (
            await connection.execute(
                select(PurgeQueueRow.failed_rounds).where(
                    PurgeQueueRow.incarnation == incarnation
                )
            )
        ).scalar_one()


async def _age_last_failure(
    registry: SQLAlchemyVectorStoreCollectionRegistry,
    incarnation: UUID,
    ago: timedelta,
) -> None:
    """Move the tombstone's last failed round back to `ago` before now, on the database clock."""
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == incarnation)
            .values(last_failed_at=_database_time_ago(registry._engine, ago))
        )


async def _failing_round(registry: SQLAlchemyVectorStoreCollectionRegistry) -> UUID:
    """One purge round whose body raises; the incarnation it claimed."""
    claimed: list[UUID] = []

    async def refused_reclamation() -> None:
        async with registry.claim_purgeable_incarnation() as claim:
            assert claim is not None
            claimed.append(claim.incarnation)
            raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError, match="the backend refused"):
        await refused_reclamation()
    return claimed[0]


async def _round(
    registry: SQLAlchemyVectorStoreCollectionRegistry, any_records_found: bool
) -> UUID | None:
    """One purge round on the oldest due tombstone, reporting `any_records_found`; its incarnation, or None."""
    async with registry.claim_purgeable_incarnation() as claim:
        if claim is None:
            return None
        claim.any_records_found = any_records_found
        return claim.incarnation


@pytest.mark.asyncio
async def test_create_registers_a_fresh_incarnation_with_the_config(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)

    incarnation = await registry.register(NAMESPACE, "c", CONFIG)
    assert await registry.mark_live(incarnation)

    registered = await registry.get(NAMESPACE, "c")
    assert registered is not None
    assert registered.incarnation == incarnation
    assert registered.config == CONFIG
    assert await registry.get(NAMESPACE, "d") is None
    assert await registry.get("other_ns", "c") is None


@pytest.mark.asyncio
async def test_a_collection_is_pending_until_marked_live(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)

    incarnation = await registry.register(NAMESPACE, "c", CONFIG)

    pending = await registry.get(NAMESPACE, "c")
    assert pending is not None
    assert (pending.incarnation, pending.config, pending.live) == (
        incarnation,
        CONFIG,
        False,
    )
    assert pending.registered_at.tzinfo is not None
    assert await registry.mark_live(incarnation)
    assert await registry.get(NAMESPACE, "c") == replace(pending, live=True)
    assert not await registry.mark_live(incarnation)


@pytest.mark.asyncio
async def test_only_the_registered_incarnation_is_marked_live(
    sqlalchemy_engine, vector_store_name
):
    """A creation whose collection was deleted while its storage was
    prepared cannot mark live a collection registered under the name since."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    deleted = await registry.register(NAMESPACE, "c", CONFIG)
    assert not await registry.mark_live(uuid4())
    await registry.unregister(NAMESPACE, "c")
    assert not await registry.mark_live(deleted)

    registered_since = await registry.register(NAMESPACE, "c", CONFIG)

    assert not await registry.mark_live(deleted)
    registered = await registry.get(NAMESPACE, "c")
    assert registered is not None
    assert (registered.incarnation, registered.live) == (registered_since, False)
    assert await registry.mark_live(registered_since)


@pytest.mark.asyncio
async def test_unregistering_an_incarnation_spares_the_name_registered_since(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    old = await registry.register(NAMESPACE, "c", CONFIG)
    await registry.unregister(NAMESPACE, "c")
    since = await registry.register(NAMESPACE, "c", CONFIG)

    await registry.unregister_incarnation(old)
    await registry.unregister_incarnation(uuid4())
    registered = await registry.get(NAMESPACE, "c")
    assert registered is not None
    assert registered.incarnation == since
    assert await _queued(registry) == [old]

    await registry.unregister_incarnation(since)
    await registry.unregister_incarnation(since)
    assert await registry.get(NAMESPACE, "c") is None
    assert sorted(await _queued(registry)) == sorted([old, since])


@pytest.mark.asyncio
async def test_a_taken_name_is_already_exists_whatever_the_config(
    sqlalchemy_engine, vector_store_name
):
    """Taken by a pending collection, then by a live one."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "c", CONFIG)

    for _ in range(2):
        with pytest.raises(VectorStoreCollectionAlreadyExistsError):
            await registry.register(NAMESPACE, "c", CONFIG)
        with pytest.raises(VectorStoreCollectionAlreadyExistsError):
            await registry.register(NAMESPACE, "c", OTHER_CONFIG)
        await registry.mark_live(incarnation)


@pytest.mark.asyncio
async def test_any_string_of_at_most_255_characters_names_a_registry(
    sqlalchemy_engine, vector_store_name
):
    name = f"Store-{vector_store_name} ".ljust(255, "x")
    registry = await _registry(sqlalchemy_engine, name)
    incarnation = await registry.register(NAMESPACE, "c", CONFIG)

    registered = await registry.get(NAMESPACE, "c")
    assert registered is not None
    assert registered.incarnation == incarnation


def test_an_engine_of_another_dialect_is_refused(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://")
    monkeypatch.setattr(engine.dialect, "name", "mssql")
    with pytest.raises(ValueError, match="mssql"):
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=engine,
            vector_store_name="store",
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


def test_a_sqlite_runtime_without_returning_is_refused(monkeypatch):
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.sqlite3.sqlite_version_info",
        (3, 34, 1),
    )
    with pytest.raises(ValueError, match="RETURNING"):
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=create_async_engine("sqlite+aiosqlite://"),
            vector_store_name="store",
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


@pytest.mark.asyncio
async def test_registries_of_two_vector_stores_share_a_database_and_nothing_else(
    sqlalchemy_engine, vector_store_name
):
    """Each registry has its own collections, and a purge round claims
    only its own tombstones."""
    first = await _registry(sqlalchemy_engine, f"{vector_store_name}_a")
    second = await _registry(sqlalchemy_engine, f"{vector_store_name}_b")

    one = await first.register(NAMESPACE, "c", CONFIG)
    two = await second.register(NAMESPACE, "c", OTHER_CONFIG)
    assert one != two
    assert await first.mark_live(one)
    assert await second.mark_live(two)
    in_first = await first.get(NAMESPACE, "c")
    in_second = await second.get(NAMESPACE, "c")
    assert in_first is not None
    assert in_second is not None
    assert in_first.incarnation == one
    assert in_second.incarnation == two

    await first.unregister(NAMESPACE, "c")
    await _age_deletion(first, one)
    still_in_second = await second.get(NAMESPACE, "c")
    assert still_in_second is not None
    assert still_in_second.incarnation == two
    assert await _round(second, any_records_found=False) is None
    assert await _round(first, any_records_found=False) == one


@pytest.mark.asyncio
async def test_concurrent_creators_get_one_winner(sqlalchemy_engine, vector_store_name):
    """Two registries on one database, the two-process shape in one process."""
    first = await _registry(sqlalchemy_engine, vector_store_name)
    second = await _registry(sqlalchemy_engine, vector_store_name)

    results = await asyncio.gather(
        *(
            registry.register(NAMESPACE, "c", CONFIG)
            for registry in (first, second) * 3
        ),
        return_exceptions=True,
    )

    winners = [r for r in results if isinstance(r, UUID)]
    losers = [
        r for r in results if isinstance(r, VectorStoreCollectionAlreadyExistsError)
    ]
    assert len(winners) == 1
    assert len(losers) == 5
    registered = await first.get(NAMESPACE, "c")
    assert registered is not None
    assert registered.incarnation == winners[0]


@pytest.mark.asyncio
async def test_delete_queues_the_incarnation_and_is_idempotent(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "c", CONFIG)

    await registry.unregister(NAMESPACE, "c")

    assert await registry.get(NAMESPACE, "c") is None
    assert await _queued(registry) == [incarnation]
    await registry.unregister(NAMESPACE, "c")
    await registry.unregister(NAMESPACE, "never")
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_recreated_name_gets_a_new_incarnation(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    old = await registry.register(NAMESPACE, "c", CONFIG)
    await registry.unregister(NAMESPACE, "c")

    new = await registry.register(NAMESPACE, "c", CONFIG)
    await registry.mark_live(new)

    assert new != old
    registered = await registry.get(NAMESPACE, "c")
    assert registered is not None
    assert registered.incarnation == new


@pytest.mark.asyncio
async def test_a_claim_names_where_the_records_are(
    sqlalchemy_engine, vector_store_name
):
    """The claim carries what a purge round needs to find the records: incarnation, namespace, configuration."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "c", OTHER_CONFIG)
    await registry.unregister(NAMESPACE, "c")
    await _age_deletion(registry, incarnation)

    async with registry.claim_purgeable_incarnation() as claim:
        assert claim is not None
        assert claim.incarnation == incarnation
        assert claim.namespace == NAMESPACE
        assert claim.config == OTHER_CONFIG
        claim.any_records_found = False


@pytest.mark.asyncio
async def test_a_tombstone_is_not_due_before_the_retention(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")

    assert await _round(registry, any_records_found=False) is None
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_due_tombstones_go_oldest_deletion_first_until_a_round_finds_nothing(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    minted = iter([UUID(int=1), UUID(int=2)])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )
    first = await registry.register(NAMESPACE, "a", CONFIG)
    second = await registry.register(NAMESPACE, "b", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    await registry.unregister(NAMESPACE, "b")
    # The later deletion, with the larger incarnation, is made the older, so
    # only the ordering by deletion can put it first.
    await _age_deletion(registry, first)
    await _age_deletion(registry, second, extra=timedelta(minutes=1))

    assert await _round(registry, any_records_found=True) == second
    # Found records: still due, and still the oldest deletion.
    assert await _round(registry, any_records_found=False) == second
    assert await _queued(registry) == [first]
    assert await _round(registry, any_records_found=False) == first
    assert await _queued(registry) == []
    assert await _round(registry, any_records_found=False) is None


@pytest.mark.asyncio
async def test_a_tombstone_queued_by_a_real_deletion_comes_due_by_the_database_clock(
    sqlalchemy_engine, vector_store_name
):
    """The claim compares the stamp the database clock wrote, by the
    database's arithmetic: with no retention the first round claims the
    tombstone, and no stamp is rewritten by the test."""
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
    )
    incarnation = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")

    assert await _round(registry, any_records_found=False) == incarnation

    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_changed_retention_applies_to_tombstones_already_queued(
    sqlalchemy_engine, vector_store_name
):
    """Only the deletion's time is stored; the retention in force when a
    claim is decided applies, including to tombstones queued under another."""
    queued_under_a_day = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await queued_under_a_day.register(NAMESPACE, "a", CONFIG)
    await queued_under_a_day.unregister(NAMESPACE, "a")
    assert await _round(queued_under_a_day, any_records_found=False) is None

    no_retention = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
    )
    assert await _round(no_retention, any_records_found=False) == incarnation
    assert await _queued(no_retention) == []


@pytest.mark.asyncio
async def test_a_round_whose_body_raises_keeps_the_tombstone_and_counts_the_failure(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)

    async def refused_reclamation() -> None:
        async with registry.claim_purgeable_incarnation() as claim:
            assert claim is not None
            assert claim.incarnation == incarnation
            claim.any_records_found = False
            raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError):
        await refused_reclamation()

    assert await _queued(registry) == [incarnation]
    assert await _failed_rounds(registry, incarnation) == 1
    # Backing off: claimed again once 30 seconds have passed since the failure.
    assert await _round(registry, any_records_found=False) is None
    await _age_last_failure(registry, incarnation, timedelta(seconds=35))
    assert await _round(registry, any_records_found=False) == incarnation


@pytest.mark.asyncio
async def test_a_failed_tombstone_backs_off_while_the_ones_behind_it_are_claimed(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    failing = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, failing, extra=timedelta(hours=1))
    later = await registry.register(NAMESPACE, "b", CONFIG)
    await registry.unregister(NAMESPACE, "b")
    await _age_deletion(registry, later)

    assert await _failing_round(registry) == failing
    assert await _round(registry, any_records_found=False) == later


@pytest.mark.asyncio
async def test_the_backoff_doubles_with_each_failure_and_runs_from_the_last_one(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    # Due for days: the backoff still runs from the last failure, not from
    # when the tombstone came due.
    await _age_deletion(registry, incarnation, extra=timedelta(days=3))

    await _failing_round(registry)
    await _age_last_failure(registry, incarnation, timedelta(seconds=35))
    await _failing_round(registry)
    # Two failures: a minute from the second.
    await _age_last_failure(registry, incarnation, timedelta(seconds=50))
    assert await _round(registry, any_records_found=True) is None
    await _age_last_failure(registry, incarnation, timedelta(seconds=70))
    assert await _round(registry, any_records_found=True) == incarnation


@pytest.mark.asyncio
async def test_the_backoff_stops_doubling_at_its_maximum(
    sqlalchemy_engine, vector_store_name
):
    registry = SQLAlchemyVectorStoreCollectionRegistry(
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=sqlalchemy_engine,
            vector_store_name=vector_store_name,
            tombstone_retention_seconds=RETENTION_SECONDS,
            purge_retry_backoff_seconds=30,
            max_purge_retry_backoff_seconds=120,
        )
    )
    await registry.startup()
    incarnation = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)

    for _ in range(5):
        await _failing_round(registry)
        await _age_last_failure(registry, incarnation, timedelta(days=1))
    # Five failures: 30 seconds doubled four times is 8 minutes, capped at 2.
    await _age_last_failure(registry, incarnation, timedelta(seconds=110))
    assert await _round(registry, any_records_found=True) is None
    await _age_last_failure(registry, incarnation, timedelta(seconds=130))
    assert await _round(registry, any_records_found=True) == incarnation


@pytest.mark.asyncio
async def test_a_tombstone_whose_rounds_keep_failing_is_dead_lettered_and_reported(
    sqlalchemy_engine, vector_store_name, caplog
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    failing = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, failing, extra=timedelta(hours=1))
    later = await registry.register(NAMESPACE, "b", CONFIG)
    await registry.unregister(NAMESPACE, "b")
    await _age_deletion(registry, later)

    with caplog.at_level(logging.ERROR):
        for _ in range(_MAX_FAILED_PURGE_ROUNDS):
            # Past its backoff, so it is the oldest tombstone claimable.
            await _age_last_failure(registry, failing, timedelta(days=1))
            assert await _failing_round(registry) == failing

    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and str(failing) in r.getMessage()
    ]
    # Skipped from now on, but kept in the queue.
    assert await _round(registry, any_records_found=False) == later
    assert await _round(registry, any_records_found=False) is None
    assert await _queued(registry) == [failing]


@pytest.mark.asyncio
async def test_a_round_that_finds_points_clears_the_failed_rounds(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)

    for _ in range(_MAX_FAILED_PURGE_ROUNDS - 1):
        await _failing_round(registry)
        await _age_last_failure(registry, incarnation, timedelta(days=1))
    assert await _round(registry, any_records_found=True) == incarnation
    assert await _failed_rounds(registry, incarnation) == 0


@pytest.mark.asyncio
async def test_a_stored_configuration_that_no_longer_validates_counts_as_a_failed_round(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == incarnation)
            .values(config={"vector_dimensions": "not a number"})
        )

    with pytest.raises(ValueError, match="vector_dimensions"):
        async with registry.claim_purgeable_incarnation():
            pass
    assert await _failed_rounds(registry, incarnation) == 1


@pytest.mark.asyncio
async def test_a_cancelled_round_is_not_a_failed_round(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)

    async def cancelled_round() -> None:
        async with registry.claim_purgeable_incarnation() as claim:
            assert claim is not None
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cancelled_round()
    assert await _failed_rounds(registry, incarnation) == 0


@pytest.mark.asyncio
async def test_a_round_that_reports_nothing_is_an_error(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)

    with pytest.raises(RuntimeError, match="without setting any_records_found"):
        async with registry.claim_purgeable_incarnation():
            pass
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_an_incarnation_awaiting_purge_is_never_reminted(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A minted incarnation colliding with a queued tombstone is rejected and re-minted."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    minted = iter([dead, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )

    fresh = await registry.register(NAMESPACE, "b", CONFIG)

    assert fresh != dead
    assert await _queued(registry) == [dead]


@pytest.mark.asyncio
async def test_creation_that_never_mints_a_free_incarnation_gives_up(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = await registry.register(NAMESPACE, "a", CONFIG)
    await registry.unregister(NAMESPACE, "a")
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: dead,
    )

    with pytest.raises(VectorStoreAttemptsExhaustedError):
        await registry.register(NAMESPACE, "b", CONFIG)
    assert await registry.get(NAMESPACE, "b") is None


@pytest.mark.asyncio
async def test_purge_rounds_keep_time_by_the_database_clock(
    sqlalchemy_engine, vector_store_name
):
    """Every time the queue stores, and the retention's cutoff, come from the
    database's clock, never the client's.

    The database's now() is shadowed by one running ten days behind the
    real clock: the tombstone's deletion is stamped by it, and a retention
    of a day, long past by the real clock, has not passed by it.
    """
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite's clock cannot be shadowed")
    schema = f"skewed_{uuid4().hex[:12]}"
    async with sqlalchemy_engine.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA {schema}"))
        await connection.execute(
            text(
                f"CREATE FUNCTION {schema}.now() RETURNS timestamptz LANGUAGE sql "
                "STABLE AS $$ SELECT pg_catalog.now() - interval '10 days' $$"
            )
        )
    skewed_engine = create_async_engine(
        sqlalchemy_engine.url,
        connect_args={
            "server_settings": {"search_path": f"{schema}, public, pg_catalog"}
        },
    )
    try:
        registry = await _registry(skewed_engine, vector_store_name)
        incarnation = await registry.register(NAMESPACE, "a", CONFIG)
        await registry.unregister(NAMESPACE, "a")

        async with skewed_engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        PurgeQueueRow.enqueued_at, text("pg_catalog.now() AS real_now")
                    ).where(PurgeQueueRow.incarnation == incarnation)
                )
            ).one()
        assert row.real_now - row.enqueued_at > timedelta(days=9)
        assert await _round(registry, any_records_found=False) is None
        assert await _queued(registry) == [incarnation]
    finally:
        await skewed_engine.dispose()
        async with sqlalchemy_engine.begin() as connection:
            await connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))


@pytest.mark.asyncio
async def test_a_claim_skips_a_tombstone_another_purger_holds(
    sqlalchemy_engine, vector_store_name
):
    """A purger never waits on another purger's claim: it takes the next due tombstone."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite holds no row locks to skip")
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    held = await registry.register(NAMESPACE, "held", CONFIG)
    free = await registry.register(NAMESPACE, "free", CONFIG)
    await registry.unregister(NAMESPACE, "held")
    await registry.unregister(NAMESPACE, "free")
    await _age_deletion(registry, held, extra=timedelta(minutes=1))
    await _age_deletion(registry, free)

    claim = None
    try:
        async with sqlalchemy_engine.connect() as other_purger, other_purger.begin():
            await other_purger.execute(
                select(PurgeQueueRow.incarnation)
                .where(PurgeQueueRow.incarnation == held)
                .with_for_update()
            )
            claim = asyncio.create_task(_round(registry, any_records_found=False))
            outcome = await _blocked_or_done(sqlalchemy_engine, claim)
            assert outcome == "done", (
                "the claim waited on a tombstone another purger holds"
            )
            assert await claim == free
    finally:
        if claim is not None and not claim.done():
            await asyncio.wait_for(claim, 30)


@pytest.mark.asyncio
async def test_racing_deletions_of_a_collection_queue_one_tombstone(
    sqlalchemy_engine, vector_store_name
):
    """Racing deletions serialize on the collection's row: the losers delete
    nothing, one tombstone is queued, and the name can be created again."""
    registries = [
        await _registry(sqlalchemy_engine, vector_store_name) for _ in range(4)
    ]
    for cycle in range(10):
        await registries[0].register(NAMESPACE, "c", CONFIG)
        await asyncio.wait_for(
            asyncio.gather(*(r.unregister(NAMESPACE, "c") for r in registries)), 30
        )
        assert await registries[0].get(NAMESPACE, "c") is None
        assert len(await _queued(registries[0])) == cycle + 1
    await registries[0].register(NAMESPACE, "c", CONFIG)


@pytest.mark.asyncio
async def test_a_deletion_racing_another_waits_and_queues_nothing_more(
    sqlalchemy_engine, vector_store_name
):
    """A deletion that finds another process's deletion in flight waits for
    it, then finds nothing to delete: one tombstone, and no error."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite serializes whole write transactions")
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register(NAMESPACE, "c", CONFIG)

    local = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            # Another process's deletion, held uncommitted.
            await remote.execute(
                delete(CollectionRow).where(CollectionRow.incarnation == incarnation)
            )
            await remote.execute(
                insert(PurgeQueueRow).values(
                    incarnation=incarnation,
                    vector_store_name=vector_store_name,
                    namespace=NAMESPACE,
                    name="c",
                    config=CONFIG.model_dump(mode="json"),
                    enqueued_at=func.now(),
                )
            )
            local = asyncio.create_task(registry.unregister(NAMESPACE, "c"))
            assert await _blocked_or_done(sqlalchemy_engine, local) == "blocked"
        await asyncio.wait_for(local, 30)
    finally:
        if local is not None and not local.done():
            local.cancel()

    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_an_incarnation_colliding_with_a_live_collection_is_reminted(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A mint rejected by the incarnation's unique constraint, with the name
    free, is a collision to mint again, not a taken name."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    live = await registry.register(NAMESPACE, "live", CONFIG)
    await registry.mark_live(live)
    minted = iter([live, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )

    fresh = await registry.register(NAMESPACE, "fresh", CONFIG)

    assert fresh != live
    registered = await registry.get(NAMESPACE, "live")
    assert registered is not None
    assert registered.incarnation == live


@pytest.mark.asyncio
async def test_a_mint_checks_the_queue_after_its_insert(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A mint colliding with a deletion in flight sees the deletion's tombstone.

    The deletion frees the incarnation's row and queues its tombstone in one
    uncommitted transaction; the colliding mint's insert waits on it. Only a
    queue check made after the insert sees the tombstone once the deletion
    commits: one made before reads the queue too early and registers a
    collection under an incarnation awaiting purge.
    """
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    victim = await registry.register(NAMESPACE, "victim", CONFIG)
    minted = iter([victim, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )
    insert_issued = asyncio.Event()

    def on_statement(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.startswith(f"INSERT INTO {CollectionRow.__tablename__}"):
            insert_issued.set()

    event.listen(sqlalchemy_engine.sync_engine, "before_cursor_execute", on_statement)
    creator = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            await remote.execute(
                delete(CollectionRow).where(CollectionRow.incarnation == victim)
            )
            await remote.execute(
                insert(PurgeQueueRow).values(
                    incarnation=victim,
                    vector_store_name=vector_store_name,
                    namespace=NAMESPACE,
                    name="victim",
                    config=CONFIG.model_dump(mode="json"),
                    enqueued_at=func.now(),
                )
            )
            creator = asyncio.create_task(registry.register(NAMESPACE, "fresh", CONFIG))
            await asyncio.wait_for(insert_issued.wait(), 30)
        fresh = await asyncio.wait_for(creator, 30)
    finally:
        event.remove(
            sqlalchemy_engine.sync_engine, "before_cursor_execute", on_statement
        )
        if creator is not None and not creator.done():
            creator.cancel()

    assert fresh != victim, "a live collection took an incarnation awaiting purge"
    assert await _queued(registry) == [victim]


@pytest.mark.asyncio
async def test_a_round_claims_one_tombstone(sqlalchemy_engine, vector_store_name):
    """A claim takes one tombstone and leaves the others to other purgers."""
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
    )
    for name in ("a", "b", "c"):
        await registry.register(NAMESPACE, name, CONFIG)
        await registry.unregister(NAMESPACE, name)

    assert await _round(registry, any_records_found=False) is not None

    assert len(await _queued(registry)) == 2


@pytest.mark.asyncio
async def test_a_persistent_database_error_surfaces_with_its_cause(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A rejected insert is retried a bounded number of times, then the
    database's own error is chained to the one that gives up."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    # A NOT NULL violation stands in for a cause that is not a collision.
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: None,
    )

    with pytest.raises(VectorStoreAttemptsExhaustedError) as raised:
        await asyncio.wait_for(registry.register(NAMESPACE, "a", CONFIG), 30)

    cause: BaseException | None = raised.value
    while cause is not None and not isinstance(cause, IntegrityError):
        cause = cause.__cause__
    assert isinstance(cause, IntegrityError)
