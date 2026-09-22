"""The SQLAlchemy collection registry: creation arbitrated by the database, deletion queued, purge claimed."""

import asyncio
import json
from collections.abc import Iterator
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import DateTime, event, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry import (
    SQLAlchemyVectorStoreCollectionRegistry,
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
NAMESPACE = "ns"


@pytest.fixture
def recorded_statements(
    sqlalchemy_engine: AsyncEngine,
) -> Iterator[list[tuple[str, Any]]]:
    """Every statement the engine executes, with its parameters, from the fixture on."""
    statements: list[tuple[str, Any]] = []

    def record(_connection, _cursor, statement, parameters, _context, _many):
        statements.append((statement, parameters))

    event.listen(sqlalchemy_engine.sync_engine, "before_cursor_execute", record)
    yield statements
    event.remove(sqlalchemy_engine.sync_engine, "before_cursor_execute", record)


@pytest.fixture
def vector_store_name() -> str:
    """A vector store name no other test (or earlier run on a shared server) used."""
    return f"test_{uuid4().hex[:12]}"


async def _registry(
    engine: AsyncEngine,
    vector_store_name: str,
    tombstone_retention: timedelta = RETENTION,
) -> SQLAlchemyVectorStoreCollectionRegistry:
    registry = SQLAlchemyVectorStoreCollectionRegistry(
        engine=engine,
        vector_store_name=vector_store_name,
        tombstone_retention=tombstone_retention,
    )
    await registry.startup()
    return registry


async def _queued(registry: SQLAlchemyVectorStoreCollectionRegistry) -> list[UUID]:
    """The registry's tombstones, oldest first."""
    async with registry._engine.connect() as connection:
        rows = await connection.execute(
            select(registry._purge_queue.c.incarnation).order_by(
                registry._purge_queue.c.enqueued_at
            )
        )
        return list(rows.scalars())


async def _clean_rounds(
    registry: SQLAlchemyVectorStoreCollectionRegistry,
) -> dict[UUID, bool]:
    """Each tombstone of the registry, and whether a round has found it clean."""
    async with registry._engine.connect() as connection:
        rows = await connection.execute(
            select(
                registry._purge_queue.c.incarnation, registry._purge_queue.c.clean_at
            )
        )
        return {row.incarnation: row.clean_at is not None for row in rows}


async def _age_clean_round(
    registry: SQLAlchemyVectorStoreCollectionRegistry,
    incarnation: UUID,
    extra: timedelta = timedelta(0),
) -> None:
    """Move the tombstone's clean round back past the retention, on the database clock."""
    async with registry._engine.begin() as connection:
        database_now = (
            await connection.execute(select(func.now(type_=DateTime(timezone=True))))
        ).scalar_one()
        await connection.execute(
            update(registry._purge_queue)
            .where(registry._purge_queue.c.incarnation == incarnation)
            .values(clean_at=database_now - RETENTION - timedelta(seconds=1) - extra)
        )


async def _age_enqueue(
    registry: SQLAlchemyVectorStoreCollectionRegistry, incarnation: UUID
) -> None:
    """Move the tombstone's enqueue stamp back a minute, on the database clock."""
    async with registry._engine.begin() as connection:
        database_now = (
            await connection.execute(select(func.now(type_=DateTime(timezone=True))))
        ).scalar_one()
        await connection.execute(
            update(registry._purge_queue)
            .where(registry._purge_queue.c.incarnation == incarnation)
            .values(enqueued_at=database_now - timedelta(minutes=1))
        )


async def _round(
    registry: SQLAlchemyVectorStoreCollectionRegistry, found: bool
) -> UUID | None:
    """One purge round on the oldest due tombstone, reporting `found`; its incarnation, or None."""
    async with registry.claim_due() as claim:
        if claim is None:
            return None
        claim.found = found
        return claim.incarnation


@pytest.mark.asyncio
async def test_create_registers_a_fresh_incarnation_with_the_config(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)

    incarnation = await registry.create(NAMESPACE, "c", CONFIG)

    registered = await registry.get(NAMESPACE, "c")
    assert registered is not None
    assert registered.incarnation == incarnation
    assert registered.config == CONFIG
    assert await registry.is_live(incarnation)
    assert await registry.get(NAMESPACE, "d") is None
    assert await registry.get("other_ns", "c") is None


@pytest.mark.asyncio
async def test_a_taken_name_is_already_exists_whatever_the_config(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    await registry.create(NAMESPACE, "c", CONFIG)

    with pytest.raises(VectorStoreCollectionAlreadyExistsError):
        await registry.create(NAMESPACE, "c", CONFIG)
    with pytest.raises(VectorStoreCollectionAlreadyExistsError):
        await registry.create(NAMESPACE, "c", OTHER_CONFIG)


@pytest.mark.parametrize(
    "invalid_name", ["", "Upper", "with-hyphen", "trailing_newline\n", "x" * 33]
)
def test_a_vector_store_name_must_be_an_identifier(invalid_name):
    with pytest.raises(ValueError, match="Vector store name"):
        SQLAlchemyVectorStoreCollectionRegistry(
            engine=create_async_engine("sqlite+aiosqlite://"),
            vector_store_name=invalid_name,
            tombstone_retention=RETENTION,
        )


def test_an_engine_of_another_dialect_is_refused(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://")
    monkeypatch.setattr(engine.dialect, "name", "mssql")
    with pytest.raises(ValueError, match="mssql"):
        SQLAlchemyVectorStoreCollectionRegistry(
            engine=engine, vector_store_name="store", tombstone_retention=RETENTION
        )


@pytest.mark.asyncio
async def test_registries_of_two_vector_stores_share_a_database_and_nothing_else(
    sqlalchemy_engine, vector_store_name
):
    """Each registry has its own collections, and a purge round claims
    only its own tombstones."""
    first = await _registry(sqlalchemy_engine, f"{vector_store_name}_a")
    second = await _registry(sqlalchemy_engine, f"{vector_store_name}_b")

    one = await first.create(NAMESPACE, "c", CONFIG)
    two = await second.create(NAMESPACE, "c", OTHER_CONFIG)
    assert one != two
    in_first = await first.get(NAMESPACE, "c")
    in_second = await second.get(NAMESPACE, "c")
    assert in_first is not None
    assert in_second is not None
    assert in_first.incarnation == one
    assert in_second.incarnation == two

    await first.delete(NAMESPACE, "c")
    assert await second.is_live(two)
    assert await _round(second, found=False) is None
    assert await _round(first, found=False) == one


@pytest.mark.asyncio
async def test_concurrent_creators_get_one_winner(sqlalchemy_engine, vector_store_name):
    """Two registries on one database, the two-process shape in one process."""
    first = await _registry(sqlalchemy_engine, vector_store_name)
    second = await _registry(sqlalchemy_engine, vector_store_name)

    results = await asyncio.gather(
        *(registry.create(NAMESPACE, "c", CONFIG) for registry in (first, second) * 3),
        return_exceptions=True,
    )

    winners = [r for r in results if isinstance(r, UUID)]
    losers = [
        r for r in results if isinstance(r, VectorStoreCollectionAlreadyExistsError)
    ]
    assert len(winners) == 1
    assert len(losers) == 5
    assert await first.is_live(winners[0])


@pytest.mark.asyncio
async def test_delete_queues_the_incarnation_and_is_idempotent(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.create(NAMESPACE, "c", CONFIG)

    await registry.delete(NAMESPACE, "c")

    assert await registry.get(NAMESPACE, "c") is None
    assert not await registry.is_live(incarnation)
    assert await _queued(registry) == [incarnation]
    await registry.delete(NAMESPACE, "c")
    await registry.delete(NAMESPACE, "never")
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_recreated_name_gets_a_new_incarnation(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    old = await registry.create(NAMESPACE, "c", CONFIG)
    await registry.delete(NAMESPACE, "c")

    new = await registry.create(NAMESPACE, "c", CONFIG)

    assert new != old
    assert await registry.is_live(new)
    assert not await registry.is_live(old)


@pytest.mark.asyncio
async def test_a_claim_names_where_the_points_are(sqlalchemy_engine, vector_store_name):
    """The tombstone carries what a purger needs to find the points: namespace, name, configuration."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.create(NAMESPACE, "c", OTHER_CONFIG)
    await registry.delete(NAMESPACE, "c")

    async with registry.claim_due() as claim:
        assert claim is not None
        assert claim.incarnation == incarnation
        assert claim.namespace == NAMESPACE
        assert claim.name == "c"
        assert claim.config == OTHER_CONFIG
        claim.found = False


@pytest.mark.asyncio
async def test_rounds_go_oldest_first_and_a_clean_round_stamps_the_tombstone(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    minted = iter([UUID(int=1), UUID(int=2)])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )
    first = await registry.create(NAMESPACE, "a", CONFIG)
    second = await registry.create(NAMESPACE, "b", CONFIG)
    await registry.delete(NAMESPACE, "a")
    await registry.delete(NAMESPACE, "b")
    # The later deletion, with the larger incarnation, is made the older, so
    # only the ordering by deletion can put it first.
    await _age_enqueue(registry, second)

    assert await _round(registry, found=True) == second
    # Found points: not stamped clean, and still the oldest deletion.
    assert await _round(registry, found=False) == second
    assert await _round(registry, found=False) == first
    # Both had a clean round less than the retention ago: nothing is due.
    assert await _round(registry, found=False) is None
    assert await _queued(registry) == [second, first]
    assert await _clean_rounds(registry) == {first: True, second: True}


@pytest.mark.asyncio
async def test_tombstones_found_clean_are_claimed_oldest_round_first(
    sqlalchemy_engine, vector_store_name
):
    """Past the retention, the tombstone whose clean round is oldest goes
    first, whatever the order of the deletions."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    first = await registry.create(NAMESPACE, "a", CONFIG)
    second = await registry.create(NAMESPACE, "b", CONFIG)
    await registry.delete(NAMESPACE, "a")
    await registry.delete(NAMESPACE, "b")
    await _age_enqueue(registry, first)
    assert await _round(registry, found=False) == first
    assert await _round(registry, found=False) == second
    await _age_clean_round(registry, first)
    await _age_clean_round(registry, second, extra=timedelta(minutes=1))

    assert await _round(registry, found=False) == second
    assert await _round(registry, found=False) == first


@pytest.mark.asyncio
async def test_a_tombstone_stamped_by_a_real_round_is_removed_when_due_again(
    sqlalchemy_engine, vector_store_name
):
    """The removal compares the stamp the database clock wrote, by the
    database's arithmetic: with no retention the second clean round removes
    the tombstone, and no stamp is rewritten by the test."""
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention=timedelta(0)
    )
    incarnation = await registry.create(NAMESPACE, "a", CONFIG)
    await registry.delete(NAMESPACE, "a")

    assert await _round(registry, found=False) == incarnation
    assert await _round(registry, found=False) == incarnation

    assert await _queued(registry) == []
    assert await _round(registry, found=False) is None


@pytest.mark.asyncio
async def test_a_changed_retention_applies_to_tombstones_already_stamped(
    sqlalchemy_engine, vector_store_name
):
    """Only the clean round's time is stored; the retention in force when a
    claim is decided applies, including to tombstones stamped under another."""
    stamped_under_a_day = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await stamped_under_a_day.create(NAMESPACE, "a", CONFIG)
    await stamped_under_a_day.delete(NAMESPACE, "a")
    assert await _round(stamped_under_a_day, found=False) == incarnation
    assert await _round(stamped_under_a_day, found=False) is None

    no_retention = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention=timedelta(0)
    )
    assert await _round(no_retention, found=False) == incarnation
    assert await _queued(no_retention) == []


@pytest.mark.asyncio
async def test_a_tombstone_is_removed_by_a_clean_round_after_the_retention(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.create(NAMESPACE, "a", CONFIG)
    await registry.delete(NAMESPACE, "a")
    assert await _round(registry, found=False) == incarnation
    await _age_clean_round(registry, incarnation)

    assert await _round(registry, found=False) == incarnation

    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_late_write_found_after_the_retention_restarts_the_rounds(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.create(NAMESPACE, "a", CONFIG)
    await registry.delete(NAMESPACE, "a")
    assert await _round(registry, found=False) == incarnation
    await _age_clean_round(registry, incarnation)

    assert await _round(registry, found=True) == incarnation

    assert await _clean_rounds(registry) == {incarnation: False}
    assert await _round(registry, found=False) == incarnation
    assert await _clean_rounds(registry) == {incarnation: True}
    assert await _round(registry, found=False) is None
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_clean_round_does_not_remove_a_tombstone_another_round_found_points_under(
    sqlalchemy_engine, vector_store_name
):
    """Two purgers on one SQLite registry can claim the same tombstone: the
    one that found nothing must not remove what the one that found points
    just un-stamped, or a later write under the incarnation is orphaned."""
    if sqlalchemy_engine.dialect.name != "sqlite":
        pytest.skip("the row lock keeps claims apart on this dialect")
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.create(NAMESPACE, "c", CONFIG)
    await registry.delete(NAMESPACE, "c")
    assert await _round(registry, found=False) == incarnation
    await _age_clean_round(registry, incarnation)

    async with registry.claim_due() as stale:
        assert stale is not None
        assert stale.incarnation == incarnation
        # A second purger claims the same entry, finds points, un-stamps it.
        async with registry.claim_due() as other:
            assert other is not None
            assert other.incarnation == incarnation
            other.found = True
        stale.found = False

    assert await _queued(registry) == [incarnation]
    assert await _clean_rounds(registry) == {incarnation: False}


@pytest.mark.asyncio
async def test_a_round_whose_body_raises_keeps_the_tombstone_as_it_was(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.create(NAMESPACE, "a", CONFIG)
    await registry.delete(NAMESPACE, "a")

    async def refused_reclamation() -> None:
        async with registry.claim_due() as claim:
            assert claim is not None
            assert claim.incarnation == incarnation
            claim.found = True
            raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError):
        await refused_reclamation()

    assert await _clean_rounds(registry) == {incarnation: False}
    assert await _round(registry, found=False) == incarnation
    assert await _clean_rounds(registry) == {incarnation: True}


@pytest.mark.asyncio
async def test_a_round_that_reports_nothing_is_an_error(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    await registry.create(NAMESPACE, "a", CONFIG)
    await registry.delete(NAMESPACE, "a")

    with pytest.raises(RuntimeError, match="without reporting"):
        async with registry.claim_due():
            pass


@pytest.mark.asyncio
async def test_an_incarnation_awaiting_purge_is_never_reminted(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A minted incarnation colliding with queued garbage is rejected and re-minted."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = await registry.create(NAMESPACE, "a", CONFIG)
    await registry.delete(NAMESPACE, "a")
    minted = iter([dead, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )

    fresh = await registry.create(NAMESPACE, "b", CONFIG)

    assert fresh != dead
    assert await _queued(registry) == [dead]


@pytest.mark.asyncio
async def test_creation_that_never_mints_a_free_incarnation_gives_up(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = await registry.create(NAMESPACE, "a", CONFIG)
    await registry.delete(NAMESPACE, "a")
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: dead,
    )

    with pytest.raises(VectorStoreAttemptsExhaustedError):
        await registry.create(NAMESPACE, "b", CONFIG)
    assert await registry.get(NAMESPACE, "b") is None


def _claims(statements: list[tuple[str, Any]], registry) -> list[tuple[str, Any]]:
    """The recorded purge claims: the bounded reads of the registry's queue."""
    return [
        (statement, parameters)
        for statement, parameters in statements
        if statement.lstrip().startswith("SELECT")
        and registry._purge_queue.name in statement
        and "LIMIT" in statement
    ]


def _rows_removed_by_filter(plan: dict) -> int:
    return plan.get("Rows Removed by Filter", 0) + sum(
        _rows_removed_by_filter(child) for child in plan.get("Plans", [])
    )


async def _assert_reads_only_what_it_returns(
    engine: AsyncEngine, claim: str, parameters: Any, index: str
) -> None:
    """The claim reads `index`, a partial index holding only candidates, and
    filters out no row it reads."""
    async with engine.connect() as connection:
        if engine.dialect.name == "postgresql":
            # A table this small may be scanned whole whatever its indexes;
            # with sequential scans off the plan shows what an index serves.
            await connection.execute(text("SET LOCAL enable_seqscan = off"))
            [[explained]] = (
                await connection.exec_driver_sql(
                    f"EXPLAIN (ANALYZE, FORMAT JSON) {claim}", parameters
                )
            ).all()
            plan = json.loads(explained) if isinstance(explained, str) else explained
            assert index in json.dumps(plan), plan
            assert _rows_removed_by_filter(plan[0]["Plan"]) == 0, plan
        else:
            details = [
                row[-1]
                for row in (
                    await connection.exec_driver_sql(
                        f"EXPLAIN QUERY PLAN {claim}", parameters
                    )
                ).all()
            ]
            assert any(d.endswith(index) or f"{index} (" in d for d in details), details


@pytest.mark.asyncio
async def test_a_claim_reads_only_the_tombstones_that_are_due(
    sqlalchemy_engine, vector_store_name, recorded_statements
):
    """Each claim goes straight to a due tombstone, however many are waiting
    out their retention: a range on its own partial index, not a filtered
    scan of the queue."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    waiting = 50
    for index in range(waiting):
        await registry.create(NAMESPACE, f"w{index}", CONFIG)
        await registry.delete(NAMESPACE, f"w{index}")
    for _ in range(waiting):
        assert await _round(registry, found=False) is not None
    fresh = await registry.create(NAMESPACE, "fresh", CONFIG)
    await registry.delete(NAMESPACE, "fresh")
    prefix = registry._purge_queue.name

    recorded_statements.clear()
    assert await _round(registry, found=False) == fresh
    [(unstamped, parameters)] = _claims(recorded_statements, registry)
    await _assert_reads_only_what_it_returns(
        sqlalchemy_engine, unstamped, parameters, f"{prefix}__ea"
    )

    aged = await _queued(registry)
    await _age_clean_round(registry, aged[0])
    recorded_statements.clear()
    assert await _round(registry, found=False) == aged[0]
    [_, (stamped, parameters)] = _claims(recorded_statements, registry)
    await _assert_reads_only_what_it_returns(
        sqlalchemy_engine, stamped, parameters, f"{prefix}__ca"
    )


@pytest.mark.asyncio
async def test_purge_rounds_keep_time_by_the_database_clock(
    sqlalchemy_engine, vector_store_name
):
    """Every time the queue stores, and the retention's cutoff, come from the
    database's clock, never the client's.

    The database's now() is shadowed by one running ten days behind the
    real clock: the tombstone's deletion and clean round are stamped by it,
    and a retention of a day, long past by the real clock, has not passed
    by it.
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
        incarnation = await registry.create(NAMESPACE, "a", CONFIG)
        await registry.delete(NAMESPACE, "a")

        assert await _round(registry, found=False) == incarnation

        queue = registry._purge_queue
        async with skewed_engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        queue.c.enqueued_at,
                        queue.c.clean_at,
                        text("pg_catalog.now() AS real_now"),
                    )
                )
            ).one()
        assert row.real_now - row.enqueued_at > timedelta(days=9)
        assert row.real_now - row.clean_at > timedelta(days=9)
        assert await _round(registry, found=False) is None
    finally:
        await skewed_engine.dispose()
        async with sqlalchemy_engine.begin() as connection:
            await connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
