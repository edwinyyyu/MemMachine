"""The SQLAlchemy collection registry: creation arbitrated by the database, deletion queued, purge claimed."""

import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import DateTime, func, select, update
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
def vector_store_name() -> str:
    """A vector store name no other test (or earlier run on a shared server) used."""
    return f"test_{uuid4().hex[:12]}"


async def _registry(
    engine: AsyncEngine, vector_store_name: str
) -> SQLAlchemyVectorStoreCollectionRegistry:
    registry = SQLAlchemyVectorStoreCollectionRegistry(
        engine=engine,
        vector_store_name=vector_store_name,
        tombstone_retention=RETENTION,
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
    registry: SQLAlchemyVectorStoreCollectionRegistry, incarnation: UUID
) -> None:
    """Move the tombstone's clean round back past the retention, on the database clock."""
    async with registry._engine.begin() as connection:
        database_now = (
            await connection.execute(select(func.now(type_=DateTime(timezone=True))))
        ).scalar_one()
        await connection.execute(
            update(registry._purge_queue)
            .where(registry._purge_queue.c.incarnation == incarnation)
            .values(clean_at=database_now - RETENTION - timedelta(seconds=1))
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
    async with registry.claim_oldest() as claim:
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


@pytest.mark.parametrize("invalid_name", ["", "Upper", "with-hyphen", "x" * 33])
def test_a_vector_store_name_must_be_an_identifier(invalid_name):
    with pytest.raises(ValueError, match="Vector store name"):
        SQLAlchemyVectorStoreCollectionRegistry(
            engine=create_async_engine("sqlite+aiosqlite://"),
            vector_store_name=invalid_name,
            tombstone_retention=RETENTION,
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

    async with registry.claim_oldest() as claim:
        assert claim is not None
        assert claim.incarnation == incarnation
        assert claim.namespace == NAMESPACE
        assert claim.name == "c"
        assert claim.config == OTHER_CONFIG
        claim.found = False


@pytest.mark.asyncio
async def test_rounds_go_oldest_first_and_a_clean_round_stamps_the_tombstone(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    first = await registry.create(NAMESPACE, "a", CONFIG)
    second = await registry.create(NAMESPACE, "b", CONFIG)
    await registry.delete(NAMESPACE, "a")
    await registry.delete(NAMESPACE, "b")
    # Two deletions in one tick of the database clock are unordered.
    await _age_enqueue(registry, first)

    assert await _round(registry, found=True) == first
    # Found points: the tombstone stays due, and it is still the oldest.
    assert await _round(registry, found=False) == first
    assert await _round(registry, found=False) == second
    # Both had a clean round less than the retention ago: nothing is due.
    assert await _round(registry, found=False) is None
    assert await _queued(registry) == [first, second]
    assert await _clean_rounds(registry) == {first: True, second: True}


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

    async with registry.claim_oldest() as stale:
        assert stale is not None
        assert stale.incarnation == incarnation
        # A second purger claims the same entry, finds points, un-stamps it.
        async with registry.claim_oldest() as other:
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
        async with registry.claim_oldest() as claim:
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
        async with registry.claim_oldest():
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
