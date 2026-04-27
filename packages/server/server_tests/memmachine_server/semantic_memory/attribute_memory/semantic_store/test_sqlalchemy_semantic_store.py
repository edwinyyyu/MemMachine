"""Tests for :class:`SQLAlchemySemanticStore` against SQLite."""

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
)
from memmachine_server.semantic_memory.attribute_memory.semantic_store import (
    SemanticAttribute,
    SemanticStorePartitionAlreadyExistsError,
)
from memmachine_server.semantic_memory.attribute_memory.semantic_store.sqlalchemy_semantic_store import (
    SQLAlchemySemanticStore,
    SQLAlchemySemanticStoreParams,
    SQLAlchemySemanticStorePartition,
)

PARTITION = "org_acme_42"
OTHER_PARTITION = "org_acme_99"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store(
    sqlite_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySemanticStore]:
    s = SQLAlchemySemanticStore(SQLAlchemySemanticStoreParams(engine=sqlite_engine))
    await s.startup()
    yield s
    await s.shutdown()


@pytest_asyncio.fixture
async def partition(
    store: SQLAlchemySemanticStore,
) -> SQLAlchemySemanticStorePartition:
    return await store.open_or_create_partition(PARTITION)


def _attr(
    *,
    topic: str = "Profile",
    category: str = "food",
    attribute: str = "favorite_pizza",
    value: str = "margherita",
    properties: dict | None = None,
) -> SemanticAttribute:
    return SemanticAttribute(
        id=uuid4(),
        topic=topic,
        category=category,
        attribute=attribute,
        value=value,
        properties=properties,
    )


# ---------------------------------------------------------------------------
# Partition lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_partition_creates_handle(
    store: SQLAlchemySemanticStore,
) -> None:
    await store.create_partition(PARTITION)
    assert await store.open_partition(PARTITION) is not None


@pytest.mark.asyncio
async def test_create_partition_raises_on_duplicate(
    store: SQLAlchemySemanticStore,
) -> None:
    await store.create_partition(PARTITION)
    with pytest.raises(SemanticStorePartitionAlreadyExistsError):
        await store.create_partition(PARTITION)


@pytest.mark.asyncio
async def test_open_partition_missing_returns_none(
    store: SQLAlchemySemanticStore,
) -> None:
    assert await store.open_partition(PARTITION) is None


@pytest.mark.asyncio
async def test_open_or_create_partition_is_idempotent(
    store: SQLAlchemySemanticStore,
) -> None:
    first = await store.open_or_create_partition(PARTITION)
    second = await store.open_or_create_partition(PARTITION)
    assert isinstance(first, SQLAlchemySemanticStorePartition)
    assert isinstance(second, SQLAlchemySemanticStorePartition)


@pytest.mark.asyncio
async def test_delete_partition_removes_data(
    store: SQLAlchemySemanticStore,
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr()
    await partition.add_attributes([a])

    await store.delete_partition(PARTITION)

    reopened = await store.open_or_create_partition(PARTITION)
    assert await reopened.get_attributes([a.id]) == {}


@pytest.mark.asyncio
async def test_delete_partition_is_idempotent(
    store: SQLAlchemySemanticStore,
) -> None:
    await store.delete_partition(PARTITION)
    await store.delete_partition(PARTITION)


@pytest.mark.parametrize(
    "bad_key",
    ["HasUpperCase", "has-dashes", "has.dots", "has spaces", "", "a" * 33],
)
@pytest.mark.asyncio
async def test_create_partition_rejects_invalid_keys(
    store: SQLAlchemySemanticStore,
    bad_key: str,
) -> None:
    with pytest.raises(ValueError, match="Partition key"):
        await store.create_partition(bad_key)


# ---------------------------------------------------------------------------
# add_attributes batching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_attributes_empty_is_noop(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    await partition.add_attributes([])
    assert await partition.list_attribute_uuids_matching() == ()


@pytest.mark.asyncio
async def test_add_attributes_round_trip(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr(properties={"source": "doc.txt"})
    await partition.add_attributes([a])

    got = (await partition.get_attributes([a.id])).get(a.id)
    assert got is not None
    assert got.topic == "Profile"
    assert got.category == "food"
    assert got.attribute == "favorite_pizza"
    assert got.value == "margherita"
    assert got.properties == {"source": "doc.txt"}


@pytest.mark.asyncio
async def test_add_attributes_batch_persists_all(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    attrs = [_attr(attribute=f"attr_{i}") for i in range(5)]
    await partition.add_attributes(attrs)

    got = await partition.get_attributes([a.id for a in attrs])
    assert set(got.keys()) == {a.id for a in attrs}


@pytest.mark.asyncio
async def test_add_attributes_raises_on_uuid_collision(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr()
    await partition.add_attributes([a])
    clash = SemanticAttribute(
        id=a.id,
        topic="Profile",
        category="food",
        attribute="dup",
        value="dup",
    )
    with pytest.raises(IntegrityError):
        await partition.add_attributes([clash])


# ---------------------------------------------------------------------------
# get_attributes / list_attributes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_attributes_missing_returns_empty(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    assert await partition.get_attributes([uuid4()]) == {}


@pytest.mark.asyncio
async def test_get_attributes_returns_only_found(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr()
    b = _attr(category="music", value="jazz")
    await partition.add_attributes([a, b])

    got = await partition.get_attributes([a.id, uuid4(), b.id])
    assert set(got.keys()) == {a.id, b.id}


@pytest.mark.asyncio
async def test_list_attributes_no_filter_streams_all(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    attrs = [_attr(attribute=f"a_{i}") for i in range(3)]
    await partition.add_attributes(attrs)

    collected = [x async for x in partition.list_attributes()]
    assert {x.id for x in collected} == {a.id for a in attrs}


@pytest.mark.asyncio
async def test_list_attributes_filter_by_system_field(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr(category="food")
    b = _attr(category="music")
    await partition.add_attributes([a, b])

    result = [
        x
        async for x in partition.list_attributes(
            filter_expr=Comparison(field="category", op="=", value="food")
        )
    ]
    assert [x.id for x in result] == [a.id]


@pytest.mark.asyncio
async def test_list_attributes_filter_by_user_metadata(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr(properties={"source": "doc1"})
    b = _attr(properties={"source": "doc2"}, category="music")
    await partition.add_attributes([a, b])

    result = [
        x
        async for x in partition.list_attributes(
            filter_expr=Comparison(field="m.source", op="=", value="doc1")
        )
    ]
    assert [x.id for x in result] == [a.id]


@pytest.mark.asyncio
async def test_list_attributes_filter_by_system_underscore_metadata(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    """``m._app_internal`` maps to the same properties key."""
    a = _attr(properties={"_app_internal": "v_0"})
    b = _attr(properties={"_app_internal": "v_1"}, category="music")
    await partition.add_attributes([a, b])

    result = [
        x
        async for x in partition.list_attributes(
            filter_expr=Comparison(field="m._app_internal", op="=", value="v_0")
        )
    ]
    assert [x.id for x in result] == [a.id]


@pytest.mark.asyncio
async def test_list_attributes_compound_filter(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr(topic="Profile", category="food")
    b = _attr(topic="Code", category="food")
    await partition.add_attributes([a, b])

    result = [
        x
        async for x in partition.list_attributes(
            filter_expr=And(
                left=Comparison(field="topic", op="=", value="Profile"),
                right=Comparison(field="category", op="=", value="food"),
            )
        )
    ]
    assert [x.id for x in result] == [a.id]


@pytest.mark.asyncio
async def test_list_attribute_uuids_matching(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr(topic="Profile")
    b = _attr(topic="Code")
    await partition.add_attributes([a, b])

    uuids = await partition.list_attribute_uuids_matching(
        filter_expr=In(field="topic", values=["Profile"])
    )
    assert uuids == (a.id,)


# ---------------------------------------------------------------------------
# Cross-partition isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partitions_are_isolated(
    store: SQLAlchemySemanticStore,
) -> None:
    p1 = await store.open_or_create_partition(PARTITION)
    p2 = await store.open_or_create_partition(OTHER_PARTITION)

    a = _attr()
    await p1.add_attributes([a])

    assert a.id in await p1.get_attributes([a.id])
    assert await p2.get_attributes([a.id]) == {}


@pytest.mark.asyncio
async def test_delete_partition_leaves_others(
    store: SQLAlchemySemanticStore,
) -> None:
    p1 = await store.open_or_create_partition(PARTITION)
    p2 = await store.open_or_create_partition(OTHER_PARTITION)

    a = _attr()
    b = _attr(category="music")
    await p1.add_attributes([a])
    await p2.add_attributes([b])

    await store.delete_partition(PARTITION)

    assert b.id in await p2.get_attributes([b.id])


# ---------------------------------------------------------------------------
# delete_attributes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_attributes_removes_rows(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr()
    b = _attr(category="music")
    await partition.add_attributes([a, b])

    await partition.delete_attributes([a.id])

    remaining = await partition.get_attributes([a.id, b.id])
    assert a.id not in remaining
    assert b.id in remaining


@pytest.mark.asyncio
async def test_delete_attributes_missing_uuids_is_noop(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    await partition.delete_attributes([uuid4(), uuid4()])


@pytest.mark.asyncio
async def test_delete_attributes_empty_is_noop(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    await partition.delete_attributes([])


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_citations_then_load(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr()
    await partition.add_attributes([a])

    h1, h2 = uuid4(), uuid4()
    await partition.add_citations(a.id, [h1, h2])

    got = (await partition.get_attributes([a.id], load_citations=True)).get(a.id)
    assert got is not None
    assert got.citations is not None
    assert set(got.citations) == {h1, h2}


@pytest.mark.asyncio
async def test_get_attributes_default_no_citations(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr()
    await partition.add_attributes([a])
    await partition.add_citations(a.id, [uuid4()])

    got = (await partition.get_attributes([a.id])).get(a.id)
    assert got is not None
    assert got.citations is None


@pytest.mark.asyncio
async def test_delete_attribute_cascades_citations(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr()
    await partition.add_attributes([a])
    await partition.add_citations(a.id, [uuid4(), uuid4()])

    await partition.delete_attributes([a.id])

    # Re-inserting with the same uuid must succeed because the FK
    # cascade cleared the prior citations.
    b = SemanticAttribute(
        id=a.id,
        topic="Profile",
        category="music",
        attribute="favorite_genre",
        value="jazz",
    )
    await partition.add_attributes([b])
    got = (await partition.get_attributes([b.id], load_citations=True)).get(b.id)
    assert got is not None
    assert got.citations == ()


@pytest.mark.asyncio
async def test_add_citations_empty_is_noop(
    partition: SQLAlchemySemanticStorePartition,
) -> None:
    a = _attr()
    await partition.add_attributes([a])
    await partition.add_citations(a.id, [])


# ---------------------------------------------------------------------------
# Params validation
# ---------------------------------------------------------------------------


def test_params_reject_ephemeral_sqlite_memory() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    bad = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    with pytest.raises((AssertionError, ValueError)):
        SQLAlchemySemanticStoreParams(engine=bad)


def _unused(_: UUID) -> None:
    pass
