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
    SQLAlchemySemanticStore,
    SQLAlchemySemanticStoreParams,
)

PARTITION = "org_acme/user_42"
OTHER_PARTITION = "org_acme/user_99"


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
    # engine is disposed by the fixture


def _attr(
    *,
    partition_id: str = PARTITION,
    topic: str = "Profile",
    category: str = "food",
    attribute: str = "favorite_pizza",
    value: str = "margherita",
    properties: dict | None = None,
) -> SemanticAttribute:
    return SemanticAttribute(
        id=uuid4(),
        partition_id=partition_id,
        topic=topic,
        category=category,
        attribute=attribute,
        value=value,
        properties=properties,
    )


# ---------------------------------------------------------------------------
# add_attribute / get_attribute / get_attributes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_and_get_attribute_round_trip(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr(properties={"source": "doc.txt"})
    await store.add_attribute(a)

    got = await store.get_attribute(a.id)
    assert got is not None
    assert got.id == a.id
    assert got.partition_id == PARTITION
    assert got.topic == "Profile"
    assert got.category == "food"
    assert got.attribute == "favorite_pizza"
    assert got.value == "margherita"
    assert got.properties == {"source": "doc.txt"}


@pytest.mark.asyncio
async def test_get_attribute_missing_returns_none(
    store: SQLAlchemySemanticStore,
) -> None:
    assert await store.get_attribute(uuid4()) is None


@pytest.mark.asyncio
async def test_get_attributes_bulk_returns_only_found(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    b = _attr(category="music", attribute="favorite_genre", value="jazz")
    missing = uuid4()

    await store.add_attribute(a)
    await store.add_attribute(b)

    got = await store.get_attributes([a.id, missing, b.id])
    assert set(got.keys()) == {a.id, b.id}
    assert got[a.id].value == "margherita"
    assert got[b.id].value == "jazz"


@pytest.mark.asyncio
async def test_get_attributes_empty_input_returns_empty(
    store: SQLAlchemySemanticStore,
) -> None:
    assert await store.get_attributes([]) == {}


@pytest.mark.asyncio
async def test_add_attribute_raises_on_id_collision(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    await store.add_attribute(a)
    # Second add with same id must raise (strict insert).
    clash = SemanticAttribute(
        id=a.id,
        partition_id=PARTITION,
        topic="Profile",
        category="food",
        attribute="dup",
        value="dup",
    )
    with pytest.raises(IntegrityError):
        await store.add_attribute(clash)


# ---------------------------------------------------------------------------
# list_attributes / list_attribute_ids_matching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_attributes_no_filter_streams_all(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    b = _attr(category="music", attribute="favorite_genre", value="jazz")
    await store.add_attribute(a)
    await store.add_attribute(b)

    collected = [x async for x in store.list_attributes()]
    assert {x.id for x in collected} == {a.id, b.id}


@pytest.mark.asyncio
async def test_list_attributes_filter_by_system_field(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr(category="food")
    b = _attr(category="music", attribute="favorite_genre", value="jazz")
    await store.add_attribute(a)
    await store.add_attribute(b)

    result = [
        x
        async for x in store.list_attributes(
            filter_expr=Comparison(field="category", op="=", value="food")
        )
    ]
    assert [x.id for x in result] == [a.id]


@pytest.mark.asyncio
async def test_list_attributes_filter_by_user_metadata(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr(properties={"source": "doc1"})
    b = _attr(properties={"source": "doc2"}, category="music")
    await store.add_attribute(a)
    await store.add_attribute(b)

    result = [
        x
        async for x in store.list_attributes(
            filter_expr=Comparison(field="m.source", op="=", value="doc1")
        )
    ]
    assert [x.id for x in result] == [a.id]


@pytest.mark.asyncio
async def test_list_attributes_compound_filter(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr(partition_id=PARTITION, category="food")
    b = _attr(partition_id=OTHER_PARTITION, category="food")
    c = _attr(partition_id=PARTITION, category="music")
    for x in (a, b, c):
        await store.add_attribute(x)

    result = [
        x
        async for x in store.list_attributes(
            filter_expr=And(
                left=Comparison(field="partition_id", op="=", value=PARTITION),
                right=Comparison(field="category", op="=", value="food"),
            )
        )
    ]
    assert [x.id for x in result] == [a.id]


@pytest.mark.asyncio
async def test_list_attribute_ids_matching(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr(partition_id=PARTITION)
    b = _attr(partition_id=OTHER_PARTITION)
    await store.add_attribute(a)
    await store.add_attribute(b)

    ids = await store.list_attribute_ids_matching(
        filter_expr=In(field="partition_id", values=[PARTITION])
    )
    assert ids == (a.id,)


@pytest.mark.asyncio
async def test_list_attribute_ids_matching_no_filter_returns_all(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    b = _attr(category="music")
    await store.add_attribute(a)
    await store.add_attribute(b)

    ids = await store.list_attribute_ids_matching()
    assert set(ids) == {a.id, b.id}


# ---------------------------------------------------------------------------
# delete_attributes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_attributes_removes_rows(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    b = _attr(category="music")
    await store.add_attribute(a)
    await store.add_attribute(b)

    await store.delete_attributes([a.id])

    assert await store.get_attribute(a.id) is None
    assert await store.get_attribute(b.id) is not None


@pytest.mark.asyncio
async def test_delete_attributes_missing_ids_is_noop(
    store: SQLAlchemySemanticStore,
) -> None:
    await store.delete_attributes([uuid4(), uuid4()])


@pytest.mark.asyncio
async def test_delete_attributes_empty_input_is_noop(
    store: SQLAlchemySemanticStore,
) -> None:
    await store.delete_attributes([])


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_citations_then_load(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    await store.add_attribute(a)

    h1, h2 = uuid4(), uuid4()
    await store.add_citations(a.id, [h1, h2])

    got = await store.get_attribute(a.id, load_citations=True)
    assert got is not None
    assert got.citations is not None
    assert set(got.citations) == {h1, h2}


@pytest.mark.asyncio
async def test_get_attribute_without_load_citations_returns_none(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    await store.add_attribute(a)
    await store.add_citations(a.id, [uuid4()])

    got = await store.get_attribute(a.id)
    assert got is not None
    assert got.citations is None


@pytest.mark.asyncio
async def test_delete_attribute_cascades_citations(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    await store.add_attribute(a)
    await store.add_citations(a.id, [uuid4(), uuid4()])

    await store.delete_attributes([a.id])

    # Adding a new attribute with the same id should succeed because the
    # citations table no longer references it.
    b = SemanticAttribute(
        id=a.id,
        partition_id=PARTITION,
        topic="Profile",
        category="music",
        attribute="favorite_genre",
        value="jazz",
    )
    await store.add_attribute(b)
    got = await store.get_attribute(b.id, load_citations=True)
    assert got is not None
    assert got.citations == ()


@pytest.mark.asyncio
async def test_add_citations_empty_is_noop(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    await store.add_attribute(a)
    await store.add_citations(a.id, [])


# ---------------------------------------------------------------------------
# list_partitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_partitions_empty(
    store: SQLAlchemySemanticStore,
) -> None:
    result = [p async for p in store.list_partitions()]
    assert result == []


@pytest.mark.asyncio
async def test_list_partitions_distinct(
    store: SQLAlchemySemanticStore,
) -> None:
    await store.add_attribute(_attr(partition_id="org_a/user_1"))
    await store.add_attribute(_attr(partition_id="org_a/user_1", category="m"))
    await store.add_attribute(_attr(partition_id="org_a/user_2"))
    await store.add_attribute(_attr(partition_id="org_b/user_3"))

    result = {p async for p in store.list_partitions()}
    assert result == {"org_a/user_1", "org_a/user_2", "org_b/user_3"}


@pytest.mark.asyncio
async def test_list_partitions_prefix_filter(
    store: SQLAlchemySemanticStore,
) -> None:
    await store.add_attribute(_attr(partition_id="org_a/user_1"))
    await store.add_attribute(_attr(partition_id="org_a/user_2"))
    await store.add_attribute(_attr(partition_id="org_b/user_3"))

    result = {p async for p in store.list_partitions(prefix="org_a/")}
    assert result == {"org_a/user_1", "org_a/user_2"}


# ---------------------------------------------------------------------------
# Lifecycle / delete_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_all_empties_tables(
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    await store.add_attribute(a)
    await store.add_citations(a.id, [uuid4()])

    await store.delete_all()

    assert await store.get_attribute(a.id) is None
    assert [x async for x in store.list_attributes()] == []


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
