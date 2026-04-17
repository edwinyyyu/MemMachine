"""Tests for :class:`AttributeMemory` orchestrator.

Uses a real :class:`SQLAlchemySemanticStore` plus an in-memory vector
collection fake.  Verifies ordering (store-first on add, vector-first on
delete), filter translation on the search path, and enrichment of
vector-query results from the store.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
)
from memmachine_server.semantic_memory.attribute_memory import (
    AttributeMemory,
    SemanticAttribute,
    SQLAlchemySemanticStore,
    SQLAlchemySemanticStoreParams,
)

from .conftest import FakeVectorStoreCollection

PARTITION = "org_acme/user_42"


@pytest_asyncio.fixture
async def store(
    sqlite_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySemanticStore]:
    s = SQLAlchemySemanticStore(SQLAlchemySemanticStoreParams(engine=sqlite_engine))
    await s.startup()
    yield s


@pytest_asyncio.fixture
async def memory(
    store: SQLAlchemySemanticStore,
    fake_vector_collection: FakeVectorStoreCollection,
) -> AttributeMemory:
    return AttributeMemory(store=store, vector_collection=fake_vector_collection)


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
# add_attribute: store-first ordering + reserved-key rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_attribute_persists_to_both_stores(
    memory: AttributeMemory,
    store: SQLAlchemySemanticStore,
    fake_vector_collection: FakeVectorStoreCollection,
) -> None:
    a = _attr()
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])

    stored = await store.get_attribute(a.id)
    assert stored is not None
    vector_rows = await fake_vector_collection.get(record_uuids=[a.id])
    assert len(vector_rows) == 1
    assert vector_rows[0].uuid == a.id
    assert vector_rows[0].properties is not None
    assert vector_rows[0].properties["_partition_id"] == PARTITION


@pytest.mark.asyncio
async def test_add_attribute_rejects_reserved_user_keys(
    memory: AttributeMemory,
    store: SQLAlchemySemanticStore,
    fake_vector_collection: FakeVectorStoreCollection,
) -> None:
    a = _attr(properties={"_custom": "bad"})
    with pytest.raises(ValueError, match="reserved"):
        await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])

    # Nothing persisted to either store.
    assert await store.get_attribute(a.id) is None
    assert await fake_vector_collection.get(record_uuids=[a.id]) == []


@pytest.mark.asyncio
async def test_add_attribute_store_failure_skips_vector() -> None:
    store_mock = AsyncMock()
    store_mock.add_attribute.side_effect = RuntimeError("store down")
    vector_mock = AsyncMock()

    memory = AttributeMemory(store=store_mock, vector_collection=vector_mock)
    a = _attr()
    with pytest.raises(RuntimeError, match="store down"):
        await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])

    store_mock.add_attribute.assert_awaited_once()
    vector_mock.upsert.assert_not_awaited()


# ---------------------------------------------------------------------------
# delete_attributes: vector-first ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_attributes_removes_from_both(
    memory: AttributeMemory,
    store: SQLAlchemySemanticStore,
    fake_vector_collection: FakeVectorStoreCollection,
) -> None:
    a = _attr()
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])

    await memory.delete_attributes([a.id])

    assert await store.get_attribute(a.id) is None
    assert await fake_vector_collection.get(record_uuids=[a.id]) == []


@pytest.mark.asyncio
async def test_delete_attributes_empty_is_noop() -> None:
    store_mock = AsyncMock()
    vector_mock = AsyncMock()
    memory = AttributeMemory(store=store_mock, vector_collection=vector_mock)
    await memory.delete_attributes([])
    store_mock.delete_attributes.assert_not_awaited()
    vector_mock.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_attributes_vector_first_then_store() -> None:
    store_mock = AsyncMock()
    vector_mock = AsyncMock()
    calls: list[str] = []
    vector_mock.delete.side_effect = lambda **_: calls.append("vector")
    store_mock.delete_attributes.side_effect = lambda _: calls.append("store")

    memory = AttributeMemory(store=store_mock, vector_collection=vector_mock)
    await memory.delete_attributes([uuid4()])

    assert calls == ["vector", "store"]


@pytest.mark.asyncio
async def test_delete_attributes_vector_failure_leaves_store_untouched() -> None:
    store_mock = AsyncMock()
    vector_mock = AsyncMock()
    vector_mock.delete.side_effect = RuntimeError("vector down")

    memory = AttributeMemory(store=store_mock, vector_collection=vector_mock)
    with pytest.raises(RuntimeError, match="vector down"):
        await memory.delete_attributes([uuid4()])

    store_mock.delete_attributes.assert_not_awaited()


# ---------------------------------------------------------------------------
# delete_attributes_matching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_attributes_matching_uses_filter(
    memory: AttributeMemory,
    store: SQLAlchemySemanticStore,
    fake_vector_collection: FakeVectorStoreCollection,
) -> None:
    keep = _attr(category="music", attribute="g", value="jazz")
    drop = _attr(category="food", attribute="p", value="pizza")
    await memory.add_attribute(attribute=keep, vector=[0.1, 0.2, 0.3])
    await memory.add_attribute(attribute=drop, vector=[0.4, 0.5, 0.6])

    await memory.delete_attributes_matching(
        filter_expr=Comparison(field="category", op="=", value="food")
    )

    assert await store.get_attribute(drop.id) is None
    assert await fake_vector_collection.get(record_uuids=[drop.id]) == []
    assert await store.get_attribute(keep.id) is not None


@pytest.mark.asyncio
async def test_delete_attributes_matching_noop_when_nothing_matches(
    memory: AttributeMemory,
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])
    await memory.delete_attributes_matching(
        filter_expr=Comparison(field="category", op="=", value="nothing")
    )
    assert await store.get_attribute(a.id) is not None


# ---------------------------------------------------------------------------
# Reads (pass-through)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_attribute_passthrough(memory: AttributeMemory) -> None:
    a = _attr()
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])
    got = await memory.get_attribute(a.id)
    assert got is not None
    assert got.id == a.id


@pytest.mark.asyncio
async def test_get_attributes_bulk(memory: AttributeMemory) -> None:
    a = _attr()
    b = _attr(category="music")
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])
    await memory.add_attribute(attribute=b, vector=[0.4, 0.5, 0.6])

    got = await memory.get_attributes([a.id, b.id])
    assert set(got.keys()) == {a.id, b.id}


@pytest.mark.asyncio
async def test_list_attributes_streams(memory: AttributeMemory) -> None:
    a = _attr()
    b = _attr(category="music")
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])
    await memory.add_attribute(attribute=b, vector=[0.4, 0.5, 0.6])

    result = [x async for x in memory.list_attributes()]
    assert {x.id for x in result} == {a.id, b.id}


@pytest.mark.asyncio
async def test_list_partitions_prefix(memory: AttributeMemory) -> None:
    await memory.add_attribute(
        attribute=_attr(partition_id="org_a/user_1"),
        vector=[0.1, 0.2, 0.3],
    )
    await memory.add_attribute(
        attribute=_attr(partition_id="org_b/user_2"),
        vector=[0.4, 0.5, 0.6],
    )
    result = {p async for p in memory.list_partitions(prefix="org_a/")}
    assert result == {"org_a/user_1"}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_attributes_with_scores(
    memory: AttributeMemory,
) -> None:
    a = _attr()
    b = _attr(category="music", attribute="g", value="jazz")
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])
    await memory.add_attribute(attribute=b, vector=[0.4, 0.5, 0.6])

    results = await memory.search(query_vector=[0.0, 0.0, 0.0])

    assert len(results) == 2
    returned_ids = {attr.id for attr, _ in results}
    assert returned_ids == {a.id, b.id}
    # Fake scorer uses -|insertion_index| so the earliest insert (a) ranks
    # highest under descending-score sort.
    assert results[0][0].id == a.id
    assert results[1][0].id == b.id


@pytest.mark.asyncio
async def test_search_translates_system_filter_for_vector_store(
    memory: AttributeMemory,
    fake_vector_collection: FakeVectorStoreCollection,
) -> None:
    a = _attr(category="food")
    b = _attr(category="music", attribute="g", value="jazz")
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])
    await memory.add_attribute(attribute=b, vector=[0.4, 0.5, 0.6])

    results = await memory.search(
        query_vector=[0.0, 0.0, 0.0],
        filter_expr=Comparison(field="category", op="=", value="food"),
    )

    returned_ids = {attr.id for attr, _ in results}
    assert returned_ids == {a.id}

    sent = fake_vector_collection.last_property_filter
    assert isinstance(sent, Comparison)
    assert sent.field == "_category"


@pytest.mark.asyncio
async def test_search_translates_user_metadata_filter(
    memory: AttributeMemory,
    fake_vector_collection: FakeVectorStoreCollection,
) -> None:
    a = _attr(properties={"source": "doc1"})
    b = _attr(
        category="music",
        attribute="g",
        value="jazz",
        properties={"source": "doc2"},
    )
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])
    await memory.add_attribute(attribute=b, vector=[0.4, 0.5, 0.6])

    await memory.search(
        query_vector=[0.0, 0.0, 0.0],
        filter_expr=Comparison(field="m.source", op="=", value="doc1"),
    )

    sent = fake_vector_collection.last_property_filter
    assert isinstance(sent, Comparison)
    # m. / metadata. prefix gets stripped for the vector store.
    assert sent.field == "source"


@pytest.mark.asyncio
async def test_search_empty_result(memory: AttributeMemory) -> None:
    assert await memory.search(query_vector=[0.0, 0.0, 0.0]) == []


@pytest.mark.asyncio
async def test_search_with_compound_and_in_filters(
    memory: AttributeMemory,
    fake_vector_collection: FakeVectorStoreCollection,
) -> None:
    await memory.add_attribute(
        attribute=_attr(partition_id="org_a/user_1", category="food"),
        vector=[0.1, 0.2, 0.3],
    )
    await memory.add_attribute(
        attribute=_attr(partition_id="org_a/user_2", category="food"),
        vector=[0.4, 0.5, 0.6],
    )
    await memory.add_attribute(
        attribute=_attr(partition_id="org_a/user_1", category="music"),
        vector=[0.7, 0.8, 0.9],
    )

    filter_expr = And(
        left=In(field="partition_id", values=["org_a/user_1"]),
        right=Comparison(field="category", op="=", value="food"),
    )
    results = await memory.search(
        query_vector=[0.0, 0.0, 0.0],
        filter_expr=filter_expr,
    )

    assert len(results) == 1
    sent = fake_vector_collection.last_property_filter
    assert isinstance(sent, And)
    left = sent.left
    right = sent.right
    assert isinstance(left, In)
    assert left.field == "_partition_id"
    assert isinstance(right, Comparison)
    assert right.field == "_category"


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_citations_persists_to_store(
    memory: AttributeMemory,
    store: SQLAlchemySemanticStore,
) -> None:
    a = _attr()
    await memory.add_attribute(attribute=a, vector=[0.1, 0.2, 0.3])
    h1 = uuid4()
    await memory.add_citations(attribute_id=a.id, history_ids=[h1])

    got = await store.get_attribute(a.id, load_citations=True)
    assert got is not None
    assert got.citations == (h1,)
