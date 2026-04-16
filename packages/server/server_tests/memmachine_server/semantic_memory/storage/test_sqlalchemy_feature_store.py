"""Tests for :class:`SQLAlchemyFeatureStore`.

Run against a file-backed SQLite database (the store's params validator
rejects ephemeral in-memory SQLite and StaticPool).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.filter.filter_parser import parse_filter
from memmachine_server.semantic_memory.storage.sqlalchemy_feature_store import (
    SQLAlchemyFeatureStore,
    SQLAlchemyFeatureStoreParams,
)


@pytest_asyncio.fixture
async def sqlite_engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'fs.sqlite'}")
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def store(sqlite_engine: AsyncEngine) -> AsyncIterator[SQLAlchemyFeatureStore]:
    store = SQLAlchemyFeatureStore(SQLAlchemyFeatureStoreParams(engine=sqlite_engine))
    await store.startup()
    yield store
    await store.delete_all()


async def _seed_feature(
    store: SQLAlchemyFeatureStore,
    *,
    set_id: str = "set-a",
    category: str = "preferences",
    feature: str = "favorite_color",
    value: str = "blue",
    tag: str = "ui",
    metadata: dict | None = None,
) -> UUID:
    fid = uuid4()
    await store.add_feature(
        feature_id=fid,
        set_id=set_id,
        category_name=category,
        feature=feature,
        value=value,
        tag=tag,
        metadata=metadata,
    )
    return fid


# ------------------------------------------------------------------ #
# CRUD
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_add_and_get_feature(store: SQLAlchemyFeatureStore) -> None:
    fid = await _seed_feature(store, metadata={"confidence": 0.95})

    feat = await store.get_feature(fid)
    assert feat is not None
    assert feat.metadata.id == fid
    assert feat.set_id == "set-a"
    assert feat.category == "preferences"
    assert feat.tag == "ui"
    assert feat.feature_name == "favorite_color"
    assert feat.value == "blue"
    assert feat.metadata.other == {"confidence": 0.95}


@pytest.mark.asyncio
async def test_get_feature_missing_returns_none(
    store: SQLAlchemyFeatureStore,
) -> None:
    assert await store.get_feature(uuid4()) is None


@pytest.mark.asyncio
async def test_get_features_bulk_skips_missing(
    store: SQLAlchemyFeatureStore,
) -> None:
    a = await _seed_feature(store, feature="a")
    b = await _seed_feature(store, feature="b")
    missing = uuid4()

    result = await store.get_features([a, b, missing])

    assert set(result.keys()) == {a, b}
    assert result[a].feature_name == "a"
    assert result[b].feature_name == "b"


@pytest.mark.asyncio
async def test_get_features_empty_sequence(
    store: SQLAlchemyFeatureStore,
) -> None:
    assert await store.get_features([]) == {}


@pytest.mark.asyncio
async def test_update_feature_only_applies_provided_fields(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store, metadata={"confidence": 0.5})

    await store.update_feature(fid, value="green")

    feat = await store.get_feature(fid)
    assert feat is not None
    assert feat.value == "green"
    assert feat.tag == "ui"  # unchanged
    assert feat.metadata.other == {"confidence": 0.5}  # unchanged


@pytest.mark.asyncio
async def test_update_feature_replaces_metadata(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store, metadata={"confidence": 0.5})

    await store.update_feature(fid, metadata={"source": "llm"})

    feat = await store.get_feature(fid)
    assert feat is not None
    assert feat.metadata.other == {"source": "llm"}


@pytest.mark.asyncio
async def test_update_feature_noop_when_no_values(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store)
    await store.update_feature(fid)
    assert (await store.get_feature(fid)) is not None


# ------------------------------------------------------------------ #
# Citations
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_add_citations_and_load_with_feature(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store)
    await store.add_citations(fid, [_uid("ep-1"), _uid("ep-2")])

    feat = await store.get_feature(fid, load_citations=True)
    assert feat is not None

    assert feat.metadata.citations is not None
    assert sorted(feat.metadata.citations) == [_uid("ep-1"), _uid("ep-2")]


@pytest.mark.asyncio
async def test_load_citations_empty_when_not_requested(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store)
    await store.add_citations(fid, [_uid("ep-1")])

    feat = await store.get_feature(fid)
    assert feat is not None
    assert feat.metadata.citations is None


@pytest.mark.asyncio
async def test_add_citations_empty_sequence_is_noop(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store)
    await store.add_citations(fid, [])
    feat = await store.get_feature(fid, load_citations=True)
    assert feat is not None
    assert feat.metadata.citations == []


@pytest.mark.asyncio
async def test_delete_feature_cascades_citations(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store)
    await store.add_citations(fid, [_uid("ep-1")])

    await store.delete_features([fid])

    # Re-create and verify no orphan citations associate to new feature.
    new_fid = await _seed_feature(store)
    feat = await store.get_feature(new_fid, load_citations=True)
    assert feat is not None
    assert feat.metadata.citations == []


# ------------------------------------------------------------------ #
# get_feature_set: filters, pagination, tag_threshold
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_get_feature_set_filter_by_tag(
    store: SQLAlchemyFeatureStore,
) -> None:
    await _seed_feature(store, feature="a", tag="ui")
    f_food = await _seed_feature(store, feature="b", tag="food")

    results = [
        f async for f in store.get_feature_set(filter_expr=parse_filter("tag = 'food'"))
    ]
    assert len(results) == 1
    assert results[0].metadata.id == f_food


@pytest.mark.asyncio
async def test_get_feature_set_filter_by_set_and_category(
    store: SQLAlchemyFeatureStore,
) -> None:
    match = await _seed_feature(store, set_id="set-a", category="preferences")
    await _seed_feature(store, set_id="set-a", category="facts")
    await _seed_feature(store, set_id="set-b", category="preferences")

    expr = parse_filter("set_id = 'set-a' AND category = 'preferences'")
    results = [f async for f in store.get_feature_set(filter_expr=expr)]
    assert [f.metadata.id for f in results] == [match]


@pytest.mark.asyncio
async def test_get_feature_set_pagination(
    store: SQLAlchemyFeatureStore,
) -> None:
    ids = [await _seed_feature(store, feature=f"f{i}") for i in range(5)]

    page0 = [f async for f in store.get_feature_set(page_size=2, page_num=0)]
    page1 = [f async for f in store.get_feature_set(page_size=2, page_num=1)]
    page2 = [f async for f in store.get_feature_set(page_size=2, page_num=2)]

    assert len(page0) == 2
    assert len(page1) == 2
    assert len(page2) == 1
    # All ids covered exactly once
    seen = {f.metadata.id for f in page0 + page1 + page2}
    assert seen == set(ids)


def _uid(label: str) -> UUID:
    return uuid5(NAMESPACE_DNS, label)


@pytest.mark.asyncio
async def test_get_feature_set_page_num_without_size_raises(
    store: SQLAlchemyFeatureStore,
) -> None:
    from memmachine_server.common.errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError):
        async for _ in store.get_feature_set(page_num=1):
            pass


@pytest.mark.asyncio
async def test_get_feature_set_tag_threshold(
    store: SQLAlchemyFeatureStore,
) -> None:
    for i in range(3):
        await _seed_feature(store, feature=f"food_{i}", tag="food")
    await _seed_feature(store, feature="ui_once", tag="ui")

    results = [f async for f in store.get_feature_set(tag_threshold=2)]
    assert all(f.tag == "food" for f in results)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_get_feature_set_load_citations(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store)
    await store.add_citations(fid, [_uid("ep-1"), _uid("ep-2")])

    results = [f async for f in store.get_feature_set(load_citations=True)]
    assert len(results) == 1
    assert results[0].metadata.citations is not None
    assert sorted(results[0].metadata.citations) == [_uid("ep-1"), _uid("ep-2")]


# ------------------------------------------------------------------ #
# Deletion
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_delete_features_empty(store: SQLAlchemyFeatureStore) -> None:
    await store.delete_features([])


@pytest.mark.asyncio
async def test_delete_feature_set_returns_deleted_ids(
    store: SQLAlchemyFeatureStore,
) -> None:
    keep = await _seed_feature(store, tag="ui")
    drop1 = await _seed_feature(store, tag="food")
    drop2 = await _seed_feature(store, tag="food")

    deleted = await store.delete_feature_set(filter_expr=parse_filter("tag = 'food'"))

    assert set(deleted) == {drop1, drop2}
    assert await store.get_feature(keep) is not None
    assert await store.get_feature(drop1) is None
    assert await store.get_feature(drop2) is None


@pytest.mark.asyncio
async def test_delete_feature_set_no_filter_deletes_all(
    store: SQLAlchemyFeatureStore,
) -> None:
    a = await _seed_feature(store, feature="a")
    b = await _seed_feature(store, feature="b")

    deleted = await store.delete_feature_set()
    assert set(deleted) == {a, b}


# ------------------------------------------------------------------ #
# History / ingestion tracking
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_history_ingestion_roundtrip(
    store: SQLAlchemyFeatureStore,
) -> None:
    await store.add_history_to_set("set-a", _uid("ep-1"))
    await store.add_history_to_set("set-a", _uid("ep-2"))
    await store.add_history_to_set("set-a", _uid("ep-3"))

    assert await store.get_history_messages_count(set_ids=["set-a"]) == 3
    assert (
        await store.get_history_messages_count(set_ids=["set-a"], is_ingested=False)
    ) == 3

    await store.mark_messages_ingested(
        set_id="set-a", history_ids=[_uid("ep-1"), _uid("ep-2")]
    )

    assert (
        await store.get_history_messages_count(set_ids=["set-a"], is_ingested=True)
    ) == 2
    assert (
        await store.get_history_messages_count(set_ids=["set-a"], is_ingested=False)
    ) == 1

    pending = [
        h
        async for h in store.get_history_messages(set_ids=["set-a"], is_ingested=False)
    ]
    assert pending == [_uid("ep-3")]


@pytest.mark.asyncio
async def test_mark_messages_ingested_requires_history_ids(
    store: SQLAlchemyFeatureStore,
) -> None:
    with pytest.raises(ValueError, match="No history ids provided"):
        await store.mark_messages_ingested(set_id="set-a", history_ids=[])


@pytest.mark.asyncio
async def test_delete_history_removes_citations(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store)
    await store.add_citations(fid, [_uid("ep-1"), _uid("ep-2")])
    await store.add_history_to_set("set-a", _uid("ep-1"))
    await store.add_history_to_set("set-a", _uid("ep-2"))

    await store.delete_history([_uid("ep-1")])

    feat = await store.get_feature(fid, load_citations=True)
    assert feat is not None
    assert feat.metadata.citations == [_uid("ep-2")]
    assert await store.get_history_messages_count() == 1


@pytest.mark.asyncio
async def test_delete_history_set(store: SQLAlchemyFeatureStore) -> None:
    await store.add_history_to_set("set-a", _uid("ep-1"))
    await store.add_history_to_set("set-b", _uid("ep-2"))

    await store.delete_history_set(["set-a"])

    assert await store.get_history_messages_count(set_ids=["set-a"]) == 0
    assert await store.get_history_messages_count(set_ids=["set-b"]) == 1


@pytest.mark.asyncio
async def test_get_history_set_ids_min_uningested(
    store: SQLAlchemyFeatureStore,
) -> None:
    # set-a: 3 uningested
    for hid in (_uid("ep-a1"), _uid("ep-a2"), _uid("ep-a3")):
        await store.add_history_to_set("set-a", hid)
    # set-b: 1 uningested
    await store.add_history_to_set("set-b", _uid("ep-b1"))

    result = [sid async for sid in store.get_history_set_ids(min_uningested_messages=2)]
    assert result == ["set-a"]


@pytest.mark.asyncio
async def test_get_history_set_ids_older_than(
    store: SQLAlchemyFeatureStore,
) -> None:
    await store.add_history_to_set("set-a", _uid("ep-1"))
    future = datetime.now(UTC) + timedelta(days=1)

    result = [sid async for sid in store.get_history_set_ids(older_than=future)]
    assert result == ["set-a"]


@pytest.mark.asyncio
async def test_purge_ingested_rows_skips_pending_sets(
    store: SQLAlchemyFeatureStore,
) -> None:
    await store.add_history_to_set("set-a", _uid("ep-1"))
    await store.add_history_to_set("set-a", _uid("ep-2"))
    await store.mark_messages_ingested(set_id="set-a", history_ids=[_uid("ep-1")])

    n = await store.purge_ingested_rows(["set-a"])

    assert n == 0
    assert await store.get_history_messages_count(set_ids=["set-a"]) == 2


@pytest.mark.asyncio
async def test_purge_ingested_rows_deletes_fully_ingested(
    store: SQLAlchemyFeatureStore,
) -> None:
    await store.add_history_to_set("set-a", _uid("ep-1"))
    await store.add_history_to_set("set-a", _uid("ep-2"))
    await store.mark_messages_ingested(
        set_id="set-a", history_ids=[_uid("ep-1"), _uid("ep-2")]
    )

    n = await store.purge_ingested_rows(["set-a"])

    assert n == 2
    assert await store.get_history_messages_count(set_ids=["set-a"]) == 0


# ------------------------------------------------------------------ #
# Set discovery
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_get_set_ids_starts_with_unions_feature_and_history_sets(
    store: SQLAlchemyFeatureStore,
) -> None:
    await _seed_feature(store, set_id="org_acme_feature_only")
    await store.add_history_to_set("org_acme_history_only", _uid("ep-1"))
    await _seed_feature(store, set_id="other_scope")

    result = sorted([sid async for sid in store.get_set_ids_starts_with("org_acme_")])
    assert result == ["org_acme_feature_only", "org_acme_history_only"]


# ------------------------------------------------------------------ #
# Lifecycle
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_delete_all_clears_everything(
    store: SQLAlchemyFeatureStore,
) -> None:
    fid = await _seed_feature(store)
    await store.add_citations(fid, [_uid("ep-1")])
    await store.add_history_to_set("set-a", _uid("ep-1"))

    await store.delete_all()

    assert await store.get_feature(fid) is None
    assert await store.get_history_messages_count() == 0
