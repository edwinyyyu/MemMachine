"""Tests for SQLAlchemyEventStore — SQLite (unit) and PostgreSQL (integration)."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_core.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
)
from memmachine_core.event.data_types import (
    Block,
    Event,
    NullContext,
    ProducerContext,
    TextBlock,
)
from memmachine_core.event.event_store import (
    EventStorePartitionAlreadyExistsError,
    EventStorePartitionConfig,
)
from memmachine_core.event.event_store.sqlalchemy_event_store import (
    BaseEventStore,
    SQLAlchemyEventStore,
    SQLAlchemyEventStoreParams,
    SQLAlchemyEventStorePartition,
)

PARTITION_KEY = "test_partition"
BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_NULL_CONTEXT = NullContext()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    *,
    ts_offset_seconds: int = 0,
    blocks: list[Block] | None = None,
    context: ProducerContext | NullContext = _NULL_CONTEXT,
    properties: dict | None = None,
) -> Event:
    return Event(
        uuid=uuid4(),
        timestamp=BASE_TIME + timedelta(seconds=ts_offset_seconds),
        context=context,
        blocks=blocks if blocks is not None else [TextBlock(text="hello")],
        properties=properties or {},
    )


def _config() -> EventStorePartitionConfig:
    """Return the default plaintext partition config."""
    return EventStorePartitionConfig()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_store(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemyEventStore]:
    store = SQLAlchemyEventStore(
        SQLAlchemyEventStoreParams(engine=sqlalchemy_sqlite_engine)
    )
    await store.startup()
    yield store
    async with sqlalchemy_sqlite_engine.begin() as conn:
        await conn.run_sync(BaseEventStore.metadata.drop_all)


@pytest_asyncio.fixture
async def pg_store(
    sqlalchemy_pg_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemyEventStore]:
    store = SQLAlchemyEventStore(
        SQLAlchemyEventStoreParams(engine=sqlalchemy_pg_engine)
    )
    await store.startup()
    yield store
    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseEventStore.metadata.drop_all)


@pytest.fixture(
    params=[
        "sqlite_store",
        pytest.param("pg_store", marks=pytest.mark.integration),
    ],
)
def store(request) -> SQLAlchemyEventStore:
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture
async def partition(
    store: SQLAlchemyEventStore,
) -> SQLAlchemyEventStorePartition:
    return await store.open_or_create_partition(PARTITION_KEY, _config())


# ===================================================================
# add_events / get_event / get_events
# ===================================================================


@pytest.mark.asyncio
async def test_add_event_and_get_event(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    event = _event()
    await partition.add_events([event])

    retrieved = await partition.get_event(event.uuid)
    assert retrieved == event


@pytest.mark.asyncio
async def test_get_event_unknown(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    assert await partition.get_event(uuid4()) is None


@pytest.mark.asyncio
async def test_add_events_and_get_events(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    events = [_event(ts_offset_seconds=i) for i in range(5)]
    await partition.add_events(events)

    retrieved = await partition.get_events([event.uuid for event in events])
    assert retrieved == {event.uuid: event for event in events}


@pytest.mark.asyncio
async def test_get_events_empty(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    assert await partition.get_events([]) == {}


@pytest.mark.asyncio
async def test_get_events_partial(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    """Unknown UUIDs are simply absent from the result."""
    event = _event()
    await partition.add_events([event])

    retrieved = await partition.get_events([event.uuid, uuid4()])
    assert set(retrieved) == {event.uuid}


@pytest.mark.asyncio
async def test_add_events_empty(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    await partition.add_events([])
    assert await partition.get_all_events() == []


@pytest.mark.asyncio
async def test_event_uuid_identity(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    """The stored UUID is the native UUID and round-trips identically."""
    event = _event()
    await partition.add_events([event])

    retrieved = await partition.get_event(event.uuid)
    assert retrieved is not None
    assert retrieved.uuid == event.uuid
    assert isinstance(retrieved.uuid, type(event.uuid))


# ===================================================================
# Content fidelity: blocks, context, properties
# ===================================================================


@pytest.mark.asyncio
async def test_multimodal_blocks_round_trip(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    """An event's ordered list of content blocks survives a write/read cycle."""
    event = _event(
        blocks=[
            TextBlock(text="first"),
            TextBlock(text="second"),
            TextBlock(text="third"),
        ]
    )
    await partition.add_events([event])

    retrieved = await partition.get_event(event.uuid)
    assert retrieved is not None
    assert retrieved.blocks == event.blocks


@pytest.mark.asyncio
async def test_typed_properties_round_trip(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    """Typed filterable properties survive a write/read cycle unchanged."""
    properties = {
        "color": "red",
        "score": 42,
        "active": True,
        "ratio": 1.5,
        "occurred_at": datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
    }
    event = _event(properties=properties)
    await partition.add_events([event])

    retrieved = await partition.get_event(event.uuid)
    assert retrieved is not None
    assert retrieved.properties == properties


@pytest.mark.asyncio
async def test_producer_context_round_trip(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    context = ProducerContext(producer="Alice")
    event = _event(context=context)
    await partition.add_events([event])

    retrieved = await partition.get_event(event.uuid)
    assert retrieved is not None
    assert retrieved.context == context


@pytest.mark.asyncio
async def test_null_context_round_trip(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    event = _event(context=NullContext())
    await partition.add_events([event])

    retrieved = await partition.get_event(event.uuid)
    assert retrieved is not None
    assert retrieved.context == NullContext()


# ===================================================================
# get_all_events
# ===================================================================


@pytest.mark.asyncio
async def test_get_all_events_empty(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    assert await partition.get_all_events() == []


@pytest.mark.asyncio
async def test_get_all_events_chronological(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    """get_all_events returns every event ordered by timestamp."""
    events = [_event(ts_offset_seconds=i) for i in range(5)]
    # Insert out of chronological order.
    await partition.add_events([events[3], events[0], events[4], events[1], events[2]])

    retrieved = await partition.get_all_events()
    assert retrieved == events


# ===================================================================
# delete_events
# ===================================================================


@pytest.mark.asyncio
async def test_delete_events(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    event = _event()
    await partition.add_events([event])

    await partition.delete_events([event.uuid])

    assert await partition.get_event(event.uuid) is None


@pytest.mark.asyncio
async def test_delete_events_partial(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    """Deleting one event leaves the others intact."""
    keep, drop = _event(ts_offset_seconds=0), _event(ts_offset_seconds=1)
    await partition.add_events([keep, drop])

    await partition.delete_events([drop.uuid])

    assert await partition.get_event(drop.uuid) is None
    assert await partition.get_event(keep.uuid) == keep


@pytest.mark.asyncio
async def test_delete_events_empty(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    await partition.delete_events([])


@pytest.mark.asyncio
async def test_delete_events_unknown(
    partition: SQLAlchemyEventStorePartition,
) -> None:
    await partition.delete_events([uuid4()])


# ===================================================================
# Partition isolation
# ===================================================================


@pytest.mark.asyncio
async def test_partition_isolation(store: SQLAlchemyEventStore) -> None:
    """Events are scoped to their partition."""
    partition_a = await store.open_or_create_partition("partition_a", _config())
    partition_b = await store.open_or_create_partition("partition_b", _config())

    event_a = _event()
    event_b = _event()
    await partition_a.add_events([event_a])
    await partition_b.add_events([event_b])

    assert await partition_a.get_event(event_b.uuid) is None
    assert await partition_b.get_event(event_a.uuid) is None
    assert [e.uuid for e in await partition_a.get_all_events()] == [event_a.uuid]
    assert [e.uuid for e in await partition_b.get_all_events()] == [event_b.uuid]


# ===================================================================
# Partition lifecycle
# ===================================================================


@pytest.mark.asyncio
async def test_create_partition(store: SQLAlchemyEventStore) -> None:
    await store.create_partition("new_partition", _config())
    assert await store.open_partition("new_partition") is not None


@pytest.mark.asyncio
async def test_create_partition_already_exists(store: SQLAlchemyEventStore) -> None:
    await store.create_partition("dup_partition", _config())
    with pytest.raises(EventStorePartitionAlreadyExistsError):
        await store.create_partition("dup_partition", _config())


@pytest.mark.asyncio
async def test_open_partition_nonexistent(store: SQLAlchemyEventStore) -> None:
    assert await store.open_partition("nonexistent") is None


@pytest.mark.asyncio
async def test_open_or_create_partition_creates(store: SQLAlchemyEventStore) -> None:
    partition = await store.open_or_create_partition("fresh", _config())
    assert partition is not None
    assert await store.open_partition("fresh") is not None


@pytest.mark.asyncio
async def test_open_or_create_partition_idempotent(
    store: SQLAlchemyEventStore,
) -> None:
    await store.create_partition("idem", _config())
    partition = await store.open_or_create_partition("idem", _config())
    assert partition is not None


@pytest.mark.asyncio
async def test_open_or_create_partition_defaults_to_plaintext_config(
    store: SQLAlchemyEventStore,
) -> None:
    partition = await store.open_or_create_partition("plaintext_default", _config())
    assert partition.config.payload_codec_config == PlaintextPayloadCodecConfig()


@pytest.mark.asyncio
async def test_delete_partition_removes_data(store: SQLAlchemyEventStore) -> None:
    partition = await store.open_or_create_partition("to_delete", _config())
    await partition.add_events([_event()])

    await store.delete_partition("to_delete")

    assert await store.open_partition("to_delete") is None


@pytest.mark.asyncio
async def test_delete_partition_cascades_events(store: SQLAlchemyEventStore) -> None:
    partition = await store.open_or_create_partition("cascade_test", _config())
    event = _event()
    await partition.add_events([event])

    await store.delete_partition("cascade_test")

    new_partition = await store.open_or_create_partition("cascade_test", _config())
    assert await new_partition.get_all_events() == []


@pytest.mark.asyncio
async def test_delete_partition_idempotent(store: SQLAlchemyEventStore) -> None:
    await store.delete_partition("never_existed")


@pytest.mark.asyncio
async def test_partition_key_validation_invalid_chars(
    store: SQLAlchemyEventStore,
) -> None:
    for bad_key in ("UPPER", "has-hyphen", "has space"):
        with pytest.raises(ValueError, match="invalid characters"):
            await store.create_partition(bad_key, _config())


@pytest.mark.asyncio
async def test_partition_key_validation_too_long(
    store: SQLAlchemyEventStore,
) -> None:
    with pytest.raises(ValueError, match="too long"):
        await store.create_partition("a" * 33, _config())
