"""Tests for SQLAlchemySegmentStore — SQLite (unit) and PostgreSQL (integration)."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, event, func, insert, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.filter.filter_parser import Comparison, parse_filter
from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionHandleStaleError,
    sqlalchemy_segment_store,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    BaseSegmentStore,
    PartitionRow,
    PurgeRow,
    SegmentRow,
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
    SQLAlchemySegmentStorePartition,
)

PARTITION_KEY = "test_partition"
BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_NULL_CONTEXT = NullContext()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(
    *,
    event_uuid: UUID | None = None,
    index: int = 0,
    offset: int = 0,
    ts_offset_seconds: int = 0,
    text: str = "hello",
    context: ProducerContext | NullContext = _NULL_CONTEXT,
    properties: dict | None = None,
) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=event_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=BASE_TIME + timedelta(seconds=ts_offset_seconds),
        block=TextBlock(text=text),
        context=context,
        properties=properties or {},
    )


def _links(*segments: Segment) -> dict[Segment, list[UUID]]:
    """Build a segment-to-derivative-UUIDs mapping with one derivative per segment."""
    return {seg: [uuid4()] for seg in segments}


def _plaintext_partition_config() -> SegmentStorePartitionConfig:
    """Return the default plaintext partition config."""
    return SegmentStorePartitionConfig()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_store(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySegmentStore]:
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_sqlite_engine)
    )
    await store.startup()
    yield store
    async with sqlalchemy_sqlite_engine.begin() as conn:
        await conn.run_sync(BaseSegmentStore.metadata.drop_all)


@pytest_asyncio.fixture
async def pg_store(
    sqlalchemy_pg_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySegmentStore]:
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_pg_engine)
    )
    await store.startup()
    yield store
    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseSegmentStore.metadata.drop_all)


@pytest.fixture(
    params=[
        "sqlite_store",
        pytest.param("pg_store", marks=pytest.mark.integration),
    ],
)
def store(request) -> SQLAlchemySegmentStore:
    return request.getfixturevalue(request.param)


@pytest.fixture
def recorded_statements(
    sqlalchemy_pg_engine: AsyncEngine,
) -> Iterator[list[str]]:
    """Every statement the engine executes, normalized to one line.

    Recording starts at fixture setup; tests clear the list after their own
    setup so assertions cover only the exercised operations.
    """
    statements: list[str] = []

    def _record(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(" ".join(statement.split()))

    event.listen(sqlalchemy_pg_engine.sync_engine, "before_cursor_execute", _record)
    yield statements
    event.remove(sqlalchemy_pg_engine.sync_engine, "before_cursor_execute", _record)


@pytest_asyncio.fixture
async def partition(
    store: SQLAlchemySegmentStore,
) -> SQLAlchemySegmentStorePartition:
    return await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )


# ===================================================================
# add_segments
# ===================================================================


@pytest.mark.asyncio
async def test_add_segments_and_get_contexts(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg(text="a")
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert seg.uuid in result
    assert len(result[seg.uuid]) == 1
    assert result[seg.uuid][0].uuid == seg.uuid


@pytest.mark.asyncio
async def test_add_segments_with_properties(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg(properties={"color": "red", "score": 42})
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    returned = result[seg.uuid][0]
    assert returned.properties == {"color": "red", "score": 42}


@pytest.mark.asyncio
async def test_add_segments_with_producer_context(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ctx = ProducerContext(producer="User")
    seg = _seg(context=ctx)
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert result[seg.uuid][0].context == ctx

    async with partition._create_session() as session:
        row = (
            await session.execute(select(SegmentRow).where(SegmentRow.uuid == seg.uuid))
        ).scalar_one()
    assert json.loads(row.context) == {"context_type": "producer", "producer": "User"}
    assert json.loads(row.block) == {"block_type": "text", "text": "hello"}


@pytest.mark.asyncio
async def test_add_segments_with_no_context(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    await partition.add_segments(_links(seg))

    async with partition._create_session() as session:
        row = (
            await session.execute(select(SegmentRow).where(SegmentRow.uuid == seg.uuid))
        ).scalar_one()
    assert json.loads(row.context) == {"context_type": "null"}

    result = await partition.get_segment_contexts([seg.uuid])
    assert result[seg.uuid][0].context == NullContext()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tz",
    [
        UTC,
        timezone(timedelta(hours=-8)),
        timezone(timedelta(hours=5, minutes=30)),
    ],
)
async def test_timestamp_roundtrips_with_timezone(
    partition: SQLAlchemySegmentStorePartition,
    tz: timezone,
) -> None:
    """A timezone-aware timestamp roundtrips with its instant and offset intact.

    SQLite's DateTime(timezone=True) discards tzinfo and stores the wall-clock
    fields verbatim, so a non-UTC timestamp that is not normalized to UTC before
    writing comes back shifted by its offset. Regression test for that bug.
    """
    ts = datetime(2024, 1, 1, 13, 30, 45, tzinfo=tz)
    seg = Segment(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=ts,
        block=TextBlock(text="tz"),
        context=_NULL_CONTEXT,
        properties={},
    )
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    returned = result[seg.uuid][0].timestamp
    # Aware-datetime equality compares absolute instants.
    assert returned == ts
    # The original UTC offset is reconstructed, not collapsed to UTC.
    assert returned.utcoffset() == ts.utcoffset()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bound",
    [
        "2024-01-01T00:00:30+00:00",
        "2024-01-01T08:00:30+08:00",  # the same instant, named in another zone
        "2023-12-31T16:00:30-08:00",  # and another
    ],
)
async def test_timestamp_filter_compares_instants_not_wall_clocks(
    partition: SQLAlchemySegmentStorePartition,
    bound: str,
) -> None:
    """A datetime bound means an instant, whatever zone it is written in.

    Companion to the write-side normalization: timestamps are stored as the UTC
    instant, so a bound compared with its own offset would be compared on its
    wall-clock digits, and `<= 08:00+08:00` would exclude a row stored at 00:00Z.
    Both backends must agree, which is why this runs against SQLite and Postgres.
    """
    early = _seg(ts_offset_seconds=0)
    late = _seg(ts_offset_seconds=60)
    await partition.add_segments(_links(early, late))

    result = await partition.get_segment_contexts(
        [early.uuid],
        max_forward_segments=5,
        property_filter=parse_filter(f"timestamp <= date('{bound}')"),
    )
    assert [s.uuid for s in result[early.uuid]] == [early.uuid]


@pytest.mark.asyncio
async def test_add_segments_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    await partition.add_segments({})


@pytest.mark.asyncio
async def test_add_multiple_derivatives_per_segment(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    d1, d2 = uuid4(), uuid4()
    await partition.add_segments({seg: [d1, d2]})

    result = await partition.get_derivative_uuids_by_segment_uuids([seg.uuid])
    assert len(result[seg.uuid]) == 2
    assert {d1, d2} == set(result[seg.uuid])


# ===================================================================
# get_segment_contexts
# ===================================================================


@pytest.mark.asyncio
async def test_contexts_empty_seeds(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_segment_contexts([])
    assert result == {}


@pytest.mark.asyncio
async def test_contexts_unknown_seed(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_segment_contexts(
        [uuid4()], max_backward_segments=2, max_forward_segments=2
    )
    assert result == {}


@pytest.mark.asyncio
async def test_contexts_seed_only(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """When max_backward=0 and max_forward=0, return just the seed."""
    seg = _seg()
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert seg.uuid in result
    ctx = result[seg.uuid]
    assert len(ctx) == 1
    assert ctx[0].uuid == seg.uuid


@pytest.mark.asyncio
async def test_contexts_backward(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await partition.add_segments(_links(*segs))

    seed = segs[3]
    result = await partition.get_segment_contexts([seed.uuid], max_backward_segments=2)
    ctx = result[seed.uuid]
    # backward(2) + seed = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [segs[1].uuid, segs[2].uuid, seed.uuid]


@pytest.mark.asyncio
async def test_contexts_forward(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await partition.add_segments(_links(*segs))

    seed = segs[1]
    result = await partition.get_segment_contexts([seed.uuid], max_forward_segments=2)
    ctx = result[seed.uuid]
    # seed + forward(2) = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [seed.uuid, segs[2].uuid, segs[3].uuid]


@pytest.mark.asyncio
async def test_contexts_backward_and_forward(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(7)]
    await partition.add_segments(_links(*segs))

    seed = segs[3]
    result = await partition.get_segment_contexts(
        [seed.uuid], max_backward_segments=2, max_forward_segments=2
    )
    ctx = result[seed.uuid]
    assert len(ctx) == 5
    uuids = [s.uuid for s in ctx]
    assert uuids == [
        segs[1].uuid,
        segs[2].uuid,
        seed.uuid,
        segs[4].uuid,
        segs[5].uuid,
    ]


@pytest.mark.asyncio
async def test_contexts_clamp_at_boundaries(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Requesting more context than available returns what exists."""
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(3)]
    await partition.add_segments(_links(*segs))

    seed = segs[0]
    result = await partition.get_segment_contexts(
        [seed.uuid], max_backward_segments=10, max_forward_segments=10
    )
    ctx = result[seed.uuid]
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [segs[0].uuid, segs[1].uuid, segs[2].uuid]


@pytest.mark.asyncio
async def test_contexts_multiple_seeds(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(10)]
    await partition.add_segments(_links(*segs))

    seed_a, seed_b = segs[2], segs[7]
    result = await partition.get_segment_contexts(
        [seed_a.uuid, seed_b.uuid],
        max_backward_segments=1,
        max_forward_segments=1,
    )
    assert seed_a.uuid in result
    assert seed_b.uuid in result
    ctx_a = [s.uuid for s in result[seed_a.uuid]]
    ctx_b = [s.uuid for s in result[seed_b.uuid]]
    assert ctx_a == [segs[1].uuid, seed_a.uuid, segs[3].uuid]
    assert ctx_b == [segs[6].uuid, seed_b.uuid, segs[8].uuid]


@pytest.mark.asyncio
async def test_contexts_with_properties(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Properties are loaded for seed and context segments."""
    ep = uuid4()
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, properties={"k": "v0"})
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, properties={"k": "v1"})
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, properties={"k": "v2"})
    await partition.add_segments(_links(s0, s1, s2))

    result = await partition.get_segment_contexts(
        [s1.uuid], max_backward_segments=1, max_forward_segments=1
    )
    ctx = result[s1.uuid]
    assert len(ctx) == 3
    assert ctx[0].properties == {"k": "v0"}
    assert ctx[1].properties == {"k": "v1"}
    assert ctx[2].properties == {"k": "v2"}


@pytest.mark.asyncio
async def test_contexts_property_filter(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Property filter excludes context rows that don't match."""
    ep = uuid4()
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, properties={"tag": "a"})
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, properties={"tag": "b"})
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, properties={"tag": "a"})
    s3 = _seg(event_uuid=ep, offset=3, ts_offset_seconds=3, properties={"tag": "a"})
    await partition.add_segments(_links(s0, s1, s2, s3))

    filt = Comparison(field="m.tag", op="=", value="a")
    result = await partition.get_segment_contexts(
        [s2.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
        property_filter=filt,
    )
    ctx = result[s2.uuid]
    uuids = [s.uuid for s in ctx]
    # s1 excluded (tag=b); s0 backward, s2 seed, s3 forward
    assert uuids == [s0.uuid, s2.uuid, s3.uuid]


@pytest.mark.asyncio
async def test_contexts_filter_by_context_producer(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """`context.producer` is not a stored property; the filter just matches nothing.

    Bare names that aren't `timestamp` are looked up as `_<name>` in the JSON
    properties (matching EventMemory's `_to_vector_record_property` convention
    for system fields). `context.producer` becomes `_context.producer`, which
    isn't a stored key on any segment, so the filter returns no contexts.
    """
    ep = uuid4()
    s0 = _seg(
        event_uuid=ep,
        offset=0,
        ts_offset_seconds=0,
        context=ProducerContext(producer="Alice"),
    )
    s1 = _seg(
        event_uuid=ep,
        offset=1,
        ts_offset_seconds=1,
        context=ProducerContext(producer="Bob"),
    )
    s2 = _seg(
        event_uuid=ep,
        offset=2,
        ts_offset_seconds=2,
        context=ProducerContext(producer="Alice"),
    )
    await partition.add_segments(_links(s0, s1, s2))

    filt = Comparison(field="context.producer", op="=", value="Alice")
    contexts = await partition.get_segment_contexts(
        [s0.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
        property_filter=filt,
    )
    assert contexts == {}


@pytest.mark.asyncio
async def test_contexts_filter_by_context_type(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """`context.context_type` is not a stored property; filter matches nothing."""
    ep = uuid4()
    s0 = _seg(
        event_uuid=ep,
        offset=0,
        ts_offset_seconds=0,
        context=ProducerContext(producer="Alice"),
    )
    s1 = _seg(
        event_uuid=ep,
        offset=1,
        ts_offset_seconds=1,
        context=NullContext(),
    )
    s2 = _seg(
        event_uuid=ep,
        offset=2,
        ts_offset_seconds=2,
        context=ProducerContext(producer="Bob"),
    )
    await partition.add_segments(_links(s0, s1, s2))

    filt = Comparison(field="context.context_type", op="=", value="producer")
    contexts = await partition.get_segment_contexts(
        [s0.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
        property_filter=filt,
    )
    assert contexts == {}


@pytest.mark.asyncio
async def test_contexts_session_isolation(store: SQLAlchemySegmentStore) -> None:
    """Context only includes segments from the same partition_key."""
    ep = uuid4()
    s_other = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0)
    s_seed = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1)
    s_after = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2)

    other_partition = await store.open_or_create_partition(
        "other_session",
        _plaintext_partition_config(),
    )
    await other_partition.add_segments(_links(s_other))

    partition = await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
    await partition.add_segments(_links(s_seed, s_after))

    result = await partition.get_segment_contexts(
        [s_seed.uuid], max_backward_segments=5, max_forward_segments=5
    )
    ctx = result[s_seed.uuid]
    uuids = [s.uuid for s in ctx]
    assert s_other.uuid not in uuids
    assert uuids == [s_seed.uuid, s_after.uuid]


@pytest.mark.asyncio
async def test_contexts_chronological_order(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Context segments are returned in chronological order."""
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await partition.add_segments(_links(*segs))

    result = await partition.get_segment_contexts(
        [segs[2].uuid], max_backward_segments=10, max_forward_segments=10
    )
    ctx = result[segs[2].uuid]
    timestamps = [s.timestamp for s in ctx]
    assert timestamps == sorted(timestamps)


@pytest.mark.asyncio
async def test_context_preserved_in_segment_contexts(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Context is preserved when retrieving segment contexts (backward/forward)."""
    ep = uuid4()
    ctx_user = ProducerContext(producer="User")
    ctx_assistant = ProducerContext(producer="Assistant")
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, context=ctx_user)
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, context=ctx_assistant)
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, context=ctx_user)
    await partition.add_segments(_links(s0, s1, s2))

    result = await partition.get_segment_contexts(
        [s1.uuid], max_backward_segments=1, max_forward_segments=1
    )
    ctx = result[s1.uuid]
    assert len(ctx) == 3
    assert ctx[0].context == ctx_user
    assert ctx[1].context == ctx_assistant
    assert ctx[2].context == ctx_user


# ===================================================================
# get_segment_uuids_by_event_uuids
# ===================================================================


@pytest.mark.asyncio
async def test_get_segment_uuids_by_event_uuids(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(3)]
    await partition.add_segments(_links(*segs))

    result = await partition.get_segment_uuids_by_event_uuids([ep])
    assert ep in result
    assert set(result[ep]) == {s.uuid for s in segs}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_event_uuids_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_segment_uuids_by_event_uuids([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_event_uuids_unknown(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_segment_uuids_by_event_uuids([uuid4()])
    assert result == {}


# ===================================================================
# get_derivative_uuids_by_segment_uuids
# ===================================================================


@pytest.mark.asyncio
async def test_get_derivative_uuids_by_segment_uuids(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    d1, d2 = uuid4(), uuid4()
    await partition.add_segments({seg: [d1, d2]})

    result = await partition.get_derivative_uuids_by_segment_uuids([seg.uuid])
    assert seg.uuid in result
    assert set(result[seg.uuid]) == {d1, d2}


@pytest.mark.asyncio
async def test_get_derivative_uuids_by_segment_uuids_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_derivative_uuids_by_segment_uuids([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_derivative_uuids_by_segment_uuids_unknown(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_derivative_uuids_by_segment_uuids([uuid4()])
    assert result == {}


# ===================================================================
# delete_segments
# ===================================================================


@pytest.mark.asyncio
async def test_delete_segments(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    seg = _seg(event_uuid=ep)
    await partition.add_segments(_links(seg))

    await partition.delete_segments([seg.uuid])

    # Segment gone.
    result = await partition.get_segment_contexts([seg.uuid])
    assert result == {}

    # Derivative cascaded.
    deriv_result = await partition.get_derivative_uuids_by_segment_uuids([seg.uuid])
    assert deriv_result == {}


@pytest.mark.asyncio
async def test_delete_segments_noop_unknown(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    await partition.delete_segments([uuid4()])


@pytest.mark.asyncio
async def test_delete_segments_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    await partition.delete_segments([])


@pytest.mark.asyncio
async def test_delete_segments_partial(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Deleting one segment leaves others intact."""
    ep = uuid4()
    s1 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0)
    s2 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1)
    await partition.add_segments(_links(s1, s2))

    await partition.delete_segments([s1.uuid])

    # s1 gone, s2 still there.
    result = await partition.get_segment_contexts([s1.uuid, s2.uuid])
    assert s1.uuid not in result
    assert s2.uuid in result


# ===================================================================
# Concurrency tests
# ===================================================================


async def _get_partition(engine: AsyncEngine) -> SQLAlchemySegmentStorePartition:
    """Create a partition handle that shares the engine."""
    store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    return await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )


@pytest.mark.asyncio
async def test_concurrent_add_disjoint(
    store: SQLAlchemySegmentStore,
) -> None:
    """Concurrent additions with disjoint segments should not interfere."""
    engine = store._engine

    async def add_batch(batch_id: int) -> None:
        part = await _get_partition(engine)
        segs = [
            _seg(ts_offset_seconds=batch_id * 10 + i, text=f"batch{batch_id}-{i}")
            for i in range(5)
        ]
        await part.add_segments(_links(*segs))

    await asyncio.gather(*(add_batch(i) for i in range(10)))

    # Verify all segments were added.
    part = await _get_partition(engine)
    async with part._create_session() as session:
        count = (
            await session.execute(select(func.count()).select_from(SegmentRow))
        ).scalar()
    assert count == 50


@pytest.mark.asyncio
async def test_concurrent_reads_during_writes(
    store: SQLAlchemySegmentStore,
) -> None:
    """Reads should not fail or block indefinitely while writes are happening."""
    engine = store._engine
    partition = await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )

    # Seed some data.
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(10)]
    await partition.add_segments(_links(*segs))

    read_results: list[int] = []

    async def reader() -> None:
        part = await _get_partition(engine)
        for _ in range(5):
            result = await part.get_segment_contexts(
                [segs[5].uuid], max_backward_segments=5, max_forward_segments=5
            )
            if segs[5].uuid in result:
                read_results.append(len(result[segs[5].uuid]))
            await asyncio.sleep(0.01)

    async def writer() -> None:
        part = await _get_partition(engine)
        for i in range(5):
            new_seg = _seg(ts_offset_seconds=100 + i)
            await part.add_segments(_links(new_seg))
            await asyncio.sleep(0.01)

    await asyncio.gather(reader(), reader(), writer())

    assert len(read_results) > 0


@pytest.mark.asyncio
async def test_concurrent_context_reads_during_deletes(
    store: SQLAlchemySegmentStore,
) -> None:
    """get_segment_contexts should not crash if segments are deleted concurrently."""
    engine = store._engine
    partition = await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )

    # Register segments across multiple events.
    events = [uuid4() for _ in range(5)]
    all_segs = [
        _seg(event_uuid=ep, offset=i, ts_offset_seconds=ep_idx * 10 + i)
        for ep_idx, ep in enumerate(events)
        for i in range(4)
    ]
    await partition.add_segments(_links(*all_segs))

    errors: list[Exception] = []

    async def context_reader() -> None:
        part = await _get_partition(engine)
        for seg in all_segs[::3]:
            try:
                await part.get_segment_contexts(
                    [seg.uuid], max_backward_segments=2, max_forward_segments=2
                )
            except Exception as e:
                errors.append(e)
            await asyncio.sleep(0.01)

    async def segment_deleter() -> None:
        part = await _get_partition(engine)
        for ep_idx in range(1, 3):
            ep_segs = all_segs[ep_idx * 4 : (ep_idx + 1) * 4]
            await part.delete_segments([s.uuid for s in ep_segs])
            await asyncio.sleep(0.02)

    await asyncio.gather(context_reader(), segment_deleter())

    assert errors == []


# ===================================================================
# Partition lifecycle
# ===================================================================


@pytest.mark.asyncio
async def test_open_or_create_partition_defaults_to_plaintext_config(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        "plaintext_default",
        _plaintext_partition_config(),
    )
    assert partition.config.payload_codec_config == PlaintextPayloadCodecConfig()


@pytest.mark.asyncio
async def test_create_partition(store: SQLAlchemySegmentStore) -> None:
    await store.create_partition("new_partition", _plaintext_partition_config())
    partition = await store.open_partition("new_partition")
    assert partition is not None


@pytest.mark.asyncio
async def test_create_partition_already_exists(store: SQLAlchemySegmentStore) -> None:
    await store.create_partition("dup_partition", _plaintext_partition_config())
    with pytest.raises(SegmentStorePartitionAlreadyExistsError):
        await store.create_partition("dup_partition", _plaintext_partition_config())


@pytest.mark.asyncio
async def test_open_partition_nonexistent(store: SQLAlchemySegmentStore) -> None:
    result = await store.open_partition("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_open_partition_existing(store: SQLAlchemySegmentStore) -> None:
    await store.create_partition("existing", _plaintext_partition_config())
    partition = await store.open_partition("existing")
    assert partition is not None


@pytest.mark.asyncio
async def test_open_or_create_partition_creates(store: SQLAlchemySegmentStore) -> None:
    partition = await store.open_or_create_partition(
        "fresh",
        _plaintext_partition_config(),
    )
    assert partition is not None
    # Verify it was actually created.
    opened = await store.open_partition("fresh")
    assert opened is not None


@pytest.mark.asyncio
async def test_open_or_create_partition_idempotent(
    store: SQLAlchemySegmentStore,
) -> None:
    await store.create_partition("idem", _plaintext_partition_config())
    partition = await store.open_or_create_partition(
        "idem",
        _plaintext_partition_config(),
    )
    assert partition is not None


@pytest.mark.asyncio
async def test_delete_partition_removes_data(store: SQLAlchemySegmentStore) -> None:
    partition = await store.open_or_create_partition(
        "to_delete",
        _plaintext_partition_config(),
    )
    seg = _seg()
    await partition.add_segments(_links(seg))

    await store.delete_partition("to_delete")

    # Partition no longer exists.
    assert await store.open_partition("to_delete") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_partition_keeps_foreign_key_enforced(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """Deleting one partition must not drop the derivative-link foreign key."""
    keeper = await pg_store.open_or_create_partition(
        "keeper_fk", _plaintext_partition_config()
    )
    await pg_store.open_or_create_partition("doomed_fk", _plaintext_partition_config())

    await pg_store.delete_partition("doomed_fk")

    # A derivative link pointing at no segment must still be rejected in the
    # partition that survived.
    with pytest.raises(IntegrityError):
        async with sqlalchemy_pg_engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO segment_store_dv_ln"
                    " (incarnation, uuid, segment_uuid)"
                    " VALUES (:incarnation, :uuid, :segment_uuid)"
                ),
                {
                    "incarnation": keeper._incarnation,
                    "uuid": uuid4(),
                    "segment_uuid": uuid4(),
                },
            )


@pytest.mark.asyncio
async def test_delete_partition_keeps_other_partitions_cascading(
    store: SQLAlchemySegmentStore,
) -> None:
    """Deleting one partition must not disable the derivative-link cascade."""
    keeper = await store.open_or_create_partition(
        "keeper",
        _plaintext_partition_config(),
    )
    await store.open_or_create_partition(
        "doomed",
        _plaintext_partition_config(),
    )
    segment = _seg()
    derivative_uuid = uuid4()
    await keeper.add_segments({segment: [derivative_uuid]})

    await store.delete_partition("doomed")

    # The foreign key from derivative links to segments must still cascade
    # for the partitions that were not deleted.
    await keeper.delete_segments([segment.uuid])
    assert await keeper.get_derivative_uuids_by_segment_uuids([segment.uuid]) == {}


@pytest.mark.asyncio
async def test_delete_partition_cascades_segments(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        "cascade_test",
        _plaintext_partition_config(),
    )
    seg = _seg()
    d1 = uuid4()
    await partition.add_segments({seg: [d1]})

    await store.delete_partition("cascade_test")

    # Re-create the partition and verify data is gone.
    new_partition = await store.open_or_create_partition(
        "cascade_test",
        _plaintext_partition_config(),
    )
    result = await new_partition.get_segment_contexts([seg.uuid])
    assert result == {}
    deriv_result = await new_partition.get_derivative_uuids_by_segment_uuids([seg.uuid])
    assert deriv_result == {}


@pytest.mark.asyncio
async def test_delete_partition_idempotent(store: SQLAlchemySegmentStore) -> None:
    await store.delete_partition("never_existed")


@pytest.mark.asyncio
async def test_partition_key_validation_invalid_chars(
    store: SQLAlchemySegmentStore,
) -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("UPPER", _plaintext_partition_config())
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("has-hyphen", _plaintext_partition_config())
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("has space", _plaintext_partition_config())


@pytest.mark.asyncio
async def test_partition_key_validation_too_long(
    store: SQLAlchemySegmentStore,
) -> None:
    with pytest.raises(ValueError, match="too long"):
        await store.create_partition("a" * 33, _plaintext_partition_config())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_context_preserved_via_lateral_join(
    pg_store: SQLAlchemySegmentStore,
) -> None:
    """Context is preserved when retrieved via the LATERAL join path (multiple seeds)."""
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
    ep = uuid4()
    ctx_user = ProducerContext(producer="User")
    ctx_assistant = ProducerContext(producer="Assistant")
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, context=ctx_user)
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, context=ctx_assistant)
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, context=ctx_user)
    s3 = _seg(event_uuid=ep, offset=3, ts_offset_seconds=3, context=ctx_assistant)
    s4 = _seg(event_uuid=ep, offset=4, ts_offset_seconds=4, context=ctx_user)
    all_segs = [s0, s1, s2, s3, s4]
    await partition.add_segments(_links(*all_segs))

    # Two seeds exercises the LATERAL join code path.
    result = await partition.get_segment_contexts(
        [s1.uuid, s3.uuid], max_backward_segments=1, max_forward_segments=1
    )

    ctx_a = result[s1.uuid]
    assert len(ctx_a) == 3
    assert ctx_a[0].context == ctx_user
    assert ctx_a[1].context == ctx_assistant
    assert ctx_a[2].context == ctx_user

    ctx_b = result[s3.uuid]
    assert len(ctx_b) == 3
    assert ctx_b[0].context == ctx_user
    assert ctx_b[1].context == ctx_assistant
    assert ctx_b[2].context == ctx_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_mixed_context_types(
    pg_store: SQLAlchemySegmentStore,
) -> None:
    """Different context types (producer, None) round-trip correctly on PG."""
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
    ctx_msg = ProducerContext(producer="User")

    s_msg = _seg(ts_offset_seconds=0, context=ctx_msg)
    s_none = _seg(ts_offset_seconds=1)

    all_segs = [s_msg, s_none]
    await partition.add_segments(_links(*all_segs))

    async with partition._create_session() as session:
        row = (
            await session.execute(
                select(SegmentRow).where(SegmentRow.uuid == s_none.uuid)
            )
        ).scalar_one()
    assert json.loads(row.context) == {"context_type": "null"}

    result = await partition.get_segment_contexts([s_msg.uuid])
    assert result[s_msg.uuid][0].context == ctx_msg

    result = await partition.get_segment_contexts([s_none.uuid])
    assert result[s_none.uuid][0].context == NullContext()


# ===================================================================
# Incarnation fencing and O(1) deletion
# ===================================================================


@pytest.mark.asyncio
async def test_stale_handle_raises_after_delete(
    store: SQLAlchemySegmentStore,
) -> None:
    """A handle held across deletion must fail loudly, not act."""
    partition = await store.open_or_create_partition(
        "fenced", _plaintext_partition_config()
    )
    seg = _seg()
    await partition.add_segments(_links(seg))

    await store.delete_partition("fenced")

    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.add_segments(_links(_seg()))
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.get_segment_contexts([seg.uuid])
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.get_derivative_uuids_by_segment_uuids([seg.uuid])


@pytest.mark.asyncio
async def test_stale_handle_raises_after_recreate(
    store: SQLAlchemySegmentStore,
) -> None:
    """Re-creating the key must not let an old handle act on the successor."""
    old_handle = await store.open_or_create_partition(
        "reborn", _plaintext_partition_config()
    )
    await store.delete_partition("reborn")
    new_handle = await store.open_or_create_partition(
        "reborn", _plaintext_partition_config()
    )

    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await old_handle.add_segments(_links(_seg()))
    await new_handle.add_segments(_links(_seg()))  # the live handle works


@pytest.mark.asyncio
async def test_recreated_partition_is_isolated_from_old_rows(
    store: SQLAlchemySegmentStore,
) -> None:
    """Old-incarnation rows are invisible to the successor before purging."""
    partition = await store.open_or_create_partition(
        "isolated", _plaintext_partition_config()
    )
    seg = _seg()
    await partition.add_segments(_links(seg))
    old_incarnation = partition._incarnation

    await store.delete_partition("isolated")
    successor = await store.open_or_create_partition(
        "isolated", _plaintext_partition_config()
    )

    assert await successor.get_segment_contexts([seg.uuid]) == {}
    # The old rows still physically exist until the purger reclaims them.
    async with successor._create_session() as session:
        remaining = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == old_incarnation)
            )
        ).scalar_one()
    assert remaining == 1


@pytest.mark.asyncio
async def test_purge_reclaims_only_dead_incarnations(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Purging erases queued incarnations and leaves live partitions alone."""
    # Small default slice, so draining takes the caller's loop.
    monkeypatch.setattr(sqlalchemy_segment_store, "_PURGE_SLICE_SEGMENTS", 2)

    live = await store.open_or_create_partition("live_p", _plaintext_partition_config())
    live_seg = _seg()
    await live.add_segments(_links(live_seg))

    doomed = await store.open_or_create_partition(
        "doomed_p", _plaintext_partition_config()
    )
    doomed_incarnation = doomed._incarnation
    await doomed.add_segments(_links(_seg(), _seg(), _seg()))
    await store.delete_partition("doomed_p")

    slices = 0
    while await store.purge_deleted_partitions():
        slices += 1
        assert slices < 10
    assert slices > 0

    async with live._create_session() as session:
        dead_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == doomed_incarnation)
            )
        ).scalar_one()
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeRow))
        ).scalar_one()
    assert dead_rows == 0
    assert queue_depth == 0
    assert (await live.get_segment_contexts([live_seg.uuid]))[live_seg.uuid]


@pytest.mark.asyncio
async def test_purge_respects_max_segments(
    store: SQLAlchemySegmentStore,
) -> None:
    """A bounded purge stops at its bound and reports remaining work."""
    doomed = await store.open_or_create_partition(
        "doomed_p", _plaintext_partition_config()
    )
    doomed_incarnation = doomed._incarnation
    await doomed.add_segments(_links(_seg(), _seg(), _seg()))
    await store.delete_partition("doomed_p")

    assert await store.purge_deleted_partitions(max_segments=2) is True

    async with doomed._create_session() as session:
        dead_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == doomed_incarnation)
            )
        ).scalar_one()
    assert dead_rows == 1

    assert await store.purge_deleted_partitions() is False
    async with doomed._create_session() as session:
        dead_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == doomed_incarnation)
            )
        ).scalar_one()
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeRow))
        ).scalar_one()
    assert dead_rows == 0
    assert queue_depth == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_landing_during_delete_is_never_orphaned(
    pg_store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write racing a delete can never leave untracked rows behind.

    The writer's registry-row pin forces delete_partition to wait, so any
    write that takes its pin commits before the incarnation is enqueued
    for purging -- its rows are always covered by the queue entry. Without
    the pin, the delete (and a purge) can complete while the write is in
    flight, and the write then lands rows under an incarnation the queue
    no longer tracks: garbage no purge will ever reclaim.
    """
    partition = await pg_store.open_or_create_partition(
        "orphan_race", _plaintext_partition_config()
    )
    incarnation = partition._incarnation

    reached_pause = asyncio.Event()
    release = asyncio.Event()
    original_insert_segments = partition._insert_segments

    async def pausing_insert_segments(session, segments) -> None:
        reached_pause.set()
        await release.wait()
        await original_insert_segments(session, segments)

    monkeypatch.setattr(partition, "_insert_segments", pausing_insert_segments)

    writer = asyncio.create_task(partition.add_segments(_links(_seg())))
    await asyncio.wait_for(reached_pause.wait(), 30)
    deleter = asyncio.create_task(pg_store.delete_partition("orphan_race"))
    await asyncio.sleep(0.4)
    if deleter.done():
        # Broken-locking world: the delete did not wait for the writer.
        # A purge in this window retires the queue entry before the
        # write lands.
        await pg_store.purge_deleted_partitions()

    release.set()
    with contextlib.suppress(SegmentStorePartitionHandleStaleError):
        await asyncio.wait_for(writer, 30)
    await asyncio.wait_for(deleter, 30)

    assert await pg_store.purge_deleted_partitions() is False
    async with partition._create_session() as session:
        leftover_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == incarnation)
            )
        ).scalar_one()
    assert leftover_rows == 0, (
        "rows landed under an incarnation the purge queue no longer "
        "tracks: the writer's registry-row pin is not draining writes "
        "before deletion commits"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_remote_delete_yields_single_queue_entry(
    pg_store: SQLAlchemySegmentStore,
) -> None:
    """Racing deletions enqueue a dead incarnation exactly once.

    Emulates a delete_partition from another process at the wire level
    (exclusive row pin, queue insert, registry delete, held uncommitted)
    and races the store's delete_partition against it: the store's delete
    must block on the row pin and, once the remote transaction commits,
    observe the registry row gone and no-op -- one queue entry, no
    integrity error surfacing to the caller.
    """
    partition = await pg_store.open_or_create_partition(
        "remote_del", _plaintext_partition_config()
    )
    incarnation = partition._incarnation

    async with partition._create_session() as remote_session:
        async with remote_session.begin():
            await remote_session.execute(
                select(PartitionRow.incarnation)
                .where(PartitionRow.partition_key == "remote_del")
                .with_for_update()
            )
            await remote_session.execute(
                insert(PurgeRow).values(
                    incarnation=incarnation,
                    partition_key="remote_del",
                    enqueued_at=datetime.now(UTC),
                )
            )
            await remote_session.execute(
                delete(PartitionRow).where(PartitionRow.partition_key == "remote_del")
            )

            local_delete = asyncio.create_task(pg_store.delete_partition("remote_del"))
            await asyncio.sleep(0.4)
            assert not local_delete.done(), (
                "delete_partition proceeded despite a concurrent deletion "
                "holding the exclusive registry-row pin"
            )
        # Remote transaction committed on exiting begin().
        await asyncio.wait_for(local_delete, 30)

    async with partition._create_session() as session:
        queue_entries = (
            await session.execute(select(func.count()).select_from(PurgeRow))
        ).scalar_one()
    assert queue_entries == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_partition_touches_only_registry_and_queue(
    pg_store: SQLAlchemySegmentStore,
    recorded_statements: list[str],
) -> None:
    """Deletion is O(1): no data-table statements, regardless of size."""
    partition = await pg_store.open_or_create_partition(
        "big_delete", _plaintext_partition_config()
    )
    await partition.add_segments(
        _links(*(_seg(ts_offset_seconds=i) for i in range(20)))
    )
    recorded_statements.clear()

    await pg_store.delete_partition("big_delete")

    touching_data = [
        s
        for s in recorded_statements
        if "segment_store_sg" in s or "segment_store_dv_ln" in s
    ]
    assert not touching_data, touching_data
    assert any("segment_store_gc" in s for s in recorded_statements)
