"""Lock-necessity tests for the SQLAlchemy segment store (PostgreSQL).

Each test pins one locking property through the public API by staging the
interleaving the lock exists to serialize. The tests use only surface
shared by the current shared-table store and the earlier partitioned
store (plus the `_insert_segments` seam for pausing a writer
mid-transaction), so the same module runs against both generations for
lock-ablation verification.

Coverage, as verified by ablating locks one at a time in both
generations:
- The writer's shared registry-row pin (`_lock_partition_for_write`)
  blocks partition deletion while a write transaction is in flight:
  `test_write_pin_blocks_partition_delete` fails with the pin ablated in
  either generation and passes with it present.
- Deletion's exclusive registry-row pin serializes racing lifecycle
  operations: the churn and concurrent-delete tests fail with it ablated
  from the shared-table store. On the partitioned store these two tests
  fail even with all locks intact -- lifecycle churn deadlocked there by
  construction (create/delete DDL lock cycles through the shared
  parents), which is what the shared-table overhaul removed.
- The ordered segment-row locks in `delete_segments` impose an
  acquisition order no engine guarantees on its own; see
  `test_overlapping_segment_deletes_do_not_deadlock` for why their
  ablation is not observable on PostgreSQL today.
"""

import asyncio
import contextlib
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.episodic_memory.event_memory.data_types import (
    NullContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    BaseSegmentStore,
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)

# Domain outcomes a racing lifecycle operation may legitimately report.
# Anything else escaping the store API (deadlock aborts, integrity
# errors, missing-relation errors) is a locking defect.
_ALLOWED_RACE_ERRORS = (
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfigMismatchError,
)

_TASK_TIMEOUT_SECONDS = 30


def _seg(index: int = 0) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=index,
        offset=0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        block=TextBlock(text="lock"),
        context=NullContext(),
        properties={},
    )


def _links(*segments: Segment) -> dict[Segment, list[UUID]]:
    return {segment: [uuid4()] for segment in segments}


async def _wait_until_blocked_or_done(
    engine: AsyncEngine,
    task: "asyncio.Task[None]",
) -> str:
    """Wait until `task` finishes ("done") or its backend waits on a lock ("blocked").

    Decided by observed database state (pg_stat_activity), not elapsed
    wall-clock time: with correct locking the task enters a lock wait
    within a few round trips, and with a lock ablated it finishes instead.
    """
    deadline = asyncio.get_running_loop().time() + _TASK_TIMEOUT_SECONDS
    while True:
        if task.done():
            return "done"
        async with engine.connect() as connection:
            blocked = (
                await connection.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE wait_event_type = 'Lock' "
                        "AND pid != pg_backend_pid()"
                    )
                )
            ).scalar_one()
        if blocked:
            return "blocked"
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("task neither blocked on a lock nor finished")
        await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def locking_store(
    sqlalchemy_pg_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySegmentStore]:
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_pg_engine)
    )
    await store.startup()
    yield store
    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseSegmentStore.metadata.drop_all)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_pin_blocks_partition_delete(
    locking_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer's shared registry-row pin makes deletion wait out writers.

    Stages the exact interleaving the pin serializes: a write transaction
    that has taken its pin but not yet committed, with a concurrent
    `delete_partition`. Deletion must block until the writer commits;
    without the pin it proceeds immediately and the write lands in a
    partition that no longer exists.
    """
    partition = await locking_store.open_or_create_partition(
        "lk_write_pin", SegmentStorePartitionConfig()
    )

    reached_pause = asyncio.Event()
    release = asyncio.Event()
    original_insert_segments = partition._insert_segments

    async def pausing_insert_segments(session, segments) -> None:
        # Runs inside the write transaction, after the write pin is taken.
        reached_pause.set()
        await release.wait()
        await original_insert_segments(session, segments)

    monkeypatch.setattr(partition, "_insert_segments", pausing_insert_segments)

    writer = asyncio.create_task(partition.add_segments(_links(_seg())))
    deleter = None
    try:
        await asyncio.wait_for(reached_pause.wait(), _TASK_TIMEOUT_SECONDS)
        deleter = asyncio.create_task(locking_store.delete_partition("lk_write_pin"))
        outcome = await _wait_until_blocked_or_done(sqlalchemy_pg_engine, deleter)
        assert outcome == "blocked", (
            "delete_partition completed while a write transaction was in "
            "flight: the writer's registry-row pin is not blocking deletion"
        )
    finally:
        # Unpause the writer even on failure so its open transaction cannot
        # wedge fixture teardown.
        release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(writer, _TASK_TIMEOUT_SECONDS)
        if deleter is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(deleter, _TASK_TIMEOUT_SECONDS)

    await writer
    await deleter
    assert await locking_store.open_partition("lk_write_pin") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_overlapping_segment_deletes_do_not_deadlock(
    locking_store: SQLAlchemySegmentStore,
) -> None:
    """Overlapping concurrent `delete_segments` stay deadlock-free.

    Regression canary rather than a lock-necessity proof: ablating the
    ordered pre-lock does not make this fail on PostgreSQL, whose current
    executor happens to acquire row locks in consistent orders for
    identical DELETE shapes (scalar array probes are sorted; bitmap scans
    lock in TID order). That consistency is engine behavior, not a
    guarantee any database documents, so the pre-lock imposes the order
    deliberately; this test catches the AB/BA cycle wherever an engine or
    plan change ever produces divergent orders.
    """
    partition = await locking_store.open_or_create_partition(
        "lk_row_order", SegmentStorePartitionConfig()
    )

    async def round_trip(rng: random.Random) -> None:
        segments = [_seg(index) for index in range(24)]
        await partition.add_segments(_links(*segments))
        uuids = [segment.uuid for segment in segments]
        forward = uuids[:16]
        backward = [*reversed(uuids[8:])]
        await asyncio.gather(
            partition.delete_segments(forward),
            partition.delete_segments(backward),
        )
        await partition.delete_segments(rng.sample(uuids, len(uuids)))

    rng = random.Random(7)
    for _ in range(10):
        await asyncio.wait_for(round_trip(rng), _TASK_TIMEOUT_SECONDS)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lifecycle_churn_completes_without_database_errors(
    locking_store: SQLAlchemySegmentStore,
) -> None:
    """Concurrent create/open/delete churn never aborts on lock cycles.

    Lifecycle operations of different partitions, and repeated
    create/delete/re-create of the same partitions, must serialize
    through the store's partition-level locking: any database deadlock
    abort or constraint violation escaping the store API is a locking
    defect. Domain errors (already exists, config mismatch) are the only
    legitimate racing outcomes.
    """
    keys = [f"lk_churn_{index}" for index in range(4)]
    config = SegmentStorePartitionConfig()

    async def worker(seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(40):
            key = rng.choice(keys)
            operation = rng.randrange(4)
            try:
                if operation == 0:
                    await locking_store.create_partition(key, config)
                elif operation == 1:
                    await locking_store.open_or_create_partition(key, config)
                elif operation == 2:
                    await locking_store.open_partition(key)
                else:
                    await locking_store.delete_partition(key)
            except _ALLOWED_RACE_ERRORS:
                pass

    await asyncio.wait_for(
        asyncio.gather(*(worker(seed) for seed in range(8))),
        _TASK_TIMEOUT_SECONDS * 4,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_partition_deletes_are_clean(
    locking_store: SQLAlchemySegmentStore,
) -> None:
    """Racing deletions of one partition serialize through the row pin.

    All racers must complete without database errors -- the losers observe
    the winner's deletion instead of double-processing the partition --
    and the partition must be cleanly re-creatable afterwards.
    """
    config = SegmentStorePartitionConfig()
    for _ in range(10):
        await locking_store.create_partition("lk_del_race", config)
        await asyncio.wait_for(
            asyncio.gather(
                *(locking_store.delete_partition("lk_del_race") for _ in range(4))
            ),
            _TASK_TIMEOUT_SECONDS,
        )
        assert await locking_store.open_partition("lk_del_race") is None
    await locking_store.create_partition("lk_del_race", config)
