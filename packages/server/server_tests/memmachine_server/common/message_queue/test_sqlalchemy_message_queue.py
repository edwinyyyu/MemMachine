"""Tests for :class:`SQLAlchemyMessageQueue` against SQLite."""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.message_queue import (
    MessageQueue,
    SQLAlchemyMessageQueue,
)


@pytest_asyncio.fixture
async def queue(
    sqlite_engine: AsyncEngine,
) -> AsyncIterator[MessageQueue]:
    q = SQLAlchemyMessageQueue(sqlite_engine)
    await q.startup()
    yield q


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_and_list_consumers(queue: MessageQueue) -> None:
    await queue.register_consumer("semantic")
    await queue.register_consumer("episodic")
    consumers = await queue.list_consumers()
    assert set(consumers) == {"semantic", "episodic"}


@pytest.mark.asyncio
async def test_register_same_consumer_twice_is_noop(queue: MessageQueue) -> None:
    await queue.register_consumer("semantic")
    await queue.register_consumer("semantic")
    assert await queue.list_consumers() == ("semantic",)


@pytest.mark.asyncio
async def test_deregister_consumer_removes_it(queue: MessageQueue) -> None:
    await queue.register_consumer("semantic")
    await queue.register_consumer("episodic")
    await queue.deregister_consumer("semantic")
    assert await queue.list_consumers() == ("episodic",)


@pytest.mark.asyncio
async def test_deregister_unknown_consumer_is_noop(queue: MessageQueue) -> None:
    await queue.deregister_consumer("nobody")


# ---------------------------------------------------------------------------
# Produce / list_pending / count_pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_produce_drops_message_when_no_consumers(
    queue: MessageQueue,
) -> None:
    mid = uuid4()
    await queue.produce("p1", mid)

    # Register after produce: no pending entry exists.
    await queue.register_consumer("semantic")
    pending = [m async for m in queue.list_pending(consumer_id="semantic")]
    assert pending == []


@pytest.mark.asyncio
async def test_produce_enqueues_for_all_registered_consumers(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    await queue.register_consumer("episodic")

    m1 = uuid4()
    await queue.produce("p1", m1)

    sem = [m async for m in queue.list_pending(consumer_id="semantic")]
    epi = [m async for m in queue.list_pending(consumer_id="episodic")]
    assert sem == [m1]
    assert epi == [m1]


@pytest.mark.asyncio
async def test_produce_same_message_twice_is_deduped(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    m = uuid4()
    await queue.produce("p1", m)
    await queue.produce("p1", m)
    pending = [x async for x in queue.list_pending(consumer_id="semantic")]
    assert pending == [m]


@pytest.mark.asyncio
async def test_list_pending_filters_by_partition(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    m1 = uuid4()
    m2 = uuid4()
    await queue.produce("p1", m1)
    await queue.produce("p2", m2)

    only_p1 = [
        m
        async for m in queue.list_pending(consumer_id="semantic", partition_ids=["p1"])
    ]
    assert only_p1 == [m1]


@pytest.mark.asyncio
async def test_list_pending_empty_partition_list_returns_empty(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    await queue.produce("p1", uuid4())
    result = [
        m async for m in queue.list_pending(consumer_id="semantic", partition_ids=[])
    ]
    assert result == []


@pytest.mark.asyncio
async def test_list_pending_respects_limit(queue: MessageQueue) -> None:
    await queue.register_consumer("semantic")
    for _ in range(5):
        await queue.produce("p1", uuid4())

    result = [m async for m in queue.list_pending(consumer_id="semantic", limit=2)]
    assert len(result) == 2


@pytest.mark.asyncio
async def test_count_pending_overall_and_by_partition(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    await queue.produce("p1", uuid4())
    await queue.produce("p1", uuid4())
    await queue.produce("p2", uuid4())

    assert await queue.count_pending(consumer_id="semantic") == 3
    assert await queue.count_pending(consumer_id="semantic", partition_ids=["p1"]) == 2
    assert await queue.count_pending(consumer_id="semantic", partition_ids=[]) == 0


# ---------------------------------------------------------------------------
# Ack
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ack_removes_only_this_consumers_pending(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    await queue.register_consumer("episodic")
    m = uuid4()
    await queue.produce("p1", m)

    await queue.ack(consumer_id="semantic", partition_id="p1", message_ids=[m])

    sem = [x async for x in queue.list_pending(consumer_id="semantic")]
    epi = [x async for x in queue.list_pending(consumer_id="episodic")]
    assert sem == []
    assert epi == [m]


@pytest.mark.asyncio
async def test_ack_unknown_message_is_noop(queue: MessageQueue) -> None:
    await queue.register_consumer("semantic")
    await queue.ack(
        consumer_id="semantic",
        partition_id="p1",
        message_ids=[uuid4()],
    )


@pytest.mark.asyncio
async def test_ack_empty_list_is_noop(queue: MessageQueue) -> None:
    await queue.register_consumer("semantic")
    m = uuid4()
    await queue.produce("p1", m)
    await queue.ack(consumer_id="semantic", partition_id="p1", message_ids=[])
    # Message still pending.
    assert await queue.count_pending(consumer_id="semantic") == 1


# ---------------------------------------------------------------------------
# list_partitions_with_pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_partitions_with_pending_distinct(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    await queue.produce("p1", uuid4())
    await queue.produce("p1", uuid4())
    await queue.produce("p2", uuid4())

    result = {
        p async for p in queue.list_partitions_with_pending(consumer_id="semantic")
    }
    assert result == {"p1", "p2"}


@pytest.mark.asyncio
async def test_list_partitions_with_pending_empty_after_ack(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    m = uuid4()
    await queue.produce("p1", m)
    await queue.ack(consumer_id="semantic", partition_id="p1", message_ids=[m])

    result = [
        p async for p in queue.list_partitions_with_pending(consumer_id="semantic")
    ]
    assert result == []


# ---------------------------------------------------------------------------
# Deregister clears pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deregister_consumer_drops_its_pending_rows(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    await queue.register_consumer("episodic")
    m = uuid4()
    await queue.produce("p1", m)

    await queue.deregister_consumer("semantic")

    # Episodic still has it.
    epi = [x async for x in queue.list_pending(consumer_id="episodic")]
    assert epi == [m]
    # Semantic has nothing.
    sem = [x async for x in queue.list_pending(consumer_id="semantic")]
    assert sem == []


# ---------------------------------------------------------------------------
# delete_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_all_clears_consumers_and_pending(
    queue: MessageQueue,
) -> None:
    await queue.register_consumer("semantic")
    await queue.produce("p1", uuid4())

    await queue.delete_all()

    assert await queue.list_consumers() == ()
    # Re-register (table was wiped), produce a new message, still works.
    await queue.register_consumer("semantic")
    m = uuid4()
    await queue.produce("p1", m)
    sem = [x async for x in queue.list_pending(consumer_id="semantic")]
    assert sem == [m]
