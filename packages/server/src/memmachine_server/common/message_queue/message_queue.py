"""Abstract interface for a shared multi-consumer ingestion queue.

:class:`MessageQueue` is a coordination layer used by memory systems
(semantic, episodic, and any future consumers) to process produced
events exactly as many times as there are registered consumers — once
per consumer — without duplicating content into the queue itself.

Model
-----
The queue stores only routing metadata: ``(partition_id, message_id)``
pairs and per-consumer ack state.  Message *content* lives in the
authoritative source store (e.g., the episode store for user events).
Consumers read a pending ``message_id`` from the queue and fetch the
payload from that source store.

Multi-consumer retention
------------------------
Producing a message records it against every *currently-registered*
consumer's pending list.  Acking removes the (consumer, message) pair.
When no registered consumer has the message pending, the message row
is implicitly deleted in the same transaction — there is no explicit
purge.

Consumers registered *after* a message is produced do not see it.
Replay/backfill is out of scope; introduce a separate API if ever
needed.

Deregistering a consumer deletes its ack state, which may free
additional messages for implicit deletion on the next ack by any
remaining consumer (or immediately, depending on the implementation).
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from uuid import UUID


class MessageQueue(ABC):
    """Multi-consumer coordination queue.

    See the module docstring for the retention model.  Implementations
    back this with SQL, Kafka, Redis Streams, or any comparable
    substrate; the ABC does not assume one.
    """

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def startup(self) -> None:
        """Initialize the queue connection and run pending migrations."""
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self) -> None:
        """Release queue resources."""
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self) -> None:
        """Delete every row in every table managed by this queue."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Consumer registration
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def register_consumer(self, consumer_id: str) -> None:
        """Register a consumer.

        Subsequent :meth:`produce` calls will enqueue against this
        consumer's pending list.  Re-registering an existing consumer
        is a no-op.
        """
        raise NotImplementedError

    @abstractmethod
    async def deregister_consumer(self, consumer_id: str) -> None:
        """Remove a consumer.

        Deletes its ack state; messages it had been blocking from
        deletion become eligible for implicit cleanup.  Deregistering
        an unknown consumer is a no-op.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_consumers(self) -> tuple[str, ...]:
        """Return the currently-registered consumer ids."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Producer side
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def produce(self, partition_id: str, message_id: UUID) -> None:
        """Record a new message for every currently-registered consumer.

        Producing with no consumers registered drops the message
        (nothing to enqueue against).  The caller is responsible for
        having produced the message payload to its authoritative store
        before calling this method.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Consumer side
    # ------------------------------------------------------------------ #

    @abstractmethod
    def list_pending(
        self,
        *,
        consumer_id: str,
        partition_ids: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[UUID]:
        """Iterate message ids this consumer has not yet acked."""
        raise NotImplementedError

    @abstractmethod
    async def count_pending(
        self,
        *,
        consumer_id: str,
        partition_ids: Iterable[str] | None = None,
    ) -> int:
        """Count the messages this consumer has not yet acked."""
        raise NotImplementedError

    @abstractmethod
    async def ack(
        self,
        *,
        consumer_id: str,
        partition_id: str,
        message_ids: Iterable[UUID],
    ) -> None:
        """Mark the given messages as processed by this consumer.

        When a message has no remaining consumers with it pending, its
        row is implicitly deleted in the same transaction.  Acking a
        message the consumer has already acked is a no-op.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Scheduling discovery
    # ------------------------------------------------------------------ #

    @abstractmethod
    def list_partitions_with_pending(
        self,
        *,
        consumer_id: str,
    ) -> AsyncIterator[str]:
        """Iterate partitions with at least one pending message for this consumer."""
        raise NotImplementedError
