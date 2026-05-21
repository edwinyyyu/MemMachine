"""
Abstract base class for an event store.

Defines an interface for persisting and retrieving raw events — the canonical
durable log shared by EventMemory and semantic memory.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from uuid import UUID

from memmachine_core.event.data_types import Event
from memmachine_core.event.event_store.data_types import (
    EventStorePartitionConfig,
)


class EventStorePartition(ABC):
    """Partition-scoped handle for an event store."""

    @property
    @abstractmethod
    def config(self) -> EventStorePartitionConfig:
        """The configuration for this partition."""
        raise NotImplementedError

    @abstractmethod
    async def add_events(self, events: Iterable[Event]) -> None:
        """
        Add events to the partition.

        Args:
            events (Iterable[Event]):
                The events to persist.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_event(self, event_uuid: UUID) -> Event | None:
        """
        Get a single event by its UUID.

        Args:
            event_uuid (UUID):
                The UUID of the event to retrieve.

        Returns:
            Event | None:
                The event, or None if it does not exist in this partition.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_events(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, Event]:
        """
        Batch fetch events by their UUIDs.

        Args:
            event_uuids (Iterable[UUID]):
                The UUIDs of the events to retrieve.

        Returns:
            dict[UUID, Event]:
                A mapping from event UUID to event. Missing UUIDs are simply
                absent from the returned mapping.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_all_events(self) -> list[Event]:
        """
        Get every event in the partition, in chronological order.

        Returns:
            list[Event]:
                All events in the partition, ordered by timestamp then UUID.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_events(self, event_uuids: Iterable[UUID]) -> None:
        """
        Delete events given by their UUIDs.

        Args:
            event_uuids (Iterable[UUID]):
                The UUIDs of the events to delete. It is idempotent; unknown
                UUIDs are ignored.
        """
        raise NotImplementedError


class EventStore(ABC):
    """
    Abstract base class for an event store.

    Manages partition-scoped handles.

    Partition keys must match `[a-z0-9_]+`
    (lowercase alphanumeric and underscores only)
    and be at most 32 bytes.
    """

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    @abstractmethod
    async def create_partition(
        self,
        partition_key: str,
        config: EventStorePartitionConfig,
    ) -> None:
        """
        Create a new partition.

        Args:
            partition_key (str):
                The key of the partition.
            config (EventStorePartitionConfig):
                Configuration for the partition.

        Raises:
            EventStorePartitionAlreadyExistsError: If the partition already exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_partition(self, partition_key: str) -> EventStorePartition | None:
        """
        Open a partition-scoped handle for an existing partition.

        Args:
            partition_key (str):
                The key of the partition.

        Returns:
            EventStorePartition | None:
                A partition-scoped handle, or None if the partition does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_or_create_partition(
        self,
        partition_key: str,
        config: EventStorePartitionConfig,
    ) -> EventStorePartition:
        """
        Open the partition if it exists, or create it if it does not.

        Args:
            partition_key (str):
                The key of the partition.
            config (EventStorePartitionConfig):
                Configuration for the partition.

        Returns:
            EventStorePartition:
                A partition-scoped handle.

        Raises:
            EventStorePartitionConfigMismatchError:
                If the partition already exists with a different configuration.
        """
        raise NotImplementedError

    @abstractmethod
    async def close_partition(self, event_store_partition: EventStorePartition) -> None:
        """
        Close a partition-scoped handle.

        Args:
            event_store_partition (EventStorePartition):
                The partition-scoped handle to close.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete a partition.

        This will delete all data in the partition. It is idempotent.

        Args:
            partition_key (str):
                The key of the partition to delete.
        """
        raise NotImplementedError
