"""Event store data models and interfaces."""

from .data_types import (
    EventStorePartitionAlreadyExistsError,
    EventStorePartitionConfig,
    EventStorePartitionConfigMismatchError,
)
from .event_store import (
    EventStore,
    EventStorePartition,
)

__all__ = [
    "EventStore",
    "EventStorePartition",
    "EventStorePartitionAlreadyExistsError",
    "EventStorePartitionConfig",
    "EventStorePartitionConfigMismatchError",
]
