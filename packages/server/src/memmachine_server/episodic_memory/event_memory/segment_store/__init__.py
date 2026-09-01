"""Segment store data models and interfaces."""

from .data_types import (
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionHandleStaleError,
    SegmentStorePermanentError,
)
from .segment_store import (
    SegmentStore,
    SegmentStorePartition,
)

__all__ = [
    "SegmentStore",
    "SegmentStorePartition",
    "SegmentStorePartitionAlreadyExistsError",
    "SegmentStorePartitionConfig",
    "SegmentStorePartitionConfigMismatchError",
    "SegmentStorePartitionHandleStaleError",
    "SegmentStorePermanentError",
]
