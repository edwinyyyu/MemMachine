"""Segment store data models and interfaces."""

from .data_types import (
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    validate_partition_key,
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
    "validate_partition_key",
]
