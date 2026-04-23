"""No-op segment store for benchmarking. All writes are discarded, reads return empty."""

from collections.abc import Iterable, Mapping
from typing import override
from uuid import UUID

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.episodic_memory.extra_memory.data_types import (
    Segment,
)
from memmachine_server.episodic_memory.extra_memory.segment_store.segment_store import (
    SegmentStore,
    SegmentStorePartition,
)


class NoopSegmentStorePartition(SegmentStorePartition):
    @override
    async def add_segments(
        self,
        segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        pass

    @override
    async def get_segment_contexts(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_segments: int = 0,
        max_forward_segments: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        return {}

    @override
    async def get_segment_uuids_by_episode_uuids(
        self,
        episode_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        return {}

    @override
    async def get_derivative_uuids_by_segment_uuids(
        self,
        segment_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        return {}

    @override
    async def delete_segments(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        pass


class NoopSegmentStore(SegmentStore):
    @override
    async def startup(self) -> None:
        pass

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def open_partition(self, partition_key: str) -> SegmentStorePartition:
        return NoopSegmentStorePartition()

    @override
    async def close_partition(
        self, segment_store_partition: SegmentStorePartition
    ) -> None:
        pass
