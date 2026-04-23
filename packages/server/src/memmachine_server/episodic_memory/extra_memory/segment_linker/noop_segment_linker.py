"""No-op segment linker for benchmarking. All writes are discarded, reads return empty."""

from collections.abc import Iterable, Mapping
from typing import override
from uuid import UUID

from memmachine_server.episodic_memory.extra_memory.segment_linker.segment_linker import (
    SegmentLinker,
    SegmentLinkerPartition,
)

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.episodic_memory.extra_memory.data_types import Segment


class NoopSegmentLinkerPartition(SegmentLinkerPartition):
    @override
    async def startup(self) -> None:
        pass

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def register_segments(
        self,
        links: Mapping[Segment, Iterable[UUID]],
        *,
        active: Iterable[UUID] | None = None,
    ) -> set[UUID]:
        all_uuids: set[UUID] = set()
        for derivative_uuids in links.values():
            all_uuids.update(derivative_uuids)
        return all_uuids

    @override
    async def get_segments_by_derivatives(
        self,
        derivative_uuids: Iterable[UUID],
        *,
        limit_per_derivative: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        return {}

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
    async def delete_segments_by_episodes(self, episode_uuids: Iterable[UUID]) -> None:
        pass

    @override
    async def delete_all_segments(self) -> None:
        pass

    @override
    async def mark_orphaned_derivatives_for_purging(self, limit: int = 10_000) -> None:
        pass

    @override
    async def get_derivatives_pending_purge(self, limit: int = 10_000) -> set[UUID]:
        return set()

    @override
    async def purge_derivatives(self, derivative_uuids: Iterable[UUID]) -> None:
        pass


class NoopSegmentLinker(SegmentLinker):
    @override
    async def startup(self) -> None:
        pass

    @override
    async def shutdown(self) -> None:
        pass

    @override
    def get_partition(self, partition_key: str) -> SegmentLinkerPartition:
        return NoopSegmentLinkerPartition()
