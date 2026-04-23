"""Simplified SQLAlchemy segment linker — no derivative sharing, no links table."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, cast, override
from uuid import UUID, uuid4

from memmachine_server.episodic_memory.extra_memory.segment_linker.segment_linker import (
    SegmentLinker,
    SegmentLinkerPartition,
)
from pydantic import BaseModel, Field, InstanceOf, JsonValue, TypeAdapter
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    and_,
    case,
    delete,
    func,
    insert,
    literal,
    or_,
    select,
    true,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine.cursor import CursorResult
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    MappedColumn,
    mapped_column,
)
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.episodic_memory.extra_memory.data_types import (
    Block,
    Context,
    Segment,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")
_ContextAdapter = TypeAdapter(Context | None)
_BlockAdapter = TypeAdapter(Block)


class DerivativeState(StrEnum):
    """Lifecycle state of a derivative."""

    ACTIVE = "A"
    PURGING = "P"


class DeletionJobStatus(StrEnum):
    """Lifecycle status of a deletion job."""

    QUEUED = "Q"
    ADDING = "A"
    STAGED = "S"


# ORM models


class BaseSegmentLinker(DeclarativeBase):
    """Base class for segment linker tables."""


class SegmentRow(BaseSegmentLinker):
    """Persisted segment."""

    __tablename__ = "simple_segment_linker_segments"

    partition_key: MappedColumn[str] = mapped_column(String, nullable=False)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    episode_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    index: MappedColumn[int] = mapped_column(Integer, nullable=False)
    offset: MappedColumn[int] = mapped_column(Integer, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    context: MappedColumn[dict[str, JsonValue] | None] = mapped_column(
        _JSON_AUTO, nullable=True
    )
    block: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False
    )

    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deletion_job_uuid: MappedColumn[UUID | None] = mapped_column(Uuid, nullable=True)

    __table_args__ = (
        Index(
            "simple_segment_linker_segments__pk_ep",
            "partition_key",
            "episode_uuid",
        ),
        Index(
            "simple_segment_linker_segments__pk_ts_ep_bk_ix",
            "partition_key",
            "timestamp",
            "episode_uuid",
            "index",
            "offset",
        ),
        Index(
            "simple_segment_linker_segments__pk_ca",
            "partition_key",
            "created_at",
        ),
        Index(
            "simple_segment_linker_segments__dj",
            "deletion_job_uuid",
        ),
    )


class SegmentPropertyRow(BaseSegmentLinker):
    """Property key-value pair for a segment."""

    __tablename__ = "simple_segment_linker_segment_properties"

    segment_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey("simple_segment_linker_segments.uuid", ondelete="CASCADE"),
        primary_key=True,
    )
    key: MappedColumn[str] = mapped_column(String, primary_key=True)
    value_bool: MappedColumn[bool] = mapped_column(Boolean, nullable=True)
    value_int: MappedColumn[int] = mapped_column(Integer, nullable=True)
    value_float: MappedColumn[float] = mapped_column(Float, nullable=True)
    value_str: MappedColumn[str] = mapped_column(String, nullable=True)
    value_datetime: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class DerivativeRow(BaseSegmentLinker):
    """Derivative record — each derivative belongs to exactly one segment."""

    __tablename__ = "simple_segment_linker_derivatives"

    partition_key: MappedColumn[str] = mapped_column(String, nullable=False)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey("simple_segment_linker_segments.uuid", ondelete="CASCADE"),
        nullable=False,
    )
    state: MappedColumn[str] = mapped_column(String(1), nullable=False)

    __table_args__ = (
        Index(
            "simple_segment_linker_derivatives__su",
            "segment_uuid",
        ),
        Index(
            "simple_segment_linker_derivatives__pk_st",
            "partition_key",
            "state",
        ),
    )


class DeletionByEpisodes(BaseModel):
    """Delete segments belonging to specific episodes."""

    type: Literal["by_episodes"] = "by_episodes"
    episode_uuids: list[UUID]


class DeletionAll(BaseModel):
    """Delete all segments in the partition."""

    type: Literal["all"] = "all"


DeletionCriteria = Annotated[
    DeletionByEpisodes | DeletionAll, Field(discriminator="type")
]
_DeletionCriteriaAdapter = TypeAdapter(DeletionCriteria)


class DeletionJobRow(BaseSegmentLinker):
    """A deletion job that logically marks segments for deletion."""

    __tablename__ = "simple_segment_linker_deletion_jobs"

    partition_key: MappedColumn[str] = mapped_column(String, nullable=False)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    status: MappedColumn[str] = mapped_column(
        String(1), nullable=False, server_default=literal(str(DeletionJobStatus.QUEUED))
    )
    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    criteria: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False
    )

    __table_args__ = (
        Index(
            "simple_segment_linker_deletion_jobs__pk_st_ca",
            "partition_key",
            "status",
            "created_at",
        ),
    )


class SQLAlchemySegmentLinkerPartition(SegmentLinkerPartition):
    """SQLAlchemy-backed partition handle (simplified, no derivative sharing)."""

    def __init__(
        self,
        partition_key: str,
        deletion_batch_size: int,
        deletion_interval: float | None,
        create_session: async_sessionmaker[AsyncSession],
        write_lock: asyncio.Lock | None = None,
    ) -> None:
        self._partition_key = partition_key
        self._create_session = create_session
        self._write_lock = write_lock

        self._deletion_batch_size = deletion_batch_size
        self._deletion_interval = deletion_interval
        self._deletion_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

    def _sqlite_write_lock(self) -> AbstractAsyncContextManager[None]:
        if self._write_lock is not None:
            return self._write_lock
        return nullcontext()

    @override
    async def startup(self) -> None:
        if self._deletion_interval is not None:
            self._shutdown_event.clear()
            self._deletion_task = asyncio.create_task(
                self._deletion_finalization_loop()
            )

    @override
    async def shutdown(self) -> None:
        if self._deletion_task is not None:
            self._shutdown_event.set()
            await self._deletion_task
            self._deletion_task = None

    async def _deletion_finalization_loop(self) -> None:
        assert self._deletion_interval is not None
        while True:
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self._deletion_interval
                )
            except TimeoutError:
                pass
            else:
                return

            try:
                await self._process_one_deletion_job()
                await self._finalize_one_staged_deletion_job()
            except Exception:
                logger.exception("Error during deletion processing cycle")

    # Registration

    @override
    async def register_segments(
        self,
        links: Mapping[Segment, Iterable[UUID]],
        *,
        active: Iterable[UUID] | None = None,
    ) -> None:
        links = {
            segment: set(derivative_uuids)
            for segment, derivative_uuids in links.items()
        }

        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            await self._insert_segments(session, links.keys())
            await self._insert_derivatives(session, links)

    async def _insert_segments(
        self,
        session: AsyncSession,
        segments: Iterable[Segment],
    ) -> None:
        segments = set(segments)

        segment_row_values = [
            {
                "uuid": segment.uuid,
                "partition_key": self._partition_key,
                "episode_uuid": segment.episode_uuid,
                "index": segment.index,
                "offset": segment.offset,
                "timestamp": segment.timestamp,
                "context": (
                    segment.context.model_dump(mode="json")
                    if segment.context is not None
                    else None
                ),
                "block": segment.block.model_dump(mode="json"),
            }
            for segment in segments
        ]
        if segment_row_values:
            await session.execute(insert(SegmentRow), segment_row_values)

        property_row_values: list[dict[str, PropertyValue | UUID]] = [
            {
                "segment_uuid": segment.uuid,
                "key": property_key,
                SQLAlchemySegmentLinkerPartition._property_type_column_name(
                    type(property_value)
                ): property_value,
            }
            for segment in segments
            for property_key, property_value in segment.properties.items()
        ]
        if property_row_values:
            await session.execute(insert(SegmentPropertyRow), property_row_values)

    async def _insert_derivatives(
        self,
        session: AsyncSession,
        links: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        derivative_rows = [
            {
                "uuid": derivative_uuid,
                "partition_key": self._partition_key,
                "segment_uuid": segment.uuid,
                "state": DerivativeState.ACTIVE,
            }
            for segment, derivative_uuids in links.items()
            for derivative_uuid in derivative_uuids
        ]
        if derivative_rows:
            await session.execute(insert(DerivativeRow), derivative_rows)

    # Retrieval

    @override
    async def get_segments_by_derivatives(
        self,
        derivative_uuids: Iterable[UUID],
        *,
        limit_per_derivative: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return {}

        if limit_per_derivative is None:
            return await self._get_segments_by_derivatives_all(
                derivative_uuids, property_filter
            )

        return await self._get_segments_by_derivatives_windowed(
            derivative_uuids, limit_per_derivative, property_filter
        )

    async def _get_segments_by_derivatives_all(
        self,
        derivative_uuids: Iterable[UUID],
        property_filter: FilterExpr | None,
    ) -> dict[UUID, list[Segment]]:
        derivative_uuids = set(derivative_uuids)

        query = (
            select(DerivativeRow.uuid, SegmentRow)
            .join(DerivativeRow, DerivativeRow.segment_uuid == SegmentRow.uuid)
            .where(
                SegmentRow.partition_key == self._partition_key,
                DerivativeRow.uuid.in_(derivative_uuids),
            )
            .order_by(
                SegmentRow.timestamp,
                SegmentRow.episode_uuid,
                SegmentRow.index,
                SegmentRow.offset,
            )
        )

        if property_filter is not None:
            query = query.where(
                SQLAlchemySegmentLinkerPartition._compile_property_filter(
                    property_filter
                )
            )

        async with self._create_session() as session:
            not_deleted_filter = self._build_not_deleted_filter(self._partition_key)
            query = query.where(not_deleted_filter)

            segment_rows = (await session.execute(query)).all()
            segment_uuids: set[UUID] = {
                segment_row.uuid for _, segment_row in segment_rows
            }
            properties_by_segments = await self._load_properties_by_segments(
                session, segment_uuids
            )

        segments_by_derivatives: defaultdict[UUID, list[Segment]] = defaultdict(list)
        for derivative_uuid, segment_row in segment_rows:
            segment = SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                segment_row, properties_by_segments[segment_row.uuid]
            )
            segments_by_derivatives[derivative_uuid].append(segment)

        return dict(segments_by_derivatives)

    async def _get_segments_by_derivatives_windowed(
        self,
        derivative_uuids: Iterable[UUID],
        limit_per_derivative: int,
        property_filter: FilterExpr | None,
    ) -> dict[UUID, list[Segment]]:
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return {}

        limit_first = limit_per_derivative // 2
        limit_last = limit_per_derivative - limit_first

        row_number_ascending = (
            func.row_number()
            .over(
                partition_by=DerivativeRow.uuid,
                order_by=[
                    SegmentRow.timestamp,
                    SegmentRow.episode_uuid,
                    SegmentRow.index,
                    SegmentRow.offset,
                ],
            )
            .label("row_number_ascending")
        )
        row_number_descending = (
            func.row_number()
            .over(
                partition_by=DerivativeRow.uuid,
                order_by=[
                    SegmentRow.timestamp.desc(),
                    SegmentRow.episode_uuid.desc(),
                    SegmentRow.index.desc(),
                    SegmentRow.offset.desc(),
                ],
            )
            .label("row_number_descending")
        )

        numbered_query = (
            select(
                DerivativeRow.uuid.label("derivative_uuid"),
                SegmentRow.uuid.label("uuid"),
                SegmentRow.episode_uuid,
                SegmentRow.index,
                SegmentRow.offset,
                SegmentRow.timestamp,
                SegmentRow.context,
                SegmentRow.block,
                row_number_ascending,
                row_number_descending,
            )
            .join(DerivativeRow, DerivativeRow.segment_uuid == SegmentRow.uuid)
            .where(
                SegmentRow.partition_key == self._partition_key,
                DerivativeRow.uuid.in_(derivative_uuids),
            )
        )
        if property_filter is not None:
            numbered_query = numbered_query.where(
                SQLAlchemySegmentLinkerPartition._compile_property_filter(
                    property_filter
                )
            )

        async with self._create_session() as session:
            not_deleted_filter = self._build_not_deleted_filter(self._partition_key)
            numbered_query = numbered_query.where(not_deleted_filter)

            numbered_subquery = numbered_query.subquery()

            limited_query = (
                select(numbered_subquery)
                .where(
                    or_(
                        numbered_subquery.c.row_number_ascending <= limit_first,
                        numbered_subquery.c.row_number_descending <= limit_last,
                    )
                )
                .order_by(
                    numbered_subquery.c.timestamp,
                    numbered_subquery.c.episode_uuid,
                    numbered_subquery.c.index,
                    numbered_subquery.c.offset,
                )
            )

            rows = (await session.execute(limited_query)).all()
            segment_uuids = {row.uuid for row in rows}
            properties_by_segments = await self._load_properties_by_segments(
                session, segment_uuids
            )

        segments_by_derivatives: defaultdict[UUID, list[Segment]] = defaultdict(list)
        for row in rows:
            segment_row = SegmentRow(
                uuid=row.uuid,
                episode_uuid=row.episode_uuid,
                index=row.index,
                offset=row.offset,
                timestamp=row.timestamp,
                context=row.context,
                block=row.block,
            )
            segment = SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                segment_row, properties_by_segments[segment_row.uuid]
            )
            segments_by_derivatives[row.derivative_uuid].append(segment)

        return dict(segments_by_derivatives)

    @override
    async def get_segment_contexts(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_segments: int = 0,
        max_forward_segments: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        seed_segment_uuids = set(seed_segment_uuids)
        if not seed_segment_uuids:
            return {}

        async with self._create_session() as session:
            not_deleted_filter = self._build_not_deleted_filter(self._partition_key)

            seed_segments_query = select(SegmentRow).where(
                SegmentRow.uuid.in_(seed_segment_uuids),
                SegmentRow.partition_key == self._partition_key,
            )
            if property_filter is not None:
                seed_segments_query = seed_segments_query.where(
                    SQLAlchemySegmentLinkerPartition._compile_property_filter(
                        property_filter
                    )
                )
            seed_segments_query = seed_segments_query.where(not_deleted_filter)
            seed_segment_rows = (
                (await session.execute(seed_segments_query)).scalars().all()
            )

            seed_segment_rows_by_uuid: dict[UUID, SegmentRow] = {
                row.uuid: row for row in seed_segment_rows
            }
            if not seed_segment_rows_by_uuid:
                return {}

            if max_backward_segments <= 0 and max_forward_segments <= 0:
                properties_by_segments = await self._load_properties_by_segments(
                    session, list(seed_segment_rows_by_uuid.keys())
                )
                return {
                    seed_segment_uuid: [
                        SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                            seed_segment_row,
                            properties_by_segments[seed_segment_uuid],
                        )
                    ]
                    for seed_segment_uuid, seed_segment_row in seed_segment_rows_by_uuid.items()
                }

            if session.bind.dialect.name != "sqlite":
                context_rows_by_seed = await self._get_context_rows_lateral(
                    session,
                    seed_segment_rows_by_uuid,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                    not_deleted_filter,
                )
            else:
                context_rows_by_seed = await self._get_context_rows_loop(
                    session,
                    seed_segment_rows_by_uuid,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                    not_deleted_filter,
                )

            all_segment_uuids = set(seed_segment_rows_by_uuid.keys())
            for backward_rows, forward_rows in context_rows_by_seed.values():
                for row in backward_rows:
                    all_segment_uuids.add(row.uuid)
                for row in forward_rows:
                    all_segment_uuids.add(row.uuid)

            properties_by_segments = await self._load_properties_by_segments(
                session, all_segment_uuids
            )

            segments_by_seed: dict[UUID, list[Segment]] = {}
            for seed_uuid, seed_row in seed_segment_rows_by_uuid.items():
                backward_rows, forward_rows = context_rows_by_seed.get(
                    seed_uuid, ([], [])
                )
                segments_by_seed[seed_uuid] = [
                    SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                        row, properties_by_segments.get(row.uuid, {})
                    )
                    for row in [*reversed(backward_rows), seed_row, *forward_rows]
                ]

            return segments_by_seed

    async def _get_context_rows_lateral(
        self,
        session: AsyncSession,
        seed_rows_by_uuid: Mapping[UUID, SegmentRow],
        max_backward_segments: int,
        max_forward_segments: int,
        property_filter: FilterExpr | None,
        not_deleted_filter: ColumnElement[bool],
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        seeds_subquery = (
            select(
                SegmentRow.uuid.label("seed_uuid"),
                SegmentRow.timestamp.label("seed_timestamp"),
                SegmentRow.episode_uuid.label("seed_episode_uuid"),
                SegmentRow.index.label("seed_index"),
                SegmentRow.offset.label("seed_offset"),
            )
            .where(SegmentRow.uuid.in_(seed_rows_by_uuid.keys()))
            .subquery("seeds")
        )

        segment_ordering_columns = tuple_(
            SegmentRow.timestamp,
            SegmentRow.episode_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        )
        seed_ordering_columns = tuple_(
            seeds_subquery.c.seed_timestamp,
            seeds_subquery.c.seed_episode_uuid,
            seeds_subquery.c.seed_index,
            seeds_subquery.c.seed_offset,
        )

        partition_key = self._partition_key

        async def get_context_rows_directional(
            range_condition: ColumnElement[bool],
            ordering: Iterable[ColumnElement | InstrumentedAttribute],
            limit: int,
        ) -> dict[UUID, list[SegmentRow]]:
            context_rows_query = (
                select(SegmentRow)
                .where(SegmentRow.partition_key == partition_key, range_condition)
                .order_by(*ordering)
                .limit(limit)
                .correlate(seeds_subquery)
            )
            if property_filter is not None:
                context_rows_query = context_rows_query.where(
                    SQLAlchemySegmentLinkerPartition._compile_property_filter(
                        property_filter
                    )
                )
            context_rows_query = context_rows_query.where(not_deleted_filter)
            lateral_subquery = context_rows_query.subquery().lateral("context")

            seed_context_join_query = select(
                seeds_subquery.c.seed_uuid,
                lateral_subquery.c.uuid,
                lateral_subquery.c.episode_uuid,
                lateral_subquery.c.index,
                lateral_subquery.c.offset,
                lateral_subquery.c.timestamp,
                lateral_subquery.c.context,
                lateral_subquery.c.block,
            ).select_from(seeds_subquery.join(lateral_subquery, true()))

            rows_by_seed: dict[UUID, list[SegmentRow]] = {
                seed_uuid: [] for seed_uuid in seed_rows_by_uuid
            }
            for row in (await session.execute(seed_context_join_query)).all():
                rows_by_seed[row.seed_uuid].append(
                    SegmentRow(
                        uuid=row.uuid,
                        partition_key=partition_key,
                        episode_uuid=row.episode_uuid,
                        index=row.index,
                        offset=row.offset,
                        timestamp=row.timestamp,
                        context=row.context,
                        block=row.block,
                    )
                )
            return rows_by_seed

        chronological_order = [
            SegmentRow.timestamp,
            SegmentRow.episode_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        ]
        reverse_chronological_order = [col.desc() for col in chronological_order]

        backward_rows_by_seed = (
            await get_context_rows_directional(
                segment_ordering_columns < seed_ordering_columns,
                reverse_chronological_order,
                max_backward_segments,
            )
            if max_backward_segments > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        forward_rows_by_seed = (
            await get_context_rows_directional(
                segment_ordering_columns > seed_ordering_columns,
                chronological_order,
                max_forward_segments,
            )
            if max_forward_segments > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        return {
            seed_uuid: (
                backward_rows_by_seed[seed_uuid],
                forward_rows_by_seed[seed_uuid],
            )
            for seed_uuid in seed_rows_by_uuid
        }

    async def _get_context_rows_loop(
        self,
        session: AsyncSession,
        seed_rows_by_uuid: Mapping[UUID, SegmentRow],
        max_backward_segments: int,
        max_forward_segments: int,
        property_filter: FilterExpr | None,
        not_deleted_filter: ColumnElement[bool],
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        context_rows_by_seed: dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]] = {}

        segment_ordering_columns = tuple_(
            SegmentRow.timestamp,
            SegmentRow.episode_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        )

        compiled_property_filter = (
            SQLAlchemySegmentLinkerPartition._compile_property_filter(property_filter)
            if property_filter is not None
            else None
        )

        for seed_uuid, seed_row in seed_rows_by_uuid.items():
            seed_ordering_values = tuple_(
                literal(seed_row.timestamp),
                literal(seed_row.episode_uuid),
                literal(seed_row.index),
                literal(seed_row.offset),
            )

            backward_rows: list[SegmentRow] = []
            if max_backward_segments > 0:
                backward_rows_query = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.partition_key == self._partition_key,
                        segment_ordering_columns < seed_ordering_values,
                    )
                    .order_by(
                        SegmentRow.timestamp.desc(),
                        SegmentRow.episode_uuid.desc(),
                        SegmentRow.index.desc(),
                        SegmentRow.offset.desc(),
                    )
                    .limit(max_backward_segments)
                )
                if compiled_property_filter is not None:
                    backward_rows_query = backward_rows_query.where(
                        compiled_property_filter
                    )
                backward_rows_query = backward_rows_query.where(not_deleted_filter)
                backward_rows = list(
                    (await session.execute(backward_rows_query)).scalars().all()
                )

            forward_rows: list[SegmentRow] = []
            if max_forward_segments > 0:
                forward_rows_query = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.partition_key == self._partition_key,
                        segment_ordering_columns > seed_ordering_values,
                    )
                    .order_by(
                        SegmentRow.timestamp,
                        SegmentRow.episode_uuid,
                        SegmentRow.index,
                        SegmentRow.offset,
                    )
                    .limit(max_forward_segments)
                )
                if compiled_property_filter is not None:
                    forward_rows_query = forward_rows_query.where(
                        compiled_property_filter
                    )
                forward_rows_query = forward_rows_query.where(not_deleted_filter)
                forward_rows = list(
                    (await session.execute(forward_rows_query)).scalars().all()
                )

            context_rows_by_seed[seed_uuid] = (backward_rows, forward_rows)

        return context_rows_by_seed

    # Deletion

    @override
    async def delete_segments_by_episodes(
        self,
        episode_uuids: Iterable[UUID],
    ) -> None:
        episode_uuids = set(episode_uuids)
        if not episode_uuids:
            return
        job_uuid = uuid4()
        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            await session.execute(
                insert(DeletionJobRow).values(
                    uuid=job_uuid,
                    partition_key=self._partition_key,
                    created_at=func.now(),
                    status=DeletionJobStatus.QUEUED,
                    criteria=DeletionByEpisodes(
                        episode_uuids=list(episode_uuids)
                    ).model_dump(mode="json"),
                )
            )
        await self._process_deletion_jobs_through(job_uuid)

    @override
    async def delete_all_segments(self) -> None:
        job_uuid = uuid4()
        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            await session.execute(
                insert(DeletionJobRow).values(
                    uuid=job_uuid,
                    partition_key=self._partition_key,
                    created_at=func.now(),
                    status=DeletionJobStatus.QUEUED,
                    criteria=DeletionAll().model_dump(mode="json"),
                )
            )
        await self._process_deletion_jobs_through(job_uuid)

    async def _process_deletion_jobs_through(self, target_job_uuid: UUID) -> None:
        async with self._create_session() as session:
            target = (
                await session.execute(
                    select(DeletionJobRow.created_at).where(
                        DeletionJobRow.uuid == target_job_uuid
                    )
                )
            ).scalar_one()

        while True:
            deletion_job = await self._get_adding_or_queued_deletion_job(
                created_at_lte=target
            )
            if deletion_job is None:
                break
            await self._stage_deletion_job(deletion_job)
            if deletion_job.uuid == target_job_uuid:
                break

    async def _process_one_deletion_job(self) -> None:
        deletion_job = await self._get_adding_or_queued_deletion_job()
        if deletion_job is not None:
            await self._stage_deletion_job(deletion_job)

    async def _stage_deletion_job(self, deletion_job: DeletionJobRow) -> None:
        while True:
            stamped = await self._stamp_batch_for_deletion_job(deletion_job)
            if stamped <= 0:
                break

        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            await session.execute(
                update(DeletionJobRow)
                .where(
                    DeletionJobRow.uuid == deletion_job.uuid,
                    DeletionJobRow.status != DeletionJobStatus.STAGED,
                )
                .values(status=DeletionJobStatus.STAGED)
            )

    async def _finalize_one_staged_deletion_job(self) -> None:
        deletion_job = await self._get_staged_deletion_job()
        if deletion_job is None:
            return

        while True:
            deleted = await self._delete_stamped_batch_for_deletion_job(deletion_job)
            if deleted <= 0:
                break

        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            await session.execute(
                delete(DeletionJobRow).where(DeletionJobRow.uuid == deletion_job.uuid)
            )

    async def _get_adding_or_queued_deletion_job(
        self,
        *,
        created_at_lte: datetime | None = None,
    ) -> DeletionJobRow | None:
        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            conditions: list[ColumnElement[bool]] = [
                DeletionJobRow.partition_key == self._partition_key,
                DeletionJobRow.status.in_(
                    [DeletionJobStatus.ADDING, DeletionJobStatus.QUEUED]
                ),
            ]
            if created_at_lte is not None:
                conditions.append(DeletionJobRow.created_at <= created_at_lte)

            adding_status_first = case(
                (DeletionJobRow.status == DeletionJobStatus.ADDING, 0),
                else_=1,
            )

            next_deletion_job_query = (
                select(DeletionJobRow)
                .where(*conditions)
                .order_by(
                    adding_status_first,
                    DeletionJobRow.created_at,
                    DeletionJobRow.uuid,
                )
                .limit(1)
            )
            if session.bind.dialect.name != "sqlite":
                next_deletion_job_query = next_deletion_job_query.with_for_update()
            next_deletion_job = (
                (await session.execute(next_deletion_job_query)).scalars().first()
            )
            if (
                next_deletion_job is not None
                and next_deletion_job.status == DeletionJobStatus.QUEUED
            ):
                await session.execute(
                    update(DeletionJobRow)
                    .where(
                        DeletionJobRow.uuid == next_deletion_job.uuid,
                        DeletionJobRow.status == DeletionJobStatus.QUEUED,
                    )
                    .values(status=DeletionJobStatus.ADDING)
                )
                next_deletion_job.status = DeletionJobStatus.ADDING
            return next_deletion_job

    async def _get_staged_deletion_job(self) -> DeletionJobRow | None:
        async with self._create_session() as session:
            staged_deletion_job_query = (
                select(DeletionJobRow)
                .where(
                    DeletionJobRow.partition_key == self._partition_key,
                    DeletionJobRow.status == DeletionJobStatus.STAGED,
                )
                .order_by(DeletionJobRow.created_at, DeletionJobRow.uuid)
                .limit(1)
            )
            return (await session.execute(staged_deletion_job_query)).scalars().first()

    async def _stamp_batch_for_deletion_job(self, deletion_job: DeletionJobRow) -> int:
        criteria = _DeletionCriteriaAdapter.validate_python(deletion_job.criteria)

        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            criteria_conditions: list[ColumnElement[bool]] = [
                SegmentRow.partition_key == self._partition_key,
                SegmentRow.created_at < deletion_job.created_at,
                SegmentRow.deletion_job_uuid.is_(None),
            ]
            match criteria:
                case DeletionByEpisodes(episode_uuids=episode_uuids):
                    criteria_conditions.append(
                        SegmentRow.episode_uuid.in_(episode_uuids)
                    )
                case DeletionAll():
                    pass

            segment_uuids_to_stamp_subquery = (
                select(SegmentRow.uuid)
                .where(*criteria_conditions)
                .limit(self._deletion_batch_size)
            )

            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(SegmentRow)
                    .where(SegmentRow.uuid.in_(segment_uuids_to_stamp_subquery))
                    .values(deletion_job_uuid=deletion_job.uuid)
                ),
            )
            return result.rowcount

    async def _delete_stamped_batch_for_deletion_job(
        self, deletion_job: DeletionJobRow
    ) -> int:
        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            batch = (
                (
                    await session.execute(
                        select(SegmentRow.uuid)
                        .where(SegmentRow.deletion_job_uuid == deletion_job.uuid)
                        .limit(self._deletion_batch_size)
                    )
                )
                .scalars()
                .all()
            )
            if not batch:
                return 0
            await session.execute(delete(SegmentRow).where(SegmentRow.uuid.in_(batch)))
            return len(batch)

    # Garbage collection

    @override
    async def mark_orphaned_derivatives_for_purging(self, limit: int = 1000) -> None:
        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            # Derivatives whose segment is stamped for deletion.
            orphans_query = (
                select(DerivativeRow.uuid)
                .join(SegmentRow, DerivativeRow.segment_uuid == SegmentRow.uuid)
                .where(
                    DerivativeRow.partition_key == self._partition_key,
                    DerivativeRow.state == DerivativeState.ACTIVE,
                    SegmentRow.deletion_job_uuid.is_not(None),
                )
                .limit(limit)
            )

            if session.bind.dialect.name != "sqlite":
                orphans_query = orphans_query.with_for_update(skip_locked=True)
            orphan_uuids = list((await session.execute(orphans_query)).scalars().all())

            if orphan_uuids:
                await session.execute(
                    update(DerivativeRow)
                    .where(DerivativeRow.uuid.in_(orphan_uuids))
                    .values(state=DerivativeState.PURGING)
                )

    @override
    async def get_derivatives_pending_purge(self, limit: int = 1000) -> set[UUID]:
        async with self._create_session() as session:
            return set(
                (
                    await session.execute(
                        select(DerivativeRow.uuid)
                        .where(
                            DerivativeRow.partition_key == self._partition_key,
                            DerivativeRow.state == DerivativeState.PURGING,
                        )
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )

    @override
    async def purge_derivatives(self, derivative_uuids: Iterable[UUID]) -> None:
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return

        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            await session.execute(
                delete(DerivativeRow).where(
                    DerivativeRow.partition_key == self._partition_key,
                    DerivativeRow.uuid.in_(derivative_uuids),
                    DerivativeRow.state == DerivativeState.PURGING,
                )
            )

    # Helpers

    async def _load_properties_by_segments(
        self,
        session: AsyncSession,
        segment_uuids: Iterable[UUID],
    ) -> dict[UUID, dict[str, PropertyValue]]:
        segment_uuids = set(segment_uuids)

        if not segment_uuids:
            return {}

        segment_property_rows = (
            (
                await session.execute(
                    select(SegmentPropertyRow).where(
                        SegmentPropertyRow.segment_uuid.in_(segment_uuids)
                    )
                )
            )
            .scalars()
            .all()
        )
        properties_by_segments: dict[UUID, dict[str, PropertyValue]] = {
            segment_uuid: {} for segment_uuid in segment_uuids
        }
        for segment_property_row in segment_property_rows:
            properties_by_segments[segment_property_row.segment_uuid][
                segment_property_row.key
            ] = SQLAlchemySegmentLinkerPartition._coalesce_property_value(
                segment_property_row
            )
        return properties_by_segments

    _COMPARISON_OPERATORS: ClassVar[
        dict[
            str,
            Callable[
                [InstrumentedAttribute[PropertyValue], PropertyValue],
                ColumnElement[bool],
            ],
        ]
    ] = {
        "=": lambda column, value: column == value,
        "!=": lambda column, value: column != value,
        ">": lambda column, value: column > value,
        "<": lambda column, value: column < value,
        ">=": lambda column, value: column >= value,
        "<=": lambda column, value: column <= value,
    }

    def _build_not_deleted_filter(
        self,
        partition_key: str,
    ) -> ColumnElement[bool]:
        stamped_jobs_subquery = select(DeletionJobRow.uuid).where(
            DeletionJobRow.partition_key == partition_key,
            DeletionJobRow.status == DeletionJobStatus.STAGED,
        )
        return or_(
            SegmentRow.deletion_job_uuid.is_(None),
            SegmentRow.deletion_job_uuid.not_in(stamped_jobs_subquery),
        )

    @staticmethod
    def _compile_property_filter(expr: FilterExpr) -> ColumnElement[bool]:
        if isinstance(expr, Comparison):
            value_column = (
                SQLAlchemySegmentLinkerPartition._property_type_column_attribute(
                    type(expr.value)
                )
            )
            comparison_op = SQLAlchemySegmentLinkerPartition._COMPARISON_OPERATORS[
                expr.op
            ]
            return (
                select(1)
                .select_from(SegmentPropertyRow)
                .where(
                    SegmentPropertyRow.segment_uuid == SegmentRow.uuid,
                    SegmentPropertyRow.key == expr.field,
                    comparison_op(value_column, expr.value),
                )
                .exists()
            )

        if isinstance(expr, In):
            first_value = expr.values[0] if expr.values else None
            value_column = (
                SegmentPropertyRow.value_int
                if isinstance(first_value, int)
                else SegmentPropertyRow.value_str
            )
            return (
                select(1)
                .select_from(SegmentPropertyRow)
                .where(
                    SegmentPropertyRow.segment_uuid == SegmentRow.uuid,
                    SegmentPropertyRow.key == expr.field,
                    value_column.in_(expr.values),
                )
                .exists()
            )

        if isinstance(expr, IsNull):
            return ~(
                select(1)
                .select_from(SegmentPropertyRow)
                .where(
                    SegmentPropertyRow.segment_uuid == SegmentRow.uuid,
                    SegmentPropertyRow.key == expr.field,
                )
                .exists()
            )

        if isinstance(expr, Not):
            return ~SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.expr)

        if isinstance(expr, And):
            return and_(
                SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.left),
                SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.right),
            )

        if isinstance(expr, Or):
            return or_(
                SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.left),
                SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.right),
            )

        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    @staticmethod
    def _property_type_column_attribute(
        property_type: type[PropertyValue],
    ) -> InstrumentedAttribute[PropertyValue]:
        if property_type is bool:
            return SegmentPropertyRow.value_bool
        if property_type is int:
            return SegmentPropertyRow.value_int
        if property_type is float:
            return SegmentPropertyRow.value_float
        if property_type is str:
            return SegmentPropertyRow.value_str
        if property_type is datetime:
            return SegmentPropertyRow.value_datetime
        raise ValueError(f"Unsupported property value type: {property_type!r}")

    @staticmethod
    def _property_type_column_name(property_type: type[PropertyValue]) -> str:
        if property_type is bool:
            return "value_bool"
        if property_type is int:
            return "value_int"
        if property_type is float:
            return "value_float"
        if property_type is str:
            return "value_str"
        if property_type is datetime:
            return "value_datetime"
        raise ValueError(f"Unsupported property value type: {property_type!r}")

    @staticmethod
    def _coalesce_property_value(row: SegmentPropertyRow) -> PropertyValue:
        if row.value_bool is not None:
            return row.value_bool
        if row.value_int is not None:
            return row.value_int
        if row.value_float is not None:
            return row.value_float
        if row.value_str is not None:
            return row.value_str
        if row.value_datetime is not None:
            return row.value_datetime
        raise ValueError(f"Property row for key={row.key!r} has no value set")

    @staticmethod
    def _segment_from_segment_row(
        row: SegmentRow,
        properties: Mapping[str, PropertyValue],
    ) -> Segment:
        context = _ContextAdapter.validate_python(row.context)
        block = _BlockAdapter.validate_python(row.block)
        return Segment(
            uuid=row.uuid,
            episode_uuid=row.episode_uuid,
            index=row.index,
            offset=row.offset,
            timestamp=row.timestamp,
            context=context,
            block=block,
            properties=dict(properties),
        )


class SQLAlchemySegmentLinkerParams(BaseModel):
    """Parameters for constructing a SQLAlchemySegmentLinker."""

    engine: InstanceOf[AsyncEngine] = Field(..., description="Async SQLAlchemy engine")
    deletion_batch_size: int = Field(
        10_000,
        description="Segments deleted per batch",
        gt=0,
    )
    deletion_interval: float | None = Field(
        None,
        description="Seconds between deletion finalization cycles. None disables periodic finalization.",
    )


class SQLAlchemySegmentLinker(SegmentLinker):
    """SQLAlchemy-backed SegmentLinker factory (simplified, no derivative sharing)."""

    def __init__(self, params: SQLAlchemySegmentLinkerParams) -> None:
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._deletion_batch_size = params.deletion_batch_size
        self._deletion_interval = params.deletion_interval

        self._use_write_lock = self._engine.dialect.name == "sqlite"
        self._write_locks: dict[str, asyncio.Lock] = {}

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseSegmentLinker.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    @override
    def get_partition(self, partition_key: str) -> SQLAlchemySegmentLinkerPartition:
        if self._use_write_lock:
            write_lock = self._write_locks.setdefault(partition_key, asyncio.Lock())
        else:
            write_lock = None
        return SQLAlchemySegmentLinkerPartition(
            partition_key=partition_key,
            deletion_batch_size=self._deletion_batch_size,
            deletion_interval=self._deletion_interval,
            create_session=self._create_session,
            write_lock=write_lock,
        )
