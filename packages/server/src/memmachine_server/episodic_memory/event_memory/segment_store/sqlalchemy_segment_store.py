"""SQLAlchemy implementation of the SegmentStore interface."""

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta, timezone
from typing import override
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    InstanceOf,
    JsonValue,
    field_validator,
)
from sqlalchemy import (
    JSON,
    CursorResult,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Uuid,
    delete,
    event,
    insert,
    literal,
    select,
    true,
    tuple_,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    MappedColumn,
    mapped_column,
)
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    normalize_filter_field,
)
from memmachine_server.common.filter.sql_filter_util import (
    FieldEncoding,
    compile_sql_filter,
)
from memmachine_server.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
)
from memmachine_server.common.payload_codec import PayloadCodec
from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
    decode_payload_codec_config,
    encode_payload_codec_config,
)
from memmachine_server.common.payload_codec.plaintext_payload_codec import (
    PlaintextPayloadCodec,
)
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)
from memmachine_server.common.utils import ensure_tz_aware, utc_offset_seconds
from memmachine_server.episodic_memory.event_memory.data_types import (
    NullContext,
    Segment,
    decode_block,
    decode_context,
    encode_block,
    encode_context,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionStaleError,
)
from memmachine_server.episodic_memory.event_memory.segment_store.segment_store import (
    SegmentStore,
    SegmentStorePartition,
)
from memmachine_server.episodic_memory.event_memory.segment_store.utils import (
    new_incarnation,
    validate_partition_key,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# ORM models


class BaseSegmentStore(DeclarativeBase):
    """Base class for segment store tables."""


class PartitionRow(BaseSegmentStore):
    """The tenant registry: one row per live partition incarnation."""

    __tablename__ = "segment_store_pt"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)
    payload_codec_config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO,
        nullable=False,
    )


class SegmentRow(BaseSegmentStore):
    """Persisted segment."""

    __tablename__ = "segment_store_sg"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    event_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    index: MappedColumn[int] = mapped_column(Integer, nullable=False)
    offset: MappedColumn[int] = mapped_column(Integer, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    timestamp_timezone_offset: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    context: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    block: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    properties: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False, default=dict
    )

    # No foreign key to the registry: registry rows and data rows are
    # deliberately decoupled so that partition deletion is a registry write
    # (O(1)) and the purge queue reclaims data rows asynchronously.
    __table_args__ = (
        Index(
            "segment_store_sg__inc_ev",
            "incarnation",
            "event_uuid",
        ),
        Index(
            "segment_store_sg__inc_ts_ev_bk_ix",
            "incarnation",
            "timestamp",
            "event_uuid",
            "index",
            "offset",
        ),
    )


class DerivativeLinkRow(BaseSegmentStore):
    """Maps a derivative UUID to its owning segment."""

    __tablename__ = "segment_store_dv_ln"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["incarnation", "segment_uuid"],
            [
                "segment_store_sg.incarnation",
                "segment_store_sg.uuid",
            ],
            ondelete="CASCADE",
        ),
        Index(
            "segment_store_dv_ln__inc_su",
            "incarnation",
            "segment_uuid",
        ),
    )


class PurgeRow(BaseSegmentStore):
    """The purge queue: one row per dead partition incarnation.

    Carries the logical key purely for forensics; the incarnation alone
    identifies the rows to reclaim.
    """

    __tablename__ = "segment_store_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    partition_key: MappedColumn[str] = mapped_column(String(255), nullable=False)
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class SQLAlchemySegmentStorePartition(SegmentStorePartition):
    """SQLAlchemy-backed partition handle."""

    def __init__(
        self,
        partition_key: str,
        incarnation: UUID,
        engine: AsyncEngine,
        config: SegmentStorePartitionConfig,
        payload_codec: PayloadCodec,
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a partition key, its incarnation, and an engine."""
        self._logical_partition_key = partition_key
        # Data rows are keyed by the incarnation alone: rows of a
        # deleted-and-recreated partition under the same logical key are
        # invisible to the new incarnation while the purge queue reclaims
        # them, and a data query cannot even be built without resolving
        # the registry first.
        self._incarnation = incarnation
        self._engine = engine
        self._config = config
        self._payload_codec = payload_codec
        self._tracker = tracker
        self._create_session = async_sessionmaker(engine, expire_on_commit=False)
        self._is_sqlite = engine.dialect.name == "sqlite"

    @override
    @property
    def config(self) -> SegmentStorePartitionConfig:
        return self._config

    async def _lock_partition_for_write(self, session: AsyncSession) -> None:
        """Pin this incarnation's registry row; raise if the handle is stale.

        The shared row lock blocks concurrent deletion (which takes the
        exclusive row lock) until the write completes; the incarnation
        predicate fences a handle that outlived its partition. SQLite
        ignores the locking clause and relies on database-level write
        serialization; the fencing check still applies.
        """
        row = (
            await session.execute(
                select(PartitionRow.partition_key)
                .where(PartitionRow.incarnation == self._incarnation)
                .with_for_update(read=True)
            )
        ).scalar_one_or_none()
        if row is None:
            raise SegmentStorePartitionStaleError(self._logical_partition_key)

    async def _ensure_partition_live(self, session: AsyncSession) -> None:
        """Raise if this handle's incarnation is no longer registered."""
        row = (
            await session.execute(
                select(PartitionRow.partition_key).where(
                    PartitionRow.incarnation == self._incarnation
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise SegmentStorePartitionStaleError(self._logical_partition_key)

    # Registration

    @override
    async def add_segments(
        self,
        segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        async with (
            self._tracker("add_segments"),
            self._create_session() as session,
            session.begin(),
        ):
            await self._lock_partition_for_write(session)
            await self._insert_segments(session, segments_to_derivative_uuids.keys())
            await self._insert_derivative_links(session, segments_to_derivative_uuids)

    async def _insert_segments(
        self,
        session: AsyncSession,
        segments: Iterable[Segment],
    ) -> None:
        """Insert segment rows."""
        segment_row_values = [
            {
                "uuid": segment.uuid,
                "incarnation": self._incarnation,
                "event_uuid": segment.event_uuid,
                "index": segment.index,
                "offset": segment.offset,
                "timestamp": ensure_tz_aware(segment.timestamp),
                "timestamp_timezone_offset": utc_offset_seconds(segment.timestamp),
                "context": self._payload_codec.encode(
                    json.dumps(encode_context(segment.context)).encode("utf-8")
                ),
                "block": self._payload_codec.encode(
                    json.dumps(encode_block(segment.block)).encode("utf-8")
                ),
                "properties": encode_properties(segment.properties),
            }
            for segment in segments
        ]
        if segment_row_values:
            await session.execute(insert(SegmentRow), segment_row_values)

    async def _insert_derivative_links(
        self,
        session: AsyncSession,
        segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        """Insert derivative rows."""
        derivative_row_values = [
            {
                "uuid": derivative_uuid,
                "incarnation": self._incarnation,
                "segment_uuid": segment.uuid,
            }
            for segment, derivative_uuids in segments_to_derivative_uuids.items()
            for derivative_uuid in derivative_uuids
        ]
        if derivative_row_values:
            await session.execute(insert(DerivativeLinkRow), derivative_row_values)

    # Retrieval

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

        async with (
            self._tracker("get_segment_contexts"),
            self._create_session() as session,
        ):
            await self._ensure_partition_live(session)
            seed_segments_query = select(SegmentRow).where(
                SegmentRow.uuid.in_(seed_segment_uuids),
                SegmentRow.incarnation == self._incarnation,
            )
            if property_filter is not None:
                seed_segments_query = seed_segments_query.where(
                    compile_sql_filter(
                        property_filter,
                        self._resolve_segment_field,
                    )
                )
            seed_segment_rows = (
                (await session.execute(seed_segments_query)).scalars().all()
            )

            seed_segment_rows_by_uuid: dict[UUID, SegmentRow] = {
                row.uuid: row for row in seed_segment_rows
            }
            if not seed_segment_rows_by_uuid:
                return {}

            # Short-circuit: no context needed.
            if max_backward_segments <= 0 and max_forward_segments <= 0:
                return {
                    seed_segment_uuid: [
                        self._segment_from_segment_row(
                            seed_segment_row,
                        )
                    ]
                    for seed_segment_uuid, seed_segment_row in seed_segment_rows_by_uuid.items()
                }

            # Get backward/forward context rows.
            if session.bind.dialect.name != "sqlite":
                context_rows_by_seed = await self._get_context_rows_lateral(
                    session,
                    seed_segment_rows_by_uuid,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                )
            else:
                context_rows_by_seed = await self._get_context_rows_loop(
                    session,
                    seed_segment_rows_by_uuid,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                )

            # Assemble results: [backward (reversed) + seed + forward].
            segments_by_seed: dict[UUID, list[Segment]] = {}
            for seed_uuid, seed_row in seed_segment_rows_by_uuid.items():
                backward_rows, forward_rows = context_rows_by_seed.get(
                    seed_uuid, ([], [])
                )
                segments_by_seed[seed_uuid] = [
                    self._segment_from_segment_row(row)
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
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Get backward/forward context using LATERAL joins (non-SQLite)."""
        seeds_subquery = (
            select(
                SegmentRow.uuid.label("seed_uuid"),
                SegmentRow.timestamp.label("seed_timestamp"),
                SegmentRow.event_uuid.label("seed_event_uuid"),
                SegmentRow.index.label("seed_index"),
                SegmentRow.offset.label("seed_offset"),
            )
            .where(
                SegmentRow.incarnation == self._incarnation,
                SegmentRow.uuid.in_(seed_rows_by_uuid.keys()),
            )
            .subquery("seeds")
        )

        segment_ordering_columns = tuple_(
            SegmentRow.timestamp,
            SegmentRow.event_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        )
        seed_ordering_columns = tuple_(
            seeds_subquery.c.seed_timestamp,
            seeds_subquery.c.seed_event_uuid,
            seeds_subquery.c.seed_index,
            seeds_subquery.c.seed_offset,
        )

        incarnation = self._incarnation

        async def get_context_rows_directional(
            range_condition: ColumnElement[bool],
            ordering: Iterable[ColumnElement | InstrumentedAttribute],
            limit: int,
        ) -> dict[UUID, list[SegmentRow]]:
            """Get context rows per seed in the specified direction."""
            # Build a LATERAL subquery that gets context rows for each seed.
            context_rows_query = (
                select(SegmentRow)
                .where(SegmentRow.incarnation == incarnation, range_condition)
                .order_by(*ordering)
                .limit(limit)
                .correlate(seeds_subquery)
            )
            if property_filter is not None:
                context_rows_query = context_rows_query.where(
                    compile_sql_filter(
                        property_filter,
                        self._resolve_segment_field,
                    )
                )
            lateral_subquery = context_rows_query.subquery().lateral("context")

            # Join each seed to its context rows via the LATERAL subquery.
            seed_context_join_query = select(
                seeds_subquery.c.seed_uuid,
                lateral_subquery.c.uuid,
                lateral_subquery.c.event_uuid,
                lateral_subquery.c.index,
                lateral_subquery.c.offset,
                lateral_subquery.c.timestamp,
                lateral_subquery.c.timestamp_timezone_offset,
                lateral_subquery.c.context,
                lateral_subquery.c.block,
                lateral_subquery.c.properties,
            ).select_from(seeds_subquery.join(lateral_subquery, true()))

            # Group result rows by seed UUID.
            rows_by_seed: dict[UUID, list[SegmentRow]] = {
                seed_uuid: [] for seed_uuid in seed_rows_by_uuid
            }
            for row in (await session.execute(seed_context_join_query)).all():
                rows_by_seed[row.seed_uuid].append(
                    SegmentRow(
                        uuid=row.uuid,
                        incarnation=incarnation,
                        event_uuid=row.event_uuid,
                        index=row.index,
                        offset=row.offset,
                        timestamp=row.timestamp,
                        timestamp_timezone_offset=row.timestamp_timezone_offset,
                        context=row.context,
                        block=row.block,
                        properties=row.properties,
                    )
                )
            return rows_by_seed

        chronological_order = [
            SegmentRow.timestamp,
            SegmentRow.event_uuid,
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
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Get backward/forward context per seed (SQLite fallback)."""
        context_rows_by_seed: dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]] = {}

        segment_ordering_columns = tuple_(
            SegmentRow.timestamp,
            SegmentRow.event_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        )

        compiled_property_filter = (
            compile_sql_filter(
                property_filter,
                self._resolve_segment_field,
            )
            if property_filter is not None
            else None
        )

        for seed_uuid, seed_row in seed_rows_by_uuid.items():
            seed_ordering_values = tuple_(
                literal(seed_row.timestamp),
                literal(seed_row.event_uuid),
                literal(seed_row.index),
                literal(seed_row.offset),
            )

            backward_rows: list[SegmentRow] = []
            if max_backward_segments > 0:
                backward_rows_query = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.incarnation == self._incarnation,
                        segment_ordering_columns < seed_ordering_values,
                    )
                    .order_by(
                        SegmentRow.timestamp.desc(),
                        SegmentRow.event_uuid.desc(),
                        SegmentRow.index.desc(),
                        SegmentRow.offset.desc(),
                    )
                    .limit(max_backward_segments)
                )
                if compiled_property_filter is not None:
                    backward_rows_query = backward_rows_query.where(
                        compiled_property_filter
                    )
                backward_rows = list(
                    (await session.execute(backward_rows_query)).scalars().all()
                )

            forward_rows: list[SegmentRow] = []
            if max_forward_segments > 0:
                forward_rows_query = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.incarnation == self._incarnation,
                        segment_ordering_columns > seed_ordering_values,
                    )
                    .order_by(
                        SegmentRow.timestamp,
                        SegmentRow.event_uuid,
                        SegmentRow.index,
                        SegmentRow.offset,
                    )
                    .limit(max_forward_segments)
                )
                if compiled_property_filter is not None:
                    forward_rows_query = forward_rows_query.where(
                        compiled_property_filter
                    )
                forward_rows = list(
                    (await session.execute(forward_rows_query)).scalars().all()
                )

            context_rows_by_seed[seed_uuid] = (backward_rows, forward_rows)

        return context_rows_by_seed

    @override
    async def get_segment_uuids_by_event_uuids(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        event_uuids = set(event_uuids)
        if not event_uuids:
            return {}

        async with (
            self._tracker("get_segment_uuids_by_event_uuids"),
            self._create_session() as session,
        ):
            await self._ensure_partition_live(session)
            query = select(SegmentRow.event_uuid, SegmentRow.uuid).where(
                SegmentRow.incarnation == self._incarnation,
                SegmentRow.event_uuid.in_(event_uuids),
            )
            rows = (await session.execute(query)).all()

        result: defaultdict[UUID, list[UUID]] = defaultdict(list)
        for event_uuid, segment_uuid in rows:
            result[event_uuid].append(segment_uuid)
        return dict(result)

    @override
    async def get_derivative_uuids_by_segment_uuids(
        self,
        segment_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        segment_uuids = set(segment_uuids)
        if not segment_uuids:
            return {}

        async with (
            self._tracker("get_derivative_uuids_by_segment_uuids"),
            self._create_session() as session,
        ):
            await self._ensure_partition_live(session)
            query = select(
                DerivativeLinkRow.segment_uuid, DerivativeLinkRow.uuid
            ).where(
                DerivativeLinkRow.incarnation == self._incarnation,
                DerivativeLinkRow.segment_uuid.in_(segment_uuids),
            )
            rows = (await session.execute(query)).all()

        result: defaultdict[UUID, list[UUID]] = defaultdict(list)
        for segment_uuid, derivative_uuid in rows:
            result[segment_uuid].append(derivative_uuid)
        return dict(result)

    # Deletion

    @override
    async def delete_segments(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        segment_uuids = set(segment_uuids)
        if not segment_uuids:
            return

        async with (
            self._tracker("delete_segments"),
            self._create_session() as session,
            session.begin(),
        ):
            await self._lock_partition_for_write(session)
            if not self._is_sqlite:
                # Lock rows in deterministic order to prevent deadlocks
                # from concurrent deletions with overlapping UUID sets.
                # SQLite relies on write serialization by the database.
                await session.execute(
                    select(SegmentRow.uuid)
                    .where(
                        SegmentRow.incarnation == self._incarnation,
                        SegmentRow.uuid.in_(segment_uuids),
                    )
                    .order_by(SegmentRow.uuid)
                    .with_for_update()
                )

            # CASCADE deletes derivatives via FK.
            await session.execute(
                delete(SegmentRow).where(
                    SegmentRow.incarnation == self._incarnation,
                    SegmentRow.uuid.in_(segment_uuids),
                )
            )

    # Helpers

    def _resolve_segment_field(
        self,
        field: str,
    ) -> tuple[ColumnElement, FieldEncoding]:
        """Map a filter field name to a segment column and encoding."""
        if field == "timestamp":
            return SegmentRow.timestamp.expression, "column"
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            key = demangle_user_metadata_key(internal_name)
            return SegmentRow.properties[key], "properties_json"
        return SegmentRow.properties[f"_{field}"], "properties_json"

    def _segment_from_segment_row(self, row: SegmentRow) -> Segment:
        """Convert a SegmentRow into a Segment."""
        context = decode_context(json.loads(self._payload_codec.decode(row.context)))
        if context is None:
            context = NullContext()
        block = decode_block(json.loads(self._payload_codec.decode(row.block)))
        properties = decode_properties(row.properties)
        original_timezone = timezone(timedelta(seconds=row.timestamp_timezone_offset))
        timestamp = ensure_tz_aware(row.timestamp).astimezone(original_timezone)
        return Segment(
            uuid=row.uuid,
            event_uuid=row.event_uuid,
            index=row.index,
            offset=row.offset,
            timestamp=timestamp,
            context=context,
            block=block,
            properties=properties,
        )


class SQLAlchemySegmentStoreParams(BaseModel):
    """
    Parameters for constructing a SQLAlchemySegmentStore.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    engine: InstanceOf[AsyncEngine] = Field(..., description="Async SQLAlchemy engine")
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, engine: AsyncEngine) -> AsyncEngine:
        assert not isinstance(engine.pool, StaticPool), (
            "Engine uses StaticPool, which shares one connection across sessions. "
            "Use a multi-connection pool instead."
        )
        db = engine.url.database
        if engine.dialect.name == "sqlite" and (db is None or db == ":memory:"):
            raise ValueError(
                "Engine uses ephemeral SQLite, where each connection gets a separate database. "
                "Use a file path instead."
            )
        return engine


class SQLAlchemySegmentStore(SegmentStore):
    """SQLAlchemy-backed SegmentStore factory."""

    def __init__(self, params: SQLAlchemySegmentStoreParams) -> None:
        """Initialize with an async SQLAlchemy engine."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="segment_store_sqlalchemy",
        )

        self._is_postgresql = self._engine.dialect.name == "postgresql"
        self._is_sqlite = self._engine.dialect.name == "sqlite"

        # SQLite requires PRAGMA foreign_keys = ON for CASCADE deletes.
        if self._is_sqlite:

            @event.listens_for(self._engine.sync_engine, "connect")
            def _enable_sqlite_fks(
                dbapi_connection: DBAPIConnection,
                _connection_record: ConnectionPoolEntry,
            ) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    # Lifecycle

    @override
    async def startup(self) -> None:
        async with self._tracker("startup"), self._engine.begin() as connection:
            await connection.run_sync(BaseSegmentStore.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    # Partition management

    @override
    async def create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
    ) -> None:
        validate_partition_key(partition_key)
        async with (
            self._tracker("create_partition"),
            self._create_session() as session,
            session.begin(),
        ):
            try:
                await session.execute(
                    insert(PartitionRow).values(
                        partition_key=partition_key,
                        incarnation=new_incarnation(),
                        payload_codec_config=encode_payload_codec_config(
                            config.payload_codec_config
                        ),
                    )
                )
            except IntegrityError as err:
                raise SegmentStorePartitionAlreadyExistsError(partition_key) from err

    @override
    async def open_partition(
        self, partition_key: str
    ) -> SQLAlchemySegmentStorePartition | None:
        validate_partition_key(partition_key)
        async with self._tracker("open_partition"):
            async with self._create_session() as session:
                partition_row = await SQLAlchemySegmentStore._get_partition_row(
                    session, partition_key
                )
            if partition_row is None:
                return None

            return await self._partition_from_partition_row(partition_row)

    @override
    async def open_or_create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
    ) -> SQLAlchemySegmentStorePartition:
        validate_partition_key(partition_key)
        async with self._tracker("open_or_create_partition"):
            return await self._open_or_create_partition(partition_key, config)

    async def _open_or_create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
    ) -> SQLAlchemySegmentStorePartition:
        async with self._create_session() as session:
            partition_row = await SQLAlchemySegmentStore._get_partition_row(
                session, partition_key
            )
        if partition_row is not None:
            SQLAlchemySegmentStore._raise_if_partition_config_mismatch(
                partition_row, config
            )
            return await self._partition_from_partition_row(partition_row)

        incarnation = new_incarnation()
        try:
            async with self._create_session() as session, session.begin():
                await session.execute(
                    insert(PartitionRow).values(
                        partition_key=partition_key,
                        incarnation=incarnation,
                        payload_codec_config=encode_payload_codec_config(
                            config.payload_codec_config
                        ),
                    )
                )
        except IntegrityError:
            pass  # Concurrent creation: partition now exists.
        else:
            payload_codec = await self._load_payload_codec(config)
            return SQLAlchemySegmentStorePartition(
                partition_key=partition_key,
                incarnation=incarnation,
                engine=self._engine,
                config=config,
                payload_codec=payload_codec,
                tracker=self._tracker,
            )

        async with self._create_session() as session:
            partition_row = await SQLAlchemySegmentStore._get_partition_row(
                session, partition_key
            )
        if partition_row is None:
            raise RuntimeError(f"Partition {partition_key!r} could not be opened")

        self._raise_if_partition_config_mismatch(partition_row, config)
        return await self._partition_from_partition_row(partition_row)

    @override
    async def close_partition(
        self, segment_store_partition: SegmentStorePartition
    ) -> None:
        pass

    @override
    async def delete_partition(self, partition_key: str) -> None:
        """Unregister the partition and queue its rows for purging.

        O(1) regardless of partition size: the exclusive row lock waits out
        in-flight writers (which hold shared pins on the row), the
        incarnation goes onto the purge queue, and the registry row is
        deleted.
        Data rows become unreachable immediately -- every operation
        resolves the registry first -- and purge_deleted_partitions
        reclaims them asynchronously.
        """
        validate_partition_key(partition_key)
        async with (
            self._tracker("delete_partition"),
            self._create_session() as session,
            session.begin(),
        ):
            row = (
                await session.execute(
                    select(PartitionRow)
                    .where(PartitionRow.partition_key == partition_key)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if row is None:
                return
            await session.execute(
                insert(PurgeRow).values(
                    incarnation=row.incarnation,
                    partition_key=partition_key,
                    enqueued_at=datetime.now(UTC),
                )
            )
            await session.execute(
                delete(PartitionRow).where(PartitionRow.partition_key == partition_key)
            )

    async def purge_deleted_partitions(
        self,
        *,
        batch_size: int = 10_000,
        max_batches: int | None = None,
    ) -> int:
        """Physically reclaim rows of deleted partitions; return rows deleted.

        Drains the purge queue with chunked deletes so long-running
        reclamation of a large partition never holds long transactions.
        Intended to be scheduled as background work; call with
        max_batches=None to purge to completion.
        """
        total = 0
        batches = 0
        async with self._tracker("purge_deleted_partitions"):
            async with self._create_session() as session:
                dead_incarnations = (
                    (await session.execute(select(PurgeRow.incarnation)))
                    .scalars()
                    .all()
                )
            for incarnation in dead_incarnations:
                while True:
                    if max_batches is not None and batches >= max_batches:
                        return total
                    async with self._create_session() as session, session.begin():
                        chunk = (
                            select(SegmentRow.uuid)
                            .where(SegmentRow.incarnation == incarnation)
                            .limit(batch_size)
                            .scalar_subquery()
                        )
                        # The link-table cascade follows each segment chunk.
                        result = await session.execute(
                            delete(SegmentRow).where(
                                SegmentRow.incarnation == incarnation,
                                SegmentRow.uuid.in_(chunk),
                            )
                        )
                    # Session.execute is typed Result; DML returns a
                    # CursorResult, which carries the affected-row count.
                    deleted = result.rowcount if isinstance(result, CursorResult) else 0
                    batches += 1
                    total += deleted
                    if deleted < batch_size:
                        break
                async with self._create_session() as session, session.begin():
                    # Sweep links whose segments were already gone, then
                    # retire the queue entry.
                    await session.execute(
                        delete(DerivativeLinkRow).where(
                            DerivativeLinkRow.incarnation == incarnation
                        )
                    )
                    await session.execute(
                        delete(PurgeRow).where(PurgeRow.incarnation == incarnation)
                    )
        return total

    # Helpers

    async def _load_payload_codec(
        self,
        config: SegmentStorePartitionConfig,
    ) -> PayloadCodec:
        """Materialize a live payload codec for a partition config."""
        match config.payload_codec_config:
            case PlaintextPayloadCodecConfig():
                return PlaintextPayloadCodec()
            case _:
                raise NotImplementedError(
                    f"Unsupported payload codec config: "
                    f"{type(config.payload_codec_config).__name__}"
                )

    async def _partition_from_partition_row(
        self,
        partition_row: PartitionRow,
    ) -> SQLAlchemySegmentStorePartition:
        """Materialize a partition handle from a registry row."""
        config = SegmentStorePartitionConfig(
            payload_codec_config=decode_payload_codec_config(
                partition_row.payload_codec_config
            )
        )
        payload_codec = await self._load_payload_codec(config)
        return SQLAlchemySegmentStorePartition(
            partition_key=partition_row.partition_key,
            incarnation=partition_row.incarnation,
            engine=self._engine,
            config=config,
            payload_codec=payload_codec,
            tracker=self._tracker,
        )

    @staticmethod
    async def _get_partition_row(
        session: AsyncSession,
        partition_key: str,
    ) -> PartitionRow | None:
        """Fetch a partition row by key."""
        return (
            await session.execute(
                select(PartitionRow).where(PartitionRow.partition_key == partition_key)
            )
        ).scalar_one_or_none()

    @staticmethod
    def _raise_if_partition_config_mismatch(
        partition_row: PartitionRow,
        config: SegmentStorePartitionConfig,
    ) -> None:
        """Raise if an existing partition row does not match the requested config."""
        existing_config = SegmentStorePartitionConfig(
            payload_codec_config=decode_payload_codec_config(
                partition_row.payload_codec_config
            )
        )
        if existing_config != config:
            raise SegmentStorePartitionConfigMismatchError(
                partition_row.partition_key,
                existing_config,
                config,
            )
