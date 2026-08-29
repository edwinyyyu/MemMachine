"""SQLAlchemy implementation of the SegmentStore interface."""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
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
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Uuid,
    delete,
    event,
    insert,
    literal,
    select,
    text,
    true,
    tuple_,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    MappedColumn,
    aliased,
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
)
from memmachine_server.episodic_memory.event_memory.segment_store.segment_store import (
    SegmentStore,
    SegmentStorePartition,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# ORM models


class BaseSegmentStore(DeclarativeBase):
    """Base class for segment store tables."""


class PartitionRow(BaseSegmentStore):
    """Tracks known partitions."""

    __tablename__ = "segment_store_pt"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    payload_codec_config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO,
        nullable=False,
    )


class SegmentRow(BaseSegmentStore):
    """Persisted segment."""

    __tablename__ = "segment_store_sg"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)

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

    __table_args__ = (
        ForeignKeyConstraint(
            ["partition_key"],
            ["segment_store_pt.partition_key"],
            ondelete="CASCADE",
        ),
        Index(
            "segment_store_sg__pk_ev",
            "partition_key",
            "event_uuid",
        ),
        Index(
            "segment_store_sg__pk_ts_ev_bk_ix",
            "partition_key",
            "timestamp",
            "event_uuid",
            "index",
            "offset",
        ),
        {"postgresql_partition_by": "LIST (partition_key)"},
    )


class DerivativeLinkRow(BaseSegmentStore):
    """Maps a derivative UUID to its owning segment."""

    __tablename__ = "segment_store_dv_ln"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["partition_key", "segment_uuid"],
            [
                "segment_store_sg.partition_key",
                "segment_store_sg.uuid",
            ],
            ondelete="CASCADE",
        ),
        Index(
            "segment_store_dv_ln__pk_su",
            "partition_key",
            "segment_uuid",
        ),
        {"postgresql_partition_by": "LIST (partition_key)"},
    )


def _pg_child_table_name(parent_table_name: str, partition_key: str) -> str:
    """Name of the PostgreSQL child table holding one partition's rows."""
    return f"{parent_table_name}_p_{partition_key}"


class SQLAlchemySegmentStorePartition(SegmentStorePartition):
    """SQLAlchemy-backed partition handle."""

    def __init__(
        self,
        partition_key: str,
        engine: AsyncEngine,
        config: SegmentStorePartitionConfig,
        payload_codec: PayloadCodec,
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a partition key and engine."""
        self._partition_key = partition_key
        self._engine = engine
        self._config = config
        self._payload_codec = payload_codec
        self._tracker = tracker
        self._create_session = async_sessionmaker(engine, expire_on_commit=False)
        self._is_sqlite = engine.dialect.name == "sqlite"

        if engine.dialect.name == "postgresql":
            # Address this partition's child tables directly instead of the
            # partitioned parents. Parent-table queries carry the partition
            # key as a bind parameter, so once the driver's prepared
            # statement switches to a cached generic plan (after five
            # executions) PostgreSQL locks EVERY child partition on every
            # execution before runtime pruning runs; with many partitions
            # and concurrent sessions this exhausts the lock table
            # ("out of shared memory") and saturates the planner. A query
            # that names the child only ever plans and locks that child.
            metadata = MetaData()
            segment_child_table = SegmentRow.__table__.to_metadata(
                metadata,
                name=_pg_child_table_name(SegmentRow.__tablename__, partition_key),
            )
            derivative_link_child_table = DerivativeLinkRow.__table__.to_metadata(
                metadata,
                name=_pg_child_table_name(
                    DerivativeLinkRow.__tablename__, partition_key
                ),
            )
            # adapt_on_names: the child Table is a fresh object with no
            # lineage to the parent's columns, so the entity's attributes
            # must map onto it by column name.
            self._segment_row = aliased(
                SegmentRow, segment_child_table, adapt_on_names=True
            )
            self._derivative_link_row = aliased(
                DerivativeLinkRow, derivative_link_child_table, adapt_on_names=True
            )
            # insert()/delete() cannot target an aliased entity; they take
            # the child Table itself (or the ORM class on other dialects).
            self._segment_dml_target = segment_child_table
            self._derivative_link_dml_target = derivative_link_child_table
        else:
            self._segment_row = SegmentRow
            self._derivative_link_row = DerivativeLinkRow
            self._segment_dml_target = SegmentRow
            self._derivative_link_dml_target = DerivativeLinkRow

    @override
    @property
    def config(self) -> SegmentStorePartitionConfig:
        return self._config

    async def _lock_partition_for_write(self, session: AsyncSession) -> None:
        """Acquire a shared lock on the partition row to prevent concurrent deletion."""
        if not self._is_sqlite:
            # Shared lock on the partition row blocks concurrent deletions
            # (which hold exclusive locks) until write completes.
            # SQLite relies on write serialization by the database.
            await session.execute(
                select(PartitionRow.partition_key)
                .where(PartitionRow.partition_key == self._partition_key)
                .with_for_update(read=True)
            )

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
                "partition_key": self._partition_key,
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
            await session.execute(insert(self._segment_dml_target), segment_row_values)

    async def _insert_derivative_links(
        self,
        session: AsyncSession,
        segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        """Insert derivative rows."""
        derivative_row_values = [
            {
                "uuid": derivative_uuid,
                "partition_key": self._partition_key,
                "segment_uuid": segment.uuid,
            }
            for segment, derivative_uuids in segments_to_derivative_uuids.items()
            for derivative_uuid in derivative_uuids
        ]
        if derivative_row_values:
            await session.execute(
                insert(self._derivative_link_dml_target), derivative_row_values
            )

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
            segment_row = self._segment_row
            seed_segments_query = select(segment_row).where(
                segment_row.uuid.in_(seed_segment_uuids),
                segment_row.partition_key == self._partition_key,
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
        segment_row = self._segment_row
        seeds_subquery = (
            select(
                segment_row.uuid.label("seed_uuid"),
                segment_row.timestamp.label("seed_timestamp"),
                segment_row.event_uuid.label("seed_event_uuid"),
                segment_row.index.label("seed_index"),
                segment_row.offset.label("seed_offset"),
            )
            .where(
                segment_row.partition_key == self._partition_key,
                segment_row.uuid.in_(seed_rows_by_uuid.keys()),
            )
            .subquery("seeds")
        )

        segment_ordering_columns = tuple_(
            segment_row.timestamp,
            segment_row.event_uuid,
            segment_row.index,
            segment_row.offset,
        )
        seed_ordering_columns = tuple_(
            seeds_subquery.c.seed_timestamp,
            seeds_subquery.c.seed_event_uuid,
            seeds_subquery.c.seed_index,
            seeds_subquery.c.seed_offset,
        )

        partition_key = self._partition_key

        async def get_context_rows_directional(
            range_condition: ColumnElement[bool],
            ordering: Iterable[ColumnElement | InstrumentedAttribute],
            limit: int,
        ) -> dict[UUID, list[SegmentRow]]:
            """Get context rows per seed in the specified direction."""
            # Build a LATERAL subquery that gets context rows for each seed.
            context_rows_query = (
                select(segment_row)
                .where(segment_row.partition_key == partition_key, range_condition)
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
                        partition_key=partition_key,
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
            segment_row.timestamp,
            segment_row.event_uuid,
            segment_row.index,
            segment_row.offset,
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
        segment_row = self._segment_row
        context_rows_by_seed: dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]] = {}

        segment_ordering_columns = tuple_(
            segment_row.timestamp,
            segment_row.event_uuid,
            segment_row.index,
            segment_row.offset,
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
                    select(segment_row)
                    .where(
                        segment_row.partition_key == self._partition_key,
                        segment_ordering_columns < seed_ordering_values,
                    )
                    .order_by(
                        segment_row.timestamp.desc(),
                        segment_row.event_uuid.desc(),
                        segment_row.index.desc(),
                        segment_row.offset.desc(),
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
                    select(segment_row)
                    .where(
                        segment_row.partition_key == self._partition_key,
                        segment_ordering_columns > seed_ordering_values,
                    )
                    .order_by(
                        segment_row.timestamp,
                        segment_row.event_uuid,
                        segment_row.index,
                        segment_row.offset,
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
            segment_row = self._segment_row
            query = select(segment_row.event_uuid, segment_row.uuid).where(
                segment_row.partition_key == self._partition_key,
                segment_row.event_uuid.in_(event_uuids),
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
            derivative_link_row = self._derivative_link_row
            query = select(
                derivative_link_row.segment_uuid, derivative_link_row.uuid
            ).where(
                derivative_link_row.partition_key == self._partition_key,
                derivative_link_row.segment_uuid.in_(segment_uuids),
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
            segment_row = self._segment_row
            if not self._is_sqlite:
                # Lock rows in deterministic order to prevent deadlocks
                # from concurrent deletions with overlapping UUID sets.
                # SQLite relies on write serialization by the database.
                await session.execute(
                    select(segment_row.uuid)
                    .where(
                        segment_row.partition_key == self._partition_key,
                        segment_row.uuid.in_(segment_uuids),
                    )
                    .order_by(segment_row.uuid)
                    .with_for_update()
                )

            # CASCADE deletes derivatives via FK.
            await session.execute(
                delete(self._segment_dml_target).where(
                    segment_row.partition_key == self._partition_key,
                    segment_row.uuid.in_(segment_uuids),
                )
            )

    # Helpers

    def _resolve_segment_field(
        self,
        field: str,
    ) -> tuple[ColumnElement, FieldEncoding]:
        """Map a filter field name to a segment column and encoding."""
        segment_row = self._segment_row
        if field == "timestamp":
            return segment_row.timestamp.expression, "column"
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            key = demangle_user_metadata_key(internal_name)
            return segment_row.properties[key], "properties_json"
        return segment_row.properties[f"_{field}"], "properties_json"

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

    _PARTITION_KEY_RE = re.compile(r"^[a-z0-9_]+$")

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

    _PG_LOCK_PARTITIONS_TABLE = text(
        "LOCK TABLE segment_store_pt IN SHARE ROW EXCLUSIVE MODE"
    )

    _PG_CHILD_TABLE_STATE = text(
        """
        SELECT child.name AS name,
               to_regclass(child.name) IS NOT NULL AS table_exists,
               EXISTS (
                   SELECT 1
                   FROM pg_inherits
                   WHERE inhrelid = to_regclass(child.name)
                     AND inhparent = to_regclass(child.parent)
               ) AS attached
        FROM unnest(cast(:names AS text[]), cast(:parents AS text[]))
             AS child(name, parent)
        """
    )

    @override
    async def create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
    ) -> None:
        SQLAlchemySegmentStore._validate_partition_key(partition_key)
        async with (
            self._tracker("create_partition"),
            self._engine.begin() as connection,
        ):
            if self._is_postgresql:
                await connection.execute(
                    SQLAlchemySegmentStore._PG_LOCK_PARTITIONS_TABLE
                )

            try:
                await connection.execute(
                    insert(PartitionRow).values(
                        partition_key=partition_key,
                        payload_codec_config=encode_payload_codec_config(
                            config.payload_codec_config
                        ),
                    )
                )
            except IntegrityError as err:
                raise SegmentStorePartitionAlreadyExistsError(partition_key) from err
            if self._is_postgresql:
                await SQLAlchemySegmentStore._create_pg_child_tables(
                    connection, partition_key
                )

    @override
    async def open_partition(
        self, partition_key: str
    ) -> SQLAlchemySegmentStorePartition | None:
        SQLAlchemySegmentStore._validate_partition_key(partition_key)
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
        SQLAlchemySegmentStore._validate_partition_key(partition_key)
        async with self._tracker("open_or_create_partition"):
            return await self._open_or_create_partition(partition_key, config)

    async def _open_or_create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
    ) -> SQLAlchemySegmentStorePartition:
        try:
            async with self._create_session() as session, session.begin():
                if self._is_postgresql:
                    await session.execute(
                        SQLAlchemySegmentStore._PG_LOCK_PARTITIONS_TABLE
                    )

                partition_row = await SQLAlchemySegmentStore._get_partition_row(
                    session, partition_key
                )
                if partition_row is None:
                    payload_codec = await self._load_payload_codec(config)
                    await session.execute(
                        insert(PartitionRow).values(
                            partition_key=partition_key,
                            payload_codec_config=encode_payload_codec_config(
                                config.payload_codec_config
                            ),
                        )
                    )
                    if self._is_postgresql:
                        connection = await session.connection()
                        await SQLAlchemySegmentStore._create_pg_child_tables(
                            connection, partition_key
                        )

                    return SQLAlchemySegmentStorePartition(
                        partition_key=partition_key,
                        engine=self._engine,
                        config=config,
                        payload_codec=payload_codec,
                        tracker=self._tracker,
                    )

                SQLAlchemySegmentStore._raise_if_partition_config_mismatch(
                    partition_row, config
                )
                return await self._partition_from_partition_row(partition_row)

        except IntegrityError:
            pass  # Concurrent creation: partition now exists.

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
        SQLAlchemySegmentStore._validate_partition_key(partition_key)
        async with (
            self._tracker("delete_partition"),
            self._engine.begin() as connection,
        ):
            if not self._is_sqlite:
                # Exclusive lock on the partition row blocks concurrent writes
                # (which hold shared locks) until deletion completes.
                # Taken before the partitions-table lock so that waiting for
                # in-flight writers does not hold a store-wide lock.
                # SQLite relies on write serialization by the database.
                await connection.execute(
                    select(PartitionRow.partition_key)
                    .where(PartitionRow.partition_key == partition_key)
                    .with_for_update()
                )
            if self._is_postgresql:
                # Same lock the create paths take, held across the child DDL,
                # so a concurrent create and delete cannot reach the two
                # parent tables in opposite order and deadlock.
                await connection.execute(
                    SQLAlchemySegmentStore._PG_LOCK_PARTITIONS_TABLE
                )
                await SQLAlchemySegmentStore._drop_pg_child_tables(
                    connection, partition_key
                )

            # CASCADE from PartitionRow deletes segments and derivatives.
            await connection.execute(
                delete(PartitionRow).where(PartitionRow.partition_key == partition_key)
            )

    # Helpers

    @staticmethod
    def _validate_partition_key(partition_key: str) -> None:
        """Validate that a partition key is safe for use in SQL identifiers."""
        if not SQLAlchemySegmentStore._PARTITION_KEY_RE.match(partition_key):
            raise ValueError(
                f"Partition key {partition_key!r} contains invalid characters. "
                "Only lowercase alphanumeric and underscores are allowed."
            )
        if len(partition_key) > 32:
            raise ValueError(
                f"Partition key {partition_key!r} is too long "
                f"({len(partition_key)} characters). Maximum is 32."
            )

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
        """Materialize a partition handle from a DB row."""
        config = SegmentStorePartitionConfig(
            payload_codec_config=decode_payload_codec_config(
                partition_row.payload_codec_config
            )
        )
        return await self._partition_from_config(
            partition_row.partition_key,
            config,
        )

    async def _partition_from_config(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
        *,
        payload_codec: PayloadCodec | None = None,
    ) -> SQLAlchemySegmentStorePartition:
        """Materialize a partition handle from config."""
        if payload_codec is None:
            payload_codec = await self._load_payload_codec(config)
        return SQLAlchemySegmentStorePartition(
            partition_key=partition_key,
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

    @staticmethod
    async def _create_pg_child_tables(
        connection: AsyncConnection, partition_key: str
    ) -> None:
        """Create PostgreSQL child partition tables for the given key."""
        for parent in (SegmentRow.__tablename__, DerivativeLinkRow.__tablename__):
            child = _pg_child_table_name(parent, partition_key)
            await connection.execute(
                text(
                    f'CREATE TABLE "{child}" PARTITION OF'
                    f" {parent} FOR VALUES IN ('{partition_key}')"
                )
            )

    @staticmethod
    async def _drop_pg_child_tables(
        connection: AsyncConnection, partition_key: str
    ) -> None:
        """
        Drop PostgreSQL child partition tables for the given key.

        An attached child is detached before it is dropped.
        Dropping a child while it is still attached requires CASCADE,
        which drops the foreign key
        between segment_store_dv_ln and segment_store_sg
        for every partition rather than only this one.

        A child that exists but is already detached
        (left behind by manual maintenance,
        or by an interrupted DETACH PARTITION CONCURRENTLY,
        which is not transactional)
        is dropped directly:
        DETACH raises "is not a partition of" on such a table,
        which would leave the partition neither deletable nor recreatable.
        """
        parents = [DerivativeLinkRow.__tablename__, SegmentRow.__tablename__]
        children = [_pg_child_table_name(parent, partition_key) for parent in parents]
        rows = (
            await connection.execute(
                SQLAlchemySegmentStore._PG_CHILD_TABLE_STATE,
                {"names": children, "parents": parents},
            )
        ).all()
        state = {row.name: (row.table_exists, row.attached) for row in rows}

        for parent, child in zip(parents, children, strict=True):
            table_exists, attached = state[child]
            if not table_exists:
                continue
            if attached:
                await connection.execute(
                    text(f'ALTER TABLE {parent} DETACH PARTITION "{child}"')
                )
            await connection.execute(text(f'DROP TABLE "{child}"'))
