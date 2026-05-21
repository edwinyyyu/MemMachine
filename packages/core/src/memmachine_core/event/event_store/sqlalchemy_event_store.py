"""SQLAlchemy implementation of the EventStore interface."""

import json
import logging
import re
from collections.abc import Iterable
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
    String,
    Uuid,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy import (
    event as sqlalchemy_event,
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
    MappedColumn,
    mapped_column,
)
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool

from memmachine_core.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
)
from memmachine_core.common.payload_codec import PayloadCodec
from memmachine_core.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
    decode_payload_codec_config,
    encode_payload_codec_config,
)
from memmachine_core.common.payload_codec.plaintext_payload_codec import (
    PlaintextPayloadCodec,
)
from memmachine_core.common.properties_json import (
    decode_properties,
    encode_properties,
)
from memmachine_core.common.utils import ensure_tz_aware, utc_offset_seconds
from memmachine_core.event.data_types import (
    Event,
    NullContext,
    decode_block,
    decode_context,
    encode_block,
    encode_context,
)
from memmachine_core.event.event_store.data_types import (
    EventStorePartitionAlreadyExistsError,
    EventStorePartitionConfig,
    EventStorePartitionConfigMismatchError,
)
from memmachine_core.event.event_store.event_store import (
    EventStore,
    EventStorePartition,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# ORM models


class BaseEventStore(DeclarativeBase):
    """Base class for event store tables."""


class PartitionRow(BaseEventStore):
    """Tracks known partitions."""

    __tablename__ = "event_store_pt"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    payload_codec_config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO,
        nullable=False,
    )


class EventRow(BaseEventStore):
    """Persisted event."""

    __tablename__ = "event_store_ev"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    timestamp_timezone_offset: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    context: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    blocks: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    properties: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False, default=dict
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["partition_key"],
            ["event_store_pt.partition_key"],
            ondelete="CASCADE",
        ),
        Index(
            "event_store_ev__pk_ts_uu",
            "partition_key",
            "timestamp",
            "uuid",
        ),
        {"postgresql_partition_by": "LIST (partition_key)"},
    )


class SQLAlchemyEventStorePartition(EventStorePartition):
    """SQLAlchemy-backed partition handle."""

    def __init__(
        self,
        partition_key: str,
        engine: AsyncEngine,
        config: EventStorePartitionConfig,
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

    @override
    @property
    def config(self) -> EventStorePartitionConfig:
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
    async def add_events(self, events: Iterable[Event]) -> None:
        event_row_values = [
            {
                "uuid": event.uuid,
                "partition_key": self._partition_key,
                "timestamp": ensure_tz_aware(event.timestamp),
                "timestamp_timezone_offset": utc_offset_seconds(event.timestamp),
                "context": self._payload_codec.encode(
                    json.dumps(encode_context(event.context)).encode("utf-8")
                ),
                "blocks": self._payload_codec.encode(
                    json.dumps([encode_block(block) for block in event.blocks]).encode(
                        "utf-8"
                    )
                ),
                "properties": encode_properties(event.properties),
            }
            for event in events
        ]
        if not event_row_values:
            return

        async with (
            self._tracker("add_events"),
            self._create_session() as session,
            session.begin(),
        ):
            await self._lock_partition_for_write(session)
            await session.execute(insert(EventRow), event_row_values)

    # Retrieval

    @override
    async def get_event(self, event_uuid: UUID) -> Event | None:
        async with (
            self._tracker("get_event"),
            self._create_session() as session,
        ):
            row = (
                await session.execute(
                    select(EventRow).where(
                        EventRow.partition_key == self._partition_key,
                        EventRow.uuid == event_uuid,
                    )
                )
            ).scalar_one_or_none()
        return self._event_from_row(row) if row is not None else None

    @override
    async def get_events(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, Event]:
        event_uuids = set(event_uuids)
        if not event_uuids:
            return {}

        async with (
            self._tracker("get_events"),
            self._create_session() as session,
        ):
            rows = (
                (
                    await session.execute(
                        select(EventRow).where(
                            EventRow.partition_key == self._partition_key,
                            EventRow.uuid.in_(event_uuids),
                        )
                    )
                )
                .scalars()
                .all()
            )
        return {row.uuid: self._event_from_row(row) for row in rows}

    @override
    async def get_all_events(self) -> list[Event]:
        async with (
            self._tracker("get_all_events"),
            self._create_session() as session,
        ):
            rows = (
                (
                    await session.execute(
                        select(EventRow)
                        .where(EventRow.partition_key == self._partition_key)
                        .order_by(EventRow.timestamp, EventRow.uuid)
                    )
                )
                .scalars()
                .all()
            )
        return [self._event_from_row(row) for row in rows]

    # Deletion

    @override
    async def delete_events(self, event_uuids: Iterable[UUID]) -> None:
        event_uuids = set(event_uuids)
        if not event_uuids:
            return

        async with (
            self._tracker("delete_events"),
            self._create_session() as session,
            session.begin(),
        ):
            await self._lock_partition_for_write(session)
            if not self._is_sqlite:
                # Lock rows in deterministic order to prevent deadlocks
                # from concurrent deletions with overlapping UUID sets.
                # SQLite relies on write serialization by the database.
                await session.execute(
                    select(EventRow.uuid)
                    .where(
                        EventRow.partition_key == self._partition_key,
                        EventRow.uuid.in_(event_uuids),
                    )
                    .order_by(EventRow.uuid)
                    .with_for_update()
                )

            await session.execute(
                delete(EventRow).where(
                    EventRow.partition_key == self._partition_key,
                    EventRow.uuid.in_(event_uuids),
                )
            )

    # Helpers

    def _event_from_row(self, row: EventRow) -> Event:
        """Convert an EventRow into an Event."""
        context = decode_context(json.loads(self._payload_codec.decode(row.context)))
        if context is None:
            context = NullContext()
        blocks = [
            decode_block(encoded_block)
            for encoded_block in json.loads(self._payload_codec.decode(row.blocks))
        ]
        properties = decode_properties(row.properties)
        original_timezone = timezone(timedelta(seconds=row.timestamp_timezone_offset))
        timestamp = ensure_tz_aware(row.timestamp).astimezone(original_timezone)
        return Event(
            uuid=row.uuid,
            timestamp=timestamp,
            context=context,
            blocks=blocks,
            properties=properties,
        )


class SQLAlchemyEventStoreParams(BaseModel):
    """
    Parameters for constructing a SQLAlchemyEventStore.

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


class SQLAlchemyEventStore(EventStore):
    """SQLAlchemy-backed EventStore factory."""

    _PARTITION_KEY_RE = re.compile(r"^[a-z0-9_]+$")

    def __init__(self, params: SQLAlchemyEventStoreParams) -> None:
        """Initialize with an async SQLAlchemy engine."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="event_store_sqlalchemy",
        )

        self._is_postgresql = self._engine.dialect.name == "postgresql"
        self._is_sqlite = self._engine.dialect.name == "sqlite"

        # SQLite requires PRAGMA foreign_keys = ON for CASCADE deletes.
        if self._is_sqlite:

            @sqlalchemy_event.listens_for(self._engine.sync_engine, "connect")
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
            await connection.run_sync(BaseEventStore.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    # Partition management

    _PG_LOCK_PARTITIONS_TABLE = text(
        "LOCK TABLE event_store_pt IN SHARE ROW EXCLUSIVE MODE"
    )

    @override
    async def create_partition(
        self,
        partition_key: str,
        config: EventStorePartitionConfig,
    ) -> None:
        SQLAlchemyEventStore._validate_partition_key(partition_key)
        async with (
            self._tracker("create_partition"),
            self._engine.begin() as connection,
        ):
            if self._is_postgresql:
                await connection.execute(SQLAlchemyEventStore._PG_LOCK_PARTITIONS_TABLE)

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
                raise EventStorePartitionAlreadyExistsError(partition_key) from err
            if self._is_postgresql:
                await SQLAlchemyEventStore._create_pg_child_tables(
                    connection, partition_key
                )

    @override
    async def open_partition(
        self, partition_key: str
    ) -> SQLAlchemyEventStorePartition | None:
        SQLAlchemyEventStore._validate_partition_key(partition_key)
        async with self._tracker("open_partition"):
            async with self._create_session() as session:
                partition_row = await SQLAlchemyEventStore._get_partition_row(
                    session, partition_key
                )
            if partition_row is None:
                return None

            return await self._partition_from_partition_row(partition_row)

    @override
    async def open_or_create_partition(
        self,
        partition_key: str,
        config: EventStorePartitionConfig,
    ) -> SQLAlchemyEventStorePartition:
        SQLAlchemyEventStore._validate_partition_key(partition_key)
        async with self._tracker("open_or_create_partition"):
            return await self._open_or_create_partition(partition_key, config)

    async def _open_or_create_partition(
        self,
        partition_key: str,
        config: EventStorePartitionConfig,
    ) -> SQLAlchemyEventStorePartition:
        try:
            async with self._create_session() as session, session.begin():
                if self._is_postgresql:
                    await session.execute(
                        SQLAlchemyEventStore._PG_LOCK_PARTITIONS_TABLE
                    )

                partition_row = await SQLAlchemyEventStore._get_partition_row(
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
                        await SQLAlchemyEventStore._create_pg_child_tables(
                            connection, partition_key
                        )

                    return SQLAlchemyEventStorePartition(
                        partition_key=partition_key,
                        engine=self._engine,
                        config=config,
                        payload_codec=payload_codec,
                        tracker=self._tracker,
                    )

                SQLAlchemyEventStore._raise_if_partition_config_mismatch(
                    partition_row, config
                )
                return await self._partition_from_partition_row(partition_row)

        except IntegrityError:
            pass  # Concurrent creation: partition now exists.

        async with self._create_session() as session:
            partition_row = await SQLAlchemyEventStore._get_partition_row(
                session, partition_key
            )
        if partition_row is None:
            raise RuntimeError(f"Partition {partition_key!r} could not be opened")

        self._raise_if_partition_config_mismatch(partition_row, config)
        return await self._partition_from_partition_row(partition_row)

    @override
    async def close_partition(self, event_store_partition: EventStorePartition) -> None:
        pass

    @override
    async def delete_partition(self, partition_key: str) -> None:
        SQLAlchemyEventStore._validate_partition_key(partition_key)
        async with (
            self._tracker("delete_partition"),
            self._engine.begin() as connection,
        ):
            if not self._is_sqlite:
                # Exclusive lock on the partition row blocks concurrent writes
                # (which hold shared locks) until deletion completes.
                # SQLite relies on write serialization by the database.
                await connection.execute(
                    select(PartitionRow.partition_key)
                    .where(PartitionRow.partition_key == partition_key)
                    .with_for_update()
                )
            if self._is_postgresql:
                await SQLAlchemyEventStore._drop_pg_child_tables(
                    connection, partition_key
                )

            # CASCADE from PartitionRow deletes events.
            await connection.execute(
                delete(PartitionRow).where(PartitionRow.partition_key == partition_key)
            )

    # Helpers

    @staticmethod
    def _validate_partition_key(partition_key: str) -> None:
        """Validate that a partition key is safe for use in SQL identifiers."""
        if not SQLAlchemyEventStore._PARTITION_KEY_RE.match(partition_key):
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
        config: EventStorePartitionConfig,
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
    ) -> SQLAlchemyEventStorePartition:
        """Materialize a partition handle from a DB row."""
        config = EventStorePartitionConfig(
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
        config: EventStorePartitionConfig,
        *,
        payload_codec: PayloadCodec | None = None,
    ) -> SQLAlchemyEventStorePartition:
        """Materialize a partition handle from config."""
        if payload_codec is None:
            payload_codec = await self._load_payload_codec(config)
        return SQLAlchemyEventStorePartition(
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
        config: EventStorePartitionConfig,
    ) -> None:
        """Raise if an existing partition row does not match the requested config."""
        existing_config = EventStorePartitionConfig(
            payload_codec_config=decode_payload_codec_config(
                partition_row.payload_codec_config
            )
        )
        if existing_config != config:
            raise EventStorePartitionConfigMismatchError(
                partition_row.partition_key,
                existing_config,
                config,
            )

    @staticmethod
    async def _create_pg_child_tables(
        connection: AsyncConnection, partition_key: str
    ) -> None:
        """Create the PostgreSQL child partition table for the given key."""
        events_child = f'"event_store_ev_p_{partition_key}"'
        await connection.execute(
            text(
                f"CREATE TABLE {events_child} PARTITION OF"
                f" event_store_ev FOR VALUES IN ('{partition_key}')"
            )
        )

    @staticmethod
    async def _drop_pg_child_tables(
        connection: AsyncConnection, partition_key: str
    ) -> None:
        """Drop the PostgreSQL child partition table for the given key."""
        events_child = f'"event_store_ev_p_{partition_key}"'
        await connection.execute(text(f"DROP TABLE IF EXISTS {events_child} CASCADE"))
