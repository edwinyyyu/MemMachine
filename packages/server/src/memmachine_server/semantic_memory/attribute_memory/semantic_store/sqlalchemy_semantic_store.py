"""SQLAlchemy implementation of the :class:`SemanticStore` interface.

Four tables live under a single :class:`DeclarativeBase`:

* ``semantic_store_pt`` — partition registry; one row per partition.
* ``semantic_store_at`` — attribute rows keyed by ``(partition_key,
  id)``.  The ``id`` is also used as ``Record.uuid`` in the paired
  :class:`~memmachine_server.common.vector_store.VectorStoreCollection`.
* ``semantic_store_ct`` — citation links between an attribute and the
  source-message ids it was extracted from.
On Postgres every data table uses ``LIST`` partitioning on
``partition_key``; each :meth:`SQLAlchemySemanticStore.create_partition`
call allocates child tables per partition, and
:meth:`SQLAlchemySemanticStore.delete_partition` drops them.
"""

import logging
import re
from collections.abc import AsyncIterator, Iterable, Mapping, MutableMapping
from datetime import datetime
from typing import Any, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf, field_validator
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKeyConstraint,
    Index,
    String,
    Uuid,
    delete,
    event,
    func,
    insert,
    select,
    text,
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
    mapped_column,
)
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from sqlalchemy.sql import Delete, Select
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
from memmachine_server.semantic_memory.attribute_memory.prompts import (
    build_consolidation_prompt,
    build_update_prompt,
)
from memmachine_server.semantic_memory.attribute_memory.semantic_store.data_types import (
    SemanticStorePartitionAlreadyExistsError,
)
from memmachine_server.semantic_memory.attribute_memory.semantic_store.semantic_store import (
    SemanticAttribute,
    SemanticStore,
    SemanticStorePartition,
)
from memmachine_server.semantic_memory.storage.text_sanitizer import sanitize_pg_text

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# ---------------------------------------------------------------------- #
# ORM models
# ---------------------------------------------------------------------- #


class BaseSemanticStore(DeclarativeBase):
    """Base class for semantic store tables."""


class PartitionRow(BaseSemanticStore):
    """Tracks known partitions."""

    __tablename__ = "semantic_store_pt"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)


class AttributeRow(BaseSemanticStore):
    """One semantic-attribute row — a single leaf of the hierarchy."""

    __tablename__ = "semantic_store_at"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    id: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    topic: MappedColumn[str] = mapped_column(String, nullable=False)
    category: MappedColumn[str] = mapped_column(String, nullable=False)
    attribute: MappedColumn[str] = mapped_column(String, nullable=False)
    value: MappedColumn[str] = mapped_column(String, nullable=False)
    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    properties: MappedColumn[dict[str, Any]] = mapped_column(
        _JSON_AUTO,
        nullable=False,
        default=dict,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["partition_key"],
            ["semantic_store_pt.partition_key"],
            ondelete="CASCADE",
        ),
        Index(
            "semantic_store_at__pk_tp",
            "partition_key",
            "topic",
        ),
        Index(
            "semantic_store_at__pk_tp_ct",
            "partition_key",
            "topic",
            "category",
        ),
        Index(
            "semantic_store_at__pk_tp_ct_at",
            "partition_key",
            "topic",
            "category",
            "attribute",
        ),
        {"postgresql_partition_by": "LIST (partition_key)"},
    )

    def to_typed_model(
        self,
        *,
        citations: tuple[UUID, ...] | None = None,
    ) -> SemanticAttribute:
        """Materialize a :class:`SemanticAttribute` from this row."""
        return SemanticAttribute(
            id=self.id,
            topic=self.topic,
            category=self.category,
            attribute=self.attribute,
            value=self.value,
            properties=dict(self.properties) if self.properties else None,
            citations=citations,
        )


class CitationRow(BaseSemanticStore):
    """Link between an attribute and a source-message id it was extracted from."""

    __tablename__ = "semantic_store_ct"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    attribute_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    history_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["partition_key", "attribute_uuid"],
            [
                "semantic_store_at.partition_key",
                "semantic_store_at.id",
            ],
            ondelete="CASCADE",
        ),
        Index(
            "semantic_store_ct__pk_at",
            "partition_key",
            "attribute_uuid",
        ),
        {"postgresql_partition_by": "LIST (partition_key)"},
    )


# ---------------------------------------------------------------------- #
# Partition handle
# ---------------------------------------------------------------------- #


class SQLAlchemySemanticStorePartition(SemanticStorePartition):
    """SQLAlchemy-backed partition handle."""

    def __init__(
        self,
        partition_key: str,
        engine: AsyncEngine,
    ) -> None:
        """Initialize with a partition key and engine."""
        self._partition_key = partition_key
        self._engine = engine
        self._create_session = async_sessionmaker(engine, expire_on_commit=False)
        self._is_sqlite = engine.dialect.name == "sqlite"

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

    # ------------------------------------------------------------------ #
    # Attribute CRUD
    # ------------------------------------------------------------------ #

    @override
    async def add_attributes(
        self,
        attributes: Iterable[SemanticAttribute],
    ) -> None:
        rows = [
            {
                "partition_key": self._partition_key,
                "id": attribute.id,
                "topic": attribute.topic,
                "category": sanitize_pg_text(
                    attribute.category, context="attribute.category"
                ),
                "attribute": sanitize_pg_text(
                    attribute.attribute, context="attribute.attribute"
                ),
                "value": sanitize_pg_text(attribute.value, context="attribute.value"),
                "properties": (
                    dict(attribute.properties) if attribute.properties else {}
                ),
            }
            for attribute in attributes
        ]
        if not rows:
            return
        async with self._create_session() as session, session.begin():
            await self._lock_partition_for_write(session)
            await session.execute(insert(AttributeRow), rows)

    @override
    async def get_attributes(
        self,
        attribute_uuids: Iterable[UUID],
        *,
        load_citations: bool = False,
    ) -> Mapping[UUID, SemanticAttribute]:
        uuids = list(attribute_uuids)
        if not uuids:
            return {}

        async with self._create_session() as session:
            rows = (
                (
                    await session.execute(
                        select(AttributeRow).where(
                            AttributeRow.partition_key == self._partition_key,
                            AttributeRow.id.in_(uuids),
                        )
                    )
                )
                .scalars()
                .all()
            )

            citations_map: Mapping[UUID, tuple[UUID, ...]] = {}
            if load_citations and rows:
                citations_map = await self._load_citations(
                    session, [row.id for row in rows]
                )

        return {
            row.id: row.to_typed_model(citations=citations_map.get(row.id))
            for row in rows
        }

    @override
    async def list_attributes(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        load_citations: bool = False,
    ) -> AsyncIterator[SemanticAttribute]:
        stmt: Select[Any] = (
            select(AttributeRow)
            .where(AttributeRow.partition_key == self._partition_key)
            .order_by(AttributeRow.created_at.asc(), AttributeRow.id.asc())
        )
        stmt = self._apply_attribute_filter(stmt, filter_expr=filter_expr)

        async with self._create_session() as session:
            result = await session.stream(stmt)

            if not load_citations:
                async for row in result.scalars():
                    yield row.to_typed_model()
                return

            rows = [row async for row in result.scalars()]
            citations_map: Mapping[UUID, tuple[UUID, ...]] = {}
            if rows:
                citations_map = await self._load_citations(
                    session, [row.id for row in rows]
                )
            for row in rows:
                yield row.to_typed_model(citations=citations_map.get(row.id))

    @override
    async def list_attribute_uuids_matching(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> tuple[UUID, ...]:
        stmt: Select[Any] = select(AttributeRow.id).where(
            AttributeRow.partition_key == self._partition_key
        )
        stmt = self._apply_attribute_filter(stmt, filter_expr=filter_expr)
        async with self._create_session() as session:
            result = await session.execute(stmt)
            return tuple(result.scalars())

    @override
    async def delete_attributes(self, attribute_uuids: Iterable[UUID]) -> None:
        uuids = set(attribute_uuids)
        if not uuids:
            return

        async with self._create_session() as session, session.begin():
            await self._lock_partition_for_write(session)
            if not self._is_sqlite:
                # Lock rows in deterministic order to prevent deadlocks
                # from concurrent deletions with overlapping UUID sets.
                # SQLite relies on write serialization by the database.
                await session.execute(
                    select(AttributeRow.id)
                    .where(
                        AttributeRow.partition_key == self._partition_key,
                        AttributeRow.id.in_(uuids),
                    )
                    .order_by(AttributeRow.id)
                    .with_for_update()
                )

            # CASCADE deletes citations via FK.
            await session.execute(
                delete(AttributeRow).where(
                    AttributeRow.partition_key == self._partition_key,
                    AttributeRow.id.in_(uuids),
                )
            )

    # ------------------------------------------------------------------ #
    # Citations
    # ------------------------------------------------------------------ #

    @override
    async def add_citations(
        self,
        attribute_uuid: UUID,
        history_uuids: Iterable[UUID],
    ) -> None:
        rows = [
            {
                "partition_key": self._partition_key,
                "attribute_uuid": attribute_uuid,
                "history_uuid": history_uuid,
            }
            for history_uuid in history_uuids
        ]
        if not rows:
            return
        async with self._create_session() as session, session.begin():
            await self._lock_partition_for_write(session)
            await session.execute(insert(CitationRow), rows)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _load_citations(
        self,
        session: AsyncSession,
        attribute_uuids: Iterable[UUID],
    ) -> Mapping[UUID, tuple[UUID, ...]]:
        uuids = list(attribute_uuids)
        if not uuids:
            return {}

        stmt = select(CitationRow.attribute_uuid, CitationRow.history_uuid).where(
            CitationRow.partition_key == self._partition_key,
            CitationRow.attribute_uuid.in_(uuids),
        )
        result = await session.execute(stmt)

        accum: MutableMapping[UUID, list[UUID]] = {uuid: [] for uuid in uuids}
        for attribute_uuid, history_uuid in result:
            accum.setdefault(attribute_uuid, []).append(history_uuid)
        return {uuid: tuple(hids) for uuid, hids in accum.items()}

    def _apply_attribute_filter[StmtT: (Select[Any], Delete)](
        self,
        stmt: StmtT,
        *,
        filter_expr: FilterExpr | None,
    ) -> StmtT:
        if filter_expr is None:
            return stmt
        clause = compile_sql_filter(
            filter_expr, SQLAlchemySemanticStorePartition._resolve_attribute_field
        )
        if isinstance(stmt, Select):
            return stmt.where(clause)
        return stmt.where(clause)

    @staticmethod
    def _resolve_attribute_field(
        field: str,
    ) -> tuple[ColumnElement[Any], FieldEncoding]:
        internal_name, is_user_property = normalize_filter_field(field)
        if is_user_property:
            key = demangle_user_metadata_key(internal_name)
            return AttributeRow.properties[key], "json"

        system_fields: dict[str, InstrumentedAttribute[Any]] = {
            "topic": AttributeRow.topic,
            "category": AttributeRow.category,
            "attribute": AttributeRow.attribute,
            "value": AttributeRow.value,
            "created_at": AttributeRow.created_at,
            "updated_at": AttributeRow.updated_at,
        }
        column = system_fields.get(internal_name)
        if column is None:
            raise ValueError(f"Unknown filter field: {field!r}")
        return column.expression, "column"


# ---------------------------------------------------------------------- #
# Store factory
# ---------------------------------------------------------------------- #


class SQLAlchemySemanticStoreParams(BaseModel):
    """Parameters for constructing a :class:`SQLAlchemySemanticStore`."""

    engine: InstanceOf[AsyncEngine] = Field(..., description="Async SQLAlchemy engine")

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, engine: AsyncEngine) -> AsyncEngine:
        assert not isinstance(engine.pool, StaticPool), (
            "Engine uses StaticPool, which shares one connection across "
            "sessions. Use a multi-connection pool instead."
        )
        db = engine.url.database
        if engine.dialect.name == "sqlite" and (db is None or db == ":memory:"):
            raise ValueError(
                "Engine uses ephemeral SQLite, where each connection gets a "
                "separate database. Use a file path instead."
            )
        return engine


class SQLAlchemySemanticStore(SemanticStore):
    """SQLAlchemy-backed :class:`SemanticStore` factory for SQLite and Postgres."""

    _PARTITION_KEY_RE = re.compile(r"^[a-z0-9_]+$")

    _PG_LOCK_PARTITIONS_TABLE = text(
        "LOCK TABLE semantic_store_pt IN SHARE ROW EXCLUSIVE MODE"
    )

    def __init__(self, params: SQLAlchemySemanticStoreParams) -> None:
        """Initialize with engine; set up session factory and SQLite FK pragma."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)

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

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BaseSemanticStore.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Partition management
    # ------------------------------------------------------------------ #

    @override
    async def create_partition(self, partition_key: str) -> None:
        SQLAlchemySemanticStore._validate_partition_key(partition_key)
        async with self._engine.begin() as connection:
            if self._is_postgresql:
                await connection.execute(
                    SQLAlchemySemanticStore._PG_LOCK_PARTITIONS_TABLE
                )

            exists = (
                await connection.execute(
                    select(PartitionRow.partition_key).where(
                        PartitionRow.partition_key == partition_key
                    )
                )
            ).scalar()
            if exists is not None:
                raise SemanticStorePartitionAlreadyExistsError(partition_key)

            await connection.execute(
                insert(PartitionRow).values(partition_key=partition_key)
            )
            if self._is_postgresql:
                await self._create_pg_child_tables(connection, partition_key)

    @override
    async def open_partition(
        self, partition_key: str
    ) -> SQLAlchemySemanticStorePartition | None:
        SQLAlchemySemanticStore._validate_partition_key(partition_key)
        async with self._create_session() as session:
            exists = (
                await session.execute(
                    select(PartitionRow.partition_key).where(
                        PartitionRow.partition_key == partition_key
                    )
                )
            ).scalar()
        if exists is None:
            return None

        return SQLAlchemySemanticStorePartition(
            partition_key=partition_key,
            engine=self._engine,
        )

    @override
    async def open_or_create_partition(
        self, partition_key: str
    ) -> SQLAlchemySemanticStorePartition:
        SQLAlchemySemanticStore._validate_partition_key(partition_key)
        try:
            async with self._engine.begin() as connection:
                if self._is_postgresql:
                    await connection.execute(
                        SQLAlchemySemanticStore._PG_LOCK_PARTITIONS_TABLE
                    )

                exists = (
                    await connection.execute(
                        select(PartitionRow.partition_key).where(
                            PartitionRow.partition_key == partition_key
                        )
                    )
                ).scalar()
                if exists is None:
                    await connection.execute(
                        insert(PartitionRow).values(partition_key=partition_key)
                    )
                    if self._is_postgresql:
                        await self._create_pg_child_tables(connection, partition_key)

        except IntegrityError:
            pass  # Concurrent creation: partition now exists.

        return SQLAlchemySemanticStorePartition(
            partition_key=partition_key,
            engine=self._engine,
        )

    @override
    async def close_partition(
        self, semantic_store_partition: SemanticStorePartition
    ) -> None:
        pass

    @override
    async def delete_partition(self, partition_key: str) -> None:
        SQLAlchemySemanticStore._validate_partition_key(partition_key)
        async with self._engine.begin() as connection:
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
                await self._drop_pg_child_tables(connection, partition_key)

            # CASCADE from PartitionRow deletes attributes and citations.
            await connection.execute(
                delete(PartitionRow).where(PartitionRow.partition_key == partition_key)
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_partition_key(partition_key: str) -> None:
        """Validate that a partition key is safe for use in SQL identifiers."""
        if not SQLAlchemySemanticStore._PARTITION_KEY_RE.match(partition_key):
            raise ValueError(
                f"Partition key {partition_key!r} contains invalid characters. "
                "Only lowercase alphanumeric and underscores are allowed."
            )
        if len(partition_key) > 32:
            raise ValueError(
                f"Partition key {partition_key!r} is too long "
                f"({len(partition_key)} characters). Maximum is 32."
            )

    @staticmethod
    async def _create_pg_child_tables(
        connection: AsyncConnection, partition_key: str
    ) -> None:
        """Create PostgreSQL child partition tables for the given key."""
        attributes_child = f'"semantic_store_at_p_{partition_key}"'
        citations_child = f'"semantic_store_ct_p_{partition_key}"'
        await connection.execute(
            text(
                f"CREATE TABLE {attributes_child} PARTITION OF"
                f" semantic_store_at FOR VALUES IN ('{partition_key}')"
            )
        )
        await connection.execute(
            text(
                f"CREATE TABLE {citations_child} PARTITION OF"
                f" semantic_store_ct FOR VALUES IN ('{partition_key}')"
            )
        )

    @staticmethod
    async def _drop_pg_child_tables(
        connection: AsyncConnection, partition_key: str
    ) -> None:
        """Drop PostgreSQL child partition tables for the given key."""
        citations_child = f'"semantic_store_ct_p_{partition_key}"'
        attributes_child = f'"semantic_store_at_p_{partition_key}"'

        # CASCADE drops cross-partition FK constraint dependencies.
        await connection.execute(
            text(f"DROP TABLE IF EXISTS {citations_child} CASCADE")
        )
        await connection.execute(
            text(f"DROP TABLE IF EXISTS {attributes_child} CASCADE")
        )


# ---------------------------------------------------------------------- #
# Logical schema for an AttributeMemory instance
# ---------------------------------------------------------------------- #


class CategoryDefinition(BaseModel):
    """A category within a topic — LLM-facing name + description."""

    name: str
    description: str = ""


class TopicDefinition(BaseModel):
    """A topic: name, description, and the categories it allows.

    The update / consolidation prompts are derived from the category
    set so that the LLM receives a consistent view of the schema on
    both paths.
    """

    name: str
    description: str = ""
    categories: tuple[CategoryDefinition, ...] = ()

    @property
    def _category_descriptions(self) -> dict[str, str]:
        return {c.name: c.description for c in self.categories}

    @property
    def update_prompt(self) -> str:
        return build_update_prompt(
            categories=self._category_descriptions,
            description=self.description,
        )

    @property
    def consolidation_prompt(self) -> str:
        return build_consolidation_prompt(
            categories=self._category_descriptions,
        )


class PartitionSchema(BaseModel):
    """Full schema bundle for an :class:`AttributeMemory` instance.

    Construction-time configuration: these values are never persisted
    by the memory.  Callers resolve them from deployment config (YAML,
    env, a config service) and pass them to :class:`AttributeMemory`.
    """

    topics: tuple[TopicDefinition, ...] = ()

    def topic(self, name: str) -> TopicDefinition | None:
        for t in self.topics:
            if t.name == name:
                return t
        return None
