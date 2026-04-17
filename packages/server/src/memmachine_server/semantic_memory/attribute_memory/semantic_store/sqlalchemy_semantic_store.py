"""SQLAlchemy implementation of the :class:`SemanticStore` interface.

Two tables live under a single :class:`DeclarativeBase`:

* ``semantic_store_attributes`` — attribute rows keyed by UUID.  The
  same UUID is used as ``Record.uuid`` in the paired
  :class:`~memmachine_server.common.vector_store.VectorStoreCollection`.
* ``semantic_store_citations`` — citation links between an attribute
  and the source-message ids it was extracted from.

Ingestion tracking is *not* in this module; see
:class:`~memmachine_server.common.message_queue.MessageQueue`.
"""

import logging
from collections.abc import AsyncIterator, Iterable, Mapping, MutableMapping
from datetime import datetime
from typing import Any, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf, field_validator
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    String,
    Uuid,
    delete,
    event,
    func,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
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
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.semantic_memory.attribute_memory.semantic_store.semantic_store import (
    SemanticAttribute,
    SemanticStore,
)
from memmachine_server.semantic_memory.storage.text_sanitizer import sanitize_pg_text

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# ---------------------------------------------------------------------- #
# ORM models
# ---------------------------------------------------------------------- #


class SemanticStoreBase(DeclarativeBase):
    """DeclarativeBase for the SQLAlchemy semantic store tables."""


class AttributeRow(SemanticStoreBase):
    """One semantic-attribute row — a single leaf of the hierarchy."""

    __tablename__ = "semantic_store_attributes"

    id: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    partition_id: MappedColumn[str] = mapped_column(String, nullable=False, index=True)
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
        Index(
            "semantic_store_attributes__pi_topic",
            "partition_id",
            "topic",
        ),
        Index(
            "semantic_store_attributes__pi_topic_cat",
            "partition_id",
            "topic",
            "category",
        ),
        Index(
            "semantic_store_attributes__pi_topic_cat_at",
            "partition_id",
            "topic",
            "category",
            "attribute",
        ),
    )

    def to_typed_model(
        self,
        *,
        citations: tuple[UUID, ...] | None = None,
    ) -> SemanticAttribute:
        """Materialize a :class:`SemanticAttribute` from this row."""
        return SemanticAttribute(
            id=self.id,
            partition_id=self.partition_id,
            topic=self.topic,
            category=self.category,
            attribute=self.attribute,
            value=self.value,
            properties=dict(self.properties) if self.properties else None,
            citations=citations,
        )


class CitationRow(SemanticStoreBase):
    """Link between an attribute and a source-message id it was extracted from."""

    __tablename__ = "semantic_store_citations"

    attribute_id: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey(
            "semantic_store_attributes.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    history_id: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)


# ---------------------------------------------------------------------- #
# Implementation
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
    """SQLAlchemy-backed :class:`SemanticStore` for SQLite and Postgres."""

    def __init__(self, params: SQLAlchemySemanticStoreParams) -> None:
        """Initialize with engine; set up session factory and SQLite FK pragma."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._is_sqlite = self._engine.dialect.name == "sqlite"

        # SQLite requires PRAGMA foreign_keys = ON for CASCADE deletes on
        # CitationRow.
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
            await connection.run_sync(SemanticStoreBase.metadata.create_all)

    @override
    async def cleanup(self) -> None:
        await self._engine.dispose()

    @override
    async def delete_all(self) -> None:
        async with self._create_session() as session, session.begin():
            # Explicit order even though CASCADE covers citations — keeps
            # the result well-defined regardless of FK enforcement state.
            await session.execute(delete(CitationRow))
            await session.execute(delete(AttributeRow))

    # ------------------------------------------------------------------ #
    # Attribute CRUD
    # ------------------------------------------------------------------ #

    @override
    async def add_attribute(self, attribute: SemanticAttribute) -> None:
        stmt = insert(AttributeRow).values(
            id=attribute.id,
            partition_id=attribute.partition_id,
            topic=attribute.topic,
            category=sanitize_pg_text(attribute.category, context="attribute.category"),
            attribute=sanitize_pg_text(
                attribute.attribute, context="attribute.attribute"
            ),
            value=sanitize_pg_text(attribute.value, context="attribute.value"),
            properties=dict(attribute.properties) if attribute.properties else {},
        )
        async with self._create_session() as session, session.begin():
            await session.execute(stmt)

    @override
    async def get_attribute(
        self,
        attribute_id: UUID,
        *,
        load_citations: bool = False,
    ) -> SemanticAttribute | None:
        async with self._create_session() as session:
            row = (
                await session.execute(
                    select(AttributeRow).where(AttributeRow.id == attribute_id)
                )
            ).scalar_one_or_none()

            if row is None:
                return None

            citations_map: Mapping[UUID, tuple[UUID, ...]] = {}
            if load_citations:
                citations_map = await self._load_citations(session, [row.id])

            return row.to_typed_model(citations=citations_map.get(row.id))

    @override
    async def get_attributes(
        self,
        attribute_ids: Iterable[UUID],
        *,
        load_citations: bool = False,
    ) -> Mapping[UUID, SemanticAttribute]:
        ids = list(attribute_ids)
        if not ids:
            return {}

        async with self._create_session() as session:
            rows = (
                (
                    await session.execute(
                        select(AttributeRow).where(AttributeRow.id.in_(ids))
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
        stmt: Select[Any] = select(AttributeRow).order_by(
            AttributeRow.created_at.asc(), AttributeRow.id.asc()
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
    async def list_attribute_ids_matching(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> tuple[UUID, ...]:
        stmt: Select[Any] = select(AttributeRow.id)
        stmt = self._apply_attribute_filter(stmt, filter_expr=filter_expr)
        async with self._create_session() as session:
            result = await session.execute(stmt)
            return tuple(result.scalars())

    @override
    async def delete_attributes(self, attribute_ids: Iterable[UUID]) -> None:
        ids = list(attribute_ids)
        if not ids:
            return
        stmt = delete(AttributeRow).where(AttributeRow.id.in_(ids))
        async with self._create_session() as session, session.begin():
            await session.execute(stmt)

    # ------------------------------------------------------------------ #
    # Citations
    # ------------------------------------------------------------------ #

    @override
    async def add_citations(
        self,
        attribute_id: UUID,
        history_ids: Iterable[UUID],
    ) -> None:
        rows = [
            {"attribute_id": attribute_id, "history_id": hid} for hid in history_ids
        ]
        if not rows:
            return
        async with self._create_session() as session, session.begin():
            await session.execute(insert(CitationRow), rows)

    # ------------------------------------------------------------------ #
    # Partition discovery
    # ------------------------------------------------------------------ #

    @override
    async def list_partitions(
        self,
        *,
        prefix: str | None = None,
    ) -> AsyncIterator[str]:
        stmt: Select[Any] = select(AttributeRow.partition_id).distinct()
        if prefix is not None:
            stmt = stmt.where(AttributeRow.partition_id.startswith(prefix))
        async with self._create_session() as session:
            result = await session.stream(stmt)
            async for partition_id in result.scalars():
                yield partition_id

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _load_citations(
        self,
        session: AsyncSession,
        attribute_ids: Iterable[UUID],
    ) -> Mapping[UUID, tuple[UUID, ...]]:
        ids = list(attribute_ids)
        if not ids:
            return {}

        stmt = select(CitationRow.attribute_id, CitationRow.history_id).where(
            CitationRow.attribute_id.in_(ids)
        )
        result = await session.execute(stmt)

        accum: MutableMapping[UUID, list[UUID]] = {aid: [] for aid in ids}
        for attribute_id, history_id in result:
            accum.setdefault(attribute_id, []).append(history_id)
        return {aid: tuple(hids) for aid, hids in accum.items()}

    def _apply_attribute_filter[StmtT: (Select[Any], Delete)](
        self,
        stmt: StmtT,
        *,
        filter_expr: FilterExpr | None,
    ) -> StmtT:
        if filter_expr is None:
            return stmt
        clause = compile_sql_filter(
            filter_expr, SQLAlchemySemanticStore._resolve_attribute_field
        )
        if clause is None:
            return stmt
        if isinstance(stmt, Select):
            return stmt.where(clause)
        return stmt.where(clause)

    @staticmethod
    def _resolve_attribute_field(
        field: str,
    ) -> tuple[
        MappedColumn[Any] | InstrumentedAttribute[Any] | ColumnElement[Any] | None,
        bool,
    ]:
        internal_name, is_user_property = normalize_filter_field(field)
        if is_user_property:
            key = demangle_user_metadata_key(internal_name)
            return AttributeRow.properties[key], True

        system_fields: dict[str, InstrumentedAttribute[Any]] = {
            "partition_id": AttributeRow.partition_id,
            "topic": AttributeRow.topic,
            "category": AttributeRow.category,
            "attribute": AttributeRow.attribute,
            "value": AttributeRow.value,
            "created_at": AttributeRow.created_at,
            "updated_at": AttributeRow.updated_at,
        }
        if internal_name in system_fields:
            return system_fields[internal_name], False
        return None, False
