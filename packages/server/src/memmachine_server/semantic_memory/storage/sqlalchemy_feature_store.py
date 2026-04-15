"""SQLAlchemy implementation of the SemanticFeatureStore interface.

Three tables live under a single :class:`DeclarativeBase`:

* ``semantic_feature_store_fe`` — feature rows keyed by UUID. The same
  UUID is also used as ``Record.uuid`` in the paired
  :class:`~memmachine_server.common.vector_store.VectorStoreCollection`.
* ``semantic_feature_store_ci`` — citation links between a feature and
  the episode IDs it was extracted from.
* ``semantic_feature_store_sh`` — history-ingestion tracking, one row
  per (set_id, history_id) pair with an ``ingested`` flag.

See ``sqlalchemy_pgvector_semantic.py`` for the old monolithic
``SemanticStorage`` implementation (kept as reference, not integrated).
"""

import logging
from collections import Counter
from collections.abc import AsyncIterator, Mapping, MutableMapping, Sequence
from datetime import datetime
from typing import Any, cast, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf, field_validator
from sqlalchemy import (
    JSON,
    Boolean,
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
    union,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    MappedColumn,
    aliased,
    mapped_column,
)
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from sqlalchemy.sql import Delete, Select
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.episode_store.episode_model import EpisodeIdT
from memmachine_server.common.errors import InvalidArgumentError
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    normalize_filter_field,
)
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.semantic_memory.semantic_model import (
    SemanticFeature,
    SetIdT,
)
from memmachine_server.semantic_memory.storage.feature_store import (
    SemanticFeatureStore,
)
from memmachine_server.semantic_memory.storage.text_sanitizer import sanitize_pg_text

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# ---------------------------------------------------------------------- #
# ORM models
# ---------------------------------------------------------------------- #


class BaseFeatureStore(DeclarativeBase):
    """Declarative base for the SQLAlchemy feature store tables."""


class FeatureRow(BaseFeatureStore):
    """One semantic feature row."""

    __tablename__ = "semantic_feature_store_fe"

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    set_id: MappedColumn[str] = mapped_column(String, nullable=False, index=True)
    category_name: MappedColumn[str] = mapped_column(String, nullable=False)
    tag: MappedColumn[str] = mapped_column(String, nullable=False)
    feature: MappedColumn[str] = mapped_column(String, nullable=False)
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
        Index("semantic_feature_store_fe__si_cat", "set_id", "category_name"),
        Index(
            "semantic_feature_store_fe__si_cat_tag",
            "set_id",
            "category_name",
            "tag",
        ),
        Index(
            "semantic_feature_store_fe__si_cat_tag_fe",
            "set_id",
            "category_name",
            "tag",
            "feature",
        ),
    )

    def to_typed_model(
        self,
        *,
        citations: Sequence[EpisodeIdT] | None = None,
    ) -> SemanticFeature:
        return SemanticFeature(
            set_id=self.set_id,
            category=self.category_name,
            tag=self.tag,
            feature_name=self.feature,
            value=self.value,
            metadata=SemanticFeature.Metadata(
                id=self.uuid,
                citations=citations,
                other=dict(self.properties) if self.properties else None,
            ),
        )


class CitationRow(BaseFeatureStore):
    """Link between a feature and an episode ID it was extracted from."""

    __tablename__ = "semantic_feature_store_ci"

    feature_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey(
            "semantic_feature_store_fe.uuid",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    history_id: MappedColumn[str] = mapped_column(String, primary_key=True)


class SetHistoryRow(BaseFeatureStore):
    """Ingestion-tracking row: one per (set_id, history_id) pair."""

    __tablename__ = "semantic_feature_store_sh"

    set_id: MappedColumn[str] = mapped_column(String, primary_key=True, index=True)
    history_id: MappedColumn[str] = mapped_column(String, primary_key=True)
    ingested: MappedColumn[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("semantic_feature_store_sh__si_ing", "set_id", "ingested"),
    )


# ---------------------------------------------------------------------- #
# Implementation
# ---------------------------------------------------------------------- #


class SQLAlchemyFeatureStoreParams(BaseModel):
    """Parameters for constructing a :class:`SQLAlchemyFeatureStore`."""

    engine: InstanceOf[AsyncEngine] = Field(
        ..., description="Async SQLAlchemy engine"
    )

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


class SQLAlchemyFeatureStore(SemanticFeatureStore):
    """SQLAlchemy-backed :class:`SemanticFeatureStore`."""

    def __init__(self, params: SQLAlchemyFeatureStoreParams) -> None:
        self._engine = params.engine
        self._create_session = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
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
            await connection.run_sync(BaseFeatureStore.metadata.create_all)

    @override
    async def cleanup(self) -> None:
        await self._engine.dispose()

    @override
    async def delete_all(self) -> None:
        async with self._create_session() as session, session.begin():
            # CitationRow CASCADEs from FeatureRow, but delete explicitly
            # so the tables are left in a known state even if a caller
            # disabled FK enforcement.
            await session.execute(delete(CitationRow))
            await session.execute(delete(FeatureRow))
            await session.execute(delete(SetHistoryRow))

    @override
    async def reset_set_ids(self, set_ids: Sequence[SetIdT]) -> None:
        # Relational backend has no per-set cache to invalidate.
        pass

    # ------------------------------------------------------------------ #
    # Feature CRUD
    # ------------------------------------------------------------------ #

    @override
    async def add_feature(
        self,
        *,
        feature_id: UUID,
        set_id: SetIdT,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        stmt = insert(FeatureRow).values(
            uuid=feature_id,
            set_id=set_id,
            category_name=category_name,
            tag=sanitize_pg_text(tag, context="feature.tag"),
            feature=sanitize_pg_text(feature, context="feature.feature"),
            value=sanitize_pg_text(value, context="feature.value"),
            properties=dict(metadata) if metadata else {},
        )
        async with self._create_session() as session, session.begin():
            await session.execute(stmt)

    @override
    async def update_feature(
        self,
        feature_id: UUID,
        *,
        set_id: SetIdT | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if set_id is not None:
            values["set_id"] = set_id
        if category_name is not None:
            values["category_name"] = category_name
        if feature is not None:
            values["feature"] = sanitize_pg_text(feature, context="feature.feature")
        if value is not None:
            values["value"] = sanitize_pg_text(value, context="feature.value")
        if tag is not None:
            values["tag"] = sanitize_pg_text(tag, context="feature.tag")
        if metadata is not None:
            values["properties"] = dict(metadata)

        if not values:
            return

        stmt = (
            update(FeatureRow).where(FeatureRow.uuid == feature_id).values(**values)
        )
        async with self._create_session() as session, session.begin():
            await session.execute(stmt)

    @override
    async def get_feature(
        self,
        feature_id: UUID,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        async with self._create_session() as session:
            row = (
                await session.execute(
                    select(FeatureRow).where(FeatureRow.uuid == feature_id)
                )
            ).scalar_one_or_none()

            if row is None:
                return None

            citations_map: Mapping[UUID, Sequence[EpisodeIdT]] = {}
            if load_citations:
                citations_map = await self._load_citations(session, [row.uuid])

            return row.to_typed_model(citations=citations_map.get(row.uuid))

    @override
    async def get_features(
        self,
        feature_ids: Sequence[UUID],
        load_citations: bool = False,
    ) -> Mapping[UUID, SemanticFeature]:
        feature_ids = list(feature_ids)
        if not feature_ids:
            return {}

        async with self._create_session() as session:
            rows = (
                (
                    await session.execute(
                        select(FeatureRow).where(FeatureRow.uuid.in_(feature_ids))
                    )
                )
                .scalars()
                .all()
            )

            citations_map: Mapping[UUID, Sequence[EpisodeIdT]] = {}
            if load_citations and rows:
                citations_map = await self._load_citations(
                    session, [row.uuid for row in rows]
                )

        return {
            row.uuid: row.to_typed_model(citations=citations_map.get(row.uuid))
            for row in rows
        }

    @override
    async def get_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> AsyncIterator[SemanticFeature]:
        if page_num is not None and page_size is None:
            raise InvalidArgumentError("Cannot specify page_num without page_size")

        stmt: Select[Any] = select(FeatureRow).order_by(
            FeatureRow.created_at.asc(), FeatureRow.uuid.asc()
        )
        stmt = self._apply_feature_filter(stmt, filter_expr=filter_expr)

        if page_size is not None:
            stmt = stmt.limit(page_size).offset(page_size * (page_num or 0))

        buffering = load_citations or (
            tag_threshold is not None and tag_threshold > 0
        )

        async with self._create_session() as session:
            result = await session.stream(stmt)

            if not buffering:
                async for row in result.scalars():
                    yield row.to_typed_model()
                return

            rows = [row async for row in result.scalars()]

            if tag_threshold is not None and tag_threshold > 0:
                counts = Counter(row.tag for row in rows)
                rows = [row for row in rows if counts[row.tag] >= tag_threshold]

            citations_map: Mapping[UUID, Sequence[EpisodeIdT]] = {}
            if load_citations and rows:
                citations_map = await self._load_citations(
                    session, [row.uuid for row in rows]
                )

            for row in rows:
                yield row.to_typed_model(citations=citations_map.get(row.uuid))

    # ------------------------------------------------------------------ #
    # Feature deletion
    # ------------------------------------------------------------------ #

    @override
    async def delete_features(
        self,
        feature_ids: Sequence[UUID],
    ) -> None:
        feature_ids = list(feature_ids)
        if not feature_ids:
            return

        stmt = delete(FeatureRow).where(FeatureRow.uuid.in_(feature_ids))
        async with self._create_session() as session, session.begin():
            await session.execute(stmt)

    @override
    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> Sequence[UUID]:
        async with self._create_session() as session, session.begin():
            stmt: Delete = delete(FeatureRow).returning(FeatureRow.uuid)
            stmt = self._apply_feature_filter(stmt, filter_expr=filter_expr)
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    # ------------------------------------------------------------------ #
    # Citations
    # ------------------------------------------------------------------ #

    @override
    async def add_citations(
        self,
        feature_id: UUID,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        if not history_ids:
            return

        rows = [
            {"feature_uuid": feature_id, "history_id": str(history_id)}
            for history_id in history_ids
        ]
        async with self._create_session() as session, session.begin():
            await session.execute(insert(CitationRow), rows)

    # ------------------------------------------------------------------ #
    # History / ingestion tracking
    # ------------------------------------------------------------------ #

    @override
    async def add_history_to_set(
        self,
        set_id: SetIdT,
        history_id: EpisodeIdT,
    ) -> None:
        stmt = insert(SetHistoryRow).values(
            set_id=set_id, history_id=str(history_id)
        )
        async with self._create_session() as session, session.begin():
            await session.execute(stmt)

    @override
    async def get_history_messages(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> AsyncIterator[EpisodeIdT]:
        stmt: Select[Any] = select(SetHistoryRow.history_id).order_by(
            SetHistoryRow.history_id.asc()
        )
        stmt = self._apply_history_filter(
            stmt, set_ids=set_ids, is_ingested=is_ingested, limit=limit
        )
        async with self._create_session() as session:
            result = await session.stream(stmt)
            async for history_id in result.scalars():
                yield EpisodeIdT(history_id)

    @override
    async def get_history_messages_count(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        stmt: Select[Any] = select(func.count(SetHistoryRow.history_id))
        stmt = self._apply_history_filter(
            stmt, set_ids=set_ids, is_ingested=is_ingested
        )
        async with self._create_session() as session:
            return (await session.execute(stmt)).scalar_one()

    @override
    async def mark_messages_ingested(
        self,
        *,
        set_id: SetIdT,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        if not history_ids:
            raise ValueError("No history ids provided")

        stmt = (
            update(SetHistoryRow)
            .where(SetHistoryRow.set_id == set_id)
            .where(SetHistoryRow.history_id.in_([str(h) for h in history_ids]))
            .values(ingested=True)
        )
        async with self._create_session() as session, session.begin():
            await session.execute(stmt)

    @override
    async def delete_history(
        self,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        if not history_ids:
            return

        hids = [str(h) for h in history_ids]
        async with self._create_session() as session, session.begin():
            await session.execute(
                delete(CitationRow).where(CitationRow.history_id.in_(hids))
            )
            await session.execute(
                delete(SetHistoryRow).where(SetHistoryRow.history_id.in_(hids))
            )

    @override
    async def delete_history_set(
        self,
        set_ids: Sequence[SetIdT],
    ) -> None:
        if not set_ids:
            return

        stmt = delete(SetHistoryRow).where(SetHistoryRow.set_id.in_(set_ids))
        async with self._create_session() as session, session.begin():
            await session.execute(stmt)

    @override
    async def get_history_set_ids(
        self,
        *,
        min_uningested_messages: int | None = None,
        older_than: datetime | None = None,
    ) -> AsyncIterator[SetIdT]:
        subqueries: list[Select[Any]] = []

        if min_uningested_messages is not None and min_uningested_messages > 0:
            subqueries.append(
                select(SetHistoryRow.set_id)
                .where(SetHistoryRow.ingested.is_(False))
                .group_by(SetHistoryRow.set_id)
                .having(func.count() >= min_uningested_messages)
            )

        if older_than is not None:
            subqueries.append(
                select(SetHistoryRow.set_id)
                .where(
                    SetHistoryRow.ingested.is_(False),
                    SetHistoryRow.created_at <= older_than,
                )
                .distinct()
            )

        if not subqueries:
            stmt: Select[Any] = select(SetHistoryRow.set_id).distinct()
        elif len(subqueries) == 1:
            stmt = subqueries[0]
        else:
            stmt = union(*subqueries)

        async with self._create_session() as session:
            result = await session.stream(stmt)
            async for set_id in result.scalars():
                if set_id is not None:
                    yield SetIdT(set_id)

    @override
    async def purge_ingested_rows(self, set_ids: Sequence[SetIdT]) -> int:
        if not set_ids:
            return 0

        # Only purge set_ids where no uningested rows remain, so the
        # (set_id, history_id) duplicate guard stays intact for pending
        # sets.
        pending_alias = aliased(SetHistoryRow)
        pending_exists = (
            select(pending_alias.set_id)
            .where(
                pending_alias.set_id == SetHistoryRow.set_id,
                pending_alias.ingested.is_(False),
            )
            .correlate(SetHistoryRow)
            .exists()
        )
        stmt = delete(SetHistoryRow).where(
            SetHistoryRow.set_id.in_(set_ids),
            SetHistoryRow.ingested.is_(True),
            ~pending_exists,
        )
        async with self._create_session() as session, session.begin():
            result = cast(CursorResult[Any], await session.execute(stmt))
            return result.rowcount

    # ------------------------------------------------------------------ #
    # Set discovery
    # ------------------------------------------------------------------ #

    @override
    async def get_set_ids_starts_with(self, prefix: str) -> AsyncIterator[SetIdT]:
        stmt = union(
            select(SetHistoryRow.set_id).where(
                SetHistoryRow.set_id.startswith(prefix)
            ),
            select(FeatureRow.set_id).where(FeatureRow.set_id.startswith(prefix)),
        )
        async with self._create_session() as session:
            result = await session.stream(stmt)
            async for set_id in result.scalars():
                yield SetIdT(set_id)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _load_citations(
        self,
        session: AsyncSession,
        feature_uuids: Sequence[UUID],
    ) -> Mapping[UUID, Sequence[EpisodeIdT]]:
        if not feature_uuids:
            return {}

        stmt = select(CitationRow.feature_uuid, CitationRow.history_id).where(
            CitationRow.feature_uuid.in_(feature_uuids)
        )
        result = await session.execute(stmt)

        citations: MutableMapping[UUID, list[EpisodeIdT]] = {
            feature_uuid: [] for feature_uuid in feature_uuids
        }
        for feature_uuid, history_id in result:
            citations.setdefault(feature_uuid, []).append(EpisodeIdT(history_id))
        return citations

    def _apply_history_filter(
        self,
        stmt: Select[Any],
        *,
        set_ids: Sequence[SetIdT] | None = None,
        is_ingested: bool | None = None,
        limit: int | None = None,
    ) -> Select[Any]:
        if set_ids is not None and len(set_ids) > 0:
            stmt = stmt.where(SetHistoryRow.set_id.in_(set_ids))
        if is_ingested is not None:
            stmt = stmt.where(SetHistoryRow.ingested == is_ingested)
        if limit is not None:
            stmt = stmt.limit(limit)
        return stmt

    def _apply_feature_filter[StmtT: (Select[Any], Delete)](
        self,
        stmt: StmtT,
        *,
        filter_expr: FilterExpr | None,
    ) -> StmtT:
        if filter_expr is None:
            return stmt
        clause = compile_sql_filter(
            filter_expr, SQLAlchemyFeatureStore._resolve_feature_field
        )
        if clause is None:
            return stmt
        return stmt.where(clause)

    @staticmethod
    def _resolve_feature_field(
        field: str,
    ) -> tuple[
        MappedColumn[Any] | InstrumentedAttribute[Any] | ColumnElement[Any] | None,
        bool,
    ]:
        internal_name, is_user_property = normalize_filter_field(field)
        if is_user_property:
            key = demangle_user_metadata_key(internal_name)
            return FeatureRow.properties[key], True

        field_mapping: dict[str, InstrumentedAttribute[Any]] = {
            "set_id": FeatureRow.set_id,
            "set": FeatureRow.set_id,
            "category_name": FeatureRow.category_name,
            "category": FeatureRow.category_name,
            "semantic_category_id": FeatureRow.category_name,
            "tag": FeatureRow.tag,
            "tag_id": FeatureRow.tag,
            "feature": FeatureRow.feature,
            "feature_name": FeatureRow.feature,
            "value": FeatureRow.value,
            "created_at": FeatureRow.created_at,
            "updated_at": FeatureRow.updated_at,
        }
        if internal_name in field_mapping:
            return field_mapping[internal_name], False
        return None, False
