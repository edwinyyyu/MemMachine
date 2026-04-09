"""SQLite + pluggable vector search engine backed vector store.

SQLite stores collection metadata, record UUIDs, and properties.
A :class:`VectorSearchEngine` provides the index for vector search.
Vectors are stored in both SQLite (source of truth) and the engine (derived index).

Each logical collection gets its own records table and engine instance.
Different namespaces always get separate native tables and indexes.

Crash recovery: intended engine operations are recorded in a SQLite
``vector_store_sqlite_pending_ops`` table within the same transaction
as the data write.  After the engine is updated, the pending entry is
cleared.  On startup, any leftover pending ops are replayed so the index
converges with the SQLite source of truth.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import override
from uuid import UUID

import numpy as np
import sqlalchemy as sa
from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import JSON, LargeBinary, String, Text, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy.pool import StaticPool

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
)
from .utils import validate_filter, validate_identifier
from .vector_search_engine import SQLKeyFilter, VectorSearchEngine
from .vector_store import VectorStore, VectorStoreCollection


class BaseSQLiteVectorStore(DeclarativeBase):
    """Base class for SQLiteVectorStore ORM models."""


class _CollectionRow(BaseSQLiteVectorStore):
    __tablename__ = "vector_store_sqlite_collections"

    namespace: MappedColumn[str] = mapped_column(String(32), primary_key=True)
    name: MappedColumn[str] = mapped_column(String(32), primary_key=True)
    config_json: MappedColumn[dict[str, JsonValue]] = mapped_column(
        JSON, nullable=False
    )


class _PendingOpRow(BaseSQLiteVectorStore):
    """Pending engine index operations for crash recovery.

    Written in the same SQLite transaction as the data change.
    Cleared after the engine index is successfully updated.
    On startup, any remaining rows are replayed.
    """

    __tablename__ = "vector_store_sqlite_pending_ops"

    id: MappedColumn[int] = mapped_column(
        sa.Integer, primary_key=True, autoincrement=True
    )
    collection_prefix: MappedColumn[str] = mapped_column(Text, nullable=False)
    op: MappedColumn[str] = mapped_column(
        String(8), nullable=False
    )  # "upsert" or "remove"
    rowid: MappedColumn[int] = mapped_column(sa.Integer, nullable=False)


class SQLiteVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by SQLite + a pluggable vector search engine.

    Each logical collection has its own records table and engine instance,
    so KNN queries search only this collection's vectors directly.
    """

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        name: str,
        config: VectorStoreCollectionConfig,
        records_table: sa.Table,
        engine: VectorSearchEngine,
        sync_engine: sa.engine.Engine,
        collection_prefix: str,
        index_path: str | None,
        save_threshold: int,
    ) -> None:
        """Initialize with session factory, table, and search engine."""
        self._create_session = create_session
        self._name = name
        self._config = config
        self._records_table = records_table
        self._engine = engine
        self._sync_engine = sync_engine
        self._metric = config.similarity_metric

        self._score_is_better = (
            (lambda a, b: a >= b)
            if config.similarity_metric.higher_is_better
            else (lambda a, b: a <= b)
        )
        self._collection_prefix = collection_prefix
        self._index_path = index_path
        self._save_threshold = save_threshold
        self._ops_since_save: int = 0

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    async def _maybe_save_engine(self) -> None:
        """Save the engine index to disk if ops since last save exceed the threshold."""
        if (
            self._index_path is not None
            and self._ops_since_save >= self._save_threshold
        ):
            await self._engine.save(self._index_path)
            self._ops_since_save = 0
            async with self._create_session() as session, session.begin():
                await session.execute(
                    sa.delete(_PendingOpRow).where(
                        _PendingOpRow.collection_prefix == self._collection_prefix,
                    )
                )

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records_list = list(records)
        if not records_list:
            return

        records_table = self._records_table
        engine = self._engine
        collection_prefix = self._collection_prefix

        async with self._create_session() as session, session.begin():
            statement = sqlite_insert(records_table).on_conflict_do_update(
                index_elements=[records_table.c.uuid],
                set_={
                    "properties": sqlite_insert(records_table).excluded.properties,
                    "vector": sqlite_insert(records_table).excluded.vector,
                },
            )
            await session.execute(
                statement,
                [
                    {
                        "uuid": str(record.uuid),
                        "properties": encode_properties(record.properties),
                        "vector": (
                            np.array(record.vector, dtype=np.float32).tobytes()
                            if record.vector is not None
                            else None
                        ),
                    }
                    for record in records_list
                ],
            )

            # Fetch rowids for engine operations
            uuid_strs = [str(record.uuid) for record in records_list]
            rows = (
                await session.execute(
                    select(records_table.c.uuid, records_table.c.rowid).where(
                        records_table.c.uuid.in_(uuid_strs),
                    )
                )
            ).all()
            uuid_to_rowid: dict[str, int] = {row.uuid: row.rowid for row in rows}

            # Record pending ops in same transaction
            pending_rowids: list[int] = []
            for record in records_list:
                if record.vector is not None:
                    rowid = uuid_to_rowid[str(record.uuid)]
                    pending_rowids.append(rowid)
            if pending_rowids:
                await session.execute(
                    sa.insert(_PendingOpRow),
                    [
                        {
                            "collection_prefix": collection_prefix,
                            "op": "upsert",
                            "rowid": rowid,
                        }
                        for rowid in pending_rowids
                    ],
                )

        # SQLite committed — update engine index
        await self._apply_engine_upserts(records_list, uuid_to_rowid, engine)

    async def _apply_engine_upserts(
        self,
        records_list: list[Record],
        uuid_to_rowid: dict[str, int],
        engine: VectorSearchEngine,
    ) -> None:
        """Update engine index after SQLite commit."""
        engine_vectors: dict[int, list[float]] = {}
        for record in records_list:
            if record.vector is not None:
                rowid = uuid_to_rowid[str(record.uuid)]
                engine_vectors[rowid] = record.vector

        if engine_vectors:
            await engine.remove(engine_vectors.keys())
            await engine.add(engine_vectors)
            self._ops_since_save += len(engine_vectors)
            await self._maybe_save_engine()

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        score_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[QueryResult]:
        query_vectors_list = list(query_vectors)
        if not query_vectors_list:
            return []

        if limit is not None and limit <= 0:
            return [QueryResult(matches=[]) for _ in query_vectors_list]

        if property_filter is not None and not validate_filter(property_filter):
            raise ValueError("Filter contains invalid field names")

        key_filter = self._build_key_filter(property_filter)
        effective_limit = limit if limit is not None else 2**31 - 1

        try:
            search_results = await self._engine.search(
                query_vectors_list, effective_limit, allowed_keys=key_filter
            )
        finally:
            if key_filter is not None:
                key_filter.close()

        results: list[QueryResult] = []
        for search_result in search_results:
            if not search_result.matches:
                results.append(QueryResult(matches=[]))
                continue
            matches = await self._build_matches(
                rowid_to_score={m.key: m.score for m in search_result.matches},
                score_threshold=score_threshold,
                return_vector=return_vector,
                return_properties=return_properties,
            )
            results.append(QueryResult(matches=matches))

        return results

    def _build_key_filter(
        self, property_filter: FilterExpr | None
    ) -> SQLKeyFilter | None:
        """Build a SQLKeyFilter for the given property filter, or None."""
        if property_filter is None:
            return None
        return SQLKeyFilter(
            sync_engine=self._sync_engine,
            records_table=self._records_table,
            filter_expression=compile_sql_filter(
                property_filter,
                lambda field: (
                    self._records_table.c.properties[field],
                    "properties_json",
                ),
            ),
        )

    async def _build_matches(
        self,
        rowid_to_score: dict[int, float],
        score_threshold: float | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Fetch records for matched rowids and build QueryMatch list."""
        records_table = self._records_table

        columns = [records_table.c.uuid, records_table.c.rowid]
        if return_properties:
            columns.append(records_table.c.properties)
        if return_vector:
            columns.append(records_table.c.vector)

        statement = select(*columns).where(
            records_table.c.rowid.in_(list(rowid_to_score.keys())),
        )

        async with self._create_session() as session:
            rows = (await session.execute(statement)).all()

        score_is_better = self._score_is_better
        matches: list[QueryMatch] = []
        for row in rows:
            score = rowid_to_score.get(row.rowid)
            if score is None:
                continue

            if score_threshold is not None and not score_is_better(
                score, score_threshold
            ):
                continue

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = None
            if return_vector and row.vector is not None:
                vector = list(np.frombuffer(row.vector, dtype=np.float32))

            matches.append(
                QueryMatch(
                    score=score,
                    record=Record(
                        uuid=UUID(row.uuid), vector=vector, properties=properties
                    ),
                )
            )

        matches.sort(
            key=lambda match: match.score,
            reverse=self._metric.higher_is_better,
        )
        return matches

    @override
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
        uuid_list = list(record_uuids)
        if not uuid_list:
            return []

        records_table = self._records_table
        uuid_strs = [str(record_uuid) for record_uuid in uuid_list]

        columns = [records_table.c.uuid]
        if return_properties:
            columns.append(records_table.c.properties)
        if return_vector:
            columns.append(records_table.c.vector)

        async with self._create_session() as session:
            rows = (
                await session.execute(
                    select(*columns).where(
                        records_table.c.uuid.in_(uuid_strs),
                    )
                )
            ).all()

        record_map: dict[UUID, Record] = {}
        for row in rows:
            uuid_val = UUID(row.uuid)

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = None
            if return_vector and row.vector is not None:
                vector = list(np.frombuffer(row.vector, dtype=np.float32))

            record_map[uuid_val] = Record(
                uuid=uuid_val, vector=vector, properties=properties
            )

        return [
            record_map[record_uuid]
            for record_uuid in uuid_list
            if record_uuid in record_map
        ]

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        uuid_list = list(record_uuids)
        if not uuid_list:
            return

        records_table = self._records_table
        engine = self._engine
        collection_prefix = self._collection_prefix
        uuid_strs = [str(record_uuid) for record_uuid in uuid_list]

        async with self._create_session() as session, session.begin():
            rows = (
                await session.execute(
                    select(records_table.c.rowid).where(
                        records_table.c.uuid.in_(uuid_strs),
                    )
                )
            ).all()
            if not rows:
                return

            rowids = [row.rowid for row in rows]

            # Record pending remove ops
            await session.execute(
                sa.insert(_PendingOpRow),
                [
                    {
                        "collection_prefix": collection_prefix,
                        "op": "remove",
                        "rowid": rowid,
                    }
                    for rowid in rowids
                ],
            )

            # Delete from SQLite
            await session.execute(
                sa.delete(records_table).where(
                    records_table.c.uuid.in_(uuid_strs),
                )
            )

        # Remove from engine index
        await engine.remove(rowids)
        self._ops_since_save += len(rowids)
        await self._maybe_save_engine()


EngineFactory = Callable[[int, SimilarityMetric], VectorSearchEngine]
"""Callable that creates a VectorSearchEngine given (ndim, metric)."""


class SQLiteVectorStoreParams(BaseModel):
    """Parameters for constructing a SQLiteVectorStore.

    Attributes:
        engine: Async SQLAlchemy engine (sqlite+aiosqlite).
        index_directory: Directory for persisting index files.
            If None, indexes are in-memory only.
        engine_factory: Factory for creating :class:`VectorSearchEngine`
            instances.  Receives ``(ndim, metric)`` and returns an engine.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ..., description="Async SQLAlchemy engine (sqlite+aiosqlite)"
    )
    index_directory: str | None = Field(
        None, description="Directory for persisting index files"
    )
    engine_factory: EngineFactory = Field(
        ..., description="Factory for creating VectorSearchEngine instances"
    )
    save_threshold: int = Field(
        1000,
        description="Number of engine operations before auto-saving the index to disk. "
        "Only applies when index_directory is set.",
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
                "Engine uses ephemeral SQLite, where each connection gets a separate "
                "database. Use a file path instead."
            )
        return engine


async def _replay_upsert_one(
    search_engine: VectorSearchEngine, label: int, vector_data: np.ndarray
) -> None:
    """Replace a single vector in the engine (crash-recovery helper)."""
    await search_engine.remove([label])
    await search_engine.add({label: list(vector_data.flat)})


async def _replay_remove_one(search_engine: VectorSearchEngine, label: int) -> None:
    """Remove a single vector from the engine (crash-recovery helper)."""
    await search_engine.remove([label])


class SQLiteVectorStore(VectorStore):
    """Vector store backed by SQLite + a pluggable vector search engine.

    Each logical collection gets its own records table and engine instance.
    Vectors are stored in SQLite as source of truth; the engine is a derived
    index that can be rebuilt.
    """

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._db_engine = params.engine
        self._index_dir = (
            Path(params.index_directory) if params.index_directory else None
        )
        self._engine_factory = params.engine_factory
        self._save_threshold = params.save_threshold
        self._create_session = async_sessionmaker(
            self._db_engine, expire_on_commit=False
        )
        self._records_tables: dict[str, sa.Table] = {}
        self._search_engines: dict[str, VectorSearchEngine] = {}
        self._sa_metadata = sa.MetaData()

        # Independent sync engine for SQLKeyFilter (same database, separate pool).
        self._sync_engine = sa.create_engine(
            str(self._db_engine.url).replace("aiosqlite", "pysqlite")
        )

    @staticmethod
    def _collection_prefix(namespace: str, name: str) -> str:
        """Unique prefix for a logical collection's native resources."""
        return f"vector_store_sqlite_{namespace}_{name}"

    @staticmethod
    def _build_records_table(table_name: str, sa_metadata: sa.MetaData) -> sa.Table:
        return sa.Table(
            table_name,
            sa_metadata,
            sa.Column("rowid", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("uuid", sa.Text, nullable=False, unique=True),
            sa.Column("properties", JSON, nullable=True),
            sa.Column("vector", LargeBinary, nullable=True),
        )

    @override
    async def startup(self) -> None:
        if self._index_dir is not None:
            self._index_dir.mkdir(parents=True, exist_ok=True)
        async with self._db_engine.begin() as connection:
            await connection.run_sync(BaseSQLiteVectorStore.metadata.create_all)
        await self._replay_pending_ops()

    async def _replay_pending_ops(self) -> None:
        """Replay any pending engine operations from a prior crash."""
        async with self._create_session() as session:
            pending = (
                (
                    await session.execute(
                        select(_PendingOpRow).order_by(_PendingOpRow.id)
                    )
                )
                .scalars()
                .all()
            )

        if not pending:
            return

        ops_by_prefix: dict[str, list[_PendingOpRow]] = defaultdict(list)
        for op in pending:
            ops_by_prefix[op.collection_prefix].append(op)

        for collection_prefix, ops in ops_by_prefix.items():
            await self._replay_prefix_ops(collection_prefix, ops)

    async def _resolve_prefix(
        self, collection_prefix: str
    ) -> tuple[VectorSearchEngine, sa.Table] | None:
        """Resolve a collection prefix to its search engine and records table.

        Returns ``None`` if no stored collection matches the prefix.
        """
        async with self._create_session() as session:
            all_collections = (
                (await session.execute(select(_CollectionRow))).scalars().all()
            )

        for collection_row in all_collections:
            if (
                self._collection_prefix(collection_row.namespace, collection_row.name)
                == collection_prefix
            ):
                config = VectorStoreCollectionConfig.model_validate(
                    collection_row.config_json
                )
                search_engine = self._get_or_create_engine(collection_prefix, config)
                records_table = self._get_or_build_records_table(collection_prefix)
                return search_engine, records_table

        return None

    async def _replay_prefix_ops(
        self, collection_prefix: str, ops: list[_PendingOpRow]
    ) -> None:
        """Replay pending ops for a single collection."""
        resolved = await self._resolve_prefix(collection_prefix)
        if resolved is None:
            async with self._create_session() as session, session.begin():
                await session.execute(
                    sa.delete(_PendingOpRow).where(
                        _PendingOpRow.collection_prefix == collection_prefix
                    )
                )
            return

        search_engine, records_table = resolved

        for op in ops:
            if op.op == "upsert":
                async with self._create_session() as session:
                    row = (
                        await session.execute(
                            select(records_table.c.vector).where(
                                records_table.c.rowid == op.rowid
                            )
                        )
                    ).scalar_one_or_none()
                if row is not None:
                    vector = np.frombuffer(row, dtype=np.float32)
                    await _replay_upsert_one(search_engine, op.rowid, vector)
            elif op.op == "remove":
                await _replay_remove_one(search_engine, op.rowid)

    @override
    async def shutdown(self) -> None:
        if self._index_dir is not None:
            saved_prefixes: list[str] = []
            for collection_prefix, search_engine in self._search_engines.items():
                path = self._index_dir / f"{collection_prefix}.idx"
                await search_engine.save(str(path))
                saved_prefixes.append(collection_prefix)
            if saved_prefixes:
                async with self._create_session() as session, session.begin():
                    await session.execute(
                        sa.delete(_PendingOpRow).where(
                            _PendingOpRow.collection_prefix.in_(saved_prefixes),
                        )
                    )
        self._search_engines.clear()
        self._records_tables.clear()

    def _get_or_build_records_table(self, collection_prefix: str) -> sa.Table:
        if collection_prefix not in self._records_tables:
            table_name = f"{collection_prefix}_records"
            self._records_tables[collection_prefix] = self._build_records_table(
                table_name, self._sa_metadata
            )
        return self._records_tables[collection_prefix]

    def _get_or_create_engine(
        self,
        collection_prefix: str,
        config: VectorStoreCollectionConfig,
    ) -> VectorSearchEngine:
        if collection_prefix not in self._search_engines:
            search_engine = self._engine_factory(
                config.vector_dimensions, config.similarity_metric
            )

            if self._index_dir is not None:
                path = self._index_dir / f"{collection_prefix}.idx"
                if path.exists():
                    search_engine.load(str(path))

            self._search_engines[collection_prefix] = search_engine
        return self._search_engines[collection_prefix]

    async def _get_stored_config(
        self,
        session: AsyncSession,
        namespace: str,
        name: str,
    ) -> VectorStoreCollectionConfig | None:
        row = (
            await session.execute(
                select(_CollectionRow.config_json).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return VectorStoreCollectionConfig.model_validate(row)

    async def _ensure_native_tables(
        self,
        session: AsyncSession,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> tuple[sa.Table, VectorSearchEngine, str]:
        """Idempotently create per-collection native resources."""
        collection_prefix = self._collection_prefix(namespace, name)
        records_table = self._get_or_build_records_table(collection_prefix)
        search_engine = self._get_or_create_engine(collection_prefix, config)

        connection = await session.connection()
        await connection.run_sync(
            self._sa_metadata.create_all,
            tables=[records_table],
        )

        # Create property indexes
        for field_name in config.properties_schema:
            index_name = f"idx_{records_table.name}_{field_name}"
            safe_field = field_name.replace("'", "''")
            await session.execute(
                sa.text(
                    f"CREATE INDEX IF NOT EXISTS [{index_name}] "
                    f"ON [{records_table.name}]"
                    f"(json_extract(properties, '$.{safe_field}'))"
                )
            )

        return records_table, search_engine, collection_prefix

    def _build_collection_handle(
        self,
        name: str,
        config: VectorStoreCollectionConfig,
        records_table: sa.Table,
        search_engine: VectorSearchEngine,
        collection_prefix: str,
    ) -> SQLiteVectorStoreCollection:
        index_path = (
            str(self._index_dir / f"{collection_prefix}.idx")
            if self._index_dir is not None
            else None
        )
        return SQLiteVectorStoreCollection(
            create_session=self._create_session,
            name=name,
            config=config,
            records_table=records_table,
            engine=search_engine,
            sync_engine=self._sync_engine,
            collection_prefix=collection_prefix,
            index_path=index_path,
            save_threshold=self._save_threshold,
        )

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")

        async with self._create_session() as session, session.begin():
            existing = await self._get_stored_config(session, namespace, name)
            if existing is not None:
                raise VectorStoreCollectionAlreadyExistsError(namespace, name)

            await self._ensure_native_tables(session, namespace, name, config)
            session.add(
                _CollectionRow(
                    namespace=namespace,
                    name=name,
                    config_json=config.model_dump(mode="json"),
                )
            )

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> VectorStoreCollection:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")

        async with self._create_session() as session, session.begin():
            existing = await self._get_stored_config(session, namespace, name)
            if existing is not None:
                if existing != config:
                    raise VectorStoreCollectionConfigMismatchError(
                        namespace, name, existing, config
                    )
                (
                    records_table,
                    search_engine,
                    collection_prefix,
                ) = await self._ensure_native_tables(session, namespace, name, existing)
                return self._build_collection_handle(
                    name, existing, records_table, search_engine, collection_prefix
                )

            (
                records_table,
                search_engine,
                collection_prefix,
            ) = await self._ensure_native_tables(session, namespace, name, config)
            session.add(
                _CollectionRow(
                    namespace=namespace,
                    name=name,
                    config_json=config.model_dump(mode="json"),
                )
            )

        return self._build_collection_handle(
            name, config, records_table, search_engine, collection_prefix
        )

    @override
    async def open_collection(
        self,
        *,
        namespace: str,
        name: str,
    ) -> VectorStoreCollection | None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        async with self._create_session() as session:
            existing = await self._get_stored_config(session, namespace, name)
        if existing is None:
            return None

        collection_prefix = self._collection_prefix(namespace, name)
        records_table = self._get_or_build_records_table(collection_prefix)
        search_engine = self._get_or_create_engine(collection_prefix, existing)
        return self._build_collection_handle(
            name, existing, records_table, search_engine, collection_prefix
        )

    @override
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        pass

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        async with self._create_session() as session, session.begin():
            existing = await self._get_stored_config(session, namespace, name)
            if existing is None:
                return

            collection_prefix = self._collection_prefix(namespace, name)
            records_table = self._get_or_build_records_table(collection_prefix)

            # Drop the per-collection records table (cascades indexes)
            await session.execute(
                sa.text(f"DROP TABLE IF EXISTS [{records_table.name}]")
            )

            # Clear pending ops for this collection
            await session.execute(
                sa.delete(_PendingOpRow).where(
                    _PendingOpRow.collection_prefix == collection_prefix
                )
            )

            # Remove from registry
            await session.execute(
                sa.delete(_CollectionRow).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )

            # Clean up in-memory caches
            self._records_tables.pop(collection_prefix, None)
            self._search_engines.pop(collection_prefix, None)
            self._sa_metadata.remove(records_table)

            # Delete index file from disk
            if self._index_dir is not None:
                index_path = self._index_dir / f"{collection_prefix}.idx"
                if index_path.exists():
                    index_path.unlink()
