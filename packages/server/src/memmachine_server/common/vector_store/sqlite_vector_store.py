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

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import override
from uuid import UUID
from weakref import WeakKeyDictionary

import numpy as np
import sqlalchemy as sa
from pydantic import BaseModel, Field, InstanceOf, JsonValue
from sqlalchemy import JSON, LargeBinary, String, Text, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column

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
        write_lock: asyncio.Lock,
        name: str,
        config: VectorStoreCollectionConfig,
        records_table: sa.Table,
        engine: VectorSearchEngine,
        db_path: str | None,
        collection_prefix: str,
    ) -> None:
        """Initialize with session factory, lock, table, and search engine."""
        self._create_session = create_session
        self._write_lock = write_lock
        self._name = name
        self._config = config
        self._records_table = records_table
        self._engine = engine
        self._db_path = db_path
        self._metric = config.similarity_metric

        self._score_is_better = (
            (lambda a, b: a >= b)
            if config.similarity_metric.higher_is_better
            else (lambda a, b: a <= b)
        )
        self._collection_prefix = collection_prefix

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    def _compile_property_filter(
        self, property_filter: FilterExpr
    ) -> sa.ColumnElement[bool]:
        """Compile a property filter against this collection's JSON properties column."""
        properties_column = self._records_table.c.properties
        return compile_sql_filter(
            property_filter,
            lambda field: (properties_column[field], "properties_json"),
        )

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records_list = list(records)
        if not records_list:
            return

        records_table = self._records_table
        engine = self._engine
        collection_prefix = self._collection_prefix

        async with self._write_lock:
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

            # SQLite committed — update engine index (still under write lock)
            await self._apply_engine_upserts(
                records_list, uuid_to_rowid, engine, collection_prefix
            )

    async def _apply_engine_upserts(
        self,
        records_list: list[Record],
        uuid_to_rowid: dict[str, int],
        engine: VectorSearchEngine,
        collection_prefix: str,
    ) -> None:
        """Update engine index after SQLite commit and clear pending ops."""
        labels: list[int] = []
        vectors: list[list[float]] = []
        for record in records_list:
            if record.vector is not None:
                rowid = uuid_to_rowid[str(record.uuid)]
                labels.append(rowid)
                vectors.append(record.vector)

        if labels:
            await asyncio.to_thread(engine.add, labels, vectors)

            async with self._create_session() as session, session.begin():
                await session.execute(
                    sa.delete(_PendingOpRow).where(
                        _PendingOpRow.collection_prefix == collection_prefix,
                        _PendingOpRow.rowid.in_(labels),
                    )
                )

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

        results: list[QueryResult] = []
        try:
            for query_vector in query_vectors_list:
                matches = await self._search_and_build_matches(
                    query_vector=list(query_vector),
                    score_threshold=score_threshold,
                    limit=limit,
                    key_filter=key_filter,
                    property_filter=property_filter if key_filter is None else None,
                    return_vector=return_vector,
                    return_properties=return_properties,
                )
                results.append(QueryResult(matches=matches))
        finally:
            if key_filter is not None:
                key_filter.close()

        return results

    def _build_key_filter(
        self, property_filter: FilterExpr | None
    ) -> SQLKeyFilter | None:
        """Build a SQLKeyFilter for the given property filter, or None.

        Returns None for in-memory databases (no file path to open a
        second sync connection) or when no filter is needed.
        """
        if property_filter is None or self._db_path is None:
            return None
        compiled = self._compile_property_filter(property_filter)
        from sqlalchemy.dialects import sqlite as sa_sqlite

        filter_sql = str(
            compiled.compile(
                dialect=sa_sqlite.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )
        return SQLKeyFilter(
            db_path=self._db_path,
            table_name=self._records_table.name,
            filter_sql=filter_sql,
        )

    async def _search_and_build_matches(
        self,
        query_vector: list[float],
        score_threshold: float | None,
        limit: int | None,
        key_filter: SQLKeyFilter | None,
        property_filter: FilterExpr | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Run engine search and build matches.

        If ``key_filter`` is provided, the engine filters during search.
        If ``property_filter`` is provided (in-memory fallback), SQL
        post-filtering is applied after fetching records.
        """
        engine = self._engine
        if len(engine) == 0:
            return []

        effective_limit = min(limit if limit is not None else len(engine), len(engine))

        result = await asyncio.to_thread(
            engine.search, query_vector, effective_limit, key_filter=key_filter
        )

        if not result.keys:
            return []

        return await self._build_matches(
            rowid_to_score=dict(zip(result.keys, result.scores, strict=True)),
            score_threshold=score_threshold,
            property_filter=property_filter,
            return_vector=return_vector,
            return_properties=return_properties,
        )

    async def _build_matches(
        self,
        rowid_to_score: dict[int, float],
        score_threshold: float | None,
        property_filter: FilterExpr | None,
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

        if property_filter is not None:
            compiled_filter = self._compile_property_filter(property_filter)
            if compiled_filter is not None:
                statement = statement.where(compiled_filter)

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

        async with self._write_lock:
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

            # Remove from engine index (still under write lock)
            await asyncio.to_thread(engine.remove, rowids)

            # Clear pending ops
            async with self._create_session() as session, session.begin():
                await session.execute(
                    sa.delete(_PendingOpRow).where(
                        _PendingOpRow.collection_prefix == collection_prefix,
                        _PendingOpRow.rowid.in_(rowids),
                    )
                )


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


def _replay_upsert_one(
    search_engine: VectorSearchEngine, label: int, vector_data: np.ndarray
) -> None:
    """Replace a single vector in the engine (crash-recovery helper)."""
    search_engine.add([label], [list(vector_data.flat)])


def _replay_remove_one(search_engine: VectorSearchEngine, label: int) -> None:
    """Remove a single vector from the engine (crash-recovery helper)."""
    search_engine.remove([label])


class SQLiteVectorStore(VectorStore):
    """Vector store backed by SQLite + a pluggable vector search engine.

    Each logical collection gets its own records table and engine instance.
    Vectors are stored in SQLite as source of truth; the engine is a derived
    index that can be rebuilt.
    """

    # Shared across all instances so that stores using the same db engine
    # serialise SQLite writes through the same lock.
    # Keyed by db engine so locks are garbage-collected when the engine is.
    _write_locks: WeakKeyDictionary[AsyncEngine, asyncio.Lock] = WeakKeyDictionary()
    _name_locks_by_engine: WeakKeyDictionary[
        AsyncEngine, defaultdict[tuple[str, str], asyncio.Lock]
    ] = WeakKeyDictionary()

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._db_engine = params.engine
        self._index_dir = (
            Path(params.index_directory) if params.index_directory else None
        )
        self._engine_factory = params.engine_factory
        self._create_session = async_sessionmaker(
            self._db_engine, expire_on_commit=False
        )
        self._records_tables: dict[str, sa.Table] = {}
        self._search_engines: dict[str, VectorSearchEngine] = {}
        self._sa_metadata = sa.MetaData()

        # Extract file path for sync SQLKeyFilter connections.
        # None for in-memory databases (falls back to SQL post-filtering).
        url = sa.make_url(str(self._db_engine.url))
        db = url.database
        self._db_path: str | None = db if db and db != ":memory:" else None

    @property
    def _write_lock(self) -> asyncio.Lock:
        return SQLiteVectorStore._write_locks.setdefault(
            self._db_engine, asyncio.Lock()
        )

    @property
    def _name_locks(self) -> defaultdict[tuple[str, str], asyncio.Lock]:
        return SQLiteVectorStore._name_locks_by_engine.setdefault(
            self._db_engine, defaultdict(asyncio.Lock)
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
                    await asyncio.to_thread(
                        _replay_upsert_one, search_engine, op.rowid, vector
                    )
            elif op.op == "remove":
                await asyncio.to_thread(_replay_remove_one, search_engine, op.rowid)

        operation_ids = [op.id for op in ops]
        async with self._create_session() as session, session.begin():
            await session.execute(
                sa.delete(_PendingOpRow).where(_PendingOpRow.id.in_(operation_ids))
            )

    @override
    async def shutdown(self) -> None:
        if self._index_dir is not None:
            for collection_prefix, search_engine in self._search_engines.items():
                path = self._index_dir / f"{collection_prefix}.idx"
                await asyncio.to_thread(search_engine.save, str(path))
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
        return SQLiteVectorStoreCollection(
            create_session=self._create_session,
            write_lock=self._write_lock,
            name=name,
            config=config,
            records_table=records_table,
            engine=search_engine,
            db_path=self._db_path,
            collection_prefix=collection_prefix,
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

        lock = self._name_locks[(namespace, name)]
        async with (
            lock,
            self._write_lock,
            self._create_session() as session,
            session.begin(),
        ):
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

        lock = self._name_locks[(namespace, name)]
        async with (
            lock,
            self._write_lock,
            self._create_session() as session,
            session.begin(),
        ):
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
        lock = self._name_locks[(namespace, name)]
        async with (
            lock,
            self._write_lock,
            self._create_session() as session,
            session.begin(),
        ):
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
