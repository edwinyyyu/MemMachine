"""
Vector store backed by SQLite + pluggable vector search engine.

Each logical collection gets its own records table and vector search engine.
A pending operations table tracks search engine operations for crash recovery:
On startup, unprocessed operations are replayed.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import override
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    Uuid,
    create_engine,
    delete,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, Session, mapped_column
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql.elements import ColumnElement

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
from .vector_search_engine import VectorSearchEngine
from .vector_store import VectorStore, VectorStoreCollection

logger = logging.getLogger(__name__)


class BaseSQLiteVectorStore(DeclarativeBase):
    """Base class for SQLiteVectorStore ORM models."""


class _CollectionRow(BaseSQLiteVectorStore):
    __tablename__ = "vector_store_sqlite_cl"

    namespace: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    name: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    config_json: MappedColumn[dict[str, JsonValue]] = mapped_column(
        JSON, nullable=False
    )


class _PendingOperationRow(BaseSQLiteVectorStore):
    """Pending engine index operations for crash recovery.

    Stores the latest unprocessed operation per (collection, record).
    Written in the same SQLite transaction as the data change.
    Marked completed after the engine processes the operation.
    Cleared after the engine is saved to disk.
    On startup, all remaining rows are replayed.
    """

    __tablename__ = "vector_store_sqlite_pd_op"

    collection_prefix: MappedColumn[str] = mapped_column(Text, primary_key=True)
    record_row_id: MappedColumn[int] = mapped_column(Integer, primary_key=True)
    operation_type: MappedColumn[str] = mapped_column(
        String(8), nullable=False
    )  # "upsert" or "delete"
    vector: MappedColumn[bytes | None] = mapped_column(LargeBinary, nullable=True)
    completed: MappedColumn[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )


class SQLiteVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by SQLite + a pluggable vector search engine.

    Each logical collection has its own records table and engine instance,
    so KNN queries search only this collection's vectors directly.
    """

    class _KeyFilter:
        """Per-candidate SQL filter using a sync SQLAlchemy session.

        Each ``__contains__`` call executes an indexed row ID lookup.
        Results are cached for the lifetime of the filter.
        The session is closed automatically when the filter is garbage-collected.
        """

        def __init__(
            self,
            sync_engine: Engine,
            records_table: Table,
            filter_expression: ColumnElement[bool],
        ) -> None:
            """Initialize with a sync engine, records table, and filter expression."""
            self._sync_engine = sync_engine
            self._records_table = records_table
            self._filter_expression = filter_expression
            self._cache: dict[int, bool] = {}
            self._session: Session | None = None

        def _get_session(self) -> Session:
            if self._session is None:
                self._session = Session(self._sync_engine)
            return self._session

        def __contains__(self, key: object) -> bool:
            """Return whether the key passes the SQL filter."""
            if not isinstance(key, int):
                return False
            if key in self._cache:
                return self._cache[key]
            row = (
                self._get_session()
                .execute(
                    select(self._records_table.c.row_id).where(
                        self._records_table.c.row_id == key,
                        self._filter_expression,
                    )
                )
                .scalar()
            )
            result = row is not None
            self._cache[key] = result
            return result

        def __del__(self) -> None:
            if self._session is not None:
                self._session.close()

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        name: str,
        config: VectorStoreCollectionConfig,
        records_table: Table,
        engine: VectorSearchEngine,
        sync_engine: Engine,
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

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    async def _maybe_save_engine(self) -> None:
        """Save the engine index to disk if completed pending operations exceed the threshold."""
        if self._index_path is None:
            return
        async with self._create_session() as session:
            count = (
                await session.execute(
                    select(func.count()).where(
                        _PendingOperationRow.collection_prefix
                        == self._collection_prefix,
                        _PendingOperationRow.completed.is_(True),
                    )
                )
            ).scalar_one()
        if count >= self._save_threshold:
            await self._engine.save(self._index_path)
            async with self._create_session() as session, session.begin():
                await session.execute(
                    delete(_PendingOperationRow).where(
                        _PendingOperationRow.collection_prefix
                        == self._collection_prefix,
                        _PendingOperationRow.completed.is_(True),
                    )
                )

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records_list = list(records)
        for record in records_list:
            if record.vector is None:
                raise ValueError(
                    f"Record {record.uuid} has vector=None, which is not allowed on input."
                )

        if not records_list:
            return

        records_table = self._records_table
        engine = self._engine
        collection_prefix = self._collection_prefix

        async with self._create_session() as session, session.begin():
            upsert_records = sqlite_insert(records_table).on_conflict_do_update(
                index_elements=[records_table.c.uuid],
                set_={
                    "properties": sqlite_insert(records_table).excluded.properties,
                },
            )
            await session.execute(
                upsert_records,
                [
                    {
                        "uuid": record.uuid,
                        "properties": encode_properties(record.properties),
                    }
                    for record in records_list
                ],
            )

            record_uuids = [record.uuid for record in records_list]
            rows = (
                await session.execute(
                    select(records_table.c.uuid, records_table.c.row_id).where(
                        records_table.c.uuid.in_(record_uuids),
                    )
                )
            ).all()
            uuid_to_row_id: dict[UUID, int] = {row.uuid: row.row_id for row in rows}

            pending_operation_values = [
                {
                    "collection_prefix": collection_prefix,
                    "record_row_id": uuid_to_row_id[record.uuid],
                    "operation_type": "upsert",
                    "vector": np.array(record.vector, dtype=np.float32).tobytes(),
                    "completed": False,
                }
                for record in records_list
            ]
            if pending_operation_values:
                upsert_pending_operation = sqlite_insert(_PendingOperationRow)
                await session.execute(
                    upsert_pending_operation.on_conflict_do_update(
                        index_elements=["collection_prefix", "record_row_id"],
                        set_={
                            "operation_type": upsert_pending_operation.excluded.operation_type,
                            "vector": upsert_pending_operation.excluded.vector,
                            "completed": upsert_pending_operation.excluded.completed,
                        },
                    ),
                    pending_operation_values,
                )

        await self._apply_engine_upserts(records_list, uuid_to_row_id, engine)

    async def _apply_engine_upserts(
        self,
        records_list: list[Record],
        uuid_to_row_id: dict[UUID, int],
        engine: VectorSearchEngine,
    ) -> None:
        """Update engine index after SQLite commit."""
        engine_vectors: dict[int, list[float]] = {
            uuid_to_row_id[record.uuid]: record.vector
            for record in records_list
            if record.vector is not None
        }

        if engine_vectors:
            await engine.remove(engine_vectors.keys())
            await engine.add(engine_vectors)
            async with self._create_session() as session, session.begin():
                await session.execute(
                    update(_PendingOperationRow)
                    .where(
                        _PendingOperationRow.collection_prefix
                        == self._collection_prefix,
                        _PendingOperationRow.record_row_id.in_(
                            list(engine_vectors.keys())
                        ),
                        _PendingOperationRow.completed.is_(False),
                    )
                    .values(completed=True)
                )
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

        search_results = await self._engine.search(
            query_vectors_list, k=effective_limit, allowed_keys=key_filter
        )

        results: list[QueryResult] = []
        for search_result in search_results:
            if not search_result.matches:
                results.append(QueryResult(matches=[]))
                continue
            matches = await self._build_matches(
                row_id_to_score={m.key: m.score for m in search_result.matches},
                score_threshold=score_threshold,
                return_vector=return_vector,
                return_properties=return_properties,
            )
            results.append(QueryResult(matches=matches))

        return results

    def _build_key_filter(
        self, property_filter: FilterExpr | None
    ) -> _KeyFilter | None:
        """Build a key filter for the given property filter, or None."""
        if property_filter is None:
            return None
        return SQLiteVectorStoreCollection._KeyFilter(
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
        row_id_to_score: dict[int, float],
        score_threshold: float | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Fetch records for matched row IDs and build QueryMatch list."""
        records_table = self._records_table
        matched_row_ids = list(row_id_to_score.keys())

        select_columns = [records_table.c.uuid, records_table.c.row_id]
        if return_properties:
            select_columns.append(records_table.c.properties)

        fetch_records = select(*select_columns).where(
            records_table.c.row_id.in_(matched_row_ids),
        )

        async with self._create_session() as session:
            matched_rows = (await session.execute(fetch_records)).all()

        vector_map: dict[int, list[float]] = {}
        if return_vector:
            vector_map = await self._engine.get_vectors(matched_row_ids)

        score_is_better = self._score_is_better
        matches: list[QueryMatch] = []
        for row in matched_rows:
            score = row_id_to_score.get(row.row_id)
            if score is None:
                continue

            if score_threshold is not None and not score_is_better(
                score, score_threshold
            ):
                continue

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = vector_map.get(row.row_id)

            matches.append(
                QueryMatch(
                    score=score,
                    record=Record(uuid=row.uuid, vector=vector, properties=properties),
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
        record_uuids = list(uuid_list)

        select_columns = [records_table.c.uuid, records_table.c.row_id]
        if return_properties:
            select_columns.append(records_table.c.properties)

        async with self._create_session() as session:
            fetched_rows = (
                await session.execute(
                    select(*select_columns).where(
                        records_table.c.uuid.in_(record_uuids),
                    )
                )
            ).all()

        vector_map: dict[int, list[float]] = {}
        if return_vector:
            vector_map = await self._engine.get_vectors(
                [row.row_id for row in fetched_rows]
            )

        record_map: dict[UUID, Record] = {}
        for row in fetched_rows:
            record_uuid = row.uuid

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = vector_map.get(row.row_id)

            record_map[record_uuid] = Record(
                uuid=record_uuid, vector=vector, properties=properties
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
        record_uuids = list(uuid_list)

        async with self._create_session() as session, session.begin():
            rows = (
                await session.execute(
                    select(records_table.c.row_id).where(
                        records_table.c.uuid.in_(record_uuids),
                    )
                )
            ).all()
            if not rows:
                return

            record_row_ids = [row.row_id for row in rows]

            upsert_pending_operation = sqlite_insert(_PendingOperationRow)
            await session.execute(
                upsert_pending_operation.on_conflict_do_update(
                    index_elements=["collection_prefix", "record_row_id"],
                    set_={
                        "operation_type": upsert_pending_operation.excluded.operation_type,
                        "completed": upsert_pending_operation.excluded.completed,
                    },
                ),
                [
                    {
                        "collection_prefix": collection_prefix,
                        "record_row_id": record_row_id,
                        "operation_type": "delete",
                        "completed": False,
                    }
                    for record_row_id in record_row_ids
                ],
            )

            await session.execute(
                delete(records_table).where(
                    records_table.c.uuid.in_(record_uuids),
                )
            )

        await engine.remove(record_row_ids)
        async with self._create_session() as session, session.begin():
            await session.execute(
                update(_PendingOperationRow)
                .where(
                    _PendingOperationRow.collection_prefix == collection_prefix,
                    _PendingOperationRow.record_row_id.in_(record_row_ids),
                    _PendingOperationRow.completed.is_(False),
                )
                .values(completed=True)
            )
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
        self._search_engines: dict[str, VectorSearchEngine] = {}
        self._sa_metadata = MetaData()

        self._sync_engine = create_engine(
            str(self._db_engine.url).replace("aiosqlite", "pysqlite")
        )

    @staticmethod
    def _collection_prefix(namespace: str, name: str) -> str:
        """Unique prefix for a logical collection's native resources."""
        return f"vector_store_sqlite_{len(namespace)}_{namespace}_{len(name)}_{name}"

    def _records_table(self, namespace: str, name: str) -> Table:
        """Get or create a SQLAlchemy Table for a per-collection records table."""
        collection_prefix = self._collection_prefix(namespace, name)
        return Table(
            f"{collection_prefix}_rc",
            self._sa_metadata,
            Column("row_id", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            Column("properties", JSON, nullable=False, default=dict),
            extend_existing=True,
        )

    @override
    async def startup(self) -> None:
        if self._index_dir is not None:
            self._index_dir.mkdir(parents=True, exist_ok=True)
        async with self._db_engine.begin() as connection:
            await connection.run_sync(BaseSQLiteVectorStore.metadata.create_all)
        await self._replay_pending_operations()

    async def _replay_pending_operations(self) -> None:
        """Replay any pending engine operations from a prior crash."""
        async with self._create_session() as session:
            pending_operations = (
                (await session.execute(select(_PendingOperationRow))).scalars().all()
            )

        if not pending_operations:
            return

        operations_by_prefix: dict[str, list[_PendingOperationRow]] = defaultdict(list)
        for operation in pending_operations:
            operations_by_prefix[operation.collection_prefix].append(operation)

        for collection_prefix, operations in operations_by_prefix.items():
            await self._replay_prefix_operations(collection_prefix, operations)

    async def _resolve_prefix(
        self, collection_prefix: str
    ) -> tuple[VectorSearchEngine, Table] | None:
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
                records_table = self._records_table(
                    collection_row.namespace, collection_row.name
                )
                return search_engine, records_table

        return None

    async def _replay_prefix_operations(
        self, collection_prefix: str, operations: list[_PendingOperationRow]
    ) -> None:
        """Replay pending operations for a single collection."""
        resolved = await self._resolve_prefix(collection_prefix)
        if resolved is None:
            async with self._create_session() as session, session.begin():
                await session.execute(
                    delete(_PendingOperationRow).where(
                        _PendingOperationRow.collection_prefix == collection_prefix
                    )
                )
            return

        search_engine, _ = resolved

        replayed_record_row_ids: list[int] = []
        for operation in operations:
            try:
                if operation.operation_type == "upsert":
                    if operation.vector is not None:
                        vector = np.frombuffer(operation.vector, dtype=np.float32)
                        await _replay_upsert_one(
                            search_engine, operation.record_row_id, vector
                        )
                    replayed_record_row_ids.append(operation.record_row_id)
                elif operation.operation_type == "delete":
                    await _replay_remove_one(search_engine, operation.record_row_id)
                    replayed_record_row_ids.append(operation.record_row_id)
            except Exception:
                logger.warning(
                    "Failed to replay %s operation for record_row_id %d in %s",
                    operation.operation_type,
                    operation.record_row_id,
                    collection_prefix,
                    exc_info=True,
                )

        if replayed_record_row_ids:
            async with self._create_session() as session, session.begin():
                await session.execute(
                    update(_PendingOperationRow)
                    .where(
                        _PendingOperationRow.collection_prefix == collection_prefix,
                        _PendingOperationRow.record_row_id.in_(replayed_record_row_ids),
                    )
                    .values(completed=True)
                )

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
                        delete(_PendingOperationRow).where(
                            _PendingOperationRow.collection_prefix.in_(saved_prefixes),
                            _PendingOperationRow.completed.is_(True),
                        )
                    )
        self._search_engines.clear()

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
    ) -> tuple[Table, VectorSearchEngine, str]:
        """Idempotently create per-collection native resources."""
        collection_prefix = self._collection_prefix(namespace, name)
        records_table = self._records_table(namespace, name)
        search_engine = self._get_or_create_engine(collection_prefix, config)

        connection = await session.connection()
        await connection.run_sync(
            self._sa_metadata.create_all,
            tables=[records_table],
        )

        return records_table, search_engine, collection_prefix

    def _build_collection_handle(
        self,
        name: str,
        config: VectorStoreCollectionConfig,
        records_table: Table,
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
        records_table = self._records_table(namespace, name)
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
            records_table = self._records_table(namespace, name)

            await session.execute(text(f"DROP TABLE IF EXISTS [{records_table.name}]"))

            await session.execute(
                delete(_PendingOperationRow).where(
                    _PendingOperationRow.collection_prefix == collection_prefix
                )
            )

            await session.execute(
                delete(_CollectionRow).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )

            self._search_engines.pop(collection_prefix, None)
            self._sa_metadata.remove(records_table)

            if self._index_dir is not None:
                index_path = self._index_dir / f"{collection_prefix}.idx"
                if index_path.exists():
                    index_path.unlink()
