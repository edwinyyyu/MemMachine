"""
Vector store backed by SQLite + sqlite-vec.

Each logical collection gets its own records table and vec0 virtual table.
Partition keys are avoided in favor of per-collection tables,
since sqlite-vec ANN indexes may not support them.
"""

import struct
from collections.abc import Iterable, Sequence
from typing import ClassVar, override
from uuid import UUID

import aiosqlite
import sqlite_vec
from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Uuid,
    delete,
    event,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool

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
from .vector_store import VectorStore, VectorStoreCollection


class BaseSQLiteVecVectorStore(DeclarativeBase):
    """Base class for SQLiteVecVectorStore ORM models."""


class _CollectionRow(BaseSQLiteVecVectorStore):
    __tablename__ = "vector_store_sqlite_vec_cl"

    namespace: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    name: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    config_json: MappedColumn[dict[str, JsonValue]] = mapped_column(
        JSON, nullable=False
    )


class SQLiteVecVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by SQLite + sqlite-vec."""

    _DISTANCE_FUNCTIONS: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "vec_distance_cosine",
        SimilarityMetric.EUCLIDEAN: "vec_distance_L2",
    }

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        name: str,
        config: VectorStoreCollectionConfig,
        records_table: Table,
        vector_table_name: str,
    ) -> None:
        """Initialize with session factory and table references."""
        self._create_session = create_session
        self._name = name
        self._config = config
        self._records_table = records_table
        self._vector_table_name = vector_table_name

        self._similarity_metric = config.similarity_metric
        self._distance_function = self._DISTANCE_FUNCTIONS[config.similarity_metric]

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    @staticmethod
    def _serialize_vector(vector: Sequence[float]) -> bytes:
        return sqlite_vec.serialize_float32(list(vector))

    @staticmethod
    def _deserialize_vector(data: bytes) -> list[float]:
        count = len(data) // 4
        return list(struct.unpack(f"={count}f", data))

    @staticmethod
    def _distance_to_score(
        distance: float, similarity_metric: SimilarityMetric
    ) -> float:
        match similarity_metric:
            case SimilarityMetric.COSINE:
                return 1.0 - distance
            case SimilarityMetric.EUCLIDEAN:
                return distance
            case _:
                raise NotImplementedError(similarity_metric)

    @staticmethod
    def _threshold_to_max_distance(
        threshold: float, similarity_metric: SimilarityMetric
    ) -> float:
        match similarity_metric:
            case SimilarityMetric.COSINE:
                return 1.0 - threshold
            case SimilarityMetric.EUCLIDEAN:
                return threshold
            case _:
                raise NotImplementedError(similarity_metric)

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records = list(records)
        if not records:
            return

        records_table = self._records_table
        vector_table_name = self._vector_table_name

        async with self._create_session() as session, session.begin():
            upsert_records = sqlite_insert(records_table).on_conflict_do_update(
                index_elements=[records_table.c.uuid],
                set_={"properties": sqlite_insert(records_table).excluded.properties},
            )
            await session.execute(
                upsert_records,
                [
                    {
                        "uuid": record.uuid,
                        "properties": encode_properties(record.properties),
                    }
                    for record in records
                ],
            )

            record_uuids = [record.uuid for record in records]
            rows = (
                await session.execute(
                    select(records_table.c.uuid, records_table.c.rowid).where(
                        records_table.c.uuid.in_(record_uuids),
                    )
                )
            ).all()
            uuid_to_rowid: dict[UUID, int] = {row.uuid: row.rowid for row in rows}

            vector_params = [
                {
                    "rowid": uuid_to_rowid[record.uuid],
                    "vector": self._serialize_vector(record.vector),
                }
                for record in records
                if record.vector is not None
            ]
            if vector_params:
                await session.execute(
                    text(f"DELETE FROM [{vector_table_name}] WHERE rowid = :rowid"),
                    vector_params,
                )
                await session.execute(
                    text(
                        f"INSERT INTO [{vector_table_name}](rowid, vector) "
                        f"VALUES (:rowid, :vector)"
                    ),
                    vector_params,
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
        query_vectors = list(query_vectors)
        if not query_vectors:
            return []

        if limit is not None and limit <= 0:
            return [QueryResult(matches=[]) for _ in query_vectors]

        if property_filter is not None and not validate_filter(property_filter):
            raise ValueError("Filter contains invalid field names")

        results: list[QueryResult] = []
        async with self._create_session() as session:
            for query_vector in query_vectors:
                query_blob = self._serialize_vector(query_vector)
                effective_limit = min(limit, 4096) if limit is not None else 4096

                knn_rows = (
                    await session.execute(
                        text(
                            f"SELECT rowid, distance FROM [{self._vector_table_name}] "
                            f"WHERE vector MATCH :query AND k = :k "
                            f"ORDER BY distance"
                        ),
                        {"query": query_blob, "k": effective_limit},
                    )
                ).all()

                if not knn_rows:
                    results.append(QueryResult(matches=[]))
                    continue

                rowid_to_distance: dict[int, float] = {
                    row.rowid: row.distance for row in knn_rows
                }
                matches = await self._build_matches(
                    session,
                    rowid_to_distance,
                    score_threshold=score_threshold,
                    property_filter=property_filter,
                    return_vector=return_vector,
                    return_properties=return_properties,
                )
                results.append(QueryResult(matches=matches))

        return results

    async def _build_matches(
        self,
        session: AsyncSession,
        rowid_to_distance: dict[int, float],
        score_threshold: float | None,
        property_filter: FilterExpr | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Fetch record data for row IDs and build matches with optional post-filter."""
        records_table = self._records_table

        matched_rowids = list(rowid_to_distance.keys())

        select_columns = [records_table.c.uuid, records_table.c.rowid]
        if return_properties:
            select_columns.append(records_table.c.properties)
        fetch_records = select(*select_columns).where(
            records_table.c.rowid.in_(matched_rowids),
        )
        if property_filter is not None:
            fetch_records = fetch_records.where(
                compile_sql_filter(
                    property_filter,
                    lambda field: (
                        records_table.c.properties[field],
                        "properties_json",
                    ),
                )
            )
        matched_rows = (await session.execute(fetch_records)).all()

        rowid_to_vector_bytes: dict[int, bytes] = {}
        if return_vector:
            rowid_to_vector_bytes = await self._fetch_vectors(
                session, [row.rowid for row in matched_rows]
            )

        matches: list[QueryMatch] = []
        for row in matched_rows:
            distance = rowid_to_distance.get(row.rowid)
            if distance is None:
                continue

            score = self._distance_to_score(distance, self._similarity_metric)
            if score_threshold is not None and (
                score < score_threshold
                if self._similarity_metric.higher_is_better
                else score > score_threshold
            ):
                continue

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = None
            if return_vector:
                vector_bytes = rowid_to_vector_bytes.get(row.rowid)
                if vector_bytes is not None:
                    vector = self._deserialize_vector(vector_bytes)

            matches.append(
                QueryMatch(
                    score=score,
                    record=Record(uuid=row.uuid, vector=vector, properties=properties),
                )
            )

        matches.sort(
            key=lambda match: match.score,
            reverse=self._similarity_metric.higher_is_better,
        )
        return matches

    async def _fetch_vectors(
        self, session: AsyncSession, rowids: list[int]
    ) -> dict[int, bytes]:
        """Fetch serialized vectors from the vec0 table by rowid."""
        if not rowids:
            return {}
        placeholders = ", ".join(f":r{i}" for i in range(len(rowids)))
        vector_rows = (
            await session.execute(
                text(
                    f"SELECT rowid, vector FROM [{self._vector_table_name}] "
                    f"WHERE rowid IN ({placeholders})"
                ),
                {f"r{i}": rowid for i, rowid in enumerate(rowids)},
            )
        ).all()
        return {row.rowid: row.vector for row in vector_rows}

    @override
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
        record_uuids = list(record_uuids)
        if not record_uuids:
            return []

        records_table = self._records_table

        async with self._create_session() as session:
            select_columns = [records_table.c.uuid, records_table.c.rowid]
            if return_properties:
                select_columns.append(records_table.c.properties)
            fetched_rows = (
                await session.execute(
                    select(*select_columns).where(
                        records_table.c.uuid.in_(record_uuids),
                    )
                )
            ).all()

            rowid_to_vector_bytes: dict[int, bytes] = {}
            if return_vector:
                rowid_to_vector_bytes = await self._fetch_vectors(
                    session, [row.rowid for row in fetched_rows]
                )

        record_map: dict[UUID, Record] = {}
        for row in fetched_rows:
            record_uuid = row.uuid

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = None
            if return_vector:
                vector_bytes = rowid_to_vector_bytes.get(row.rowid)
                if vector_bytes is not None:
                    vector = self._deserialize_vector(vector_bytes)

            record_map[record_uuid] = Record(
                uuid=record_uuid, vector=vector, properties=properties
            )

        return [
            record_map[record_uuid]
            for record_uuid in record_uuids
            if record_uuid in record_map
        ]

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        uuid_list = list(record_uuids)
        if not uuid_list:
            return

        records_table = self._records_table
        vector_table_name = self._vector_table_name
        record_uuids = list(uuid_list)

        async with self._create_session() as session, session.begin():
            rows = (
                await session.execute(
                    select(records_table.c.rowid).where(
                        records_table.c.uuid.in_(record_uuids),
                    )
                )
            ).all()
            if not rows:
                return

            record_rowids = [row.rowid for row in rows]

            placeholders = ", ".join(
                f":r{index}" for index in range(len(record_rowids))
            )
            await session.execute(
                text(
                    f"DELETE FROM [{vector_table_name}] WHERE rowid IN ({placeholders})"
                ),
                {f"r{index}": row_id for index, row_id in enumerate(record_rowids)},
            )

            await session.execute(
                delete(records_table).where(
                    records_table.c.uuid.in_(record_uuids),
                )
            )


class SQLiteVecVectorStoreParams(BaseModel):
    """Parameters for constructing a SQLiteVecVectorStore.

    Attributes:
        engine: Async SQLAlchemy engine with sqlite-vec loaded.
            Use :meth:`SQLiteVecVectorStore.create_engine` to create one.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="Async SQLAlchemy engine (sqlite+aiosqlite) with sqlite-vec loaded",
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


class SQLiteVecVectorStore(VectorStore):
    """Vector store backed by SQLite + sqlite-vec.

    Each logical collection gets its own records table and vec0 virtual table.
    """

    _SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "cosine",
        SimilarityMetric.EUCLIDEAN: "L2",
    }

    def __init__(self, params: SQLiteVecVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._records_tables: dict[str, Table] = {}
        self._sa_metadata = MetaData()

    @staticmethod
    def create_engine(url: str) -> AsyncEngine:
        """Create an ``AsyncEngine`` with sqlite-vec loaded on every connection.

        Args:
            url: SQLAlchemy database URL, e.g. ``"sqlite+aiosqlite:///path.db"``
                 or ``"sqlite+aiosqlite:///:memory:"``.
        """
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(url)

        @event.listens_for(engine.sync_engine, "connect")
        def _on_connect(
            dbapi_connection: DBAPIConnection,
            _connection_record: ConnectionPoolEntry,
        ) -> None:
            async def _load_extension(
                aio_connection: aiosqlite.Connection,
            ) -> None:
                await aio_connection.enable_load_extension(True)
                await aio_connection.load_extension(sqlite_vec.loadable_path())
                await aio_connection.enable_load_extension(False)

            dbapi_connection.run_async(_load_extension)

        return engine

    @staticmethod
    def _collection_prefix(namespace: str, name: str) -> str:
        """Unique prefix for a logical collection's native resources."""
        return f"vector_store_sqlite_vec_{namespace}_{name}"

    @staticmethod
    def _validate_metric(similarity_metric: SimilarityMetric) -> None:
        if (
            similarity_metric
            not in SQLiteVecVectorStore._SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE
        ):
            supported = ", ".join(
                similarity_metric.value
                for similarity_metric in SQLiteVecVectorStore._SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE
            )
            raise ValueError(
                f"sqlite-vec only supports {supported} similarity metrics, "
                f"got {similarity_metric.value!r}"
            )

    @staticmethod
    def _build_records_table(table_name: str, sa_metadata: MetaData) -> Table:
        """Build a SQLAlchemy Core Table for a per-collection records table."""
        return Table(
            table_name,
            sa_metadata,
            Column("rowid", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            Column("properties", JSON, nullable=True),
        )

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BaseSQLiteVecVectorStore.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        self._records_tables.clear()

    def _get_or_build_records_table(self, collection_prefix: str) -> Table:
        if collection_prefix not in self._records_tables:
            table_name = f"{collection_prefix}_rc"
            self._records_tables[collection_prefix] = self._build_records_table(
                table_name, self._sa_metadata
            )
        return self._records_tables[collection_prefix]

    async def _get_stored_config(
        self, session: AsyncSession, namespace: str, name: str
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
    ) -> tuple[Table, str]:
        """Idempotently create per-collection native tables."""
        collection_prefix = self._collection_prefix(namespace, name)
        records_table = self._get_or_build_records_table(collection_prefix)
        vector_table_name = f"{collection_prefix}_vc"
        distance_metric_value = self._SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE[
            config.similarity_metric
        ]

        connection = await session.connection()
        await connection.run_sync(
            self._sa_metadata.create_all,
            tables=[records_table],
        )

        await session.execute(
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS [{vector_table_name}] USING vec0("
                f"vector float[{config.vector_dimensions}] distance_metric={distance_metric_value}"
                f")"
            )
        )

        for field_name in config.properties_schema:
            safe_field = field_name.replace("'", "''")
            index_name = f"idx_{records_table.name}_{field_name}_v"
            await session.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS [{index_name}] "
                    f"ON [{records_table.name}]"
                    f"(json_extract(properties, '$.{safe_field}.v'))"
                )
            )

        return records_table, vector_table_name

    def _build_collection_handle(
        self,
        name: str,
        config: VectorStoreCollectionConfig,
        records_table: Table,
        vector_table_name: str,
    ) -> SQLiteVecVectorStoreCollection:
        return SQLiteVecVectorStoreCollection(
            create_session=self._create_session,
            name=name,
            config=config,
            records_table=records_table,
            vector_table_name=vector_table_name,
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
        self._validate_metric(config.similarity_metric)

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
        self._validate_metric(config.similarity_metric)

        async with self._create_session() as session, session.begin():
            existing = await self._get_stored_config(session, namespace, name)
            if existing is not None:
                if existing != config:
                    raise VectorStoreCollectionConfigMismatchError(
                        namespace, name, existing, config
                    )
                records_table, vector_table_name = await self._ensure_native_tables(
                    session, namespace, name, existing
                )
                return self._build_collection_handle(
                    name, existing, records_table, vector_table_name
                )

            records_table, vector_table_name = await self._ensure_native_tables(
                session, namespace, name, config
            )
            session.add(
                _CollectionRow(
                    namespace=namespace,
                    name=name,
                    config_json=config.model_dump(mode="json"),
                )
            )

        return self._build_collection_handle(
            name, config, records_table, vector_table_name
        )

    @override
    async def open_collection(
        self, *, namespace: str, name: str
    ) -> VectorStoreCollection | None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        async with self._create_session() as session:
            existing = await self._get_stored_config(session, namespace, name)
        if existing is None:
            return None

        collection_prefix = self._collection_prefix(namespace, name)
        records_table = self._get_or_build_records_table(collection_prefix)
        vector_table_name = f"{collection_prefix}_vc"
        return self._build_collection_handle(
            name, existing, records_table, vector_table_name
        )

    @override
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        pass  # No resources to release

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
            vector_table_name = f"{collection_prefix}_vc"

            await session.execute(text(f"DROP TABLE IF EXISTS [{vector_table_name}]"))
            await session.execute(text(f"DROP TABLE IF EXISTS [{records_table.name}]"))

            await session.execute(
                delete(_CollectionRow).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )

            self._records_tables.pop(collection_prefix, None)
            self._sa_metadata.remove(records_table)
