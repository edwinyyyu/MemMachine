"""SQLite-based vector store implementation using sqlite-vec."""

import asyncio
import json
import logging
import sqlite3
import struct
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import override
from uuid import UUID

import sqlite_vec
from pydantic import BaseModel, Field, InstanceOf
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .data_types import (
    CollectionAlreadyExistsError,
    CollectionConfig,
    CollectionConfigMismatchError,
    QueryMatch,
    QueryResult,
    Record,
)
from .utils import validate_filter, validate_identifier
from .vector_store import Collection, VectorStore

logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {SimilarityMetric.COSINE, SimilarityMetric.EUCLIDEAN}

_METRIC_TO_VEC0: dict[SimilarityMetric, str] = {
    SimilarityMetric.COSINE: "cosine",
    SimilarityMetric.EUCLIDEAN: "L2",
}

_METRIC_TO_DISTANCE_FN: dict[SimilarityMetric, str] = {
    SimilarityMetric.COSINE: "vec_distance_cosine",
    SimilarityMetric.EUCLIDEAN: "vec_distance_L2",
}

_BindParams = dict[str, str | int | float | bytes | None]


def _vec_table_name(namespace: str, dimensions: int, metric: SimilarityMetric) -> str:
    """Return the shared vec0 virtual table name for a (namespace, dimensions, metric) combo."""
    return f"_vs_vec_{namespace}_{dimensions}_{_METRIC_TO_VEC0[metric]}"


def _validate_metric(metric: SimilarityMetric) -> None:
    if metric not in _SUPPORTED_METRICS:
        msg = f"Similarity metric {metric.value!r} is not supported by sqlite-vec. Use COSINE or EUCLIDEAN."
        raise ValueError(msg)


def _serialize_vector(vector: list[float]) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    return sqlite_vec.serialize_float32(vector)


def _deserialize_vector(blob: bytes) -> list[float]:
    """Deserialize bytes from sqlite-vec to a float vector."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _serialize_properties(
    properties: dict[str, PropertyValue] | None,
) -> str | None:
    """Serialize properties dict to JSON string."""
    if properties is None:
        return None

    def _convert(v: PropertyValue) -> PropertyValue | str:
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    return json.dumps({k: _convert(v) for k, v in properties.items()})


def _deserialize_properties(
    raw: str | None,
    schema: dict[str, type[PropertyValue]] | None,
) -> dict[str, PropertyValue] | None:
    """Deserialize JSON string to properties dict, using schema for type coercion."""
    if raw is None:
        return None
    data: dict[str, PropertyValue] = json.loads(raw)
    if schema is None:
        return data
    result: dict[str, PropertyValue] = {}
    for k, v in data.items():
        expected_type = schema.get(k)
        if expected_type is datetime and isinstance(v, str):
            result[k] = datetime.fromisoformat(v)
        else:
            result[k] = v
    return result


def _serialize_schema(
    schema: Mapping[str, type[PropertyValue]] | None,
) -> str | None:
    """Serialize properties_schema to JSON."""
    if schema is None:
        return None
    return json.dumps({k: PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[v] for k, v in schema.items()})


def _deserialize_schema(
    raw: str | None,
) -> dict[str, type[PropertyValue]] | None:
    """Deserialize JSON to properties_schema."""
    if raw is None:
        return None
    data: dict[str, str] = json.loads(raw)
    return {k: PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE[v] for k, v in data.items()}


def _distance_to_similarity(distance: float, metric: SimilarityMetric) -> float:
    """Convert a distance value to a similarity score."""
    if metric == SimilarityMetric.COSINE:
        return 1.0 - distance
    # EUCLIDEAN
    return 1.0 / (1.0 + distance)


def _similarity_to_max_distance(
    threshold: float,
    metric: SimilarityMetric,
) -> float:
    """Convert a similarity threshold to a maximum distance for SQL filtering."""
    if metric == SimilarityMetric.COSINE:
        return 1.0 - threshold
    # EUCLIDEAN: similarity = 1/(1+d) => d = (1-t)/t
    if threshold <= 0:
        return float("inf")
    return (1.0 - threshold) / threshold


class _FilterCompiler:
    """Compiles FilterExpr trees into SQL WHERE clauses with bound parameters."""

    def __init__(self) -> None:
        self._param_counter = 0
        self.params: _BindParams = {}

    def _next_param(self) -> str:
        self._param_counter += 1
        return f"_fv{self._param_counter}"

    def compile(self, expr: FilterExpr) -> str:
        """Compile a FilterExpr into a SQL WHERE clause fragment."""
        if isinstance(expr, Comparison):
            return self._compile_comparison(expr)
        if isinstance(expr, In):
            return self._compile_in(expr)
        if isinstance(expr, IsNull):
            return self._compile_is_null(expr)
        if isinstance(expr, And):
            left = self.compile(expr.left)
            right = self.compile(expr.right)
            return f"({left}) AND ({right})"
        if isinstance(expr, Or):
            left = self.compile(expr.left)
            right = self.compile(expr.right)
            return f"({left}) OR ({right})"
        if isinstance(expr, Not):
            inner = self.compile(expr.expr)
            return f"NOT ({inner})"
        msg = f"Unsupported filter expression type: {type(expr)}"
        raise TypeError(msg)

    def _compile_comparison(self, comp: Comparison) -> str:
        json_path = f"json_extract(r.properties, '$.{comp.field}')"
        pname = self._next_param()
        self.params[pname] = self._bind_value(comp.value)
        return f"{json_path} {comp.op} :{pname}"

    def _compile_in(self, expr: In) -> str:
        json_path = f"json_extract(r.properties, '$.{expr.field}')"
        placeholders = []
        for v in expr.values:
            pname = self._next_param()
            self.params[pname] = self._bind_value(v)
            placeholders.append(f":{pname}")
        return f"{json_path} IN ({', '.join(placeholders)})"

    def _compile_is_null(self, expr: IsNull) -> str:
        json_path = f"json_extract(r.properties, '$.{expr.field}')"
        return f"{json_path} IS NULL"

    @staticmethod
    def _bind_value(
        value: PropertyValue,
    ) -> str | int | float | None:
        """Convert a filter value to a SQLite-compatible bind value."""
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (int, float, str)) or value is None:
            return value
        msg = f"Unsupported filter value type: {type(value)}"
        raise TypeError(msg)


class SQLiteVectorStoreParams(BaseModel):
    """Parameters for SQLiteVectorStore.

    Attributes:
        engine: SQLAlchemy async engine (sqlite+aiosqlite).
        metrics_factory: Optional MetricsFactory for collecting usage metrics.

    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="SQLAlchemy async engine (sqlite+aiosqlite)",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class SQLiteCollection(Collection):
    """A collection backed by SQLite + sqlite-vec."""

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        name: str,
        dimensions: int,
        metric: SimilarityMetric,
        schema: dict[str, type[PropertyValue]] | None,
        vec_table: str,
        tracker: OperationTracker,
    ) -> None:
        self._engine = engine
        self._name = name
        self._dimensions = dimensions
        self._metric = metric
        self._schema = schema
        self._vec_table = vec_table
        self._tracker = tracker

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        """Upsert records in the collection."""
        async with self._tracker("upsert"):
            records_list = list(records)

            async with self._engine.begin() as conn:
                for record in records_list:
                    uuid_str = str(record.uuid)
                    props_json = _serialize_properties(record.properties)

                    result = await conn.execute(
                        text(
                            "SELECT rowid FROM _vs_records "
                            "WHERE collection_name = :coll AND uuid = :uuid"
                        ),
                        {"coll": self._name, "uuid": uuid_str},
                    )
                    existing = result.fetchone()

                    if existing is not None:
                        await self._upsert_existing(
                            conn,
                            existing[0],
                            props_json,
                            record.vector,
                        )
                    else:
                        await self._upsert_new(
                            conn,
                            uuid_str,
                            props_json,
                            record.vector,
                        )

    async def _upsert_existing(
        self,
        conn: object,
        row_id: int,
        props_json: str | None,
        vector: list[float] | None,
    ) -> None:
        """Update an existing record's properties and optionally its vector."""
        await conn.execute(  # type: ignore[union-attr]
            text("UPDATE _vs_records SET properties = :props WHERE rowid = :rowid"),
            {"props": props_json, "rowid": row_id},
        )
        if vector is not None:
            vec_blob = _serialize_vector(vector)
            await conn.execute(  # type: ignore[union-attr]
                text(f"UPDATE {self._vec_table} SET embedding = :emb WHERE rowid = :rowid"),
                {"emb": vec_blob, "rowid": row_id},
            )

    async def _upsert_new(
        self,
        conn: object,
        uuid_str: str,
        props_json: str | None,
        vector: list[float] | None,
    ) -> None:
        """Insert a new record and optionally its vector."""
        result = await conn.execute(  # type: ignore[union-attr]
            text(
                "INSERT INTO _vs_records (collection_name, uuid, properties) "
                "VALUES (:coll, :uuid, :props)"
            ),
            {"coll": self._name, "uuid": uuid_str, "props": props_json},
        )
        row_id = result.lastrowid

        if vector is not None:
            vec_blob = _serialize_vector(vector)
            await conn.execute(  # type: ignore[union-attr]
                text(
                    f"INSERT INTO {self._vec_table} (rowid, embedding, collection_name) "
                    f"VALUES (:rowid, :emb, :coll)"
                ),
                {"rowid": row_id, "emb": vec_blob, "coll": self._name},
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
        """Query for records matching the criteria by query vectors."""
        async with self._tracker("query"):
            if property_filter is not None:
                if not validate_filter(property_filter):
                    raise ValueError("Filter contains an invalid property key")

            results: list[QueryResult] = []
            for query_vector in query_vectors:
                query_blob = _serialize_vector(list(query_vector))

                if property_filter is None:
                    matches = await self._query_knn(
                        query_blob,
                        score_threshold,
                        limit,
                        return_vector,
                        return_properties,
                    )
                else:
                    matches = await self._query_filtered(
                        query_blob,
                        score_threshold,
                        limit,
                        property_filter,
                        return_vector,
                        return_properties,
                    )
                results.append(QueryResult(matches=matches))

            return results

    async def _query_knn(
        self,
        query_blob: bytes,
        score_threshold: float | None,
        limit: int | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Execute a pure KNN query via vec0 MATCH."""
        knn_limit = limit if limit is not None else 100

        async with self._engine.connect() as conn:
            knn_rows = await conn.execute(
                text(
                    f"SELECT knn.rowid, knn.distance "
                    f"FROM {self._vec_table} knn "
                    f"WHERE knn.embedding MATCH :query "
                    f"  AND knn.k = :k "
                    f"  AND knn.collection_name = :coll "
                    f"ORDER BY knn.distance"
                ),
                {"query": query_blob, "k": knn_limit, "coll": self._name},
            )
            knn_results = knn_rows.fetchall()

            if not knn_results:
                return []

            rowid_to_distance: dict[int, float] = {
                row[0]: row[1] for row in knn_results
            }
            rowid_list = list(rowid_to_distance.keys())
            placeholders = ", ".join(f":rid{i}" for i in range(len(rowid_list)))
            bind_params: _BindParams = {
                f"rid{i}": rid for i, rid in enumerate(rowid_list)
            }

            select_parts = ["r.rowid", "r.uuid"]
            if return_properties:
                select_parts.append("r.properties")

            rec_rows = await conn.execute(
                text(
                    f"SELECT {', '.join(select_parts)} "
                    f"FROM _vs_records r "
                    f"WHERE r.rowid IN ({placeholders})"
                ),
                bind_params,
            )
            rec_results = rec_rows.fetchall()

            rowid_to_rec: dict[int, tuple[str, str | None]] = {}
            for row in rec_results:
                rowid_to_rec[row[0]] = (
                    row[1],
                    row[2] if return_properties else None,
                )

            rowid_to_vec: dict[int, list[float]] = {}
            if return_vector:
                vec_rows = await conn.execute(
                    text(
                        f"SELECT rowid, embedding FROM {self._vec_table} "
                        f"WHERE rowid IN ({placeholders})"
                    ),
                    bind_params,
                )
                for vrow in vec_rows.fetchall():
                    rowid_to_vec[vrow[0]] = _deserialize_vector(vrow[1])

        matches: list[QueryMatch] = []
        for rid in rowid_list:
            if rid not in rowid_to_rec:
                continue
            uuid_str, props_raw = rowid_to_rec[rid]
            distance = rowid_to_distance[rid]
            similarity = _distance_to_similarity(distance, self._metric)

            if score_threshold is not None and similarity < score_threshold:
                continue

            record = Record(
                uuid=UUID(uuid_str),
                vector=rowid_to_vec.get(rid) if return_vector else None,
                properties=_deserialize_properties(props_raw, self._schema)
                if return_properties
                else None,
            )
            matches.append(QueryMatch(score=similarity, record=record))
        return matches

    async def _query_filtered(
        self,
        query_blob: bytes,
        score_threshold: float | None,
        limit: int | None,
        property_filter: FilterExpr,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Execute a filtered query using brute-force distance calculation."""
        distance_fn = _METRIC_TO_DISTANCE_FN[self._metric]
        compiler = _FilterCompiler()
        filter_clause = compiler.compile(property_filter)

        select_parts = [
            "r.uuid",
            "r.properties",
            f"{distance_fn}(v.embedding, :query) AS distance",
        ]
        if return_vector:
            select_parts.append("v.embedding")

        where_parts = [
            "r.collection_name = :coll",
            filter_clause,
        ]
        bind_params: _BindParams = {"coll": self._name, "query": query_blob}
        bind_params.update(compiler.params)

        if score_threshold is not None:
            max_dist = _similarity_to_max_distance(score_threshold, self._metric)
            where_parts.append(f"{distance_fn}(v.embedding, :query) <= :max_dist")
            bind_params["max_dist"] = max_dist

        limit_clause = "LIMIT :lim" if limit is not None else ""
        if limit is not None:
            bind_params["lim"] = limit

        sql = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM _vs_records r "
            f"JOIN {self._vec_table} v ON v.rowid = r.rowid "
            f"WHERE {' AND '.join(where_parts)} "
            f"ORDER BY distance ASC "
            f"{limit_clause}"
        )

        matches: list[QueryMatch] = []
        async with self._engine.connect() as conn:
            rows = await conn.execute(text(sql), bind_params)
            for row in rows.fetchall():
                uuid_val = UUID(row[0])
                props_raw = row[1]
                distance = row[2]
                similarity = _distance_to_similarity(distance, self._metric)

                vector = None
                if return_vector:
                    vector = _deserialize_vector(row[3])

                record = Record(
                    uuid=uuid_val,
                    vector=vector,
                    properties=_deserialize_properties(props_raw, self._schema)
                    if return_properties
                    else None,
                )
                matches.append(QueryMatch(score=similarity, record=record))
        return matches

    @override
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
        """Get records from the collection by their UUIDs."""
        async with self._tracker("get"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return []

            uuid_strs = [str(u) for u in uuid_list]

            placeholders = ", ".join(f":u{i}" for i in range(len(uuid_strs)))
            bind_params: _BindParams = {f"u{i}": s for i, s in enumerate(uuid_strs)}
            bind_params["coll"] = self._name

            select_parts = ["r.uuid"]
            if return_properties:
                select_parts.append("r.properties")
            if return_vector:
                select_parts.append("v.embedding")

            select_clause = ", ".join(select_parts)

            if return_vector:
                sql = (
                    f"SELECT {select_clause} "
                    f"FROM _vs_records r "
                    f"LEFT JOIN {self._vec_table} v ON v.rowid = r.rowid "
                    f"WHERE r.collection_name = :coll AND r.uuid IN ({placeholders})"
                )
            else:
                sql = (
                    f"SELECT {select_clause} "
                    f"FROM _vs_records r "
                    f"WHERE r.collection_name = :coll AND r.uuid IN ({placeholders})"
                )

            async with self._engine.connect() as conn:
                rows = await conn.execute(text(sql), bind_params)
                rows_list = rows.fetchall()

            uuid_to_record: dict[str, Record] = {}
            for row in rows_list:
                idx = 0
                uuid_val = UUID(row[idx])
                idx += 1

                props = None
                if return_properties:
                    props = _deserialize_properties(row[idx], self._schema)
                    idx += 1

                vector = None
                if return_vector:
                    raw_vec = row[idx]
                    if raw_vec is not None:
                        vector = _deserialize_vector(raw_vec)

                uuid_to_record[str(uuid_val)] = Record(
                    uuid=uuid_val,
                    vector=vector,
                    properties=props,
                )

            return [uuid_to_record[str(u)] for u in uuid_list if str(u) in uuid_to_record]

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        """Delete records from the collection by their UUIDs."""
        async with self._tracker("delete"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return

            uuid_strs = [str(u) for u in uuid_list]

            placeholders = ", ".join(f":u{i}" for i in range(len(uuid_strs)))
            bind_params: _BindParams = {f"u{i}": s for i, s in enumerate(uuid_strs)}
            bind_params["coll"] = self._name

            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        f"DELETE FROM {self._vec_table} "
                        f"WHERE rowid IN ("
                        f"  SELECT rowid FROM _vs_records "
                        f"  WHERE collection_name = :coll AND uuid IN ({placeholders})"
                        f")"
                    ),
                    bind_params,
                )

                await conn.execute(
                    text(
                        f"DELETE FROM _vs_records "
                        f"WHERE collection_name = :coll AND uuid IN ({placeholders})"
                    ),
                    bind_params,
                )


class SQLiteVectorStore(VectorStore):
    """SQLite-backed vector store using sqlite-vec for native vector search."""

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        super().__init__()
        self._engine = params.engine
        self._tracker = OperationTracker(params.metrics_factory, prefix="vector_store_sqlite")
        self._name_locks: defaultdict[tuple[str, str], asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    @override
    async def startup(self) -> None:
        """Initialize the store: load sqlite-vec extension and create metadata tables."""

        @event.listens_for(self._engine.sync_engine, "connect")
        def _load_vec_extension(
            dbapi_conn: sqlite3.Connection, _connection_record: object
        ) -> None:
            dbapi_conn.enable_load_extension(True)
            sqlite_vec.load(dbapi_conn)
            dbapi_conn.enable_load_extension(False)

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS _vs_collections ("
                    "  namespace TEXT NOT NULL,"
                    "  name TEXT NOT NULL,"
                    "  vector_dimensions INTEGER NOT NULL,"
                    "  similarity_metric TEXT NOT NULL,"
                    "  properties_schema TEXT,"
                    "  PRIMARY KEY (namespace, name)"
                    ")"
                )
            )
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS _vs_records ("
                    "  rowid INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  collection_name TEXT NOT NULL,"
                    "  uuid TEXT NOT NULL,"
                    "  properties TEXT,"
                    "  UNIQUE(collection_name, uuid)"
                    ")"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_vs_records_collection "
                    "ON _vs_records(collection_name)"
                )
            )

    @override
    async def shutdown(self) -> None:
        """Dispose of the engine."""
        await self._engine.dispose()

    def _build_collection_handle(
        self,
        name: str,
        config: CollectionConfig,
        vec_table: str,
    ) -> SQLiteCollection:
        return SQLiteCollection(
            engine=self._engine,
            name=name,
            dimensions=config.vector_dimensions,
            metric=config.similarity_metric,
            schema=dict(config.properties_schema) if config.properties_schema else None,
            vec_table=vec_table,
            tracker=self._tracker,
        )

    async def _create_native_collection(
        self,
        conn: object,
        namespace: str,
        name: str,
        config: CollectionConfig,
    ) -> str:
        """Create the vec0 table and property indexes. Returns the vec_table name."""
        _validate_metric(config.similarity_metric)
        vec_table = _vec_table_name(namespace, config.vector_dimensions, config.similarity_metric)
        vec0_metric = _METRIC_TO_VEC0[config.similarity_metric]

        await conn.execute(  # type: ignore[union-attr]
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {vec_table} USING vec0("
                f"  embedding float[{config.vector_dimensions}],"
                f"  collection_name text partition key,"
                f'  distance_metric="{vec0_metric}"'
                f")"
            )
        )

        if config.properties_schema:
            safe_name = name.replace("'", "''")
            for field_name in config.properties_schema:
                safe_field = field_name.replace("'", "''")
                idx_name = f"idx_vs_prop_{name}_{field_name}"
                await conn.execute(  # type: ignore[union-attr]
                    text(
                        f"CREATE INDEX IF NOT EXISTS [{idx_name}] "
                        f"ON _vs_records(json_extract(properties, '$.{safe_field}')) "
                        f"WHERE collection_name = '{safe_name}'"
                    )
                )

        return vec_table

    async def _register_collection(
        self,
        conn: object,
        namespace: str,
        name: str,
        config: CollectionConfig,
    ) -> None:
        """Write the logical collection entry to the registry."""
        await conn.execute(  # type: ignore[union-attr]
            text(
                "INSERT INTO _vs_collections (namespace, name, vector_dimensions, similarity_metric, properties_schema) "
                "VALUES (:namespace, :name, :dims, :metric, :schema)"
            ),
            {
                "namespace": namespace,
                "name": name,
                "dims": config.vector_dimensions,
                "metric": config.similarity_metric.value,
                "schema": _serialize_schema(config.properties_schema),
            },
        )

    async def _get_registry_entry(
        self,
        namespace: str,
        name: str,
    ) -> CollectionConfig | None:
        """Retrieve the config for a logical collection, or None if not found."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT vector_dimensions, similarity_metric, properties_schema "
                    "FROM _vs_collections WHERE namespace = :namespace AND name = :name"
                ),
                {"namespace": namespace, "name": name},
            )
            row = result.fetchone()
            if row is None:
                return None
            return CollectionConfig(
                vector_dimensions=row[0],
                similarity_metric=row[1],
                properties_schema=_deserialize_schema(row[2]),
            )

    @staticmethod
    def _validate_namespace_and_name(namespace: str, name: str) -> None:
        if not validate_identifier(namespace):
            raise ValueError(
                f"Namespace {namespace!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        if not validate_identifier(name):
            raise ValueError(
                f"Name {name!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: CollectionConfig,
    ) -> None:
        """Create a logical collection in the vector store."""
        SQLiteVectorStore._validate_namespace_and_name(namespace, name)
        lock = self._name_locks[(namespace, name)]
        async with lock, self._tracker("create_collection"):
            if await self._get_registry_entry(namespace, name) is not None:
                raise CollectionAlreadyExistsError(namespace, name)

            async with self._engine.begin() as conn:
                await self._create_native_collection(conn, namespace, name, config)
                await self._register_collection(conn, namespace, name, config)

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: CollectionConfig,
    ) -> SQLiteCollection:
        """Open the collection if it exists, or create and return it."""
        SQLiteVectorStore._validate_namespace_and_name(namespace, name)
        lock = self._name_locks[(namespace, name)]
        async with lock, self._tracker("open_or_create_collection"):
            existing_config = await self._get_registry_entry(namespace, name)
            if existing_config is not None:
                if existing_config != config:
                    raise CollectionConfigMismatchError(
                        namespace, name, existing_config, config
                    )
                vec_table = _vec_table_name(
                    namespace,
                    existing_config.vector_dimensions,
                    existing_config.similarity_metric,
                )
                return self._build_collection_handle(name, existing_config, vec_table)

            async with self._engine.begin() as conn:
                vec_table = await self._create_native_collection(
                    conn, namespace, name, config
                )
                await self._register_collection(conn, namespace, name, config)
            return self._build_collection_handle(name, config, vec_table)

    @override
    async def open_collection(
        self, *, namespace: str, name: str
    ) -> SQLiteCollection | None:
        """Get a handle to a logical collection in the vector store."""
        SQLiteVectorStore._validate_namespace_and_name(namespace, name)
        existing_config = await self._get_registry_entry(namespace, name)
        if existing_config is None:
            return None
        vec_table = _vec_table_name(
            namespace,
            existing_config.vector_dimensions,
            existing_config.similarity_metric,
        )
        return self._build_collection_handle(name, existing_config, vec_table)

    @override
    async def close_collection(self, *, collection: Collection) -> None:
        """No-op; SQLite collection handles require no explicit close."""

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        """Delete a logical collection from the vector store."""
        SQLiteVectorStore._validate_namespace_and_name(namespace, name)
        lock = self._name_locks[(namespace, name)]
        async with lock, self._tracker("delete_collection"):
            existing_config = await self._get_registry_entry(namespace, name)
            if existing_config is None:
                return

            vec_table = _vec_table_name(
                namespace,
                existing_config.vector_dimensions,
                existing_config.similarity_metric,
            )

            async with self._engine.begin() as conn:
                # Drop property indexes
                if existing_config.properties_schema:
                    for field_name in existing_config.properties_schema:
                        idx_name = f"idx_vs_prop_{name}_{field_name}"
                        await conn.execute(text(f"DROP INDEX IF EXISTS [{idx_name}]"))

                # Delete vectors from the shared vec0 table
                await conn.execute(
                    text(f"DELETE FROM {vec_table} WHERE collection_name = :coll"),
                    {"coll": name},
                )

                # Delete records
                await conn.execute(
                    text("DELETE FROM _vs_records WHERE collection_name = :coll"),
                    {"coll": name},
                )

                # Delete metadata
                await conn.execute(
                    text(
                        "DELETE FROM _vs_collections "
                        "WHERE namespace = :namespace AND name = :name"
                    ),
                    {"namespace": namespace, "name": name},
                )
