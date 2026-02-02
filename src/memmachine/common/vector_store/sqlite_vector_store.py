"""SQLite-backed vector store using sqlite-vec for vector operations."""

import json
import re
import sqlite3
import struct
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from uuid import UUID

import aiosqlite
import sqlite_vec
from pydantic import BaseModel, Field

from memmachine.common.data_types import FilterablePropertyValue, SimilarityMetric
from memmachine.common.errors import ResourceNotFoundError
from memmachine.common.filter.filter_parser import And, Comparison, FilterExpr, Or

from .data_types import PropertyValue, QueryResult, Record

_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")

_PYTHON_TYPE_TO_SQLITE: dict[type[PropertyValue], str] = {
    bool: "INTEGER",
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    datetime: "TEXT",
}

_TYPE_NAME_TO_PYTHON: dict[str, type[PropertyValue]] = {
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "datetime": datetime,
}

_OVERSAMPLING_FACTOR = 4

_METRIC_TO_VEC0: dict[SimilarityMetric, str] = {
    SimilarityMetric.COSINE: "cosine",
    SimilarityMetric.EUCLIDEAN: "L2",
    SimilarityMetric.MANHATTAN: "L1",
}


def _distance_to_similarity(distance: float, metric: SimilarityMetric) -> float:
    """Convert a sqlite-vec distance value to a similarity score."""
    if metric == SimilarityMetric.COSINE:
        return 1.0 - distance
    # L2 / L1: unbounded distances → bounded similarity via 1/(1+d)
    return 1.0 / (1.0 + distance)


def _validate_collection_name(name: str) -> None:
    """Raise ``ValueError`` if *name* contains unsafe characters."""
    if not _NAME_RE.match(name):
        msg = (
            f"Invalid collection name '{name}': "
            "only alphanumeric characters and underscores are allowed."
        )
        raise ValueError(msg)


def _records_table(name: str) -> str:
    return f"_coll_{name}_records"


def _vec_table(name: str) -> str:
    return f"_coll_{name}_vec"


def _serialize_vector(vec: Sequence[float]) -> bytes:
    """Pack a float list into the little-endian binary format sqlite-vec expects."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _deserialize_vector(blob: bytes) -> list[float]:
    """Unpack a sqlite-vec binary blob back to a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _python_type_name(t: type[PropertyValue]) -> str:
    """Return the canonical short name used in the stored schema JSON."""
    for name, cls in _TYPE_NAME_TO_PYTHON.items():
        if cls is t:
            return name
    msg = f"Unsupported property type: {t}"
    raise ValueError(msg)


def _render_sqlite_filter_expr(
    expr: FilterExpr,
    table_alias: str,
) -> tuple[str, list[FilterablePropertyValue]]:
    """Convert a ``FilterExpr`` tree to a SQL fragment with ``?`` params."""
    if isinstance(expr, Comparison):
        return _render_comparison(expr, table_alias)
    if isinstance(expr, And):
        left_sql, left_params = _render_sqlite_filter_expr(expr.left, table_alias)
        right_sql, right_params = _render_sqlite_filter_expr(expr.right, table_alias)
        return f"({left_sql} AND {right_sql})", [*left_params, *right_params]
    if isinstance(expr, Or):
        left_sql, left_params = _render_sqlite_filter_expr(expr.left, table_alias)
        right_sql, right_params = _render_sqlite_filter_expr(expr.right, table_alias)
        return f"({left_sql} OR {right_sql})", [*left_params, *right_params]
    msg = f"Unsupported filter expression type: {type(expr)}"
    raise TypeError(msg)


def _render_comparison(
    comp: Comparison,
    table_alias: str,
) -> tuple[str, list[FilterablePropertyValue]]:
    col = f"{table_alias}.{comp.field}"
    match comp.op:
        case "is_null":
            return f"{col} IS NULL", []
        case "is_not_null":
            return f"{col} IS NOT NULL", []
        case "in":
            values = comp.value if isinstance(comp.value, list) else [comp.value]
            placeholders = ", ".join("?" for _ in values)
            return f"{col} IN ({placeholders})", [_coerce_param(v) for v in values]
        case "=" | ">" | "<" | ">=" | "<=":
            return f"{col} {comp.op} ?", [_coerce_param(comp.value)]
        case _:
            msg = f"Unsupported operator: {comp.op}"
            raise ValueError(msg)


def _coerce_param(
    value: FilterablePropertyValue | list[FilterablePropertyValue],
) -> FilterablePropertyValue:
    """Coerce a parameter value for SQLite binding."""
    if isinstance(value, list):
        msg = "Cannot bind a list as a single parameter."
        raise TypeError(msg)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return int(value)
    return value


def _restore_property_value(
    raw: float | str,
    target_type: type[PropertyValue],
) -> PropertyValue:
    """Convert a raw SQLite value back to the expected Python type."""
    if target_type is bool:
        return bool(raw)
    if target_type is datetime:
        return datetime.fromisoformat(str(raw))
    return raw


def _build_properties(
    row: sqlite3.Row,
    offset: int,
    prop_cols: list[str],
    schema: dict[str, type[PropertyValue]],
) -> dict[str, PropertyValue]:
    """Extract property columns from a row tuple starting at *offset*."""
    properties: dict[str, PropertyValue] = {}
    for i, col in enumerate(prop_cols):
        val = row[offset + i]
        if val is not None:
            properties[col] = _restore_property_value(val, schema[col])
    return properties


async def _fetch_vector(
    db: aiosqlite.Connection,
    vec_table: str,
    rowid: int,
) -> list[float] | None:
    """Load a single vector by rowid, returning ``None`` if absent."""
    vec_row = await db.execute_fetchall(
        f"SELECT embedding FROM {vec_table} WHERE rowid = ?",
        (rowid,),
    )
    if vec_row:
        return _deserialize_vector(vec_row[0][0])
    return None


class SQLiteCollection:
    """A collection backed by SQLite + sqlite-vec."""

    def __init__(self, store: "SQLiteVectorStore", name: str) -> None:
        """Initialize with a reference to the parent store and collection name."""
        self._store = store
        self._name = name

    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records into the collection."""
        db = self._store.ensure_db()
        meta = await self._store.get_collection_meta(self._name)
        schema: dict[str, type[PropertyValue]] = meta["schema"]
        prop_cols = list(schema.keys())

        rec_table = _records_table(self._name)
        vec_table = _vec_table(self._name)

        for record in records:
            uuid_str = str(record.uuid)

            # Check if UUID already exists — delete old rows first
            row = await db.execute_fetchall(
                f"SELECT rowid FROM {rec_table} WHERE uuid = ?",
                (uuid_str,),
            )
            if row:
                old_rowid = row[0][0]
                await db.execute(
                    f"DELETE FROM {vec_table} WHERE rowid = ?",
                    (old_rowid,),
                )
                await db.execute(
                    f"DELETE FROM {rec_table} WHERE rowid = ?",
                    (old_rowid,),
                )

            # Build column list and values for the records table
            columns = ["uuid"]
            values: list[FilterablePropertyValue | str] = [uuid_str]
            if record.properties:
                for col in prop_cols:
                    if col in record.properties:
                        values.append(_coerce_param(record.properties[col]))
                        columns.append(col)

            placeholders = ", ".join("?" for _ in columns)
            col_list = ", ".join(columns)
            cursor = await db.execute(
                f"INSERT INTO {rec_table} ({col_list}) VALUES ({placeholders})",
                tuple(values),
            )
            new_rowid = cursor.lastrowid

            # Insert vector
            if record.vector is not None:
                vec_bytes = _serialize_vector(record.vector)
                await db.execute(
                    f"INSERT INTO {vec_table} (rowid, embedding) VALUES (?, ?)",
                    (new_rowid, vec_bytes),
                )

        await db.commit()

    async def query(
        self,
        *,
        query_vector: Sequence[float],
        similarity_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[QueryResult]:
        """Query records by vector similarity."""
        db = self._store.ensure_db()
        meta = await self._store.get_collection_meta(self._name)
        schema: dict[str, type[PropertyValue]] = meta["schema"]
        prop_cols = list(schema.keys())
        metric: SimilarityMetric = meta["metric"]

        vec_tbl = _vec_table(self._name)
        rec_table = _records_table(self._name)

        effective_limit = limit if limit is not None else 100
        k = (
            effective_limit * _OVERSAMPLING_FACTOR
            if property_filter
            else effective_limit
        )

        # Step 1: KNN search on vec0
        vec_bytes = _serialize_vector(query_vector)
        knn_rows = await db.execute_fetchall(
            f"SELECT rowid, distance FROM {vec_tbl} WHERE embedding MATCH ? AND k = ?",
            (vec_bytes, k),
        )

        if not knn_rows:
            return []

        rowids = [r[0] for r in knn_rows]
        distances = {r[0]: r[1] for r in knn_rows}

        # Step 2: Fetch matching records with optional filter
        rows, params = self._build_query_sql(
            rec_table, rowids, prop_cols, return_properties, property_filter
        )
        fetched = await db.execute_fetchall(rows, tuple(params))

        # Step 3: Build results
        results: list[QueryResult] = []
        for row in fetched:
            result = await self._row_to_query_result(
                row,
                distances,
                metric,
                prop_cols,
                schema,
                return_properties,
                return_vector,
                vec_tbl,
                db,
            )
            if similarity_threshold is not None and result.score < similarity_threshold:
                continue
            results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:effective_limit]

    @staticmethod
    def _build_query_sql(
        rec_table: str,
        rowids: list[int],
        prop_cols: list[str],
        return_properties: bool,
        property_filter: FilterExpr | None,
    ) -> tuple[str, list[FilterablePropertyValue | int | str | bytes]]:
        """Build the SQL and params for the records lookup."""
        placeholders = ", ".join("?" for _ in rowids)
        select_cols = ["r.rowid", "r.uuid"]
        if return_properties:
            select_cols.extend(f"r.{col}" for col in prop_cols)

        sql = (
            f"SELECT {', '.join(select_cols)} "
            f"FROM {rec_table} r "
            f"WHERE r.rowid IN ({placeholders})"
        )
        params: list[FilterablePropertyValue | int | str | bytes] = list(rowids)

        if property_filter:
            filter_sql, filter_params = _render_sqlite_filter_expr(property_filter, "r")
            sql += f" AND {filter_sql}"
            params.extend(filter_params)

        return sql, params

    @staticmethod
    async def _row_to_query_result(
        row: sqlite3.Row,
        distances: dict[int, float],
        metric: SimilarityMetric,
        prop_cols: list[str],
        schema: dict[str, type[PropertyValue]],
        return_properties: bool,
        return_vector: bool,
        vec_tbl: str,
        db: aiosqlite.Connection,
    ) -> QueryResult:
        """Convert a fetched row into a ``QueryResult``."""
        rowid = row[0]
        uuid_str = row[1]
        similarity = _distance_to_similarity(distances[rowid], metric)

        properties = (
            _build_properties(row, 2, prop_cols, schema)
            if return_properties and prop_cols
            else None
        )
        vector = await _fetch_vector(db, vec_tbl, rowid) if return_vector else None

        return QueryResult(
            score=similarity,
            record=Record(uuid=UUID(uuid_str), vector=vector, properties=properties),
        )

    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        """Get records by UUID."""
        db = self._store.ensure_db()
        meta = await self._store.get_collection_meta(self._name)
        schema: dict[str, type[PropertyValue]] = meta["schema"]
        prop_cols = list(schema.keys())

        rec_table = _records_table(self._name)
        vec_tbl = _vec_table(self._name)

        uuid_strs = [str(u) for u in record_uuids]
        if not uuid_strs:
            return []

        placeholders = ", ".join("?" for _ in uuid_strs)
        select_cols = ["r.rowid", "r.uuid"]
        if return_properties:
            select_cols.extend(f"r.{col}" for col in prop_cols)

        sql = (
            f"SELECT {', '.join(select_cols)} "
            f"FROM {rec_table} r "
            f"WHERE r.uuid IN ({placeholders})"
        )
        rows = await db.execute_fetchall(sql, tuple(uuid_strs))

        row_map: dict[str, sqlite3.Row] = {row[1]: row for row in rows}

        results: list[Record] = []
        for uuid_str in uuid_strs:
            if uuid_str not in row_map:
                continue
            row = row_map[uuid_str]
            results.append(
                await self._row_to_record(
                    row,
                    prop_cols,
                    schema,
                    return_properties,
                    return_vector,
                    vec_tbl,
                    db,
                )
            )
        return results

    @staticmethod
    async def _row_to_record(
        row: sqlite3.Row,
        prop_cols: list[str],
        schema: dict[str, type[PropertyValue]],
        return_properties: bool,
        return_vector: bool,
        vec_tbl: str,
        db: aiosqlite.Connection,
    ) -> Record:
        """Convert a fetched row into a ``Record``."""
        rowid = row[0]
        uuid_str = row[1]

        properties = (
            _build_properties(row, 2, prop_cols, schema)
            if return_properties and prop_cols
            else None
        )
        vector = await _fetch_vector(db, vec_tbl, rowid) if return_vector else None

        return Record(uuid=UUID(uuid_str), vector=vector, properties=properties)

    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """Delete records by UUID."""
        db = self._store.ensure_db()
        rec_table = _records_table(self._name)
        vec_table = _vec_table(self._name)

        # Verify collection exists (raises if not)
        await self._store.get_collection_meta(self._name)

        uuid_list = list(record_uuids)
        if not uuid_list:
            return

        uuid_strs = [str(u) for u in uuid_list]
        placeholders = ", ".join("?" for _ in uuid_strs)

        # Look up rowids
        rows = await db.execute_fetchall(
            f"SELECT rowid FROM {rec_table} WHERE uuid IN ({placeholders})",
            tuple(uuid_strs),
        )
        rowids = [r[0] for r in rows]

        if rowids:
            rid_placeholders = ", ".join("?" for _ in rowids)
            await db.execute(
                f"DELETE FROM {vec_table} WHERE rowid IN ({rid_placeholders})",
                tuple(rowids),
            )
            await db.execute(
                f"DELETE FROM {rec_table} WHERE rowid IN ({rid_placeholders})",
                tuple(rowids),
            )
            await db.commit()


class SQLiteVectorStoreParams(BaseModel):
    """Parameters for SQLiteVectorStore."""

    db_path: str = Field(
        default=":memory:",
        description="Path to the SQLite database file, or ':memory:' for an in-memory database.",
    )


class SQLiteVectorStore:
    """Vector store backed by SQLite with sqlite-vec."""

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        """Initialize the SQLite vector store with configuration parameters."""
        self._db_path = params.db_path
        self._db: aiosqlite.Connection | None = None

    async def startup(self) -> None:
        """Open the database connection and initialise extensions/tables."""
        self._db = await aiosqlite.connect(self._db_path)
        # Enable extension loading and load sqlite-vec
        await self._db.execute("select 1")  # ensure connection is open
        self._db._conn.enable_load_extension(True)  # noqa: SLF001
        sqlite_vec.load(self._db._conn)  # noqa: SLF001
        self._db._conn.enable_load_extension(False)  # noqa: SLF001

        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS _collections (
                name                TEXT PRIMARY KEY,
                vector_dimensions   INTEGER NOT NULL,
                similarity_metric   TEXT NOT NULL,
                properties_schema   TEXT
            )
            """
        )
        await self._db.commit()

    async def shutdown(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        """Create a new collection."""
        _validate_collection_name(collection_name)
        if similarity_metric not in _METRIC_TO_VEC0:
            supported = ", ".join(m.value for m in _METRIC_TO_VEC0)
            msg = f"Unsupported similarity metric '{similarity_metric.value}'. Supported: {supported}."
            raise ValueError(msg)

        db = self.ensure_db()

        schema_json: str | None = None
        if properties_schema:
            schema_json = json.dumps(
                {k: _python_type_name(v) for k, v in properties_schema.items()}
            )

        await db.execute(
            "INSERT INTO _collections (name, vector_dimensions, similarity_metric, properties_schema) "
            "VALUES (?, ?, ?, ?)",
            (collection_name, vector_dimensions, similarity_metric.value, schema_json),
        )

        # Create records table
        prop_columns = ""
        if properties_schema:
            parts = []
            for prop_name, prop_type in properties_schema.items():
                _validate_collection_name(prop_name)  # reuse name validation
                sqlite_type = _PYTHON_TYPE_TO_SQLITE[prop_type]
                parts.append(f"{prop_name} {sqlite_type}")
            prop_columns = ", " + ", ".join(parts)

        rec_table = _records_table(collection_name)
        await db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {rec_table} (
                rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid    TEXT NOT NULL UNIQUE
                {prop_columns}
            )
            """
        )

        # Create indexes on property columns
        if properties_schema:
            for prop_name in properties_schema:
                await db.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{rec_table}_{prop_name} "
                    f"ON {rec_table} ({prop_name})"
                )

        # Create vec0 virtual table
        vec_tbl = _vec_table(collection_name)
        vec0_metric = _METRIC_TO_VEC0[similarity_metric]
        await db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {vec_tbl} "
            f"USING vec0(embedding float[{vector_dimensions}] distance_metric={vec0_metric})"
        )

        await db.commit()

    async def get_collection(self, collection_name: str) -> SQLiteCollection:
        """Get a collection handle (verifies existence)."""
        self.ensure_db()
        await self.get_collection_meta(collection_name)
        return SQLiteCollection(self, collection_name)

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection and all its data."""
        _validate_collection_name(collection_name)
        db = self.ensure_db()

        # Verify existence
        await self.get_collection_meta(collection_name)

        await db.execute(f"DROP TABLE IF EXISTS {_vec_table(collection_name)}")
        await db.execute(f"DROP TABLE IF EXISTS {_records_table(collection_name)}")
        await db.execute("DELETE FROM _collections WHERE name = ?", (collection_name,))
        await db.commit()

    async def get_collection_meta(self, name: str) -> dict:
        """Read collection metadata; raise ``ResourceNotFoundError`` if absent."""
        _validate_collection_name(name)
        db = self.ensure_db()
        rows = await db.execute_fetchall(
            "SELECT vector_dimensions, similarity_metric, properties_schema "
            "FROM _collections WHERE name = ?",
            (name,),
        )
        if not rows:
            msg = f"Collection '{name}' not found."
            raise ResourceNotFoundError(msg)
        dims, metric, schema_json = rows[0]
        schema: dict[str, type[PropertyValue]] = {}
        if schema_json:
            raw = json.loads(schema_json)
            schema = {k: _TYPE_NAME_TO_PYTHON[v] for k, v in raw.items()}
        return {
            "dimensions": dims,
            "metric": SimilarityMetric(metric),
            "schema": schema,
        }

    def ensure_db(self) -> aiosqlite.Connection:
        """Return the active connection or raise if not started."""
        if self._db is None:
            msg = "Vector store is not started. Call startup() first."
            raise RuntimeError(msg)
        return self._db
