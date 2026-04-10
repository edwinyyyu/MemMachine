"""Milvus-based vector store implementation."""

import asyncio
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, ClassVar, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf
from pymilvus import (
    AsyncMilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from memmachine_server.common.data_types import (
    PropertyValue,
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine_server.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.filter.filter_parser import (
    In as FilterIn,
)
from memmachine_server.common.filter.filter_parser import (
    IsNull as FilterIsNull,
)
from memmachine_server.common.filter.filter_parser import (
    Not as FilterNot,
)
from memmachine_server.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker
from memmachine_server.common.utils import ensure_tz_aware, utc_offset_seconds

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

# System field names use MixedCase so they cannot collide with user property
# names, which are strictly [a-z0-9_]+.
_FIELD_UUID = "Uuid"
_FIELD_VECTOR = "Vector"

# Companion field suffix for datetime timezone offsets.
# MixedCase so it cannot collide with user property names ([a-z0-9_]+).
_TZ_SUFFIX = "Tz"

_PK_MAX_LENGTH = 36
_VARCHAR_MAX_LENGTH = 65535
_DEFAULT_SEARCH_LIMIT = 16384
_UPSERT_BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def _to_milvus_collection_name(namespace: str, name: str) -> str:
    """Build a deterministic, injective Milvus collection name.

    Format: ``mm_{len(namespace)}_{namespace}_{name}``

    The length prefix makes the mapping provably injective — the boundary
    between namespace and name is always unambiguous.
    """
    return f"mm_{len(namespace)}_{namespace}_{name}"


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _datetime_to_micros(dt: datetime) -> int:
    """Convert a timezone-aware datetime to UTC microseconds since epoch."""
    return int(ensure_tz_aware(dt).timestamp() * 1_000_000)


def _micros_to_datetime(micros: int, offset_seconds: int) -> datetime:
    """Reconstruct a datetime from UTC microseconds and a UTC offset."""
    utc_dt = datetime.fromtimestamp(micros / 1_000_000, tz=UTC)
    tz = timezone(timedelta(seconds=offset_seconds))
    return utc_dt.astimezone(tz)


# ---------------------------------------------------------------------------
# Filter expression → Milvus boolean expression string
# ---------------------------------------------------------------------------


def _format_filter_value(value: PropertyValue) -> str:
    """Format a single property value for a Milvus filter expression."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, datetime):
        return str(_datetime_to_micros(value))
    msg = f"Unsupported filter value type: {type(value)}"
    raise TypeError(msg)


_COMPARISON_OP_MAP: dict[str, str] = {
    "=": "==",
    "!=": "!=",
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
}


def _build_milvus_filter(expr: FilterExpr, datetime_fields: set[str]) -> str:
    """Convert a FilterExpr tree into a Milvus boolean expression string."""
    if isinstance(expr, FilterComparison):
        op = _COMPARISON_OP_MAP[expr.op]
        if expr.field in datetime_fields and isinstance(expr.value, datetime):
            formatted = str(_datetime_to_micros(expr.value))
        else:
            formatted = _format_filter_value(expr.value)
        return f"{expr.field} {op} {formatted}"
    if isinstance(expr, FilterIn):
        values_str = ", ".join(_format_filter_value(v) for v in expr.values)
        return f"{expr.field} in [{values_str}]"
    if isinstance(expr, FilterIsNull):
        return f"{expr.field} is null"
    if isinstance(expr, FilterNot):
        return f"not ({_build_milvus_filter(expr.expr, datetime_fields)})"
    if isinstance(expr, FilterAnd):
        left = _build_milvus_filter(expr.left, datetime_fields)
        right = _build_milvus_filter(expr.right, datetime_fields)
        return f"({left}) and ({right})"
    if isinstance(expr, FilterOr):
        left = _build_milvus_filter(expr.left, datetime_fields)
        right = _build_milvus_filter(expr.right, datetime_fields)
        return f"({left}) or ({right})"
    msg = f"Unsupported filter expression type: {type(expr)}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _build_output_fields(
    properties_schema: Mapping[str, type[PropertyValue]],
    *,
    return_vector: bool,
    return_properties: bool,
) -> list[str]:
    """Build the output_fields list for Milvus search/get calls."""
    fields: list[str] = []
    if return_vector:
        fields.append(_FIELD_VECTOR)
    if return_properties:
        for prop_name, prop_type in properties_schema.items():
            fields.append(prop_name)
            if prop_type is datetime:
                fields.append(f"{prop_name}{_TZ_SUFFIX}")
    return fields


# ---------------------------------------------------------------------------
# Record serialization / deserialization
# ---------------------------------------------------------------------------


def _build_record_data(
    record: Record,
    properties_schema: Mapping[str, type[PropertyValue]],
) -> dict[str, Any]:
    """Serialize a Record to a Milvus-compatible dict for upsert."""
    data: dict[str, Any] = {
        _FIELD_UUID: str(record.uuid),
        _FIELD_VECTOR: record.vector,
    }
    for prop_name, prop_type in properties_schema.items():
        value = record.properties.get(prop_name) if record.properties else None
        if value is None:
            data[prop_name] = None
            if prop_type is datetime:
                data[f"{prop_name}{_TZ_SUFFIX}"] = None
        elif prop_type is datetime and isinstance(value, datetime):
            data[prop_name] = _datetime_to_micros(value)
            data[f"{prop_name}{_TZ_SUFFIX}"] = utc_offset_seconds(value)
        else:
            data[prop_name] = value
    return data


def _extract_entity(hit: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Milvus search hit or get result into a single dict.

    Search results nest output fields in ``entity``; get results put them
    at the top level.
    """
    if "entity" in hit:
        entity = dict(hit["entity"])
        entity[_FIELD_UUID] = hit["id"]
        if _FIELD_VECTOR in hit:
            entity[_FIELD_VECTOR] = hit[_FIELD_VECTOR]
        return entity
    return hit


def _parse_record(
    data: dict[str, Any],
    properties_schema: Mapping[str, type[PropertyValue]],
    *,
    include_vector: bool,
    include_properties: bool,
) -> Record:
    """Deserialize a Milvus result dict to a Record."""
    uuid = UUID(str(data[_FIELD_UUID]))

    vector: list[float] | None = None
    if include_vector:
        raw_vec = data.get(_FIELD_VECTOR)
        if raw_vec is not None:
            vector = list(raw_vec)

    properties: dict[str, PropertyValue] | None = None
    if include_properties:
        properties = {}
        for prop_name, prop_type in properties_schema.items():
            value = data.get(prop_name)
            if value is None:
                continue
            if prop_type is datetime:
                offset = data.get(f"{prop_name}{_TZ_SUFFIX}") or 0
                properties[prop_name] = _micros_to_datetime(int(value), int(offset))
            else:
                properties[prop_name] = value

    return Record(uuid=uuid, vector=vector, properties=properties)


# ---------------------------------------------------------------------------
# Config reconstruction from schema + index
# ---------------------------------------------------------------------------


def _to_data_type(raw: DataType | int) -> DataType:
    """Coerce an integer or DataType enum to DataType."""
    if isinstance(raw, DataType):
        return raw
    return DataType(raw)


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------


class MilvusVectorStoreParams(BaseModel):
    """Parameters for MilvusVectorStore.

    Attributes:
        client:
            Async Milvus client instance (lifecycle managed externally).
        index_type:
            Milvus vector index type (default: ``AUTOINDEX``).
            Examples: ``AUTOINDEX``, ``HNSW``, ``DISKANN``, ``SCANN``,
            ``IVF_FLAT``, ``IVF_SQ8``.
        metrics_factory:
            An instance of MetricsFactory for collecting usage metrics.
    """

    client: InstanceOf[AsyncMilvusClient] = Field(
        ...,
        description="Async Milvus client instance",
    )
    index_type: str = Field(
        "AUTOINDEX",
        description=(
            "Milvus vector index type. "
            "Examples: AUTOINDEX, HNSW, DISKANN, SCANN, IVF_FLAT, IVF_SQ8"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class MilvusCollection(Collection):
    """A collection backed by Milvus."""

    def __init__(
        self,
        *,
        client: AsyncMilvusClient,
        collection_name: str,
        properties_schema: Mapping[str, type[PropertyValue]],
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a Milvus client and collection name."""
        self._client = client
        self._collection_name = collection_name
        self._properties_schema = properties_schema
        self._tracker = tracker
        self._datetime_fields: set[str] = {
            name for name, ptype in properties_schema.items() if ptype is datetime
        }

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records into the collection."""
        async with self._tracker("upsert"):
            record_list = list(records)
            if not record_list:
                return

            with_vec: list[Record] = []
            without_vec: list[Record] = []
            for record in record_list:
                (with_vec if record.vector is not None else without_vec).append(record)

            # For records without vectors, fetch existing vectors and merge.
            if without_vec:
                existing = await self._client.get(
                    collection_name=self._collection_name,
                    ids=[str(r.uuid) for r in without_vec],
                    output_fields=[_FIELD_VECTOR],
                )
                existing_by_id: dict[str, list[float]] = {}
                for row in existing:
                    entity = _extract_entity(row)
                    pk = str(entity[_FIELD_UUID])
                    vec = entity.get(_FIELD_VECTOR)
                    if vec is not None:
                        existing_by_id[pk] = list(vec)

                for record in without_vec:
                    vec = existing_by_id.get(str(record.uuid))
                    if vec is None:
                        msg = (
                            f"Cannot upsert record {record.uuid} without a vector: "
                            "record does not exist and no vector was provided"
                        )
                        raise ValueError(msg)
                    with_vec.append(
                        Record(
                            uuid=record.uuid, vector=vec, properties=record.properties
                        )
                    )

            data = [_build_record_data(r, self._properties_schema) for r in with_vec]
            for i in range(0, len(data), _UPSERT_BATCH_SIZE):
                batch = data[i : i + _UPSERT_BATCH_SIZE]
                await self._client.upsert(
                    collection_name=self._collection_name, data=batch
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
            if property_filter and not validate_filter(property_filter):
                raise ValueError("Filter contains an invalid property key")

            filter_str = ""
            if property_filter:
                filter_str = _build_milvus_filter(
                    property_filter, self._datetime_fields
                )

            effective_limit = limit if limit is not None else _DEFAULT_SEARCH_LIMIT
            output_fields = _build_output_fields(
                self._properties_schema,
                return_vector=return_vector,
                return_properties=return_properties,
            )

            query_vector_lists = [list(v) for v in query_vectors]
            if not query_vector_lists:
                return []

            results = await self._client.search(
                collection_name=self._collection_name,
                data=query_vector_lists,
                filter=filter_str,
                limit=effective_limit,
                output_fields=output_fields or None,
            )

            query_results: list[QueryResult] = []
            for batch in results:
                matches: list[QueryMatch] = []
                for hit in batch:
                    score = float(hit["distance"])
                    if score_threshold is not None and score < score_threshold:
                        continue
                    entity = _extract_entity(hit)
                    record = _parse_record(
                        entity,
                        self._properties_schema,
                        include_vector=return_vector,
                        include_properties=return_properties,
                    )
                    matches.append(QueryMatch(score=score, record=record))
                query_results.append(QueryResult(matches=matches))

            return query_results

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

            output_fields = _build_output_fields(
                self._properties_schema,
                return_vector=return_vector,
                return_properties=return_properties,
            )
            uuid_strings = [str(u) for u in uuid_list]
            rows = await self._client.get(
                collection_name=self._collection_name,
                ids=uuid_strings,
                output_fields=output_fields or None,
            )

            rows_by_id: dict[str, dict[str, Any]] = {}
            for row in rows:
                entity = _extract_entity(row)
                rows_by_id[str(entity[_FIELD_UUID])] = entity

            records: list[Record] = []
            for uid in uuid_list:
                entity = rows_by_id.get(str(uid))
                if entity is None:
                    continue
                records.append(
                    _parse_record(
                        entity,
                        self._properties_schema,
                        include_vector=return_vector,
                        include_properties=return_properties,
                    )
                )
            return records

    @override
    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """Delete records from the collection by their UUIDs."""
        async with self._tracker("delete"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return
            await self._client.delete(
                collection_name=self._collection_name,
                ids=[str(u) for u in uuid_list],
            )


class MilvusVectorStore(VectorStore):
    """Asynchronous Milvus-based implementation of VectorStore.

    Uses one native Milvus collection per logical collection.  Collection
    configuration is reconstructed from the native schema and vector index,
    so no external registry is needed.
    """

    _SIMILARITY_METRIC_TO_MILVUS_METRIC_TYPE: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "COSINE",
        SimilarityMetric.DOT: "IP",
        SimilarityMetric.EUCLIDEAN: "L2",
    }

    _MILVUS_METRIC_TYPE_TO_SIMILARITY_METRIC: ClassVar[dict[str, SimilarityMetric]] = {
        v: k for k, v in _SIMILARITY_METRIC_TO_MILVUS_METRIC_TYPE.items()
    }

    _PROPERTY_TYPE_TO_MILVUS_DTYPE: ClassVar[dict[type[PropertyValue], DataType]] = {
        bool: DataType.BOOL,
        int: DataType.INT64,
        float: DataType.DOUBLE,
        str: DataType.VARCHAR,
        datetime: DataType.INT64,
    }

    @staticmethod
    def _milvus_metric_type(metric: SimilarityMetric) -> str:
        """Convert a SimilarityMetric to a Milvus metric type string."""
        mapping = MilvusVectorStore._SIMILARITY_METRIC_TO_MILVUS_METRIC_TYPE
        if metric not in mapping:
            msg = f"Milvus does not support the {metric.value} similarity metric"
            raise ValueError(msg)
        return mapping[metric]

    @staticmethod
    def _build_schema(config: CollectionConfig) -> CollectionSchema:
        """Build a Milvus CollectionSchema from a CollectionConfig."""
        fields: list[FieldSchema] = [
            FieldSchema(
                name=_FIELD_UUID,
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=_PK_MAX_LENGTH,
            ),
            FieldSchema(
                name=_FIELD_VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=config.vector_dimensions,
            ),
        ]
        dtype_map = MilvusVectorStore._PROPERTY_TYPE_TO_MILVUS_DTYPE
        for prop_name, prop_type in config.properties_schema.items():
            milvus_dtype = dtype_map[prop_type]
            kwargs: dict[str, Any] = {
                "name": prop_name,
                "dtype": milvus_dtype,
                "nullable": True,
            }
            if milvus_dtype == DataType.VARCHAR:
                kwargs["max_length"] = _VARCHAR_MAX_LENGTH
            fields.append(FieldSchema(**kwargs))
            if prop_type is datetime:
                fields.append(
                    FieldSchema(
                        name=f"{prop_name}{_TZ_SUFFIX}",
                        dtype=DataType.INT64,
                        nullable=True,
                    )
                )
        return CollectionSchema(fields=fields)

    @staticmethod
    def _extract_vector_dimensions(fields: list[dict[str, Any]]) -> int:
        """Extract the vector dimensions from the vector field in a Milvus schema."""
        for field in fields:
            if field["name"] == _FIELD_VECTOR:
                return field["params"]["dim"]
        msg = f"Schema has no vector field '{_FIELD_VECTOR}'"
        raise ValueError(msg)

    @staticmethod
    def _extract_properties_schema(
        fields: list[dict[str, Any]],
    ) -> dict[str, type[PropertyValue]]:
        """Reconstruct the properties_schema from Milvus field descriptors."""
        tz_companions: set[str] = {
            f["name"][: -len(_TZ_SUFFIX)]
            for f in fields
            if f["name"].endswith(_TZ_SUFFIX)
        }
        dtype_map: dict[DataType, type[PropertyValue]] = {
            DataType.BOOL: bool,
            DataType.DOUBLE: float,
            DataType.VARCHAR: str,
        }
        schema: dict[str, type[PropertyValue]] = {}
        for field in fields:
            name = field["name"]
            if name in (_FIELD_UUID, _FIELD_VECTOR) or name.endswith(_TZ_SUFFIX):
                continue
            dtype = _to_data_type(field["type"])
            if dtype == DataType.INT64:
                schema[name] = datetime if name in tz_companions else int
            elif dtype in dtype_map:
                schema[name] = dtype_map[dtype]
        return schema

    @staticmethod
    async def _extract_similarity_metric(
        client: AsyncMilvusClient,
        milvus_collection_name: str,
    ) -> SimilarityMetric:
        """Read the similarity metric from the vector index."""
        index_names: list[str] = await client.list_indexes(
            collection_name=milvus_collection_name,
        )
        for idx_name in index_names:
            idx_info = await client.describe_index(
                collection_name=milvus_collection_name,
                index_name=idx_name,
            )
            if idx_info.get("field_name") == _FIELD_VECTOR:
                metric_str = idx_info.get("metric_type", "")
                return MilvusVectorStore._MILVUS_METRIC_TYPE_TO_SIMILARITY_METRIC.get(
                    metric_str, SimilarityMetric.COSINE
                )
        return SimilarityMetric.COSINE

    @staticmethod
    async def _reconstruct_config(
        client: AsyncMilvusClient,
        milvus_collection_name: str,
    ) -> CollectionConfig:
        """Reconstruct a CollectionConfig from a Milvus collection's schema and index."""
        info = await client.describe_collection(
            collection_name=milvus_collection_name,
        )
        fields: list[dict[str, Any]] = info["fields"]
        return CollectionConfig(
            vector_dimensions=MilvusVectorStore._extract_vector_dimensions(fields),
            similarity_metric=await MilvusVectorStore._extract_similarity_metric(
                client, milvus_collection_name
            ),
            properties_schema=MilvusVectorStore._extract_properties_schema(fields),
        )

    def __init__(self, params: MilvusVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client: AsyncMilvusClient = params.client
        self._index_type = params.index_type
        self._tracker = OperationTracker(
            params.metrics_factory, prefix="vector_store_milvus"
        )
        self._name_locks: defaultdict[tuple[str, str], asyncio.Lock] = defaultdict(
            asyncio.Lock
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

    def _build_collection_handle(
        self, milvus_collection_name: str, config: CollectionConfig
    ) -> MilvusCollection:
        return MilvusCollection(
            client=self._client,
            collection_name=milvus_collection_name,
            properties_schema=config.properties_schema,
            tracker=self._tracker,
        )

    async def _create_milvus_collection(
        self, milvus_collection_name: str, config: CollectionConfig
    ) -> None:
        """Create the native Milvus collection, index, and load it."""
        schema = MilvusVectorStore._build_schema(config)
        await self._client.create_collection(
            collection_name=milvus_collection_name,
            schema=schema,
        )
        metric_type = MilvusVectorStore._milvus_metric_type(config.similarity_metric)
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=_FIELD_VECTOR,
            index_type=self._index_type,
            metric_type=metric_type,
        )
        await self._client.create_index(
            collection_name=milvus_collection_name,
            index_params=index_params,
        )
        await self._client.load_collection(collection_name=milvus_collection_name)

    @override
    async def startup(self) -> None:
        """No-op; client lifecycle is managed externally."""

    @override
    async def shutdown(self) -> None:
        """No-op; client lifecycle is managed externally."""

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: CollectionConfig,
    ) -> None:
        """Create a logical collection in the Milvus vector store."""
        self._validate_namespace_and_name(namespace, name)
        lock = self._name_locks[(namespace, name)]
        async with lock, self._tracker("create_collection"):
            milvus_name = _to_milvus_collection_name(namespace, name)
            if await self._client.has_collection(collection_name=milvus_name):
                raise CollectionAlreadyExistsError(namespace, name)
            await self._create_milvus_collection(milvus_name, config)

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: CollectionConfig,
    ) -> MilvusCollection:
        """Open the collection if it exists, or create and return it."""
        self._validate_namespace_and_name(namespace, name)
        lock = self._name_locks[(namespace, name)]
        async with lock, self._tracker("open_or_create_collection"):
            milvus_name = _to_milvus_collection_name(namespace, name)
            if await self._client.has_collection(collection_name=milvus_name):
                existing_config = await MilvusVectorStore._reconstruct_config(
                    self._client, milvus_name
                )
                if existing_config != config:
                    raise CollectionConfigMismatchError(
                        namespace, name, existing_config, config
                    )
                return self._build_collection_handle(milvus_name, existing_config)
            await self._create_milvus_collection(milvus_name, config)
            return self._build_collection_handle(milvus_name, config)

    @override
    async def open_collection(
        self, *, namespace: str, name: str
    ) -> MilvusCollection | None:
        """Get a collection handle from the Milvus vector store."""
        self._validate_namespace_and_name(namespace, name)
        milvus_name = _to_milvus_collection_name(namespace, name)
        if not await self._client.has_collection(collection_name=milvus_name):
            return None
        config = await MilvusVectorStore._reconstruct_config(self._client, milvus_name)
        return self._build_collection_handle(milvus_name, config)

    @override
    async def close_collection(self, *, collection: Collection) -> None:
        """No-op; Milvus collection handles require no explicit close."""

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        """Delete a logical collection from the Milvus vector store."""
        self._validate_namespace_and_name(namespace, name)
        lock = self._name_locks[(namespace, name)]
        async with lock, self._tracker("delete_collection"):
            milvus_name = _to_milvus_collection_name(namespace, name)
            await self._client.drop_collection(collection_name=milvus_name)
