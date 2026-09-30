"""Milvus-based vector store implementation."""

import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID

from pydantic import Field, InstanceOf
from pymilvus import AsyncMilvusClient, DataType
from pymilvus.exceptions import MilvusException

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
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
from memmachine_server.common.metrics_factory import OperationTracker
from memmachine_server.common.properties_json import (
    PROPERTY_VALUE_KEY,
    encode_properties,
)
from memmachine_server.common.utils import ensure_tz_aware, utc_offset_seconds

from .collection_registry import RegisteredCollection
from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionConfig,
)
from .declared_properties import require_declared_types
from .registry_backed_vector_store import (
    RegistryBackedVectorStore,
    RegistryBackedVectorStoreCollection,
    RegistryBackedVectorStoreParams,
)
from .utils import require_dimensions, require_valid_query_vector, validate_filter

_ID_FIELD = "id"
_RECORD_UUID_FIELD = "record_uuid"
_PARTITION_KEY_FIELD = "partition_key"
"""The native partition-key field, holding the incarnation of the collection an entity belongs to.

A collection created again under a deleted one's name gets a fresh
incarnation, so the deleted collection's entities are not part of it.
"""
_VECTOR_FIELD = "vector"
_PROPERTIES_FIELD = "properties"
"""A JSON field holding the properties the collection's schema does not declare."""
_DECLARED_FIELD_PREFIX = "_p_"
"""The prefix of the typed field holding a declared property."""
_OFFSET_FIELD_PREFIX = "_tz_"
"""The prefix of the field holding a declared datetime's UTC offset in seconds, which TIMESTAMPTZ drops."""

_UUID_LENGTH = 36
"""The length of a UUID's hyphenated text form (RFC 9562), the form stored."""
_PRIMARY_ID_LENGTH = 2 * _UUID_LENGTH + 1
"""The length of a primary key, `"{incarnation}:{record_uuid}"`."""
_FALSE_EXPR = f"{_ID_FIELD} in []"
"""A Milvus expression no entity satisfies; Milvus refuses a bare `false`."""

_DECLARED_DATA_TYPES: dict[type[PropertyValue], DataType] = {
    bool: DataType.BOOL,
    int: DataType.INT64,
    float: DataType.DOUBLE,
    str: DataType.VARCHAR,
    datetime: DataType.TIMESTAMPTZ,
}

# The index AUTOINDEX builds on CPU from Milvus 2.6.10, named so every server
# builds it, whatever its version or AUTOINDEX configuration; partition-key
# isolation needs the HNSW family.
_VECTOR_INDEX_TYPE = "HNSW_SQ"
_VECTOR_INDEX_PARAMS: dict[str, Any] = {
    "M": 18,
    "efConstruction": 240,
    "sq_type": "SQ4U",
    "refine": True,
    "refine_type": "FP16",
}
# Candidates per result rescored against the half-precision vectors.
_SEARCH_REFINE_K = 8


def _expr_string(value: str) -> str:
    """Return a Milvus expression string literal."""
    return json.dumps(value)


def _literal(value: PropertyValue) -> str:
    """Return a Milvus expression literal for a property value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, datetime):
        return _expr_string(ensure_tz_aware(value).astimezone(UTC).isoformat())
    return _expr_string(value)


def _declared_literal(value: PropertyValue) -> str:
    """Return a Milvus expression literal for a declared property's value.

    A datetime is a TIMESTAMPTZ instant literal, which compares instants
    whatever the offsets.
    """
    if isinstance(value, datetime):
        return f"ISO '{ensure_tz_aware(value).astimezone(UTC).isoformat()}'"
    return _literal(value)


def _fits(value: PropertyValue, declared_type: type[PropertyValue]) -> bool:
    """Whether a value can be compared with, or stored in, a declared property."""
    if isinstance(value, bool):
        return declared_type is bool
    if isinstance(value, int | float):
        return declared_type in (int, float)
    return isinstance(value, declared_type)


def _absent(key: str, declared: Mapping[str, type[PropertyValue]]) -> str:
    """A Milvus expression true exactly where the property has no value."""
    if key in declared:
        return f"{_DECLARED_FIELD_PREFIX}{key} is null"
    return f"not exists {_PROPERTIES_FIELD}[{_expr_string(key)}]"


def _condition(
    expr: FilterComparison | FilterIn,
    declared: Mapping[str, type[PropertyValue]],
) -> str:
    """A Milvus expression for one condition; false or null where the property has no value."""
    values = list(expr.values) if isinstance(expr, FilterIn) else [expr.value]
    declared_type = declared.get(expr.field)
    if declared_type is None:
        target = (
            f"{_PROPERTIES_FIELD}[{_expr_string(expr.field)}]"
            f"[{_expr_string(PROPERTY_VALUE_KEY)}]"
        )
        render = _literal
    else:
        # A value of another type never equals or orders against the property.
        values = [value for value in values if _fits(value, declared_type)]
        target = f"{_DECLARED_FIELD_PREFIX}{expr.field}"
        render = _declared_literal
    if not values:
        return _FALSE_EXPR
    match expr:
        case FilterIn():
            return f"{target} in [{', '.join(render(value) for value in values)}]"
        case FilterComparison(op="="):
            return f"{target} == {render(values[0])}"
        case FilterComparison(op=op):
            return f"{target} {op} {render(values[0])}"


def _milvus_filter(
    expr: FilterExpr,
    declared: Mapping[str, type[PropertyValue]],
    *,
    negate: bool = False,
) -> str:
    """Compile a filter, or with `negate` its complement, into a Milvus expression.

    A negation holds where the property has no value. Milvus evaluates a
    condition on a null the SQL way, so negation is pushed down to the
    conditions.
    """
    match expr:
        case FilterNot(operand):
            return _milvus_filter(operand, declared, negate=not negate)
        case FilterAnd(left, right) | FilterOr(left, right):
            operator = "&&" if isinstance(expr, FilterAnd) != negate else "||"
            return f" {operator} ".join(
                f"({_milvus_filter(operand, declared, negate=negate)})"
                for operand in (left, right)
            )
        case FilterIsNull(field):
            absent = _absent(field, declared)
            return f"not ({absent})" if negate else absent
        case FilterComparison(field, "!=", value):
            equal = FilterComparison(field=field, op="=", value=value)
            return _milvus_filter(equal, declared, negate=not negate)
        case FilterComparison() | FilterIn():
            condition = _condition(expr, declared)
            if negate:
                return f"(not ({condition})) || ({_absent(expr.field, declared)})"
            return condition
        case _:
            raise TypeError(f"Unsupported filter expression type: {type(expr)}")


def _require_every_key_accepted(result: Mapping[str, int], sent: int) -> None:
    """Raise unless Milvus accepted the delete of every primary key sent.

    Milvus counts the primary keys a delete accepts, present or not, so a
    delete it rejected counts fewer; pymilvus's async client returns rather
    than raises for one.
    """
    accepted = result["delete_count"]
    if accepted != sent:
        raise MilvusException(
            message=f"Milvus accepted the delete of {accepted} of {sent} primary keys"
        )


def _incarnation_filter(incarnation: UUID) -> str:
    """A Milvus expression matching the entities of one collection incarnation."""
    return f"{_PARTITION_KEY_FIELD} == {_expr_string(str(incarnation))}"


class MilvusVectorStoreCollection(RegistryBackedVectorStoreCollection):
    """A logical collection backed by Milvus."""

    @staticmethod
    def _passes_threshold(
        score: float,
        threshold: float | None,
        similarity_metric: SimilarityMetric,
    ) -> bool:
        if threshold is None:
            return True
        if similarity_metric.higher_is_better:
            return score >= threshold
        return score <= threshold

    def __init__(
        self,
        *,
        client: AsyncMilvusClient,
        native_collection_name: str,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
        tracker: OperationTracker,
        is_live: Callable[[UUID], Awaitable[bool]],
        request_timeout_seconds: int,
    ) -> None:
        """Initialize with a Milvus client and the incarnation the handle is bound to."""
        super().__init__(
            namespace=namespace,
            name=name,
            incarnation=incarnation,
            config=config,
            tracker=tracker,
            is_live=is_live,
        )
        self._client = client
        self._native_collection_name = native_collection_name
        self._request_timeout_seconds = request_timeout_seconds

    def _primary_id(self, record_uuid: UUID) -> str:
        """The primary key of a record: the incarnation and the record UUID.

        Collections sharing a native collection never share a primary key, and
        neither do a deleted collection and one created again under its name.
        """
        return f"{self._incarnation}:{record_uuid}"

    def _build_entity(self, record: Record) -> dict[str, Any]:
        """Build a Milvus entity from a vector store record."""
        declared = self._config.indexed_properties_schema
        require_declared_types(record.properties, declared)
        require_dimensions(record.vector, self._config.vector_dimensions)
        entity: dict[str, Any] = {
            _ID_FIELD: self._primary_id(record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: str(self._incarnation),
            _VECTOR_FIELD: record.vector,
            _PROPERTIES_FIELD: encode_properties(
                {
                    key: value
                    for key, value in record.properties.items()
                    if key not in declared
                }
            ),
        }
        for key, declared_type in declared.items():
            value = record.properties.get(key)
            if isinstance(value, datetime):
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = ensure_tz_aware(
                    value
                ).isoformat()
                entity[f"{_OFFSET_FIELD_PREFIX}{key}"] = utc_offset_seconds(value)
            elif declared_type is datetime:
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = None
                entity[f"{_OFFSET_FIELD_PREFIX}{key}"] = None
            else:
                entity[f"{_DECLARED_FIELD_PREFIX}{key}"] = value
        return entity

    def _score(self, distance: float) -> float:
        """The store's score for a distance Milvus returned.

        Milvus returns cosine similarity and inner product as they are, and
        the squared Euclidean distance.
        """
        if self._config.similarity_metric is SimilarityMetric.EUCLIDEAN:
            return math.sqrt(max(distance, 0.0))
        return distance

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        async with self._tracker("upsert"):
            records = list(records)
            if not records:
                return

            entities = [self._build_entity(record) for record in records]
            await self._fence()
            await self._client.upsert(
                collection_name=self._native_collection_name,
                data=entities,
                timeout=self._request_timeout_seconds,
            )
            await self._fence()

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            if not query_vectors:
                return []
            for query_vector in query_vectors:
                require_valid_query_vector(query_vector, self._config.vector_dimensions)
            if limit <= 0:
                return [QueryResult(matches=[]) for _ in query_vectors]

            await self._fence()
            filter_expr = _incarnation_filter(self._incarnation)
            if property_filter is not None:
                if not validate_filter(property_filter):
                    raise ValueError("Filter contains an invalid property key")
                property_expr = _milvus_filter(
                    property_filter, self._config.indexed_properties_schema
                )
                filter_expr = f"({filter_expr}) && ({property_expr})"

            raw_results = await self._client.search(
                collection_name=self._native_collection_name,
                data=query_vectors,
                filter=filter_expr,
                limit=limit,
                search_params={"params": {"refine_k": _SEARCH_REFINE_K}},
                output_fields=[_RECORD_UUID_FIELD],
                anns_field=_VECTOR_FIELD,
                timeout=self._request_timeout_seconds,
            )

            results: list[QueryResult] = []
            for raw_matches in raw_results:
                matches: list[QueryMatch] = []
                for raw_match in raw_matches:
                    entity = cast(Mapping[str, Any], raw_match["entity"])
                    score = self._score(raw_match["distance"])
                    if not self._passes_threshold(
                        score, score_threshold, self._config.similarity_metric
                    ):
                        continue

                    matches.append(
                        QueryMatch(
                            score=score,
                            record_uuid=UUID(str(entity[_RECORD_UUID_FIELD])),
                        )
                    )

                matches.sort(
                    key=lambda match: match.score,
                    reverse=self._config.similarity_metric.higher_is_better,
                )
                results.append(QueryResult(matches=matches))

            return results

    @override
    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        async with self._tracker("delete"):
            uuid_list = list(record_uuids)
            if not uuid_list:
                return
            await self._fence()
            primary_ids = [self._primary_id(uuid) for uuid in uuid_list]
            result = await self._client.delete(
                collection_name=self._native_collection_name,
                ids=primary_ids,
                timeout=self._request_timeout_seconds,
            )
            _require_every_key_accepted(result, len(primary_ids))
            await self._fence()


class MilvusVectorStoreParams(RegistryBackedVectorStoreParams):
    """
    Parameters for MilvusVectorStore.

    Attributes:
        client (AsyncMilvusClient):
            Async Milvus client instance.
        request_timeout_seconds (int):
            Seconds any request to Milvus may take (default: 30).
        max_varchar_length (int):
            Bytes a declared string property can hold: the length of its
            VARCHAR field, at most the server's proxy.maxVarCharLength
            (default: 65535).
        purge_batch_size (int):
            The most entities one purge round lists and deletes, at most the
            server's quotaAndLimits.limits.maxQueryResultWindow (default:
            10000).
    """

    client: InstanceOf[AsyncMilvusClient] = Field(
        ...,
        description="Async Milvus client instance",
    )
    request_timeout_seconds: int = Field(
        30, gt=0, description="Seconds any request to Milvus may take"
    )
    max_varchar_length: int = Field(
        65535,
        gt=0,
        description=(
            "Bytes a declared string property can hold: the length of its "
            "VARCHAR field, at most the server's proxy.maxVarCharLength"
        ),
    )
    purge_batch_size: int = Field(
        10000,
        gt=0,
        description=(
            "The most entities one purge round lists and deletes, at most the "
            "server's quotaAndLimits.limits.maxQueryResultWindow"
        ),
    )


class MilvusVectorStore(RegistryBackedVectorStore[MilvusVectorStoreCollection]):
    """Asynchronous Milvus-based implementation of VectorStore.

    A logical collection is the entities carrying its incarnation in the
    partition-key field.

    Reads run at Milvus's default consistency level, Bounded: a query
    reflects every write that returned at least the server's
    `common.gracefulTime` (5 s by default) before it began.
    """

    _SIMILARITY_METRIC_TO_MILVUS_METRIC: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "COSINE",
        SimilarityMetric.DOT: "IP",
        SimilarityMetric.EUCLIDEAN: "L2",
    }

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        """Check if an exception indicates a resource already exists."""
        message = str(error).lower()
        return "already exist" in message or "already exists" in message

    @staticmethod
    def _build_native_collection_name(
        namespace: str, config: VectorStoreCollectionConfig
    ) -> str:
        """Build a deterministic native collection name from namespace and config."""
        digest = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
        return f"memmachine_{namespace}__{digest}"

    @staticmethod
    def _validate_metric(similarity_metric: SimilarityMetric) -> None:
        if (
            similarity_metric
            not in MilvusVectorStore._SIMILARITY_METRIC_TO_MILVUS_METRIC
        ):
            supported = ", ".join(
                metric.value
                for metric in MilvusVectorStore._SIMILARITY_METRIC_TO_MILVUS_METRIC
            )
            raise ValueError(
                f"Milvus only supports {supported} similarity metrics, "
                f"got {similarity_metric.value!r}"
            )

    def __init__(self, params: MilvusVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__(params, metrics_prefix="vector_store_milvus")
        self._client = params.client
        self._request_timeout_seconds = params.request_timeout_seconds
        self._max_varchar_length = params.max_varchar_length
        self._purge_batch_size = params.purge_batch_size

    @override
    def _build_collection_handle(
        self, namespace: str, name: str, registered: RegisteredCollection
    ) -> MilvusVectorStoreCollection:
        return MilvusVectorStoreCollection(
            client=self._client,
            native_collection_name=MilvusVectorStore._build_native_collection_name(
                namespace, registered.config
            ),
            namespace=namespace,
            name=name,
            incarnation=registered.incarnation,
            config=registered.config,
            tracker=self._tracker,
            is_live=self._collection_registry.is_live,
            request_timeout_seconds=self._request_timeout_seconds,
        )

    @override
    async def _create_native_collection(
        self, namespace: str, config: VectorStoreCollectionConfig
    ) -> None:
        # Created, indexed and loaded as separate steps, each when missing.
        self._validate_metric(config.similarity_metric)
        native_collection_name = MilvusVectorStore._build_native_collection_name(
            namespace, config
        )
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=_VECTOR_FIELD,
            index_name=_VECTOR_FIELD,
            index_type=_VECTOR_INDEX_TYPE,
            metric_type=self._SIMILARITY_METRIC_TO_MILVUS_METRIC[
                config.similarity_metric
            ],
            params=_VECTOR_INDEX_PARAMS,
        )
        for key in config.indexed_properties_schema:
            index_params.add_index(
                field_name=f"{_DECLARED_FIELD_PREFIX}{key}",
                index_name=f"{_DECLARED_FIELD_PREFIX}{key}",
                index_type="AUTOINDEX",
            )

        async def _create_collection() -> None:
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name=_ID_FIELD,
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=_PRIMARY_ID_LENGTH,
            )
            schema.add_field(
                field_name=_RECORD_UUID_FIELD,
                datatype=DataType.VARCHAR,
                max_length=_UUID_LENGTH,
            )
            schema.add_field(
                field_name=_PARTITION_KEY_FIELD,
                datatype=DataType.VARCHAR,
                max_length=_UUID_LENGTH,
                is_partition_key=True,
            )
            schema.add_field(
                field_name=_VECTOR_FIELD,
                datatype=DataType.FLOAT_VECTOR,
                dim=config.vector_dimensions,
            )
            schema.add_field(
                field_name=_PROPERTIES_FIELD,
                datatype=DataType.JSON,
            )
            for key, declared_type in config.indexed_properties_schema.items():
                if declared_type is str:
                    schema.add_field(
                        field_name=f"{_DECLARED_FIELD_PREFIX}{key}",
                        datatype=DataType.VARCHAR,
                        max_length=self._max_varchar_length,
                        nullable=True,
                    )
                else:
                    schema.add_field(
                        field_name=f"{_DECLARED_FIELD_PREFIX}{key}",
                        datatype=_DECLARED_DATA_TYPES[declared_type],
                        nullable=True,
                    )
                if declared_type is datetime:
                    schema.add_field(
                        field_name=f"{_OFFSET_FIELD_PREFIX}{key}",
                        datatype=DataType.INT32,
                        nullable=True,
                    )

            # Without index_params, so the indexes and the load below are
            # steps of their own.
            await self._client.create_collection(
                collection_name=native_collection_name,
                schema=schema,
                properties={"partitionkey.isolation": True},
                timeout=self._request_timeout_seconds,
            )

        if not await self._client.has_collection(
            native_collection_name,
            timeout=self._request_timeout_seconds,
        ):
            try:
                await _create_collection()
            except MilvusException as exc:
                if not MilvusVectorStore._is_already_exists_error(exc):
                    raise

        # Index names are their field names, so the ones present say which
        # fields are indexed. An index a racing creator made is created again
        # without error.
        existing = set(
            await self._client.list_indexes(
                native_collection_name,
                timeout=self._request_timeout_seconds,
            )
        )
        missing = self._client.prepare_index_params()
        missing.extend(
            index for index in index_params if index.index_name not in existing
        )
        if missing:
            await self._client.create_index(
                native_collection_name,
                missing,
                timeout=self._request_timeout_seconds,
            )
        # A no-op when the collection is already loaded.
        await self._client.load_collection(
            native_collection_name,
            timeout=self._request_timeout_seconds,
        )

    @override
    async def _purge_round(
        self, namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> bool:
        # Lists up to a batch of the incarnation's entities and deletes them by
        # primary key. A batch keeps each delete small; one filter-delete of a
        # large incarnation stalls every tenant while Milvus applies it.
        native_collection_name = MilvusVectorStore._build_native_collection_name(
            namespace, config
        )
        if not await self._client.has_collection(
            native_collection_name,
            timeout=self._request_timeout_seconds,
        ):
            # The native collection is gone with everything in it.
            return False
        listed = await self._client.query(
            collection_name=native_collection_name,
            filter=_incarnation_filter(incarnation),
            output_fields=[_ID_FIELD],
            limit=self._purge_batch_size,
            timeout=self._request_timeout_seconds,
        )
        primary_ids = [entity[_ID_FIELD] for entity in listed]
        if primary_ids:
            result = await self._client.delete(
                collection_name=native_collection_name,
                ids=primary_ids,
                timeout=self._request_timeout_seconds,
            )
            _require_every_key_accepted(result, len(primary_ids))
        return bool(primary_ids)
