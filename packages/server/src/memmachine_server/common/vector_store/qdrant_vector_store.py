"""Qdrant-based vector store implementation."""

import asyncio
import hashlib
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID, uuid4, uuid5
from weakref import WeakKeyDictionary

import grpc
import grpc.aio
from pydantic import BaseModel, Field, InstanceOf
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from memmachine_server.common.data_types import (
    OrderedValue,
    PropertyValue,
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
from memmachine_server.common.utils import ensure_tz_aware

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionHandleStaleError,
)
from .utils import validate_filter, validate_identifier
from .vector_store import VectorStore, VectorStoreCollection

# Point payload keys (stored on every Qdrant point).
# System keys use _SYSTEM_KEY_PREFIX, which contains a hyphen. Hyphens are valid in
# Qdrant but forbidden by _IDENTIFIER_RE, so system keys can never collide with user keys.
_SYSTEM_KEY_PREFIX = "sys-"
_PAYLOAD_INCARNATION = f"{_SYSTEM_KEY_PREFIX}incarnation"
"""The payload key naming the collection incarnation a point belongs to.

Points carry the incarnation, never the collection name: a collection
deleted and re-created under the same name gets a fresh incarnation, and
its predecessor's points are invisible to it while the purge reclaims them.
"""
_PAYLOAD_RECORD_UUID = f"{_SYSTEM_KEY_PREFIX}record_uuid"
"""The payload key holding the record's uuid, which the point id is derived from."""


def _incarnation_filter(incarnation: UUID) -> models.Filter:
    """Build a Qdrant filter that matches the points of one collection incarnation."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_PAYLOAD_INCARNATION,
                match=models.MatchValue(value=incarnation.hex),
            ),
        ],
    )


def _point_id(incarnation: UUID, record_uuid: UUID) -> UUID:
    """The id of a record's point: one per (incarnation, record uuid).

    Point ids are global within a physical collection, which logical
    collections share. Deriving the id from the incarnation keeps a stale
    handle, or a co-tenant, from overwriting a point that a live incarnation
    holds under the same record uuid.
    """
    return uuid5(incarnation, str(record_uuid))


class QdrantVectorStoreCollection(VectorStoreCollection):
    """A collection backed by Qdrant."""

    _RANGE_OPERATORS: ClassVar[dict[str, str]] = {
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
    }

    @staticmethod
    def _build_qdrant_filter(expr: FilterExpr) -> models.Filter:
        """Convert a FilterExpr tree into a Qdrant Filter."""
        if isinstance(expr, FilterComparison):
            return QdrantVectorStoreCollection._build_qdrant_comparison(expr)
        if isinstance(expr, FilterIn):
            return QdrantVectorStoreCollection._in_filter(expr.field, expr.values)
        if isinstance(expr, FilterIsNull):
            return QdrantVectorStoreCollection._null_filter(expr.field, negate=False)
        if isinstance(expr, FilterNot):
            return models.Filter(
                must_not=[QdrantVectorStoreCollection._build_qdrant_filter(expr.expr)]
            )
        if isinstance(expr, FilterAnd):
            left = QdrantVectorStoreCollection._build_qdrant_filter(expr.left)
            right = QdrantVectorStoreCollection._build_qdrant_filter(expr.right)
            return models.Filter(must=[left, right])
        if isinstance(expr, FilterOr):
            left = QdrantVectorStoreCollection._build_qdrant_filter(expr.left)
            right = QdrantVectorStoreCollection._build_qdrant_filter(expr.right)
            return models.Filter(should=[left, right])
        message = f"Unsupported filter expression type: {type(expr)}"
        raise TypeError(message)

    @staticmethod
    def _build_qdrant_comparison(comparison: FilterComparison) -> models.Filter:
        """Convert a Comparison into a Qdrant Filter."""
        field = comparison.field
        operator = comparison.op
        value = comparison.value

        if operator in ("=", "!="):
            negate = operator == "!="
            if isinstance(value, float):
                return QdrantVectorStoreCollection._float_eq_filter(
                    field, value, negate=negate
                )
            if isinstance(value, datetime):
                return QdrantVectorStoreCollection._datetime_eq_filter(
                    field, value, negate=negate
                )
            return QdrantVectorStoreCollection._match_filter(
                field, value, negate=negate
            )
        if operator in QdrantVectorStoreCollection._RANGE_OPERATORS:
            if not isinstance(value, OrderedValue):
                message = (
                    f"Range filter on '{field}' requires a numeric or datetime value, "
                    f"got {type(value).__name__}"
                )
                raise TypeError(message)
            return QdrantVectorStoreCollection._range_filter(
                field, value, QdrantVectorStoreCollection._RANGE_OPERATORS[operator]
            )

        message = f"Unsupported filter operator: {operator}"
        raise ValueError(message)

    @staticmethod
    def _match_filter(
        field: str,
        value: bool | int | str,
        *,
        negate: bool,
    ) -> models.Filter:
        condition = models.FieldCondition(
            key=field,
            match=models.MatchValue(value=value),
        )
        if negate:
            return models.Filter(must_not=[condition])
        return models.Filter(must=[condition])

    @staticmethod
    def _float_eq_filter(field: str, value: float, *, negate: bool) -> models.Filter:
        """Use a range filter for float equality since MatchValue doesn't accept floats."""
        condition = models.FieldCondition(
            key=field,
            range=models.Range(gte=value, lte=value),
        )
        if negate:
            return models.Filter(must_not=[condition])
        return models.Filter(must=[condition])

    @staticmethod
    def _datetime_eq_filter(
        field: str, value: datetime, *, negate: bool
    ) -> models.Filter:
        """Use a DatetimeRange filter for datetime equality since MatchValue doesn't accept datetimes."""
        value = ensure_tz_aware(value)
        condition = models.FieldCondition(
            key=field,
            range=models.DatetimeRange(gte=value, lte=value),
        )
        if negate:
            return models.Filter(must_not=[condition])
        return models.Filter(must=[condition])

    @staticmethod
    def _in_filter(field: str, value: list[int] | list[str]) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=field,
                    match=models.MatchAny(any=value),
                ),
            ],
        )

    @staticmethod
    def _range_filter(
        field: str,
        value: OrderedValue,
        range_parameter: str,
    ) -> models.Filter:
        if isinstance(value, datetime):
            value = ensure_tz_aware(value)
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key=field,
                        range=models.DatetimeRange(**{range_parameter: value}),
                    ),
                ],
            )

        return models.Filter(
            must=[
                models.FieldCondition(
                    key=field,
                    range=models.Range(**{range_parameter: value}),
                ),
            ],
        )

    @staticmethod
    def _null_filter(field: str, *, negate: bool) -> models.Filter:
        condition = models.IsEmptyCondition(
            is_empty=models.PayloadField(key=field),
        )
        if negate:
            return models.Filter(must_not=[condition])
        return models.Filter(must=[condition])

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        collection_name: str,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
        tracker: OperationTracker,
        is_live: Callable[[str, str, UUID], Awaitable[bool]],
    ) -> None:
        """Initialize with a Qdrant client and the incarnation the handle is bound to."""
        self._client = client
        self._tracker = tracker
        self._collection_name = collection_name
        self._namespace = namespace
        self._name = name
        self._incarnation = incarnation
        self._config = config
        self._is_live = is_live

    async def _fence(self) -> None:
        """Raise if this handle's incarnation is no longer the collection's.

        Qdrant has no transactions, so the check and the operation are two
        calls; a deletion landing between them leaves points under a dead
        incarnation, which the purge reclaims like any other.
        """
        if not await self._is_live(self._namespace, self._name, self._incarnation):
            raise VectorStoreCollectionHandleStaleError(self._namespace, self._name)

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        """The configuration for this collection."""
        return self._config

    def _build_payload(
        self,
        properties: dict[str, PropertyValue] | None,
    ) -> dict[str, PropertyValue]:
        """Build Qdrant-compatible payload from record properties."""
        payload: dict[str, PropertyValue] = {
            _PAYLOAD_INCARNATION: self._incarnation.hex,
        }
        if properties:
            for key, value in properties.items():
                if value is None:
                    continue
                if isinstance(value, datetime):
                    payload[key] = ensure_tz_aware(value)
                else:
                    payload[key] = value
        return payload

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records into the collection."""
        async with self._tracker("upsert"):
            await self._fence()
            points: list[models.PointStruct] = []
            for record in records:
                payload = self._build_payload(record.properties)
                payload[_PAYLOAD_RECORD_UUID] = str(record.uuid)
                points.append(
                    models.PointStruct(
                        id=_point_id(self._incarnation, record.uuid),
                        vector=record.vector,
                        payload=payload,
                    )
                )
            if points:
                await self._upsert_with_backoff(points)

    async def _upsert_with_backoff(self, points: Iterable[models.PointStruct]) -> None:
        """Upsert points, splitting the batch in half on failure and retrying."""
        points = list(points)
        try:
            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
        except (ResponseHandlingException, UnexpectedResponse):
            if len(points) <= 1:
                raise
            mid = len(points) // 2
            await self._upsert_with_backoff(points[:mid])
            await self._upsert_with_backoff(points[mid:])

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        min_cosine_similarity: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        """Query for records matching the criteria by query vectors."""
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            if not query_vectors:
                return []
            await self._fence()

            incarnation_filter = _incarnation_filter(self._incarnation)
            if property_filter:
                if not validate_filter(property_filter):
                    raise ValueError("Filter contains an invalid property key")
                property_qdrant_filter = (
                    QdrantVectorStoreCollection._build_qdrant_filter(property_filter)
                )
                qdrant_filter = models.Filter(
                    must=[incarnation_filter, property_qdrant_filter]
                )
            else:
                qdrant_filter = incarnation_filter

            requests = [
                models.QueryRequest(
                    query=query_vector,
                    filter=qdrant_filter,
                    score_threshold=min_cosine_similarity,
                    limit=limit,
                    with_vector=False,
                    with_payload=[_PAYLOAD_RECORD_UUID],
                )
                for query_vector in query_vectors
            ]

            batch_results = await self._client.query_batch_points(
                collection_name=self._collection_name,
                requests=requests,
            )

            query_results: list[QueryResult] = []
            for batch in batch_results:
                matches = [
                    QueryMatch(
                        cosine_similarity=point.score,
                        record_uuid=UUID(
                            cast(dict[str, Any], point.payload)[_PAYLOAD_RECORD_UUID]
                        ),
                    )
                    for point in batch.points
                ]
                query_results.append(QueryResult(matches=matches))

            return query_results

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
            await self._fence()

            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.PointIdsList(
                    points=[
                        _point_id(self._incarnation, record_uuid)
                        for record_uuid in uuid_list
                    ],
                ),
            )


class QdrantVectorStoreParams(BaseModel):
    """
    Parameters for QdrantVectorStore.

    Attributes:
        client (AsyncQdrantClient):
            Async Qdrant client instance.
        registry_replication_factor (int):
            Replication factor for registry collections. Write consistency factor is
            set to match so all replicas confirm writes before returning, guaranteeing
            read-your-writes from any available replica.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).

    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )
    registry_replication_factor: int = Field(
        1,
        description=(
            "Replication factor for registry collections. Write consistency factor is "
            "set to match so all replicas confirm writes before returning, guaranteeing "
            "read-your-writes from any available replica"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class QdrantVectorStore(VectorStore):
    """Asynchronous Qdrant-based implementation of VectorStore.

    A logical collection is identified to callers by its (namespace, name)
    pair and inside the store by an incarnation minted when it is created.
    Its points carry the incarnation, never the name, so a collection deleted
    and re-created under the same pair starts empty and its predecessor's
    points are never adopted by, or reclaimed out from under, the successor.
    A handle is bound to one incarnation: once that incarnation is deleted,
    every operation of the handle raises `VectorStoreCollectionHandleStaleError`.

    `delete_collection` is two registry writes: the collection's live entry
    goes, making it unreachable at once, and an entry in the store's purge
    queue follows. `purge_deleted_collections` reclaims the points afterward,
    oldest deletion first, one collection per call. Qdrant has no
    transactions: a write that passed its fence before the deletion lands
    under the dead incarnation and is reclaimed with it, and a crash between
    the two registry writes leaves the points unreachable and unreclaimed, a
    leak, never a collection the purge takes from under a live one.

    Physical collections are shared by every logical collection of a
    namespace with the same configuration; the per-namespace registry holds
    one point per live logical collection, and the purge queue, one per
    store, one point per deleted incarnation.
    """

    _QDRANT_DISTANCE: ClassVar[models.Distance] = models.Distance.COSINE

    _PROPERTY_TYPE_TO_INDEX_TYPE: ClassVar[
        dict[type[PropertyValue], models.PayloadSchemaType]
    ] = {
        bool: models.PayloadSchemaType.BOOL,
        int: models.PayloadSchemaType.INTEGER,
        float: models.PayloadSchemaType.FLOAT,
        str: models.PayloadSchemaType.KEYWORD,
        datetime: models.PayloadSchemaType.DATETIME,
    }

    # Registry collection keys (stored on registry points, one per live logical
    # collection; the point id is uuid5 of the name).
    _REGISTRY_SUFFIX: ClassVar[str] = "__registry"
    _REGISTRY_NAME: ClassVar[str] = "name"
    _REGISTRY_INCARNATION: ClassVar[str] = "incarnation"
    _REGISTRY_VECTOR_DIMENSIONS: ClassVar[str] = "vector_dimensions"
    _REGISTRY_INDEXED_PROPERTIES_SCHEMA: ClassVar[str] = "indexed_properties_schema"

    # Purge queue keys (stored on queue points, one per deleted incarnation; the
    # point id is the incarnation). The queue is one collection per store, so a
    # sweep serves every namespace with one scroll; its name cannot collide with
    # a namespace's `<namespace>__<sha256>` or `<namespace>__registry`.
    _PURGE_QUEUE_COLLECTION: ClassVar[str] = "memmachine__purge_queue"
    _QUEUE_NAMESPACE: ClassVar[str] = "namespace"
    _QUEUE_NAME: ClassVar[str] = "name"
    _QUEUE_NATIVE_COLLECTION: ClassVar[str] = "native_collection"
    _QUEUE_DELETED_AT: ClassVar[str] = "deleted_at"

    # Every collection this store creates accepts filters on unindexed payload
    # keys: a caller may filter on any property, declared or not, and Qdrant
    # Cloud's default strict mode would reject the undeclared ones.
    _STRICT_MODE: ClassVar[models.StrictModeConfig] = models.StrictModeConfig(
        enabled=False
    )

    # Fixed UUID namespace for deterministic registry point IDs.
    _REGISTRY_UUID_NAMESPACE: ClassVar[UUID] = UUID(
        "a3c1f6d2-4b8e-4f2a-9c7d-1e5f8a0b3d6c"
    )

    # Keyed by client so locks are garbage-collected when the client is.
    _name_locks: ClassVar[
        WeakKeyDictionary[
            AsyncQdrantClient,
            defaultdict[tuple[str, str], asyncio.Lock],
        ]
    ] = WeakKeyDictionary()

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        """Check if an exception indicates a resource already exists."""
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 409
        if isinstance(error, grpc.aio.AioRpcError):
            return error.code() == grpc.StatusCode.ALREADY_EXISTS
        if isinstance(error, ValueError):
            return "already exists" in str(error).lower()
        return False

    @staticmethod
    def _is_not_found_error(error: Exception) -> bool:
        """Check if an exception indicates a resource was not found."""
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 404
        if isinstance(error, grpc.aio.AioRpcError):
            return error.code() == grpc.StatusCode.NOT_FOUND
        if isinstance(error, ValueError):
            return "not found" in str(error).lower()
        return False

    @staticmethod
    def _registry_collection_name(namespace: str) -> str:
        """Return the registry collection name for a namespace."""
        return f"{namespace}{QdrantVectorStore._REGISTRY_SUFFIX}"

    @staticmethod
    def _registry_point_uuid(name: str) -> UUID:
        """Return a deterministic UUID for a logical collection name."""
        return uuid5(QdrantVectorStore._REGISTRY_UUID_NAMESPACE, name)

    @staticmethod
    def _build_native_collection_name(
        namespace: str, config: VectorStoreCollectionConfig
    ) -> str:
        """Build a deterministic native collection name from namespace and config."""
        digest = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
        return f"{namespace}__{digest}"

    def __init__(self, params: QdrantVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._client: AsyncQdrantClient = params.client

        self._registry_replication_factor = params.registry_replication_factor

        self._hnsw_m = 16

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_qdrant",
        )

        self._client_name_locks = QdrantVectorStore._name_locks.setdefault(
            self._client, defaultdict(asyncio.Lock)
        )

    @override
    async def startup(self) -> None:
        """Create the store's purge queue; the client's lifecycle is managed externally."""
        await self._ensure_purge_queue_collection()

    @override
    async def shutdown(self) -> None:
        """No-op; client lifecycle is managed externally."""

    async def _ensure_namespace_registry_collection(self, namespace: str) -> None:
        """Idempotently create the registry collection for a namespace."""
        registry_collection_name = QdrantVectorStore._registry_collection_name(
            namespace
        )
        try:
            await self._client.create_collection(
                collection_name=registry_collection_name,
                vectors_config=models.VectorParams(
                    size=1,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,
                ),
                replication_factor=self._registry_replication_factor,
                write_consistency_factor=self._registry_replication_factor,
                strict_mode_config=QdrantVectorStore._STRICT_MODE,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
            if not QdrantVectorStore._is_already_exists_error(e):
                raise

    async def _ensure_purge_queue_collection(self) -> None:
        """Idempotently create the purge queue and the index its sweep orders by."""
        try:
            await self._client.create_collection(
                collection_name=QdrantVectorStore._PURGE_QUEUE_COLLECTION,
                vectors_config=models.VectorParams(
                    size=1,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,
                ),
                replication_factor=self._registry_replication_factor,
                write_consistency_factor=self._registry_replication_factor,
                strict_mode_config=QdrantVectorStore._STRICT_MODE,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
            if not QdrantVectorStore._is_already_exists_error(e):
                raise
        try:
            await self._client.create_payload_index(
                collection_name=QdrantVectorStore._PURGE_QUEUE_COLLECTION,
                field_name=QdrantVectorStore._QUEUE_DELETED_AT,
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
            if not QdrantVectorStore._is_already_exists_error(e):
                raise

    async def _get_registry_entry(
        self, namespace: str, name: str
    ) -> dict[str, Any] | None:
        """
        Retrieve the registry entry for a logical collection name.

        Verifies the stored name matches
        to guard against SHA-1 collisions in the uuid5 point ID.
        """
        registry_collection_name = QdrantVectorStore._registry_collection_name(
            namespace
        )
        point_uuid = QdrantVectorStore._registry_point_uuid(name)
        try:
            points = await self._client.retrieve(
                collection_name=registry_collection_name,
                ids=[point_uuid],
                with_payload=True,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
            if QdrantVectorStore._is_not_found_error(e):
                return None
            raise

        if not points:
            return None

        payload = cast(dict[str, Any], points[0].payload)
        if payload.get(QdrantVectorStore._REGISTRY_NAME) != name:
            return None

        return payload

    @staticmethod
    def _parse_entry(entry: Mapping[str, Any]) -> VectorStoreCollectionConfig:
        """Parse a VectorStoreCollectionConfig from a registry entry."""
        return VectorStoreCollectionConfig(
            vector_dimensions=entry[QdrantVectorStore._REGISTRY_VECTOR_DIMENSIONS],
            indexed_properties_schema=entry[
                QdrantVectorStore._REGISTRY_INDEXED_PROPERTIES_SCHEMA
            ],
        )

    async def _is_live(self, namespace: str, name: str, incarnation: UUID) -> bool:
        """Whether the collection's live entry still names this incarnation."""
        entry = await self._get_registry_entry(namespace, name)
        return (
            entry is not None
            and entry.get(QdrantVectorStore._REGISTRY_INCARNATION) == incarnation.hex
        )

    def _build_collection_handle(
        self,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
        incarnation: UUID,
    ) -> QdrantVectorStoreCollection:
        """Build a handle bound to one incarnation of the collection."""
        return QdrantVectorStoreCollection(
            client=self._client,
            collection_name=QdrantVectorStore._build_native_collection_name(
                namespace, config
            ),
            namespace=namespace,
            name=name,
            incarnation=incarnation,
            config=config,
            tracker=self._tracker,
            is_live=self._is_live,
        )

    async def _create_native_collection(
        self, namespace: str, config: VectorStoreCollectionConfig
    ) -> None:
        """Idempotently create the native Qdrant collection and payload indexes."""
        native_collection_name = QdrantVectorStore._build_native_collection_name(
            namespace, config
        )
        distance = QdrantVectorStore._QDRANT_DISTANCE
        # The collection and its indexes are created under separate guards. Sharing
        # one meant a collection that already existed - a second worker, or a retry
        # after a crash between the two calls - raised on create_collection, took
        # the already-exists path, and left the collection with no payload indexes
        # at all. The lock above is keyed on the client object, so it serialises
        # callers within a process and not across uvicorn workers.
        try:
            await self._client.create_collection(
                collection_name=native_collection_name,
                vectors_config=models.VectorParams(
                    size=config.vector_dimensions, distance=distance
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,
                    payload_m=self._hnsw_m,
                ),
                strict_mode_config=QdrantVectorStore._STRICT_MODE,
            )
        except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
            if not QdrantVectorStore._is_already_exists_error(e):
                raise

        indexes: list[tuple[str, Any]] = [
            (
                _PAYLOAD_INCARNATION,
                models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
        ]
        for prop_name, prop_type in config.indexed_properties_schema.items():
            index_type = QdrantVectorStore._PROPERTY_TYPE_TO_INDEX_TYPE.get(prop_type)
            if index_type is not None:
                indexes.append((prop_name, index_type))

        for field_name, field_schema in indexes:
            try:
                await self._client.create_payload_index(
                    collection_name=native_collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except (UnexpectedResponse, grpc.aio.AioRpcError, ValueError) as e:
                if not QdrantVectorStore._is_already_exists_error(e):
                    raise

    async def _register_collection(
        self,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
        incarnation: UUID,
    ) -> None:
        """Write the logical collection's live entry to the registry."""
        registry_name = QdrantVectorStore._registry_collection_name(namespace)
        point_uuid = QdrantVectorStore._registry_point_uuid(name)
        await self._client.upsert(
            collection_name=registry_name,
            points=[
                models.PointStruct(
                    id=point_uuid,
                    vector=[0.0],
                    payload={
                        QdrantVectorStore._REGISTRY_NAME: name,
                        QdrantVectorStore._REGISTRY_INCARNATION: incarnation.hex,
                        QdrantVectorStore._REGISTRY_VECTOR_DIMENSIONS: config.vector_dimensions,
                        QdrantVectorStore._REGISTRY_INDEXED_PROPERTIES_SCHEMA: config.model_dump(
                            mode="json"
                        )["indexed_properties_schema"],
                    },
                ),
            ],
            wait=True,
        )

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        """Create a logical collection in the Qdrant vector store."""
        if not validate_identifier(namespace):
            raise ValueError(
                f"Namespace {namespace!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        if not validate_identifier(name):
            raise ValueError(
                f"Name {name!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        async with (
            self._client_name_locks[(namespace, name)],
            self._tracker("create_collection"),
        ):
            await self._ensure_namespace_registry_collection(namespace)
            if await self._get_registry_entry(namespace, name) is not None:
                raise VectorStoreCollectionAlreadyExistsError(namespace, name)
            await self._create_native_collection(namespace, config)
            await self._register_collection(namespace, name, config, uuid4())

    @override
    async def get_collection(
        self, *, namespace: str, name: str
    ) -> QdrantVectorStoreCollection | None:
        """Get a collection handle from the vector store."""
        if not validate_identifier(namespace):
            raise ValueError(
                f"Namespace {namespace!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        if not validate_identifier(name):
            raise ValueError(
                f"Name {name!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        entry = await self._get_registry_entry(namespace, name)
        if entry is None:
            return None
        return self._build_collection_handle(
            namespace,
            name,
            QdrantVectorStore._parse_entry(entry),
            UUID(entry[QdrantVectorStore._REGISTRY_INCARNATION]),
        )

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        """Delete a logical collection from the Qdrant vector store."""
        if not validate_identifier(namespace):
            raise ValueError(
                f"Namespace {namespace!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        if not validate_identifier(name):
            raise ValueError(
                f"Name {name!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        async with (
            self._client_name_locks[(namespace, name)],
            self._tracker("delete_collection"),
        ):
            entry = await self._get_registry_entry(namespace, name)
            if entry is None:
                return
            incarnation = UUID(entry[QdrantVectorStore._REGISTRY_INCARNATION])
            config = QdrantVectorStore._parse_entry(entry)
            native_collection_name = QdrantVectorStore._build_native_collection_name(
                namespace, config
            )

            # The live entry goes first, so the collection is unreachable
            # before anything else happens; the purge entry follows. The
            # delete names the incarnation it read, so a creation that raced
            # in from another process keeps its entry.
            await self._client.delete(
                collection_name=QdrantVectorStore._registry_collection_name(namespace),
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.HasIdCondition(
                                has_id=[QdrantVectorStore._registry_point_uuid(name)]
                            ),
                            models.FieldCondition(
                                key=QdrantVectorStore._REGISTRY_INCARNATION,
                                match=models.MatchValue(value=incarnation.hex),
                            ),
                        ]
                    ),
                ),
                wait=True,
            )
            await self._client.upsert(
                collection_name=QdrantVectorStore._PURGE_QUEUE_COLLECTION,
                points=[
                    models.PointStruct(
                        id=incarnation,
                        vector=[0.0],
                        payload={
                            QdrantVectorStore._QUEUE_NAMESPACE: namespace,
                            QdrantVectorStore._QUEUE_NAME: name,
                            QdrantVectorStore._QUEUE_NATIVE_COLLECTION: native_collection_name,
                            QdrantVectorStore._QUEUE_DELETED_AT: int(
                                datetime.now(UTC).timestamp() * 1_000_000
                            ),
                        },
                    ),
                ],
                wait=True,
            )

    @override
    async def purge_deleted_collections(self) -> bool:
        # One dead incarnation per call, oldest first: its points go by a
        # filter on the incarnation, one server-side operation, then the
        # queue entry. Concurrent purgers may claim the same entry; every
        # step is idempotent, so the loser does empty work.
        async with self._tracker("purge_deleted_collections"):
            entries, _ = await self._client.scroll(
                collection_name=QdrantVectorStore._PURGE_QUEUE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=QdrantVectorStore._QUEUE_DELETED_AT,
                            range=models.Range(gte=0),
                        )
                    ]
                ),
                order_by=models.OrderBy(
                    key=QdrantVectorStore._QUEUE_DELETED_AT,
                    direction=models.Direction.ASC,
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if not entries:
                return False
            [entry] = entries
            incarnation = UUID(str(entry.id))
            native_collection_name = cast(dict[str, Any], entry.payload)[
                QdrantVectorStore._QUEUE_NATIVE_COLLECTION
            ]

            # A physical collection an operator dropped has nothing left to
            # reclaim; its entries retire.
            if await self._client.collection_exists(native_collection_name):
                await self._client.delete(
                    collection_name=native_collection_name,
                    points_selector=models.FilterSelector(
                        filter=_incarnation_filter(incarnation),
                    ),
                    wait=True,
                )
            await self._client.delete(
                collection_name=QdrantVectorStore._PURGE_QUEUE_COLLECTION,
                points_selector=models.PointIdsList(points=[incarnation]),
                wait=True,
            )
            return True
