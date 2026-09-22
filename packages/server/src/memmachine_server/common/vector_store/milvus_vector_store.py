"""Milvus-based vector store implementation."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf
from pymilvus import DataType, MilvusClient
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
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)
from memmachine_server.common.utils import compute_similarity, ensure_tz_aware

from .collection_registry import RegisteredCollection, VectorStoreCollectionRegistry
from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    VectorStoreCollectionHandleStaleError,
)
from .utils import require_identifiers, validate_filter
from .vector_store import VectorStore, VectorStoreCollection

_ID_FIELD = "id"
_RECORD_UUID_FIELD = "record_uuid"
_PARTITION_KEY_FIELD = "partition_key"
"""The native partition-key field; holds the incarnation, never the collection's name.

A collection deleted and re-created under the same name gets a fresh
incarnation, and its predecessor's entities are invisible to it while the
purge reclaims them.
"""
_VECTOR_FIELD = "vector"
_PROPERTIES_FIELD = "properties"
_PROPERTY_FILTER_PREFIX = "_p_"

_MAX_UUID_LENGTH = 36
_MAX_PRIMARY_ID_LENGTH = 128
_INCARNATION_HEX_LENGTH = 32
_FALSE_EXPR = f'{_ID_FIELD} == "__memmachine_no_match__"'

# Consecutive lost creation races before open-or-create gives up: every
# retry requires another process to have created and then deleted the
# collection in between, so this depth means something else is wrong.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10


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


def _property_field(field: str) -> str:
    """Return the dynamic Milvus field used for property filtering."""
    return f"{_PROPERTY_FILTER_PREFIX}{field}"


def _normalize_property_filter_value(value: PropertyValue) -> PropertyValue:
    """Normalize property values stored in dynamic filter fields."""
    if isinstance(value, datetime):
        return ensure_tz_aware(value).astimezone(UTC).isoformat()
    return value


def _incarnation_filter(incarnation: UUID) -> str:
    """A Milvus expression matching the entities of one collection incarnation."""
    return f"{_PARTITION_KEY_FIELD} == {_expr_string(incarnation.hex)}"


class MilvusVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by Milvus."""

    _RANGE_OPERATORS: ClassVar[set[str]] = {">", ">=", "<", "<="}

    @staticmethod
    def _build_milvus_filter(expr: FilterExpr) -> str:
        """Convert a FilterExpr tree into a Milvus filter expression."""
        if isinstance(expr, FilterComparison):
            return MilvusVectorStoreCollection._build_milvus_comparison(expr)
        if isinstance(expr, FilterIn):
            if not expr.values:
                return _FALSE_EXPR
            values = ", ".join(_literal(value) for value in expr.values)
            return f"{_property_field(expr.field)} in [{values}]"
        if isinstance(expr, FilterIsNull):
            return f"{_property_field(expr.field)} is null"
        if isinstance(expr, FilterNot):
            return (
                f"not ({MilvusVectorStoreCollection._build_milvus_filter(expr.expr)})"
            )
        if isinstance(expr, FilterAnd):
            left = MilvusVectorStoreCollection._build_milvus_filter(expr.left)
            right = MilvusVectorStoreCollection._build_milvus_filter(expr.right)
            return f"({left}) && ({right})"
        if isinstance(expr, FilterOr):
            left = MilvusVectorStoreCollection._build_milvus_filter(expr.left)
            right = MilvusVectorStoreCollection._build_milvus_filter(expr.right)
            return f"({left}) || ({right})"
        message = f"Unsupported filter expression type: {type(expr)}"
        raise TypeError(message)

    @staticmethod
    def _build_milvus_comparison(comparison: FilterComparison) -> str:
        """Convert a Comparison into a Milvus filter expression."""
        field = _property_field(comparison.field)
        operator = "==" if comparison.op == "=" else comparison.op
        return f"{field} {operator} {_literal(comparison.value)}"

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

    @staticmethod
    def _primary_id(incarnation: UUID, record_uuid: UUID) -> str:
        """Build a native primary key unique within a shared native collection."""
        return f"{incarnation.hex}:{record_uuid}"

    def __init__(
        self,
        *,
        client: MilvusClient,
        native_collection_name: str,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
        tracker: OperationTracker,
        is_live: Callable[[UUID], Awaitable[bool]],
    ) -> None:
        """Initialize with a Milvus client and the incarnation the handle is bound to."""
        self._client = client
        self._native_collection_name = native_collection_name
        self._namespace = namespace
        self._name = name
        self._incarnation = incarnation
        self._config = config
        self._tracker = tracker
        self._is_live = is_live

    async def _fence(self) -> None:
        """Raise if this handle's incarnation is no longer the collection's.

        Called before every operation, to refuse a handle known to be
        dead, and after a write, so a write completed under an incarnation
        that died meanwhile raises instead of reporting success. Milvus has
        no transactions, so a write can still land under a dead
        incarnation: between the two checks, or after a check that never
        ran; the tombstone's purge rounds reclaim it. A read is not checked
        after: a collection deleted while a read is in flight keeps its
        entities until a purge round claims its tombstone, so the read returns
        what it saw, a snapshot from before the deletion, as a read that
        happened to run just before it would have.
        """
        if not await self._is_live(self._incarnation):
            raise VectorStoreCollectionHandleStaleError(self._namespace, self._name)

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        """The configuration for this collection."""
        return self._config

    def _build_entity(self, record: Record) -> dict[str, Any]:
        """Build a Milvus entity from a vector store record."""
        if record.vector is None:
            raise ValueError(
                f"Record {record.uuid} has vector=None, which is not allowed on input."
            )

        properties = record.properties if record.properties is not None else {}
        entity: dict[str, Any] = {
            _ID_FIELD: self._primary_id(self._incarnation, record.uuid),
            _RECORD_UUID_FIELD: str(record.uuid),
            _PARTITION_KEY_FIELD: self._incarnation.hex,
            _VECTOR_FIELD: record.vector,
            _PROPERTIES_FIELD: encode_properties(properties),
        }
        # Explicit nulls clear stale dynamic fields during native Milvus upserts.
        for key in self._config.indexed_properties_schema:
            entity[_property_field(key)] = None
        for key, value in properties.items():
            entity[_property_field(key)] = _normalize_property_filter_value(value)
        return entity

    @staticmethod
    def _parse_record(
        entity: Mapping[str, Any],
        *,
        return_vector: bool,
        return_properties: bool,
    ) -> Record:
        """Parse a Milvus entity into a vector store record."""
        vector: list[float] | None = None
        if return_vector:
            raw_vector = entity.get(_VECTOR_FIELD)
            if raw_vector is not None:
                vector = list(cast(Sequence[float], raw_vector))

        properties: dict[str, PropertyValue] | None = None
        if return_properties:
            properties = decode_properties(
                cast(Mapping | None, entity.get(_PROPERTIES_FIELD))
            )

        return Record(
            uuid=UUID(str(entity[_RECORD_UUID_FIELD])),
            vector=vector,
            properties=properties,
        )

    def _output_fields(
        self, *, return_vector: bool, return_properties: bool
    ) -> list[str]:
        fields = [_RECORD_UUID_FIELD]
        if return_vector:
            fields.append(_VECTOR_FIELD)
        if return_properties:
            fields.append(_PROPERTIES_FIELD)
        return fields

    @staticmethod
    def _score_from_entity_vector(
        query_vector: Sequence[float],
        entity: Mapping[str, Any],
        similarity_metric: SimilarityMetric,
    ) -> float:
        raw_vector = entity.get(_VECTOR_FIELD)
        if raw_vector is None:
            raise ValueError("Milvus search result did not include the vector field")
        return compute_similarity(
            list(query_vector),
            [list(cast(Sequence[float], raw_vector))],
            similarity_metric,
        )[0]

    def _partition_filter(self) -> str:
        return _incarnation_filter(self._incarnation)

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records into the collection."""
        async with self._tracker("upsert"):
            records = list(records)
            if not records:
                return

            await self._fence()
            entities = [self._build_entity(record) for record in records]

            def _upsert() -> None:
                self._client.upsert(
                    collection_name=self._native_collection_name,
                    data=entities,
                )

            await asyncio.to_thread(_upsert)
            await self._fence()

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[QueryResult]:
        """Query for records matching the criteria by query vectors."""
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            if not query_vectors:
                return []
            if limit <= 0:
                return [QueryResult(matches=[]) for _ in query_vectors]

            await self._fence()
            filter_expr = self._partition_filter()
            if property_filter is not None:
                if not validate_filter(property_filter):
                    raise ValueError("Filter contains an invalid property key")
                property_expr = self._build_milvus_filter(property_filter)
                filter_expr = f"({filter_expr}) && ({property_expr})"

            raw_results = await asyncio.to_thread(
                self._client.search,
                collection_name=self._native_collection_name,
                data=query_vectors,
                filter=filter_expr,
                limit=limit,
                # Milvus Lite returns COSINE as distance, while Zilliz Cloud
                # returns it as similarity. Fetch vectors and compute scores
                # locally so MemMachine score semantics stay consistent.
                output_fields=self._output_fields(
                    return_vector=True,
                    return_properties=return_properties,
                ),
                anns_field=_VECTOR_FIELD,
            )

            results: list[QueryResult] = []
            for query_vector, raw_matches in zip(
                query_vectors, raw_results, strict=True
            ):
                matches: list[QueryMatch] = []
                for raw_match in raw_matches:
                    entity = cast(Mapping[str, Any], raw_match["entity"])
                    score = self._score_from_entity_vector(
                        query_vector,
                        entity,
                        self._config.similarity_metric,
                    )
                    if not self._passes_threshold(
                        score, score_threshold, self._config.similarity_metric
                    ):
                        continue

                    matches.append(
                        QueryMatch(
                            score=score,
                            record=self._parse_record(
                                entity,
                                return_vector=return_vector,
                                return_properties=return_properties,
                            ),
                        )
                    )

                matches.sort(
                    key=lambda match: match.score,
                    reverse=self._config.similarity_metric.higher_is_better,
                )
                results.append(QueryResult(matches=matches))

            return results

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

            await self._fence()
            primary_ids = [
                self._primary_id(self._incarnation, uuid) for uuid in uuid_list
            ]
            raw_records = await asyncio.to_thread(
                self._client.get,
                collection_name=self._native_collection_name,
                ids=primary_ids,
                output_fields=self._output_fields(
                    return_vector=return_vector,
                    return_properties=return_properties,
                ),
            )

            records_by_uuid = {
                record.uuid: record
                for record in (
                    self._parse_record(
                        cast(Mapping[str, Any], raw_record),
                        return_vector=return_vector,
                        return_properties=return_properties,
                    )
                    for raw_record in raw_records
                )
            }
            records = [
                records_by_uuid[record_uuid]
                for record_uuid in uuid_list
                if record_uuid in records_by_uuid
            ]
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
            await self._fence()
            primary_ids = [
                self._primary_id(self._incarnation, uuid) for uuid in uuid_list
            ]
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self._native_collection_name,
                ids=primary_ids,
            )
            await self._fence()


class MilvusVectorStoreParams(BaseModel):
    """
    Parameters for MilvusVectorStore.

    Attributes:
        client (MilvusClient): Milvus client instance.
        collection_registry (VectorStoreCollectionRegistry):
            The registry of the Milvus deployment the client reaches: which
            collections exist, under which incarnation and configuration,
            and which dead incarnations await purge. Milvus arbitrates none
            of that, so the registry lives where a primary key and a
            transaction can. Every store on the deployment, in any
            process, uses this registry, and no store on another
            deployment does.
        consistency_level (str): Collection consistency level for newly created collections.
        metrics_factory (MetricsFactory | None): Metrics factory for collecting usage metrics.
    """

    client: InstanceOf[MilvusClient] = Field(
        ...,
        description="Milvus client instance",
    )
    collection_registry: InstanceOf[VectorStoreCollectionRegistry] = Field(
        ...,
        description="The registry of the deployment the client reaches",
    )
    consistency_level: str = Field(
        default="Session",
        description="Milvus consistency level for newly created collections",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class MilvusVectorStore(VectorStore):
    """Asynchronous Milvus-based implementation of VectorStore.

    A logical collection is a partition-key value, the incarnation of its
    life, inside a native collection shared by the logical collections of
    one namespace and configuration. The catalog is the `VectorStoreCollectionRegistry`
    the store is given: it mints the incarnations and arbitrates creation,
    deletion and reclamation across processes, which Milvus, with no
    transactions or unique constraints, cannot. Any process sharing the
    Milvus backend and the registry may serve any collection.
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
        super().__init__()
        self._client = params.client
        self._consistency_level = params.consistency_level
        self._collection_registry = params.collection_registry
        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="vector_store_milvus",
        )

    @override
    async def startup(self) -> None:
        """Ready the registry; the client's lifecycle is managed externally."""
        await self._collection_registry.startup()

    @override
    async def shutdown(self) -> None:
        """No-op; client lifecycle is managed externally."""

    def _build_collection_handle(
        self, namespace: str, name: str, registered: RegisteredCollection
    ) -> MilvusVectorStoreCollection:
        """Build a MilvusVectorStoreCollection handle bound to the registered incarnation."""
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
        )

    async def _create_native_collection(
        self, namespace: str, config: VectorStoreCollectionConfig
    ) -> None:
        """Idempotently create the native Milvus collection."""
        self._validate_metric(config.similarity_metric)
        native_collection_name = MilvusVectorStore._build_native_collection_name(
            namespace, config
        )
        if await asyncio.to_thread(self._client.has_collection, native_collection_name):
            return

        def _create_collection() -> None:
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(
                field_name=_ID_FIELD,
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=_MAX_PRIMARY_ID_LENGTH,
            )
            schema.add_field(
                field_name=_RECORD_UUID_FIELD,
                datatype=DataType.VARCHAR,
                max_length=_MAX_UUID_LENGTH,
            )
            schema.add_field(
                field_name=_PARTITION_KEY_FIELD,
                datatype=DataType.VARCHAR,
                max_length=_INCARNATION_HEX_LENGTH,
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

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name=_VECTOR_FIELD,
                index_type="AUTOINDEX",
                metric_type=self._SIMILARITY_METRIC_TO_MILVUS_METRIC[
                    config.similarity_metric
                ],
            )

            self._client.create_collection(
                collection_name=native_collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self._consistency_level,
            )

        try:
            await asyncio.to_thread(_create_collection)
        except MilvusException as exc:
            if not MilvusVectorStore._is_already_exists_error(exc):
                raise

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        """Create a logical collection in the Milvus vector store."""
        require_identifiers(namespace, name)
        self._validate_metric(config.similarity_metric)
        async with self._tracker("create_collection"):
            # The native collection first, the registry row last: a crash
            # between the two leaves an empty native collection the next
            # creation of the same configuration adopts, never a row whose
            # entities have nowhere to go. The registry's primary key is the
            # arbiter: a racing creator on any process loses here, never in
            # Milvus.
            await self._create_native_collection(namespace, config)
            await self._collection_registry.create(namespace, name, config)

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> MilvusVectorStoreCollection:
        """Open the collection if it exists, or create and return it."""
        require_identifiers(namespace, name)
        self._validate_metric(config.similarity_metric)
        async with self._tracker("open_or_create_collection"):
            attempts = 0
            # Read-then-create, retried: losing the create means a racing
            # creator won (open its row), and finding no row after losing
            # means a racing deleter removed the winner (create again).
            while True:
                registered = await self._collection_registry.get(namespace, name)
                if registered is not None:
                    if registered.config != config:
                        raise VectorStoreCollectionConfigMismatchError(
                            namespace, name, registered.config, config
                        )
                    return self._build_collection_handle(namespace, name, registered)
                await self._create_native_collection(namespace, config)
                try:
                    incarnation = await self._collection_registry.create(
                        namespace, name, config
                    )
                except VectorStoreCollectionAlreadyExistsError as err:
                    attempts += 1
                    if attempts >= _MAX_OPEN_OR_CREATE_ATTEMPTS:
                        raise VectorStoreAttemptsExhaustedError(
                            f"Opening or creating collection ({namespace!r}, "
                            f"{name!r}) made no progress after "
                            f"{_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts"
                        ) from err
                    continue
                return self._build_collection_handle(
                    namespace,
                    name,
                    RegisteredCollection(incarnation=incarnation, config=config),
                )

    @override
    async def open_collection(
        self, *, namespace: str, name: str
    ) -> MilvusVectorStoreCollection | None:
        """Get a collection handle from the vector store."""
        require_identifiers(namespace, name)
        registered = await self._collection_registry.get(namespace, name)
        if registered is None:
            return None
        return self._build_collection_handle(namespace, name, registered)

    @override
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        """No-op; Milvus collection handles require no explicit close."""

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        """Delete a logical collection from the Milvus vector store."""
        require_identifiers(namespace, name)
        async with self._tracker("delete_collection"):
            # One registry transaction: the collection is unreachable when
            # it commits, and its entities wait on the queue for the purge.
            await self._collection_registry.delete(namespace, name)

    @override
    async def purge_deleted_collections(self) -> bool:
        # One purge round per call, on the tombstone that came due first: the
        # claim is a row lock the registry holds while one entity is looked
        # for and, if there is one, the entities go by filter in a single
        # server-side operation. The registry keeps or removes the tombstone
        # by what the round found.
        async with (
            self._tracker("purge_deleted_collections"),
            self._collection_registry.claim_due() as claim,
        ):
            if claim is None:
                return False
            native_collection_name = MilvusVectorStore._build_native_collection_name(
                claim.namespace, claim.config
            )
            incarnation_filter = _incarnation_filter(claim.incarnation)
            if await asyncio.to_thread(
                self._client.has_collection, native_collection_name
            ):
                held = await asyncio.to_thread(
                    self._client.query,
                    collection_name=native_collection_name,
                    filter=incarnation_filter,
                    output_fields=[_ID_FIELD],
                    limit=1,
                )
                claim.found = bool(list(held))
            else:
                # The native collection is gone with everything in it.
                claim.found = False
            if claim.found:
                await asyncio.to_thread(
                    self._client.delete,
                    collection_name=native_collection_name,
                    filter=incarnation_filter,
                )
            return claim.found
