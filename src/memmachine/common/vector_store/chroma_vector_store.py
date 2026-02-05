"""Chroma DB implementation of the vector store interface."""

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

if TYPE_CHECKING:
    from chromadb.api.types import Include

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection as ChromaDBCollection
from chromadb.config import Settings

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    Or,
)

from .data_types import PropertyValue, QueryResult, Record
from .vector_store import Collection, VectorStore


def _similarity_metric_to_chroma(metric: SimilarityMetric) -> str:
    """Convert SimilarityMetric to Chroma distance function name."""
    mapping = {
        SimilarityMetric.COSINE: "cosine",
        SimilarityMetric.DOT: "ip",  # inner product
        SimilarityMetric.EUCLIDEAN: "l2",
    }
    if metric not in mapping:
        raise ValueError(
            f"Chroma does not support {metric.value} distance. "
            f"Supported: cosine, dot, euclidean"
        )
    return mapping[metric]


def _convert_filter_to_chroma(expr: FilterExpr) -> dict[str, Any]:
    """Convert a FilterExpr tree to Chroma's Where format."""
    if isinstance(expr, Comparison):
        return _convert_comparison_to_chroma(expr)
    if isinstance(expr, And):
        left = _convert_filter_to_chroma(expr.left)
        right = _convert_filter_to_chroma(expr.right)
        return {"$and": [left, right]}
    if isinstance(expr, Or):
        left = _convert_filter_to_chroma(expr.left)
        right = _convert_filter_to_chroma(expr.right)
        return {"$or": [left, right]}
    raise TypeError(f"Unsupported filter expression type: {type(expr)}")


def _convert_comparison_to_chroma(comp: Comparison) -> dict[str, Any]:
    """Convert a Comparison to Chroma's Where format."""
    field = comp.field
    op = comp.op
    value = comp.value

    # Convert datetime values to ISO format strings for Chroma
    if isinstance(value, datetime):
        value = value.isoformat()
    elif isinstance(value, list):
        value = [v.isoformat() if isinstance(v, datetime) else v for v in value]

    op_mapping = {
        "=": "$eq",
        ">": "$gt",
        "<": "$lt",
        ">=": "$gte",
        "<=": "$lte",
        "in": "$in",
    }

    if op == "is_null":
        # Chroma doesn't have direct null check; use $eq with None
        return {field: {"$eq": None}}
    if op == "is_not_null":
        return {field: {"$ne": None}}
    if op in op_mapping:
        chroma_op = op_mapping[op]
        if op == "=":
            # Simple equality can use shorthand
            return {field: value}
        return {field: {chroma_op: value}}
    raise ValueError(f"Unsupported comparison operator: {op}")


def _distance_to_similarity(distance: float, metric: str) -> float:
    """Convert Chroma distance to similarity score."""
    if metric == "cosine":
        # Chroma cosine distance is 1 - cosine_similarity
        return 1.0 - distance
    if metric == "ip":
        # Inner product: negative distance (Chroma uses -ip as distance)
        return -distance
    if metric == "l2":
        # L2 distance: convert to similarity (smaller distance = higher similarity)
        # Use 1 / (1 + distance) to get a value in (0, 1]
        return 1.0 / (1.0 + distance)
    return 1.0 - distance


def _serialize_properties(
    properties: dict[str, PropertyValue] | None,
) -> dict[str, str | int | float | None] | None:
    """Serialize properties for storage in Chroma."""
    if properties is None:
        return None
    result: dict[str, str | int | float | None] = {}
    for key, value in properties.items():
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, bool):
            # Chroma doesn't support booleans directly, store as int
            result[key] = int(value)
        else:
            result[key] = value
    return result


def _deserialize_properties(
    metadata: dict[str, Any] | None,
) -> dict[str, PropertyValue] | None:
    """Deserialize properties from Chroma metadata."""
    if metadata is None:
        return None
    # Filter out internal metadata keys if any
    return {k: v for k, v in metadata.items() if not k.startswith("_")}


class ChromaCollection(Collection):
    """Chroma DB implementation of Collection."""

    def __init__(
        self,
        store: "ChromaVectorStore",
        collection_name: str,
        distance_fn: str,
    ) -> None:
        """Initialize ChromaCollection."""
        self._store = store
        self._collection_name = collection_name
        self._distance_fn = distance_fn

    def _get_collection(self) -> ChromaDBCollection:
        """Get the underlying Chroma collection."""
        return self._store.get_client().get_collection(
            name=self._collection_name,
            embedding_function=None,
        )

    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """Upsert records in the collection."""
        records_list = list(records)
        if not records_list:
            return

        ids: list[str] = []
        embeddings: list[Sequence[float]] = []
        metadatas: list[dict[str, str | int | float | None]] = []

        for record in records_list:
            ids.append(str(record.uuid))
            if record.vector is not None:
                embeddings.append(record.vector)
            props = _serialize_properties(record.properties)
            if props is not None:
                metadatas.append(props)

        collection = self._get_collection()
        collection.upsert(
            ids=ids,
            embeddings=embeddings if embeddings else None,
            metadatas=cast(Any, metadatas) if metadatas else None,
        )

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
        """Query for records matching the criteria by vector similarity."""
        collection = self._get_collection()

        # Build query parameters
        where: dict[str, Any] | None = None
        if property_filter is not None:
            where = _convert_filter_to_chroma(property_filter)

        # Chroma requires n_results; default to a reasonable number if not specified
        n_results = limit if limit is not None else 10

        # Build include list with proper literal types
        include: Include = ["distances"]
        if return_vector:
            include = ["embeddings", *include]
        if return_properties:
            include = ["metadatas", *include]

        results = collection.query(
            query_embeddings=[list(query_vector)],
            n_results=n_results,
            where=where,
            include=include,
        )

        # Process results
        query_results: list[QueryResult] = []

        result_ids = results.get("ids") or [[]]
        ids = result_ids[0]
        result_distances = results.get("distances") or [[]]
        distances = result_distances[0] if result_distances else []
        result_embeddings = results.get("embeddings") or [[]]
        embeddings_list = result_embeddings[0] if return_vector and result_embeddings else []
        result_metadatas = results.get("metadatas") or [[]]
        metadatas_list = result_metadatas[0] if return_properties and result_metadatas else []

        for i, record_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0.0
            similarity = _distance_to_similarity(distance, self._distance_fn)

            # Apply similarity threshold
            if similarity_threshold is not None and similarity < similarity_threshold:
                continue

            vector = list(embeddings_list[i]) if embeddings_list and i < len(embeddings_list) else None
            metadata = metadatas_list[i] if metadatas_list and i < len(metadatas_list) else None
            properties = _deserialize_properties(metadata)

            record = Record(
                uuid=UUID(record_id),
                vector=vector if return_vector else None,
                properties=properties if return_properties else None,
            )
            query_results.append(QueryResult(score=similarity, record=record))

        return query_results

    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        """Get records from the collection by their UUIDs."""
        uuid_list = list(record_uuids)
        if not uuid_list:
            return []

        collection = self._get_collection()

        # Build include list with proper literal types
        include: Include = []
        if return_vector:
            include = ["embeddings", *include]
        if return_properties:
            include = ["metadatas", *include]

        ids = [str(uuid) for uuid in uuid_list]
        results = collection.get(
            ids=ids,
            include=include if include else ["metadatas"],
        )

        # Build a map for ordering
        id_to_index = {id_str: i for i, id_str in enumerate(ids)}

        records: list[Record | None] = [None] * len(uuid_list)

        result_ids = results.get("ids") or []
        result_embeddings = results.get("embeddings") or []
        embeddings_list = result_embeddings if return_vector else []
        result_metadatas = results.get("metadatas") or []
        metadatas_list = result_metadatas if return_properties else []

        for i, record_id in enumerate(result_ids):
            vector = list(embeddings_list[i]) if embeddings_list and i < len(embeddings_list) else None
            metadata = metadatas_list[i] if metadatas_list and i < len(metadatas_list) else None
            properties = _deserialize_properties(metadata)

            record = Record(
                uuid=UUID(record_id),
                vector=vector if return_vector else None,
                properties=properties if return_properties else None,
            )

            # Place in correct position to maintain order
            original_idx = id_to_index.get(record_id)
            if original_idx is not None:
                records[original_idx] = record

        # Filter out None values (records that weren't found)
        return [r for r in records if r is not None]

    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """Delete records from the collection by their UUIDs."""
        uuid_list = list(record_uuids)
        if not uuid_list:
            return

        collection = self._get_collection()
        ids = [str(uuid) for uuid in uuid_list]
        collection.delete(ids=ids)


class ChromaVectorStore(VectorStore):
    """Chroma DB implementation of VectorStore."""

    def __init__(
        self,
        *,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """
        Initialize ChromaVectorStore.

        Args:
            persist_directory: Path for persistent storage. If None, uses in-memory storage.
            host: Chroma server host for client mode.
            port: Chroma server port for client mode.

        """
        self._persist_directory = persist_directory
        self._host = host
        self._port = port
        self._client: ClientAPI | None = None
        self._collection_metadata: dict[str, dict[str, Any]] = {}

    def get_client(self) -> ClientAPI:
        """Get the Chroma client. Raises RuntimeError if not started."""
        if self._client is None:
            raise RuntimeError("ChromaVectorStore not started. Call startup() first.")
        return self._client

    async def startup(self) -> None:
        """Initialize the Chroma client."""
        if self._host and self._port:
            # Connect to remote Chroma server
            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port,
            )
        elif self._persist_directory:
            # Use persistent storage
            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            # Use in-memory storage
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

    async def shutdown(self) -> None:
        """Shutdown the Chroma client."""
        # Chroma doesn't require explicit cleanup
        self._client = None

    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        """Create a collection in the vector store."""
        if self._client is None:
            raise RuntimeError("ChromaVectorStore not started. Call startup() first.")

        distance_fn = _similarity_metric_to_chroma(similarity_metric)

        # Store metadata about the collection
        self._collection_metadata[collection_name] = {
            "vector_dimensions": vector_dimensions,
            "distance_fn": distance_fn,
            "properties_schema": properties_schema,
        }

        self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_fn},
            embedding_function=None,
        )

    async def get_collection(self, collection_name: str) -> Collection:
        """Get a collection from the vector store."""
        if self._client is None:
            raise RuntimeError("ChromaVectorStore not started. Call startup() first.")

        # Get distance function from stored metadata, defaulting to cosine
        metadata = self._collection_metadata.get(collection_name, {})
        distance_fn = metadata.get("distance_fn", "cosine")

        return ChromaCollection(
            store=self,
            collection_name=collection_name,
            distance_fn=distance_fn,
        )

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector store."""
        if self._client is None:
            raise RuntimeError("ChromaVectorStore not started. Call startup() first.")

        self._client.delete_collection(name=collection_name)

        # Clean up metadata
        self._collection_metadata.pop(collection_name, None)
