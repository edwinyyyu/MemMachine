"""Qdrant-based vector store implementation."""

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf
from qdrant_client import AsyncQdrantClient, models

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    Or,
)

from .data_types import PropertyValue, QueryResult, Record
from .vector_store import Collection, VectorStore

QDRANT_SIMILARITY_METRIC_MAP = {
    SimilarityMetric.COSINE: models.Distance.COSINE,
    SimilarityMetric.DOT: models.Distance.DOT,
    SimilarityMetric.EUCLIDEAN: models.Distance.EUCLID,
    SimilarityMetric.MANHATTAN: models.Distance.MANHATTAN,
}

QDRANT_TYPE_MAP = {
    bool: models.PayloadSchemaType.BOOL,
    int: models.PayloadSchemaType.INTEGER,
    float: models.PayloadSchemaType.FLOAT,
    str: models.PayloadSchemaType.KEYWORD,
    datetime: models.PayloadSchemaType.DATETIME,
}


def _extract_uuid(point_id: int | str | UUID) -> UUID:
    """Extract UUID from Qdrant point ID."""
    if isinstance(point_id, UUID):
        return point_id
    if isinstance(point_id, str):
        return UUID(point_id)
    raise TypeError(f"Expected UUID or string point ID, got int: {point_id}")


def _extract_vector(vector: object) -> list[float] | None:
    """Extract single dense vector from Qdrant vector response."""
    if vector is None:
        return None
    if isinstance(vector, dict):
        raise TypeError("Named vectors not supported, expected single dense vector")
    if not isinstance(vector, list):
        raise TypeError(f"Expected list vector, got {type(vector)}")
    if vector and isinstance(vector[0], list):
        raise TypeError("Multi-vectors not supported, expected single dense vector")
    result: list[float] = []
    for v in vector:
        if not isinstance(v, (int, float)):
            raise TypeError(f"Expected numeric vector element, got {type(v)}")
        result.append(float(v))
    return result


def _make_range_condition(
    field: str,
    value: float | datetime,
    *,
    gt: bool = False,
    lt: bool = False,
    gte: bool = False,
    lte: bool = False,
) -> models.FieldCondition:
    """Create a range condition for numeric or datetime values."""
    if isinstance(value, datetime):
        return models.FieldCondition(
            key=field,
            range=models.DatetimeRange(
                gt=value if gt else None,
                lt=value if lt else None,
                gte=value if gte else None,
                lte=value if lte else None,
            ),
        )
    return models.FieldCondition(
        key=field,
        range=models.Range(
            gt=value if gt else None,
            lt=value if lt else None,
            gte=value if gte else None,
            lte=value if lte else None,
        ),
    )


def _make_equality_condition(field: str, value: object) -> models.FieldCondition:
    """Create an equality condition for the given field and value."""
    if isinstance(value, (float, datetime)):
        return _make_range_condition(field, value, gte=True, lte=True)
    if isinstance(value, (int, str)):
        return models.FieldCondition(key=field, match=models.MatchValue(value=value))
    raise TypeError(f"Unsupported value type for '=' operator: {type(value)}")


def _make_in_condition(field: str, value: object) -> models.FieldCondition:
    """Create an 'in' condition for the given field and list of values."""
    if not isinstance(value, list):
        raise TypeError(f"'in' operator requires a list, got {type(value)}")
    if not value:
        raise ValueError("'in' operator requires a non-empty list")
    int_values: list[int] = [v for v in value if isinstance(v, int)]
    if len(int_values) == len(value):
        return models.FieldCondition(key=field, match=models.MatchAny(any=int_values))
    str_values: list[str] = [v for v in value if isinstance(v, str)]
    if len(str_values) == len(value):
        return models.FieldCondition(key=field, match=models.MatchAny(any=str_values))
    raise TypeError("'in' operator requires homogeneous list of int or str")


def _convert_comparison_to_qdrant(comp: Comparison) -> models.Filter:
    """Convert a Comparison to a Qdrant Filter."""
    field = comp.field
    op = comp.op
    value = comp.value

    condition: models.FieldCondition | models.IsNullCondition
    if op == "=":
        condition = _make_equality_condition(field, value)
    elif op == "in":
        condition = _make_in_condition(field, value)
    elif op in (">", "<", ">=", "<="):
        if not isinstance(value, (int, float, datetime)):
            raise TypeError(f"Range operator '{op}' requires numeric or datetime value")
        range_ops = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte"}
        condition = _make_range_condition(field, value, **{range_ops[op]: True})
    elif op == "is_null":
        condition = models.IsNullCondition(is_null=models.PayloadField(key=field))
    elif op == "is_not_null":
        condition = models.IsNullCondition(is_null=models.PayloadField(key=field))
        return models.Filter(must_not=[condition])
    else:
        raise ValueError(f"Unsupported comparison operator: {op}")

    return models.Filter(must=[condition])


def _convert_filter_to_qdrant(expr: FilterExpr) -> models.Filter:
    """Convert a FilterExpr to a Qdrant Filter."""
    if isinstance(expr, Comparison):
        return _convert_comparison_to_qdrant(expr)

    if isinstance(expr, And):
        left_filter = _convert_filter_to_qdrant(expr.left)
        right_filter = _convert_filter_to_qdrant(expr.right)
        return models.Filter(must=[left_filter, right_filter])

    if isinstance(expr, Or):
        left_filter = _convert_filter_to_qdrant(expr.left)
        right_filter = _convert_filter_to_qdrant(expr.right)
        return models.Filter(should=[left_filter, right_filter])

    raise TypeError(f"Unsupported filter expression type: {type(expr)}")


class QdrantVectorStoreParams(BaseModel):
    """
    Parameters for QdrantVectorStore.

    Attributes:
        client (AsyncQdrantClient):
            Async Qdrant client instance.

    """

    client: InstanceOf[AsyncQdrantClient] = Field(
        ...,
        description="Async Qdrant client instance",
    )


class QdrantCollection(Collection):
    """Qdrant-based implementation of Collection."""

    def __init__(self, client: AsyncQdrantClient, name: str) -> None:
        """Initialize QdrantCollection."""
        self._client = client
        self._name = name

    @override
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        points: list[models.PointStruct] = []
        for record in records:
            if record.vector is None:
                raise ValueError(f"Record {record.uuid} has no vector")
            points.append(
                models.PointStruct(
                    id=record.uuid,
                    payload=record.properties,
                    vector=record.vector,
                )
            )
        await self._client.upsert(collection_name=self._name, points=points)

    @override
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
        query_response = await self._client.query_points(
            collection_name=self._name,
            query=list(query_vector),
            score_threshold=similarity_threshold,
            limit=limit or 1000,
            query_filter=_convert_filter_to_qdrant(property_filter) if property_filter else None,
            with_vectors=return_vector,
            with_payload=return_properties,
        )
        results: list[QueryResult] = []
        for point in query_response.points:
            point_uuid = _extract_uuid(point.id)
            point_vector = _extract_vector(point.vector) if return_vector else None
            results.append(
                QueryResult(
                    score=point.score,
                    record=Record(
                        uuid=point_uuid,
                        vector=point_vector,
                        properties=point.payload,
                    ),
                )
            )
        return results

    @override
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        qdrant_records = await self._client.retrieve(
            collection_name=self._name,
            ids=list(record_uuids),
            with_vectors=return_vector,
            with_payload=return_properties,
        )
        results: list[Record] = []
        for record in qdrant_records:
            record_uuid = _extract_uuid(record.id)
            record_vector = _extract_vector(record.vector) if return_vector else None
            results.append(
                Record(
                    uuid=record_uuid,
                    vector=record_vector,
                    properties=record.payload,
                )
            )
        return results

    @override
    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        await self._client.delete(
            collection_name=self._name,
            points_selector=models.PointIdsList(
                points=list(record_uuids),
            ),
        )


class QdrantVectorStore(VectorStore):
    """Qdrant-based implementation of VectorStore."""

    def __init__(self, params: QdrantVectorStoreParams) -> None:
        """Initialize QdrantVectorStore."""
        self._client = params.client

    @override
    async def startup(self) -> None:
        pass

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        if properties_schema is None:
            properties_schema = {}

        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dimensions,
                distance=QDRANT_SIMILARITY_METRIC_MAP[similarity_metric],
            ),
        )

        index_creation_tasks = [
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=property_name,
                field_schema=QDRANT_TYPE_MAP[property_type],
            )
            for property_name, property_type in properties_schema.items()
        ]

        await asyncio.gather(*index_creation_tasks)

    @override
    async def get_collection(self, collection_name: str) -> Collection:
        return QdrantCollection(client=self._client, name=collection_name)

    @override
    async def delete_collection(self, collection_name: str) -> None:
        await self._client.delete_collection(collection_name=collection_name)
