"""In-memory VectorStoreCollection implementation for testing."""

import math
import operator
from collections.abc import Iterable, Sequence
from typing import override
from uuid import UUID

from memmachine_core.common import PropertyValue
from memmachine_core.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
)
from memmachine_core.common.vector_store import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollection,
    VectorStoreCollectionConfig,
)

# ---------------------------------------------------------------------------
# Filter evaluation
# ---------------------------------------------------------------------------

_ORDERING_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


def _comparable(
    properties: dict[str, PropertyValue], field: str, value: PropertyValue
) -> PropertyValue | None:
    """Return the stored value if the field holds one of `value`'s type, else None.

    A predicate matches only a field holding a comparable value, so an absent
    field and a field of another type are alike non-matches.
    """
    prop = properties.get(field)
    return prop if type(prop) is type(value) else None


def evaluate_filter(expr: FilterExpr, properties: dict[str, PropertyValue]) -> bool:
    """Evaluate a FilterExpr against a properties dict."""
    match expr:
        case Equals(field, value):
            return _comparable(properties, field, value) == value
        case NotEquals(field, value):
            prop = _comparable(properties, field, value)
            return prop is not None and prop != value
        case Ordering(field, op, value):
            prop = _comparable(properties, field, value)
            return prop is not None and bool(_ORDERING_OPS[op](prop, value))
        case In(field, values):
            return any(_comparable(properties, field, v) == v for v in values)
        case IsMissing(field):
            return field not in properties
        case And(operands):
            return all(evaluate_filter(o, properties) for o in operands)
        case Or(operands):
            return any(evaluate_filter(o, properties) for o in operands)
        case Not(operand):
            return not evaluate_filter(operand, properties)


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(a, b) / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# InMemoryVectorStoreCollection
# ---------------------------------------------------------------------------


class InMemoryVectorStoreCollection(VectorStoreCollection):
    """
    In-memory VectorStoreCollection for testing.

    Scores matches by cosine similarity, with full FilterExpr evaluation
    on record properties.
    """

    def __init__(self, collection_config: VectorStoreCollectionConfig) -> None:
        self.collection_config = collection_config
        self.records: dict[UUID, Record] = {}

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self.collection_config

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        for record in records:
            self.records[record.uuid] = Record(
                uuid=record.uuid,
                vector=list(record.vector),
                properties=dict(record.properties),
            )

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        min_cosine_similarity: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        results: list[QueryResult] = []
        for query_vector in query_vectors:
            qv = list(query_vector)
            matches: list[QueryMatch] = []
            for record in self.records.values():
                if property_filter is not None and not evaluate_filter(
                    property_filter, record.properties
                ):
                    continue
                cosine_similarity = _cosine_similarity(qv, record.vector)
                if (
                    min_cosine_similarity is not None
                    and cosine_similarity < min_cosine_similarity
                ):
                    continue
                matches.append(
                    QueryMatch(
                        cosine_similarity=cosine_similarity,
                        record_uuid=record.uuid,
                    )
                )
            matches.sort(key=lambda m: m.cosine_similarity, reverse=True)
            if limit is not None:
                matches = matches[:limit]
            results.append(QueryResult(matches=matches))
        return results

    @override
    async def get_cosine_similarity(
        self,
        *,
        query_vector: Sequence[float],
        record_uuids: Iterable[UUID],
    ) -> dict[UUID, float]:
        qv = list(query_vector)
        similarities: dict[UUID, float] = {}
        for uid in record_uuids:
            record = self.records.get(uid)
            if record is None:
                continue
            similarities[uid] = _cosine_similarity(qv, record.vector)
        return similarities

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        for uid in record_uuids:
            self.records.pop(uid, None)
