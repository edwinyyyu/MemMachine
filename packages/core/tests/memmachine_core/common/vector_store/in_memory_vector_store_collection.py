"""In-memory VectorStoreCollection implementation for testing."""

import math
import operator
from collections.abc import Iterable, Sequence
from uuid import UUID

from memmachine_core.common import PropertyValue
from memmachine_core.common.filter import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
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

_COMPARISON_OPS = {
    "=": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


def _evaluate_comparison(prop: PropertyValue, op: str, value: PropertyValue) -> bool:
    fn = _COMPARISON_OPS.get(op)
    if fn is None:
        raise ValueError(f"Unknown comparison op: {op!r}")
    return bool(fn(prop, value))


def evaluate_filter(expr: FilterExpr, properties: dict[str, PropertyValue]) -> bool:
    """Evaluate a FilterExpr against a properties dict."""
    match expr:
        case Comparison(field=field, op=op, value=value):
            prop = properties.get(field)
            if prop is None:
                return False
            return _evaluate_comparison(prop, op, value)
        case In(field=field, values=values):
            return properties.get(field) in values
        case IsNull(field=field):
            return field not in properties
        case And(left=left, right=right):
            return evaluate_filter(left, properties) and evaluate_filter(
                right, properties
            )
        case Or(left=left, right=right):
            return evaluate_filter(left, properties) or evaluate_filter(
                right, properties
            )
        case Not(expr=inner):
            return not evaluate_filter(inner, properties)
        case _:
            raise TypeError(f"Unknown filter expression type: {type(expr)}")


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
    def config(self) -> VectorStoreCollectionConfig:
        return self.collection_config

    async def upsert(self, *, records: Iterable[Record]) -> None:
        for record in records:
            self.records[record.uuid] = Record(
                uuid=record.uuid,
                vector=list(record.vector),
                properties=dict(record.properties),
            )

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

    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        for uid in record_uuids:
            self.records.pop(uid, None)
