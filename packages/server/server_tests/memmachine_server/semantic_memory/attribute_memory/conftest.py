"""Shared fixtures for attribute_memory tests.

Provides a file-backed SQLite engine so :class:`SQLAlchemySemanticStore`
validators (which reject ``:memory:`` + ``StaticPool``) are satisfied,
and a minimal in-memory :class:`VectorStoreCollection` fake for
orchestrator tests.
"""

from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from uuid import UUID

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.vector_store import (
    Record,
    VectorStoreCollection,
)
from memmachine_server.common.vector_store.data_types import (
    QueryMatch,
    QueryResult,
    VectorStoreCollectionConfig,
)


@pytest_asyncio.fixture
async def sqlite_engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    """File-based SQLite engine for integration tests."""
    db_path = tmp_path / "attribute_memory_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    yield engine
    await engine.dispose()


class FakeVectorStoreCollection(VectorStoreCollection):
    """Minimal in-memory :class:`VectorStoreCollection` for orchestrator tests.

    * ``upsert`` records are kept in insertion order.
    * ``query`` returns every stored record that passes the property
      filter, scored by ``-|index|`` (later inserts rank higher).
    * ``delete`` removes records by UUID.
    * ``get`` returns records by UUID in the requested order.

    This does **not** implement real vector similarity; the orchestrator
    tests only need to verify that the right records are returned and
    enriched, and that the right property filter is forwarded.
    """

    def __init__(self, config: VectorStoreCollectionConfig) -> None:
        self._config = config
        self._records: dict[UUID, Record] = {}
        self._order: list[UUID] = []
        self.last_property_filter: FilterExpr | None = None

    @property
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    async def upsert(self, *, records: Iterable[Record]) -> None:
        for record in records:
            if record.uuid not in self._records:
                self._order.append(record.uuid)
            self._records[record.uuid] = record

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
        self.last_property_filter = property_filter
        query_vectors_list = list(query_vectors)
        results: list[QueryResult] = []
        for _ in query_vectors_list:
            matches: list[QueryMatch] = []
            for i, uid in enumerate(self._order):
                record = self._records[uid]
                if property_filter is not None and not _match(
                    property_filter, record.properties or {}
                ):
                    continue
                projected = Record(
                    uuid=record.uuid,
                    vector=record.vector if return_vector else None,
                    properties=record.properties if return_properties else None,
                )
                matches.append(QueryMatch(score=-float(i), record=projected))
            matches.sort(key=lambda m: m.score, reverse=True)
            if limit is not None:
                matches = matches[:limit]
            results.append(QueryResult(matches=matches))
        return results

    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
        out: list[Record] = []
        for uid in record_uuids:
            record = self._records.get(uid)
            if record is None:
                continue
            out.append(
                Record(
                    uuid=record.uuid,
                    vector=record.vector if return_vector else None,
                    properties=record.properties if return_properties else None,
                )
            )
        return out

    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        for uid in record_uuids:
            self._records.pop(uid, None)
            if uid in self._order:
                self._order.remove(uid)


def _match(expr: FilterExpr, props: dict) -> bool:
    """Minimal filter evaluator for the fake vector store.

    Supports the subset of :mod:`filter_parser` nodes used in orchestrator
    tests: ``Comparison`` (with ``=``) and the logical combinators.
    """
    from memmachine_server.common.filter.filter_parser import (
        And,
        Comparison,
        In,
        IsNull,
        Not,
        Or,
    )

    if isinstance(expr, Comparison):
        return props.get(expr.field) == expr.value
    if isinstance(expr, In):
        return props.get(expr.field) in expr.values
    if isinstance(expr, IsNull):
        return props.get(expr.field) is None
    if isinstance(expr, And):
        return _match(expr.left, props) and _match(expr.right, props)
    if isinstance(expr, Or):
        return _match(expr.left, props) or _match(expr.right, props)
    if isinstance(expr, Not):
        return not _match(expr.expr, props)
    raise TypeError(f"Unsupported filter node: {type(expr).__name__}")


@pytest_asyncio.fixture
async def fake_vector_collection() -> FakeVectorStoreCollection:
    """In-memory vector collection with the system-field schema."""
    from memmachine_server.semantic_memory.attribute_memory.vector_adapter import (
        SYSTEM_PROPERTIES_SCHEMA,
    )

    return FakeVectorStoreCollection(
        VectorStoreCollectionConfig(
            vector_dimensions=3,
            similarity_metric=SimilarityMetric.COSINE,
            properties_schema={**SYSTEM_PROPERTIES_SCHEMA},
        )
    )
