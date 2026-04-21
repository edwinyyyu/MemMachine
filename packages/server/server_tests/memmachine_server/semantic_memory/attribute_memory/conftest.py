"""Shared fixtures for attribute_memory tests."""

import hashlib
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.reranker import Reranker
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
    """In-memory :class:`VectorStoreCollection` for orchestrator tests.

    * ``upsert`` records are kept in insertion order.
    * ``query`` returns every stored record that passes the property
      filter, scored by ``-|index|`` (later inserts rank higher).
    * ``delete`` removes records by UUID.
    * ``get`` returns records by UUID in the requested order.
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
    """Minimal filter evaluator for :class:`FakeVectorStoreCollection`."""
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
    from memmachine_server.semantic_memory.attribute_memory import AttributeMemory

    return FakeVectorStoreCollection(
        VectorStoreCollectionConfig(
            vector_dimensions=3,
            similarity_metric=SimilarityMetric.COSINE,
            properties_schema={
                **AttributeMemory.expected_vector_store_collection_schema()
            },
        )
    )


# --------------------------------------------------------------------------- #
# Deterministic fake dependencies for AttributeMemory orchestrator tests.
# --------------------------------------------------------------------------- #


class FakeEmbedder(Embedder):
    """Deterministic 3-dim embedder backed by a SHA-256 prefix."""

    def __init__(self, dimensions: int = 3) -> None:
        self._dim = dimensions
        self.ingest_calls: list[list[str]] = []
        self.search_calls: list[list[str]] = []

    async def ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        batch = [str(x) for x in inputs]
        self.ingest_calls.append(batch)
        return [self._embed(x) for x in batch]

    async def search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        batch = [str(q) for q in queries]
        self.search_calls.append(batch)
        return [self._embed(x) for x in batch]

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self._dim)]

    @property
    def model_id(self) -> str:
        return "fake-embedder"

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class FakeLanguageModel(LanguageModel):
    """Scripted LLM: tests queue responses, each call pops one.

    Real LLM backends parse the raw JSON response into the
    ``output_format`` pydantic model before returning.  Tests push
    either already-parsed model instances or raw dicts; dicts are
    validated through a :class:`TypeAdapter` here to mirror that
    contract.
    """

    def __init__(self) -> None:
        self.parsed_responses: list[Any] = []
        self.parsed_calls: list[dict[str, Any]] = []
        self.raise_on_parsed: Exception | None = None

    async def generate_parsed_response(
        self,
        output_format: type,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> Any:
        from pydantic import TypeAdapter

        self.parsed_calls.append(
            {
                "output_format": output_format,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if self.raise_on_parsed is not None:
            raise self.raise_on_parsed
        if not self.parsed_responses:
            return None
        raw = self.parsed_responses.pop(0)
        return TypeAdapter(output_format).validate_python(raw)

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        raise NotImplementedError

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any, int, int]:
        raise NotImplementedError


class FakeReranker(Reranker):
    """Scores candidates by reversed index order (first = highest)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        self.calls.append((query, list(candidates)))
        n = len(candidates)
        return [float(n - i) for i in range(n)]


@pytest_asyncio.fixture
async def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest_asyncio.fixture
async def fake_llm() -> FakeLanguageModel:
    return FakeLanguageModel()


@pytest_asyncio.fixture
async def fake_reranker() -> FakeReranker:
    return FakeReranker()
