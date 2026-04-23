"""No-op vector store for benchmarking. All writes are discarded, reads return empty."""

import threading
from collections.abc import Iterable, Sequence
from typing import override
from uuid import UUID

from memmachine_server.common.filter.filter_parser import FilterExpr

from .data_types import CollectionConfig, QueryResult, Record
from .vector_store import Collection, VectorStore


class _Counter:
    """Thread-safe counter shared across all NoopCollections."""

    def __init__(self, report_every: int = 10_000) -> None:
        self._total = 0
        self._report_every = report_every
        self._lock = threading.Lock()

    def add(self, n: int) -> None:
        with self._lock:
            prev = self._total
            self._total += n
            if self._total // self._report_every > prev // self._report_every:
                print(
                    f"[noop vector store] {self._total} total vectors upserted",
                    flush=True,
                )


class NoopCollection(Collection):
    def __init__(self, counter: _Counter) -> None:
        self._counter = counter

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        self._counter.add(sum(1 for _ in records))

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
        return [QueryResult(matches=[]) for _ in query_vectors]

    @override
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
        return []

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        pass


class NoopVectorStore(VectorStore):
    def __init__(self) -> None:
        self._counter = _Counter()

    @override
    async def startup(self) -> None:
        pass

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def create_collection(
        self, *, namespace: str, name: str, config: CollectionConfig
    ) -> None:
        pass

    @override
    async def open_or_create_collection(
        self, *, namespace: str, name: str, config: CollectionConfig
    ) -> Collection:
        return NoopCollection(self._counter)

    @override
    async def open_collection(self, *, namespace: str, name: str) -> Collection | None:
        return NoopCollection(self._counter)

    @override
    async def close_collection(self, *, collection: Collection) -> None:
        pass

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        pass
