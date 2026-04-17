"""Attribute memory: orchestrator for the paired :class:`SemanticStore` and vector collection.

Enforces the fault-tolerance ordering rules between the relational
source of truth and the vector index.  This module is a thin
convenience layer — no state beyond the two handles, no lifecycle
ownership (the caller constructs and disposes the dependencies).

Ordering rules
--------------
* **Add**: :class:`SemanticStore` first, then
  :class:`VectorStoreCollection`.  A partial failure leaves a row in
  the store without a corresponding vector record, which is tolerable
  because the invariant is ``vector.uuids ⊆ store.uuids``.  The
  upstream delivery layer retries to close the gap.
* **Delete**: :class:`VectorStoreCollection` first, then
  :class:`SemanticStore`.  Same invariant is preserved: a partial
  failure can leak an orphan store row, but never an orphan vector
  record.
"""

from collections.abc import AsyncIterator, Iterable, Mapping
from uuid import UUID

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.vector_store import VectorStoreCollection
from memmachine_server.semantic_memory.attribute_memory.semantic_store.semantic_store import (
    SemanticAttribute,
    SemanticStore,
)
from memmachine_server.semantic_memory.attribute_memory.vector_adapter import (
    build_vector_record,
    translate_filter_for_vector_store,
    validate_attribute_properties,
)


class AttributeMemory:
    """Coordinate writes and searches across the semantic and vector stores."""

    def __init__(
        self,
        *,
        store: SemanticStore,
        vector_collection: VectorStoreCollection,
    ) -> None:
        """Bind to existing store handles; caller owns their lifecycle."""
        self._store = store
        self._vector = vector_collection

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    async def add_attribute(
        self,
        *,
        attribute: SemanticAttribute,
        vector: list[float],
    ) -> None:
        """Persist an attribute to both stores, store first.

        Rejects user properties with reserved (``_``-prefixed) keys
        before touching either store.
        """
        validate_attribute_properties(attribute.properties)
        await self._store.add_attribute(attribute)
        record = build_vector_record(attribute, vector)
        await self._vector.upsert(records=[record])

    async def add_citations(
        self,
        *,
        attribute_id: UUID,
        history_ids: Iterable[UUID],
    ) -> None:
        """Record source-message ids as evidence for an attribute."""
        await self._store.add_citations(attribute_id, history_ids)

    async def delete_attributes(
        self,
        attribute_ids: Iterable[UUID],
    ) -> None:
        """Delete attributes from both stores, vector first."""
        ids = tuple(attribute_ids)
        if not ids:
            return
        await self._vector.delete(record_uuids=ids)
        await self._store.delete_attributes(ids)

    async def delete_attributes_matching(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        """Delete attributes matching a relational filter, vector first."""
        ids = await self._store.list_attribute_ids_matching(filter_expr=filter_expr)
        if not ids:
            return
        await self.delete_attributes(ids)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    async def get_attribute(
        self,
        attribute_id: UUID,
        *,
        load_citations: bool = False,
    ) -> SemanticAttribute | None:
        """Fetch a single attribute by id."""
        return await self._store.get_attribute(
            attribute_id, load_citations=load_citations
        )

    async def get_attributes(
        self,
        attribute_ids: Iterable[UUID],
        *,
        load_citations: bool = False,
    ) -> Mapping[UUID, SemanticAttribute]:
        """Bulk-load attributes by id."""
        return await self._store.get_attributes(
            attribute_ids, load_citations=load_citations
        )

    async def list_attributes(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        load_citations: bool = False,
    ) -> AsyncIterator[SemanticAttribute]:
        """Stream attributes matching a relational filter."""
        async for attribute in self._store.list_attributes(
            filter_expr=filter_expr,
            load_citations=load_citations,
        ):
            yield attribute

    async def list_partitions(
        self,
        *,
        prefix: str | None = None,
    ) -> AsyncIterator[str]:
        """Iterate partition ids, optionally filtered by prefix."""
        async for partition_id in self._store.list_partitions(prefix=prefix):
            yield partition_id

    async def search(
        self,
        *,
        query_vector: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_expr: FilterExpr | None = None,
        load_citations: bool = False,
    ) -> list[tuple[SemanticAttribute, float]]:
        """Similarity search enriched from the semantic store.

        Returns ``(attribute, score)`` pairs ordered by descending
        match quality.  The filter is translated to vector-store
        property keys (system fields become ``_``-prefixed; user
        metadata ``m.X`` / ``metadata.X`` becomes ``X``) before
        being submitted.
        """
        vector_filter = translate_filter_for_vector_store(filter_expr)
        results = await self._vector.query(
            query_vectors=[query_vector],
            limit=top_k,
            score_threshold=score_threshold,
            property_filter=vector_filter,
            return_properties=False,
        )
        if not results:
            return []
        matches = results[0].matches
        if not matches:
            return []
        uuids = [m.record.uuid for m in matches]
        attribute_map = await self._store.get_attributes(
            uuids, load_citations=load_citations
        )
        return [
            (attribute_map[m.record.uuid], m.score)
            for m in matches
            if m.record.uuid in attribute_map
        ]
