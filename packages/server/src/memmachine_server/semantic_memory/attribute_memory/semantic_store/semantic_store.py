"""Abstract interface for the relational store of semantic memory.

Defines :class:`SemanticStore`, the source of truth for structured
semantic-memory data: attributes and citations.  Embeddings live in a
paired :class:`~memmachine_server.common.vector_store.VectorStoreCollection`;
the two are linked by identity — each attribute's UUID is also its
vector record's UUID.  Ingestion coordination (tracking which episodes
a consumer has processed) lives in
:class:`~memmachine_server.common.message_queue.MessageQueue`, not here.

Hierarchy
---------

    partition_id -> topic -> category -> attribute -> value

* ``partition_id`` is an opaque scoping/routing key composed by the
  upper layer (typically from org, project, and subject identifiers).
  The store does not interpret its structure; prefix queries on it are
  the only structural operation.
* ``topic`` is schema-fixed by the prompt author — an area of
  knowledge (e.g., ``Profile``, ``CodeKnowledge``).
* ``category`` is LLM-emitted classification within the topic (e.g.,
  ``food``, ``music``).  A fact lives in exactly one category.
* ``attribute`` and ``value`` are the leaf key/value pair.

System vs user metadata
-----------------------
Hierarchy fields are *system* metadata: first-class columns here,
``_``-prefixed keys in the vector store's flat property map.
User-supplied metadata lives in ``properties`` and is addressed via
``m.`` / ``metadata.`` prefixes in the filter DSL.  Keys beginning
with ``_`` are reserved; the orchestrator rejects them at input.

Fault tolerance
---------------
This store is the source of truth.  The orchestrator adds here first,
vector store second; deletes vector store first, here second.  The
resulting invariant is ``vector_store.uuids ⊆ semantic_store.uuids``;
partial failure can only leak orphan vector records, recoverable by
the upstream delivery layer (message queue + retries).

No idempotency guarantees
-------------------------
Mutating methods are not idempotent by contract.  :meth:`add_attribute`
raises on id collision; deletes pass through whatever the backend
does.  At-least-once delivery upstream (via
:class:`~memmachine_server.common.message_queue.MessageQueue`) is the
recovery mechanism, not store-level dedup.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from memmachine_server.common.filter.filter_parser import FilterExpr


class SemanticAttribute(BaseModel):
    """A single leaf in the semantic hierarchy.

    One row represents one fact: under ``partition_id`` / ``topic`` /
    ``category``, the named ``attribute`` has the given ``value``.

    ``citations`` is populated only on read paths when the caller
    passes ``load_citations=True``.  :meth:`SemanticStore.add_attribute`
    ignores this field; citations are attached separately via
    :meth:`SemanticStore.add_citations`.
    """

    id: UUID
    partition_id: str
    topic: str
    category: str
    attribute: str
    value: str
    properties: dict[str, Any] | None = None
    citations: tuple[UUID, ...] | None = None


class SemanticStore(ABC):
    """Relational storage for semantic attributes and citations.

    See the module docstring for the data model, ordering contract, and
    failure semantics.
    """

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def startup(self) -> None:
        """Initialize the storage connection and run pending migrations."""
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self) -> None:
        """Release storage resources."""
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self) -> None:
        """Delete every row in every table managed by this store."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Attribute CRUD
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def add_attribute(self, attribute: SemanticAttribute) -> None:
        """Persist a new attribute.

        Raises on id collision (UUID4 makes collision a caller bug,
        not a routine state).  The orchestrator must use
        ``attribute.id`` when writing the matching record to the
        vector store.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_attribute(
        self,
        attribute_id: UUID,
        *,
        load_citations: bool = False,
    ) -> SemanticAttribute | None:
        """Fetch a single attribute by id, or ``None`` if not found."""
        raise NotImplementedError

    @abstractmethod
    async def get_attributes(
        self,
        attribute_ids: Iterable[UUID],
        *,
        load_citations: bool = False,
    ) -> Mapping[UUID, SemanticAttribute]:
        """Bulk-load attributes by id.

        The primary enrichment path after a vector-store query: pass
        the UUIDs returned by the vector store here to hydrate full
        :class:`SemanticAttribute` objects.  IDs with no matching row
        are silently omitted from the result.
        """
        raise NotImplementedError

    @abstractmethod
    def list_attributes(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        load_citations: bool = False,
    ) -> AsyncIterator[SemanticAttribute]:
        """Stream attributes matching a relational filter.

        For vector-similarity search, query the
        :class:`~memmachine_server.common.vector_store.VectorStoreCollection`
        first, then enrich via :meth:`get_attributes`.

        Filter field naming:
          * System fields: ``partition_id``, ``topic``, ``category``,
            ``attribute``, ``value`` (bare names).
          * User metadata: ``m.<key>`` or ``metadata.<key>``.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_attribute_ids_matching(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> tuple[UUID, ...]:
        """Return attribute ids matching the filter, without fetching rows.

        Intended for the delete path: call this first, delete matching
        records from the vector store, then call
        :meth:`delete_attributes` with the same ids.  This preserves
        the delete-vector-then-store ordering rule.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_attributes(self, attribute_ids: Iterable[UUID]) -> None:
        """Delete attributes by id.

        Missing ids pass through the backend (typically a no-op).  The
        caller must have deleted matching vector records *before*
        calling this method, per the ordering rule.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Citations
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def add_citations(
        self,
        attribute_id: UUID,
        history_ids: Iterable[UUID],
    ) -> None:
        """Associate source message ids as citations for an attribute."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Partition discovery
    # ------------------------------------------------------------------ #

    @abstractmethod
    def list_partitions(
        self,
        *,
        prefix: str | None = None,
    ) -> AsyncIterator[str]:
        """Iterate distinct ``partition_id`` values, optionally filtered by prefix.

        Prefix filtering supports the upper layer's composite
        ``partition_id`` convention (e.g., ``org_X/project_Y/...``).
        """
        raise NotImplementedError
