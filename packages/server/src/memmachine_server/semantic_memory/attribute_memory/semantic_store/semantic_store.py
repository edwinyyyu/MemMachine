"""Abstract interface for the relational store of semantic memory.

Defines :class:`SemanticStore`, a partition-handle factory, and
:class:`SemanticStorePartition`, the partition-scoped handle that
exposes all data operations.  The store is the source of truth for
attributes and citations; embeddings live in a paired
:class:`~memmachine_server.common.vector_store.VectorStoreCollection`
(itself the vector-side partition handle), linked by identity — each
attribute's UUID is also its vector record's UUID.

Hierarchy
---------

    topic -> category -> attribute -> value

Scoped to the partition the handle belongs to.

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

Partition keys
--------------
Partition keys must match ``[a-z0-9_]+`` and be at most 32 bytes —
the constraint comes from Postgres LIST partitioning, where the key
ends up in a child-table identifier.

Fault tolerance
---------------
This store is the source of truth.  The orchestrator adds here first,
vector store second; deletes vector store first, here second.  The
resulting invariant is ``vector_store.uuids ⊆ semantic_store.uuids``;
partial failure can only leak orphan vector records, recoverable by
the upstream delivery layer (message queue + retries).

No idempotency guarantees
-------------------------
Mutating methods are not idempotent by contract.
:meth:`SemanticStorePartition.add_attribute` raises on id collision;
deletes pass through whatever the backend does.  At-least-once
delivery upstream is the recovery mechanism, not store-level dedup.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from memmachine_server.common.filter.filter_parser import FilterExpr


class SemanticAttribute(BaseModel):
    """A single leaf in the semantic hierarchy.

    One row represents one fact: under ``topic`` / ``category``, the
    named ``attribute`` has the given ``value``.  The owning partition
    is implicit in the :class:`SemanticStorePartition` handle used to
    insert and read the row.

    ``citations`` is populated only on read paths when the caller
    passes ``load_citations=True``.
    :meth:`SemanticStorePartition.add_attribute` ignores this field;
    citations are attached separately via
    :meth:`SemanticStorePartition.add_citations`.
    """

    id: UUID
    topic: str
    category: str
    attribute: str
    value: str
    properties: dict[str, Any] | None = None
    citations: tuple[UUID, ...] | None = None


class SemanticStorePartition(ABC):
    """Partition-scoped handle for a semantic store.

    All data operations are scoped to the partition this handle was
    opened against.  See the module docstring for ordering and
    idempotency contracts.
    """

    # ------------------------------------------------------------------ #
    # Attribute CRUD
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def add_attributes(
        self,
        attributes: Iterable[SemanticAttribute],
    ) -> None:
        """Persist new attributes into this partition in one transaction.

        Raises on uuid collision (UUID4 makes collision a caller bug,
        not a routine state).  The orchestrator must use each
        ``attribute.id`` when writing the matching record to the
        paired vector store.  Citations carried on
        :attr:`SemanticAttribute.citations` are ignored here; attach
        them separately via :meth:`add_citations`.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_attributes(
        self,
        attribute_uuids: Iterable[UUID],
        *,
        load_citations: bool = False,
    ) -> Mapping[UUID, SemanticAttribute]:
        """Bulk-load attributes by uuid.

        The primary enrichment path after a vector-store query: pass
        the UUIDs returned by the vector store here to hydrate full
        :class:`SemanticAttribute` objects.  UUIDs with no matching row
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

        For vector-similarity search, query the paired
        :class:`~memmachine_server.common.vector_store.VectorStoreCollection`
        first, then enrich via :meth:`get_attributes`.

        Filter field naming:
          * System fields: ``topic``, ``category``, ``attribute``,
            ``value`` (bare names).
          * User metadata: ``m.<key>`` or ``metadata.<key>``.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_attribute_uuids_matching(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> tuple[UUID, ...]:
        """Return attribute uuids matching the filter, without fetching rows.

        Intended for the delete path: call this first, delete matching
        records from the vector store, then call
        :meth:`delete_attributes` with the same uuids.  This preserves
        the delete-vector-then-store ordering rule.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_attributes(self, attribute_uuids: Iterable[UUID]) -> None:
        """Delete attributes by uuid.

        Missing uuids pass through the backend (typically a no-op).
        The caller must have deleted matching vector records *before*
        calling this method, per the ordering rule.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Citations
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def add_citations(
        self,
        attribute_uuid: UUID,
        history_uuids: Iterable[UUID],
    ) -> None:
        """Associate source message uuids as citations for an attribute."""
        raise NotImplementedError


class SemanticStore(ABC):
    """Partition-handle factory for relational semantic-memory storage.

    Manages partition lifecycle (create / open / delete); all data
    operations live on :class:`SemanticStorePartition` handles.

    Partition keys must match ``[a-z0-9_]+`` and be at most 32 bytes.
    """

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Partition management
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def create_partition(self, partition_key: str) -> None:
        """Create a new partition.

        Raises:
            SemanticStorePartitionAlreadyExistsError: If the partition
                already exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_partition(self, partition_key: str) -> SemanticStorePartition | None:
        """Open a handle for an existing partition, or ``None`` if missing."""
        raise NotImplementedError

    @abstractmethod
    async def open_or_create_partition(
        self, partition_key: str
    ) -> SemanticStorePartition:
        """Open the partition if it exists, or create it if it does not."""
        raise NotImplementedError

    @abstractmethod
    async def close_partition(
        self, semantic_store_partition: SemanticStorePartition
    ) -> None:
        """Close a partition-scoped handle."""
        raise NotImplementedError

    @abstractmethod
    async def delete_partition(self, partition_key: str) -> None:
        """Delete a partition and all data within it.  Idempotent."""
        raise NotImplementedError
