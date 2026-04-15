"""Abstract interface for the relational side of semantic feature storage.

Defines :class:`SemanticFeatureStore`, which owns all structured /
relational data for semantic memory: features, citations, and
history-ingestion tracking.  Embeddings and similarity search are
delegated to a
:class:`~memmachine_server.common.vector_store.VectorStoreCollection`.

Each feature's primary key is a :class:`~uuid.UUID`, used directly as
``Record.uuid`` in the vector store.  The orchestrator generates the
UUID via ``uuid4()`` before writing to either store; the same UUID
identifies the feature row and its embedding record.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from memmachine_server.common.episode_store.episode_model import EpisodeIdT
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.semantic_memory.semantic_model import (
    SemanticFeature,
    SetIdT,
)


class SemanticFeatureStore(ABC):
    """Relational storage for semantic features, citations, and history tracking.

    Implementations persist all structured data **except** embeddings.
    Vector storage and similarity search are handled by a separate
    :class:`~memmachine_server.common.vector_store.VectorStoreCollection`
    instance; the two stores are linked by identity — each feature's
    UUID primary key is also its ``Record.uuid`` in the vector store.

    The orchestrator generates the UUID via ``uuid4()``, passes it to
    :meth:`add_feature`, and uses the same value for the corresponding
    vector-store ``Record``.  Deletion methods that take a filter return
    the UUIDs of the deleted features so the caller can clean up the
    vector store.
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
        """Release storage resources (connections, pools, etc.)."""
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self) -> None:
        """Delete all data (features, history, citations)."""
        raise NotImplementedError

    @abstractmethod
    async def reset_set_ids(self, set_ids: Sequence[SetIdT]) -> None:
        """Reset backend-specific indexes or caches for the given set_ids."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Feature CRUD
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def add_feature(
        self,
        *,
        feature_id: UUID,
        set_id: SetIdT,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a new feature with the caller-supplied UUID.

        The same *feature_id* must be used as ``Record.uuid`` when the
        caller upserts the corresponding embedding into the vector store.
        """
        raise NotImplementedError

    @abstractmethod
    async def update_feature(
        self,
        feature_id: UUID,
        *,
        set_id: SetIdT | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Update relational fields of an existing feature.

        Only non-``None`` arguments are applied.  No embedding parameter
        — if *value* changed the caller must also upsert the vector
        record; if *set_id*, *category_name*, or *tag* changed the
        caller must also update the vector record's properties.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_feature(
        self,
        feature_id: UUID,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        """Fetch a single feature by id."""
        raise NotImplementedError

    @abstractmethod
    async def get_features(
        self,
        feature_ids: Sequence[UUID],
        load_citations: bool = False,
    ) -> Mapping[UUID, SemanticFeature]:
        """Bulk-load features by id.

        This is the primary enrichment path after a vector-store query:
        pass the UUIDs returned by
        :meth:`~memmachine_server.common.vector_store.VectorStoreCollection.query`
        to hydrate full :class:`SemanticFeature` objects.

        Returns:
            Mapping from *feature_id* to ``SemanticFeature``.
            IDs that do not match any stored feature are silently
            omitted from the result.
        """
        raise NotImplementedError

    @abstractmethod
    def get_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> AsyncIterator[SemanticFeature]:
        """Query features using relational filters only.

        For vector-similarity search the orchestrator should query the
        ``VectorStoreCollection`` first, then enrich via
        :meth:`get_features`.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Feature deletion
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def delete_features(
        self,
        feature_ids: Sequence[UUID],
    ) -> None:
        """Delete features by id.

        The caller is responsible for deleting the matching records
        from the vector store using the same IDs.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> Sequence[UUID]:
        """Delete features matching the filter; return the deleted IDs.

        The caller must delete the matching records from the vector
        store using the returned IDs.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Citations
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def add_citations(
        self,
        feature_id: UUID,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        """Associate episode IDs as citations for a feature."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # History / ingestion tracking
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def add_history_to_set(
        self,
        set_id: SetIdT,
        history_id: EpisodeIdT,
    ) -> None:
        """Record that a history message belongs to a feature set."""
        raise NotImplementedError

    @abstractmethod
    def get_history_messages(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> AsyncIterator[EpisodeIdT]:
        """Retrieve history message IDs with optional ingestion filter."""
        raise NotImplementedError

    @abstractmethod
    async def get_history_messages_count(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        """Count history messages matching the filter."""
        raise NotImplementedError

    @abstractmethod
    async def mark_messages_ingested(
        self,
        *,
        set_id: SetIdT,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        """Mark history messages as ingested for the given set."""
        raise NotImplementedError

    @abstractmethod
    async def delete_history(
        self,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        """Delete history references and their citation associations."""
        raise NotImplementedError

    @abstractmethod
    async def delete_history_set(
        self,
        set_ids: Sequence[SetIdT],
    ) -> None:
        """Delete all history rows for the given set_ids."""
        raise NotImplementedError

    @abstractmethod
    def get_history_set_ids(
        self,
        *,
        min_uningested_messages: int | None = None,
        older_than: datetime | None = None,
    ) -> AsyncIterator[SetIdT]:
        """Return set_ids with pending ingestion work."""
        raise NotImplementedError

    @abstractmethod
    async def purge_ingested_rows(self, set_ids: Sequence[SetIdT]) -> int:
        """Delete fully-ingested history rows for the given set_ids.

        Skips sets that still have un-ingested messages to preserve
        the duplicate guard.

        Returns:
            Number of rows deleted.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Set discovery
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_set_ids_starts_with(self, prefix: str) -> AsyncIterator[SetIdT]:
        """Return set_ids matching the given prefix."""
        raise NotImplementedError
