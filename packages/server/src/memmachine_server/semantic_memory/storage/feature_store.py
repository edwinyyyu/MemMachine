"""Abstract interface for the relational side of semantic feature storage.

This module defines :class:`SemanticFeatureStore`, which owns all
structured / relational data for semantic memory: features, citations,
and history-ingestion tracking.  It does **not** handle embeddings or
vector similarity search — those are delegated to a
:class:`~memmachine_server.common.vector_store.VectorStoreCollection`.

Each feature carries a ``vector_uuid`` (:class:`~uuid.UUID`) that links
it to the corresponding :class:`~memmachine_server.common.vector_store.Record`
in the vector store.  The mapping is 1-to-1.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from memmachine_server.common.episode_store.episode_model import EpisodeIdT
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.semantic_memory.semantic_model import (
    FeatureIdT,
    SemanticFeature,
    SetIdT,
)


class SemanticFeatureStore(ABC):
    """Relational storage for semantic features, citations, and history tracking.

    Implementations persist all structured data **except** embeddings.
    Vector storage and similarity search are handled by a separate
    ``VectorStoreCollection`` instance; the two stores are linked by
    ``vector_uuid``.

    Lifecycle
    ---------
    The orchestrator is responsible for:

    * generating a ``vector_uuid`` (via ``uuid4()``) before writing,
    * upserting the corresponding ``Record`` into the vector store,
    * deleting vector-store records when features are removed (using the
      ``vector_uuid`` values returned by the delete methods).
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
        vector_uuid: UUID,
        set_id: SetIdT,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> FeatureIdT:
        """Persist a new feature and return its relational id.

        The caller is responsible for upserting a
        :class:`~memmachine_server.common.vector_store.Record` with the
        same *vector_uuid* into the vector store.

        Args:
            vector_uuid: UUID that identifies this feature's record in the
                vector store.
            set_id: Feature-set this feature belongs to.
            category_name: Semantic category.
            feature: Feature name / key.
            value: Textual value.
            tag: Tag label.
            metadata: Optional user-defined key-value pairs.

        Returns:
            The newly assigned ``FeatureIdT``.
        """
        raise NotImplementedError

    @abstractmethod
    async def update_feature(
        self,
        feature_id: FeatureIdT,
        *,
        set_id: SetIdT | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Update relational fields of an existing feature.

        Only non-``None`` arguments are applied.  This method does **not**
        accept an embedding — if the *value* changed the caller must also
        update the vector store via
        :meth:`~memmachine_server.common.vector_store.VectorStoreCollection.upsert`.
        If *set_id*, *category_name*, or *tag* changed the caller must
        also update the corresponding vector-store properties.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        """Fetch a single feature by its relational id."""
        raise NotImplementedError

    @abstractmethod
    async def get_features_by_vector_uuids(
        self,
        vector_uuids: Sequence[UUID],
        load_citations: bool = False,
    ) -> Mapping[UUID, SemanticFeature]:
        """Bulk-load features by their vector-store UUIDs.

        This is the primary enrichment method: after querying the vector
        store the orchestrator calls this to hydrate full
        :class:`SemanticFeature` objects for the matched UUIDs.

        Returns:
            Mapping from *vector_uuid* to ``SemanticFeature``.
            UUIDs that do not match any stored feature are silently
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
        :meth:`get_features_by_vector_uuids`.

        Args:
            filter_expr: Relational filter tree
                (``set_id``, ``category``, ``tag``, etc.).
            page_size: Maximum results per page.
            page_num: Zero-based page number (requires *page_size*).
            tag_threshold: Only return features whose tag appears at
                least this many times in the result set.
            load_citations: Whether to eagerly load citation episode IDs.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Feature → vector UUID lookups
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_vector_uuid(
        self,
        feature_id: FeatureIdT,
    ) -> UUID | None:
        """Look up the vector-store UUID for a single feature.

        Returns ``None`` if the feature does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_vector_uuids(
        self,
        feature_ids: Sequence[FeatureIdT],
    ) -> Mapping[FeatureIdT, UUID]:
        """Bulk look up vector-store UUIDs for multiple features.

        Returns:
            Mapping from *feature_id* to *vector_uuid*.
            Feature IDs that do not exist are silently omitted.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Feature deletion
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def delete_features(
        self,
        feature_ids: Sequence[FeatureIdT],
    ) -> Sequence[UUID]:
        """Delete features by relational id.

        The caller must also delete the corresponding records from the
        vector store using the returned UUIDs.

        Returns:
            The ``vector_uuid`` values of the deleted features.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> Sequence[UUID]:
        """Delete features matching the filter.

        The caller must also delete the corresponding records from the
        vector store using the returned UUIDs.

        Returns:
            The ``vector_uuid`` values of the deleted features.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Citations
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def add_citations(
        self,
        feature_id: FeatureIdT,
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
