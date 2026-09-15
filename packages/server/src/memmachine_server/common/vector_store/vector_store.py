"""
Abstract base class for a vector store.

Defines the interface for adding, querying, and deleting records.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from uuid import UUID

from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)

from .data_types import (
    QueryResult,
    Record,
    VectorStoreCollectionConfig,
)


class VectorStoreCollection(ABC):
    """
    A logical collection in a vector store.

    Identified by a (namespace, name) pair.
    All data operations are scoped to this logical collection.

    A handle is bound to the collection as it was when the handle was
    obtained. What its operations do once the collection has been deleted,
    or deleted and re-created under the same pair, is implementation-defined;
    see `VectorStore`.

    Implementations must support storing and filtering on record properties
    not declared in the configured indexed properties schema.

    The schema exists to support indexing on fixed-type record properties.
    Record properties not declared in the schema may have mixed-type values.
    """

    @property
    @abstractmethod
    def config(self) -> VectorStoreCollectionConfig:
        """The configuration for this collection."""
        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """
        Upsert records in the collection.

        Insert records with new UUIDs,
        and update records with existing UUIDs.

        Args:
            records (Iterable[Record]):
                Iterable of records to upsert.
                Records containing properties
                not in the indexed properties schema
                are allowed.
        """
        raise NotImplementedError

    @abstractmethod
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        min_cosine_similarity: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        """
        Query for records matching the criteria by query vectors.

        Answers with UUIDs and scores. Stored properties are filterable but
        never returned: this store is not the authority for a record's
        content, and its copy is only as fresh as the last write to it -- a
        caller that needs a record's fields reads them from whatever owns
        them.

        Args:
            query_vectors (Iterable[Sequence[float]]):
                The vectors to compare against.
            limit (int):
                Maximum number of matching records to return per query vector.
            min_cosine_similarity (float | None):
                If provided, only return matches whose cosine similarity
                is greater than or equal to this value
                (default: None).
            property_filter (FilterExpr | None):
                Filter expression tree.
                If None or empty, no property filtering is applied
                (default: None).

        Returns:
            list[QueryResult]:
                Results for each query vector,
                ordered as in the input iterable.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete records from the collection by their UUIDs.

        Args:
            record_uuids (Iterable[UUID]):
                Iterable of UUIDs of the records to delete.
        """
        raise NotImplementedError


class VectorStore(ABC):
    """
    Abstract base class for a vector store.

    A given logical collection identified by a (namespace, name) pair
    must be managed by at most one process at a time.
    The consumer is responsible for sharding names across processes.

    Different namespaces are fully independent (separate native collections).
    Multiple logical collections with the same (namespace, vector dimensions, indexed properties schema)
    may share a native collection to reduce overhead.

    Lifecycle, guaranteed by every implementation: `delete_collection` makes
    the collection unreachable through `get_collection` at once and is
    idempotent, and a collection created under a deleted pair starts empty
    and never adopts its predecessor's records.

    Implementation-defined: whether deletion reclaims the records at once or
    defers that to `purge_deleted_collections`; and what a handle held
    across a deletion does. An implementation that binds each handle to one
    incarnation of its collection raises `VectorStoreCollectionHandleStaleError`
    from every operation of a stale handle and never lets one reach a
    successor's records. An implementation that does not may fail with a
    backend error, or, once storage exists again under the same pair, let
    the stale handle reach it. Each implementation states which it is.

    Naming constraints:
        - Namespaces, names, and property keys must match `[a-z0-9_]+`
          (lowercase alphanumeric and underscores only).
        - Each identifier must be at most 32 bytes.
    """

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    @abstractmethod
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        """
        Create a logical collection in the vector store and return a handle to it.

        A (namespace, name) pair uniquely identifies a collection.
        The configuration (dimensions, schema) is fixed at creation time.

        Args:
            namespace (str):
                Groups related collections and guarantees storage
                isolation at the native collection level.
            name (str):
                Name to identify the collection within a namespace.
            config (VectorStoreCollectionConfig):
                Configuration for the collection.

        Raises:
            VectorStoreCollectionAlreadyExistsError: If a collection with the same
                (namespace, name) already exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_collection(
        self, *, namespace: str, name: str
    ) -> VectorStoreCollection | None:
        """
        Get a handle to a logical collection in the vector store.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.

        Returns:
            VectorStoreCollection | None:
                A handle to the collection, or None if it does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        """
        Delete a logical collection from the vector store.

        The collection is unreachable through `get_collection` at once, and
        its records are reclaimed then or by `purge_deleted_collections`, as
        the implementation defines. Idempotent.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_deleted_collections(self) -> bool:
        """
        Physically reclaim storage for deleted collections, bounded per call.

        The sweeper: reclaims what `delete_collection` deferred, for every
        namespace this store serves, oldest deletion first. Each call does a
        bounded amount of work, sized so it does not noticeably degrade
        concurrent request serving, and is safe to repeat and to run
        concurrently from any process. The store never schedules it; a
        deployment must run it somewhere (the server's resource manager runs
        it in the background). Implementations that reclaim in
        `delete_collection` return False without doing anything.

        Returns:
            bool:
                True if another call may reclaim more. False if this call
                found nothing to claim.
        """
        raise NotImplementedError
