"""
Abstract base class for a vector store.

A store is one collection: a body of records searched together, with one
dimensionality and one declared schema, named at construction. Within it,
a partition holds one tenant's records, and `VectorStorePartition` is the
handle a data consumer holds for it.
"""

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from uuid import UUID

from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.filter import FilterExpr

from .data_types import QueryResult, Record, VectorStorePartitionAlreadyExistsError


class VectorStorePartition(ABC):
    """
    One partition of a vector store, bound to its key.

    All data operations are scoped to the partition. The handle owns
    nothing: a caller builds one with `VectorStore.get_partition`, drops it,
    and builds another at will.

    A partition stores the properties its store declares
    (`indexed_properties`), typed and indexed for filtering during a
    search, and no others: a record or a filter naming an undeclared key is
    rejected, so an undeclared key never exists in the store, neither
    stored write-only nor scanned for.
    """

    @property
    @abstractmethod
    def partition_key(self) -> str:
        """The key this handle is bound to."""
        raise NotImplementedError

    @property
    @abstractmethod
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        """The declared schema: every key this partition stores and filters on."""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_filter_nodes(self) -> frozenset[type]:
        """
        The filter node classes the backend evaluates during a search.

        A `query` whose filter uses any other node raises
        `UnsupportedFilterError`; a caller routes such a predicate to a store
        that evaluates it afterward.
        """
        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """
        Upsert records in the partition.

        Insert records with new UUIDs,
        and update records with existing UUIDs.

        Args:
            records (Iterable[Record]):
                Iterable of records to upsert.

        Raises:
            UndeclaredPropertyKeyError:
                If a record carries a key the store has not declared;
                raised before anything is sent.
            PropertyTypeMismatchError:
                If a record's value is not of its key's declared type;
                raised before anything is sent.
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
                A filtered search returns fewer when the filter admits fewer.
            min_cosine_similarity (float | None):
                If provided, only return matches whose cosine similarity
                is greater than or equal to this value
                (default: None).
            property_filter (FilterExpr | None):
                Filter expression tree over declared keys, evaluated during
                the search.
                If None, no property filtering is applied
                (default: None).

        Returns:
            list[QueryResult]:
                Results for each query vector,
                ordered as in the input iterable.

        Raises:
            UndeclaredPropertyKeyError:
                If the filter names a key the store has not declared.
            UnsupportedFilterError:
                If the filter uses a node outside `supported_filter_nodes`.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete records from the partition by their UUIDs.

        Args:
            record_uuids (Iterable[UUID]):
                Iterable of UUIDs of the records to delete.
        """
        raise NotImplementedError


async def get_or_create_partition(
    store: "VectorStore", partition_key: str
) -> VectorStorePartition:
    """
    The partition, created if it does not exist.

    Losing a creation race to a concurrent creator is fine: the winner's
    partition is the one returned.
    """
    partition = await store.get_partition(partition_key)
    if partition is not None:
        return partition
    with contextlib.suppress(VectorStorePartitionAlreadyExistsError):
        await store.create_partition(partition_key)
    partition = await store.get_partition(partition_key)
    if partition is None:
        raise RuntimeError(
            f"Partition {partition_key!r} of collection {store.collection!r} was "
            "deleted between its creation and this lookup"
        )
    return partition


class VectorStore(ABC):
    """
    Abstract base class for a vector store.

    A store is one collection, named at construction with its vector
    dimensions and its declared schema; the composition root builds one
    store per collection it needs, and the name is what keeps two stores
    over one engine or one client apart. Every partition of the store
    shares the collection's dimensions and schema, and the schema is fixed
    for the life of the store's data: changing it is a migration.

    A given partition must be managed by at most one process at a time.
    The consumer is responsible for sharding partition keys across
    processes.

    Naming constraints:
        - Collection names must match `[a-z0-9_]+` and be at most 64 bytes.
        - Partition keys and property keys must match `[a-z0-9_]+`
          (lowercase alphanumeric and underscores only) and be at most
          32 bytes.
    """

    @property
    @abstractmethod
    def collection(self) -> str:
        """The collection this store is."""
        raise NotImplementedError

    @property
    @abstractmethod
    def vector_dimensions(self) -> int:
        """Dimensionality of every vector in the store."""
        raise NotImplementedError

    @property
    @abstractmethod
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        """The declared schema every partition of this store carries."""
        raise NotImplementedError

    @abstractmethod
    async def provision(self) -> None:
        """
        Create the collection's durable resources, idempotently.

        The native collection and its payload indexes, or the tables and
        indexes beside the data; whatever must exist before a partition can
        be created. Run once per deployment change by whoever owns the
        schema, before `startup`; never by a request.
        """
        raise NotImplementedError

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    @abstractmethod
    async def create_partition(self, partition_key: str) -> None:
        """
        Create a partition.

        Args:
            partition_key (str):
                The key of the partition.

        Raises:
            VectorStorePartitionAlreadyExistsError: If the partition already exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_partition(self, partition_key: str) -> VectorStorePartition | None:
        """
        Get a handle for an existing partition.

        Args:
            partition_key (str):
                The key of the partition.

        Returns:
            VectorStorePartition | None:
                A handle bound to the partition, or None if the partition
                does not exist.

        Raises:
            VectorStorePartitionSchemaMismatchError:
                If the partition was created under other dimensions or
                another declared schema than this store's.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete a partition and all of its records.

        Idempotent.

        Args:
            partition_key (str):
                The key of the partition to delete.
        """
        raise NotImplementedError
