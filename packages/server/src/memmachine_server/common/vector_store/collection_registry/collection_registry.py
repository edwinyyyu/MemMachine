"""
Abstract base class for a vector store collection registry.

Defines the interface for registering, retrieving,
and deregistering logical collections.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from memmachine_server.common.data_types import ConcurrencyScope
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)


class CollectionRegistryEntry(BaseModel):
    """
    Registered identity of a logical vector store collection.

    Stores the collection's resolved identity alongside its config:
    the native collection name is fixed at registration,
    so later changes to config serialization
    cannot silently repoint the collection at a new native collection,
    and the partition key carries a per-registration generation,
    so records written through handles held across a deregistration
    stay invisible and are never resurrected by a re-registration.

    One entry format is shared by all vector store backends;
    extend it by adding optional fields with defaults
    rather than with backend-specific entry types.
    A new field's default must reproduce the behavior
    that existed before the field:
    stored entries predating it are read through that default.
    A change that cannot be expressed that way is breaking:
    introduce an entry format version field (default 1) at that point --
    the rule above guarantees pre-existing entries classify correctly --
    and migrate deliberately.
    """

    config: VectorStoreCollectionConfig = Field(
        ...,
        description="Configuration for the collection",
    )
    native_collection_name: str = Field(
        ...,
        description="Native collection storing the records",
    )
    partition_key: str = Field(
        ...,
        description=("Generation-scoped partition key for the collection's records"),
    )


class CollectionAlreadyRegisteredError(Exception):
    """Raised when registering a collection that is already registered."""

    def __init__(self, namespace: str, name: str) -> None:
        """Initialize with the namespace and name of the existing collection."""
        self.namespace = namespace
        self.name = name
        super().__init__(f"Collection ({namespace!r}, {name!r}) is already registered.")


class CollectionRegistry(ABC):
    """
    Abstract base class for a vector store collection registry.

    A collection registry is a durable catalog of a vector store's
    logical collections: a mapping from (namespace, name)
    to an immutable CollectionRegistryEntry.
    It is the metadata authority that makes collection lifecycle
    operations atomic across processes sharing the same backing store;
    the vector store holds a registry dedicated to it
    and cannot reach any other registry through it.

    Entries are immutable:
    a collection holds the entry it was registered with
    until it is deregistered.
    Implementations must make registration atomic
    across processes sharing the same backing store.

    Stored entries are canonicalized by serialization,
    so a returned entry may differ from the object it was registered from
    (e.g. defaults filled in), but is identical for every reader.

    Namespaces and names follow the VectorStore naming constraints.
    """

    @property
    @abstractmethod
    def concurrency_scope(self) -> ConcurrencyScope:
        """
        Widest boundary for concurrent use of this registry.

        Instances of the registry's vector store deployed within the
        declared scope observe the same registered collections,
        and registration stays atomic between them.
        """
        raise NotImplementedError

    @abstractmethod
    async def startup(self) -> None:
        """Prepare the registry's backing storage. Idempotent."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    @abstractmethod
    async def register(
        self, *, namespace: str, name: str, entry: CollectionRegistryEntry
    ) -> None:
        """
        Atomically register a collection.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.
            entry (CollectionRegistryEntry):
                Registered identity of the collection.

        Raises:
            CollectionAlreadyRegisteredError:
                If the collection is already registered,
                regardless of whether its stored entry
                matches the provided entry.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, *, namespace: str, name: str) -> CollectionRegistryEntry | None:
        """
        Get the stored entry for a collection.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.

        Returns:
            CollectionRegistryEntry | None:
                The stored entry, or None if the collection is not registered.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_or_register(
        self, *, namespace: str, name: str, entry: CollectionRegistryEntry
    ) -> tuple[CollectionRegistryEntry, bool]:
        """
        Atomically register a collection if it is not registered.

        Never compares entries:
        if the collection is already registered,
        its stored entry is returned unchanged
        regardless of the provided entry.
        Callers enforce their own config equality policy.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.
            entry (CollectionRegistryEntry):
                Registered identity to store if the collection
                is not registered.

        Returns:
            tuple[CollectionRegistryEntry, bool]:
                The stored entry,
                and whether this call registered the collection.
        """
        raise NotImplementedError

    @abstractmethod
    async def deregister(self, *, namespace: str, name: str) -> None:
        """
        Deregister a collection.

        It is idempotent.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.
        """
        raise NotImplementedError
