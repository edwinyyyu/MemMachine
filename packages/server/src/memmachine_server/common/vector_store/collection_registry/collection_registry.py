"""
Abstract base class for a collection registry.

The catalog of a vector store whose backend cannot arbitrate one: which
logical collections exist, under which incarnation and configuration, and
which deleted incarnations await purge. Its calls are arbitrated across
every process sharing it: registration mints an incarnation no registered or
queued collection carries, unregistration makes the collection unreachable
when it returns, and a purge claim goes to one purger at a time.

Stores share a registry exactly when their clients connect to the same
Qdrant, or the same Milvus database.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from uuid import UUID

from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)


@dataclass(frozen=True)
class RegisteredCollection:
    """
    A registered collection.

    Its `incarnation` is the value its records carry, `config` the
    configuration it was created with, and `live` whether its storage is
    prepared: a collection is pending until it is marked live.
    """

    incarnation: UUID
    config: VectorStoreCollectionConfig
    live: bool


@dataclass
class PurgeClaim:
    """
    One purge round's claim on a tombstone.

    The registry fills in `incarnation`, the value the deleted collection's
    records carry, and `namespace` and `config`, the deleted collection's,
    which locate its records in the store. The round sets `any_records_found`
    before the claim ends: whether it found records under the incarnation.
    """

    incarnation: UUID
    namespace: str
    config: VectorStoreCollectionConfig
    any_records_found: bool | None = None


class VectorStoreCollectionRegistry(ABC):
    """
    The collection registry of one vector store.

    A deleted collection's incarnation waits on a queue as a tombstone. A
    write checked as live before the deletion can land in the backend after
    it, so a tombstone's purge starts once a retention, longer than any
    write can be in flight, has passed since the deletion. The tombstone is
    removed when a purge round finds nothing, and its incarnation is not
    minted again before then.
    """

    @abstractmethod
    async def startup(self) -> None:
        """Make the registry ready for use; its owner calls it before the first use."""
        raise NotImplementedError

    @abstractmethod
    async def register(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> UUID:
        """
        Register a new collection, pending, under a freshly minted incarnation.

        The (namespace, name) is arbitrated across processes, and the
        incarnation is one no registered or queued collection carries, so the
        new collection starts empty and no purge reclaims its records. The
        collection is pending until `mark_live` marks it, and holds its
        (namespace, name) meanwhile.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
            config (VectorStoreCollectionConfig):
                The configuration the collection is created with.

        Returns:
            UUID: The incarnation the collection's records carry.

        Raises:
            VectorStoreCollectionAlreadyExistsError:
                The (namespace, name) is taken, by a live or a pending
                collection.
            VectorStoreAttemptsExhaustedError:
                The registry gave up after repeated inserts were rejected
                with the (namespace, name) free.
        """
        raise NotImplementedError

    @abstractmethod
    async def mark_live(self, namespace: str, name: str, incarnation: UUID) -> bool:
        """
        Mark a pending collection live, once its storage is prepared.

        The collection is marked if it is still registered, pending, under
        the incarnation.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
            incarnation (UUID): The incarnation `register` returned.

        Returns:
            bool:
                Whether the collection was pending under the incarnation, and
                is now live.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, namespace: str, name: str) -> RegisteredCollection | None:
        """
        Look up the collection registered under a (namespace, name).

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.

        Returns:
            RegisteredCollection | None:
                The collection, pending or live, or None when there is none.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(
        self, namespace: str, name: str, *, incarnation: UUID | None = None
    ) -> None:
        """
        Unregister a collection and queue its incarnation for purge.

        The collection, live or pending, is unreachable when this returns,
        and purge rounds reclaim its records later. Idempotent.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
            incarnation (UUID | None):
                If given, unregister the collection only if it is registered
                under this incarnation (default: None).
        """
        raise NotImplementedError

    @abstractmethod
    def claim_purgeable_incarnation(
        self,
    ) -> AbstractAsyncContextManager[PurgeClaim | None]:
        """
        Claim a due tombstone for one purge round, run in the body of the context.

        A tombstone is due once the retention has passed since its
        deletion. In the body, the caller deletes records under
        `claim.incarnation`, which `claim.namespace` and `claim.config`
        locate, and sets `claim.any_records_found`. A round that
        found no records removes the tombstone and frees its incarnation. A
        body that raises is a failed round: the tombstone is claimed again
        after a backoff that grows with each consecutive failure, and one
        whose rounds keep failing is dead-lettered and reported. A round must
        be safe to repeat, since a registry may hand one tombstone to two
        purgers.

        Returns:
            AbstractAsyncContextManager[PurgeClaim | None]:
                The claim, or None when no tombstone is due.
        """
        raise NotImplementedError
