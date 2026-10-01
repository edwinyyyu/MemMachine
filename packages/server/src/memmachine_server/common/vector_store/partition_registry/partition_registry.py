"""
Abstract base classes for a collection registry and its registrations.

The catalog of a vector store whose backend cannot arbitrate one: which
logical collections exist, under which incarnation and configuration, and
which deleted incarnations await purge. Its calls are arbitrated across
every process sharing it: registration mints an incarnation no registered or
queued collection carries, unregistration makes the collection unreachable
when it returns, and a purge claim hands out a due tombstone, possibly to
two purgers at once.

The registry is addressed by (namespace, name). Registering a collection
answers a `PendingRegistration`, through which its creator marks it live or
abandons it; resolving a name answers a `LiveRegistration`, through which a
handle checks that the collection has not been deleted. Each acts on one life
of the collection alone.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from uuid import UUID

from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)


@dataclass(frozen=True)
class Registration:
    """
    One life of a registered collection, from its registration to its deletion.

    Its fields belong to this life and never change: `incarnation` is the
    value its records carry, and `config` the configuration it was created
    with.
    """

    namespace: str
    name: str
    config: VectorStoreCollectionConfig
    incarnation: UUID


@dataclass(frozen=True)
class PendingRegistration(Registration, ABC):
    """
    A collection's registration while its creator prepares its storage.

    A registry implementation supplies the methods, each of which writes the
    registry.
    """

    @abstractmethod
    async def mark_live(self) -> "LiveRegistration":
        """
        Mark this pending collection as live.

        Called once, when the collection's storage is prepared. A concurrent
        deletion of the collection either follows the mark or makes it raise.

        Returns:
            LiveRegistration: The same life of the collection, live.

        Raises:
            VectorStorePartitionDeletedError:
                If this life is no longer pending: the collection was
                deleted.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self) -> None:
        """
        Unregister this life of the collection and queue its incarnation for purge.

        The collection is unreachable when this returns, and purge rounds
        reclaim its records later. A collection registered since under the
        same (namespace, name) is another life and stays. Idempotent.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class LiveRegistration(Registration, ABC):
    """
    A live collection's registration.

    A registry implementation supplies the method, which reads the registry.
    """

    @abstractmethod
    async def require_current(self) -> None:
        """
        Raise unless this life is still the collection registered under its (namespace, name).

        One read of the registry: a deletion committed before the read makes
        it raise, and one committed after does not.

        Raises:
            VectorStorePartitionHandleStaleError:
                If the collection was deleted, whether or not another was
                registered under the (namespace, name) since.
        """
        raise NotImplementedError


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


class VectorStorePartitionRegistry(ABC):
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
    ) -> PendingRegistration:
        """
        Register a new pending collection under a freshly minted incarnation.

        The (namespace, name) is arbitrated across processes, and the
        incarnation is one no registered or queued collection carries, so the
        new collection starts empty and no purge reclaims its records. The
        collection is pending until its registration's `mark_live`, and holds
        its (namespace, name) meanwhile.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
            config (VectorStoreCollectionConfig):
                The configuration the collection is created with.

        Returns:
            PendingRegistration: The new collection's registration.

        Raises:
            VectorStorePartitionAlreadyExistsError:
                The (namespace, name) is taken, by a live or a pending
                collection.
            VectorStoreAttemptsExhaustedError:
                The registry gave up after repeated attempts to register
                the free (namespace, name) failed.
        """
        raise NotImplementedError

    @abstractmethod
    async def resolve(self, namespace: str, name: str) -> LiveRegistration | None:
        """
        Resolve a (namespace, name) to the live collection registered under it.

        One read of the registry decides the outcome.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.

        Returns:
            LiveRegistration | None:
                The live collection's registration, or None when no
                collection is registered under the (namespace, name).

        Raises:
            VectorStorePartitionPendingError:
                If the collection registered under the (namespace, name) is
                pending.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self, namespace: str, name: str) -> None:
        """
        Unregister the collection under a (namespace, name) and queue its incarnation for purge.

        The collection, pending or live, is unreachable when this returns,
        and purge rounds reclaim its records later. Idempotent.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
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
        body that raises is a failed round, and the tombstone stays for a
        later claim; a registry may delay a failed tombstone's next claim, or
        stop claiming one whose rounds keep failing, leaving its records in
        place. A round must be safe to repeat, since a registry may hand one
        tombstone to two purgers.

        Returns:
            AbstractAsyncContextManager[PurgeClaim | None]:
                The claim, or None when no tombstone is due.
        """
        raise NotImplementedError
