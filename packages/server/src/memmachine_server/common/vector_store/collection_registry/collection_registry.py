"""
Abstract base class for a collection registry.

The catalog of a vector store whose backend holds only points: which
logical collections exist, under which incarnation and configuration, and
which dead incarnations await purge. A registry keys its collections by
namespace and name. Its calls are arbitrated across every process sharing
it: creation mints an incarnation no live or queued collection carries,
deletion makes the collection unreachable when it returns, and a purge
claim is handed to one purger at a time.

A registry belongs to one vector deployment: every store whose client
reaches that deployment uses it, and stores on other deployments use
other registries, since a store reclaims its registry's tombstones
through its own client.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)


@dataclass(frozen=True)
class RegisteredCollection:
    """A live collection: the incarnation its points carry and the configuration it was created with."""

    incarnation: UUID
    config: VectorStoreCollectionConfig


@dataclass
class PurgeClaim:
    """A claimed tombstone: the incarnation to purge, where its points are, and what the round found.

    Mutable for `found`, which the purger sets before the claim ends: True
    when the backend still held points under the incarnation, False when
    it held none.
    """

    incarnation: UUID
    namespace: str
    name: str
    config: VectorStoreCollectionConfig
    clean_at: datetime | None
    found: bool | None = None


class VectorStoreCollectionRegistry(ABC):
    """
    The collection registry of one vector store.

    A queue entry is a dead incarnation's tombstone. The backend holds the
    points, and a write the registry read as live can land there after
    the purge that followed the deletion, so one purge cannot be the last:
    the entry stays through purge rounds until a round finds nothing, then
    through a retention, then through one more round that finds nothing
    again. Only then is it removed, and until then the incarnation is
    never re-minted.
    """

    @abstractmethod
    async def startup(self) -> None:
        """Ready the registry, idempotently."""
        raise NotImplementedError

    @abstractmethod
    async def create(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> UUID:
        """
        Register a new collection under a freshly minted incarnation.

        The (namespace, name) is arbitrated across processes, and the
        incarnation is one no live or queued collection carries, so no
        points can be adopted by, or reclaimed out from under, the new
        collection.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
            config (VectorStoreCollectionConfig):
                The configuration the collection is created with.

        Returns:
            UUID: The incarnation the collection's points carry.

        Raises:
            VectorStoreCollectionAlreadyExistsError: The (namespace, name) is taken.
            VectorStoreAttemptsExhaustedError:
                Every minted incarnation was rejected for another reason.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, namespace: str, name: str) -> RegisteredCollection | None:
        """
        Look up the live collection under a (namespace, name).

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.

        Returns:
            RegisteredCollection | None:
                The live collection, or None when there is none.
        """
        raise NotImplementedError

    @abstractmethod
    async def is_live(self, incarnation: UUID) -> bool:
        """
        Whether a collection is still registered under an incarnation.

        Args:
            incarnation (UUID): The incarnation a handle is bound to.

        Returns:
            bool: Whether the incarnation's collection is live.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, namespace: str, name: str) -> None:
        """
        Unregister a collection and queue its incarnation for purge.

        The collection is unreachable when this returns, and its points
        are reclaimed by the purge rounds that claim its tombstone. It is
        idempotent: no collection under the key is the no-op case.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
        """
        raise NotImplementedError

    @abstractmethod
    def claim_oldest(self) -> AbstractAsyncContextManager[PurgeClaim | None]:
        """
        Claim the oldest tombstone due for a purge round, for the body of the context.

        The context yields None when no tombstone is due: the queue is
        empty, or every entry had a clean round less than the retention
        ago. A registry that can hold a claim hands the entry to no other
        purger for the body's duration; one that cannot lets a doubly
        claimed entry cost a repeated, idempotent round and never a missed
        one.

        The body purges and sets `found`. A round that found points keeps
        the entry due, so rounds continue; a round that found none stamps
        the entry clean the first time and removes it when it is the round
        due after the retention. A body that raises leaves the entry as it
        was, for a later claim.

        Returns:
            AbstractAsyncContextManager[PurgeClaim | None]:
                The claim, held for the body of the context.
        """
        raise NotImplementedError
