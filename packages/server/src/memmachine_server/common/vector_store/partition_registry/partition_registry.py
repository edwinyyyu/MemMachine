"""
Abstract base class for a partition registry.

The catalog of a vector store whose backend cannot arbitrate one: which
partitions exist, under which incarnation and schema, and which deleted
incarnations await purge. A registry belongs to one vector store. Its calls
are arbitrated across every process sharing it: registration mints an
incarnation no registered or queued partition carries, unregistration makes
the partition unreachable when it returns, and a purge claim hands out a due
tombstone, possibly to two purgers at once.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from memmachine_server.common.vector_store.data_types import PartitionSchema


@dataclass(frozen=True)
class RegisteredPartition:
    """
    A registered partition.

    Its `incarnation` is the value its records carry, `schema` what it was
    created under, `live` whether its storage is prepared (a partition is
    pending until it is marked live), and `registered_at` when it was
    registered, on the registry's clock.
    """

    incarnation: UUID
    schema: PartitionSchema
    live: bool
    registered_at: datetime


@dataclass
class PurgeClaim:
    """
    One purge round's claim on a tombstone.

    The registry fills in `incarnation`, the value the deleted partition's
    records carry in the store. The round sets `any_records_found` before the
    claim ends: whether it found records under the incarnation.
    """

    incarnation: UUID
    any_records_found: bool | None = None


class VectorStorePartitionRegistry(ABC):
    """
    The partition registry of one vector store, identified by the store's name.

    A store name identifies one store among all the stores whose registries
    are kept together, so stores that must stay apart have distinct names.

    A deleted partition's incarnation waits on a queue as a tombstone. A
    write checked as live before the deletion can land in the backend after
    it, so a tombstone's purge starts once a retention, longer than any
    write can be in flight, has passed since the deletion. The tombstone is
    removed when a purge round finds nothing, and its incarnation is not
    minted again before then.
    """

    @abstractmethod
    async def provision(self) -> None:
        """Create the registry's durable resources, idempotently."""
        raise NotImplementedError

    @abstractmethod
    async def register(self, partition_key: str, schema: PartitionSchema) -> UUID:
        """
        Register a new partition, pending, under a freshly minted incarnation.

        The partition key is arbitrated across processes, and the
        incarnation is one no registered or queued partition carries, so the
        new partition starts empty and no purge reclaims its records. The
        partition is pending until `mark_live` marks it, and holds its key
        meanwhile.

        Args:
            partition_key (str): The key of the partition.
            schema (PartitionSchema):
                What the partition is created under: its store's
                dimensions, metric and declared schema.

        Returns:
            UUID: The incarnation the partition's records carry.

        Raises:
            VectorStorePartitionAlreadyExistsError:
                The partition key is taken, by a live or a pending partition.
            VectorStoreAttemptsExhaustedError:
                The registry gave up after repeated attempts to register
                the free partition key failed.
        """
        raise NotImplementedError

    @abstractmethod
    async def mark_live(self, incarnation: UUID) -> bool:
        """
        Mark the partition registered, pending, under an incarnation live.

        Called once the partition's storage is prepared.

        Args:
            incarnation (UUID): The incarnation `register` returned.

        Returns:
            bool:
                Whether a partition was registered, pending, under the
                incarnation, and is now live.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, partition_key: str) -> RegisteredPartition | None:
        """
        Look up the partition registered under a key.

        Args:
            partition_key (str): The key of the partition.

        Returns:
            RegisteredPartition | None:
                The partition, pending or live, or None when there is none.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self, partition_key: str) -> None:
        """
        Unregister the partition under a key and queue its incarnation for purge.

        The partition, pending or live, is unreachable when this returns, and
        purge rounds reclaim its records later. Idempotent.

        Args:
            partition_key (str): The key of the partition.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister_incarnation(self, incarnation: UUID) -> None:
        """
        Unregister the partition registered under an incarnation and queue the incarnation for purge.

        As `unregister`, for a caller holding the incarnation: a partition
        registered since under the same key carries another incarnation and
        stays. Idempotent.

        Args:
            incarnation (UUID): The incarnation the partition is registered under.
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
        `claim.incarnation` in the store and sets `claim.any_records_found`.
        A round that found no records removes the tombstone and frees its
        incarnation. A body that raises is a failed round, and the tombstone
        stays for a later claim; a registry may delay a failed tombstone's
        next claim, or stop claiming one whose rounds keep failing, leaving
        its records in place. A round must be safe to repeat, since a
        registry may hand one tombstone to two purgers.

        Returns:
            AbstractAsyncContextManager[PurgeClaim | None]:
                The claim, or None when no tombstone is due.
        """
        raise NotImplementedError
