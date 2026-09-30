"""
Base classes for a vector store whose collections a collection registry arbitrates.

The registry mints each collection's incarnation and arbitrates creation,
deletion and reclamation across processes. A collection is registered
pending, its storage is prepared, and it is marked live; only a live
collection is opened. The backend holds records, each carrying its
collection's incarnation, and a subclass decides how: it prepares a new
collection's storage, builds a handle for one collection, and purges a
deleted incarnation's records.
"""

import asyncio
import logging
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .collection_registry import RegisteredCollection, VectorStoreCollectionRegistry
from .data_types import (
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    VectorStoreCollectionHandleStaleError,
)
from .utils import require_identifiers
from .vector_store import VectorStore, VectorStoreCollection

logger = logging.getLogger(__name__)

# Attempts before open-or-create gives up, a second apart: a name that stays
# taken by a collection that never becomes live is one left pending, or
# something else is wrong.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10
_OPEN_OR_CREATE_RETRY_DELAY_SECONDS = 1


class RegistryBackedVectorStoreCollection(VectorStoreCollection):
    """A handle bound to one incarnation of a logical collection.

    For subclasses: `_incarnation` is the incarnation the handle is bound to,
    `_tracker` times each operation, and `_fence()` raises once the collection
    has been deleted.
    """

    def __init__(
        self,
        *,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
        tracker: OperationTracker,
        get_registered_collection: Callable[
            [str, str], Awaitable[RegisteredCollection | None]
        ],
    ) -> None:
        """Initialize with the incarnation the handle is bound to, and the registry's lookup."""
        self._namespace = namespace
        self._name = name
        self._incarnation = incarnation
        self._config = config
        self._tracker = tracker
        self._get_registered_collection = get_registered_collection

    async def _fence(self) -> None:
        """Raise if this handle's collection has been deleted.

        The collection registered under the handle's name carries the
        handle's incarnation until it is deleted; one created again under
        the name carries another. Called before every operation, and again
        after a write, so a write that raced the deletion raises instead of
        reporting success. Such a write may still have landed; the purge
        reclaims it.
        """
        registered = await self._get_registered_collection(self._namespace, self._name)
        if registered is None or registered.incarnation != self._incarnation:
            raise VectorStoreCollectionHandleStaleError(self._namespace, self._name)

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config


class RegistryBackedVectorStoreParams(BaseModel):
    """
    Parameters for a RegistryBackedVectorStore.

    Attributes:
        collection_registry (VectorStoreCollectionRegistry):
            Registry of the store's collections, shared by the stores, in any
            process, whose clients connect to the same database, and by no
            other store. Started by the caller.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    collection_registry: InstanceOf[VectorStoreCollectionRegistry] = Field(
        ...,
        description=(
            "Registry of the store's collections, shared by the stores, in any "
            "process, whose clients connect to the same database, and by no "
            "other store. Started by the caller"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class RegistryBackedVectorStore[CollectionT: RegistryBackedVectorStoreCollection](
    VectorStore
):
    """A vector store whose collections a VectorStoreCollectionRegistry arbitrates.

    Any process connected to the same database, with the same registry, may
    serve any collection. A collection is pending until its storage is
    prepared: meanwhile `open_collection` answers None,
    `open_or_create_collection` waits for it, and creating its name raises
    VectorStoreCollectionAlreadyExistsError. One a crash left pending is
    deleted like any other.

    For subclasses: `_collection_registry` is the registry and `_tracker` times
    each operation, and a subclass implements `_prepare_storage`,
    `_build_collection_handle` and `_purge_round`.
    """

    def __init__(
        self, params: RegistryBackedVectorStoreParams, *, metrics_prefix: str
    ) -> None:
        """Initialize with the registry and the prefix of the store's metrics."""
        super().__init__()
        self._collection_registry = params.collection_registry
        self._tracker = OperationTracker(params.metrics_factory, prefix=metrics_prefix)

    @override
    async def startup(self) -> None:
        # The caller owns the client's and the registry's lifecycles.
        pass

    @override
    async def shutdown(self) -> None:
        # The caller owns the client's and the registry's lifecycles.
        pass

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        require_identifiers(namespace, name)
        async with self._tracker("create_collection"):
            # The registry's primary key decides a creation race. The
            # collection stays pending, invisible, until its storage is
            # prepared; one deleted meanwhile was created, then deleted.
            incarnation = await self._collection_registry.register(
                namespace, name, config
            )
            await self._prepare_storage_or_unregister(
                namespace, name, config, incarnation
            )
            await self._collection_registry.mark_live(incarnation)

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> CollectionT:
        require_identifiers(namespace, name)
        async with self._tracker("open_or_create_collection"):
            # Read-then-create, retried: a pending collection is another
            # creator's (open it once it is live), losing the create means
            # another creator took the name meanwhile, and losing the mark
            # means a deleter removed this one while its storage was prepared
            # (create again).
            pending = False
            for attempt in range(_MAX_OPEN_OR_CREATE_ATTEMPTS):
                if attempt:
                    await asyncio.sleep(_OPEN_OR_CREATE_RETRY_DELAY_SECONDS)
                registered = await self._collection_registry.get(namespace, name)
                pending = registered is not None and not registered.live
                if registered is not None:
                    if registered.config != config:
                        raise VectorStoreCollectionConfigMismatchError(
                            namespace, name, registered.config, config
                        )
                    if registered.live:
                        return self._build_collection_handle(
                            namespace, name, registered
                        )
                    continue
                try:
                    incarnation = await self._collection_registry.register(
                        namespace, name, config
                    )
                except VectorStoreCollectionAlreadyExistsError:
                    continue
                await self._prepare_storage_or_unregister(
                    namespace, name, config, incarnation
                )
                if await self._collection_registry.mark_live(incarnation):
                    return self._build_collection_handle(
                        namespace,
                        name,
                        RegisteredCollection(
                            incarnation=incarnation, config=config, live=True
                        ),
                    )
            if pending:
                raise VectorStoreAttemptsExhaustedError(
                    f"Collection ({namespace!r}, {name!r}) stayed pending through "
                    f"{_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts to open it; if its "
                    "creation was abandoned, delete it to create it again"
                )
            raise VectorStoreAttemptsExhaustedError(
                f"Opening or creating collection ({namespace!r}, {name!r}) made "
                f"no progress after {_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts"
            )

    async def _prepare_storage_or_unregister(
        self,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
        incarnation: UUID,
    ) -> None:
        """Prepare a pending collection's storage, unregistering it if that raises.

        A collection the registry cannot unregister then stays pending until
        it is deleted.
        """
        try:
            await self._prepare_storage(namespace, config, incarnation)
        except Exception:
            try:
                await self._collection_registry.unregister_incarnation(incarnation)
            except Exception:
                logger.exception(
                    "Could not unregister collection (%r, %r) after its "
                    "storage preparation failed; it stays pending until "
                    "deleted",
                    namespace,
                    name,
                )
            raise

    @override
    async def open_collection(self, *, namespace: str, name: str) -> CollectionT | None:
        require_identifiers(namespace, name)
        registered = await self._collection_registry.get(namespace, name)
        if registered is None or not registered.live:
            return None
        return self._build_collection_handle(namespace, name, registered)

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        require_identifiers(namespace, name)
        async with self._tracker("delete_collection"):
            # One registry transaction: the collection is unreachable when
            # it commits, and its records wait on the queue for the purge.
            await self._collection_registry.unregister(namespace, name)

    @override
    async def purge_deleted_collections(self) -> bool:
        # One purge round per call, on the tombstone that came due first.
        async with (
            self._tracker("purge_deleted_collections"),
            self._collection_registry.claim_purgeable_incarnation() as claim,
        ):
            if claim is None:
                return False
            claim.any_records_found = await self._purge_round(
                claim.namespace, claim.config, claim.incarnation
            )
            return True

    @abstractmethod
    async def _prepare_storage(
        self,
        namespace: str,
        config: VectorStoreCollectionConfig,
        incarnation: UUID,
    ) -> None:
        """
        Prepare the storage a newly registered collection needs.

        The collection is registered, pending, under the incarnation, and is
        marked live once this returns. Its storage may be shared with the
        other collections of its namespace and configuration, such as a
        native collection they are all stored in, and may be its own. Shared
        storage is prepared by any number of processes at once, so preparing
        it must be idempotent and safe to race; what serves one namespace and
        configuration serves no other. Whatever a failed or interrupted call
        leaves must be recoverable: completed by a later call for the same
        namespace and configuration, or reclaimed by the purge rounds of the
        incarnation once its pending collection is deleted.

        Args:
            namespace (str):
                Namespace of the collection being created.
            config (VectorStoreCollectionConfig):
                Configuration of the collection being created.
            incarnation (UUID):
                The incarnation the collection's records will carry.

        Raises:
            Exception:
                Whatever the backend raises. The collection is then
                unregistered, or, if the registry cannot unregister it, stays
                pending until it is deleted.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_collection_handle(
        self, namespace: str, name: str, registered: RegisteredCollection
    ) -> CollectionT:
        """
        Build a handle bound to a registered collection's incarnation.

        The collection is live: its storage was prepared before it was
        marked so.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.
            registered (RegisteredCollection):
                The collection's incarnation and configuration.

        Returns:
            CollectionT:
                A handle bound to the incarnation, whose operations raise
                once the collection is deleted.
        """
        raise NotImplementedError

    @abstractmethod
    async def _purge_round(
        self, namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> bool:
        """
        Delete records carrying a deleted incarnation, and return whether it found any.

        Runs under a purge claim, after the tombstone's retention. Returning
        False removes the tombstone, so it must return False only when no
        record under the incarnation remains; a round may delete some of the
        records and return True, to be run again. It may delete storage that
        holds only the incarnation's records, which a failed or interrupted
        preparation may have left part-made. It must be safe to repeat, and
        to run on two purgers at once. Storage that is gone holds no record.

        Args:
            namespace (str):
                Namespace of the deleted collection.
            config (VectorStoreCollectionConfig):
                Configuration of the deleted collection; with the namespace,
                it locates the collection's records.
            incarnation (UUID):
                The incarnation the deleted collection's records carry.

        Returns:
            bool:
                Whether the round found records under the incarnation.

        Raises:
            Exception:
                Whatever the backend raises. The round then counts as
                failed, and the tombstone is claimed again after a backoff.
        """
        raise NotImplementedError
