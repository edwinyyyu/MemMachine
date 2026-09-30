"""
Base classes for a vector store whose collections a collection registry arbitrates.

A logical collection is the records carrying its incarnation inside a native
collection shared by the logical collections of one namespace and
configuration. The registry mints incarnations and arbitrates creation,
deletion and reclamation across processes; a subclass provides the native
operations.
"""

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

# Consecutive lost creation races before open-or-create gives up: every
# retry requires another process to have created and then deleted the
# collection in between, so this depth means something else is wrong.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10


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
        is_live: Callable[[UUID], Awaitable[bool]],
    ) -> None:
        """Initialize with the incarnation the handle is bound to."""
        self._namespace = namespace
        self._name = name
        self._incarnation = incarnation
        self._config = config
        self._tracker = tracker
        self._is_live = is_live

    async def _fence(self) -> None:
        """Raise if this handle's collection has been deleted.

        Called before every operation, and again after a write, so a write
        that raced the deletion raises instead of reporting success. Such a
        write may still have landed; the purge reclaims it.
        """
        if not await self._is_live(self._incarnation):
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
    serve any collection.

    For subclasses: `_collection_registry` is the registry and `_tracker` times
    each operation, and a subclass implements `_create_native_collection`,
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
            # The native collection first, the registry row last, so a crash
            # between the two leaves at worst an empty native collection,
            # which the next creation of the same configuration uses. The
            # registry's primary key decides a creation race.
            await self._create_native_collection(namespace, config)
            await self._collection_registry.register(namespace, name, config)

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
            attempts = 0
            # Read-then-create, retried: losing the create means a racing
            # creator won (open its row), and finding no row after losing
            # means a racing deleter removed the winner (create again).
            while True:
                registered = await self._collection_registry.get(namespace, name)
                if registered is not None:
                    if registered.config != config:
                        raise VectorStoreCollectionConfigMismatchError(
                            namespace, name, registered.config, config
                        )
                    return self._build_collection_handle(namespace, name, registered)
                await self._create_native_collection(namespace, config)
                try:
                    incarnation = await self._collection_registry.register(
                        namespace, name, config
                    )
                except VectorStoreCollectionAlreadyExistsError as err:
                    attempts += 1
                    if attempts >= _MAX_OPEN_OR_CREATE_ATTEMPTS:
                        raise VectorStoreAttemptsExhaustedError(
                            f"Opening or creating collection ({namespace!r}, "
                            f"{name!r}) made no progress after "
                            f"{_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts"
                        ) from err
                    continue
                return self._build_collection_handle(
                    namespace,
                    name,
                    RegisteredCollection(incarnation=incarnation, config=config),
                )

    @override
    async def open_collection(self, *, namespace: str, name: str) -> CollectionT | None:
        require_identifiers(namespace, name)
        registered = await self._collection_registry.get(namespace, name)
        if registered is None:
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
            return claim.any_records_found

    @abstractmethod
    async def _create_native_collection(
        self, namespace: str, config: VectorStoreCollectionConfig
    ) -> None:
        """
        Ensure the native collection of a namespace and configuration exists.

        The native collection holds the records of every collection with
        this namespace and configuration, and of no other, so two
        configurations never share one. It is ensured before each
        registration, by any number of processes at once, so this must be
        idempotent and safe to race, and must complete a native collection
        a failed call left part-built.

        Args:
            namespace (str):
                Namespace of the collection being created.
            config (VectorStoreCollectionConfig):
                Configuration of the collection being created.

        Raises:
            Exception:
                Whatever the backend raises. The collection is then not
                registered, and the next creation of the configuration
                completes what this call left.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_collection_handle(
        self, namespace: str, name: str, registered: RegisteredCollection
    ) -> CollectionT:
        """
        Build a handle bound to a registered collection's incarnation.

        The collection's native collection was ensured before it was
        registered.

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
        records and return True, to be run again. It must be safe to repeat,
        and to run on two purgers at once. The native collection may be
        gone, in which case no record remains.

        Args:
            namespace (str):
                Namespace of the deleted collection.
            config (VectorStoreCollectionConfig):
                Configuration of the deleted collection; with the namespace,
                it names the native collection.
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
