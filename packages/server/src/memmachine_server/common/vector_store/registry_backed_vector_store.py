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
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .collection_registry import RegisteredCollection, VectorStoreCollectionRegistry
from .data_types import (
    QueryResult,
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    VectorStoreCollectionHandleStaleError,
    VectorStoreCollectionPendingError,
)
from .utils import (
    require_declared_types,
    require_dimensions,
    require_identifiers,
    require_valid_query_vector,
    require_valid_score_threshold,
    validate_filter,
)
from .vector_store import VectorStore, VectorStoreCollection

logger = logging.getLogger(__name__)

# Attempts open-or-create makes, _OPEN_OR_CREATE_RETRY_DELAY_SECONDS apart,
# before it gives up on a collection that stays pending or a name it keeps
# losing.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10
_OPEN_OR_CREATE_RETRY_DELAY_SECONDS = 1


class RegistryBackedVectorStoreCollection(VectorStoreCollection):
    """A handle bound to one incarnation of a logical collection.

    Each operation checks its inputs and that the collection is still live:
    `upsert` before and after its backend call, `query` before it, and
    `delete` after it.

    For subclasses: `_incarnation` is the incarnation the handle is bound to,
    and a subclass implements the backend calls `_upsert`, `_query` and
    `_delete`.
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
        the name carries another. A check after a write makes one that raced
        the deletion raise instead of reporting success; whatever such a
        write landed, the purge reclaims.
        """
        registered = await self._get_registered_collection(self._namespace, self._name)
        if registered is None or registered.incarnation != self._incarnation:
            raise VectorStoreCollectionHandleStaleError(self._namespace, self._name)

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        async with self._tracker("upsert"):
            records = list(records)
            for record in records:
                require_declared_types(
                    record.properties, self._config.indexed_properties_schema
                )
                require_dimensions(record.vector, self._config.vector_dimensions)
            await self._fence()
            if not records:
                return
            await self._upsert(records)
            await self._fence()

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            for query_vector in query_vectors:
                require_valid_query_vector(query_vector, self._config.vector_dimensions)
            require_valid_score_threshold(score_threshold)
            if property_filter is not None and not validate_filter(property_filter):
                raise ValueError("Filter contains an invalid property key")
            await self._fence()
            if not query_vectors:
                return []
            if limit <= 0:
                return [QueryResult(matches=[]) for _ in query_vectors]
            return await self._query(
                query_vectors,
                limit=limit,
                score_threshold=score_threshold,
                property_filter=property_filter,
            )

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        async with self._tracker("delete"):
            # One check, after: a delete adds nothing a purge must reclaim,
            # and a stale handle's delete reaches only its own incarnation.
            record_uuids = list(record_uuids)
            if record_uuids:
                await self._delete(record_uuids)
            await self._fence()

    @abstractmethod
    async def _upsert(self, records: list[Record]) -> None:
        """
        Write records to the backend under the handle's incarnation.

        Called between two liveness checks, with at least one record, each
        already checked against the collection's configuration. A record
        replaces the one with its UUID. The records are durable when it
        returns; a call that raises may have written some of them.

        Args:
            records (list[Record]): The records to write.

        Raises:
            Exception: Whatever the backend raises; the upsert raises it.
        """
        raise NotImplementedError

    @abstractmethod
    async def _query(
        self,
        query_vectors: list[list[float]],
        *,
        limit: int,
        score_threshold: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryResult]:
        """
        Search the handle's incarnation's records for each query vector.

        Called after a liveness check, with at least one query vector and a
        limit of at least 1, the vectors, threshold and filter already
        checked.

        Args:
            query_vectors (list[list[float]]): The vectors to search for.
            limit (int): The most matches to return per query vector.
            score_threshold (float | None):
                The score a match must reach, or None for any.
            property_filter (FilterExpr | None):
                The condition a match's properties must meet, or None.

        Returns:
            list[QueryResult]:
                One result per query vector, in order, its matches best
                first.

        Raises:
            Exception: Whatever the backend raises; the query raises it.
        """
        raise NotImplementedError

    @abstractmethod
    async def _delete(self, record_uuids: list[UUID]) -> None:
        """
        Delete the handle's incarnation's records with these UUIDs.

        Called before a liveness check, with at least one UUID, possibly
        through a handle whose collection was deleted. A UUID with no record
        under the incarnation is skipped. The deletions are durable when it
        returns; a call that raises may have deleted some of the records.

        Args:
            record_uuids (list[UUID]): The UUIDs of the records to delete.

        Raises:
            Exception: Whatever the backend raises; the delete raises it.
        """
        raise NotImplementedError


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
    prepared: meanwhile opening it raises VectorStoreCollectionPendingError,
    `open_or_create_collection` waits a bounded time for it, and creating its
    name raises VectorStoreCollectionAlreadyExistsError. A collection a crash
    left pending stays pending until `delete_collection` deletes it.

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
        # Unregistrations after a failed preparation, held until done so the
        # garbage collector cannot drop one whose creation was cancelled.
        self._unregistrations: set[asyncio.Task[None]] = set()

    @override
    async def startup(self) -> None:
        # The caller owns the registry's lifecycle and that of any client a
        # subclass is given.
        pass

    @override
    async def shutdown(self) -> None:
        # The caller owns the registry's lifecycle and that of any client a
        # subclass is given.
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
            # The registry decides a creation race. The collection stays
            # pending until its storage is prepared; if it is deleted
            # meanwhile, mark_live finds nothing to mark, and the creation
            # counts as one followed by a deletion.
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
            registered: RegisteredCollection | None = None
            for attempt in range(_MAX_OPEN_OR_CREATE_ATTEMPTS):
                if attempt:
                    await asyncio.sleep(_OPEN_OR_CREATE_RETRY_DELAY_SECONDS)
                registered = await self._collection_registry.get(namespace, name)
                if registered is not None:
                    if registered.config != config:
                        raise VectorStoreCollectionConfigMismatchError(
                            namespace, name, registered.config, config
                        )
                    if registered.live:
                        return self._build_collection_handle(
                            namespace, name, registered.incarnation, config
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
                        namespace, name, incarnation, config
                    )
            # The last lookup found the collection pending.
            if registered is not None:
                raise VectorStoreCollectionPendingError(
                    namespace, name, registered.registered_at
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
        """Prepare a pending collection's storage, unregistering it if that raises or is cancelled.

        A collection the registry cannot unregister then stays pending until
        it is deleted.
        """
        try:
            await self._prepare_storage(namespace, config, incarnation)
        except BaseException:
            # Shielded, so a cancelled creation still frees the name.
            unregistration = asyncio.create_task(
                self._collection_registry.unregister_incarnation(incarnation)
            )
            self._unregistrations.add(unregistration)
            unregistration.add_done_callback(self._unregistrations.discard)
            try:
                await asyncio.shield(unregistration)
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
        if registered is None:
            return None
        if not registered.live:
            raise VectorStoreCollectionPendingError(
                namespace, name, registered.registered_at
            )
        return self._build_collection_handle(
            namespace, name, registered.incarnation, registered.config
        )

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        require_identifiers(namespace, name)
        async with self._tracker("delete_collection"):
            # The collection is unreachable once unregister returns, and its
            # incarnation awaits the purge.
            await self._collection_registry.unregister(namespace, name)

    @override
    async def purge_deleted_collections(self) -> bool:
        # One purge round per call, on a due tombstone.
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

        The collection is registered as pending under the incarnation, and is
        marked live once this returns. Its storage may be its own or shared
        with the other collections of its namespace and configuration. Shared
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
        self,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
    ) -> CollectionT:
        """
        Build a handle bound to a live collection's incarnation.

        The collection's storage was prepared before it was marked live.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.
            incarnation (UUID):
                The incarnation the collection is registered under.
            config (VectorStoreCollectionConfig):
                The configuration the collection was created with.

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
        to run on two purgers at once. A round that finds the incarnation's
        storage missing returns False.

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
                failed, and the tombstone stays for a later claim.
        """
        raise NotImplementedError
