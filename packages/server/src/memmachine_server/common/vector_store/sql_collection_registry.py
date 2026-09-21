"""
A collection registry in SQL, for the stores whose backend holds only points.

Qdrant and Milvus have no transactions, unique constraints or conditional
writes, so a catalog kept inside them cannot arbitrate two processes
creating, deleting or reclaiming the same logical collection. The registry
lives in the deployment's relational database instead: one table pair per
backend kind, shared by every store on that kind of backend, with a row per
live logical collection keyed by backend, namespace and name, whose
incarnation is the value every point of that life carries, and a purge
queue of dead incarnations claimed oldest-first. Creation is an insert the
primary key arbitrates, deletion is one transaction, and a purge claim is a
row lock the database hands to one purger at a time.

A queue entry is the dead incarnation's tombstone. The backend holds the
points, and a write the registry read as live can land there after the
purge that followed the deletion, so one purge cannot be the last: the
entry stays through purge rounds until a round finds nothing, then
through a retention measured on the database clock, then through one
more round that finds nothing again. Only then is it removed, and until
then the incarnation is never re-minted.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import NamedTuple
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Uuid,
    delete,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from .data_types import (
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
)
from .utils import _IDENTIFIER_MAX_BYTES

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10
_BACKEND_ID_MAX_LENGTH = 255


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected; retry with a fresh incarnation."""


class RegisteredCollection(NamedTuple):
    """A live collection: the incarnation its points carry and the configuration it was created with."""

    incarnation: UUID
    config: VectorStoreCollectionConfig


@dataclass
class PurgeClaim:
    """A claimed tombstone: the incarnation to purge, where its points are, and what the round found.

    The purger sets `found` before the claim ends: True when the backend
    still held points under the incarnation, False when it held none.
    """

    incarnation: UUID
    namespace: str
    name: str
    config: VectorStoreCollectionConfig
    clean_at: datetime | None
    found: bool | None = None


class SqlCollectionRegistry:
    """The collection registry of one store, in a table pair shared by its backend kind.

    `table_prefix` names the backend kind: every store on that kind of
    backend shares `{prefix}_ct` and `{prefix}_gc`, and `backend`, the id
    of the configured backend the store is on, is part of every key, so
    stores on different backends keep apart in one database.
    `tombstone_retention` is how long a dead incarnation's entry outlives
    the first purge round that found nothing; it must exceed, by orders
    of magnitude, the longest a write to the backend can be in flight.
    """

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        table_prefix: str,
        backend: str,
        tombstone_retention: timedelta,
    ) -> None:
        """Bind to the registry tables of one backend kind, for one backend."""
        self._engine = engine
        self._backend = backend
        self._tombstone_retention = tombstone_retention
        metadata = MetaData()
        self._collections = Table(
            f"{table_prefix}_ct",
            metadata,
            Column("backend", String(_BACKEND_ID_MAX_LENGTH), primary_key=True),
            Column("namespace", String(_IDENTIFIER_MAX_BYTES), primary_key=True),
            Column("name", String(_IDENTIFIER_MAX_BYTES), primary_key=True),
            Column("incarnation", Uuid, nullable=False, unique=True),
            # The configuration the collection was created with: the handle
            # is built from it, and open-or-create compares against it.
            Column("config_json", JSON, nullable=False),
        )
        self._purge_queue = Table(
            f"{table_prefix}_gc",
            metadata,
            Column("incarnation", Uuid, primary_key=True),
            Column("backend", String(_BACKEND_ID_MAX_LENGTH), nullable=False),
            Column("namespace", String(_IDENTIFIER_MAX_BYTES), nullable=False),
            Column("name", String(_IDENTIFIER_MAX_BYTES), nullable=False),
            # The configuration names the native collection the points are in.
            Column("config_json", JSON, nullable=False),
            Column("enqueued_at", DateTime(timezone=True), nullable=False),
            # When a purge round last found nothing under the incarnation;
            # cleared by a round that finds something. The entry is removed
            # by a round that finds nothing a retention after this.
            Column("clean_at", DateTime(timezone=True), nullable=True),
            Index(f"{table_prefix}_gc__ba_ea", "backend", "enqueued_at"),
        )
        self._metadata = metadata

    async def provision(self) -> None:
        """Create the registry tables, idempotently.

        Two provisioners racing on an empty database can both find the
        tables absent and both issue the DDL; the loser's fails, and its
        second pass finds the winner's tables and creates nothing. Any
        other failure fails the second pass too.
        """
        try:
            await self._create_tables()
        except DBAPIError:
            await self._create_tables()

    async def _create_tables(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(self._metadata.create_all)

    async def create(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> UUID:
        """Register a new collection under a freshly minted incarnation.

        The primary key arbitrates the (namespace, name) across processes;
        the incarnation's unique constraint and the in-transaction queue
        check reject an incarnation that is live or still awaiting purge,
        so no points can be adopted by, or reclaimed out from under, a new
        collection.

        Raises:
            VectorStoreCollectionAlreadyExistsError: The (namespace, name) is taken.
            VectorStoreAttemptsExhaustedError:
                Every minted incarnation was rejected for another reason.
        """
        attempts = 0
        while True:
            incarnation = uuid4()
            try:
                await self._insert(namespace, name, incarnation, config)
            except _RegistryInsertRejectedError as err:
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Creating collection ({namespace!r}, {name!r}) on backend "
                        f"{self._backend!r} made no progress after "
                        f"{_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue
            return incarnation

    async def _insert(
        self,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
    ) -> None:
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    insert(self._collections).values(
                        backend=self._backend,
                        namespace=namespace,
                        name=name,
                        incarnation=incarnation,
                        config_json=config.model_dump(mode="json"),
                    )
                )
                queued = (
                    await connection.execute(
                        select(self._purge_queue.c.incarnation).where(
                            self._purge_queue.c.incarnation == incarnation
                        )
                    )
                ).scalar_one_or_none()
                if queued is not None:
                    logger.warning(
                        "Incarnation %s minted for collection (%r, %r) on backend %r "
                        "collides with garbage awaiting purge; re-minting",
                        incarnation,
                        namespace,
                        name,
                        self._backend,
                    )
                    raise _RegistryInsertRejectedError(str(incarnation))
        except IntegrityError as err:
            if await self.get(namespace, name) is not None:
                raise VectorStoreCollectionAlreadyExistsError(namespace, name) from err
            logger.warning(
                "Registry insert for collection (%r, %r) on backend %r with "
                "incarnation %s failed and no row exists under the key; "
                "retrying with a fresh incarnation",
                namespace,
                name,
                self._backend,
                incarnation,
            )
            raise _RegistryInsertRejectedError(str(incarnation)) from err

    async def get(self, namespace: str, name: str) -> RegisteredCollection | None:
        """The live collection under the (namespace, name), or None."""
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        self._collections.c.incarnation,
                        self._collections.c.config_json,
                    ).where(
                        self._collections.c.backend == self._backend,
                        self._collections.c.namespace == namespace,
                        self._collections.c.name == name,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        return RegisteredCollection(
            incarnation=row.incarnation,
            config=VectorStoreCollectionConfig.model_validate(row.config_json),
        )

    async def is_live(self, incarnation: UUID) -> bool:
        """Whether a collection is still registered under this incarnation."""
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(self._collections.c.name).where(
                        self._collections.c.incarnation == incarnation
                    )
                )
            ).scalar_one_or_none()
        return row is not None

    async def delete(self, namespace: str, name: str) -> None:
        """Unregister the collection and queue its incarnation for purge, in one transaction.

        Idempotent: no row under the key is the no-op case. The collection
        is unreachable as soon as the transaction commits; its points are
        reclaimed by the purge rounds that claim the entry, its tombstone.
        """
        # The DELETE goes first: it takes the row's write lock, so racing
        # deleters serialize on it and the loser deletes nothing, on
        # PostgreSQL and on SQLite alike.
        async with self._engine.begin() as connection:
            row = (
                await connection.execute(
                    delete(self._collections)
                    .where(
                        self._collections.c.backend == self._backend,
                        self._collections.c.namespace == namespace,
                        self._collections.c.name == name,
                    )
                    .returning(
                        self._collections.c.incarnation,
                        self._collections.c.config_json,
                    )
                )
            ).one_or_none()
            if row is None:
                return
            await connection.execute(
                insert(self._purge_queue).values(
                    incarnation=row.incarnation,
                    backend=self._backend,
                    namespace=namespace,
                    name=name,
                    config_json=row.config_json,
                    enqueued_at=func.now(),
                )
            )

    @asynccontextmanager
    async def claim_oldest(self) -> AsyncIterator[PurgeClaim | None]:
        """Claim this backend's oldest tombstone due for a purge round.

        Yields None when no tombstone is due: the queue is empty, or every
        entry had a clean round less than the retention ago. The claim is a
        row lock held for the body: on PostgreSQL a concurrent purger skips
        the locked entry and takes the next; on SQLite the writers
        serialize at the end of the round, so a doubly claimed entry costs
        a repeated, idempotent round and never a missed one.

        The body purges and sets `found`. A round that found points keeps
        the entry and clears `clean_at`, so rounds continue; a round that
        found none stamps `clean_at` the first time and removes the entry
        when it is the round due after the retention. A body that raises
        leaves the entry as it was, for a later claim.
        """
        async with self._engine.begin() as connection:
            database_now = (
                await connection.execute(
                    select(func.now(type_=DateTime(timezone=True)))
                )
            ).scalar_one()
            row = (
                await connection.execute(
                    select(
                        self._purge_queue.c.incarnation,
                        self._purge_queue.c.namespace,
                        self._purge_queue.c.name,
                        self._purge_queue.c.config_json,
                        self._purge_queue.c.clean_at,
                    )
                    .where(
                        self._purge_queue.c.backend == self._backend,
                        or_(
                            self._purge_queue.c.clean_at.is_(None),
                            self._purge_queue.c.clean_at
                            <= database_now - self._tombstone_retention,
                        ),
                    )
                    .order_by(self._purge_queue.c.enqueued_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).one_or_none()
            if row is None:
                yield None
                return
            claim = PurgeClaim(
                incarnation=row.incarnation,
                namespace=row.namespace,
                name=row.name,
                config=VectorStoreCollectionConfig.model_validate(row.config_json),
                clean_at=row.clean_at,
            )
            yield claim
            if claim.found is None:
                raise RuntimeError(
                    f"Purge round for incarnation {claim.incarnation} on backend "
                    f"{self._backend!r} ended without reporting what it found"
                )
            entry = self._purge_queue.c.incarnation == claim.incarnation
            if claim.found:
                await connection.execute(
                    update(self._purge_queue).where(entry).values(clean_at=None)
                )
            elif claim.clean_at is None:
                await connection.execute(
                    update(self._purge_queue).where(entry).values(clean_at=func.now())
                )
            else:
                await connection.execute(delete(self._purge_queue).where(entry))
