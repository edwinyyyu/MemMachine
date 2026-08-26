"""SQLAlchemy-backed collection registry implementation."""

import re
from typing import override

from pydantic import (
    BaseModel,
    Field,
    InstanceOf,
    JsonValue,
    TypeAdapter,
    field_validator,
)
from sqlalchemy import (
    JSON,
    Column,
    Insert,
    MetaData,
    String,
    Table,
    delete,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.pool import StaticPool

from memmachine_server.common.vector_store.utils import validate_identifier

from .collection_registry import (
    CollectionAlreadyRegisteredError,
    CollectionRegistry,
    CollectionRegistryEntry,
)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")

_TABLE_NAME_PREFIX = "collection_registry_"

# Registry names become SQL identifier components:
# prefix (20 bytes) + name (32 bytes) stays within
# PostgreSQL's 63-byte identifier limit.
_REGISTRY_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_MAX_REGISTRY_NAME_BYTES = 32

_MAX_KEY_LENGTH = 255

_SUPPORTED_DIALECTS = ("postgresql", "sqlite")

_ENTRY_ADAPTER: TypeAdapter[CollectionRegistryEntry] = TypeAdapter(
    CollectionRegistryEntry
)


class SQLAlchemyCollectionRegistryParams(BaseModel):
    """
    Parameters for SQLAlchemyCollectionRegistry.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine.
            Must use a PostgreSQL or SQLite dialect.
        name (str):
            Name identifying which vector store this is the registry for.
            Determines the registry's table name,
            so it must match `[a-z0-9_]+`
            and be at most 32 bytes.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="Async SQLAlchemy engine with a PostgreSQL or SQLite dialect",
    )
    name: str = Field(
        ...,
        description=(
            "Name identifying which vector store this is the registry for; "
            "determines the registry's table name, "
            "so it must match [a-z0-9_]+ and be at most 32 bytes"
        ),
    )

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, engine: AsyncEngine) -> AsyncEngine:
        assert not isinstance(engine.pool, StaticPool), (
            "Engine uses StaticPool, which shares one connection across sessions. "
            "Use a multi-connection pool instead."
        )
        db = engine.url.database
        if engine.dialect.name == "sqlite" and (db is None or db == ":memory:"):
            raise ValueError(
                "Engine uses ephemeral SQLite, where each connection gets a separate database. "
                "Use a file path instead."
            )
        if engine.dialect.name not in _SUPPORTED_DIALECTS:
            raise ValueError(
                f"Engine dialect {engine.dialect.name!r} is not supported. "
                f"Supported dialects: {', '.join(_SUPPORTED_DIALECTS)}."
            )
        return engine

    @field_validator("name")
    @classmethod
    def _validate_name(cls, name: str) -> str:
        if not _REGISTRY_NAME_RE.match(name):
            raise ValueError(
                f"Registry name {name!r} must match [a-z0-9_]+ "
                "(lowercase alphanumeric and underscores only)"
            )
        if len(name.encode()) > _MAX_REGISTRY_NAME_BYTES:
            raise ValueError(
                f"Registry name {name!r} must be at most "
                f"{_MAX_REGISTRY_NAME_BYTES} bytes"
            )
        return name


class SQLAlchemyCollectionRegistry(CollectionRegistry):
    """
    Asynchronous SQLAlchemy-backed implementation of CollectionRegistry.

    Each registry owns a dedicated table named after the registry
    (`collection_registry_<name>`),
    so registries sharing a database never collide
    and an instance can only reach its own registry.

    Registration is atomic across processes:
    the table's primary key is the arbiter,
    via plain INSERT for register
    and native INSERT .. ON CONFLICT DO NOTHING for get_or_register.
    """

    def __init__(self, params: SQLAlchemyCollectionRegistryParams) -> None:
        """Initialize the collection registry with the provided parameters."""
        super().__init__()
        self._engine = params.engine
        self._name = params.name

        self._table = Table(
            _TABLE_NAME_PREFIX + params.name,
            MetaData(),
            Column("key", String(_MAX_KEY_LENGTH), primary_key=True),
            Column("entry", _JSON_AUTO, nullable=False),
        )

    @staticmethod
    def _key(namespace: str, name: str) -> str:
        """
        Build the storage key for a collection.

        The separator is outside the identifier charset ([a-z0-9_]+),
        so distinct (namespace, name) pairs can never produce the same key.
        """
        return f"{namespace}/{name}"

    @staticmethod
    def _validate_collection_identifiers(namespace: str, name: str) -> None:
        """Validate a collection's namespace and name."""
        if not validate_identifier(namespace):
            raise ValueError(
                f"Namespace {namespace!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )
        if not validate_identifier(name):
            raise ValueError(
                f"Name {name!r} must match [a-z0-9_]+ and be at most 32 bytes"
            )

    def _insert_ignoring_conflicts(self, key: str, dumped_entry: JsonValue) -> Insert:
        """Build a dialect-native INSERT that ignores primary key conflicts."""
        if self._engine.dialect.name == "postgresql":
            return (
                postgresql_insert(self._table)
                .values(key=key, entry=dumped_entry)
                .on_conflict_do_nothing()
            )
        return (
            sqlite_insert(self._table)
            .values(key=key, entry=dumped_entry)
            .on_conflict_do_nothing()
        )

    @override
    async def startup(self) -> None:
        """Idempotently create the registry's table."""
        async with self._engine.begin() as connection:
            await connection.run_sync(self._table.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        """No-op; engine lifecycle is managed externally."""

    @override
    async def register(
        self, *, namespace: str, name: str, entry: CollectionRegistryEntry
    ) -> None:
        """Atomically register a collection in the registry table."""
        SQLAlchemyCollectionRegistry._validate_collection_identifiers(namespace, name)
        dumped_entry = _ENTRY_ADAPTER.dump_python(entry, mode="json")
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    insert(self._table).values(
                        key=SQLAlchemyCollectionRegistry._key(namespace, name),
                        entry=dumped_entry,
                    )
                )
        except IntegrityError as err:
            raise CollectionAlreadyRegisteredError(namespace, name) from err

    @override
    async def get(self, *, namespace: str, name: str) -> CollectionRegistryEntry | None:
        """Get the stored entry for a collection."""
        SQLAlchemyCollectionRegistry._validate_collection_identifiers(namespace, name)
        async with self._engine.connect() as connection:
            result = await connection.execute(
                select(self._table.c.entry).where(
                    self._table.c.key
                    == SQLAlchemyCollectionRegistry._key(namespace, name)
                )
            )
            row = result.one_or_none()
        if row is None:
            return None
        return _ENTRY_ADAPTER.validate_python(row.entry)

    @override
    async def get_or_register(
        self, *, namespace: str, name: str, entry: CollectionRegistryEntry
    ) -> tuple[CollectionRegistryEntry, bool]:
        """Atomically register a collection if it is not registered."""
        SQLAlchemyCollectionRegistry._validate_collection_identifiers(namespace, name)

        stored_entry = await self.get(namespace=namespace, name=name)
        if stored_entry is not None:
            return stored_entry, False

        dumped_entry = _ENTRY_ADAPTER.dump_python(entry, mode="json")
        async with self._engine.begin() as connection:
            result = await connection.execute(
                self._insert_ignoring_conflicts(
                    SQLAlchemyCollectionRegistry._key(namespace, name),
                    dumped_entry,
                )
            )
            registered = result.rowcount == 1
        if registered:
            return _ENTRY_ADAPTER.validate_python(dumped_entry), True

        # Lost a concurrent registration race; the winner's entry is stored.
        stored_entry = await self.get(namespace=namespace, name=name)
        if stored_entry is None:
            raise RuntimeError(
                f"Collection ({namespace!r}, {name!r}) in registry {self._name!r} "
                "was deregistered during concurrent registration"
            )
        return stored_entry, False

    @override
    async def deregister(self, *, namespace: str, name: str) -> None:
        """Idempotently deregister a collection."""
        SQLAlchemyCollectionRegistry._validate_collection_identifiers(namespace, name)
        async with self._engine.begin() as connection:
            await connection.execute(
                delete(self._table).where(
                    self._table.c.key
                    == SQLAlchemyCollectionRegistry._key(namespace, name)
                )
            )
