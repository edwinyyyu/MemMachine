"""Abstract interface for storing semantic configuration data."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID

from memmachine_server.semantic_memory.semantic_model import (
    SemanticCategory,
    SetIdT,
    SetTypeEntry,
)


@runtime_checkable
class SemanticConfigStorage(Protocol):
    """Contract for persisting and retrieving semantic memory configuration."""

    async def startup(self) -> None: ...

    async def delete_all(self) -> None: ...

    async def set_setid_config(
        self,
        *,
        set_id: SetIdT,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None: ...

    @dataclass(frozen=True)
    class Config:
        """Configuration values tied to a specific set identifier."""

        embedder_name: str | None
        llm_name: str | None
        disabled_categories: list[str] | None
        categories: list[SemanticCategory]

    async def get_setid_config(
        self,
        *,
        set_id: SetIdT,
    ) -> Config: ...

    async def register_set_id_set_type(
        self,
        *,
        set_id: SetIdT,
        set_type_id: UUID,
    ) -> None: ...

    @dataclass(frozen=True)
    class Category:
        """Represents a semantic category as stored in the database."""

        id: UUID
        name: str
        prompt: str
        description: str | None

    async def get_category(
        self,
        *,
        category_id: UUID,
    ) -> Category | None: ...

    async def get_category_set_ids(
        self,
        *,
        category_id: UUID,
    ) -> list[SetIdT]: ...

    async def create_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> UUID: ...

    async def clone_category(
        self,
        *,
        category_id: UUID,
        new_set_id: SetIdT,
        new_name: str,
    ) -> UUID: ...

    async def delete_category(
        self,
        *,
        category_id: UUID,
    ) -> None: ...

    async def add_disabled_category_to_setid(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None: ...

    async def remove_disabled_category_from_setid(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None: ...

    async def create_set_type_category(
        self,
        *,
        set_type_id: UUID,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> UUID: ...

    async def get_set_type_categories(
        self,
        *,
        set_type_id: UUID,
    ) -> list[SemanticCategory]: ...

    @dataclass(frozen=True)
    class Tag:
        """Represents a tag associated with a category as represented in the database."""

        id: UUID
        name: str
        description: str

    async def get_tag(
        self,
        *,
        tag_id: UUID,
    ) -> Tag | None: ...

    async def add_tag(
        self,
        *,
        category_id: UUID,
        tag_name: str,
        description: str,
    ) -> UUID: ...

    async def update_tag(
        self,
        *,
        tag_id: UUID,
        tag_name: str,
        tag_description: str,
    ) -> None: ...

    async def delete_tag(
        self,
        *,
        tag_id: UUID,
    ) -> None: ...

    async def add_set_type_id(
        self,
        *,
        org_id: str,
        org_level_set: bool = False,
        metadata_tags: list[str],
        name: str | None = None,
        description: str | None = None,
    ) -> UUID: ...

    async def list_set_type_ids(self, *, org_id: str) -> list[SetTypeEntry]: ...

    async def delete_set_type_id(self, *, set_type_id: UUID) -> None: ...
