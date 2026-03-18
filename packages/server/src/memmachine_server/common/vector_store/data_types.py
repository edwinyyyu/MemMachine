"""Data types for vector store."""

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, field_serializer

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
    SimilarityMetric,
)


class CollectionConfig(BaseModel):
    """Configuration for a logical collection in a vector store."""

    vector_dimensions: int
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    properties_schema: Mapping[str, type[PropertyValue]] | None = None

    @field_serializer("properties_schema")
    def _serialize_properties_schema(
        self, v: Mapping[str, type[PropertyValue]] | None
    ) -> dict[str, str] | None:
        if v is None:
            return None
        return {k: PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[val] for k, val in v.items()}


class CollectionAlreadyExistsError(Exception):
    """Raised when creating a collection that already exists."""

    def __init__(self, namespace: str, name: str) -> None:
        """Initialize with the namespace and name of the existing collection."""
        self.namespace = namespace
        self.name = name
        super().__init__(f"Collection ({namespace!r}, {name!r}) already exists.")


class CollectionConfigurationMismatchError(Exception):
    """Raised when opening a collection with a different configuration than it was created with."""

    def __init__(
        self,
        namespace: str,
        name: str,
        existing_config: CollectionConfig,
        requested_config: CollectionConfig,
    ) -> None:
        """Initialize with the namespace, name, and configurations."""
        self.namespace = namespace
        self.name = name
        self.existing_config = existing_config
        self.requested_config = requested_config
        super().__init__(
            f"Collection ({namespace!r}, {name!r}) already exists with a different configuration. "
            f"Existing config: {existing_config.model_dump_json()}, "
            f"requested config: {requested_config.model_dump_json()}."
        )


@dataclass(kw_only=True)
class Record:
    """A record in the vector store."""

    uuid: UUID
    vector: list[float] | None = None
    properties: dict[str, PropertyValue] | None = None

    def __eq__(self, other: object) -> bool:
        """Compare nodes by UID, vector, and properties."""
        if not isinstance(other, Record):
            return False
        return (
            self.uuid == other.uuid
            and self.vector == other.vector
            and self.properties == other.properties
        )

    def __hash__(self) -> int:
        """Hash a record by its UID."""
        return hash(self.uuid)


@dataclass(kw_only=True)
class QueryMatch:
    """A single vector store query match."""

    score: float
    record: Record


@dataclass(kw_only=True)
class QueryResult:
    """Result of a vector store query."""

    matches: list[QueryMatch]
