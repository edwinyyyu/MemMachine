"""Client-side API models (Python 3.10+ compatible).

This module contains Pydantic models for the MemMachine client that serialize
to the same JSON format as the server models. The models are kept compatible
with Python 3.10+ by using (str, Enum) instead of StrEnum.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

import regex
from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)
from typing_extensions import Self

UTC = timezone.utc

logger = logging.getLogger(__name__)

DEFAULT_ORG_AND_PROJECT_ID = "universal"


# --------------------------------------------------------------------------------------
# Enums - Using (str, Enum) for Python 3.10 compatibility
# --------------------------------------------------------------------------------------


class MemoryType(str, Enum):
    """Memory type."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class EpisodeType(str, Enum):
    """Episode type."""

    MESSAGE = "message"


class ContentType(str, Enum):
    """Enumeration for the type of content within an Episode."""

    STRING = "string"


class ResourceStatus(str, Enum):
    """Status of a resource."""

    READY = "ready"
    FAILED = "failed"
    PENDING = "pending"


# --------------------------------------------------------------------------------------
# ID Type Aliases and Validators
# --------------------------------------------------------------------------------------

EpisodeIdT = str
SetIdT = str
FeatureIdT = str


class InvalidNameError(ValueError):
    """Custom error for invalid names."""


class InvalidTimestampError(ValueError):
    """Custom error for invalid timestamps."""


def _is_valid_name(v: str) -> str:
    if not regex.fullmatch(r"^[\p{L}\p{N}_:-]+$", v):
        raise InvalidNameError(
            "ID can only contain letters, numbers, underscore, hyphen, "
            f"colon, or Unicode characters, found: '{v}'",
        )
    return v


def _validate_int_compatible(v: str) -> str:
    try:
        int(v)
    except ValueError as e:
        raise ValueError("ID must be int-compatible") from e
    return v


IntCompatibleId = Annotated[str, AfterValidator(_validate_int_compatible), Field(...)]

SafeId = Annotated[str, AfterValidator(_is_valid_name), Field(...)]
SafeIdWithDefault = Annotated[SafeId, Field(default=DEFAULT_ORG_AND_PROJECT_ID)]


# --------------------------------------------------------------------------------------
# Episode Models
# --------------------------------------------------------------------------------------


class EpisodeEntry(BaseModel):
    """Payload used when creating a new episode entry."""

    content: Annotated[
        str, Field(..., description="The content payload of the episode.")
    ]
    producer_id: Annotated[
        str, Field(..., description="Identifier of the episode producer.")
    ]
    producer_role: Annotated[
        str,
        Field(..., description="Role of the producer (e.g., user/assistant/system)."),
    ]
    produced_for_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Identifier of the intended recipient of the episode.",
        ),
    ]
    episode_type: Annotated[
        EpisodeType | None,
        Field(
            default=None,
            description="The type of episode being stored (e.g., message).",
        ),
    ]
    metadata: Annotated[
        dict[str, JsonValue] | None,
        Field(
            default=None, description="Optional metadata associated with the episode."
        ),
    ]
    created_at: Annotated[
        AwareDatetime | None,
        Field(default=None, description="Timestamp when the episode was created."),
    ]


class EpisodeResponse(EpisodeEntry):
    """Episode data returned in search responses."""

    uid: Annotated[
        EpisodeIdT, Field(..., description="Unique identifier for the episode.")
    ]
    score: Annotated[
        float | None,
        Field(default=None, description="Optional relevance score for the episode."),
    ]


class Episode(BaseModel):
    """Episode data returned in list responses."""

    uid: Annotated[
        EpisodeIdT, Field(..., description="Unique identifier for the episode.")
    ]
    content: Annotated[
        str, Field(..., description="The content payload of the episode.")
    ]
    session_key: Annotated[
        str, Field(..., description="Session key associated with the episode.")
    ]
    created_at: Annotated[
        AwareDatetime, Field(..., description="Timestamp when the episode was created.")
    ]

    producer_id: Annotated[
        str, Field(..., description="Identifier of the episode producer.")
    ]
    producer_role: Annotated[
        str,
        Field(..., description="Role of the producer (e.g., user/assistant/system)."),
    ]
    produced_for_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Identifier of the intended recipient of the episode.",
        ),
    ]

    sequence_num: Annotated[
        int, Field(default=0, description="Sequence number within the session.")
    ]

    episode_type: Annotated[
        EpisodeType,
        Field(
            default=EpisodeType.MESSAGE, description="The type of episode being stored."
        ),
    ]
    content_type: Annotated[
        ContentType,
        Field(default=ContentType.STRING, description="Content type of the episode."),
    ]
    filterable_metadata: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Metadata indexed for filtering."),
    ]
    metadata: Annotated[
        dict[str, JsonValue] | None,
        Field(
            default=None, description="Optional metadata associated with the episode."
        ),
    ]

    def __hash__(self) -> int:
        """Hash an episode by its UID."""
        return hash(self.uid)


# --------------------------------------------------------------------------------------
# Semantic Memory Models
# --------------------------------------------------------------------------------------


class SemanticFeature(BaseModel):
    """Semantic memory entry returned in API responses."""

    class Metadata(BaseModel):
        """Storage metadata for a semantic feature, including id and citations."""

        citations: Annotated[
            list[EpisodeIdT] | None,
            Field(
                default=None, description="Episode IDs cited by this semantic feature."
            ),
        ]
        id: Annotated[
            FeatureIdT | None,
            Field(default=None, description="Identifier for the semantic feature."),
        ]
        other: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Additional storage metadata."),
        ]

    set_id: Annotated[
        SetIdT | None,
        Field(default=None, description="Identifier of the semantic set."),
    ]
    category: Annotated[
        str, Field(..., description="Category of the semantic feature.")
    ]
    tag: Annotated[
        str, Field(..., description="Tag associated with the semantic feature.")
    ]
    feature_name: Annotated[
        str, Field(..., description="Name of the semantic feature.")
    ]
    value: Annotated[str, Field(..., description="Value of the semantic feature.")]
    metadata: Annotated[
        Metadata,
        Field(
            default_factory=Metadata,
            description="Storage metadata for the semantic feature.",
        ),
    ]


# --------------------------------------------------------------------------------------
# Project Models
# --------------------------------------------------------------------------------------


class ProjectConfig(BaseModel):
    """Project configuration model."""

    reranker: Annotated[
        str,
        Field(
            default="",
            description="The name of the reranker model to use for this project.",
        ),
    ]

    embedder: Annotated[
        str,
        Field(
            default="",
            description="The name of the embedder model to use for this project.",
        ),
    ]


class _WithOrgAndProj(BaseModel):
    org_id: Annotated[
        SafeIdWithDefault,
        Field(description="The unique identifier of the organization."),
    ]
    project_id: Annotated[
        SafeIdWithDefault,
        Field(description="The identifier of the project."),
    ]


class CreateProjectSpec(BaseModel):
    """Specification model for creating a new project."""

    org_id: Annotated[
        SafeId, Field(description="The unique identifier of the organization.")
    ]
    project_id: Annotated[SafeId, Field(description="The identifier of the project.")]
    description: Annotated[
        str,
        Field(default="", description="A human-readable description of the project."),
    ]
    config: ProjectConfig = Field(
        default_factory=ProjectConfig,
        description="Configuration settings associated with this project.",
    )


class ProjectResponse(BaseModel):
    """Response model returned after project operations."""

    org_id: Annotated[
        SafeId, Field(description="The unique identifier of the organization.")
    ]
    project_id: Annotated[SafeId, Field(description="The identifier of the project.")]
    description: Annotated[
        str,
        Field(default="", description="A human-readable description of the project."),
    ]
    config: Annotated[
        ProjectConfig,
        Field(
            default_factory=ProjectConfig,
            description="Configuration settings associated with this project.",
        ),
    ]


class GetProjectSpec(BaseModel):
    """Specification model for retrieving a project."""

    org_id: Annotated[
        SafeId, Field(description="The unique identifier of the organization.")
    ]
    project_id: Annotated[SafeId, Field(description="The identifier of the project.")]


class EpisodeCountResponse(BaseModel):
    """Response model representing the number of episodes associated with a project."""

    count: Annotated[
        int,
        Field(
            ...,
            description="The total number of episodic memories in the project.",
            ge=0,
        ),
    ]


class DeleteProjectSpec(BaseModel):
    """Specification model for deleting a project."""

    org_id: Annotated[
        SafeId, Field(description="The unique identifier of the organization.")
    ]
    project_id: Annotated[SafeId, Field(description="The identifier of the project.")]


# --------------------------------------------------------------------------------------
# Memory Message Models
# --------------------------------------------------------------------------------------

# Type alias for timestamp input
TimestampInput = datetime | int | float | str | None


class MemoryMessage(BaseModel):
    """Model representing a memory message."""

    content: Annotated[
        str, Field(..., description="The content or text of the message.")
    ]
    producer: Annotated[
        str,
        Field(
            default="user", description="The sender of the message. Defaults to 'user'."
        ),
    ]
    produced_for: Annotated[
        str,
        Field(default="", description="The intended recipient of the message."),
    ]
    timestamp: Annotated[
        datetime,
        Field(
            default_factory=lambda: datetime.now(UTC),
            description="The timestamp when the message was created.",
        ),
    ]
    role: Annotated[
        str,
        Field(default="", description="The role of the message in a conversation."),
    ]
    metadata: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description="Additional metadata associated with the message.",
        ),
    ]
    episode_type: Annotated[
        EpisodeType | None,
        Field(default=None, description="The type of an episode (e.g., 'message')."),
    ]

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: TimestampInput) -> datetime:
        if v is None:
            return datetime.now(UTC)

        # Already a datetime
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=UTC)

        # Unix timestamp (seconds or milliseconds)
        if isinstance(v, (int, float)):
            # Heuristic: > 10^12 is probably milliseconds
            if v > 1_000_000_000_000:
                v = v / 1000
            return datetime.fromtimestamp(v, tz=UTC)

        # String date
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                pass

        raise InvalidTimestampError(f"Unsupported timestamp: {v}")


# --------------------------------------------------------------------------------------
# Add Memory Models
# --------------------------------------------------------------------------------------


class AddMemoriesSpec(_WithOrgAndProj):
    """Specification model for adding memories."""

    types: Annotated[
        list[MemoryType],
        Field(default_factory=list, description="A list of memory types to include."),
    ]

    messages: Annotated[
        list[MemoryMessage],
        Field(min_length=1, description="A list of messages to be added."),
    ]


class AddMemoryResult(BaseModel):
    """Response model for adding memories."""

    uid: Annotated[
        str, Field(..., description="The unique identifier of the memory message.")
    ]


class AddMemoriesResponse(BaseModel):
    """Response model for adding memories."""

    results: Annotated[
        list[AddMemoryResult],
        Field(..., description="The list of results for each added memory message."),
    ]


# --------------------------------------------------------------------------------------
# Search Memory Models
# --------------------------------------------------------------------------------------


class SearchMemoriesSpec(_WithOrgAndProj):
    """Specification model for searching memories."""

    top_k: Annotated[
        int,
        Field(default=10, description="The maximum number of memories to return."),
    ]
    query: Annotated[str, Field(..., description="The natural language query.")]
    filter: Annotated[
        str,
        Field(default="", description="An optional string filter applied to metadata."),
    ]
    expand_context: Annotated[
        int,
        Field(
            default=0,
            description="The number of additional episodes to include around each match.",
        ),
    ]
    score_threshold: Annotated[
        float | None,
        Field(
            default=None, description="The minimum score for a memory to be included."
        ),
    ]
    types: Annotated[
        list[MemoryType],
        Field(default_factory=list, description="A list of memory types to include."),
    ]


class EpisodicSearchShortTermMemory(BaseModel):
    """Short-term episodic memory search results."""

    episodes: Annotated[
        list[EpisodeResponse],
        Field(..., description="Matched short-term episodic entries."),
    ]
    episode_summary: Annotated[
        list[str], Field(..., description="Summaries of matched short-term episodes.")
    ]


class EpisodicSearchLongTermMemory(BaseModel):
    """Long-term episodic memory search results."""

    episodes: Annotated[
        list[EpisodeResponse],
        Field(..., description="Matched long-term episodic entries."),
    ]


class EpisodicSearchResult(BaseModel):
    """Episodic payload returned by `/memories/search`."""

    long_term_memory: Annotated[
        EpisodicSearchLongTermMemory,
        Field(..., description="Long-term episodic search results."),
    ]
    short_term_memory: Annotated[
        EpisodicSearchShortTermMemory,
        Field(..., description="Short-term episodic search results."),
    ]


class SearchResultContent(BaseModel):
    """Payload for SearchResult.content returned by `/memories/search`."""

    model_config = ConfigDict(extra="forbid")

    episodic_memory: Annotated[
        EpisodicSearchResult | None,
        Field(default=None, description="Episodic memory search results."),
    ]
    semantic_memory: Annotated[
        list[SemanticFeature] | None,
        Field(default=None, description="Semantic memory search results."),
    ]


class SearchResult(BaseModel):
    """Response model for memory search results."""

    status: Annotated[
        int,
        Field(default=0, description="The status code of the search operation."),
    ]
    content: Annotated[
        SearchResultContent,
        Field(..., description="The dictionary containing the memory search results."),
    ]


# --------------------------------------------------------------------------------------
# List Memory Models
# --------------------------------------------------------------------------------------


class ListMemoriesSpec(_WithOrgAndProj):
    """Specification model for listing memories."""

    page_size: Annotated[
        int,
        Field(
            default=100,
            description="The maximum number of memories to return per page.",
        ),
    ]
    page_num: Annotated[
        int,
        Field(default=0, description="The zero-based page number to retrieve."),
    ]
    filter: Annotated[
        str,
        Field(default="", description="An optional string filter applied to metadata."),
    ]
    type: Annotated[
        MemoryType | None,
        Field(default=None, description="The specific memory type to list."),
    ]


class ListResultContent(BaseModel):
    """Payload for ListResult.content returned by `/memories/list`."""

    model_config = ConfigDict(extra="forbid")

    episodic_memory: Annotated[
        list[Episode] | None,
        Field(default=None, description="Listed episodic memory entries."),
    ]
    semantic_memory: Annotated[
        list[SemanticFeature] | None,
        Field(default=None, description="Listed semantic memory entries."),
    ]


class ListResult(BaseModel):
    """Response model for memory list results."""

    status: Annotated[
        int,
        Field(default=0, description="The status code of the list operation."),
    ]
    content: Annotated[
        ListResultContent,
        Field(..., description="The dictionary containing the memory list results."),
    ]


# --------------------------------------------------------------------------------------
# Delete Memory Models
# --------------------------------------------------------------------------------------


class DeleteMemoriesSpec(_WithOrgAndProj):
    """Specification model for deleting memories."""

    episodic_memory_uids: Annotated[
        list[EpisodeIdT],
        Field(default=[], description="A list of unique IDs of episodic memories."),
    ]

    semantic_memory_uids: Annotated[
        list[FeatureIdT],
        Field(default=[], description="A list of unique IDs of semantic memories."),
    ]


class DeleteEpisodicMemorySpec(_WithOrgAndProj):
    """Specification model for deleting episodic memories."""

    episodic_id: Annotated[
        SafeId,
        Field(default="", description="The unique ID of the specific episodic memory."),
    ]
    episodic_ids: Annotated[
        list[SafeId],
        Field(default=[], description="A list of unique IDs of episodic memories."),
    ]

    def get_ids(self) -> list[str]:
        """Get a list of episodic IDs to delete."""
        id_set = set(self.episodic_ids)
        if len(self.episodic_id) > 0:
            id_set.add(self.episodic_id)
        id_set = {i.strip() for i in id_set if i and i.strip()}
        return sorted(id_set)

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        """Ensure at least one ID is provided."""
        if len(self.get_ids()) == 0:
            raise ValueError("At least one episodic ID must be provided")
        return self


class DeleteSemanticMemorySpec(_WithOrgAndProj):
    """Specification model for deleting semantic memories."""

    semantic_id: Annotated[
        SafeId,
        Field(default="", description="The unique ID of the specific semantic memory."),
    ]
    semantic_ids: Annotated[
        list[SafeId],
        Field(default=[], description="A list of unique IDs of semantic memories."),
    ]

    def get_ids(self) -> list[str]:
        """Get a list of semantic IDs to delete."""
        id_set = set(self.semantic_ids)
        if len(self.semantic_id) > 0:
            id_set.add(self.semantic_id)
        id_set = {i.strip() for i in id_set if len(i.strip()) > 0}
        return sorted(id_set)

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        """Ensure at least one ID is provided."""
        if len(self.get_ids()) == 0:
            raise ValueError("At least one semantic ID must be provided")
        return self


# --------------------------------------------------------------------------------------
# Configuration Models
# --------------------------------------------------------------------------------------


class ResourceInfo(BaseModel):
    """Information about a configured resource."""

    name: Annotated[
        str, Field(..., description="The unique name/identifier of the resource.")
    ]
    provider: Annotated[
        str, Field(..., description="The provider type of the resource.")
    ]
    status: Annotated[
        ResourceStatus, Field(..., description="The current status of the resource.")
    ]
    error: Annotated[
        str | None,
        Field(default=None, description="The error message if operation failed."),
    ]


class ResourcesStatus(BaseModel):
    """Status of all configured resources."""

    embedders: Annotated[
        list[ResourceInfo],
        Field(
            default_factory=list, description="The status of all configured embedders."
        ),
    ]
    language_models: Annotated[
        list[ResourceInfo],
        Field(
            default_factory=list,
            description="The status of all configured language models.",
        ),
    ]
    rerankers: Annotated[
        list[ResourceInfo],
        Field(
            default_factory=list, description="The status of all configured rerankers."
        ),
    ]
    databases: Annotated[
        list[ResourceInfo],
        Field(
            default_factory=list, description="The status of all configured databases."
        ),
    ]


class GetConfigResponse(BaseModel):
    """Response model for configuration retrieval."""

    resources: Annotated[
        ResourcesStatus,
        Field(..., description="The status of all configured resources."),
    ]


class UpdateLongTermMemorySpec(BaseModel):
    """Partial update for long-term memory configuration."""

    embedder: str | None = Field(default=None, description="The ID of the embedder.")
    reranker: str | None = Field(default=None, description="The ID of the reranker.")
    vector_graph_store: str | None = Field(
        default=None, description="The ID of the vector graph store."
    )


class UpdateShortTermMemorySpec(BaseModel):
    """Partial update for short-term memory configuration."""

    llm_model: str | None = Field(
        default=None, description="The ID of the language model."
    )
    message_capacity: int | None = Field(
        default=None, gt=0, description="The maximum message capacity."
    )


class UpdateEpisodicMemorySpec(BaseModel):
    """Partial update for episodic memory configuration."""

    long_term_memory: UpdateLongTermMemorySpec | None = Field(
        default=None, description="Partial update for long-term memory settings."
    )
    short_term_memory: UpdateShortTermMemorySpec | None = Field(
        default=None, description="Partial update for short-term memory settings."
    )
    long_term_memory_enabled: bool | None = Field(
        default=None, description="Whether long-term episodic memory is enabled."
    )
    short_term_memory_enabled: bool | None = Field(
        default=None, description="Whether short-term episodic memory is enabled."
    )
    enabled: bool | None = Field(
        default=None, description="Whether episodic memory is enabled."
    )


class UpdateSemanticMemorySpec(BaseModel):
    """Partial update for semantic memory configuration."""

    enabled: bool | None = Field(
        default=None, description="Whether semantic memory is enabled."
    )
    database: str | None = Field(default=None, description="The ID of the database.")
    llm_model: str | None = Field(
        default=None, description="The ID of the language model."
    )
    embedding_model: str | None = Field(
        default=None, description="The ID of the embedder."
    )
    ingestion_trigger_messages: int | None = Field(
        default=None, gt=0, description="Number of messages triggering ingestion."
    )
    ingestion_trigger_age_seconds: int | None = Field(
        default=None, gt=0, description="Max age of uningested messages."
    )


class UpdateMemoryConfigSpec(BaseModel):
    """Specification for updating memory configuration."""

    episodic_memory: Annotated[
        UpdateEpisodicMemorySpec | None,
        Field(
            default=None,
            description="Partial update for episodic memory configuration.",
        ),
    ]
    semantic_memory: Annotated[
        UpdateSemanticMemorySpec | None,
        Field(
            default=None,
            description="Partial update for semantic memory configuration.",
        ),
    ]


class UpdateMemoryConfigResponse(BaseModel):
    """Response model for memory configuration update."""

    success: Annotated[bool, Field(..., description="Whether the operation succeeded.")]
    message: Annotated[
        str, Field(..., description="Status message describing the result.")
    ]


class AddEmbedderSpec(BaseModel):
    """Specification for adding a new embedder."""

    name: Annotated[
        str, Field(..., description="Unique name/identifier for the embedder.")
    ]
    provider: Annotated[str, Field(..., description="The embedder provider type.")]
    config: Annotated[
        dict[str, Any], Field(..., description="Provider-specific configuration.")
    ]


class AddLanguageModelSpec(BaseModel):
    """Specification for adding a new language model."""

    name: Annotated[
        str, Field(..., description="Unique name/identifier for the language model.")
    ]
    provider: Annotated[
        str, Field(..., description="The language model provider type.")
    ]
    config: Annotated[
        dict[str, Any], Field(..., description="Provider-specific configuration.")
    ]


class UpdateResourceResponse(BaseModel):
    """Response model for resource update operations."""

    success: Annotated[bool, Field(..., description="Whether the operation succeeded.")]
    status: Annotated[
        ResourceStatus,
        Field(..., description="Current status of the resource after operation."),
    ]
    error: Annotated[
        str | None,
        Field(default=None, description="Error message if operation failed."),
    ]


class DeleteResourceResponse(BaseModel):
    """Response model for resource deletion operations."""

    success: Annotated[bool, Field(..., description="Whether the deletion succeeded.")]
    message: Annotated[
        str, Field(..., description="Status message describing the result.")
    ]


# --------------------------------------------------------------------------------------
# Error and Version Models
# --------------------------------------------------------------------------------------


class RestErrorModel(BaseModel):
    """Model representing an error response."""

    code: Annotated[
        int, Field(..., description="The http status code if the operation failed.")
    ]
    message: Annotated[str, Field(..., description="A descriptive error message.")]
    internal_error: Annotated[
        str, Field(..., description="The real error that triggered the exception.")
    ]
    exception: Annotated[str, Field(..., description="The exception details.")]
    trace: Annotated[
        str, Field(..., description="The stack trace of the exception if available.")
    ]


class Version(BaseModel):
    """Model representing version information."""

    server_version: Annotated[
        str, Field(..., description="The version of the MemMachine server.")
    ]
    client_version: Annotated[
        str, Field(..., description="The version of the MemMachine client.")
    ]

    def __str__(self) -> str:
        """Return the version as a string."""
        return "\n".join(
            [
                f"server: {self.server_version}",
                f"client: {self.client_version}",
            ]
        )


# --------------------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------------------

__all__ = [
    "AddEmbedderSpec",
    "AddLanguageModelSpec",
    "AddMemoriesResponse",
    # Add Memory Models
    "AddMemoriesSpec",
    "AddMemoryResult",
    "ContentType",
    "CreateProjectSpec",
    "DeleteEpisodicMemorySpec",
    # Delete Memory Models
    "DeleteMemoriesSpec",
    "DeleteProjectSpec",
    "DeleteResourceResponse",
    "DeleteSemanticMemorySpec",
    "Episode",
    "EpisodeCountResponse",
    # Episode Models
    "EpisodeEntry",
    "EpisodeResponse",
    "EpisodeType",
    "EpisodicSearchLongTermMemory",
    "EpisodicSearchResult",
    "EpisodicSearchShortTermMemory",
    "GetConfigResponse",
    "GetProjectSpec",
    # Errors
    "InvalidNameError",
    "InvalidTimestampError",
    # List Memory Models
    "ListMemoriesSpec",
    "ListResult",
    "ListResultContent",
    # Memory Message Models
    "MemoryMessage",
    # Enums
    "MemoryType",
    # Project Models
    "ProjectConfig",
    "ProjectResponse",
    # Configuration Models
    "ResourceInfo",
    "ResourceStatus",
    "ResourcesStatus",
    # Error and Version Models
    "RestErrorModel",
    # Search Memory Models
    "SearchMemoriesSpec",
    "SearchResult",
    "SearchResultContent",
    # Semantic Models
    "SemanticFeature",
    "UpdateEpisodicMemorySpec",
    "UpdateLongTermMemorySpec",
    "UpdateMemoryConfigResponse",
    "UpdateMemoryConfigSpec",
    "UpdateResourceResponse",
    "UpdateSemanticMemorySpec",
    "UpdateShortTermMemorySpec",
    "Version",
]
