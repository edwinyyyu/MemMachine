"""Data types for AttributeMemory ingestion.

Ported from :mod:`memmachine_server.episodic_memory.event_memory.data_types`
(kept only the types up to and including :class:`Event`; post-Event
types like ``Segment`` / ``Derivative`` / ``QueryResult`` aren't used
by the attribute-memory pipeline).
"""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import (
    Annotated,
    Literal,
    cast,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
)

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)

# Block: leaf content type


class Text(BaseModel):
    """Plain text."""

    type: Literal["text"] = "text"
    text: str


class Image(BaseModel):
    """An image."""

    type: Literal["image"] = "image"


class Audio(BaseModel):
    """Audio content."""

    type: Literal["audio"] = "audio"


class Video(BaseModel):
    """Video content."""

    type: Literal["video"] = "video"


class FileRef(BaseModel):
    """Reference to a file."""

    type: Literal["file_ref"] = "file_ref"


Block = Annotated[
    Text | Image | Audio | Video | FileRef,
    Field(discriminator="type"),
]


class MessageContext(BaseModel):
    """The content is communicated by a source."""

    type: Literal["message"] = "message"
    source: str


class CitationContext(BaseModel):
    """The content is cited from a source."""

    type: Literal["citation"] = "citation"
    source: str
    source_type: str | None = None
    location: str | None = None


ContextUnion = MessageContext | CitationContext

Context = Annotated[
    ContextUnion,
    Field(discriminator="type"),
]


# Body: top-level event payload


class Content(BaseModel):
    """A list of item blocks with optional context."""

    type: Literal["content"] = "content"
    context: Context | None = None
    items: list[Block]


class ReadFile(BaseModel):
    """Request the system to read a file."""

    type: Literal["read_file"] = "read_file"
    file: FileRef


Body = Annotated[
    Content | ReadFile,
    Field(discriminator="type"),
]


# Event: the top-level ingestion input


class Event(BaseModel):
    """An event."""

    uuid: UUID
    timestamp: datetime
    body: Body
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("properties", mode="before")
    @classmethod
    def _deserialize_properties(cls, v: object) -> object:
        if not isinstance(v, Mapping):
            return v
        try:
            return decode_properties(cast(Mapping[str, JsonValue], v))
        except (TypeError, ValueError):
            return v

    @field_serializer("properties")
    def _serialize_properties(
        self, v: dict[str, PropertyValue]
    ) -> dict[str, dict[str, bool | int | float | str]]:
        return encode_properties(v)

    def __hash__(self) -> int:
        """Hash an event by its UUID."""
        return hash(self.uuid)


# ---------------------------------------------------------------------- #
# LLM-mutation commands
# ---------------------------------------------------------------------- #


class CommandType(StrEnum):
    """Mutation actions emitted by the LLM against the profile."""

    ADD = "add"
    DELETE = "delete"


class Command(BaseModel):
    """One LLM-emitted mutation against the attribute profile.

    The LLM never sees attribute UUIDs — a ``DELETE`` is matched by
    ``(topic, category, attribute, value)`` at apply time, and an
    ``ADD`` creates a fresh attribute with a new UUID.
    """

    command: CommandType
    category: str
    attribute: str
    value: str

    @field_validator("category", "attribute", "value", mode="after")
    @classmethod
    def _strip_null_bytes(cls, v: str) -> str:
        if "\x00" in v:
            return v.replace("\x00", "")
        return v
