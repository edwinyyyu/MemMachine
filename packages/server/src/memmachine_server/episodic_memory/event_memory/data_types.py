"""Data types for EventMemory."""

from collections.abc import Mapping
from datetime import datetime, tzinfo
from typing import (
    Annotated,
    Literal,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    InstanceOf,
    JsonValue,
    TypeAdapter,
    field_serializer,
    field_validator,
)

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)

# Block: leaf content type.
#
# Different Block types do not just represent different modalities;
# they represent different content types,
# each requiring distinct downstream processing logic.
# Plain text, JSON, and HTML may all be processed differently
# despite sharing the text modality.


class TextBlock(BaseModel):
    """Plain text block."""

    block_type: Literal["text"] = "text"
    text: str


Block = Annotated[
    TextBlock,
    Field(discriminator="block_type"),
]


class ProducerContext(BaseModel):
    """The content is produced by a producer."""

    context_type: Literal["producer"] = "producer"
    producer: str


class NullContext(BaseModel):
    """No context is attached."""

    context_type: Literal["null"] = "null"


ContextUnion = ProducerContext | NullContext

Context = Annotated[
    ContextUnion,
    Field(discriminator="context_type"),
]

_CONTEXT_ADAPTER = TypeAdapter(Context | None)
_BLOCK_ADAPTER = TypeAdapter(Block)


def encode_context(context: Context | None) -> dict[str, JsonValue] | None:
    """Encode a context into JSON-compatible data."""
    return _CONTEXT_ADAPTER.dump_python(context, mode="json")


def decode_context(encoded: Mapping[str, JsonValue] | None) -> Context | None:
    """Decode a context from JSON-compatible data."""
    return _CONTEXT_ADAPTER.validate_python(encoded)


def encode_block(block: Block) -> dict[str, JsonValue]:
    """Encode a block into JSON-compatible data."""
    return _BLOCK_ADAPTER.dump_python(block, mode="json")


def decode_block(encoded: Mapping[str, JsonValue]) -> Block:
    """Decode a block from JSON-compatible data."""
    return _BLOCK_ADAPTER.validate_python(encoded)


# Event, Segment, Derivative: core data models for EventMemory.


class Event(BaseModel):
    """Some content, along with its associated context and properties.

    `session_id` names the conversation or stream the event belongs to;
    `source_id` names the entity responsible for it. Both are optional and
    filterable at search: `None` is null, the record carries no key for it,
    and `None` in a typed id list (or `IS NULL`) selects it.
    """

    uuid: UUID
    timestamp: datetime
    session_id: str | None = None
    source_id: str | None = None
    context: Context = Field(default_factory=NullContext)
    blocks: list[Block]
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    @field_validator("properties", mode="before")
    @classmethod
    def _deserialize_properties(cls, v: object) -> object:
        if not isinstance(v, Mapping):
            return v
        try:
            return decode_properties(v)
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


class Segment(BaseModel):
    """Snapshot of an event, representing a smaller unit of content.

    `session_id`, `source_id`, `context`, `timestamp` and `properties` are
    copied verbatim from the event.
    """

    uuid: UUID
    event_uuid: UUID
    index: int
    offset: int
    timestamp: datetime
    session_id: str | None = None
    source_id: str | None = None
    context: Context = Field(default_factory=NullContext)
    block: Block
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    @field_validator("properties", mode="before")
    @classmethod
    def _deserialize_properties(cls, v: object) -> object:
        if not isinstance(v, Mapping):
            return v
        try:
            return decode_properties(v)
        except (TypeError, ValueError):
            # Not type-tagged data (e.g. plain PropertyValue from code).
            return v

    @field_serializer("properties")
    def _serialize_properties(
        self, v: dict[str, PropertyValue]
    ) -> dict[str, dict[str, bool | int | float | str]]:
        return encode_properties(v)

    def __hash__(self) -> int:
        """Hash a segment by its UUID."""
        return hash(self.uuid)


class Derivative(BaseModel):
    """Information derived from a segment.

    `session_id`, `source_id`, `context`, `timestamp` and `properties` are
    copied verbatim from the segment.
    """

    uuid: UUID
    segment_uuid: UUID
    timestamp: datetime
    session_id: str | None = None
    source_id: str | None = None
    context: Context = Field(default_factory=NullContext)
    block: Block
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    @field_validator("properties", mode="before")
    @classmethod
    def _deserialize_properties(cls, v: object) -> object:
        if not isinstance(v, Mapping):
            return v
        try:
            return decode_properties(v)
        except (TypeError, ValueError):
            return v

    @field_serializer("properties")
    def _serialize_properties(
        self, v: dict[str, PropertyValue]
    ) -> dict[str, dict[str, bool | int | float | str]]:
        return encode_properties(v)

    def __hash__(self) -> int:
        """Hash a derivative by its UUID."""
        return hash(self.uuid)


# FormatOptions: options for formatting query result.

# CLDR datetime style. Ordered from compact to verbose.
DateTimeStyle = Literal["short", "medium", "long", "full"]


class FormatOptions(BaseModel):
    """Options for formatting."""

    date_style: DateTimeStyle | None = "full"
    time_style: DateTimeStyle | None = "long"
    locale: str = "en_US"
    timezone: InstanceOf[tzinfo] | None = None


# Results and options.


class SearchHit(BaseModel):
    """One matched derivative with the context window around its segment."""

    score: float
    """Cosine similarity of the matched derivative."""
    seed: int
    """Index in `segments` of the matched segment."""
    segments: list[Segment]
    """The context window, in the store's order."""


class Neighborhood(BaseModel):
    """The segments around an anchor, never the anchor itself: its open neighborhood."""

    before: list[Segment]
    """In order, ending just before the anchor."""
    after: list[Segment]
    """In order, starting just after the anchor."""


class EvictionOptions(BaseModel):
    """How EventMemory trims clusters of near-duplicate derivatives."""

    similarity_threshold: float = Field(ge=-1.0, le=1.0)
    """Cosine similarity at or above which two derivatives are one cluster."""
    search_limit: int = Field(gt=0)
    """Stored neighbors consulted per new derivative."""
    target_size: int = Field(gt=0)
    """A cluster larger than this is trimmed to it."""
