"""Data types for EventMemory.

The `Event` data model and its companions (`Context`, `Block`, and the
encode/decode helpers) now live in `memmachine_core.event.data_types` — the
canonical core location. They are re-exported here so existing EventMemory
imports keep working. `Segment`, `Derivative`, and the query-result/formatting
types remain EventMemory-internal and are defined below.
"""

from collections.abc import Mapping
from datetime import datetime, tzinfo
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    InstanceOf,
    field_serializer,
    field_validator,
)

from memmachine_core.common.data_types import PropertyValue
from memmachine_core.common.properties_json import (
    decode_properties,
    encode_properties,
)
from memmachine_core.event.data_types import (
    Block,
    Context,
    ContextUnion,
    Event,
    NullContext,
    ProducerContext,
    TextBlock,
    decode_block,
    decode_context,
    encode_block,
    encode_context,
)

__all__ = [
    "Block",
    "Context",
    "ContextUnion",
    "DateTimeStyle",
    "Derivative",
    "Event",
    "FormatOptions",
    "NullContext",
    "ProducerContext",
    "QueryResult",
    "ScoredSegmentContext",
    "Segment",
    "TextBlock",
    "decode_block",
    "decode_context",
    "encode_block",
    "encode_context",
]


# Segment, Derivative: EventMemory index internals.


class Segment(BaseModel):
    """Snapshot of an event, representing a smaller unit of content."""

    uuid: UUID
    event_uuid: UUID
    index: int
    offset: int
    timestamp: datetime
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
    """Information derived from a segment."""

    uuid: UUID
    segment_uuid: UUID
    timestamp: datetime
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


# QueryResult: the result of a memory query.


class ScoredSegmentContext(BaseModel):
    """A segment context anchored on a seed segment, with a score."""

    score: float
    seed_segment_uuid: UUID
    segments: list[Segment]


class QueryResult(BaseModel):
    """Memory query result, ordered by reranker score."""

    scored_segment_contexts: list[ScoredSegmentContext]
