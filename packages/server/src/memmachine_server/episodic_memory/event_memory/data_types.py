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


class RewriteContext(BaseModel):
    """Carries a separate text to embed/derive while keeping the segment
    text intact for display.

    Used when the segment has been rewritten or transformed for display
    purposes but we want the embedding/derivation step to see a
    different (e.g. verbatim-augmented or multi-form) text. Display
    formatting treats this like NullContext (no header prefix).
    """

    context_type: Literal["rewrite"] = "rewrite"
    text_to_embed: str


class SurroundingEvent(BaseModel):
    """A neighboring event the producer of the current event was exposed to."""

    producer: str
    text: str


class SurroundingEventsContext(BaseModel):
    """Producer context augmented with the immediately surrounding events.

    Equivalent to `ProducerContext` for display and embedding purposes
    (the producer is prepended to the text), but additionally carries
    the events occurring before and after the current event in the same
    sequence. Segmenters and derivers that benefit from neighbor
    awareness (e.g. resolving second-person addressees, anaphora, or
    references to prior turns) can read `before` and `after`.
    """

    context_type: Literal["surrounding_events"] = "surrounding_events"
    producer: str
    before: list[SurroundingEvent] = Field(default_factory=list)
    after: list[SurroundingEvent] = Field(default_factory=list)


class RawSegmentEventContext(BaseModel):
    """Experimental: bundles producer, before-neighbor events, and the
    current event's full text.

    Used by experimental raw-segmenter + LLM-deriver paradigms where the
    segment's block.text stays a raw chunk shown to the answerer, while
    the deriver generates retrieval-optimized derivatives using full
    neighbor + current-event context. Display formatting matches
    ProducerContext (``{producer}: <text>``).
    """

    context_type: Literal["raw_segment_event"] = "raw_segment_event"
    producer: str
    before: list[SurroundingEvent] = Field(default_factory=list)
    current_event_text: str


class DecoupledRetrievalContext(BaseModel):
    """Decouples a segment's retrieval representations from its display
    text.

    ``block.text`` stays the verbatim text shown to the answering model
    -- a raw conversation chunk -- and keeps its distinct role. The
    retrieval channels instead score separate texts carried here:

    - ``text_to_embed`` is embedded for semantic (vector) retrieval.
    - ``text_to_score_bm25`` is concatenated across the segments of a
      retrieved context and scored by BM25 lexical retrieval.

    Both are typically clean, reference-resolved rewrites, so neither
    retrieval channel sees the conversational vocatives, greetings, and
    filler that the raw display text carries (those false-match
    name-bearing queries when BM25 scores raw turns). Display formatting
    matches ``ProducerContext`` (``{producer}: <block.text>``).
    """

    context_type: Literal["decoupled_retrieval"] = "decoupled_retrieval"
    producer: str
    text_to_embed: str
    text_to_score_bm25: str


ContextUnion = (
    ProducerContext
    | NullContext
    | RewriteContext
    | SurroundingEventsContext
    | RawSegmentEventContext
    | DecoupledRetrievalContext
)

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
    """An event."""

    uuid: UUID
    timestamp: datetime
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
