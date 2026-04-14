"""
Event memory data types with serialization and formatting.

These types duplicate the internal server types in
memmachine_server.episodic_memory.event_memory.data_types so that
memmachine-client can deserialize and format event memory results
without depending on memmachine-server.
"""

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, tzinfo
from typing import Annotated, Literal
from uuid import UUID

from babel.dates import format_date, format_datetime
from pydantic import (
    BaseModel,
    Field,
    InstanceOf,
    field_serializer,
    field_validator,
)

from .properties import PropertyValue, decode_properties, encode_properties

# Block: leaf content type


class EventMemoryText(BaseModel):
    """Plain text."""

    type: Literal["text"] = "text"
    text: str


class EventMemoryImage(BaseModel):
    """An image."""

    type: Literal["image"] = "image"


class EventMemoryAudio(BaseModel):
    """Audio content."""

    type: Literal["audio"] = "audio"


class EventMemoryVideo(BaseModel):
    """Video content."""

    type: Literal["video"] = "video"


class EventMemoryFileRef(BaseModel):
    """Reference to a file."""

    type: Literal["file_ref"] = "file_ref"


EventMemoryBlock = Annotated[
    EventMemoryText
    | EventMemoryImage
    | EventMemoryAudio
    | EventMemoryVideo
    | EventMemoryFileRef,
    Field(discriminator="type"),
]


# Context: contextual information about the content


class EventMemoryMessageContext(BaseModel):
    """The content is communicated by a source."""

    type: Literal["message"] = "message"
    source: str


class EventMemoryCitationContext(BaseModel):
    """The content is cited from a source."""

    type: Literal["citation"] = "citation"
    source: str
    source_type: str | None = None
    location: str | None = None


EventMemoryContext = Annotated[
    EventMemoryMessageContext | EventMemoryCitationContext,
    Field(discriminator="type"),
]


# Segment


class EventMemorySegment(BaseModel):
    """Snapshot of an event, representing a smaller unit of content."""

    uuid: UUID
    event_uuid: UUID
    index: int
    offset: int
    timestamp: datetime
    context: EventMemoryContext | None = None
    block: EventMemoryBlock
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
        """Hash a segment by its UUID."""
        return hash(self.uuid)


# FormatOptions


class EventMemoryFormatOptions(BaseModel):
    """Options for formatting."""

    datetime_style: Literal["short", "medium", "long", "full"] | None = "full"
    include_time: bool = True
    locale: str = "en_US"
    timezone: InstanceOf[tzinfo] | None = None


# ScoredSegmentContext + QueryResult


class EventMemoryScoredSegmentContext(BaseModel):
    """A segment context anchored on a seed segment, with a score."""

    seed_segment_uuid: UUID
    score: float
    segments: list[EventMemorySegment]

    def to_string(self, format_options: EventMemoryFormatOptions | None = None) -> str:
        """Format this segment context as a string."""
        return format_segment_context(self.segments, format_options=format_options)


class EventMemoryQueryResult(BaseModel):
    """Memory query result, ordered by reranker score."""

    scored_segment_contexts: list[EventMemoryScoredSegmentContext]

    def build_context(self, max_num_segments: int) -> list[EventMemorySegment]:
        """Select the top segments within a budget.

        Iterates contexts in score order, accumulating segments until the
        limit is reached. When a context would exceed the limit, segments
        nearest the seed are prioritized. Deduplicates across contexts.

        Args:
            max_num_segments: The maximum number of segments to return.

        Returns:
            Deduplicated segments ordered chronologically.
        """
        unified: set[EventMemorySegment] = set()

        for scored_context in self.scored_segment_contexts:
            context = scored_context.segments

            if len(unified) >= max_num_segments:
                break
            if (len(unified) + len(context)) <= max_num_segments:
                unified.update(context)
            else:
                seed_index = next(
                    index
                    for index, segment in enumerate(context)
                    if segment.uuid == scored_context.seed_segment_uuid
                )

                for segment in sorted(
                    context,
                    key=lambda s: _seed_proximity(s, context, seed_index),
                ):
                    if len(unified) >= max_num_segments:
                        break
                    unified.add(segment)

        return sorted(
            unified,
            key=lambda segment: (
                segment.timestamp,
                segment.event_uuid,
                segment.index,
                segment.offset,
            ),
        )

    def to_string(
        self,
        max_num_segments: int,
        format_options: EventMemoryFormatOptions | None = None,
    ) -> str:
        """Build context and format as a string."""
        segments = self.build_context(max_num_segments)
        return format_segment_context(segments, format_options=format_options)


# Formatting utilities


def format_segment_context(
    segments: Iterable[EventMemorySegment],
    *,
    format_options: EventMemoryFormatOptions | None = None,
) -> str:
    """Format a sequence of segments as a human-readable string."""
    if format_options is None:
        format_options = EventMemoryFormatOptions()

    context_string = ""
    last_segment: EventMemorySegment | None = None
    accumulated_text = ""
    first = True

    for segment in segments:
        is_continuation = (
            last_segment is not None
            and segment.event_uuid == last_segment.event_uuid
            and segment.index == last_segment.index
        )

        if not is_continuation:
            if not first:
                context_string += json.dumps(accumulated_text) + "\n"
            first = False
            accumulated_text = ""
            context_string += _segment_header(segment, format_options)

        text = _extract_text(segment.block)
        if text is not None:
            accumulated_text += text
        elif not is_continuation:
            context_string += f"[{segment.block.type}]\n"

        last_segment = segment

    if not first:
        context_string += json.dumps(accumulated_text) + "\n"

    return context_string.strip()


def _extract_text(block: EventMemoryBlock) -> str | None:
    """Extract text from a block, if it contains text."""
    match block:
        case EventMemoryText(text=text):
            return text
        case _:
            return None


def _segment_header(
    segment: EventMemorySegment, format_options: EventMemoryFormatOptions
) -> str:
    """Build the header emitted before a segment."""
    datetime_style = format_options.datetime_style
    if datetime_style is None:
        timestamp_prefix = ""
    else:
        formatted_timestamp = _format_timestamp(
            segment.timestamp,
            datetime_style=datetime_style,
            include_time=format_options.include_time,
            locale=format_options.locale,
            timezone=format_options.timezone,
        )
        timestamp_prefix = f"[{formatted_timestamp}] "

    match segment.context:
        case EventMemoryMessageContext(source=source):
            return f"{timestamp_prefix}{source}: "
        case EventMemoryCitationContext(source=source):
            return f"{timestamp_prefix}From '{source}': "
        case _:
            return timestamp_prefix


def _format_timestamp(
    timestamp: datetime,
    *,
    datetime_style: Literal["short", "medium", "long", "full"],
    include_time: bool,
    locale: str,
    timezone: tzinfo | None,
) -> str:
    """Format a timestamp."""
    normalized_timestamp = (
        timestamp.astimezone(timezone) if timezone is not None else timestamp
    )
    if include_time:
        return format_datetime(
            normalized_timestamp, format=datetime_style, locale=locale
        )
    return format_date(normalized_timestamp, format=datetime_style, locale=locale)


def _seed_proximity(
    segment: EventMemorySegment,
    context: list[EventMemorySegment],
    seed_index: int,
) -> float:
    """Score a segment by its proximity to the seed. Lower is closer."""
    idx_offset = context.index(segment) - seed_index
    if idx_offset >= 0:
        # Forward context is more useful than backward.
        return (idx_offset - 0.5) / 2
    return -idx_offset
