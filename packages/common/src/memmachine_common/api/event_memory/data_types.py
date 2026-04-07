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
    """Options for formatting query results."""

    timezone: InstanceOf[tzinfo] | None = None
    show_timezone_label: bool = True


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

            timestamp = _format_timestamp(segment.timestamp, format_options)

            match segment.context:
                case EventMemoryMessageContext(source=source):
                    context_string += f"{timestamp} {source}: "
                case EventMemoryCitationContext(source=source):
                    context_string += f"{timestamp} From '{source}': "
                case _:
                    context_string += f"{timestamp} "

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


def _format_timestamp(
    timestamp: datetime, format_options: EventMemoryFormatOptions
) -> str:
    """Format a timestamp as a bracketed date/time string."""
    display_timestamp = (
        timestamp.astimezone(format_options.timezone)
        if format_options.timezone is not None
        else timestamp
    )
    date_str = display_timestamp.date().strftime("%A, %B %d, %Y")
    time_str = display_timestamp.time().strftime("%I:%M %p")
    if format_options.show_timezone_label:
        tz_label = _format_timezone(display_timestamp)
        if tz_label:
            time_str += " " + tz_label
    return f"[{date_str} at {time_str}]"


def _format_timezone(timestamp: datetime) -> str:
    """Format the timezone of a datetime as a UTC offset string."""
    offset = timestamp.utcoffset()
    if offset is None:
        return ""
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if seconds:
        return f"UTC{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


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
