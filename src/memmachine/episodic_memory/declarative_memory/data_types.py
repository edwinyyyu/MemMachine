"""Data structures for declarative episodic memory entries."""

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import JsonValue

from memmachine.common.data_types import PropertyValue


@dataclass(kw_only=True)
class MessageContent:
    """Content formatted as a message (source: text)."""

    type: str = field(default="message", init=False)
    text: str
    source: str = ""

    def format(self) -> str:
        """Format as a message (source: text)."""
        if self.source:
            return f"{self.source}: {self.text}"
        return self.text


@dataclass(kw_only=True)
class TextContent:
    """Content formatted as plain text."""

    type: str = field(default="text", init=False)
    text: str
    source: str = ""

    def format(self) -> str:
        """Format as plain text."""
        return self.text


@dataclass(kw_only=True)
class ConversationContent:
    """Content from a conversation with full producer attribution."""

    type: str = field(default="conversation", init=False)
    text: str
    source: str = ""
    producer_id: str = ""
    producer_role: str = ""
    produced_for_id: str | None = None

    def format(self) -> str:
        """Format as a message (source: text)."""
        if self.source:
            return f"{self.source}: {self.text}"
        return self.text


EpisodeContent = MessageContent | TextContent | ConversationContent


@dataclass(kw_only=True)
class Episode:
    """A single episodic memory entry."""

    uid: str
    timestamp: datetime
    content: EpisodeContent
    properties: dict[str, PropertyValue] = field(
        default_factory=dict,
    )
    extra: dict[str, JsonValue] = field(
        default_factory=dict,
    )

    def __eq__(self, other: object) -> bool:
        """Compare episodes by UID."""
        if not isinstance(other, Episode):
            return False
        return (
            self.uid == other.uid
            and self.timestamp == other.timestamp
            and self.content == other.content
            and self.properties == other.properties
            and self.extra == other.extra
        )

    def __hash__(self) -> int:
        """Hash an episode by its UID."""
        return hash(self.uid)


@dataclass(kw_only=True)
class Derivative:
    """A derived episodic memory linked to a source episode."""

    uid: str
    timestamp: datetime
    content: str
    properties: dict[str, PropertyValue] = field(
        default_factory=dict,
    )
    extra: dict[str, JsonValue] = field(
        default_factory=dict,
    )

    def __eq__(self, other: object) -> bool:
        """Compare derivatives by UID."""
        if not isinstance(other, Derivative):
            return False
        return (
            self.uid == other.uid
            and self.timestamp == other.timestamp
            and self.content == other.content
            and self.properties == other.properties
            and self.extra == other.extra
        )

    def __hash__(self) -> int:
        """Hash a derivative by its UID."""
        return hash(self.uid)


_MANGLE_PROPERTY_KEY_PREFIX = "filterable_"


def mangle_property_key(key: str) -> str:
    """Prefix property keys with the mangling token."""
    return _MANGLE_PROPERTY_KEY_PREFIX + key


def demangle_property_key(mangled_key: str) -> str:
    """Remove the mangling prefix from a property key."""
    return mangled_key.removeprefix(_MANGLE_PROPERTY_KEY_PREFIX)


def is_mangled_property_key(candidate_key: str) -> bool:
    """Check whether the provided key contains the mangling prefix."""
    return candidate_key.startswith(_MANGLE_PROPERTY_KEY_PREFIX)
