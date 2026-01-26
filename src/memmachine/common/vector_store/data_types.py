"""Data types for vector store."""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

# Types that can be used as property values.
PropertyValue = bool | int | float | str | datetime


@dataclass(kw_only=True)
class Record:
    """A record in the vector store."""

    uuid: UUID
    vector: list[float]
    properties: dict[str, PropertyValue] = field(default_factory=dict)

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
