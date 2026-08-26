"""Common data types for MemMachine."""

from datetime import datetime
from enum import Enum, IntEnum
from typing import Final

PropertyValue = bool | int | float | str | datetime
"""Type for stored property values."""

PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME: Final[dict[type[PropertyValue], str]] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    datetime: "datetime",
}

PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE: Final[dict[str, type[PropertyValue]]] = {
    v: k for k, v in PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME.items()
}

FilterValue = bool | int | float | str | datetime | list[int] | list[str]
"""Type for filter expression values (includes list types for IN clauses)."""

OrderedValue = int | float | datetime
"""Type for values that can be ordered/sorted."""


class ConcurrencyScope(IntEnum):
    """
    Widest safe deployment boundary for concurrent resource management.

    The scope within which concurrent instances of a component may safely
    manage the same resources. Ordered by breadth, so the effective scope
    of a composed system is the minimum of its parts' scopes.
    """

    PROCESS = 1
    """Concurrent management is safe only within a single process."""

    MACHINE = 2
    """Concurrent management is safe across processes on one machine."""

    CLUSTER = 3
    """Concurrent management is safe across machines."""


class SimilarityMetric(Enum):
    """Similarity metrics supported by embedding operations."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

    @property
    def higher_is_better(self) -> bool:
        """Whether a higher score indicates a better match."""
        return self in (SimilarityMetric.COSINE, SimilarityMetric.DOT)


class ExternalServiceAPIError(Exception):
    """Raised when an API error occurs for an external service."""
