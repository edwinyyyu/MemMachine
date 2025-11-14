"""
Common data types for MemMachine.
"""

from enum import Enum

FilterablePropertyValue = bool | int | str
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class SimilarityMetric(Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class ExternalServiceAPIError(Exception):
    """
    Raised when an API error occurs for an external service.
    """

    pass
