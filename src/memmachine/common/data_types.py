"""
Common data types for MemMachine.
"""

from typing import TypeVar

T = TypeVar("T")
Nested = T | list["Nested"] | dict[str, "Nested"]


class ExternalServiceAPIError(Exception):
    """
    Raised when an API error occurs for an external service.
    """

    pass
