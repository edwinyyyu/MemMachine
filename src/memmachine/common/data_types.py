"""
Common data types for MemMachine.
"""

from typing import Any

from pydantic import BaseModel

from .data_types import Nested

from typing import TypeVar

T = TypeVar("T")
Nested = T | list["Nested"] | dict[str, "Nested"]

class ResourceDefinition(BaseModel):
    type: str
    variant: str
    config: dict[str, Any]
    dependencies: Nested[str] # TODO @edwinyyyu: documentation


class ExternalServiceAPIError(Exception):
    """
    Raised when an API error occurs for an external service.
    """

    pass
