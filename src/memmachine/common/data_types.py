"""
Common data types for MemMachine.
"""

from typing import Any, TypeAlias

from pydantic import BaseModel

from typing import TypeVar

T = TypeVar("T")
Nested: TypeAlias = T | list["Nested"] | dict[str, "Nested"]


# Type alias for JSON-compatible data structures.
JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)

# Type alias for configuration values.
ConfigValue: TypeAlias = (
    None | bool | int | float | str | list["ConfigValue"] | dict[str, "ConfigValue"]
)


class ResourceDefinition(BaseModel):
    type: str
    variant: str
    config: dict[str, ConfigValue]
    dependencies: dict[str, Nested[str]]  # TODO @edwinyyyu: documentation


class ExternalServiceAPIError(Exception):
    """
    Raised when an API error occurs for an external service.
    """

    pass
