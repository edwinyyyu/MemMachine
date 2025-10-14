from typing import Any

from pydantic import BaseModel

from .data_types import Nested

class ResourceDefinition(BaseModel):
    type: str
    variant: str
    config: dict[str, Any]
    dependencies: Nested[str]
