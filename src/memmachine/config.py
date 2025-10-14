from typing import Any

from pydantic import BaseModel

class ResourceDefinition(BaseModel):
    type: str
    variant: str
    config: dict[str, Any]
    dependency_ids: dict[str, str]
