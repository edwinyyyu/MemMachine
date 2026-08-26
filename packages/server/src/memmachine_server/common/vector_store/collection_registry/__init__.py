"""Collection registry interface and implementations."""

from .collection_registry import (
    CollectionAlreadyRegisteredError,
    CollectionRegistry,
    CollectionRegistryEntry,
)

__all__ = [
    "CollectionAlreadyRegisteredError",
    "CollectionRegistry",
    "CollectionRegistryEntry",
]
