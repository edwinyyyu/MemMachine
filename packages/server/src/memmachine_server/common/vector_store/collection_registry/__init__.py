"""Collection registry interface and implementations."""

from .collection_registry import (
    CollectionRegistry,
    PurgeClaim,
    RegisteredCollection,
)

__all__ = [
    "CollectionRegistry",
    "PurgeClaim",
    "RegisteredCollection",
]
