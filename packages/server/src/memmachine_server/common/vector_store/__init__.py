"""Public exports for vector store."""

from .data_types import (
    IndexedProperties,
    PartitionSchema,
    PropertyTypeMismatchError,
    QueryResult,
    Record,
    UndeclaredPropertyKeyError,
    UnsupportedFilterError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionSchemaMismatchError,
    validate_collection_name,
)
from .vector_store import VectorStore, VectorStorePartition, get_or_create_partition

__all__ = [
    "IndexedProperties",
    "PartitionSchema",
    "PropertyTypeMismatchError",
    "QueryResult",
    "Record",
    "UndeclaredPropertyKeyError",
    "UnsupportedFilterError",
    "VectorStore",
    "VectorStorePartition",
    "VectorStorePartitionAlreadyExistsError",
    "VectorStorePartitionSchemaMismatchError",
    "get_or_create_partition",
    "validate_collection_name",
]
