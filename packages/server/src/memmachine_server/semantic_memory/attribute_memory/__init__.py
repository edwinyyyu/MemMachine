"""Attribute memory: partition-scoped semantic-memory engine.

Only abstractions and the public orchestrator are re-exported.
Concrete backends (``SQLAlchemySemanticStore``, ``PartitionSchema``,
``TopicDefinition``, ``CategoryDefinition``) live in
:mod:`.semantic_store.sqlalchemy_semantic_store` and must be imported
from there directly.
"""

from .attribute_memory import AttributeMemory
from .ingest_config import IngestConfig
from .semantic_store import (
    SemanticAttribute,
    SemanticStore,
    SemanticStorePartition,
    SemanticStorePartitionAlreadyExistsError,
)

__all__ = [
    "AttributeMemory",
    "IngestConfig",
    "SemanticAttribute",
    "SemanticStore",
    "SemanticStorePartition",
    "SemanticStorePartitionAlreadyExistsError",
]
