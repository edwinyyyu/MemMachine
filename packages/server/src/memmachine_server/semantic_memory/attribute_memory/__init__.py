"""Attribute memory: partition-scoped semantic-memory engine.

Only abstractions and the public orchestrator are re-exported.
Concrete backends (``SQLAlchemySemanticStore``, ``PartitionSchema``,
``TopicDefinition``, ``CategoryDefinition``) live in
:mod:`.semantic_store.sqlalchemy_semantic_store` and must be imported
from there directly.
"""

from .attribute_memory import AttributeMemory
from .clustering_config import ClusteringConfig
from .semantic_store import (
    SemanticAttribute,
    SemanticStore,
    SemanticStorePartition,
    SemanticStorePartitionAlreadyExistsError,
)

__all__ = [
    "AttributeMemory",
    "ClusteringConfig",
    "SemanticAttribute",
    "SemanticStore",
    "SemanticStorePartition",
    "SemanticStorePartitionAlreadyExistsError",
]
