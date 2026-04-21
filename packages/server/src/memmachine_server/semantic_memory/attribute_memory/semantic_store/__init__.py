"""Relational storage interface for attribute memory.

Only the abstract interface is re-exported here.  Concrete backends
(``SQLAlchemySemanticStore``, ``PartitionSchema``, etc.) live in
:mod:`.sqlalchemy_semantic_store` and must be imported from there.
"""

from .data_types import SemanticStorePartitionAlreadyExistsError
from .semantic_store import (
    SemanticAttribute,
    SemanticStore,
    SemanticStorePartition,
)

__all__ = [
    "SemanticAttribute",
    "SemanticStore",
    "SemanticStorePartition",
    "SemanticStorePartitionAlreadyExistsError",
]
