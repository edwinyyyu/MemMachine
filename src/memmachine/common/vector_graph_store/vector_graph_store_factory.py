"""
Factory for VectorGraphStore instances.
"""

from typing import Any

from memmachine.common.factory import Factory

from .vector_graph_store import VectorGraphStore


class VectorGraphStoreFactory(Factory):
    """
    Factory for VectorGraphStore instances.
    """

    @staticmethod
    def create(
        variant: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> VectorGraphStore:
        match variant:
            case "neo4j":
                from .neo4j_vector_graph_store import (
                    Neo4jVectorGraphStore,
                    Neo4jVectorGraphStoreConfig,
                )

                return Neo4jVectorGraphStore(Neo4jVectorGraphStoreConfig(**config))
            case _:
                raise ValueError(f"Unknown VectorGraphStore variant: {variant}")
