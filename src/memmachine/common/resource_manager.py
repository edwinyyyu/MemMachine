"""
Resource manager for creating and managing resources
based on their definitions and dependencies.
"""

from collections import deque
from collections.abc import Mapping
from typing import Any, Iterable

from memmachine.common.data_types import ResourceDefinition
from memmachine.common.embedder.embedder_factory import EmbedderFactory
from memmachine.common.factory import Factory
from memmachine.common.language_model.language_model_factory import (
    LanguageModelFactory,
)
from memmachine.common.metrics_manager.metrics_manager_factory import (
    MetricsManagerFactory,
)
from memmachine.common.reranker.reranker_factory import RerankerFactory
from memmachine.common.utils import get_nested_values
from memmachine.common.vector_graph_store.vector_graph_store_factory import (
    VectorGraphStoreFactory,
)

"""
Each entry in resource_definitions should look like this:
```
resource_id: {
    "type": "<TYPE>",
    "provider": "<VARIANT>",
    "config": {
        ... <CONFIGURATION> ...
    }
}
```
"""

# Map resource types to their corresponding factory classes
resource_factory_map: dict[str, type[Factory]] = {
    "embedder": EmbedderFactory,
    "language_model": LanguageModelFactory,
    "metrics_manager": MetricsManagerFactory,
    "reranker": RerankerFactory,
    "vector_graph_store": VectorGraphStoreFactory,
}


class ResourceManager:
    """
    Resource manager for creating and managing resources
    based on their definitions and dependencies.
    """

    def __init__(
        self,
        resource_cache: dict[str, Any] | None = None,
    ):
        self._resource_cache = (
            resource_cache.copy() if resource_cache is not None else {}
        )

    def get_resource(self, resource_id: str) -> Any:
        """
        Get a resource by its ID from the resource cache.
        """
        return self._resource_cache.get(resource_id)

    def resolve_resources(self, resource_ids: Iterable[str]) -> dict[str, Any]:
        """
        Resolve an iterable of resource IDs to a dict
        mapping resource IDs to their corresponding resources.
        """
        return {
            resource_id: resource
            for resource_id in resource_ids
            if (resource := self.get_resource(resource_id)) is not None
        }

    def create_resources(
        self,
        resource_definitions: Mapping[str, ResourceDefinition],
    ):
        """
        Create resources
        based on their definitions and dependencies.
        """

        # Map from resource ID to a set of dependency resource IDs
        resource_dependency_graph = {}
        for resource_id, resource_definition in resource_definitions.items():
            dependency_ids = get_nested_values(resource_definition.dependencies)
            resource_dependency_graph[resource_id] = set(dependency_ids)

        def order_resources(
            resource_dependency_graph: dict[str, set[str]],
            resource_cache: dict[str, Any],
        ) -> list[str]:
            """
            Order resources based on their dependencies
            using a topological sort.
            """
            ordered_resource_ids = []

            dependency_counts = {
                resource_id: 0 for resource_id in resource_dependency_graph.keys()
            }
            dependent_resource_ids: dict[str, set[str]] = {
                resource_id: set() for resource_id in resource_dependency_graph.keys()
            }

            for (
                resource_id,
                dependency_ids,
            ) in resource_dependency_graph.items():
                for dependency_id in dependency_ids:
                    # Check that the dependency exists in either the resource definitions or the resource cache.
                    if (
                        dependency_id not in resource_dependency_graph.keys()
                        and dependency_id not in resource_cache.keys()
                    ):
                        raise ValueError(
                            f"Dependency {dependency_id} "
                            f"for resource {resource_id} "
                            "found in neither resource definitions nor resource cache"
                        )

                    # Only count depdencies that have not been initialized yet.
                    if dependency_id in resource_dependency_graph.keys():
                        dependency_counts[resource_id] += 1
                        dependent_resource_ids[dependency_id].add(resource_id)

            queue = deque(
                [
                    resource_id
                    for resource_id, count in dependency_counts.items()
                    if count == 0
                ]
            )

            while queue:
                resource_id = queue.popleft()
                ordered_resource_ids.append(resource_id)

                for dependent_resource_id in dependent_resource_ids[resource_id]:
                    dependency_counts[dependent_resource_id] -= 1
                    if dependency_counts[dependent_resource_id] == 0:
                        queue.append(dependent_resource_id)

            if len(ordered_resource_ids) != len(resource_dependency_graph):
                raise ValueError("Cyclic dependency detected in resource definitions")

            return ordered_resource_ids

        ordered_resource_ids = order_resources(resource_dependency_graph)

        initialized_resources = {}
        for resource_id in ordered_resource_ids:
            if resource_id in self._resource_cache:
                continue

            resource_definition = resource_definitions[resource_id]

            resource_factory = resource_factory_map[resource_definition.type]

            initialized_resource = resource_factory.create(
                provider=resource_definition.provider,
                config=resource_definition.config,
                injections=self._resource_cache,
            )

            initialized_resources[resource_id] = initialized_resource
            self._resource_cache[resource_id] = initialized_resource
