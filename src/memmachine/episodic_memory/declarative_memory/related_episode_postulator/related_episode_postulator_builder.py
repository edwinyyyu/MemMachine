"""
Builder for RelatedEpisodePostulator instances.
"""

from typing import Any

from memmachine.common.builder import Builder

from .related_episode_postulator import RelatedEpisodePostulator


class RelatedEpisodePostulatorBuilder(Builder):
    """
    Builder for RelatedEpisodePostulator instances.
    """

    @staticmethod
    def get_dependency_ids(variant: str, config: dict[str, Any]) -> set[str]:
        dependency_ids = set()

        match variant:
            case "null":
                pass
            case "previous":
                dependency_ids.add(config["vector_graph_store_id"])

        return dependency_ids

    @staticmethod
    def build(
        variant: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> RelatedEpisodePostulator:
        match variant:
            case "null":
                from .null_related_episode_postulator import (
                    NullRelatedEpisodePostulator,
                )

                return NullRelatedEpisodePostulator()
            case "previous":
                from .previous_related_episode_postulator import (
                    PreviousRelatedEpisodePostulator,
                )

                populated_config = {
                    "vector_graph_store": injections[config["vector_graph_store_id"]],
                    "search_limit": config.get("search_limit", 1),
                    "filterable_property_keys": config.get(
                        "filterable_property_keys", set()
                    ),
                }
                return PreviousRelatedEpisodePostulator(populated_config)
            case _:
                raise ValueError(f"Unknown RelatedEpisodePostulator variant: {variant}")
