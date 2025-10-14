"""
Factory for RelatedEpisodePostulator instances.
"""

from typing import Any

from memmachine.common.factory import Factory
from .related_episode_postulator import RelatedEpisodePostulator


class RelatedEpisodePostulatorFactory(Factory):
    """
    Factory for RelatedEpisodePostulator instances.
    """

    @staticmethod
    def create(
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
                    key: value
                    for key, value in config.items()
                    if key != "vector_graph_store_id"
                } | {"vector_graph_store": injections[config["vector_graph_store_id"]]}
                return PreviousRelatedEpisodePostulator(populated_config)
            case _:
                raise ValueError(f"Unknown RelatedEpisodePostulator variant: {variant}")
