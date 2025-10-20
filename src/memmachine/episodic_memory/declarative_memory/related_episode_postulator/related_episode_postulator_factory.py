"""
Factory for RelatedEpisodePostulator instances.
"""

from typing import Any

from memmachine.common.data_types import ConfigValue, Nested
from memmachine.common.factory import Factory

from .related_episode_postulator import RelatedEpisodePostulator


class RelatedEpisodePostulatorFactory(Factory):
    """
    Factory for RelatedEpisodePostulator instances.
    """

    @staticmethod
    def create(
        provider: str,
        config: dict[str, ConfigValue],
        dependencies: dict[str, Nested[str]],
        injections: dict[str, Any],
    ) -> RelatedEpisodePostulator:
        match provider:
            case "null":
                from .null_related_episode_postulator import (
                    NullRelatedEpisodePostulator,
                )

                return NullRelatedEpisodePostulator()
            case "previous":
                from .previous_related_episode_postulator import (
                    PreviousRelatedEpisodePostulator,
                )

                return PreviousRelatedEpisodePostulator(
                    dict(config) | Factory.inject_dependencies(dependencies, injections)
                )
            case _:
                raise ValueError(
                    f"Unknown RelatedEpisodePostulator provider: {provider}"
                )
