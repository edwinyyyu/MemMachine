"""
Factory for Embedder instances.
"""

from typing import Any

from memmachine.common.data_types import Nested, ConfigValue
from memmachine.common.factory import Factory
from memmachine.common.metrics_manager.metrics_manager import MetricsManager

from .embedder import Embedder

class EmbedderFactory(Factory):
    """
    Factory for Embedder instances.
    """

    @staticmethod
    def create(
        variant: str,
        config: dict[str, ConfigValue],
        dependencies: Nested[str],
        injections: dict[str, Any],
    ) -> Embedder:
        match variant:
            case "openai":
                from .openai_embedder import OpenAIEmbedder

                # TODO: Temporary until refactoring of OpenAIEmbedder is done,
                # so that we do not union config and injected dependencies.
                if not isinstance(dependencies, dict):
                    raise TypeError("Dependencies must be a dictionary for OpenAIEmbedder")

                return OpenAIEmbedder(
                    dict(config) |
                    Factory.inject_dependencies(
                        dependencies, injections
                    )
                )

            case _:
                raise ValueError(f"Unknown Embedder variant: {variant}")
