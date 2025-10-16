"""
Factory for Embedder instances.
"""

from typing import Any

from memmachine.common.data_types import Nested, ConfigValue
from memmachine.common.factory import Factory

from .embedder import Embedder


class EmbedderFactory(Factory):
    """
    Factory for Embedder instances.
    """

    @staticmethod
    def create(
        variant: str,
        config: dict[str, ConfigValue],
        dependencies: dict[str, Nested[str]],
        injections: dict[str, Any],
    ) -> Embedder:
        match variant:
            case "openai":
                from .openai_embedder import OpenAIEmbedder

                # TODO: Temporary until refactoring of OpenAIEmbedder is done,
                # so that we do not union config and injected dependencies.
                return OpenAIEmbedder(
                    dict(config) | Factory.inject_dependencies(dependencies, injections)
                )

            case _:
                raise ValueError(f"Unknown Embedder variant: {variant}")
