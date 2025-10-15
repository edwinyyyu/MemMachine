"""
Factory for Embedder instances.
"""

from typing import Any

from memmachine.common.data_types import Nested
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
        config: dict[str, Any],
        dependencies: Nested[str],
        injections: dict[str, Any],
    ) -> Embedder:
        match variant:
            case "openai":
                from .openai_embedder import OpenAIEmbedder

                injected_metrics_manager_id = config.get("metrics_manager_id")
                if injected_metrics_manager_id is None:
                    injected_metrics_manager = None
                elif not isinstance(injected_metrics_manager_id, str):
                    raise TypeError("metrics_manager_id must be a string if provided")
                else:
                    injected_metrics_manager = injections.get(
                        injected_metrics_manager_id
                    )
                    if injected_metrics_manager is None:
                        raise ValueError(
                            "MetricsManager with id "
                            f"{injected_metrics_manager_id} "
                            "not found in injections"
                        )
                    elif not isinstance(injected_metrics_manager, MetricsManager):
                        raise TypeError(
                            "Injected dependency with id "
                            f"{injected_metrics_manager_id} "
                            "is not a MetricsManager"
                        )

                return OpenAIEmbedder(
                    {
                        key: value
                        for key, value in config.items()
                        if key != "metrics_manager_id"
                    }
                    | {"metrics_manager": injected_metrics_manager}
                )
            case _:
                raise ValueError(f"Unknown Embedder variant: {variant}")
