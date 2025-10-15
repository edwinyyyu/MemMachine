"""
Factory for LanguageModel instances.
"""

from typing import Any

from memmachine.common.data_types import Nested, ConfigValue
from memmachine.common.factory import Factory
from memmachine.common.metrics_manager.metrics_manager import MetricsManager

from .language_model import LanguageModel


class LanguageModelFactory(Factory):
    """
    Factory for LanguageModel instances.
    """

    @staticmethod
    def create(
        variant: str,
        config: dict[str, ConfigValue],
        dependencies: Nested[str],
        injections: dict[str, Any],
    ) -> LanguageModel:
        def get_metrics_manager(config: dict[str, Any]):
            injected_metrics_manager_id = config.get("metrics_manager_id")
            if injected_metrics_manager_id is None:
                injected_metrics_manager = None
            elif not isinstance(injected_metrics_manager_id, str):
                raise TypeError("metrics_manager_id must be a string if provided")
            else:
                injected_metrics_manager = injections.get(injected_metrics_manager_id)
                if injected_metrics_manager is None:
                    raise ValueError(
                        "MetricsManager with id "
                        f"{injected_metrics_manager_id} "
                        "not found in injections"
                    )
                if not isinstance(injected_metrics_manager, MetricsManager):
                    raise TypeError(
                        "Injected dependency with id "
                        f"{injected_metrics_manager_id} "
                        "is not a MetricsManager"
                    )
            return injected_metrics_manager

        match variant:
            case "openai":
                from .openai_language_model import OpenAILanguageModel

                return OpenAILanguageModel(
                    {
                        key: value
                        for key, value in config.items()
                        if key != "metrics_manager_id"
                    }
                    | {
                        "metrics_manager": get_metrics_manager(config),
                    }
                )

            case "vllm" | "sglang" | "openai-compatible":
                from .openai_compatible_language_model import (
                    OpenAICompatibleLanguageModel,
                )

                return OpenAICompatibleLanguageModel(
                    {
                        key: value
                        for key, value in config.items()
                        if key != "metrics_manager_id"
                    }
                    | {
                        "metrics_manager": get_metrics_manager(config),
                    }
                )

            case _:
                raise ValueError(f"Unknown LanguageModel variant: {variant}")
