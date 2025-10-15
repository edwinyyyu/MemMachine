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
        dependencies: dict[str, Nested[str]],
        injections: dict[str, Any],
    ) -> LanguageModel:
        match variant:
            case "openai":
                from .openai_language_model import OpenAILanguageModel

                # TODO: Temporary until refactoring of OpenAILanguageModel is done,
                # so that we do not union config and injected dependencies.
                return OpenAILanguageModel(
                    dict(config) | Factory.inject_dependencies(dependencies, injections)
                )

            case "vllm" | "sglang" | "openai-compatible":
                from .openai_compatible_language_model import (
                    OpenAICompatibleLanguageModel,
                )

                # TODO: Temporary until refactoring of OpenAICompatibleLanguageModel is done,
                # so that we do not union config and injected dependencies.
                return OpenAICompatibleLanguageModel(
                    dict(config) | Factory.inject_dependencies(dependencies, injections)
                )

            case _:
                raise ValueError(f"Unknown LanguageModel variant: {variant}")
