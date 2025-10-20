"""
Factory for LanguageModel instances.
"""

from typing import Any

from memmachine.common.data_types import ConfigValue, Nested
from memmachine.common.factory import Factory

from .language_model import LanguageModel


class LanguageModelFactory(Factory):
    """
    Factory for LanguageModel instances.
    """

    @staticmethod
    def create(
        provider: str,
        config: dict[str, ConfigValue],
        dependencies: dict[str, Nested[str]],
        injections: dict[str, Any],
    ) -> LanguageModel:
        match provider:
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
                raise ValueError(f"Unknown LanguageModel provider: {provider}")
