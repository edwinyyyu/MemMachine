"""
Factory for DerivativeDeriver instances.
"""

from typing import Any

from memmachine.common.data_types import ConfigValue, Nested
from memmachine.common.factory import Factory

from .derivative_deriver import DerivativeDeriver


class DerivativeDeriverFactory(Factory):
    """
    Factory for DerivativeDeriver instances.
    """

    @staticmethod
    def create(
        provider: str,
        config: dict[str, ConfigValue],
        dependencies: dict[str, Nested[str]],
        injections: dict[str, Any],
    ) -> DerivativeDeriver:
        match provider:
            case "concatenation":
                from .concatenation_derivative_deriver import (
                    ConcatenationDerivativeDeriver,
                )

                populated_config = config
                return ConcatenationDerivativeDeriver(populated_config)
            case "identity":
                from .identity_derivative_deriver import (
                    IdentityDerivativeDeriver,
                )

                populated_config = config
                return IdentityDerivativeDeriver(populated_config)
            case "sentence":
                from .sentence_derivative_deriver import (
                    SentenceDerivativeDeriver,
                )

                populated_config = config
                return SentenceDerivativeDeriver(populated_config)
            case _:
                raise ValueError(f"Unknown DerivativeDeriver provider: {provider}")
