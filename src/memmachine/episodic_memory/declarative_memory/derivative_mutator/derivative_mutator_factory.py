"""
Factory for DerivativeMutator instances.
"""

from typing import Any

from memmachine.common.data_types import Nested, ConfigValue
from memmachine.common.factory import Factory
from .derivative_mutator import DerivativeMutator


class DerivativeMutatorFactory(Factory):
    """
    Factory for DerivativeMutator instances.
    """

    @staticmethod
    def create(
        provider: str,
        config: dict[str, ConfigValue],
        dependencies: dict[str, Nested[str]],
        injections: dict[str, Any],
    ) -> DerivativeMutator:
        match provider:
            case "identity":
                from .identity_derivative_mutator import (
                    IdentityDerivativeMutator,
                )

                return IdentityDerivativeMutator()
            case "metadata":
                from .metadata_derivative_mutator import (
                    MetadataDerivativeMutator,
                )

                populated_config = config
                return MetadataDerivativeMutator(populated_config)
            case "third-person-rewrite":
                from .third_person_rewrite_derivative_mutator import (
                    ThirdPersonRewriteDerivativeMutator,
                )

                return ThirdPersonRewriteDerivativeMutator(
                    dict(config) | Factory.inject_dependencies(dependencies, injections)
                )
            case _:
                raise ValueError(f"Unknown DerivativeMutator provider: {provider}")
