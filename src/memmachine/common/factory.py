"""
Abstract base class for a factory that constructs objects
based on their definitions and dependencies.
"""

from abc import ABC, abstractmethod
from typing import Any

from .data_types import Nested, ConfigValue

class Factory(ABC):
    """
    Abstract base class for a factory that constructs objects
    based on their definitions and dependencies.
    """

    @staticmethod
    @abstractmethod
    def create(
        variant: str,
        config: dict[str, ConfigValue],
        dependencies: Nested[str],
        injections: dict[str, Any]
    ) -> Any:
        """
        Create the resource
        based on its variant,
        configuration,
        and injected dependencies.

        Args:
            variant (str):
                The variant of the resource to create.
            config (dict[str, ConfigValue]):
                The configuration dictionary for the resource.
            dependencies (Nested[str]):
                The nested structure (list/dict) of dependency IDs to wire.
            injections (dict[str, Any]):
                A dictionary of injected dependencies,
                where keys are dependency IDs
                and values are the corresponding resource instances.
        """
        raise NotImplementedError
