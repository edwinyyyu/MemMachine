"""
Abstract base class for a factory that constructs objects
based on their definitions and dependencies.
"""

from abc import ABC, abstractmethod
from typing import Any


class Factory(ABC):
    """
    Abstract base class for a factory that constructs objects
    based on their definitions and dependencies.
    """

    @staticmethod
    @abstractmethod
    def create(variant: str, config: dict[str, Any], injections: dict[str, Any]) -> Any:
        """
        Build the resource
        based on its variant,
        configuration,
        and injected dependencies.

        Args:
            variant (str):
                The variant of the resource to build.
            config (dict[str, Any]):
                The configuration dictionary for the resource.
            injections (dict[str, Any]):
                A dictionary of injected dependencies,
                where keys are dependency IDs
                and values are the corresponding resource instances.
        """
        raise NotImplementedError
