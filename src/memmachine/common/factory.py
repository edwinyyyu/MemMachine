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
        dependencies: dict[str, Nested[str]],
        injections: dict[str, Any],
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
            dependencies (dict[str, Nested[str]]):
                The nested structure (list/dict) of dependency IDs to wire.
            injections (dict[str, Any]):
                A dictionary of injected dependencies,
                where keys are dependency IDs
                and values are the corresponding resource instances.
        """
        raise NotImplementedError

    @staticmethod
    def inject_dependencies(
        dependencies: dict[str, Nested[str]], injections: dict[str, Any]
    ) -> dict[str, Nested[Any]]:
        """
        Inject dependencies into a nested structure
        by replacing IDs with their corresponding instances.

        Args:
            dependencies (dict[str, Nested[str]]):
                The nested structure (list/dict) containing dependency IDs.
            injections (dict[str, Any]):
                A dictionary mapping IDs to their corresponding instances.
        """
        return Factory._inject_nested_ids(dependencies, injections)

    @staticmethod
    def _inject_nested_ids(
        nested: Nested[str],
        injections: dict[str, Any],
    ) -> Nested[Any]:
        """
        Recursively replace IDs in a nested structure
        with their corresponding values from an injections dictionary.

        Args:
            nested_ids (Nested[str]):
                A nested structure (list/dict) containing string IDs.
            injections (dict[str, Any]):
                A dictionary mapping IDs to their corresponding instances.
        """
        if isinstance(nested, str):
            if nested not in injections:
                raise ValueError(f"Instance with id {nested} not found in injections")
            return injections[nested]
        elif isinstance(nested, list):
            return [Factory._inject_nested_ids(item, injections) for item in nested]
        elif isinstance(nested, dict):
            return {
                key: Factory._inject_nested_ids(value, injections)
                for key, value in nested.items()
            }
        else:
            raise TypeError("Nested structure must be a string, list, or dict")
