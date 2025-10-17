"""
Factory for MetricsManager instances.
"""

from typing import Any

from memmachine.common.data_types import Nested, ConfigValue
from memmachine.common.factory import Factory

from .metrics_manager import MetricsManager


class MetricsManagerFactory(Factory):
    """
    Factory for MetricsManager instances.
    """

    @staticmethod
    def create(
        provider: str,
        config: dict[str, ConfigValue],
        dependencies: dict[str, Nested[str]],
        injections: dict[str, Any],
    ) -> MetricsManager:
        match provider:
            case "prometheus":
                from .prometheus_metrics_manager import (
                    PrometheusMetricsManager,
                )

                return PrometheusMetricsManager()
            case _:
                raise ValueError(f"Unknown MetricsManager provider: {provider}")
