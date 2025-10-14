"""
Factory for MetricsManager instances.
"""

from typing import Any

from memmachine.common.factory import Factory

from .metrics_manager import MetricsManager


class MetricsManagerFactory(Factory):
    """
    Factory for MetricsManager instances.
    """

    @staticmethod
    def create(
        variant: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> MetricsManager:
        match variant:
            case "prometheus":
                from .prometheus_metrics_manager import (
                    PrometheusMetricsManager,
                )

                return PrometheusMetricsManager()
            case _:
                raise ValueError(f"Unknown MetricsManager variant: {variant}")
