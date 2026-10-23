"""Prometheus metrics output shared by the server's worker processes."""

import os

from prometheus_client import REGISTRY, CollectorRegistry, multiprocess


def metrics_registry() -> CollectorRegistry:
    """Return the registry a scrape reads.

    Each worker process keeps its own registry, so a scrape answered from one
    of them would report that worker alone. With PROMETHEUS_MULTIPROC_DIR set,
    every process writes its metrics to that directory, and the returned
    registry merges them.
    """
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return REGISTRY
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry
