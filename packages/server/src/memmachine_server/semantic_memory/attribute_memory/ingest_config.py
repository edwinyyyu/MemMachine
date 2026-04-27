"""Ingestion configuration for :class:`AttributeMemory`.

This dataclass keeps the tuning knobs that affect prompt construction
and automatic consolidation.
"""

from dataclasses import dataclass


@dataclass
class IngestConfig:
    """How :meth:`AttributeMemory.ingest` shapes prompts and consolidation."""

    max_features_per_update: int = 50
    consolidation_threshold: int = 20
