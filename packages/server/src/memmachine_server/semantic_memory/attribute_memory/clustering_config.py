"""Clustering configuration for :class:`AttributeMemory.ingest`.

Bundles the parameters that govern how events are grouped into
clusters, when a cluster is flushed to the LLM, and how idle
clusters are garbage-collected.  Passed once at
:class:`AttributeMemory` construction — not persisted by the memory.
"""

from dataclasses import dataclass, field
from datetime import timedelta

from memmachine_server.semantic_memory.attribute_memory.clustering import (
    ClusterSplitterProtocol,
    NoOpClusterSplitter,
)
from memmachine_server.semantic_memory.attribute_memory.data_types import (
    ClusterParams,
)


@dataclass
class ClusteringConfig:
    """How :meth:`AttributeMemory.ingest` groups and flushes events.

    * ``enabled`` — when ``False``, :meth:`AttributeMemory.ingest`
      bypasses cluster assignment entirely and processes each event
      immediately against the topic-wide profile.  When ``True``, the
      remaining settings govern clustered ingestion.
    * ``cluster_params`` — similarity threshold and time-gap limits
      for :class:`ClusterManager.assign`.
    * ``splitter`` — optional per-cluster split strategy; defaults to
      :class:`NoOpClusterSplitter`.
    * ``trigger_messages`` — flush a cluster once it holds at least
      this many pending events.  Set to ``0`` to disable the size
      trigger and rely solely on age.  Set to ``1`` to flush on every
      call (no batching).
    * ``trigger_age`` — flush a cluster once its oldest pending event
      is at least this old.  Set to ``None`` to disable the age
      trigger.
    * ``idle_ttl`` — clusters with no pending events and no activity
      within this window are GC'd.  Set to ``None`` to disable GC.
    * ``max_clusters_per_run`` — cap on clusters processed per ingest
      call, preventing a single call from monopolizing an LLM quota.
    * ``max_features_per_update`` — cap on the number of existing
      attributes shown to the LLM as "old profile" for each topic
      during feature extraction.  Bounds prompt size and reduces the
      chance of context-length overflow.
    * ``consolidation_threshold`` — after flushing clusters, any
      ``(topic, category)`` pair whose attribute count reaches this
      value is auto-consolidated by :meth:`AttributeMemory.ingest`.
      Set to ``0`` to disable auto-consolidation.
    """

    enabled: bool = True
    cluster_params: ClusterParams = field(default_factory=ClusterParams)
    splitter: ClusterSplitterProtocol = field(default_factory=NoOpClusterSplitter)
    trigger_messages: int = 5
    trigger_age: timedelta | None = timedelta(minutes=5)
    idle_ttl: timedelta | None = timedelta(days=1)
    max_clusters_per_run: int = 5
    max_features_per_update: int = 50
    consolidation_threshold: int = 20
