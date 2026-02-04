"""
MemMachine Client - A Python client library for MemMachine memory system.

This module provides a high-level interface for interacting with MemMachine's
episodic and profile memory systems.
"""

from .client import MemMachineClient
from .config import Config
from .langgraph import (
    MemMachineTools,
    create_add_memory_tool,
    create_search_memory_tool,
)
from .memory import Memory
from .models import (
    # Memory Models
    AddMemoriesResponse,
    AddMemoryResult,
    # Enums
    ContentType,
    # Episode Models
    Episode,
    # Project Models
    EpisodeCountResponse,
    EpisodeEntry,
    EpisodeResponse,
    EpisodeType,
    ListResult,
    MemoryType,
    ProjectConfig,
    ProjectResponse,
    ResourceStatus,
    SearchResult,
    SemanticFeature,
)
from .project import Project

__all__ = [
    # Memory Models
    "AddMemoriesResponse",
    "AddMemoryResult",
    # Client Classes
    "Config",
    # Enums
    "ContentType",
    # Episode Models
    "Episode",
    # Project Models
    "EpisodeCountResponse",
    "EpisodeEntry",
    "EpisodeResponse",
    "EpisodeType",
    "ListResult",
    "MemMachineClient",
    # LangGraph Tools
    "MemMachineTools",
    "Memory",
    "MemoryType",
    "Project",
    "ProjectConfig",
    "ProjectResponse",
    "ResourceStatus",
    "SearchResult",
    "SemanticFeature",
    "create_add_memory_tool",
    "create_search_memory_tool",
]
