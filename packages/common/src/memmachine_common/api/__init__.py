"""Shared API definitions for MemMachine client and server."""

from enum import Enum


class MemoryType(Enum):
    """Memory type."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class EpisodeType(Enum):
    """Episode type."""

    MESSAGE = "message"


__all__ = ["EpisodeType", "MemoryType"]
