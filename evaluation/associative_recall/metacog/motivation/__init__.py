"""Standalone motivation generator module.

Importable by any harness. Exports:
    - MotivationState (TypedDict-shaped dataclass-style mapping)
    - VALID_STATES (frozenset of allowed categorical state strings)
    - MOTIVATION_SCHEMA (json_schema for the LLM call)
    - update_motivation(...) async function
    - initial_motivation_state() helper
"""

from .motivation import (
    MOTIVATION_SCHEMA,
    VALID_STATES,
    MotivationState,
    initial_motivation_state,
    update_motivation,
)

__all__ = [
    "MOTIVATION_SCHEMA",
    "VALID_STATES",
    "MotivationState",
    "initial_motivation_state",
    "update_motivation",
]
