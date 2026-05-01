"""Shared scenario types for round 16b."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    idx: int
    text: str
    expected_world: str  # ground-truth world for this turn


@dataclass
class FactCheck:
    """A ground-truth fact that should (or should not) be stored in a world."""

    description: str
    world: str  # the world this fact belongs to
    must_contain: list[str]  # any entry text matching one of these substrings counts
    must_not_world: list[str] = field(
        default_factory=list
    )  # worlds that must NOT contain it


@dataclass
class QACheck:
    qid: str
    question: str
    expected_world: str  # which world the retrieve should target
    must_include: list[str]  # answer text must include at least one
    must_exclude: list[str] = field(default_factory=list)  # answer must NOT include any


@dataclass
class Scenario:
    name: str
    turns: list[Turn]
    fact_checks: list[FactCheck]
    qa_checks: list[QACheck]
