"""Deterministic substring grader (same spirit as round 9's grader).

For each Question, check:
  - expected_contains: every phrase must appear (case-insensitive substring)
  - expected_absent: none of these should appear
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Verdict:
    qid: str
    passed: bool
    reason: str
    answer: str


def grade_one(question, answer: str) -> Verdict:
    ans_low = answer.lower()
    missing = [p for p in question.expected_contains if p.lower() not in ans_low]
    forbidden = [p for p in question.expected_absent if p.lower() in ans_low]
    if missing:
        return Verdict(question.qid, False, f"missing={missing}", answer)
    if forbidden:
        return Verdict(question.qid, False, f"forbidden={forbidden}", answer)
    return Verdict(question.qid, True, "ok", answer)


def grade_all(questions, answers: dict[str, str]) -> list[Verdict]:
    return [grade_one(q, answers.get(q.qid, "")) for q in questions]
