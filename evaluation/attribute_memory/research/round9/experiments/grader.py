"""Deterministic grader for round 9 state-tracking questions.

For each Question:
- check `expected_contains`: every phrase (case-insensitive substring) must
  appear SOMEWHERE in the answer.
- check `expected_absent`: none of these phrases should appear.

For the "multi" questions where entities are listed, we require the right
entity names to appear.

We return per-question verdict + summary counts.
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
    verdicts = []
    for q in questions:
        a = answers.get(q.qid, "")
        verdicts.append(grade_one(q, a))
    return verdicts


def summarize(verdicts: list[Verdict]) -> dict:
    total = len(verdicts)
    passed = sum(1 for v in verdicts if v.passed)
    by_kind = {}
    # We'll accept the qid->kind mapping from outside
    return {"total": total, "passed": passed, "rate": passed / total if total else 0.0}
