"""numerical_state — User's quantities evolve over time.

User mentions numerical state at different points: weight, salary, savings,
counts. Later asks about current values, max/min, deltas.

Total ~50 turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    idx: int
    text: str
    kind: str
    mentions: list[str] = field(default_factory=list)


FILLER = ["Long day.", "Tired.", "Slack laggy.", "Need lunch.", "Coffee good."]


def generate():
    turns = []
    fc = 0

    def add(text, kind, mentions=None):
        turns.append(
            Turn(idx=len(turns) + 1, text=text, kind=kind, mentions=mentions or [])
        )

    def filler():
        nonlocal fc
        add(FILLER[fc % len(FILLER)], "filler")
        fc += 1

    filler()
    filler()

    # Salary progression
    add(
        "Started at $120k base when I joined Stripe in 2020.",
        "salary",
        ["User", "Stripe"],
    )
    filler()
    add("Got a raise to $145k base last year.", "salary", ["User"])
    filler()
    add(
        "Just signed Notion offer at $220k base — big jump.",
        "salary",
        ["User", "Notion"],
    )
    filler()
    filler()

    # Weight progression
    add("Hit 175 lb at the start of marathon training.", "weight", ["User"])
    filler()
    add("Down to 162 lb after three months of running.", "weight", ["User"])
    filler()
    add("Currently 168 lb — gained some back during the holidays.", "weight", ["User"])
    filler()
    filler()

    # Savings
    add(
        "Got my emergency fund up to 3 months expenses last spring.",
        "savings",
        ["User"],
    )
    filler()
    add("Now at 6 months emergency fund.", "savings", ["User"])
    filler()
    filler()

    # Step count
    add("Daily step goal is 10,000 — usually hit it.", "habit", ["User"])
    filler()
    add("Yesterday only got 3,200 steps. Bad day.", "habit", ["User"])
    filler()
    add("Made up for it: 18,000 steps today on a long walk.", "habit", ["User"])
    filler()
    filler()

    for i, t in enumerate(turns, start=1):
        t.idx = i
    return turns


@dataclass
class GroundTruth:
    pass


def ground_truth(turns):
    return GroundTruth()


@dataclass
class Question:
    qid: str
    kind: str
    question: str
    expected_contains: list[str]
    expected_absent: list[str] = field(default_factory=list)


def build_questions(gt):
    return [
        Question(
            qid="Q01",
            kind="latest",
            question="What is User's current base salary?",
            expected_contains=["220", "Notion"],
        ),
        Question(
            qid="Q02",
            kind="first",
            question="What was User's starting salary at Stripe?",
            expected_contains=["120"],
        ),
        Question(
            qid="Q03",
            kind="delta",
            question="By how much did User's salary increase from Stripe start to Notion offer?",
            expected_contains=["100", "$100"],
        ),
        Question(
            qid="Q04",
            kind="latest",
            question="What's User's current weight?",
            expected_contains=["168"],
        ),
        Question(
            qid="Q05",
            kind="min",
            question="What was User's lowest recorded weight?",
            expected_contains=["162"],
        ),
        Question(
            qid="Q06",
            kind="latest",
            question="How many months of emergency fund does User have?",
            expected_contains=["6", "six"],
        ),
        Question(
            qid="Q07",
            kind="extreme",
            question="What was User's worst step day recently?",
            expected_contains=["3,200", "3200"],
        ),
        Question(
            qid="Q08",
            kind="extreme",
            question="What was User's best step day recently?",
            expected_contains=["18,000", "18000"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
