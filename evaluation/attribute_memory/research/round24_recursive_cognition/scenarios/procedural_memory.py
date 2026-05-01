"""procedural_memory — User establishes how-to procedures over time.

User describes how they do tasks ("I always X by doing Y"). Later asks
"How do I usually do X?" and expects retrieval of the procedure.

Tests procedural recall — different from event/fact recall because the
"event" is the establishment of a method, not a single happening.

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


FILLER = ["Long day.", "Tired.", "Slack laggy.", "Coffee was good.", "Need lunch."]


def generate() -> list[Turn]:
    turns: list[Turn] = []
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

    # Procedure 1: deploy via dry-run first
    add(
        "I always do a dry-run before any prod deploy. Caught a typo last week that way.",
        "procedure",
        ["User"],
    )
    filler()
    filler()
    add(
        "Yesterday's deploy: dry-run first, then actual. Standard flow.",
        "procedure_use",
        ["User"],
    )
    filler()
    filler()

    # Procedure 2: weekly review on Sundays
    add(
        "Every Sunday morning I do a weekly review — past wins, next week priorities.",
        "procedure",
        ["User"],
    )
    filler()
    filler()
    add(
        "This Sunday's review: closed 12 tickets, top priority is the auth refactor.",
        "procedure_use",
        ["User"],
    )
    filler()
    filler()

    # Procedure 3: code review heuristic
    add(
        "My code review rule: if I can't summarize the diff in two sentences, it's too big.",
        "procedure",
        ["User"],
    )
    filler()
    filler()
    add(
        "Sent this PR back today — three pages of changes, way too big per my rule.",
        "procedure_use",
        ["User"],
    )
    filler()
    filler()

    # Procedure 4: cooking — pasta water
    add(
        "I always salt my pasta water like the sea — about 2 tablespoons per pot.",
        "procedure",
        ["User"],
    )
    filler()
    filler()
    add(
        "Last night's spaghetti: salted the water properly, came out perfect.",
        "procedure_use",
        ["User"],
    )
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
            kind="procedure",
            question="What does User always do before a prod deploy?",
            expected_contains=["dry-run", "dry run"],
        ),
        Question(
            qid="Q02",
            kind="procedure",
            question="When does User do their weekly review?",
            expected_contains=["Sunday"],
        ),
        Question(
            qid="Q03",
            kind="procedure",
            question="What's User's rule for whether a PR is too big?",
            expected_contains=["two sentences", "summarize"],
        ),
        Question(
            qid="Q04",
            kind="procedure",
            question="How does User salt pasta water?",
            expected_contains=["2 tablespoons", "like the sea"],
        ),
        Question(
            qid="Q05",
            kind="procedure_use",
            question="Has User caught any typos via their pre-deploy procedure?",
            expected_contains=["yes", "typo"],
        ),
        Question(
            qid="Q06",
            kind="procedure_use",
            question="What was the top priority in this Sunday's weekly review?",
            expected_contains=["auth refactor"],
        ),
        Question(
            qid="Q07",
            kind="procedure_use",
            question="Did User reject any PRs recently? Why?",
            expected_contains=["yes", "too big", "three pages"],
        ),
        Question(
            qid="Q08",
            kind="procedure",
            question="What habits does User follow for prod deploys?",
            expected_contains=["dry-run", "dry run"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
