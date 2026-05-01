"""unfinished_business — action items User asked about but never resolved.

User mentions tasks/questions/follow-ups. Some get resolved, some don't.
Question: "what items are still open?"

Tests absence-detection: the SIGNAL is what's MISSING (no follow-up).

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


FILLER = ["Long day.", "Tired.", "Coffee good.", "Need lunch."]


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

    # Item 1: book hotel for Tokyo trip — RESOLVED
    add("Need to book a hotel for the Tokyo trip next month.", "task", ["User"])
    filler()
    filler()
    add("Booked the Tokyo hotel — Ginza area, 5 nights.", "task_done", ["User"])
    filler()
    filler()

    # Item 2: ask Marcus about Q3 budget — UNRESOLVED
    add(
        "Need to ask Marcus about the Q3 budget for our team.",
        "task",
        ["User", "Marcus"],
    )
    filler()
    filler()
    filler()

    # Item 3: review the auth RFC — RESOLVED
    add("TODO: review Alice's auth RFC document.", "task", ["User", "Alice"])
    filler()
    add("Read the auth RFC — left detailed comments.", "task_done", ["User"])
    filler()
    filler()

    # Item 4: schedule dentist appointment — UNRESOLVED
    add(
        "Should schedule a dentist appointment, my last cleaning was over a year ago.",
        "task",
        ["User"],
    )
    filler()
    filler()
    filler()

    # Item 5: figure out what to get partner for anniversary — UNRESOLVED
    add(
        "Anniversary is in 3 weeks — still no idea what to get for a gift.",
        "task",
        ["User"],
    )
    filler()
    filler()

    # Item 6: renew car registration — RESOLVED
    add("Car registration expires this month — need to renew.", "task", ["User"])
    filler()
    add("Got the car registration renewed at the DMV today.", "task_done", ["User"])
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
            kind="open",
            question="What action items does User still have open (not yet resolved)?",
            expected_contains=["Marcus", "dentist", "anniversary"],
        ),
        Question(
            qid="Q02",
            kind="open_specific",
            question="Did User figure out the anniversary gift yet?",
            expected_contains=["no"],
        ),
        Question(
            qid="Q03",
            kind="open_specific",
            question="Has User scheduled the dentist appointment?",
            expected_contains=["no"],
        ),
        Question(
            qid="Q04",
            kind="open_specific",
            question="Has User asked Marcus about the Q3 budget?",
            expected_contains=["no"],
        ),
        Question(
            qid="Q05",
            kind="closed",
            question="Did User book the Tokyo hotel?",
            expected_contains=["yes", "Ginza"],
        ),
        Question(
            qid="Q06",
            kind="closed",
            question="Did User review the auth RFC?",
            expected_contains=["yes", "comments"],
        ),
        Question(
            qid="Q07",
            kind="closed",
            question="Did User renew the car registration?",
            expected_contains=["yes", "DMV"],
        ),
        Question(
            qid="Q08",
            kind="count_open",
            question="How many of User's recent action items are still unresolved?",
            expected_contains=["3", "three"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
