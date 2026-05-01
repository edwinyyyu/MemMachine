"""temporal_proximity — "what was X right before/after Y?"

Tests retrieval anchored on temporal adjacency rather than topic.
"What were we discussing right before the power went out?" requires
finding the power-outage event AND adjacent events in time.

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

    # Day 1: morning standup, lunch, power outage, recovery
    add("Morning standup discussed the auth refactor priorities.", "morning", ["User"])
    filler()
    add(
        "Lunch with the design team — they showed the new dashboard mocks.",
        "midday",
        ["User"],
    )
    filler()
    add(
        "Right after lunch, the office had a power outage for 30 minutes.",
        "incident",
        ["User"],
    )
    filler()
    add(
        "Once power was back, I jumped into reviewing Bob's PR on caching.",
        "afternoon",
        ["User", "Bob"],
    )
    filler()
    filler()

    # Day 2: deploy, breakage, fix
    add("Started the day with a planned deploy to staging.", "morning", ["User"])
    filler()
    add("Deploy went sideways — auth service wouldn't start.", "incident", ["User"])
    filler()
    add(
        "Spent two hours debugging — turned out to be a missing env var.",
        "fix",
        ["User"],
    )
    filler()
    add(
        "After fixing the deploy, finally got to my 1:1 with Marcus.",
        "afternoon",
        ["User", "Marcus"],
    )
    filler()
    filler()

    # Day 3: trip
    add(
        "Flew to SF for the Stripe offsite Tuesday morning.", "trip", ["User", "Stripe"]
    )
    filler()
    add(
        "Right before the flight, I grabbed a sandwich at the airport deli.",
        "trip_detail",
        ["User"],
    )
    filler()
    add(
        "Offsite kicked off with a strategy session led by the CFO.",
        "trip_detail",
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
            kind="proximity",
            question="What was User doing right before the power outage?",
            expected_contains=["lunch", "design", "dashboard"],
        ),
        Question(
            qid="Q02",
            kind="proximity",
            question="What did User do right after the power came back?",
            expected_contains=["Bob", "PR", "caching"],
        ),
        Question(
            qid="Q03",
            kind="proximity",
            question="What did User do right before the deploy went sideways?",
            expected_contains=["planned", "deploy", "staging"],
        ),
        Question(
            qid="Q04",
            kind="proximity",
            question="What did User do after fixing the deploy?",
            expected_contains=["Marcus", "1:1"],
        ),
        Question(
            qid="Q05",
            kind="proximity",
            question="What did User eat right before the SF flight?",
            expected_contains=["sandwich", "deli"],
        ),
        Question(
            qid="Q06",
            kind="proximity",
            question="What was the first thing on Day 1's morning agenda?",
            expected_contains=["standup", "auth"],
        ),
        Question(
            qid="Q07",
            kind="proximity_specific",
            question="What kicked off the offsite?",
            expected_contains=["strategy", "CFO"],
        ),
        Question(
            qid="Q08",
            kind="duration",
            question="How long was the power outage?",
            expected_contains=["30 minutes"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
