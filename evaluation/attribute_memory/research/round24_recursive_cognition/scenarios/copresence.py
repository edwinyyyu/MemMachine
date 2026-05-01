"""copresence — have two entities met / been in same situation?

User describes events with multiple people. Later asks "have X and Y met?"
A human remembers the full cast of an event, not just headline content.

Total ~40 turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    idx: int
    text: str
    kind: str
    mentions: list[str] = field(default_factory=list)


FILLER = ["Long day.", "Tired.", "Coffee good."]


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

    add(
        "Had dinner with Alice and Bob at Joe's last Friday.",
        "meet",
        ["User", "Alice", "Bob"],
    )
    filler()
    filler()

    add(
        "Zoom call with Carla, Dave, and Eve about the Q3 plan.",
        "meet",
        ["User", "Carla", "Dave", "Eve"],
    )
    filler()
    filler()

    add("Bob and I grabbed coffee on Tuesday — solo.", "meet", ["User", "Bob"])
    filler()
    filler()

    add(
        "Big Stripe offsite Wednesday: Alice, Carla, Frank, Marcus all in the same room.",
        "meet_group",
        ["User", "Alice", "Carla", "Frank", "Marcus"],
    )
    filler()
    filler()

    add(
        "Sara dropped by my apartment Saturday — first time meeting my partner Jamie.",
        "meet",
        ["User", "Sara", "Jamie"],
    )
    filler()
    filler()

    # Frank only at Stripe offsite, never with others I've talked to outside work
    add("Frank works at Stripe in security.", "context", ["Frank", "Stripe"])
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
            kind="met",
            question="Have Alice and Bob met?",
            expected_contains=["yes", "dinner", "Joe"],
        ),
        Question(
            qid="Q02",
            kind="met",
            question="Have Carla and Dave been in a meeting together?",
            expected_contains=["yes", "Zoom", "Q3"],
        ),
        Question(
            qid="Q03",
            kind="met",
            question="Has Sara met User's partner Jamie?",
            expected_contains=["yes", "Saturday"],
        ),
        Question(
            qid="Q04",
            kind="not_met",
            question="Have Bob and Eve been in a meeting together (based on what I've told you)?",
            expected_contains=["no", "not"],
        ),
        Question(
            qid="Q05",
            kind="met",
            question="Has Alice met Carla?",
            expected_contains=["yes", "offsite"],
        ),
        Question(
            qid="Q06",
            kind="not_met",
            question="Have Bob and Frank ever been in the same room?",
            expected_contains=["no", "not"],
        ),
        Question(
            qid="Q07",
            kind="who_at",
            question="Who was at the Stripe offsite?",
            expected_contains=["Alice", "Carla", "Frank", "Marcus"],
        ),
        Question(
            qid="Q08",
            kind="who_at",
            question="Who was on the Q3 plan Zoom call?",
            expected_contains=["Carla", "Dave", "Eve"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
