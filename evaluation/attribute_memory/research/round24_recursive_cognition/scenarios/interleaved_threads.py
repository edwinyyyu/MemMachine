"""interleaved_threads — work topics + home topics interleaved.

User alternates between work and home contexts. Memory must let queries
about each thread retrieve the right thread without bleed.

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


FILLER = ["Coffee.", "Tired."]


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

    # interleave work + home
    add(
        "Marcus is reviewing my deploy strategy proposal at Stripe.",
        "work",
        ["User", "Marcus", "Stripe"],
    )
    filler()
    add(
        "My partner Jamie is looking at houses in Park Slope.",
        "home",
        ["User", "Jamie"],
    )
    filler()
    add(
        "Got my staff promotion approved at Stripe — official next quarter.",
        "work",
        ["User", "Stripe"],
    )
    filler()
    add(
        "Jamie loved the brownstone on 8th Ave — going back this weekend.",
        "home",
        ["User", "Jamie"],
    )
    filler()
    add("Started mentoring Carla on the platform team.", "work", ["User", "Carla"])
    filler()
    add("Made an offer on the brownstone — $1.4M, all cash.", "home", ["User"])
    filler()
    add("Q3 platform OKRs locked: API SLO 99.95%.", "work", ["User"])
    filler()
    add("Brownstone offer accepted! Closing in 45 days.", "home", ["User"])
    filler()
    add(
        "Carla shipped her first PR solo — proud of her growth.",
        "work",
        ["User", "Carla"],
    )
    filler()
    add(
        "Jamie's planning the move logistics — boxes arriving next week.",
        "home",
        ["User", "Jamie"],
    )
    filler()
    add("API SLO holding well — 99.97% MTD.", "work", ["User"])
    filler()
    add("Officially closed on the brownstone today.", "home", ["User"])
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
            kind="work",
            question="What's User's promotion status at Stripe?",
            expected_contains=["staff", "approved"],
        ),
        Question(
            qid="Q02",
            kind="home",
            question="Did User and Jamie buy a house?",
            expected_contains=["yes", "brownstone"],
        ),
        Question(
            qid="Q03",
            kind="home_specific",
            question="What was the price of the house?",
            expected_contains=["1.4M", "1.4 million"],
        ),
        Question(
            qid="Q04",
            kind="work_specific",
            question="Who is User mentoring at work?",
            expected_contains=["Carla"],
        ),
        Question(
            qid="Q05",
            kind="work_specific",
            question="What's User's team's API SLO target?",
            expected_contains=["99.95"],
        ),
        Question(
            qid="Q06",
            kind="home_specific",
            question="Where is the new house?",
            expected_contains=["Park Slope", "8th Ave"],
        ),
        Question(
            qid="Q07",
            kind="work",
            question="Who is reviewing User's deploy strategy proposal?",
            expected_contains=["Marcus"],
        ),
        Question(
            qid="Q08",
            kind="thread_separation",
            question="Did Carla help with the house purchase?",
            expected_contains=["no"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
