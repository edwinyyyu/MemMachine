"""speech_acts — distinguish hypothetical/speculation/question from fact.

User says "Maybe Alice will join" (speculation), "I think Bob is leaving"
(belief), "Is Carla here?" (question). Memory should NOT treat these as
asserted facts.

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

    # Speculation that turns out false
    add(
        "Maybe Alice will leave the company — heard rumors.",
        "speculation",
        ["User", "Alice"],
    )
    filler()
    filler()
    add(
        "Talked to Alice today — she's staying. The rumors were wrong.",
        "fact",
        ["User", "Alice"],
    )
    filler()
    filler()

    # Belief that's correct (different speech act, same outcome)
    add("I think Bob is going on parental leave next month.", "belief", ["User", "Bob"])
    filler()
    filler()
    add("Bob's leave starts March 15th — confirmed.", "fact", ["User", "Bob"])
    filler()
    filler()

    # Question (no assertion at all)
    add("Wait, is Carla actually a manager now?", "question", ["User", "Carla"])
    filler()
    filler()
    add(
        "Carla's not a manager — she's a senior engineer. Got that wrong in my head.",
        "fact",
        ["User", "Carla"],
    )
    filler()
    filler()

    # Hypothetical
    add("If Dave gets the promo, he'll move to NY.", "hypothetical", ["User", "Dave"])
    filler()
    filler()
    add("Dave didn't get the promo — staying put.", "fact", ["User", "Dave"])
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
            kind="actual",
            question="Did Alice leave the company?",
            expected_contains=["no", "staying"],
        ),
        Question(
            qid="Q02",
            kind="actual",
            question="When does Bob's parental leave start?",
            expected_contains=["March 15"],
        ),
        Question(
            qid="Q03",
            kind="actual",
            question="Is Carla a manager?",
            expected_contains=["no", "senior engineer"],
        ),
        Question(
            qid="Q04",
            kind="actual",
            question="Did Dave move to NY?",
            expected_contains=["no"],
        ),
        Question(
            qid="Q05",
            kind="speech_act",
            question="What did User initially suspect about Alice?",
            expected_contains=["leave", "rumors"],
        ),
        Question(
            qid="Q06",
            kind="speech_act",
            question="What was User's earlier wrong belief about Carla?",
            expected_contains=["manager"],
        ),
        Question(
            qid="Q07",
            kind="speech_act",
            question="What was conditional on Dave getting the promo?",
            expected_contains=["NY", "move"],
        ),
        Question(
            qid="Q08",
            kind="speech_act",
            question="Was User's belief about Bob correct?",
            expected_contains=["yes", "right", "March"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
