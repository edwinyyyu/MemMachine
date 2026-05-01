"""confidence_assessment — distinguish hedged from confident statements.

User's statements vary in certainty: definite, hedged, speculative.
Memory should preserve confidence cues so reader can answer
"how sure was X about Y?"

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


FILLER = ["Long day.", "Coffee good.", "Tired."]


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

    # Confident
    add(
        "Alice is 100% certain the deadline is March 30. She reviewed the contract herself.",
        "confident",
        ["Alice"],
    )
    filler()
    filler()

    # Hedged
    add(
        "Bob thinks the deadline is probably March 30, but isn't sure — he didn't read the fine print.",
        "hedged",
        ["Bob"],
    )
    filler()
    filler()

    # Vague
    add(
        "Carla muttered something about late March for the deadline.",
        "vague",
        ["Carla"],
    )
    filler()
    filler()

    # Confident different topic
    add("Dave swears the new release will go out next Friday.", "confident", ["Dave"])
    filler()
    filler()

    # Speculative
    add(
        "Eve speculated the new release MIGHT slip if QA finds anything.",
        "speculative",
        ["Eve"],
    )
    filler()
    filler()

    # User's own confidence
    add("I'm 100% set on moving to Seattle in June.", "user_confident", ["User"])
    filler()
    filler()
    add(
        "Wondering if the move should actually be August instead. Maybe.",
        "user_hedged",
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
            kind="confidence",
            question="How sure is Alice about the March 30 deadline?",
            expected_contains=["100", "certain", "very sure"],
        ),
        Question(
            qid="Q02",
            kind="confidence",
            question="How sure is Bob about the March 30 deadline?",
            expected_contains=["not sure", "probably", "isn't"],
        ),
        Question(
            qid="Q03",
            kind="confidence",
            question="How confident is Carla about the deadline?",
            expected_contains=["vague", "muttered", "low"],
        ),
        Question(
            qid="Q04",
            kind="confidence",
            question="Is Dave confident the release will go out next Friday?",
            expected_contains=["yes", "swears"],
        ),
        Question(
            qid="Q05",
            kind="confidence",
            question="Did Eve assert the release WILL slip, or was it speculation?",
            expected_contains=["speculation", "might"],
        ),
        Question(
            qid="Q06",
            kind="user_confidence",
            question="Is User definitely moving to Seattle in June?",
            expected_contains=["maybe", "August", "wondering"],
        ),
        Question(
            qid="Q07",
            kind="contrast",
            question="Whose deadline opinion is most reliable: Alice, Bob, or Carla?",
            expected_contains=["Alice"],
        ),
        Question(
            qid="Q08",
            kind="contrast",
            question="Whose release prediction is more confident: Dave or Eve?",
            expected_contains=["Dave"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
