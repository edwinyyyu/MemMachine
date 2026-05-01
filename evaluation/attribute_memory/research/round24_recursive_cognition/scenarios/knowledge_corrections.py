"""knowledge_corrections — User corrects prior statements explicitly.

User states something earlier; later says "actually, I was wrong, it's Y."
Memory should reflect the corrected version, not the original.

Different from silent_contradiction (which is implicit). Here the user
explicitly retracts.

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

    # Correction 1: wrong about Marcus's role
    add(
        "Marcus is the new CTO at Acme — heard it from a friend.",
        "claim",
        ["User", "Marcus", "Acme"],
    )
    filler()
    filler()
    add(
        "Actually, I was wrong about Marcus — he's the head of engineering, not CTO.",
        "correction",
        ["User", "Marcus"],
    )
    filler()
    filler()

    # Correction 2: wrong about Sara's wedding date
    add("Sara's wedding is in April.", "claim", ["User", "Sara"])
    filler()
    filler()
    add(
        "Correction: Sara's wedding got moved to June, not April.",
        "correction",
        ["User", "Sara"],
    )
    filler()
    filler()

    # Correction 3: wrong about User's own job start
    add("I'm starting at Notion next Monday.", "claim", ["User", "Notion"])
    filler()
    filler()
    add(
        "Update: Notion start date pushed back two weeks. Now starting end of month.",
        "correction",
        ["User", "Notion"],
    )
    filler()
    filler()

    # Correction 4: wrong about Daisy's breed
    add("Daisy's a golden retriever puppy.", "claim", ["User", "Daisy"])
    filler()
    filler()
    add(
        "Vet said Daisy is actually a golden retriever / lab mix, not pure golden.",
        "correction",
        ["User", "Daisy"],
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
            kind="post_correction",
            question="What is Marcus's role at Acme?",
            expected_contains=["head of engineering"],
            expected_absent=["CTO"],
        ),
        Question(
            qid="Q02",
            kind="post_correction",
            question="When is Sara's wedding?",
            expected_contains=["June"],
            expected_absent=["April"],
        ),
        Question(
            qid="Q03",
            kind="post_correction",
            question="When does User start at Notion?",
            expected_contains=["end of month", "two weeks"],
        ),
        Question(
            qid="Q04",
            kind="post_correction",
            question="What breed is Daisy?",
            expected_contains=["mix", "lab"],
        ),
        Question(
            qid="Q05",
            kind="correction_aware",
            question="Did User have to correct anything they said about Marcus?",
            expected_contains=["yes", "CTO", "head"],
        ),
        Question(
            qid="Q06",
            kind="correction_aware",
            question="Has the Sara wedding date been corrected from an earlier statement?",
            expected_contains=["yes", "April", "June"],
        ),
        Question(
            qid="Q07",
            kind="post_correction",
            question="Is Marcus the CTO at Acme?",
            expected_contains=["no"],
        ),
        Question(
            qid="Q08",
            kind="post_correction",
            question="Is Daisy a pure golden retriever?",
            expected_contains=["no", "mix"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
