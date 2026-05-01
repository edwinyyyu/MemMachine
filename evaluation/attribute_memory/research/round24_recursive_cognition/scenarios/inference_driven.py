"""inference_driven — world knowledge determines what to retrieve.

User states some facts. A later question requires JOINING the user's
facts with general world knowledge to answer.

E.g., "I'm allergic to shellfish" + question "Can I eat at this seafood
restaurant?" requires inferring that seafood restaurants have shellfish.

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


FILLER = ["Long day.", "Tired.", "Slack laggy.", "Coffee good."]


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

    # Diet-related facts
    add("I'm allergic to shellfish — anaphylaxis risk.", "fact", ["User"])
    filler()
    add("I've been vegan for 3 years.", "fact", ["User"])
    filler()
    filler()

    # Travel-related facts
    add("My passport expires next March.", "fact", ["User"])
    filler()
    add("I have TSA PreCheck through 2027.", "fact", ["User"])
    filler()
    filler()

    # Work-related facts
    add("I'm on call every other Sunday for the next quarter.", "fact", ["User"])
    filler()
    add("My main project deadline is end of November.", "fact", ["User"])
    filler()
    filler()

    # Health facts
    add("I'm on a beta blocker for my heart.", "fact", ["User"])
    filler()
    add(
        "I had ACL surgery last year — knee still gets sore on stairs.",
        "fact",
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
            kind="world_knowledge",
            question="Should User eat at a seafood restaurant? Why or why not?",
            expected_contains=["no", "shellfish", "allergic"],
        ),
        Question(
            qid="Q02",
            kind="world_knowledge",
            question="Can User eat eggs?",
            expected_contains=["no", "vegan"],
        ),
        Question(
            qid="Q03",
            kind="world_knowledge",
            question="Can User travel internationally next April?",
            expected_contains=["no", "passport", "expire"],
        ),
        Question(
            qid="Q04",
            kind="world_knowledge",
            question="Should User schedule a Sunday brunch in three weeks?",
            expected_contains=["may", "depends", "on call", "Sunday"],
        ),
        Question(
            qid="Q05",
            kind="world_knowledge",
            question="Should User do high-impact running?",
            expected_contains=["no", "knee", "ACL"],
        ),
        Question(
            qid="Q06",
            kind="world_knowledge",
            question="Can User start a heavy gym program right away?",
            expected_contains=["caution", "knee", "ACL"],
        ),
        Question(
            qid="Q07",
            kind="schedule_join",
            question="Will User be free for an off-site at end of November?",
            expected_contains=["no", "deadline"],
        ),
        Question(
            qid="Q08",
            kind="world_knowledge",
            question="At airport security, does User need to remove shoes?",
            expected_contains=["no", "PreCheck"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
