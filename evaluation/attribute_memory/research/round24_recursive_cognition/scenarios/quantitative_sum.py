"""quantitative_sum — totals/deltas requiring computation across facts.

User mentions individual amounts. Question requires summing them.

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

    # Spending events
    add("Spent $85 on groceries Monday.", "spend", ["User"])
    filler()
    add("Date night dinner Tuesday was $120.", "spend", ["User"])
    filler()
    add("Wednesday: paid $45 for the climbing gym monthly fee.", "spend", ["User"])
    filler()
    add("Thursday lunch with team — my share was $30.", "spend", ["User"])
    filler()
    add("Saturday: $200 for new running shoes.", "spend", ["User"])
    filler()
    filler()

    # Income / savings
    add("Got $1,500 freelance check this month.", "income", ["User"])
    filler()
    add("Saved $800 to the emergency fund this month.", "save", ["User"])
    filler()
    filler()

    # Travel miles
    add("Flew 4,500 miles for the Tokyo conference.", "travel", ["User"])
    filler()
    add("Then 1,200 miles to LA for Sara's wedding.", "travel", ["User"])
    filler()
    add("Plus 350-mile drive for the family reunion.", "travel", ["User"])
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
            kind="sum",
            question="How much did User spend in total this past week (groceries, dinners, gym, lunch, shoes)?",
            expected_contains=["480"],
        ),
        Question(
            qid="Q02",
            kind="sum",
            question="How much was User's biggest single expense recently?",
            expected_contains=["200", "shoes"],
        ),
        Question(
            qid="Q03",
            kind="net",
            question="What was User's net cash flow this month between freelance income and savings deposit?",
            expected_contains=["700"],
        ),
        Question(
            qid="Q04",
            kind="sum",
            question="Total flying miles User did between Tokyo and LA?",
            expected_contains=["5,700", "5700"],
        ),
        Question(
            qid="Q05",
            kind="sum",
            question="Total miles User traveled (flights + drive)?",
            expected_contains=["6,050", "6050"],
        ),
        Question(
            qid="Q06",
            kind="recall",
            question="What did User spend on Tuesday?",
            expected_contains=["120", "date night", "dinner"],
        ),
        Question(
            qid="Q07",
            kind="recall",
            question="What was the climbing gym fee?",
            expected_contains=["45"],
        ),
        Question(
            qid="Q08",
            kind="extreme",
            question="What's the longest single trip User took recently?",
            expected_contains=["Tokyo", "4,500", "4500"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
