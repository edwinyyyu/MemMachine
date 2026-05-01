"""preference_drift — User's opinions and tastes change over time.

User states preferences ("love sushi"), later changes ("sick of sushi").
Memory should track both the current preference and the trajectory.

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


FILLER = ["Long day.", "Tired.", "Coffee good.", "Need lunch.", "Slack laggy."]


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

    # Sushi: love → tired
    add("I could eat sushi every day. Best food in the world.", "preference", ["User"])
    filler()
    filler()
    add("Sushi twice this week. Still love it.", "preference_continuation", ["User"])
    filler()
    add(
        "Honestly burnt out on sushi after weeks of it. Need a break.",
        "preference_change",
        ["User"],
    )
    filler()
    filler()

    # Music: classical → ambient
    add(
        "Lately I've been all about classical music. Mostly Bach.",
        "preference",
        ["User"],
    )
    filler()
    filler()
    add(
        "Found a new ambient artist I'm obsessed with — moved on from classical for now.",
        "preference_change",
        ["User"],
    )
    filler()
    filler()

    # Coffee: light roast → dark roast
    add(
        "Light roast all day. Anything darker tastes burnt to me.",
        "preference",
        ["User"],
    )
    filler()
    filler()
    add(
        "Tried a really nice dark roast at the new cafe — converted. Dark roast era now.",
        "preference_change",
        ["User"],
    )
    filler()
    filler()

    # Travel: cities → nature
    add(
        "I prefer city trips to nature trips — give me a museum over a hiking trail.",
        "preference",
        ["User"],
    )
    filler()
    filler()
    add(
        "Just got back from Patagonia. Cities feel boring now.",
        "preference_change",
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
            kind="current",
            question="Does User currently want sushi?",
            expected_contains=["no", "burnt out", "break"],
        ),
        Question(
            qid="Q02",
            kind="current",
            question="What music genre is User into now?",
            expected_contains=["ambient"],
        ),
        Question(
            qid="Q03",
            kind="current",
            question="What roast does User prefer now?",
            expected_contains=["dark"],
        ),
        Question(
            qid="Q04",
            kind="current",
            question="Does User prefer city or nature trips lately?",
            expected_contains=["nature"],
        ),
        Question(
            qid="Q05",
            kind="historical",
            question="What was User's earlier music preference?",
            expected_contains=["classical", "Bach"],
        ),
        Question(
            qid="Q06",
            kind="historical",
            question="Did User used to prefer light roast?",
            expected_contains=["yes", "light"],
        ),
        Question(
            qid="Q07",
            kind="trajectory",
            question="Has User's view on sushi changed? How?",
            expected_contains=["yes", "loved", "burnt"],
        ),
        Question(
            qid="Q08",
            kind="trajectory",
            question="Has User's preference about travel evolved?",
            expected_contains=["yes", "cities", "nature"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
