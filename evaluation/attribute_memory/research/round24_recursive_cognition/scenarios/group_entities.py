"""group_entities — group references that sometimes mean specific people.

User refers to "the team" — sometimes meaning all, sometimes a subset.
Tests entity polymorphism and contextual disambiguation.

Setup:
  - User's team has 4 members: Alice, Bob, Carla, Dave.
  - User refers to "the team" in different contexts.
  - Sometimes only specific members were involved.

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


FILLER = [
    "Long day.",
    "Tired.",
    "Slack laggy.",
    "Inbox messy.",
    "Need lunch.",
    "Heard a siren.",
    "Coffee was good.",
]


def generate() -> list[Turn]:
    turns: list[Turn] = []
    fc = 0

    def add(text: str, kind: str, mentions: list[str] | None = None):
        turns.append(
            Turn(idx=len(turns) + 1, text=text, kind=kind, mentions=mentions or [])
        )

    def filler():
        nonlocal fc
        add(FILLER[fc % len(FILLER)], "filler")
        fc += 1

    filler()
    filler()

    # Establish team members
    add(
        "My team has four people: Alice, Bob, Carla, and Dave.",
        "intro",
        ["User", "Alice", "Bob", "Carla", "Dave"],
    )
    filler()
    add("Alice is the senior on the platform team.", "detail", ["Alice"])
    filler()
    add("Bob is our infrastructure engineer.", "detail", ["Bob"])
    filler()
    add("Carla joined two months ago — front end.", "detail", ["Carla"])
    filler()
    add("Dave is the team lead.", "detail", ["Dave"])
    filler()
    filler()

    # Group references — meaning all
    add(
        "The whole team came to lunch on Friday — first time all four together.",
        "all_team",
        ["User", "Alice", "Bob", "Carla", "Dave"],
    )
    filler()
    filler()

    # Group references — meaning subset (named subset)
    add(
        "Bob and Carla worked late on the deploy automation last night.",
        "subset",
        ["Bob", "Carla"],
    )
    filler()
    filler()

    # Implicit subset (only some on call)
    add(
        "Alice and Dave handled the on-call rotation this week.",
        "subset",
        ["Alice", "Dave"],
    )
    filler()
    filler()

    # Group reference but ambiguous
    add("The team is wrapping up sprint 23.", "ambiguous_team", ["User"])
    filler()
    add(
        "Got positive feedback from the team about the new sprint cadence.",
        "ambiguous_team",
        ["User"],
    )
    filler()
    filler()

    # New member joins
    add(
        "Eve joined the team last Monday — security background.", "team_change", ["Eve"]
    )
    filler()
    add(
        "Now the team is five — Alice, Bob, Carla, Dave, Eve.",
        "team_state",
        ["User", "Alice", "Bob", "Carla", "Dave", "Eve"],
    )
    filler()
    filler()

    for i, t in enumerate(turns, start=1):
        t.idx = i
    return turns


@dataclass
class GroundTruth:
    pass


def ground_truth(turns: list[Turn]) -> GroundTruth:
    return GroundTruth()


@dataclass
class Question:
    qid: str
    kind: str
    question: str
    expected_contains: list[str]
    expected_absent: list[str] = field(default_factory=list)


def build_questions(gt: GroundTruth) -> list[Question]:
    return [
        Question(
            qid="Q01",
            kind="group_size",
            question="How many people are currently on User's team?",
            expected_contains=["5", "five"],
        ),
        Question(
            qid="Q02",
            kind="group_membership",
            question="Who are the members of User's team currently? List names.",
            expected_contains=["Alice", "Bob", "Carla", "Dave", "Eve"],
        ),
        Question(
            qid="Q03",
            kind="role",
            question="Who is the team lead?",
            expected_contains=["Dave"],
        ),
        Question(
            qid="Q04",
            kind="role",
            question="Who is the infrastructure engineer?",
            expected_contains=["Bob"],
        ),
        Question(
            qid="Q05",
            kind="subset",
            question="Who worked on the deploy automation late last night?",
            expected_contains=["Bob", "Carla"],
        ),
        Question(
            qid="Q06",
            kind="subset",
            question="Who handled on-call this week?",
            expected_contains=["Alice", "Dave"],
        ),
        Question(
            qid="Q07",
            kind="state_change",
            question="Did anyone new join User's team recently?",
            expected_contains=["yes", "Eve"],
        ),
        Question(
            qid="Q08",
            kind="role",
            question="What's Eve's background?",
            expected_contains=["security"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
