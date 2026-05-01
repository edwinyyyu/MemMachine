"""evolving_terminology — same concept renamed over time.

Stresses DSU's merge-across-rename ability. Concepts are introduced under
one label, then later renamed (sometimes multiple times). Questions ask
about the concept under any of its labels.

Scenarios in this benchmark:
  - "gamma" → "v2" → "production auth"
  - "the dashboard refactor" → "Phoenix" → "the new admin panel"
  - "Project Helios" → "the analytics pipeline"

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
    "Cafe again.",
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

    # ===== gamma → v2 → production auth =====
    add(
        "Calling the new auth system 'gamma' for now — internal codename.",
        "intro",
        ["User"],
    )
    filler()
    add("Gamma is replacing the old session-based login.", "detail", ["User"])
    filler()
    filler()
    add(
        "Officially renamed gamma to v2 — marketing wanted something simpler.",
        "rename",
        ["User"],
    )
    filler()
    add("V2 is going through QA this week.", "detail", ["User"])
    filler()
    filler()
    add(
        "V2 shipped to prod today as the new production auth system.",
        "rename",
        ["User"],
    )
    filler()
    add(
        "Production auth had a brief outage last night — investigating.",
        "event",
        ["User"],
    )
    filler()
    filler()

    # ===== dashboard refactor → Phoenix → admin panel =====
    add("Starting the dashboard refactor next sprint.", "intro", ["User"])
    filler()
    add("Eng team voted to call the dashboard refactor 'Phoenix.'", "rename", ["User"])
    filler()
    add("Phoenix is reusing 60% of the old code.", "detail", ["User"])
    filler()
    filler()
    add("Phoenix shipped — it's now the new admin panel.", "rename", ["User"])
    filler()
    add("New admin panel got positive feedback from internal users.", "event", ["User"])
    filler()
    filler()

    # ===== Project Helios → analytics pipeline =====
    add(
        "Kicking off Project Helios — sounds dramatic but it's just analytics.",
        "intro",
        ["User"],
    )
    filler()
    add("Helios is the analytics pipeline rewrite.", "rename", ["User"])
    filler()
    add("The analytics pipeline is now in beta.", "detail", ["User"])
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
            kind="rename",
            question="What was originally called 'gamma'?",
            expected_contains=["production auth", "v2", "auth"],
        ),
        Question(
            qid="Q02",
            kind="rename",
            question="Has v2 had any outages?",
            expected_contains=["yes", "outage"],
        ),
        Question(
            qid="Q03",
            kind="rename",
            question="What is Phoenix in this context?",
            expected_contains=["dashboard", "admin panel"],
        ),
        Question(
            qid="Q04",
            kind="rename",
            question="Did the dashboard refactor ship?",
            expected_contains=["yes", "shipped", "admin"],
        ),
        Question(
            qid="Q05",
            kind="rename",
            question="What's the analytics pipeline's internal codename?",
            expected_contains=["Helios"],
        ),
        Question(
            qid="Q06",
            kind="rename",
            question="What is Project Helios?",
            expected_contains=["analytics", "pipeline"],
        ),
        Question(
            qid="Q07",
            kind="state",
            question="What's the current state of v2?",
            expected_contains=["prod", "shipped"],
        ),
        Question(
            qid="Q08",
            kind="state_evolved",
            question="What is the current name of what started as 'gamma'?",
            expected_contains=["production auth"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
