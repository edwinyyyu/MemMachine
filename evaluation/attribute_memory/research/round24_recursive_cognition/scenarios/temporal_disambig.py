"""temporal_disambig — multiple events on different days/times.

User mentions multiple meetings, multiple visits, etc. Later question
references "the meeting on Tuesday" or "last week's incident" requiring
temporal disambiguation.

Tests whether the system tracks event timing well enough to distinguish
between similar events at different times.

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
    "Need lunch.",
    "Coffee was good.",
    "Heard a siren.",
    "Cafe again.",
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

    # ===== Multiple meetings on different days =====
    add(
        "Had a 1:1 with Marcus on Monday — discussed the roadmap.",
        "meeting",
        ["User", "Marcus"],
    )
    filler()
    add("Tuesday's standup ran long because of the deploy issue.", "meeting", ["User"])
    filler()
    add(
        "Wednesday's team retrospective went well — closed 8 tickets.",
        "meeting",
        ["User"],
    )
    filler()
    add(
        "Thursday afternoon I had coffee with Priya about the new project.",
        "meeting",
        ["User", "Priya"],
    )
    filler()
    add("Friday's all-hands was about Q2 priorities.", "meeting", ["User"])
    filler()
    filler()

    # ===== Multiple incidents =====
    add("Pager went off at 2am Monday — auth service down.", "incident", ["User"])
    filler()
    add(
        "Wednesday afternoon: deploy pipeline broke for 30 minutes.",
        "incident",
        ["User"],
    )
    filler()
    add(
        "Friday evening: cache eviction bug caused stale reads. Fixed in 20 min.",
        "incident",
        ["User"],
    )
    filler()
    filler()

    # ===== Visits to a place at different times =====
    add("Lunch at Joe's Pizza on Tuesday — solo, quick bite.", "visit", ["User"])
    filler()
    add("Joe's Pizza again Friday — group dinner with the team.", "visit", ["User"])
    filler()
    add("Sunday brunch at Joe's Pizza — sat outside.", "visit", ["User"])
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
            kind="temporal",
            question="What did User discuss with Marcus on Monday?",
            expected_contains=["roadmap"],
        ),
        Question(
            qid="Q02",
            kind="temporal",
            question="Which day did User meet with Priya?",
            expected_contains=["Thursday"],
        ),
        Question(
            qid="Q03",
            kind="temporal_count",
            question="How many incidents did User have last week?",
            expected_contains=["3", "three"],
        ),
        Question(
            qid="Q04",
            kind="temporal",
            question="What happened at Tuesday's standup?",
            expected_contains=["deploy"],
        ),
        Question(
            qid="Q05",
            kind="temporal_count",
            question="How many times did User go to Joe's Pizza?",
            expected_contains=["3", "three"],
        ),
        Question(
            qid="Q06",
            kind="temporal_specific",
            question="What was the Friday Joe's Pizza visit about?",
            expected_contains=["group", "team", "dinner"],
        ),
        Question(
            qid="Q07",
            kind="temporal",
            question="What was the Friday evening incident?",
            expected_contains=["cache", "stale"],
        ),
        Question(
            qid="Q08",
            kind="temporal",
            question="What was the Wednesday meeting?",
            expected_contains=["retrospective"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
