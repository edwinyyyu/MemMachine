"""multi_session — explicit session breaks; user references prior sessions.

Tests memory continuity across what would be separate sessions in a real
agent (the underlying corpus is one stream, but turn text references
"last week" / "the other day" / "earlier this month").

Scenarios:
  - Session 1: introduces a project; session 2 references it
  - Session 1: User states a goal; session 3 reports progress
  - Session 1-3: ongoing book-club; multiple events referenced

Total ~80 turns.
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
    "Need lunch.",
    "Slack laggy.",
    "Cafe again.",
    "Heard a siren.",
    "Coffee good.",
    "Inbox messy.",
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

    # ===== SESSION 1 — first chat =====
    add("[NEW SESSION — Monday morning]", "session_start")
    add(
        "Starting Project Atlas — the analytics dashboard rewrite.",
        "intro",
        ["User", "Atlas"],
    )
    filler()
    add("Initial scope: 6 weeks, 3 engineers.", "detail", ["User"])
    filler()
    filler()
    add("Goal: ship Atlas v1 by end of Q2.", "goal", ["User", "Atlas"])
    filler()
    add(
        "Started reading 'Designing Data-Intensive Applications' for the Atlas work.",
        "task",
        ["User"],
    )
    filler()
    add("Booking a kickoff meeting with the team for Wednesday.", "task", ["User"])
    filler()
    filler()

    # ===== SESSION 2 — next day =====
    add("[NEW SESSION — Tuesday afternoon, day after]", "session_start")
    add(
        "Yesterday's plan for Atlas got revised — actually 8 weeks not 6.",
        "update_prior",
        ["User", "Atlas"],
    )
    filler()
    add(
        "Ran into Marcus at the gym — he had thoughts on Atlas's data model.",
        "event",
        ["User", "Marcus", "Atlas"],
    )
    filler()
    add(
        "DDIA chapter 3 on indexes is super relevant to the Atlas plan.",
        "task",
        ["User", "Atlas"],
    )
    filler()
    filler()

    # ===== SESSION 3 — three weeks later =====
    add("[NEW SESSION — three weeks later]", "session_start")
    add(
        "Atlas is at the halfway mark. Not bad given the 8-week timeline.",
        "progress",
        ["User", "Atlas"],
    )
    filler()
    add("Got to the Sagas chapter in DDIA — directly applicable.", "task", ["User"])
    filler()
    filler()
    add(
        "Tom on infra wants to review the Atlas backend choices.",
        "event",
        ["User", "Tom", "Atlas"],
    )
    filler()
    filler()

    # ===== SESSION 4 — final =====
    add("[NEW SESSION — Q2 end]", "session_start")
    add(
        "Atlas v1 shipped today. Slightly past Q2 (one week over).",
        "milestone",
        ["User", "Atlas"],
    )
    filler()
    add(
        "Marcus's data-model feedback was right — saved us a refactor.",
        "reflection",
        ["User", "Marcus", "Atlas"],
    )
    filler()
    add("Finished DDIA — solid book.", "task", ["User"])
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
            kind="continuity",
            question="What is Project Atlas?",
            expected_contains=["analytics", "dashboard"],
        ),
        Question(
            qid="Q02",
            kind="continuity",
            question="How long did Project Atlas actually take?",
            expected_contains=["8 weeks", "eight"],
        ),
        Question(
            qid="Q03",
            kind="continuity",
            question="Did Atlas ship by end of Q2?",
            expected_contains=["one week", "past"],
        ),
        Question(
            qid="Q04",
            kind="continuity",
            question="Was Atlas's original scope 6 weeks or 8 weeks?",
            expected_contains=["6", "six"],
        ),
        Question(
            qid="Q05",
            kind="continuity",
            question="What book was User reading during Atlas?",
            expected_contains=["Designing", "Data", "DDIA"],
        ),
        Question(
            qid="Q06",
            kind="continuity",
            question="Did User finish DDIA?",
            expected_contains=["yes", "finished"],
        ),
        Question(
            qid="Q07",
            kind="continuity",
            question="Whose feedback did User end up valuing on Atlas's data model?",
            expected_contains=["Marcus"],
        ),
        Question(
            qid="Q08",
            kind="continuity",
            question="Did Tom contribute to Atlas?",
            expected_contains=["yes", "review", "backend"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
