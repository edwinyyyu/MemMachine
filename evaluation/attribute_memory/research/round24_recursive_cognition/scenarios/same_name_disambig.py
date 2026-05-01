"""same_name_disambig — three different Alices in three contexts.

Tests the writer/DSU's ability to keep distinct entities with the same surface
name. R23/R24's DSU starts each mention in its own class; the writer must
NOT merge mentions of "Alice" that refer to different people, while still
merging same-Alice mentions across turns.

Three Alices:
  - Alice #1: User's new neighbor (nurse, lives across the hall)
  - Alice #2: User's colleague on the platform team (engineer)
  - Alice #3: User's friend Sara's younger sister (architect, getting married)

Each Alice has its own context arc. Entity resolution must be context-aware,
not surface-only.

Pairs are interleaved so the writer can't just "first Alice wins."

Total ~70 turns.
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
    "Coffee was good this morning.",
    "Long day on calls.",
    "Stomach hurts a bit.",
    "Weather's nice today.",
    "Slack is being laggy again.",
    "Inbox is a mess.",
    "412 unread emails — yikes.",
    "Need to grab lunch soon.",
    "Going to take a quick break.",
    "Tired today.",
    "Working from a cafe.",
    "Heard a fire alarm earlier.",
    "Got a haircut yesterday.",
    "Kitchen is a mess again.",
]


def generate() -> list[Turn]:
    turns: list[Turn] = []
    fc = 0

    def add(text: str, kind: str, mentions: list[str] | None = None):
        turns.append(
            Turn(
                idx=len(turns) + 1,
                text=text,
                kind=kind,
                mentions=mentions or [],
            )
        )

    def filler():
        nonlocal fc
        add(FILLER[fc % len(FILLER)], "filler")
        fc += 1

    # Lead-in
    filler()
    filler()

    # ============= ARC 1: Alice the neighbor =============
    add(
        "New neighbor moved in across the hall — she's a nurse at Mt Sinai.",
        "intro",
        ["User"],
    )
    filler()
    filler()
    add(
        "My neighbor introduced herself — her name is Alice.",
        "name_reveal",
        ["User", "Alice"],
    )
    filler()
    add(
        "Alice mentioned the building's hallway repaint is coming.",
        "detail",
        ["User", "Alice"],
    )
    filler()
    filler()

    # ============= ARC 2: Alice the colleague =============
    # (totally different Alice, work context)
    add(
        "There's a colleague on the platform team I keep hearing good things about.",
        "intro",
        ["User"],
    )
    filler()
    filler()
    add(
        "That colleague is Alice — she fixed the deploy bug last sprint.",
        "name_reveal",
        ["User", "Alice"],
    )
    filler()
    add(
        "Alice on platform team is a senior engineer apparently.",
        "detail",
        ["User", "Alice"],
    )
    filler()
    filler()

    # ============= ARC 3: Alice, Sara's sister (different person) =============
    add(
        "Sara's younger sister is getting married next month.",
        "intro",
        ["User", "Sara"],
    )
    filler()
    filler()
    add(
        "Sara's sister — her name's Alice — is an architect.",
        "name_reveal",
        ["User", "Sara", "Alice"],
    )
    filler()
    add("Alice is getting married in Vermont actually.", "detail", ["User", "Alice"])
    filler()
    filler()

    # ============= Cross-references that should NOT confuse =============
    add(
        "Alice — the neighbor — invited me to a building meeting.",
        "context_alice",
        ["User", "Alice"],
    )
    filler()
    add(
        "Coffee with Alice from work tomorrow about the platform refactor.",
        "context_alice",
        ["User", "Alice"],
    )
    filler()
    add(
        "Got Alice's wedding invite in the mail today.",
        "context_alice",
        ["User", "Alice"],
    )
    filler()
    filler()

    # ============= Update facts about each =============
    add(
        "Alice the neighbor is hosting a dinner this weekend.",
        "update",
        ["User", "Alice"],
    )
    filler()
    add(
        "Alice on platform team got promoted to staff engineer.",
        "update",
        ["User", "Alice"],
    )
    filler()
    add(
        "Alice (Sara's sister) is having the rehearsal in Burlington.",
        "update",
        ["User", "Sara", "Alice"],
    )
    filler()
    filler()
    filler()

    for i, t in enumerate(turns, start=1):
        t.idx = i
    return turns


@dataclass
class GroundTruth:
    facts: list[dict] = field(default_factory=list)


def ground_truth(turns: list[Turn]) -> GroundTruth:
    return GroundTruth(
        facts=[
            {"alice_role": "neighbor", "job": "nurse", "place": "Mt Sinai"},
            {"alice_role": "colleague", "team": "platform", "title": "staff engineer"},
            {
                "alice_role": "sara's sister",
                "job": "architect",
                "wedding_place": "Vermont",
                "rehearsal_place": "Burlington",
            },
        ]
    )


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
            kind="disambig",
            question="What does Alice the neighbor do for a living?",
            expected_contains=["nurse"],
        ),
        Question(
            qid="Q02",
            kind="disambig",
            question="What team is User's colleague Alice on?",
            expected_contains=["platform"],
        ),
        Question(
            qid="Q03",
            kind="disambig",
            question="What is Sara's sister Alice's job?",
            expected_contains=["architect"],
        ),
        Question(
            qid="Q04",
            kind="disambig",
            question="Where is Alice's wedding (Sara's sister Alice)?",
            expected_contains=["Vermont"],
        ),
        Question(
            qid="Q05",
            kind="disambig",
            question="Where is the rehearsal for the wedding?",
            expected_contains=["Burlington"],
        ),
        Question(
            qid="Q06",
            kind="disambig",
            question="Did Alice the colleague get promoted?",
            expected_contains=["yes", "staff"],
        ),
        Question(
            qid="Q07",
            kind="disambig",
            question="Where does Alice the neighbor work?",
            expected_contains=["Mt Sinai"],
        ),
        Question(
            qid="Q08",
            kind="entity_count",
            question="How many distinct people named Alice does User know? Answer with a number.",
            expected_contains=["3", "three"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    gt = ground_truth(turns)
    qs = build_questions(gt)
    print(f"turns: {len(turns)}")
    for t in turns[:30]:
        print(f"  t{t.idx} [{t.kind}] {t.text}")
    print(f"# questions: {len(qs)}")
    for q in qs:
        print(f"  {q.qid}: {q.question} -> {q.expected_contains}")
