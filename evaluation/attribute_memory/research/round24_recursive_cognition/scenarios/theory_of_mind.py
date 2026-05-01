"""theory_of_mind — what does someone else think, and can they be wrong?

User reports beliefs/expectations of OTHER people, separate from facts.
The system should track:
  - what X believed at time T (even if later revealed wrong)
  - the difference between what X thinks and what is actually true
  - whether X was correct in retrospect

Cognition pass should fire on these as @beliefs / @expectations.

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
    "Email pile.",
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

    # ===== Belief 1: Marcus thinks deploy is fine, then it breaks =====
    add(
        "Marcus says the new deploy is totally fine — passed all tests.",
        "belief",
        ["User", "Marcus"],
    )
    filler()
    filler()
    add(
        "Marcus is confident the migration will go smoothly.",
        "belief",
        ["User", "Marcus"],
    )
    filler()
    add("Prod is down — the migration deadlocked the users table.", "actual", ["User"])
    filler()
    add(
        "Marcus had to roll back. Lesson: he was wrong about the migration.",
        "actual",
        ["User", "Marcus"],
    )
    filler()
    filler()

    # ===== Belief 2: Priya thinks the customer is happy, but they're churning =====
    add(
        "Priya thinks Acme Corp is super happy with us — they renewed last quarter.",
        "belief",
        ["User", "Priya", "Acme"],
    )
    filler()
    filler()
    add(
        "Acme just sent a churn notice — they're moving to a competitor.",
        "actual",
        ["User", "Acme"],
    )
    filler()
    add(
        "Priya was surprised — she really thought Acme was thrilled.",
        "actual",
        ["User", "Priya", "Acme"],
    )
    filler()
    filler()

    # ===== Belief 3: Alex thinks Sara is a vegetarian (correct belief) =====
    add(
        "Alex mentioned that Sara is vegetarian — he's planning the menu around that.",
        "belief",
        ["User", "Alex", "Sara"],
    )
    filler()
    filler()
    add(
        "Sara confirmed she's been vegetarian for years — Alex was right.",
        "actual",
        ["User", "Sara", "Alex"],
    )
    filler()
    filler()

    # ===== Belief 4: User and friend disagree about Tom's role =====
    add(
        "Pat thinks Tom is the CTO at Acme but Tom's actually the head of design.",
        "belief",
        ["User", "Pat", "Tom", "Acme"],
    )
    filler()
    filler()
    add(
        "Confirmed today — Tom is head of design, not CTO. Pat was mistaken.",
        "actual",
        ["User", "Tom", "Pat"],
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
            kind="belief",
            question="What did Marcus initially think about the migration?",
            expected_contains=["fine", "smooth", "confident"],
        ),
        Question(
            qid="Q02",
            kind="correctness",
            question="Was Marcus correct about the migration?",
            expected_contains=["no", "wrong", "deadlock", "rollback"],
        ),
        Question(
            qid="Q03",
            kind="belief",
            question="What did Priya think about Acme's relationship with us?",
            expected_contains=["happy", "thrilled"],
        ),
        Question(
            qid="Q04",
            kind="correctness",
            question="Was Priya right about Acme?",
            expected_contains=["no", "wrong", "churn"],
        ),
        Question(
            qid="Q05",
            kind="belief",
            question="What does Alex think Sara's diet is?",
            expected_contains=["vegetarian"],
        ),
        Question(
            qid="Q06",
            kind="correctness",
            question="Was Alex right about Sara's diet?",
            expected_contains=["yes", "right", "correct"],
        ),
        Question(
            qid="Q07",
            kind="belief",
            question="What did Pat believe about Tom's role at Acme?",
            expected_contains=["CTO"],
        ),
        Question(
            qid="Q08",
            kind="correctness",
            question="Is Pat right about Tom being CTO?",
            expected_contains=["no", "wrong", "design"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
