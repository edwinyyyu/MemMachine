"""aggregation_counts — count/aggregate over event history.

Tests retrieval of ALL events of a kind so the LLM can count them.
Vector-only memory tends to surface only the top-K most-similar; it cannot
return "all 7 visits to Tokyo." A semantic-memory system with entity-keyed
retrieval should pull the entity's full event chain.

Counts to test:
  - 4 cats over the years (Luna, Mochi, Toby, Pepper)
  - 6 visits to Tokyo (different reasons)
  - 3 different jobs at startups
  - 5 marathons run

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
    "Long week.",
    "Tired today.",
    "Slack laggy.",
    "Inbox messy.",
    "Coffee was good.",
    "Weather is fine.",
    "Need lunch.",
    "Quick break.",
    "Working from cafe.",
    "Heard a fire alarm.",
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

    # ===== Cats over the years =====
    add("Adopted my first cat Luna in 2018.", "event", ["User", "Luna"])
    filler()
    add("Got Mochi (Siamese) in 2020.", "event", ["User", "Mochi"])
    filler()
    add("Adopted Toby last year — he's a tabby.", "event", ["User", "Toby"])
    filler()
    add(
        "Brought home Pepper this spring — fourth cat now.", "event", ["User", "Pepper"]
    )
    filler()
    filler()

    # ===== Tokyo visits =====
    add(
        "Went to Tokyo for the first time back in 2019 with Sarah.",
        "event",
        ["User", "Tokyo", "Sarah"],
    )
    filler()
    add("Tokyo trip again in 2020 — work conference.", "event", ["User", "Tokyo"])
    filler()
    add("Visited Tokyo in spring 2021 for ramen.", "event", ["User", "Tokyo"])
    filler()
    add("Tokyo for two weeks in 2022 — vacation.", "event", ["User", "Tokyo"])
    filler()
    add("Quick Tokyo layover in 2023 on the way to Seoul.", "event", ["User", "Tokyo"])
    filler()
    add(
        "Most recent Tokyo visit was last month — saw cherry blossoms.",
        "event",
        ["User", "Tokyo"],
    )
    filler()
    filler()

    # ===== Startup jobs =====
    add("First startup job was at Vercel in 2017.", "event", ["User", "Vercel"])
    filler()
    add("Joined Linear in 2019 — second startup.", "event", ["User", "Linear"])
    filler()
    add("Now at Resend, my third startup.", "event", ["User", "Resend"])
    filler()
    filler()

    # ===== Marathons =====
    add("Ran my first marathon in Boston 2017.", "event", ["User", "Boston"])
    filler()
    add("Did Chicago Marathon 2018.", "event", ["User", "Chicago"])
    filler()
    add("NYC Marathon 2019 — third marathon.", "event", ["User", "NYC"])
    filler()
    add("Berlin Marathon last fall.", "event", ["User", "Berlin"])
    filler()
    add(
        "Just signed up for my fifth — Tokyo Marathon next year.",
        "event",
        ["User", "Tokyo"],
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
            kind="count",
            question="How many cats has User had over the years? List them.",
            expected_contains=["4", "four", "Luna", "Mochi", "Toby", "Pepper"],
        ),
        Question(
            qid="Q02",
            kind="count",
            question="How many times has User visited Tokyo?",
            expected_contains=["6", "six"],
        ),
        Question(
            qid="Q03",
            kind="count",
            question="How many startups has User worked at?",
            expected_contains=["3", "three"],
        ),
        Question(
            qid="Q04",
            kind="count",
            question="List User's marathons in chronological order.",
            expected_contains=["Boston", "Chicago", "NYC", "Berlin", "Tokyo"],
        ),
        Question(
            qid="Q05",
            kind="count_recent",
            question="When was User's most recent Tokyo trip?",
            expected_contains=["last month", "cherry blossoms"],
        ),
        Question(
            qid="Q06",
            kind="count_first",
            question="When did User adopt their first cat? Which one?",
            expected_contains=["2018", "Luna"],
        ),
        Question(
            qid="Q07",
            kind="count",
            question="How many marathons has User run (or signed up for)?",
            expected_contains=["5", "five"],
        ),
        Question(
            qid="Q08",
            kind="count_subset",
            question="Which cat is a Siamese?",
            expected_contains=["Mochi"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
    for q in qs:
        print(f"  {q.qid}: {q.question}")
