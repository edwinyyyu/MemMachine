"""perspective_separation — multiple people have views on same topic.

User reports what Alice thinks, what Bob thinks, what Carla thinks
about the same subject. Memory must attribute correctly when asked
about each person's view.

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

    # Topic: Q3 reorg
    add("Alice thinks the Q3 reorg is overdue and a good idea.", "view", ["Alice"])
    filler()
    add("Bob is skeptical of the Q3 reorg — worries about disruption.", "view", ["Bob"])
    filler()
    add(
        "Carla doesn't have strong feelings either way on the reorg.", "view", ["Carla"]
    )
    filler()
    filler()

    # Topic: new office space
    add(
        "Alice loves the new office space — natural light, big windows.",
        "view",
        ["Alice"],
    )
    filler()
    add("Bob hates the open layout — too noisy.", "view", ["Bob"])
    filler()
    add(
        "Carla thinks it's fine, just wishes for more meeting rooms.", "view", ["Carla"]
    )
    filler()
    filler()

    # Topic: hiring philosophy
    add(
        "Alice prefers hiring senior engineers who can hit the ground running.",
        "view",
        ["Alice"],
    )
    filler()
    add("Bob prefers hiring junior engineers and training them up.", "view", ["Bob"])
    filler()
    add("Carla wants a mix.", "view", ["Carla"])
    filler()
    filler()

    # Some shared ground for contrast
    add(
        "All three of them agree the deploy pipeline needs investment.",
        "shared_view",
        ["Alice", "Bob", "Carla"],
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
            kind="attribute",
            question="What does Alice think about the Q3 reorg?",
            expected_contains=["good", "overdue"],
        ),
        Question(
            qid="Q02",
            kind="attribute",
            question="What does Bob think about the Q3 reorg?",
            expected_contains=["skeptical", "disruption"],
        ),
        Question(
            qid="Q03",
            kind="attribute",
            question="What does Alice think of the new office?",
            expected_contains=["love", "natural light"],
        ),
        Question(
            qid="Q04",
            kind="attribute",
            question="What does Bob think of the new office?",
            expected_contains=["hate", "open layout", "noisy"],
        ),
        Question(
            qid="Q05",
            kind="attribute",
            question="What's Alice's hiring philosophy?",
            expected_contains=["senior", "ground running"],
        ),
        Question(
            qid="Q06",
            kind="attribute",
            question="What's Bob's hiring philosophy?",
            expected_contains=["junior", "training"],
        ),
        Question(
            qid="Q07",
            kind="agreement",
            question="What do Alice, Bob, and Carla all agree on?",
            expected_contains=["deploy pipeline", "investment"],
        ),
        Question(
            qid="Q08",
            kind="contrast",
            question="On hiring, where do Alice and Bob disagree?",
            expected_contains=["senior", "junior"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
