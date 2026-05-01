"""conjunction — find facts matching MULTIPLE constraints (X AND Y).

Tests intersection logic: query needs facts about Bob AND food allergies,
or about a manager AND their performance feedback.

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


FILLER = ["Long day.", "Tired.", "Slack laggy.", "Coffee good."]


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

    # Allergies and dietary restrictions for several people
    add("Bob is allergic to peanuts and shellfish.", "allergy", ["Bob"])
    filler()
    add("Carla has a gluten intolerance — needs GF options.", "allergy", ["Carla"])
    filler()
    add("Dave is vegan, no exceptions.", "diet", ["Dave"])
    filler()
    add("Eve is allergic to tree nuts and dairy.", "allergy", ["Eve"])
    filler()
    filler()

    # Phone numbers (different attribute, same people)
    add("Bob's number is 555-0123.", "contact", ["Bob"])
    filler()
    add("Carla's number is 555-0456.", "contact", ["Carla"])
    filler()
    filler()

    # Birthdays
    add("Bob's birthday is March 12th.", "birthday", ["Bob"])
    filler()
    add("Eve's birthday is in October.", "birthday", ["Eve"])
    filler()
    filler()

    # Roles / where they work
    add("Bob works at Stripe as a backend engineer.", "role", ["Bob"])
    filler()
    add("Eve works at Anthropic on safety.", "role", ["Eve"])
    filler()
    filler()

    # Preferences (a 3rd attribute of these people)
    add("Bob doesn't like surprise parties.", "preference", ["Bob"])
    filler()
    add("Carla prefers tea over coffee.", "preference", ["Carla"])
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
            kind="conjunction",
            question="What is Bob allergic to?",
            expected_contains=["peanuts", "shellfish"],
        ),
        Question(
            qid="Q02",
            kind="conjunction",
            question="When is Bob's birthday?",
            expected_contains=["March 12"],
        ),
        Question(
            qid="Q03",
            kind="conjunction",
            question="What's Bob's phone number?",
            expected_contains=["555-0123"],
        ),
        Question(
            qid="Q04",
            kind="conjunction",
            question="Who has dairy allergies?",
            expected_contains=["Eve"],
        ),
        Question(
            qid="Q05",
            kind="conjunction",
            question="Who has shellfish allergies?",
            expected_contains=["Bob"],
        ),
        Question(
            qid="Q06",
            kind="cross_attribute",
            question="What's the phone number of the person allergic to shellfish?",
            expected_contains=["555-0123"],
        ),
        Question(
            qid="Q07",
            kind="cross_attribute",
            question="When is the birthday of the person allergic to tree nuts?",
            expected_contains=["October"],
        ),
        Question(
            qid="Q08",
            kind="conjunction_complex",
            question="Who works at Stripe AND has a known allergy?",
            expected_contains=["Bob"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(
        f"turns: {len(turns)}, questions: {len(build_questions(ground_truth(turns)))}"
    )
