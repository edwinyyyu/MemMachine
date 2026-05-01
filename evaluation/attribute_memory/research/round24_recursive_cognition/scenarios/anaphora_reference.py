"""anaphora_reference — "that thing we discussed", "the same as last time".

User refers back to earlier topics with vague pronouns or definite
references. Retrieval must resolve the referent before searching.

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


FILLER = ["Long day.", "Tired.", "Coffee good.", "Need lunch."]


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

    # Topic 1: roof leak — discussed once, referred back later
    add(
        "Got a quote for fixing the roof leak — $4,200, big number.",
        "topic_intro",
        ["User"],
    )
    filler()
    add(
        "Roof contractor is licensed and insured, well-reviewed.",
        "topic_detail",
        ["User"],
    )
    filler()
    filler()

    # Topic 2: career change
    add(
        "Seriously considering switching from engineering to product management.",
        "topic_intro",
        ["User"],
    )
    filler()
    add(
        "Product manager at Notion makes about the same as my SWE comp.",
        "topic_detail",
        ["User"],
    )
    filler()
    filler()

    # Topic 3: car decision
    add(
        "Test-drove a 2024 Civic — 32 mpg, decent infotainment.",
        "topic_intro",
        ["User"],
    )
    filler()
    add(
        "Civic is $28k out the door. Considering it seriously.",
        "topic_detail",
        ["User"],
    )
    filler()
    filler()

    # NOW: anaphora references
    add(
        "Talked to my partner about that thing we discussed — we're going for it.",
        "anaphora",
        ["User"],
    )
    filler()
    # We don't have ground truth on what "that thing" is — model needs to infer
    # from temporal proximity (most recent durable topic = car decision)
    add("OK, we're buying the car.", "anaphora_resolve", ["User"])
    filler()
    filler()

    add("Got the same quote as before for the leak.", "anaphora", ["User"])
    filler()
    add("Going with that contractor I mentioned earlier.", "anaphora", ["User"])
    filler()
    filler()

    add("Did the same thing as last week.", "vague_anaphora", ["User"])
    # Intentionally unresolvable — too vague
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
            kind="anaphora",
            question="What did User end up buying after talking to their partner?",
            expected_contains=["car", "Civic"],
        ),
        Question(
            qid="Q02",
            kind="anaphora",
            question="Who did User decide to go with for the leak?",
            expected_contains=["roof", "licensed", "insured"],
        ),
        Question(
            qid="Q03",
            kind="recall",
            question="How much was the roof leak quote?",
            expected_contains=["$4,200", "4200"],
        ),
        Question(
            qid="Q04",
            kind="recall",
            question="What career change is User considering?",
            expected_contains=["product management", "PM"],
        ),
        Question(
            qid="Q05",
            kind="recall",
            question="What car did User test-drive?",
            expected_contains=["Civic"],
        ),
        Question(
            qid="Q06",
            kind="recall",
            question="What's the Civic's mpg?",
            expected_contains=["32"],
        ),
        Question(
            qid="Q07",
            kind="anaphora_resolution",
            question="When User said 'we're going for it', what were they referring to?",
            expected_contains=["car", "Civic"],
        ),
        Question(
            qid="Q08",
            kind="vague",
            question="What did User do that was 'the same thing as last week'?",
            expected_contains=["unclear", "not", "specified"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
