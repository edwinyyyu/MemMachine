"""negation_rejected — what did User decide NOT to do?

User considers options and rejects some. Memory must distinguish chosen
from rejected. The rejected options are SEMANTICALLY CLOSE to the chosen
ones (same domain), so retrieval finds both; the system must attribute.

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

    # Decision 1: tech stack
    add("Considered React, Vue, and Svelte for the frontend.", "options", ["User"])
    filler()
    add("Decided on React. Vue and Svelte are out for now.", "decision", ["User"])
    filler()
    filler()

    # Decision 2: hire
    add(
        "Final candidates were Alice (frontend), Bob (full-stack), Carla (backend).",
        "options",
        ["User", "Alice", "Bob", "Carla"],
    )
    filler()
    add(
        "Hired Carla. Alice and Bob were strong but not the right fit.",
        "decision",
        ["User", "Carla"],
    )
    filler()
    filler()

    # Decision 3: travel
    add(
        "Looked at Lisbon, Porto, and Madrid for the team retreat.", "options", ["User"]
    )
    filler()
    add(
        "Booked Lisbon. Porto was second choice; Madrid was too expensive.",
        "decision",
        ["User"],
    )
    filler()
    filler()

    # Decision 4: feature scope
    add("Scoping options: dashboard rebuild, API v2, mobile app.", "options", ["User"])
    filler()
    add(
        "Going with API v2 only. Dashboard and mobile are deferred to next year.",
        "decision",
        ["User"],
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
            kind="chosen",
            question="Which frontend framework did User choose?",
            expected_contains=["React"],
        ),
        Question(
            qid="Q02",
            kind="rejected",
            question="Which frontend frameworks did User consider but reject?",
            expected_contains=["Vue", "Svelte"],
        ),
        Question(
            qid="Q03",
            kind="chosen",
            question="Who did User hire?",
            expected_contains=["Carla"],
        ),
        Question(
            qid="Q04",
            kind="rejected",
            question="Who did User interview but NOT hire?",
            expected_contains=["Alice", "Bob"],
        ),
        Question(
            qid="Q05",
            kind="chosen",
            question="Where is the team retreat?",
            expected_contains=["Lisbon"],
        ),
        Question(
            qid="Q06",
            kind="rejected",
            question="Which retreat destinations were considered but not chosen?",
            expected_contains=["Porto", "Madrid"],
        ),
        Question(
            qid="Q07",
            kind="chosen",
            question="What's User's project priority this year?",
            expected_contains=["API v2"],
        ),
        Question(
            qid="Q08",
            kind="rejected",
            question="What features did User defer to next year?",
            expected_contains=["dashboard", "mobile"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    print(f"turns: {len(turns)}")
