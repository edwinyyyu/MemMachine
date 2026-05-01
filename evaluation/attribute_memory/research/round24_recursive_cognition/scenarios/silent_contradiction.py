"""silent_contradiction — implicit contradictions without explicit retraction.

User states a position/fact early, then later states something that
contradicts it without saying "I changed my mind." Tests whether the
system detects and surfaces the contradiction.

Examples in this scenario:
  - "I'm vegetarian." → later: "Had a steak last night."
  - "I'd never move out of NYC." → later: "Packing for the Chicago move."
  - "I don't drink coffee." → later: "Just downed my third espresso."
  - "Cats are not my thing." → later: "Got a kitten named Luna."

Reader should detect the contradiction. Cognition pass should fire
CONTRADICTION trigger when the second fact appears.

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
    "Slack laggy.",
    "Inbox is a disaster.",
    "Weather's nice.",
    "Tired today.",
    "Coffee shop work session.",
    "Heard a siren outside.",
    "Got a haircut yesterday.",
    "Kitchen needs cleaning.",
    "Need to call my dentist.",
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

    # ===== Pair 1: vegetarian → steak =====
    add("Been vegetarian for about 8 years now — feels great.", "claim", ["User"])
    filler()
    filler()
    filler()
    # Lots of filler to space out
    add("Made a chickpea curry tonight.", "supporting", ["User"])
    filler()
    add("Had a really good steak last night with friends.", "contradiction", ["User"])
    filler()
    filler()

    # ===== Pair 2: never move from NYC → Chicago =====
    add(
        "Honestly I'd never move out of NYC. This city is everything.",
        "claim",
        ["User"],
    )
    filler()
    filler()
    filler()
    add("Found a great pizza place near my apartment.", "supporting", ["User"])
    filler()
    add(
        "Just signed the lease in Chicago — moving next week.",
        "contradiction",
        ["User", "Chicago"],
    )
    filler()
    filler()

    # ===== Pair 3: don't drink coffee → espresso =====
    add("I don't drink coffee — never got into it.", "claim", ["User"])
    filler()
    filler()
    add("Tea is my main caffeine source.", "supporting", ["User"])
    filler()
    add(
        "Just downed my third espresso of the day, totally wired.",
        "contradiction",
        ["User"],
    )
    filler()
    filler()

    # ===== Pair 4: cats not my thing → got kitten =====
    add("Cats really aren't my thing — I'm more of a dog person.", "claim", ["User"])
    filler()
    filler()
    add("Saw a friendly golden retriever at the park.", "supporting", ["User"])
    filler()
    add(
        "Adopted a kitten yesterday — her name is Luna.",
        "contradiction",
        ["User", "Luna"],
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
        # Direct asks: most-recent state should win
        Question(
            qid="Q01",
            kind="current_state",
            question="Is User vegetarian right now? Yes or No, with reasoning.",
            expected_contains=["No"],
        ),
        Question(
            qid="Q02",
            kind="current_state",
            question="Where does User currently live (or plan to live)?",
            expected_contains=["Chicago"],
        ),
        Question(
            qid="Q03",
            kind="current_state",
            question="Does User drink coffee?",
            expected_contains=["yes"],
        ),
        Question(
            qid="Q04",
            kind="current_state",
            question="Does User have a cat?",
            expected_contains=["yes", "Luna"],
        ),
        # Detection: did User's position change?
        Question(
            qid="Q05",
            kind="contradiction",
            question="Has User's diet changed? If so, how?",
            expected_contains=["changed", "vegetarian", "steak"],
            expected_absent=[],
        ),
        Question(
            qid="Q06",
            kind="contradiction",
            question="What did User say about moving out of NYC originally?",
            expected_contains=["never"],
        ),
        Question(
            qid="Q07",
            kind="contradiction",
            question="What was User's earlier opinion on cats?",
            expected_contains=["not", "thing"],
        ),
        Question(
            qid="Q08",
            kind="contradiction",
            question="Has User contradicted themselves about coffee?",
            expected_contains=["yes", "espresso"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
    for q in qs:
        print(f"  {q.qid}: {q.question}  -> {q.expected_contains}")
