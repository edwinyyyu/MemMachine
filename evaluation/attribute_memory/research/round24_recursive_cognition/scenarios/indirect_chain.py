"""indirect_chain — multi-hop entity resolution at retrieval time.

Tests whether the reader can compose chains of facts:
  Sara is User's friend.
  Sara's husband is Alex.
  Alex works at Stripe.
  → "Where does User's friend's husband work?" should answer "Stripe"

The writer captures each link as a separate fact. The reader must compose.
This stresses RETRIEVAL: queries with no direct surface match must still
pull the chain via entity-graph traversal (or via multiple retrieval probes).

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
    "Coffee was good this morning.",
    "Long day on calls.",
    "Stomach hurts.",
    "Weather's fine.",
    "Slack is laggy.",
    "Inbox is a mess.",
    "Email avalanche.",
    "Need lunch.",
    "Quick break.",
    "Tired today.",
    "Working from a cafe.",
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

    # Chain 1: User -> friend Sara -> husband Alex -> employer Stripe -> CEO Patrick
    add("Sara is one of my closest friends from college.", "chain", ["User", "Sara"])
    filler()
    filler()
    add(
        "Sara's husband Alex is great — they got married last year.",
        "chain",
        ["User", "Sara", "Alex"],
    )
    filler()
    add("Alex works at Stripe as a designer.", "chain", ["Alex"])
    filler()
    filler()
    add(
        "Patrick Collison runs Stripe — Alex met him at an offsite.",
        "chain",
        ["Patrick", "Alex"],
    )
    filler()
    filler()

    # Chain 2: User -> sister Jamie -> partner Robin -> dog Cooper
    add("My sister Jamie lives in Brooklyn.", "chain", ["User", "Jamie"])
    filler()
    filler()
    add("Jamie's partner Robin is a teacher.", "chain", ["Jamie", "Robin"])
    filler()
    add(
        "Robin and Jamie just adopted a dog named Cooper.",
        "chain",
        ["Robin", "Jamie", "Cooper"],
    )
    filler()
    filler()

    # Chain 3: User -> boss Marcus -> mentor Olivia -> son Theo
    add(
        "My boss Marcus has been at the company for 10 years.",
        "chain",
        ["User", "Marcus"],
    )
    filler()
    filler()
    add(
        "Marcus's mentor Olivia is a VP at a different company.",
        "chain",
        ["Marcus", "Olivia"],
    )
    filler()
    add("Olivia's son Theo is starting college this fall.", "chain", ["Olivia", "Theo"])
    filler()
    filler()

    # Cross-references that should NOT confuse
    add("Had coffee with my boss's mentor today.", "indirect_use", ["User"])
    filler()
    add(
        "My friend's husband sent me a Stripe gift card for my birthday.",
        "indirect_use",
        ["User"],
    )
    filler()
    add("Cooper (my sister's dog) is hilarious.", "indirect_use", ["User", "Cooper"])
    filler()
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
            kind="indirect",
            question="Where does User's friend's husband work?",
            expected_contains=["Stripe"],
        ),
        Question(
            qid="Q02",
            kind="indirect",
            question="Who is the CEO of the company that User's friend's husband works at?",
            expected_contains=["Patrick"],
        ),
        Question(
            qid="Q03",
            kind="indirect",
            question="What is User's sister's partner's profession?",
            expected_contains=["teacher"],
        ),
        Question(
            qid="Q04",
            kind="indirect",
            question="What is the name of User's sister's dog?",
            expected_contains=["Cooper"],
        ),
        Question(
            qid="Q05",
            kind="indirect",
            question="What is User's boss's mentor's name?",
            expected_contains=["Olivia"],
        ),
        Question(
            qid="Q06",
            kind="indirect",
            question="Who is User's boss's mentor's son?",
            expected_contains=["Theo"],
        ),
        Question(
            qid="Q07",
            kind="direct",
            question="What's User's friend's name?",
            expected_contains=["Sara"],
        ),
        Question(
            qid="Q08",
            kind="direct",
            question="Where does Alex work?",
            expected_contains=["Stripe"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
    for q in qs:
        print(f"  {q.qid}: {q.question}  -> {q.expected_contains}")
