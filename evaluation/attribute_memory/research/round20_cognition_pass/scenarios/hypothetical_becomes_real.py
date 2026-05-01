"""Hypothetical-becomes-real scenario.

Tests the cognition loop:
  1. User states a conditional: "If I get hired at @Notion, my boss will be @Sam"
  2. Filler turns
  3. Trigger event: "I got the offer from @Notion"
  4. (cognition pass should surface the prior expectation — User expects @Sam)
  5. More filler
  6. Met-the-boss event: "Met my new boss, his name is @Sam"

Questions:
  - Did User expect Sam to be the boss before meeting? (cognition-reflective)
  - Who is User's boss? (factual; should retrieve "Sam")
  - What did User say would happen if hired at Notion? (history-recall)

Design:
  - Total 60 turns, 1 conditional pair.
  - Multiple expectation/cognition pairs to test the loop generally.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    idx: int
    text: str
    kind: str  # "filler" | "conditional" | "trigger" | "named" | "expectation"
    mentions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


FILLER = [
    "Coffee was good this morning.",
    "Long day ahead.",
    "Stomach hurts, too much espresso.",
    "Email avalanche.",
    "Slack is laggy.",
    "Tired. Need a nap.",
    "Going to grab lunch.",
    "Weather is nice today.",
    "Inbox at 412 unread.",
    "Slow afternoon.",
    "Just got back from a run.",
    "Pretty mellow morning.",
    "Watching a movie tonight.",
    "Coffee was good this morning.",
    "Should hydrate more.",
]


def generate() -> list[Turn]:
    """Return ~60 turns with 2 hypothetical-becomes-real pairs."""
    turns: list[Turn] = []
    fcursor = 0

    def add(text: str, kind: str, **kw) -> Turn:
        t = Turn(idx=len(turns) + 1, text=text, kind=kind, **kw)
        turns.append(t)
        return t

    def filler():
        nonlocal fcursor
        text = FILLER[fcursor % len(FILLER)]
        fcursor += 1
        add(text, "filler")

    # Lead-in
    for _ in range(2):
        filler()

    # Pair 1: conditional → trigger → named met
    add(
        "If I get hired at @Notion, my boss will be Sam.",
        "conditional",
        mentions=["User", "Notion", "Sam"],
        metadata={"hypothesis": "@Notion -> @Sam as boss"},
    )
    for _ in range(4):
        filler()
    add(
        "I'm pretty hopeful about the Notion application.",
        "expectation",
        mentions=["User", "Notion"],
    )
    for _ in range(5):
        filler()
    add(
        "I got the offer from Notion!",
        "trigger",
        mentions=["User", "Notion"],
        metadata={"trigger_for": "@Notion"},
    )
    for _ in range(3):
        filler()
    add(
        "First day at Notion next Monday.",
        "trigger",
        mentions=["User", "Notion"],
    )
    for _ in range(5):
        filler()
    add(
        "Met my new boss today, his name is Sam.",
        "named",
        mentions=["User", "Sam"],
        metadata={"resolves": "@Notion -> @Sam"},
    )

    # Pair 2: another conditional pair to test generalization
    for _ in range(3):
        filler()
    add(
        "Thinking of moving to Berlin — if I do, I'll buy a bike for sure.",
        "conditional",
        mentions=["User", "Berlin"],
        metadata={"hypothesis": "@Berlin -> bike"},
    )
    for _ in range(6):
        filler()
    add(
        "Decided. Moving to Berlin next month.",
        "trigger",
        mentions=["User", "Berlin"],
        metadata={"trigger_for": "@Berlin"},
    )
    for _ in range(5):
        filler()
    add(
        "Picked up a bike yesterday — a used Bianchi.",
        "named",
        mentions=["User"],
        metadata={"resolves": "@Berlin -> bike"},
    )

    for i, t in enumerate(turns, start=1):
        t.idx = i
    return turns


@dataclass
class GroundTruth:
    pairs: list[dict] = field(default_factory=list)


def ground_truth(turns: list[Turn]) -> GroundTruth:
    gt = GroundTruth()
    by_idx = {t.idx: t for t in turns}
    pair: dict = {}
    for t in turns:
        if t.kind == "conditional":
            if pair:
                gt.pairs.append(pair)
            pair = {
                "conditional_turn": t.idx,
                "conditional_text": t.text,
                "hypothesis": t.metadata.get("hypothesis"),
                "trigger_turn": None,
                "named_turn": None,
            }
        elif t.kind == "trigger" and pair and pair.get("trigger_turn") is None:
            pair["trigger_turn"] = t.idx
            pair["trigger_text"] = t.text
        elif t.kind == "named" and pair:
            pair["named_turn"] = t.idx
            pair["named_text"] = t.text
            gt.pairs.append(pair)
            pair = {}
    if pair:
        gt.pairs.append(pair)
    return gt


@dataclass
class Question:
    qid: str
    kind: str
    question: str
    expected_contains: list[str]
    expected_absent: list[str] = field(default_factory=list)


def build_questions(gt: GroundTruth) -> list[Question]:
    return [
        # Pair 1
        Question(
            qid="Q01",
            kind="cognition_reflective",
            question="Did User expect Sam to be his boss before meeting Sam?",
            expected_contains=["yes", "expected", "Notion"],
        ),
        Question(
            qid="Q02",
            kind="factual",
            question="Who is User's boss at Notion?",
            expected_contains=["Sam"],
        ),
        Question(
            qid="Q03",
            kind="history",
            question="What did User predict would happen if hired at Notion?",
            expected_contains=["Sam", "boss"],
        ),
        # Pair 2
        Question(
            qid="Q04",
            kind="cognition_reflective",
            question="What did User intend to do if moving to Berlin?",
            expected_contains=["bike"],
        ),
        Question(
            qid="Q05",
            kind="factual",
            question="Did User end up buying a bike?",
            expected_contains=["yes", "Bianchi"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    gt = ground_truth(turns)
    qs = build_questions(gt)
    print(f"turns: {len(turns)}")
    for p in gt.pairs:
        print(f"  {p}")
    print(f"# questions: {len(qs)}")
    for q in qs:
        print(f"  [{q.qid}] {q.kind}: {q.question}")
