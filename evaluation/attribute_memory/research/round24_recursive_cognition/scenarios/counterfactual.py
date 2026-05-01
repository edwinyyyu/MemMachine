"""counterfactual — regret / what-if reasoning over alternative paths.

User describes choices and reflects on alternatives. Tests whether the
system tracks both the actual outcome AND the counterfactual the user
considered.

Examples:
  - User considered Anthropic vs Google offer, took Google. Later regrets.
  - User chose Brooklyn over Queens for housing. Reflects on what Queens
    would have been like.
  - User stayed at Stripe instead of joining a friend's startup; later sees
    that startup IPO.

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
    "Coffee was good.",
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

    # ===== Choice 1: Anthropic vs Google offers (chose Google) =====
    add(
        "Got offers from both Anthropic and Google. Hard call.",
        "decision",
        ["User", "Anthropic", "Google"],
    )
    filler()
    filler()
    add(
        "Took the Google offer — more money, bigger team.",
        "outcome",
        ["User", "Google"],
    )
    filler()
    filler()
    add(
        "Now at Google, six months in. The Anthropic team I would've joined "
        "just shipped something I'd love to have built. Big regret.",
        "regret",
        ["User", "Anthropic", "Google"],
    )
    filler()
    filler()

    # ===== Choice 2: Brooklyn vs Queens (chose Brooklyn) =====
    add(
        "Picking between Brooklyn and Queens for the next apartment.",
        "decision",
        ["User", "Brooklyn", "Queens"],
    )
    filler()
    filler()
    add("Signed the Brooklyn lease — better commute.", "outcome", ["User", "Brooklyn"])
    filler()
    add("Brooklyn's been great so far — no regrets.", "no_regret", ["User", "Brooklyn"])
    filler()
    filler()

    # ===== Choice 3: Stayed at Stripe over friend's startup =====
    add(
        "Friend's startup is recruiting me — equity-heavy, risky.", "decision", ["User"]
    )
    filler()
    add(
        "Decided to stay at Stripe — too risky to bet on a Series A.",
        "outcome",
        ["User", "Stripe"],
    )
    filler()
    filler()
    add(
        "That friend's startup just hit unicorn status. Big public IPO.",
        "missed_outcome",
        ["User"],
    )
    filler()
    add(
        "Honestly thinking about whether I should've taken that bet.",
        "regret",
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
            kind="actual",
            question="Which company did User end up working at: Anthropic or Google?",
            expected_contains=["Google"],
        ),
        Question(
            qid="Q02",
            kind="counterfactual",
            question="Did User regret choosing Google over Anthropic?",
            expected_contains=["yes", "regret"],
        ),
        Question(
            qid="Q03",
            kind="counterfactual",
            question="Where did User live: Brooklyn or Queens?",
            expected_contains=["Brooklyn"],
        ),
        Question(
            qid="Q04",
            kind="counterfactual",
            question="Did User regret choosing Brooklyn?",
            expected_contains=["no"],
        ),
        Question(
            qid="Q05",
            kind="missed_opportunity",
            question="Did User join their friend's startup?",
            expected_contains=["no"],
        ),
        Question(
            qid="Q06",
            kind="missed_opportunity",
            question="What happened to the friend's startup that User declined to join?",
            expected_contains=["unicorn", "IPO"],
        ),
        Question(
            qid="Q07",
            kind="reflection",
            question="Has User had second thoughts about staying at Stripe?",
            expected_contains=["yes", "regret", "thinking"],
        ),
        Question(
            qid="Q08",
            kind="alternatives_listed",
            question="What alternative job did User pass on for Google?",
            expected_contains=["Anthropic"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
