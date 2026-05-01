"""Multi-batch coref scenario.

Anonymous descriptors at one turn ("my new boss"), then later turns NAME the
entity ("his name is Marcus"). Pairs are spaced so they would straddle a
batch=5 boundary but fit inside a sliding window of 10-20 turns.

Tests whether the sliding writer can resolve descriptor->name across what
used to be a writer-batch boundary.

Pairs (8 total):
  - "my new boss" -> "his name is Marcus" (gap 17)
  - "the team I'm joining" -> "...it's the platform team" (gap 11)
  - "this guy at the gym" -> "his name is Theo" (gap 14)
  - "a colleague" -> "Priya" (gap 9)
  - "an old friend" -> "Sana" (gap 8)
  - "our new neighbor" -> "her name is Alice" (gap 13)
  - "my new mentor" -> "Quentin" (gap 16)
  - "the senior" -> "Nadia" (gap 12)

Total ~50 turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    idx: int
    text: str
    kind: str  # "descriptor" | "name" | "filler"
    mentions: list[str] = field(default_factory=list)
    # If a descriptor turn, what descriptor key it sets.
    descriptor_key: tuple[str, str] | None = None  # (@User, predicate)
    descriptor_label: str | None = None  # the anonymous phrase
    # If a name turn, the (predicate, name) it resolves and the descriptor
    # turn it should bind to.
    resolves_to: str | None = None
    resolves_predicate: tuple[str, str] | None = None
    binds_descriptor_turn: int | None = None


# Carefully scripted: descriptor at turn D, name at turn D+gap. Filler in between.
PAIRS = [
    # (descriptor_text, predicate, descriptor_label, name_text, name, gap)
    (
        "My new boss started this week.",
        ("@User", "boss"),
        "my new boss",
        "Oh — his name is Marcus, by the way.",
        "Marcus",
        17,
    ),
    (
        "Joining a new team next month.",
        ("@User", "team"),
        "new team",
        "It's officially the platform team.",
        "platform",
        11,
    ),
    (
        "There's this guy at the gym I always see.",
        ("@User", "gym_buddy"),
        "guy at the gym",
        "Found out his name — he's Theo.",
        "Theo",
        14,
    ),
    (
        "A colleague helped me debug the deploy issue.",
        ("@User", "colleague"),
        "a colleague",
        "It was Priya who fixed it actually.",
        "Priya",
        9,
    ),
    (
        "Ran into an old friend at the airport.",
        ("@User", "old_friend"),
        "an old friend",
        "It was Sana, hadn't seen her in years.",
        "Sana",
        8,
    ),
    (
        "Our new neighbor moved in upstairs.",
        ("@User", "neighbor"),
        "new neighbor",
        "Her name is Alice — she's super friendly.",
        "Alice",
        13,
    ),
    (
        "Got assigned a new mentor at work.",
        ("@User", "mentor"),
        "new mentor",
        "He's Quentin, the staff engineer on infra.",
        "Quentin",
        16,
    ),
    (
        "The senior on my project is sharp.",
        ("@User", "senior"),
        "the senior",
        "Her name's Nadia and she's been here forever.",
        "Nadia",
        12,
    ),
]

FILLER_POOL = [
    "Coffee was good this morning.",
    "Weather is nice today.",
    "Long day, slow afternoon.",
    "Working from a cafe.",
    "Just got back from a run.",
    "Tired. Need a nap.",
    "Going to grab lunch soon.",
    "Been on calls all morning.",
    "Stomach hurts, too much espresso.",
    "Slack is laggy.",
    "Email avalanche this morning.",
    "Watching a movie tonight.",
    "Should hydrate more.",
    "Inbox is at 412 unread.",
    "Pretty mellow morning.",
]


def generate() -> list[Turn]:
    turns: list[Turn] = []
    fcursor = 0

    def add(text: str, kind: str, **kw) -> Turn:
        t = Turn(idx=len(turns) + 1, text=text, kind=kind, **kw)
        turns.append(t)
        return t

    def filler():
        nonlocal fcursor
        text = FILLER_POOL[fcursor % len(FILLER_POOL)]
        fcursor += 1
        add(text, "filler")

    # Lead-in: 2 filler turns
    for _ in range(2):
        filler()

    for desc_text, pred, desc_label, name_text, name, gap in PAIRS:
        d = add(
            desc_text,
            "descriptor",
            mentions=["User"],
            descriptor_key=pred,
            descriptor_label=desc_label,
        )
        # gap-1 filler turns between descriptor and name
        for _ in range(gap - 1):
            filler()
        add(
            name_text,
            "name",
            mentions=["User", name],
            resolves_to=name,
            resolves_predicate=pred,
            binds_descriptor_turn=d.idx,
        )
        # short gap between pairs
        for _ in range(2):
            filler()

    for i, t in enumerate(turns, start=1):
        t.idx = i
    return turns


@dataclass
class GroundTruth:
    pairs: list[dict] = field(default_factory=list)


def ground_truth(turns: list[Turn]) -> GroundTruth:
    gt = GroundTruth()
    for t in turns:
        if t.kind == "name":
            gt.pairs.append(
                {
                    "name": t.resolves_to,
                    "predicate": t.resolves_predicate,
                    "name_turn": t.idx,
                    "descriptor_turn": t.binds_descriptor_turn,
                }
            )
    return gt


@dataclass
class Question:
    qid: str
    kind: str
    question: str
    expected_contains: list[str]
    expected_absent: list[str] = field(default_factory=list)


def build_questions(gt: GroundTruth) -> list[Question]:
    qs: list[Question] = []
    # For each pair, ask:
    #   "Who is User's <predicate>?" -> name
    pred_qs = [
        (("@User", "boss"), "Who is User's boss?"),
        (("@User", "team"), "What team is User joining?"),
        (("@User", "gym_buddy"), "Who's the guy User sees at the gym?"),
        (("@User", "colleague"), "Which colleague helped User debug a deploy?"),
        (("@User", "old_friend"), "Which old friend did User run into at the airport?"),
        (("@User", "neighbor"), "Who is User's new neighbor?"),
        (("@User", "mentor"), "Who is User's mentor?"),
        (("@User", "senior"), "Who is the senior on User's project?"),
    ]
    pair_by_pred = {tuple(p["predicate"]): p for p in gt.pairs}
    for i, (pred, q) in enumerate(pred_qs, start=1):
        p = pair_by_pred.get(pred)
        if p:
            qs.append(
                Question(
                    qid=f"Q{i:02d}",
                    kind="coref",
                    question=q,
                    expected_contains=[p["name"]],
                )
            )
    return qs


if __name__ == "__main__":
    turns = generate()
    gt = ground_truth(turns)
    qs = build_questions(gt)
    print(f"turns: {len(turns)}")
    for p in gt.pairs:
        gap = p["name_turn"] - p["descriptor_turn"]
        print(
            f"  {p['name']}: descriptor t{p['descriptor_turn']} -> name t{p['name_turn']} (gap={gap})"
        )
    print(f"# questions: {len(qs)}")
