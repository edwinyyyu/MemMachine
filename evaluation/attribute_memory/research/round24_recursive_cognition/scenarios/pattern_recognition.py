"""pattern_recognition — analogical patterns across distinct events.

User encounters the SAME structural problem in different surface forms,
spaced apart in conversation. The system should recognize the pattern
when asked.

Patterns tested:
  - Race condition: deploy queue (event 1), Postgres double-write (event 2),
    Slack double-message (event 3)
  - Off-by-one: pagination cursor (event 1), array index in dashboard (event 2)
  - Cache invalidation: stale React state (event 1), Redis miss after deploy
    (event 2), CDN headers stale (event 3)
  - Concurrency lock contention: user-update lock (event 1), Stripe webhook
    lock (event 2)

Stress tests cognition's ability to extract patterns from varied surface text.

Total ~60 turns.
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
    "Coffee was good.",
    "Long day.",
    "Tired.",
    "Slack laggy.",
    "Email pile.",
    "Need lunch.",
    "Cafe trip.",
    "Heard a siren.",
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

    # ===== Race condition pattern (3 instances) =====
    add(
        "Two deploys hit the queue simultaneously and both succeeded — "
        "ended up with two services running. Need to add a lock.",
        "race1",
        ["User"],
    )
    filler()
    filler()
    add(
        "Found a bug today: two API requests both updated the same Postgres row, "
        "the second one's update got silently dropped. Classic.",
        "race2",
        ["User"],
    )
    filler()
    filler()
    add(
        "Slack alerts: webhook fired twice in 200ms and both posted, "
        "now there are duplicate notifications.",
        "race3",
        ["User"],
    )
    filler()
    filler()

    # ===== Off-by-one pattern (2 instances) =====
    add(
        "Pagination is broken — cursor advances past the last record by one. "
        "Skipping the final page entirely.",
        "obo1",
        ["User"],
    )
    filler()
    filler()
    add(
        "Dashboard chart shows day 31 of the month even when month has 30 days. "
        "Loop bound is wrong.",
        "obo2",
        ["User"],
    )
    filler()
    filler()

    # ===== Cache invalidation pattern (3 instances) =====
    add(
        "React component caching the old user object after profile update. "
        "Need to bust the state when the prop changes.",
        "cache1",
        ["User"],
    )
    filler()
    filler()
    add(
        "Redis is serving stale config after the deploy — missed the eviction "
        "step in the rollout.",
        "cache2",
        ["User"],
    )
    filler()
    filler()
    add(
        "CDN serving old assets — turns out the cache-control headers were "
        "longer than the deploy interval.",
        "cache3",
        ["User"],
    )
    filler()
    filler()

    # ===== Concurrency lock contention (2 instances) =====
    add(
        "User-update endpoint slow because of lock contention on the "
        "users table. Switching to row-level locks.",
        "lock1",
        ["User"],
    )
    filler()
    filler()
    add(
        "Stripe webhook backfill ran into the same kind of issue — "
        "single global lock serializing 10K events.",
        "lock2",
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
            kind="pattern",
            question="What recurring bug pattern has User hit in the deploy queue, Postgres rows, and Slack webhook? Name the pattern.",
            expected_contains=["race", "concurren"],
        ),
        Question(
            qid="Q02",
            kind="pattern",
            question="How many distinct race-condition incidents has User encountered?",
            expected_contains=["3", "three"],
        ),
        Question(
            qid="Q03",
            kind="pattern",
            question="What kind of bug was the pagination cursor issue? It's an example of a classic off-by-N error.",
            expected_contains=["off-by-one", "off by one"],
        ),
        Question(
            qid="Q04",
            kind="pattern",
            question="What's the common pattern between the React component, Redis config, and CDN asset bugs?",
            expected_contains=["cache", "stale", "invalidat"],
        ),
        Question(
            qid="Q05",
            kind="pattern",
            question="Has User encountered lock contention before? Where?",
            expected_contains=["yes", "users table", "Stripe"],
        ),
        Question(
            qid="Q06",
            kind="analogy",
            question="If a webhook is firing twice, which prior bug pattern does it match?",
            expected_contains=["race", "deploy", "Postgres"],
        ),
        Question(
            qid="Q07",
            kind="pattern_count",
            question="How many cache-invalidation incidents has User had?",
            expected_contains=["3", "three"],
        ),
        Question(
            qid="Q08",
            kind="pattern_specific",
            question="What was wrong with the Postgres row update?",
            expected_contains=["race", "two", "dropped", "concurrent"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
