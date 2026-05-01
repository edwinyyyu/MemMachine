"""system_references — referencing a large system from local context.

User describes parts of a larger system over time. Later, a small detail
("the rate limiter is acting up") should retrieve the larger context
("our auth middleware uses Redis-backed sliding-window rate limiting").

Tests whether the system can connect a local detail to the broader system
it belongs to, even when the original context is far back in history.

Scenarios:
  - The auth middleware system (introduced over multiple turns)
  - The data pipeline (Kafka → Spark → Redshift)
  - The frontend stack (Next.js + Tailwind + tRPC)

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
    "Need lunch.",
    "Cafe again.",
    "Heard a siren.",
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

    # ===== Auth middleware system =====
    add(
        "Built our new auth middleware in Go — runs as a sidecar in front of every service.",
        "system_intro",
        ["User"],
    )
    filler()
    add(
        "Auth middleware uses JWT tokens with RS256 signing.", "system_detail", ["User"]
    )
    filler()
    add(
        "Auth middleware has Redis-backed sliding-window rate limiting at 100 req/min/user.",
        "system_detail",
        ["User"],
    )
    filler()
    add(
        "Logging via OpenTelemetry to Honeycomb for the auth middleware.",
        "system_detail",
        ["User"],
    )
    filler()
    filler()

    # ===== Data pipeline =====
    add(
        "Our analytics pipeline: events flow Kafka → Spark Streaming → Redshift.",
        "system_intro",
        ["User"],
    )
    filler()
    add(
        "The pipeline uses Avro schema evolution for backward compatibility.",
        "system_detail",
        ["User"],
    )
    filler()
    filler()

    # ===== Frontend stack =====
    add(
        "Frontend is Next.js 14 with Tailwind and tRPC for type-safe APIs.",
        "system_intro",
        ["User"],
    )
    filler()
    add("State management via Zustand on the frontend.", "system_detail", ["User"])
    filler()
    filler()

    # ===== Localized issue mentions (need to connect back) =====
    add(
        "The rate limiter is acting up — getting 429s on legitimate traffic.",
        "issue",
        ["User"],
    )
    filler()
    add("Honeycomb is showing weird auth latency this morning.", "issue", ["User"])
    filler()
    add(
        "Avro deserialization errors in the pipeline since the deploy.",
        "issue",
        ["User"],
    )
    filler()
    add("Tailwind classes not purging in prod build.", "issue", ["User"])
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
            kind="system_lookup",
            question="What is the rate limiter that's having issues part of?",
            expected_contains=["auth middleware"],
        ),
        Question(
            qid="Q02",
            kind="system_detail",
            question="What signing algorithm does User's auth middleware use?",
            expected_contains=["RS256"],
        ),
        Question(
            qid="Q03",
            kind="system_lookup",
            question="Honeycomb is showing weird auth latency. What system is Honeycomb attached to?",
            expected_contains=["auth middleware", "OpenTelemetry"],
        ),
        Question(
            qid="Q04",
            kind="system_detail",
            question="What's User's data pipeline architecture?",
            expected_contains=["Kafka", "Spark", "Redshift"],
        ),
        Question(
            qid="Q05",
            kind="system_lookup",
            question="Avro deserialization errors are happening — what's the broader pipeline they're part of?",
            expected_contains=["Kafka", "Spark", "analytics pipeline"],
        ),
        Question(
            qid="Q06",
            kind="system_detail",
            question="What frontend framework does User use?",
            expected_contains=["Next.js"],
        ),
        Question(
            qid="Q07",
            kind="system_lookup",
            question="Tailwind purging issue — what stack is this part of?",
            expected_contains=["Next.js", "frontend"],
        ),
        Question(
            qid="Q08",
            kind="system_detail",
            question="What state management library is User using on the frontend?",
            expected_contains=["Zustand"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
