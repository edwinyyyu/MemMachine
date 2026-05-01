"""project_context — connecting many small actions into a larger coherent project.

User makes many task-level statements over time related to a larger project.
Question asks about the OVERALL project, requiring retrieval to cluster
related events.

This is a clustering / summarization task. Each task statement is a small
fact; the project as a whole is implicit.

Project arc: building a new API from scratch
  - Schema design
  - Auth endpoints
  - Rate limiting
  - Caching layer
  - Webhook ingestion
  - Deployment pipeline
  - Monitoring
  - Production launch

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
    "Coffee good.",
    "Slack laggy.",
    "Tired today.",
    "Long calls.",
    "Need lunch.",
    "Cafe trip.",
    "Heard a siren.",
    "Got a haircut.",
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

    # Many small task-level statements; never explicitly framed as "project."
    add(
        "Sketched the API schema today — REST with cursor pagination.", "task", ["User"]
    )
    filler()
    add("Wrote the auth endpoints; OAuth + bearer token.", "task", ["User"])
    filler()
    add("Added rate limiting via Redis.", "task", ["User"])
    filler()
    filler()
    add("Set up Prometheus metrics for the API.", "task", ["User"])
    filler()
    add("Webhook ingestion endpoint done.", "task", ["User"])
    filler()
    add("CI/CD pipeline configured for the API.", "task", ["User"])
    filler()
    filler()
    add("Added a caching layer with TTLs.", "task", ["User"])
    filler()
    add("Wrote the API docs in OpenAPI format.", "task", ["User"])
    filler()
    add("Deployed the API to staging.", "task", ["User"])
    filler()
    filler()
    add("Stress-tested the API at 5K rps — held up well.", "task", ["User"])
    filler()
    add("API is live on prod as of Friday.", "milestone", ["User"])
    filler()
    filler()

    # Reflection prompts (User reflects on the work)
    add(
        "Honestly, took the API from zero to prod in three weeks.",
        "reflection",
        ["User"],
    )
    filler()
    add(
        "The hardest part was the rate limiting — Redis tuning was tricky.",
        "reflection",
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
            kind="project_summary",
            question="What's the major project User has been working on recently?",
            expected_contains=["API"],
        ),
        Question(
            qid="Q02",
            kind="project_components",
            question="What components does User's API have? Name at least 3.",
            expected_contains=["auth", "rate limit", "cach"],
        ),
        Question(
            qid="Q03",
            kind="project_state",
            question="Is the API live in production?",
            expected_contains=["yes", "prod"],
        ),
        Question(
            qid="Q04",
            kind="project_duration",
            question="How long did the API project take?",
            expected_contains=["three", "3 weeks"],
        ),
        Question(
            qid="Q05",
            kind="project_difficulty",
            question="What was the hardest part of the API project?",
            expected_contains=["rate limit", "Redis"],
        ),
        Question(
            qid="Q06",
            kind="project_components",
            question="Did User write API documentation?",
            expected_contains=["yes", "OpenAPI", "doc"],
        ),
        Question(
            qid="Q07",
            kind="project_components",
            question="Did User set up monitoring for the API?",
            expected_contains=["yes", "Prometheus", "metrics"],
        ),
        Question(
            qid="Q08",
            kind="project_pacing",
            question="What was the API's load test result?",
            expected_contains=["5K", "rps", "well"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
