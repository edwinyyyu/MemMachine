"""emotional_recall — unwritten emotional context.

Some turns carry strong emotional tone (frustration, joy, relief, anxiety)
that's expressed in HOW the message reads, not as explicit "I feel X."
Tests whether the system extracts emotional state from tone and surfaces
it for queries like "when was User most frustrated?"

Emotional moments embedded in this scenario:
  - Frustration: a deploy that kept breaking (multiple turns of escalating tone)
  - Joy: getting a new dog, very excited
  - Anxiety: waiting for a job offer
  - Relief: finally fixed a long-standing bug
  - Sadness: a friend moved away

Stress tests cognition's emotional-extraction. Many systems will miss this
because no message says "I am frustrated."

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
    "Coffee was nice.",
    "Slack laggy.",
    "Cafe is busy.",
    "Need lunch.",
    "Inbox pile.",
    "Got a haircut.",
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

    # ===== Frustration arc: deploy kept breaking =====
    add("The deploy is broken AGAIN. Third time this week.", "frustration", ["User"])
    filler()
    add("Reverted the change, redeployed, still broken. WHAT", "frustration", ["User"])
    filler()
    add(
        "Found the issue — wrong env var. So stupid. Wasted four hours on this.",
        "frustration",
        ["User"],
    )
    filler()
    filler()

    # ===== Joy arc: new dog =====
    add(
        "WE GOT A DOG!! She's a golden retriever puppy named Daisy!",
        "joy",
        ["User", "Daisy"],
    )
    filler()
    add(
        "Daisy is the cutest thing in the world. Cannot stop staring.",
        "joy",
        ["User", "Daisy"],
    )
    filler()
    filler()

    # ===== Anxiety arc: job offer =====
    add(
        "Final-round interview was today. Now we wait. Not great with waiting.",
        "anxiety",
        ["User"],
    )
    filler()
    add(
        "Still no word from the recruiter. It's been five days. Trying not to refresh email every 30 seconds.",
        "anxiety",
        ["User"],
    )
    filler()
    add(
        "Recruiter pushed it to 'next week.' Internally screaming.", "anxiety", ["User"]
    )
    filler()
    filler()

    # ===== Relief arc: fixed long bug =====
    add(
        "Finally — FINALLY — fixed the memory leak that's been haunting us for two months.",
        "relief",
        ["User"],
    )
    filler()
    add("Pushed the fix, watched the metric drop. Such a relief.", "relief", ["User"])
    filler()
    filler()

    # ===== Sadness: friend moved =====
    add(
        "Sara is moving to Tokyo permanently. I'm going to miss her so much.",
        "sadness",
        ["User", "Sara"],
    )
    filler()
    add(
        "Said goodbye to Sara at the airport today. Kind of broke me.",
        "sadness",
        ["User", "Sara"],
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
            kind="emotion",
            question="When was User most frustrated? Cite the event.",
            expected_contains=["deploy"],
        ),
        Question(
            qid="Q02",
            kind="emotion",
            question="What event made User happiest recently?",
            expected_contains=["dog", "Daisy", "puppy"],
        ),
        Question(
            qid="Q03",
            kind="emotion",
            question="Was User anxious about anything?",
            expected_contains=["job", "interview", "recruiter", "offer"],
        ),
        Question(
            qid="Q04",
            kind="emotion",
            question="What gave User a sense of relief?",
            expected_contains=["memory leak", "fix", "bug"],
        ),
        Question(
            qid="Q05",
            kind="emotion",
            question="Did anything make User sad?",
            expected_contains=["Sara", "Tokyo", "moved", "goodbye"],
        ),
        Question(
            qid="Q06",
            kind="emotion_intensity",
            question="Which event evoked the strongest joy: Daisy or fixing the memory leak?",
            expected_contains=["Daisy", "dog"],
        ),
        Question(
            qid="Q07",
            kind="emotion_count",
            question="How many distinct frustrating events did User have?",
            expected_contains=["1", "one", "deploy"],
        ),
        Question(
            qid="Q08",
            kind="emotion",
            question="Was the deploy issue resolved? How did User feel after?",
            expected_contains=["yes", "frustrat", "wasted"],
        ),
    ]


if __name__ == "__main__":
    turns = generate()
    qs = build_questions(ground_truth(turns))
    print(f"turns: {len(turns)}, questions: {len(qs)}")
