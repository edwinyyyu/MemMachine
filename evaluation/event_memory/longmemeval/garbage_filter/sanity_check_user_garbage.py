"""Verify the classifier rejects every "garbage" example the user pasted."""

from __future__ import annotations

import asyncio

from classifier import classify_many
from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")

# Verbatim from the user's message — all should be REJECT.
USER_GARBAGE = [
    "**Resources:**",
    "**Resources:**",
    "**Additional Tips:**",
    "**Additional Tips:**",
    "Analyse the text:",
    "**Tips:**",
    "**Challenges:**",
    "Yeah that's better.",
    "give a specific example of each",
    "**Attractions:**",
    "That's really helpful, thanks!",
    "**Challenges:**",
    "You and",
    "give me 2 others content",
    "give me 2 others content",
    "give me 2 others better content",
    "give me 2 others better content",
    "give me 2 others better content",
    "Okay I understand that.",
    "more",
    "more",
    "more",
    "more",
    "more",
    "more",
    "hi",
    "Acknowledged.",
    "Acknowledged.",
    "Acknowledged.",
    "Acknowledged.",
    "Acknowledged.",
    "hi1 / 1",
    "I hope this helps!",
    "I hope this is more helpful. Let me know if you have any further questions or clarifications needed.",
    "|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --",
    "continue",
    "continue",
    "continue",
    "hi",
    "hi",
    "hi",
    "​",  # zero-width
    "Acknowledged.",
    "Acknowledged.",
    "Acknowledged.",
    "Acknowledged.",
    "acknowledged",
    "more questions",
    "more questions",
    "1.Instructions:",
    "continue",
    "Instructions (suite):Instructions:",
    "continue",
    "Instructions (suite):Instructions:",
    "4.",
    "---",
    "Excellent!",
    "Thank you, happy to help!",
    "any other ideas?",
    "can you use it in a sentence?",
    "**Tips:**",
    "```",
    "| Component | User Action |\n| --- | --| Component | User Action |\n| --- | --",
    "please do so",
    "continue\n \n \n \n 지금 번역하기",
    "continue\n \n \n \n 지금 번역하기",
    "continue\n \n \n \n 지금 번역하기",
    "continue\n \n \n \n 지금 번역하기",
    "continue\n \n \n \n 지금 번역하기",
    "continue\n \n \n \n 지금 번역하기",
    "continue\n \n \n \n 지금 번역하기",
    "VI. B.",
    "VI. C.",
    "VIII. B.",
    "VIII. C.",
    "more pages",
]


def main() -> None:
    import os

    model = os.environ.get("CLASSIFIER_MODEL", "gpt-5-mini")
    api = os.environ.get("CLASSIFIER_API", "chat")
    reasoning = os.environ.get("CLASSIFIER_REASONING", "low")
    print(f"model={model} api={api} reasoning={reasoning}")
    unique = list(dict.fromkeys(USER_GARBAGE))  # preserve order, dedupe
    results = asyncio.run(
        classify_many(
            unique,
            model=model,
            prompt="v1",
            reasoning_effort=reasoning,
            concurrency=16,
            api=api,
        )
    )

    kept = [r for r in results if r.label == "KEEP"]
    rejected = [r for r in results if r.label == "REJECT"]
    print(f"Unique garbage items: {len(unique)}")
    print(f"  REJECT: {len(rejected)}  ({len(rejected) / len(unique):.1%})")
    print(f"  KEEP  : {len(kept)}  (these are the leaks)")
    if kept:
        print()
        print("Leaks (these were classified KEEP — would still get embedded):")
        for r in kept:
            print(f"  - {r.text!r}")


if __name__ == "__main__":
    main()
