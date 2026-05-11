"""Check v9 segmenter on short single-event replies.

User concern: locomo ingest feeds the segmenter ONE Event at a time.
A complex prior message ("What's your favorite food?") is in a separate
event from its reply ("Pizza."). When the segmenter sees just "Pizza."
with no surrounding context, does v9 drop it as filler or keep it as
substance?

This probe sends representative short replies through v9 at 3 reps each
and prints what comes back.
"""

from __future__ import annotations

import asyncio
import os

import openai
from dotenv import load_dotenv
from probe_segmenter_F_natural import call
from probe_segmenter_F_natural_v9 import PROMPT_F_NATURAL_V9
from probe_segmenter_F_natural_v19 import PROMPT_F_NATURAL_V19

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


# Each entry is a single-event passage — what the segmenter actually
# sees in locomo. The expected column reflects intuition.
CASES: list[tuple[str, str, str]] = [
    # short affirmatives / commitments — all KEEP for reconstruction
    ("bare-yes", "Yes.", "KEEP — binary affirmative"),
    ("bare-no", "No.", "KEEP — binary negative"),
    ("yes-with-fact", "Yes, the meeting is at 3pm.", "KEEP — affirmative + time"),
    ("no-with-rationale", "No, I'm allergic to peanuts.", "KEEP — negative + reason"),
    ("yes-please", "Yes please!", "KEEP — affirmative"),
    ("ok-sounds-good", "OK, sounds good.", "KEEP — agreement / commitment"),
    ("got-it", "Got it.", "KEEP — commitment / understanding"),
    ("sure-thing", "Sure thing.", "KEEP — affirmative commitment"),
    ("yes-1pm", "Yes — 1pm works.", "KEEP — affirmative + time"),
    ("agree-strong", "I agree completely.", "KEEP — agreement"),
    ("yeah-cool", "Yeah, cool.", "KEEP — informal affirmative + approval"),
    # short content answers — KEEP
    ("single-noun-answer", "Pizza.", "KEEP — content answer"),
    ("single-noun-place", "Tokyo.", "KEEP — named entity"),
    ("single-number", "42.", "KEEP — content answer (bare number)"),
    ("preference-short", "Vanilla, definitely.", "KEEP — preference"),
    ("personal-fact-short", "I'm vegan.", "KEEP — personal fact"),
    # pure social plumbing — DROP
    ("greeting", "Hi!", "DROP — pure greeting"),
    ("thanks", "Thanks!", "DROP — pure thanks"),
    ("signoff", "Best, Sarah.", "DROP — pure sign-off"),
]


async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for prompt_name, prompt in [
        ("v9", PROMPT_F_NATURAL_V9),
        ("v19", PROMPT_F_NATURAL_V19),
    ]:
        for model_name, reasoning in [
            ("gpt-5-nano", "low"),
            ("gpt-5.4-nano", "medium"),
        ]:
            print(f"\n# {prompt_name} on {model_name} @ {reasoning} reasoning")
            print(
                f"{'case':25s} | {'passage':32s} | {'segs (rep1 / rep2 / rep3)':50s} | expected"
            )
            print("-" * 150)
            for case_id, passage, expected in CASES:
                results: list[list[str]] = []
                for _ in range(3):
                    p = prompt.format(passage=passage)
                    segs = await call(client, model_name, p, reasoning)
                    results.append(segs)
                cells = " / ".join(str(r) if r else "[]" for r in results)
                print(
                    f"{case_id:25s} | {passage[:32]:32s} | {cells[:50]:50s} | {expected}"
                )

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
