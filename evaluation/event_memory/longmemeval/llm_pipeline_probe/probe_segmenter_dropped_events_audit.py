"""Audit short-prompt drop quality on the 52 LoCoMo events that v47s
dropped (when used in the routed segmenter on group 0).

Tests a candidate short prompt against every dropped event:
  - did the candidate also drop it? (true positive / true negative)
  - did the candidate keep it? (false drop recovered / false keep)

The 52 events are the v33-kept, routed-dropped set from group 0.
Manually graded into:
  KEEP (substantive content present somewhere in the message): 18 cases
  GREY  (referent-bearing question, opinion, or thin specifics): 6 cases
  DROP (pure filler -- greeting / sign-off / generic affirmation): 28 cases

A good short prompt should KEEP all the KEEP cases, ideally KEEP the
GREY cases (conservative), and DROP the DROP cases.
"""

from __future__ import annotations

import asyncio
import json
import os

import openai
from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "segments": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["segments"],
}


# Manual grading: KEEP / GREY / DROP for each of the 52 dropped events.
EVENTS: list[tuple[str, str, str]] = [
    ("DROP", "D10:1", "Hey Melanie! Just wanted to say hi!"),
    ("GREY", "D10:15", "Cool! What did it look like?"),
    (
        "DROP",
        "D10:2",
        "Hey Caroline! Good to talk to you again. What's up? Anything new since last time?",
    ),
    ("DROP", "D11:17", "Great chatting with you! Feel free to reach out any time."),
    (
        "KEEP",
        "D11:5",
        "Wow, that's awesome! How did it feel being part of that community? [Attached a photo of a band performing on stage with a sign that says all are welcome]",
    ),
    ("DROP", "D12:19", "Sounds great, Mel! We'll make some awesome memories!"),
    (
        "DROP",
        "D12:7",
        "That's amazing! You put so much effort and passion into it. Your creativity really shines. Seeing how art can be a source of self-expression and growth is truly inspiring. You're killing it!",
    ),
    (
        "DROP",
        "D12:9",
        "Glad you found something that makes you so happy! Surrounding ourselves with things that bring us joy is important. Life's too short to do anything else!",
    ),
    ("DROP", "D14:28", "Wow, did you make that? It looks so real!"),
    (
        "KEEP",
        "D14:9",
        "Thanks Mel, really appreciate your kind words. It means a lot to me that you can feel the sense of peace and serenity. Makes me feel connected.",
    ),
    (
        "DROP",
        "D15:1",
        "Hey Melanie, great to hear from you. What's been up since we talked?",
    ),
    ("KEEP", "D15:20", "That's awesome! What type of guitar? Been playing long?"),
    (
        "KEEP",
        "D15:22",
        "Music's amazing, isn't it? Any songs that have deep meaning for you?",
    ),
    (
        "KEEP",
        "D16:18",
        "The sign was just a precaution, I had a great time. But thank you for your concern, you're so thoughtful!",
    ),
    (
        "DROP",
        "D16:19",
        "Phew! Glad it all worked out and you had a good time at the park!",
    ),
    (
        "DROP",
        "D16:20",
        "Yeah, it was so much fun! Those joyful moments definitely show us life's beauty.",
    ),
    (
        "GREY",
        "D17:18",
        "Nope, never been to something like that. What was it about? What made it so special?",
    ),
    (
        "DROP",
        "D17:2",
        "Hey Caroline! Great to hear from you! Wow, what an amazing journey. Congrats!",
    ),
    (
        "KEEP",
        "D17:20",
        "Wow, sounds amazing! What was the event like? Those posters are great!",
    ),
    (
        "GREY",
        "D17:6",
        "Thanks, Caroline! Appreciate your help. Got any tips for getting started on it?",
    ),
    (
        "DROP",
        "D18:10",
        "Our loved ones give us strength to tackle any challenge - it's amazing!",
    ),
    ("KEEP", "D18:16", "Wow, great pic! Is that recent? Looks like you all had fun!"),
    (
        "GREY",
        "D18:20",
        "Wow, that's awesome! What do you love most about camping with your fam?",
    ),
    (
        "KEEP",
        "D18:8",
        "Kids are amazingly resilient in tough situations. They have an amazing ability to bounce back.",
    ),
    (
        "DROP",
        "D19:12",
        "Absolutely! I'm so glad we can always be there for each other.",
    ),
    ("DROP", "D19:14", "Glad you had support. Being yourself is great!"),
    ("DROP", "D1:1", "Hey Mel! Good to see you! How have you been?"),
    (
        "KEEP",
        "D1:16",
        "Thanks, Caroline! Painting's a fun way to express my feelings and get creative. It's a great way to relax after a long day.",
    ),
    (
        "DROP",
        "D1:2",
        "Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?",
    ),
    (
        "GREY",
        "D1:4",
        "Wow, that's cool, Caroline! What happened that was so awesome? Did you hear any inspiring stories?",
    ),
    (
        "KEEP",
        "D2:13",
        "That's great, Caroline! Loving the inclusivity and support. Anything you're excited for in the adoption process?",
    ),
    (
        "KEEP",
        "D3:19",
        "Looks like you had a great day! How was it? You all look so happy!",
    ),
    (
        "KEEP",
        "D3:20",
        "It so fun! We played games, ate good food, and just hung out together. Family moments make life awesome.",
    ),
    (
        "DROP",
        "D3:21",
        "Glad you had a great time. Cherish the moments - they're the best!",
    ),
    (
        "DROP",
        "D3:9",
        "Yeah Mel, let's spread love and understanding! Thanks for the support and encouragement. We can tackle life's challenges together! We got this!",
    ),
    (
        "KEEP",
        "D4:12",
        "What kind of counseling and mental health services do you want to persue?",
    ),
    ("DROP", "D4:7", "Glad you made some new family mems. How was it? Anything fun?"),
    (
        "DROP",
        "D5:14",
        "Sounds awesome, Caroline! Have a great time and learn a lot. Have fun!",
    ),
    ("DROP", "D5:15", "Cool, thanks Mel! Can't wait. I'll keep ya posted. Bye!"),
    (
        "DROP",
        "D5:9",
        "Nice job! You really put in the work and it definitely shows. Your creativity looks great!",
    ),
    ("DROP", "D6:1", "Hey Mel! Long time no talk. Lots has been going on since then!"),
    ("DROP", "D6:2", "Hey Caroline! Missed you. Anything new? Spill the beans!"),
    ("KEEP", "D6:8", "What kind of books you got in your library?"),
    (
        "GREY",
        "D7:12",
        "That sounds awesome! What did you take away from it to use in your life?",
    ),
    (
        "KEEP",
        "D7:17",
        "Ah, they're adorable! What are their names? Pets sure do bring so much joy to us!",
    ),
    ("DROP", "D8:1", "Hey Mel, what's up? Been a busy week since we talked."),
    (
        "GREY",
        "D8:15",
        "Wow, what a great day! Glad everyone could make it. What was your favorite part?",
    ),
    ("KEEP", "D8:18", "Wow, looks awesome! Did you join in?"),
    ("DROP", "D8:20", "Wow, what an experience! How did it make you feel?"),
    (
        "KEEP",
        "D8:34",
        "We enjoy hiking in the mountains and exploring forests. It's a cool way to connect with nature and each other.",
    ),
    (
        "KEEP",
        "D9:10",
        "Seeing my mentee's face light up when they saw the support was the best! Such a special moment.",
    ),
    ("KEEP", "D9:9", "Wow! What's the best part you remember from it?"),
]


async def call_short(client, model, prompt_template, passage, reasoning):
    prompt = prompt_template.format(passage=passage)
    kwargs = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "segments_response",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    }
    if model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    raw = (resp.output_text or "").strip()
    return json.loads(raw).get("segments", [])


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, choices=["v47s", "v48s"])
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--reasoning", default="low")
    args = parser.parse_args()

    if args.prompt == "v47s":
        from probe_segmenter_short_v47 import PROMPT_SHORT_V47 as PROMPT
    else:
        from probe_segmenter_short_v48 import PROMPT_SHORT_V48 as PROMPT

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(8)

    async def go(passage):
        async with sem:
            return await call_short(client, args.model, PROMPT, passage, args.reasoning)

    print(f"# audit: {args.prompt} on 52 dropped events from LoCoMo group_0")
    print(f"# model={args.model} reasoning={args.reasoning} reps={args.reps}")
    print()

    # Per-rep results
    per_event_kept: list[list[bool]] = []
    for rep in range(args.reps):
        segs_list = await asyncio.gather(*(go(text) for _, _, text in EVENTS))
        kept_this_rep = [len(s) > 0 for s in segs_list]
        per_event_kept.append(kept_this_rep)

    # Aggregate: how many reps kept each event
    print(f"{'grade':5s} {'dia':8s} {'kept_reps':10s}  text")
    counts_by_grade = {"KEEP": [0, 0], "GREY": [0, 0], "DROP": [0, 0]}  # [kept, total]
    for i, (grade, dia, text) in enumerate(EVENTS):
        kept_reps = sum(per_event_kept[r][i] for r in range(args.reps))
        # majority-kept counts as "kept" for grade aggregation
        is_kept = kept_reps > args.reps / 2
        counts_by_grade[grade][1] += 1
        if is_kept:
            counts_by_grade[grade][0] += 1
        marker = (
            "✓"
            if (grade in ("KEEP", "GREY") and is_kept)
            or (grade == "DROP" and not is_kept)
            else "✗"
        )
        print(
            f"{grade:5s} {dia:8s} {kept_reps}/{args.reps}        {marker} {text[:80]}"
        )

    print()
    print("# summary")
    for grade in ["KEEP", "GREY", "DROP"]:
        kept, total = counts_by_grade[grade]
        want_kept = grade in ("KEEP", "GREY")
        correct = kept if want_kept else (total - kept)
        print(f"  {grade}: kept {kept}/{total}, correct {correct}/{total}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
