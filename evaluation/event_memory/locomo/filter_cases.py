"""Tiny case-level harness to iterate the filter prompt cheaply.

Instead of a $8 / 68-min full benchmark per prompt edit, judge candidate
prompts against a hand-labeled set of the KNOWN failure cases (pulled from
the v3 run's gold_drop_docs, rendered with full dates as the filter sees
them) plus correct-DROP controls. A clean prompt must flip the Bucket-C
KEEPs while leaving the DROP controls dropped.

Cases are TEST INPUTS, not prompt examples -- the prompt's own examples stay
neutral/non-bench (see feedback_neutral_prompt_examples).

Usage:  uv run python filter_cases.py
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

MODEL, EFFORT = "gpt-5-nano", "low"

# (question substring, doc substring to disambiguate, expected, label)
# expected KEEP = filter wrongly dropped (Bucket C component/ordinal/interval)
# expected DROP = correct drop (Bucket A mismatch / resonance / filler)
CASE_SPECS = [
    ("months passed between Andrew adopting Toby and Buddy",
     "adopted another pup", "KEEP", "interval-endpoint (Buddy adoption)"),
    ("How many screenplays has Joanna written",
     "loss, identity", "KEEP", "count-instance (a screenplay)"),
    ("What game was the second tournament",
     "Street Fighter", "KEEP", "ordinal-instance (the game)"),
    ("biggest stressor in Andrew",
     "Cooking has been helping me de-stress", "DROP",
     "indirect/no-stressor-named (weak gold)"),
    ("What instrument is John learning",
     "learning this instrument", "DROP", "speaker-mismatch (James not John)"),
    ("How does Calvin plan to jumpstart his inspiration",
     "immerse myself", "DROP", "speaker-mismatch (Dave not Calvin)"),
    ("tattoo does Audrey have on her arm",
     "dog and sunflowers", "KEEP", "photo-caption (tattoo design in caption)"),
]

# fully synthetic controls (full-date rendered) -- must DROP
SYNTHETIC = [
    ("How many pets do I have?",
     '[Friday, May 20, 2022, 7:49 PM] Sam: "7"',
     "DROP", "resonance (bare number)"),
    ("When was the second script shown on the big screens?",
     '[Tuesday, October 25, 2022, 8:16 PM] Nate: "Hey Joanna, what\'s '
     'been up since we last chatted? How\'s it going?"',
     "DROP", "filler/greeting (no content)"),
    ("What instrument is John learning?",
     '[Sunday, March 27, 2022, 12:40 AM] James: "Hey John! I\'m '
     'challenging myself -- I\'m learning this instrument, quite a journey."',
     "DROP", "about-ness mismatch (James learning, not John)"),
]


def load_cases() -> list[tuple[str, str, str, str]]:
    d = json.load(open("swiss-probe-filter-p100-v3.json"))
    pool: list[tuple[str, str]] = []  # (question, doc)
    for r in d["records"]:
        for doc in r.get("gold_drop_docs", []):
            pool.append((r["question"], doc))
    cases = []
    for qsub, dsub, exp, label in CASE_SPECS:
        hit = next(
            (
                (q, doc) for q, doc in pool
                if qsub.lower() in q.lower() and dsub.lower() in doc.lower()
            ),
            None,
        )
        if hit:
            cases.append((hit[0], hit[1], exp, label))
        else:
            print(f"  [warn] no doc found for: {label}")
    for q, doc, exp, label in SYNTHETIC:
        cases.append((q, doc, exp, label))
    return cases


# ---- candidate prompts ------------------------------------------------

PROMPTS: dict[str, str] = {}

PROMPTS["v4_clean"] = """\
You are given a QUERY and one candidate ITEM (shown with its date and \
speaker, as it will appear when answering). Decide whether the ITEM supplies \
any information the QUERY's answer would use (KEEP) or none (DROP).

"A piece" counts, not only a direct answer: a single component the answer is \
built from (one event, date, name, or quantity to combine with others), or a \
detail identifying what the QUERY refers to. The ITEM need not look like the \
answer.

Handle every ambiguity by ONE rule: read the ITEM in the way MOST FAVORABLE \
to the QUERY -- settle which specific instance it is, and resolve any \
reference it contains ("it", "this", "there"), in the QUERY's favor. Resolve \
only what the ITEM leaves open; never add a connection the ITEM does not \
itself state. If even one PART of the ITEM supplies a piece the answer uses, \
KEEP it -- ignore surrounding greetings or small talk. DROP only if, even \
under the most-favorable reading, no part supplies any piece.

- "when did he win his third tournament" + "I won another tournament [date]": \
states a dated win he had; only WHICH-numbered is open -> settle favorably \
-> KEEP.
- "months between adopting A and B" + an item reporting one of those \
adoptions with its date: supplies one endpoint -> KEEP.
- "how many pets" + "7": nothing in "7" states it is about pets; relevance \
would require INVENTING that link -> forbidden -> DROP.

QUERY:
{query}

ITEM:
{doc}

Reply with exactly one token: KEEP or DROP."""


PROMPTS["v5"] = """\
You are given a QUERY and one candidate ITEM. The ITEM begins with its date \
and speaker in brackets -- like [Friday, May 20, 2022, 7:49 PM] Speaker: \
"..." -- and that date and speaker ARE part of the ITEM's information: a date \
in the bracket is a date the ITEM supplies. Decide whether the ITEM supplies \
any information the QUERY's answer would use (KEEP) or none (DROP).

You are NOT answering the QUERY and NOT computing its answer. You do not need \
the ITEM to state a total, a count, an ordinal ("the second", "the third"), \
a duration, or the final answer. KEEP the ITEM if it supplies even one piece \
the answer is built from -- a single event, a date (including the bracketed \
one), a name, or one instance of the kind the QUERY counts or chooses among. \
The counting, ordering, and arithmetic happen later, with all items together.

- "how many X has P done" or "which X was the Nth": KEEP every ITEM reporting \
one X by P, even if it never says how many or which number -- it is one \
instance to be counted later.
- "time between A and B": KEEP an ITEM reporting A or B with its date -- it \
is one endpoint.
- If only PART of the ITEM supplies a piece, KEEP it; ignore surrounding \
greetings or small talk.

DROP the ITEM only if it supplies no such piece. The single thing you may not \
do is invent what the ITEM is ABOUT: if making it relevant requires supposing \
it concerns the QUERY's subject when nothing in the ITEM -- its text, date, \
or speaker -- says so, DROP it. ("7" for "how many pets": nothing in it is \
about pets -> DROP. A message where one person reports THEIR OWN action, for \
a QUERY about someone else's action -> DROP.) But combining a piece the ITEM \
does state with other items later is NOT invention.

QUERY:
{query}

ITEM:
{doc}

Reply with exactly one token: KEEP or DROP."""


PROMPTS["v6"] = """\
You are given a QUERY and one candidate ITEM. The ITEM begins with its date \
and speaker in brackets -- like [Friday, May 20, 2022, 7:49 PM] Speaker: \
"..." -- and may contain a bracketed attachment description like [Attached a \
photo of ...: a red bicycle leaning on a fence]. The date, the speaker, AND \
the attachment description are ALL part of the ITEM's information: a date in \
the bracket is a date the ITEM supplies, and an attachment description can \
state facts (an object's appearance, a name, a place). Decide whether the \
ITEM supplies any information the QUERY's answer would use (KEEP) or none \
(DROP).

You are NOT answering the QUERY and NOT computing its answer. You do not need \
the ITEM to state a total, a count, an ordinal ("the second", "the third"), \
a duration, or the final answer. KEEP the ITEM if it supplies even one piece \
the answer is built from -- a single event, a date (including the bracketed \
one), a name, a description, or one instance of the kind the QUERY counts or \
chooses among. The counting, ordering, and arithmetic happen later, with all \
items together.

- "how many X has P done" or "which X was the Nth": KEEP every ITEM reporting \
one X by P, even if it never says how many or which number -- it is one \
instance to be counted later.
- "time between A and B": KEEP an ITEM reporting A or B with its date -- it \
is one endpoint.
- If only PART of the ITEM (including its attachment description) supplies a \
piece, KEEP it; ignore surrounding greetings or small talk.

DROP the ITEM only if it supplies no such piece. The single thing you may not \
do is invent what the ITEM is ABOUT: if making it relevant requires supposing \
it concerns the QUERY's subject when nothing in the ITEM -- its text, date, \
speaker, or attachment -- says so, DROP it. ("7" for "how many pets": nothing \
in it is about pets -> DROP. A message where one person reports THEIR OWN \
action, for a QUERY about someone else's action -> DROP.) But combining a \
piece the ITEM does state with other items later is NOT invention.

QUERY:
{query}

ITEM:
{doc}

Reply with exactly one token: KEEP or DROP."""


PROMPTS["v7"] = """\
You are given a QUERY and one candidate ITEM. The ITEM begins with its date \
and speaker in brackets -- like [Friday, May 20, 2022, 7:49 PM] Speaker: \
"..." -- and that date and speaker ARE part of the ITEM's information: a date \
in the bracket is a date the ITEM supplies. Decide whether the ITEM supplies \
any information the QUERY's answer would use (KEEP) or none (DROP).

You are NOT answering the QUERY and NOT computing its answer. You do not need \
the ITEM to state a total, a count, an ordinal ("the second", "the third"), \
a duration, or the final answer.

When the QUERY is about specific named instances of a TYPE -- the months \
between adopting A and B (type: adoptions), how many X someone has done \
(type: X-events), which Y was the Nth (type: Y-events) -- the answer is \
assembled from the individual events of that TYPE. So KEEP any ITEM that \
reports ONE event of that type by that person, WITH its date, EVEN IF the \
ITEM names a different instance, or names none at all. A single such event \
can rarely name itself as "B" or "the third" -- which event is which is \
resolved later by comparing all the events together. Requiring the ITEM to \
already identify itself as the specific instance is the main mistake to avoid.

Otherwise, KEEP the ITEM if it supplies even one piece the answer is built \
from -- a single event, a date (including the bracketed one), a name, or a \
detail that pins down what the QUERY refers to. If only PART of the ITEM \
supplies a piece, KEEP it; ignore surrounding greetings or small talk.

DROP the ITEM only if it supplies no such piece. The single thing you may not \
do is invent what the ITEM is ABOUT: if making it relevant requires supposing \
it concerns the QUERY's subject when nothing in the ITEM -- its text, date, \
or speaker -- says so, DROP it. ("7" for "how many pets": nothing in it is \
about pets -> DROP. One person reporting THEIR OWN action, for a QUERY about \
someone else's action -> DROP.) Note this is different from an event of the \
right type that simply names a different instance -- that you KEEP.

QUERY:
{query}

ITEM:
{doc}

Reply with exactly one token: KEEP or DROP."""


async def judge(client, prompt_tmpl, question, doc) -> str:
    prompt = prompt_tmpl.format(query=question, doc=doc)
    try:
        resp = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            extra_body={"reasoning_effort": EFFORT},
        )
    except Exception:
        resp = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
        )
    t = (resp.choices[0].message.content or "").strip().upper()
    return "DROP" if t.startswith("DROP") else "KEEP"


async def main() -> None:
    load_dotenv(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/"
        "locomo/.env"
    )
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cases = load_cases()
    print(f"\n{len(cases)} cases\n")
    # import the live module prompt as the baseline to compare against
    from swiss_rerank_probe import FILTER_PROMPT
    variants = {"v3_current": FILTER_PROMPT, **PROMPTS}
    N = 5  # samples per case -> majority vote (verdicts are stochastic)
    for name, tmpl in variants.items():
        # keep_rate[i] = fraction of N samples that returned KEEP for case i
        all_v = await asyncio.gather(*(
            asyncio.gather(*(
                judge(client, tmpl, q, doc) for _ in range(N)
            )) for q, doc, _, _ in cases
        ))
        keep_rate = [sum(v == "KEEP" for v in vs) / N for vs in all_v]
        maj = ["KEEP" if kr >= 0.5 else "DROP" for kr in keep_rate]
        correct = sum(
            1 for (_, _, exp, _), m in zip(cases, maj) if m == exp
        )
        print(f"=== {name}: {correct}/{len(cases)} by majority (N={N}) ===")
        for (q, doc, exp, label), m, kr in zip(cases, maj, keep_rate):
            mark = "OK " if m == exp else "XX "
            print(f"  {mark}[want {exp:4s} got {m:4s} keep={kr:.0%}] {label}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
