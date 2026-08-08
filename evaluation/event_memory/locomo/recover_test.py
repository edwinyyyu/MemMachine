"""Try to recover the avoidable residual drops: context prompt + v7's
endpoint/instance rule + larger forward window. Avoidable cases should flip
to KEEP; unavoidable false-gold must STAY DROP (over-keep guard).
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from locomo_models import load_locomo_dataset
from context_filter_test import CTX_PROMPT as CTX_V1
from audit_ctx_residual import build_conv_index

MODEL, EFFORT, N = "gpt-5-nano", "low", 7

CTX_V3 = """\
You are given a QUERY and a short slice of conversation. Exactly one turn is \
marked >>> CANDIDATE >>>. Judge ONLY the marked CANDIDATE: decide whether IT \
supplies any information the QUERY's answer would use (KEEP) or none (DROP). \
The other turns are CONTEXT -- use them only to resolve what the CANDIDATE \
refers to (a name, a "he"/"it", which instance it is); do not judge them.

You are NOT answering the QUERY and NOT computing its answer. You do not need \
the CANDIDATE to state a total, a count, an ordinal, a duration, or the final \
answer. KEEP the CANDIDATE if it supplies even one piece the answer is built \
from, once CONTEXT resolves what it refers to:
- For a SPAN question -- "how long did X take", "how long did someone do X \
before/until some EVENT", "how many months/years between A and B" -- KEEP any \
item reporting an event that BOUNDS the span: its start, its end, or the \
EVENT it runs up to. The bounding event counts EVEN IF the item describes \
that event rather than the span's activity. Example: for "how long did she \
rehearse before the premiere", an item reporting the premiere (or opening \
night) is the end bound -> KEEP, though it never mentions rehearsing and \
states no duration.
- For "how many X" or "which X was the Nth": one instance of that kind (one \
award, one trip) is a needed piece. KEEP it, even if it never says how many \
or which number.
- Any event, date (including the bracketed one), or name the answer draws on.

DROP only if, even read in context, the CANDIDATE supplies no such piece -- \
it is about a different subject, a different person's action, or carries no \
event/fact the answer would use (e.g. it reports a meal when the QUERY asks \
about a pet, or a breed when the QUERY asks a name). Do not invent \
connections the CANDIDATE plus its context do not state.

QUERY:
{query}

CONVERSATION:
{conversation}

Reply with exactly one token: KEEP or DROP."""

# context-conditioning + endpoint/instance keeping (neutral prompt examples)
CTX_V2 = """\
You are given a QUERY and a short slice of conversation. Exactly one turn is \
marked >>> CANDIDATE >>>. Judge ONLY the marked CANDIDATE: decide whether IT \
supplies any information the QUERY's answer would use (KEEP) or none (DROP). \
The other turns are CONTEXT -- use them only to resolve what the CANDIDATE \
refers to (a name, a "he"/"it", which instance it is); do not judge them.

You are NOT answering the QUERY and NOT computing its answer. You do not need \
the CANDIDATE to state a total, a count, an ordinal, a duration, or the final \
answer. KEEP the CANDIDATE if it supplies even one piece the answer is built \
from, once CONTEXT resolves what it refers to:
- For "how long did X take" or "how many months/years between A and B": an \
event that is one ENDPOINT of that span -- its start or its end (e.g. \
"I finished the marathon", "we signed the lease", "I started the new job") -- \
is a needed piece. KEEP it, though it states no duration.
- For "how many X" or "which X was the Nth": one instance of that kind (one \
award, one trip) is a needed piece. KEEP it, even if it never says how many \
or which number.
- Any event, date (including the bracketed one), or name the answer draws on.

DROP only if, even read in context, the CANDIDATE supplies no such piece -- \
it is about a different subject, a different person's action, or carries no \
event/fact the answer would use (e.g. it reports a meal when the QUERY asks \
about a pet, or gives a breed when the QUERY asks a name). Do not invent \
connections the CANDIDATE plus its context do not state.

QUERY:
{query}

CONVERSATION:
{conversation}

Reply with exactly one token: KEEP or DROP."""

# (question, anchor dia_id, expected, label)
CASES = [
    # avoidable -> want KEEP (interval endpoints / instances)
    ("How long did John practice chess for before winning the chess tournament?",
     "D30:4", "KEEP", "endpoint: won the tournament"),
    ("How long did James and Samantha date for before deciding to move in together?",
     "D29:8", "KEEP", "endpoint: decided to move in"),
    ("How long was the car modification workshop in San Francisco?",
     "D14:1", "KEEP", "endpoint: went to the SF workshop"),
    ("How long did Dave's work on the Ford Mustang take?",
     "D20:1", "KEEP", "endpoint: worked on the Mustang engine"),
    ("When did Joanna start writing her third screenplay?",
     "D12:14", "KEEP", "instance: wrote a screenplay"),
    ("How many months passed between Andrew adopting Toby and Buddy?",
     "D24:2", "KEEP", "endpoint: adopted a pup (name 4 fwd)"),
    # unavoidable false-gold -> want DROP (over-keep guard)
    ("When did Nate get Tilly for Joanna?",
     "D24:2", "DROP", "false-gold: about a recipe"),
    ("What recipes has Joanna made?",
     "D21:3", "DROP", "false-gold: about data loss"),
    ("What kind of lighting does Nate's gaming room have?",
     "D10:2", "DROP", "false-gold: about a trilogy"),
    ("What are the names of Audrey's dogs?",
     "D19:12", "DROP", "wrong-attribute: gives breeds not names"),
]


def window(order, msgs, anchor, back=2, fwd=8):
    if anchor not in order:
        return None
    i = order.index(anchor)
    lo, hi = max(0, i - back), min(len(order), i + fwd + 1)
    return "\n".join(
        f"[{msgs[d]['sp']}, {msgs[d]['dt']}]"
        f"{' >>> CANDIDATE >>> ' if d == anchor else ''}: {msgs[d]['t']}"
        for d in order[lo:hi]
    )


async def vote(client, tmpl, q, conv):
    async def one():
        try:
            r = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user",
                    "content": tmpl.format(query=q, conversation=conv)}],
                extra_body={"reasoning_effort": EFFORT})
        except Exception:
            r = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user",
                    "content": tmpl.format(query=q, conversation=conv)}])
        t = (r.choices[0].message.content or "").strip().upper()
        return 0.0 if t.startswith("DROP") else 1.0
    return sum(await asyncio.gather(*(one() for _ in range(N)))) / N


async def main():
    load_dotenv(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/"
        "locomo/.env"
    )
    idx = build_conv_index(load_locomo_dataset("../../data/locomo10_c2sub.json"))
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for tname, tmpl in [("CTX_V2 (+endpoint rule)", CTX_V2), ("CTX_V3 (+bounding-event rule)", CTX_V3)]:
        ok = 0
        print(f"\n=== {tname} ===")
        for q, anchor, exp, label in CASES:
            msgs, order, _ = idx[q]
            conv = window(order, msgs, anchor)
            kr = await vote(client, tmpl, q, conv)
            got = "KEEP" if kr >= 0.5 else "DROP"
            mark = "OK " if got == exp else "XX "
            ok += got == exp
            print(f"  {mark}[want {exp:4s} got {got:4s} keep={kr:.0%}] {label}")
        print(f"  -> {ok}/{len(CASES)} correct")


if __name__ == "__main__":
    asyncio.run(main())
