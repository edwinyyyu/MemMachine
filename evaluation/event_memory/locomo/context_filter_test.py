"""Test: does conditioning the filter judgment on surrounding context flip
the local-coreference cases (Buddy adoption) from DROP to KEEP, while still
dropping a genuinely-irrelevant control?

Renders the anchor turn in situ within its neighborhood, marked, and asks the
LLM to judge ONLY the marked candidate (neighbors are reference-resolution
context). Compares to the isolated (no-context) judgment.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from locomo_models import load_locomo_dataset
from swiss_rerank_probe import FILTER_PROMPT as V5_NOCTX

MODEL, EFFORT, N = "gpt-5-nano", "low", 7

CTX_PROMPT = """\
You are given a QUERY and a short slice of conversation. Exactly one turn is \
marked >>> CANDIDATE >>>. Judge ONLY the marked CANDIDATE: decide whether IT \
supplies any information the QUERY's answer would use (KEEP) or none (DROP). \
The other turns are CONTEXT -- use them only to resolve what the CANDIDATE \
refers to (a name, a "he"/"it", which instance it is); do not judge them.

You are NOT answering the QUERY. KEEP the CANDIDATE if it supplies even one \
piece the answer is built from -- an event, a date, a name, or one instance \
of the kind the QUERY counts or chooses among -- once the CONTEXT is used to \
resolve what it refers to. DROP only if, even read in context, it supplies no \
such piece. Do not invent connections the CANDIDATE plus its context do not \
state.

QUERY:
{query}

CONVERSATION:
{conversation}

Reply with exactly one token: KEEP or DROP."""


def find_conv():
    data = load_locomo_dataset("../../data/locomo10_c2sub.json")
    for item in data:
        if "conversation" not in item:
            continue
        conv = item["conversation"]
        msgs = {}
        order = []
        i = 0
        while True:
            i += 1
            sid = f"session_{i}"
            if sid not in conv:
                break
            dt = conv.get(f"{sid}_date_time", "")
            for m in conv[sid]:
                msgs[m["dia_id"]] = {"sp": m["speaker"], "t": m.get("text", ""),
                                     "dt": dt}
                order.append(m["dia_id"])
        if any(d.startswith("D24:") and "another pup"
               in msgs[d]["t"].lower() for d in msgs):
            return msgs, order
    raise SystemExit("conv not found")


def window(order, msgs, anchor, k=4):
    idx = order.index(anchor)
    lo, hi = max(0, idx - k), min(len(order), idx + k + 1)
    lines = []
    for d in order[lo:hi]:
        m = msgs[d]
        mark = " >>> CANDIDATE >>>" if d == anchor else ""
        lines.append(f"[{m['sp']}, {m['dt']}]{mark}: {m['t']}")
    return "\n".join(lines)


async def vote(client, prompt) -> float:
    async def one():
        try:
            r = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                extra_body={"reasoning_effort": EFFORT})
        except Exception:
            r = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}])
        t = (r.choices[0].message.content or "").strip().upper()
        return 0.0 if t.startswith("DROP") else 1.0
    return sum(await asyncio.gather(*(one() for _ in range(N)))) / N


async def main() -> None:
    load_dotenv(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/"
        "locomo/.env"
    )
    msgs, order = find_conv()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # KEEP target: Buddy adoption (anchor names no one; D24:6 names Buddy)
    q_buddy = "How many months passed between Andrew adopting Toby and Buddy?"
    anchor = "D24:2"
    cand_noctx = (f'[{msgs[anchor]["sp"]}, {msgs[anchor]["dt"]}]: '
                  f'{msgs[anchor]["t"]}')
    noctx = V5_NOCTX.format(query=q_buddy, doc=cand_noctx)
    ctx = CTX_PROMPT.format(query=q_buddy,
                            conversation=window(order, msgs, anchor, k=4))

    # DROP control: an unrelated turn (greeting) judged WITH context -- must
    # still drop (context must not make everything keep)
    ctrl_anchor = None
    for d in order:
        if d.startswith("D24:") and "what's been" in msgs[d]["t"].lower():
            ctrl_anchor = d
            break
    print("Buddy adoption anchor D24:2 (names no one in the turn itself):")
    print("  no-context KEEP-rate :", await vote(client, noctx))
    print("  +context  KEEP-rate  :", await vote(client, ctx))
    print("  (context window includes D24:6 'I named him Buddy')")
    if ctrl_anchor:
        cctx = CTX_PROMPT.format(
            query=q_buddy, conversation=window(order, msgs, ctrl_anchor, k=4))
        print(f"\nDROP control {ctrl_anchor} (greeting) +context KEEP-rate:",
              await vote(client, cctx))


if __name__ == "__main__":
    asyncio.run(main())
