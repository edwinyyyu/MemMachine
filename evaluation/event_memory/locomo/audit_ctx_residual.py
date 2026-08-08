"""Audit what the CONTEXT-conditioned filter still drops, to bucket residual
gold-drops as reasonable (false-gold) vs avoidable (real error).

For each gold-drop question where A1_ctx still dropped >=1 gold, re-judge each
GOLD evidence message in its conversation context (majority vote), and print
the ones that DROP with their text + whether the answer is recoverable from
the surrounding turns.
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from locomo_models import load_locomo_dataset
from context_filter_test import CTX_PROMPT

MODEL, EFFORT, N, K = "gpt-5-nano", "low", 3, 4


def build_conv_index(data):
    """question -> (msgs dict, ordered dia_ids, evidence dia_ids)."""
    out = {}
    for item in data:
        if "conversation" not in item:
            continue
        conv = item["conversation"]
        msgs, order = {}, []
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
        for qa in item["qa"]:
            out[qa["question"]] = (msgs, order, qa.get("evidence", []))
    return out


def window(order, msgs, anchor, k=K):
    if anchor not in order:
        return None
    idx = order.index(anchor)
    lo = max(0, idx - (k // 3))
    hi = min(len(order), idx + (k - k // 3) + 1)
    lines = []
    for d in order[lo:hi]:
        m = msgs[d]
        mark = " >>> CANDIDATE >>> " if d == anchor else ""
        lines.append(f"[{m['sp']}, {m['dt']}]{mark}: {m['t']}")
    return "\n".join(lines)


async def vote(client, q, conv) -> float:
    async def one():
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user",
                           "content": CTX_PROMPT.format(query=q, conversation=conv)}],
                extra_body={"reasoning_effort": EFFORT})
        except Exception:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user",
                           "content": CTX_PROMPT.format(query=q, conversation=conv)}])
        t = (r.choices[0].message.content or "").strip().upper()
        return 0.0 if t.startswith("DROP") else 1.0
    return sum(await asyncio.gather(*(one() for _ in range(N)))) / N


async def main() -> None:
    load_dotenv(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/"
        "locomo/.env"
    )
    golddrop = json.load(open("ctxfilter-A-golddrop.json"))
    still = [r["question"] for r in golddrop["records"]
             if r["A1_ctx"]["gold_dropped"] > 0]
    idx = build_conv_index(load_locomo_dataset("../../data/locomo10_c2sub.json"))
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"{len(still)} questions where context filter still drops gold\n")
    dropped = []
    for q in still:
        if q not in idx:
            continue
        msgs, order, evidence = idx[q]
        for e in evidence:
            conv = window(order, msgs, e)
            if conv is None:
                continue
            kr = await vote(client, q, conv)
            if kr < 0.5:  # still dropped in context
                # does any neighbor in the window name/resolve it?
                dropped.append((q, e, msgs[e]["sp"], msgs[e]["t"]))
    print(f"=== {len(dropped)} gold still DROPPED in context (read & bucket) ===\n")
    for q, e, sp, t in dropped:
        print(f"Q: {q}")
        print(f"   [{e} {sp}]: {t[:160]}\n")


if __name__ == "__main__":
    asyncio.run(main())
