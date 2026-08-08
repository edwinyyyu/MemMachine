"""Run the CURRENT module filter prompt (v5) over every gold the prior run
dropped, with majority vote, to find systematic over-DROP issues cheaply.

For each (question, dropped-gold-doc) from swiss-probe-filter-p100-v3.json,
vote N times. Print which v5 now KEEPS (recovered) vs still DROPS, and dump
the still-dropped ones (full text) for bucketing: systematic bug / false-gold
/ implicit-edge.

Usage:  uv run python run_v5_on_drops.py
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from swiss_rerank_probe import FILTER_PROMPT  # current = v5

MODEL, EFFORT, N = "gpt-5-nano", "low", 3


async def vote(client, question, doc) -> float:
    async def one():
        p = FILTER_PROMPT.format(query=question, doc=doc)
        try:
            r = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": p}],
                extra_body={"reasoning_effort": EFFORT})
        except Exception:
            r = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": p}])
        t = (r.choices[0].message.content or "").strip().upper()
        return 0.0 if t.startswith("DROP") else 1.0
    return sum(await asyncio.gather(*(one() for _ in range(N)))) / N


async def main() -> None:
    load_dotenv(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/"
        "locomo/.env"
    )
    d = json.load(open("swiss-probe-filter-p100-v3.json"))
    pairs = [
        (r["category"], r["question"], doc)
        for r in d["records"] for doc in r.get("gold_drop_docs", [])
    ]
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    rates = await asyncio.gather(*(vote(client, q, doc) for _, q, doc in pairs))

    recovered = [(p, r) for p, r in zip(pairs, rates) if r >= 0.5]
    still = [(p, r) for p, r in zip(pairs, rates) if r < 0.5]
    print(f"\n{len(pairs)} prior-dropped gold | v5 KEEPS {len(recovered)} "
          f"(recovered) | still DROPS {len(still)}")
    by_cat = {}
    for (cat, _, _), r in zip(pairs, rates):
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0 if r >= 0.5 else 1] += 1
    print("by cat [recovered, still-dropped]:",
          {c: v for c, v in sorted(by_cat.items())})
    print(f"\n===== {len(still)} STILL DROPPED (read & bucket) =====")
    for (cat, q, doc), r in sorted(still, key=lambda x: x[0][0]):
        print(f"\n[c{cat} keep={r:.0%}] Q: {q}")
        print(f"   {doc}")


if __name__ == "__main__":
    asyncio.run(main())
