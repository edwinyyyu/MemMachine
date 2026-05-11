"""F (length-capped, schema-enforced) vs F-natural (no length guidance).

Goal: measure whether dropping the length budget from prompt + schema
buys us higher coverage at the cost of bigger segments.
"""

from __future__ import annotations

import argparse
import asyncio
import os

import openai
from compare_R_vs_RS import coverage
from dotenv import load_dotenv
from probe_segmenter_F import PROMPT_F, build_schema, verbatim_ratio
from probe_segmenter_F_natural import PROMPT_F_NATURAL, SCHEMA_NATURAL
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


async def call_with_schema(client, model, prompt, reasoning, schema):
    kwargs: dict = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "s",
                "schema": schema,
                "strict": True,
            }
        },
    }
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    try:
        resp = await client.responses.create(**kwargs)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    raw = (resp.output_text or "").strip()
    try:
        import json

        return json.loads(raw).get("segments", []), None
    except Exception as e:
        return None, f"parse: {e}"


async def run_F(client, model, reasoning, src, max_chars):
    schema = build_schema(max_chars)
    prompt = PROMPT_F.format(passage=src, max_chars=max_chars)
    return await call_with_schema(client, model, prompt, reasoning, schema)


async def run_Fn(client, model, reasoning, src):
    prompt = PROMPT_F_NATURAL.format(passage=src)
    return await call_with_schema(client, model, prompt, reasoning, SCHEMA_NATURAL)


def stats(segs, src):
    if not segs:
        return {"n": 0, "max_len": 0, "med_len": 0, "verb": 1.0, "cov": 0.0}
    lens = sorted(len(s) for s in segs)
    return {
        "n": len(segs),
        "max_len": lens[-1],
        "med_len": lens[len(lens) // 2],
        "verb": sum(verbatim_ratio(s, src) for s in segs) / len(segs),
        "cov": coverage(" ".join(segs), src),
    }


CELLS = [
    ("gpt-5-nano", "low"),
    ("gpt-5-nano", "medium"),
    ("gpt-5.4-nano", "low"),
    ("gpt-5.4-nano", "medium"),
    ("gpt-4o-mini", ""),
    ("gpt-4.1-nano", ""),
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--per-bucket", type=int, default=3)
    parser.add_argument("--min-length", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()

    bins = collect()
    samples = []
    for b, ts in bins.items():
        for t in [x for x in ts if len(x) >= args.min_length][: args.per_bucket]:
            samples.append((b, t))

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)
    print(f"Test: {len(samples)} chunks   F max_chars={args.max_chars}")
    print()
    print(
        f"{'cell':24s} | {'mode':9s} | {'cov':>4s} {'verb':>4s} "
        f"{'med':>4s} {'max':>4s} {'n':>3s}"
    )
    print("-" * 72)

    for model, reasoning in CELLS:

        async def one(src):
            async with sem:
                f, fe = await run_F(client, model, reasoning, src, args.max_chars)
                fn, fne = await run_Fn(client, model, reasoning, src)
            return (f, fe), (fn, fne)

        results = await asyncio.gather(*(one(t) for _, t in samples))
        f_rows = [
            stats(s, src)
            for ((s, e), _), (_, src) in zip(results, samples, strict=False)
            if e is None
        ]
        fn_rows = [
            stats(s, src)
            for (_, (s, e)), (_, src) in zip(results, samples, strict=False)
            if e is None
        ]

        def show(rows, label):
            n = len(rows)
            print(
                f"{model + ' ' + (reasoning or '(default)'):24s} | {label:9s} | "
                f"{sum(r['cov'] for r in rows) / n:>3.0%} "
                f"{sum(r['verb'] for r in rows) / n:>3.0%} "
                f"{sum(r['med_len'] for r in rows) / n:>4.0f} "
                f"{sum(r['max_len'] for r in rows) / n:>4.0f} "
                f"{sum(r['n'] for r in rows) / n:>3.0f}"
            )

        show(f_rows, "F (cap)")
        show(fn_rows, "Fnatural")
        print("-" * 72)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
