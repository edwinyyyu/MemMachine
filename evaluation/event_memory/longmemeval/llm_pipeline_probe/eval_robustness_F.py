"""Robustness eval for Mode F across model x reasoning, NO safety net.

Mode F: full passage to LLM → JSON list of segment strings via strict
json_schema with per-segment maxLength. No pre-split of any kind.

Three goals (same as before):
  1. ~0 hallucinations          → measured by verbatim ratio per segment.
                                   With strong prompt, should be ≥99%.
  2. high coverage of memorable → distinctive-token recall vs source.
     content
  3. robust across cells        → above stable across model x reasoning.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re

import openai
from compare_R_vs_RS import coverage
from dotenv import load_dotenv
from probe_segmenter_F import PROMPT_F, build_schema, verbatim_ratio
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


SCAFFOLDING_PATTERNS = [
    r"\bwhat a great question\b",
    r"\bI hope this helps\b",
    r"\bgreat question\b",
    r"\bglad I could help\b",
    r"\blet me know if you have any other questions\b",
    r"\bfeel free to (ask|reach out)\b",
]
_SCAFFOLD_RE = re.compile("|".join(SCAFFOLDING_PATTERNS), re.IGNORECASE)


async def call_f(client, model, prompt, reasoning, schema):
    kwargs: dict = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "segments_response",
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
        return None, f"api_error: {type(e).__name__}: {e}"
    raw = (resp.output_text or "").strip()
    try:
        import json

        return json.loads(raw), None
    except Exception as e:
        return None, f"parse_error: {e}"


async def go_one(client, model, reasoning, src, max_chars):
    schema = build_schema(max_chars)
    prompt = PROMPT_F.format(passage=src, max_chars=max_chars)
    parsed, err = await call_f(client, model, prompt, reasoning, schema)
    if parsed is None:
        return [], err
    return parsed.get("segments", []), None


async def run_cell(client, model, reasoning, samples, max_chars, sem):
    async def go(text):
        async with sem:
            return await go_one(client, model, reasoning, text, max_chars)

    rep = await asyncio.gather(*(go(t) for _, t in samples))

    rows = []
    for (bucket, src), (segs, err) in zip(samples, rep, strict=False):
        if err:
            rows.append({"err": err})
            continue
        verb = sum(verbatim_ratio(s, src) for s in segs) / max(len(segs), 1)
        seg_lens = [len(s) for s in segs]
        cov = coverage(" ".join(segs), src) if segs else 0.0
        scaffold_hits = sum(len(_SCAFFOLD_RE.findall(s)) for s in segs)
        rows.append(
            {
                "verbatim": verb,
                "cov": cov,
                "max_seg_len": max(seg_lens) if seg_lens else 0,
                "n_segs": len(segs),
                "over_cap": sum(1 for L in seg_lens if max_chars < L),
                "scaffold_hits": scaffold_hits,
                "err": None,
            }
        )
    return rows


def summarize(model, reasoning, rows):
    errs = [r for r in rows if r.get("err")]
    ok = [r for r in rows if not r.get("err")]
    if not ok:
        return f"{model:14s} {reasoning:9s}  ALL FAILED ({errs[0]['err']})"
    n = len(ok)
    schema_ok = (n / (n + len(errs))) * 100
    return (
        f"{model:14s} {reasoning:9s}  "
        f"schema_ok={schema_ok:.0f}%  "
        f"cov={sum(r['cov'] for r in ok) / n:.0%}  "
        f"verbatim={sum(r['verbatim'] for r in ok) / n:.0%}  "
        f"scaffold/chunk={sum(r['scaffold_hits'] for r in ok) / n:.2f}  "
        f"avg_segs={sum(r['n_segs'] for r in ok) / n:.1f}  "
        f"avg_maxlen={sum(r['max_seg_len'] for r in ok) / n:.0f}  "
        f"over_cap_rate={sum(r['over_cap'] for r in ok) / sum(r['n_segs'] for r in ok):.0%}"
    )


CELLS = [
    ("gpt-5-nano", "minimal"),
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
    parser.add_argument("--per-bucket", type=int, default=4)
    parser.add_argument("--min-length", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()

    bins = collect()
    samples = []
    for b, ts in bins.items():
        for t in [x for x in ts if len(x) >= args.min_length][: args.per_bucket]:
            samples.append((b, t))
    print(f"Test: {len(samples)} chunks  max_chars={args.max_chars}")
    print()

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)
    summaries = []
    for model, reasoning in CELLS:
        rows = await run_cell(client, model, reasoning, samples, args.max_chars, sem)
        summary = summarize(model, reasoning or "(default)", rows)
        summaries.append(summary)
        print(summary)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
