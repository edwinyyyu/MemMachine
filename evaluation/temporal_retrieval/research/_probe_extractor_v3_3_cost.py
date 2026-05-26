"""Cost A/B: v3.2 vs v3.3 extractor — input/output tokens + latency.

Goal: quantify token savings from dropping `surface` and `granularity`
from the output schema (and corresponding prompt rules).

Methodology
-----------
- Run a fixed set of representative texts through both prompts AGAINST
  THE RAW API (bypassing the on-disk cache).
- Each call: wall-clock latency, input tokens, output tokens, reasoning
  tokens.
- Repeat each text N_REPEATS times to dampen noise.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_extractor_v3_3_cost
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

from temporal_retrieval.extractor_v3_2 import (
    MODEL as V32_MODEL,
    SINGLE_PASS_SYSTEM_V3_2,
    V3_2_JSON_SCHEMA,
)
from temporal_retrieval.extractor_v3_3 import (
    MODEL as V33_MODEL,
    SINGLE_PASS_SYSTEM_V3_3,
    V3_3_JSON_SCHEMA,
)
from temporal_retrieval.extractor_common import full_ref_context

from ._common import setup_env

setup_env()
if not os.environ.get("OPENAI_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


N_REPEATS = 3
CONCURRENCY = 4
REF = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


TEXTS = [
    ("pinpoint",       "Shipped the v5 release on March 15, 2024."),
    ("month",          "Worked on the migration in October 2023."),
    ("quarter",        "Q4 2023 was rough for the team."),
    ("year",           "In 2007 we launched the original product."),
    ("fuzzy_around",   "Around 2008 the team grew to ten people."),
    ("multi_te",       "Shipped in Q3 2023; replaced by the v2 in Q4 2024."),
    ("deictic",        "Yesterday I reviewed the new dashboard."),
    ("policy_skip",    "Policy: backups must run within the last hour."),
]


client = AsyncOpenAI()
sem = asyncio.Semaphore(CONCURRENCY)


async def one_call(system_prompt: str, schema: dict, model: str,
                   text: str) -> dict:
    ctx = full_ref_context(REF)
    user = f"{ctx}\n\nPassage:\n{text}"
    format_config: ResponseFormatTextJSONSchemaConfigParam = {
        "type": "json_schema",
        "strict": True,
        **schema,
    }
    text_config: ResponseTextConfigParam = {"format": format_config}

    async with sem:
        t0 = time.perf_counter()
        resp = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            text=text_config,
        )
        dt = time.perf_counter() - t0

    usage = resp.usage
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    reasoning_tok = 0
    out_det = getattr(usage, "output_tokens_details", None)
    if out_det is not None:
        reasoning_tok = int(getattr(out_det, "reasoning_tokens", 0) or 0)
    return {
        "latency_s": dt,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "reasoning_tok": reasoning_tok,
        "response_chars": len(resp.output_text or ""),
    }


async def measure(name: str, prompt: str, schema: dict, model: str) -> list[dict]:
    print(f"\n--- {name} ---", flush=True)
    tasks = []
    for (label, text) in TEXTS:
        for r in range(N_REPEATS):
            tasks.append((label, r, one_call(prompt, schema, model, text)))
    results = await asyncio.gather(*[t[2] for t in tasks])
    rows = []
    for (label, r, _), res in zip(tasks, results, strict=False):
        res["label"] = label
        res["rep"] = r
        rows.append(res)
        print(f"  {label:14s} rep={r}  lat={res['latency_s']:5.2f}s  "
              f"in={res['in_tok']:5d}  out={res['out_tok']:5d}  "
              f"(reasoning={res['reasoning_tok']})", flush=True)
    return rows


def summarize(name: str, rows: list[dict]) -> dict:
    lat = [r["latency_s"] for r in rows]
    in_t = [r["in_tok"] for r in rows]
    out_t = [r["out_tok"] for r in rows]
    reason_t = [r["reasoning_tok"] for r in rows]
    chars = [r["response_chars"] for r in rows]

    def m(xs):
        return mean(xs) if xs else 0.0

    def s(xs):
        return stdev(xs) if len(xs) > 1 else 0.0

    print(f"\n  {name} SUMMARY across {len(rows)} calls:", flush=True)
    print(f"    latency:   {m(lat):6.2f} ± {s(lat):5.2f}s", flush=True)
    print(f"    input_tok: {m(in_t):7.0f}", flush=True)
    print(f"    output_tok:{m(out_t):7.0f} ± {s(out_t):6.1f}  (incl reasoning={m(reason_t):.0f})", flush=True)
    print(f"    resp_chars:{m(chars):7.0f}", flush=True)
    return {"name": name, "lat": m(lat), "lat_std": s(lat),
            "in": m(in_t), "out": m(out_t), "out_std": s(out_t),
            "reasoning": m(reason_t), "chars": m(chars)}


async def main():
    print("PROMPT TEMPLATE SIZE (characters):", flush=True)
    print(f"  v3.2 prompt: {len(SINGLE_PASS_SYSTEM_V3_2):6d} chars", flush=True)
    print(f"  v3.3 prompt: {len(SINGLE_PASS_SYSTEM_V3_3):6d} chars", flush=True)
    delta = len(SINGLE_PASS_SYSTEM_V3_3) - len(SINGLE_PASS_SYSTEM_V3_2)
    pct = 100.0 * delta / len(SINGLE_PASS_SYSTEM_V3_2)
    print(f"  Δ           : {delta:+6d} chars ({pct:+.1f}%)", flush=True)

    print(f"\nTexts: {len(TEXTS)}  Repeats: {N_REPEATS}", flush=True)
    print(f"Total LLM calls per variant: {len(TEXTS) * N_REPEATS}", flush=True)

    v32 = await measure("v3.2", SINGLE_PASS_SYSTEM_V3_2, V3_2_JSON_SCHEMA, V32_MODEL)
    v33 = await measure("v3.3", SINGLE_PASS_SYSTEM_V3_3, V3_3_JSON_SCHEMA, V33_MODEL)

    s32 = summarize("v3.2", v32)
    s33 = summarize("v3.3", v33)

    print("\n" + "=" * 70, flush=True)
    print("COMPARISON", flush=True)
    print("=" * 70, flush=True)

    def line(a, b, label, unit=""):
        diff = b - a
        pct = 100.0 * diff / a if a else 0.0
        print(f"  {label:14s}  v3.2 {a:8.2f}{unit}   v3.3 {b:8.2f}{unit}   "
              f"Δ {diff:+7.2f}{unit} ({pct:+5.1f}%)", flush=True)

    line(s32["lat"], s33["lat"], "latency (s)", "s")
    line(s32["in"], s33["in"], "input_tok")
    line(s32["out"], s33["out"], "output_tok")
    line(s32["reasoning"], s33["reasoning"], "reasoning_tok")
    line(s32["chars"], s33["chars"], "resp_chars")


if __name__ == "__main__":
    asyncio.run(main())
