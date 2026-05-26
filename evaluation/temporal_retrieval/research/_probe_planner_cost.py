"""Cost A/B: DNF vs Tree planner — input/output tokens + latency.

Goal: if accuracy is the same, pick the cheaper one.

Methodology
-----------
- Run a fixed set of representative queries (single leaf, AND, OR, NOT,
  no-scope) through both prompts AGAINST THE RAW API (bypassing the
  on-disk cache so we get a real LLM round-trip every time).
- For each call, capture: wall-clock latency, input tokens, output
  tokens, and the (reasoning + cached) breakdown returned by the API.
- Repeat each query N_REPEATS times to dampen noise, then average.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._probe_planner_cost
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from statistics import mean, stdev

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

from temporal_retrieval.planner import (
    MODEL as DNF_MODEL,
    PLAN_PROMPT as DNF_PROMPT,
    _PLAN_JSON_SCHEMA as DNF_SCHEMA,
)
from temporal_retrieval.planner_tree import (
    MODEL as TREE_MODEL,
    TREE_PLAN_PROMPT as TREE_PROMPT,
    _PLAN_JSON_SCHEMA as TREE_SCHEMA,
)

from ._common import setup_env

setup_env()
if not os.environ.get("OPENAI_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


N_REPEATS = 3  # samples per query
CONCURRENCY = 4  # cap on simultaneous in-flight calls

# Representative queries — mix of shapes the planner sees in production.
QUERIES = [
    ("simple_leaf",       "What did I work on in Q4 2023?",                            "2024-01-15T12:00:00Z"),
    ("relative_deictic",  "What happened last quarter?",                               "2024-04-15T12:00:00Z"),
    ("after_leaf",        "Updates after the migration shipped",                       "2024-06-01T12:00:00Z"),
    ("composition_and",   "in 2024 not in summer",                                     "2025-01-01T12:00:00Z"),
    ("explicit_or",       "in Q1 or Q4 of 2023",                                       "2024-03-01T12:00:00Z"),
    ("extremum_latest",   "What was my latest budget review",                          "2024-09-01T12:00:00Z"),
    ("no_scope",          "Notes from the team retreat",                               "2024-07-01T12:00:00Z"),
    ("complex_excl",      "Recent changes in Q1 2024 excluding February",              "2024-05-01T12:00:00Z"),
]


client = AsyncOpenAI()
sem = asyncio.Semaphore(CONCURRENCY)


async def one_call(prompt: str, schema: dict, schema_name: str,
                   model: str, query: str, ref_time: str) -> dict:
    """Single uncached round-trip. Returns timing + usage."""
    text_to_send = prompt.format(query=query, ref_time=ref_time)
    format_config: ResponseFormatTextJSONSchemaConfigParam = {
        "type": "json_schema",
        "name": schema_name,
        "strict": True,
        "schema": schema,
    }
    text_config: ResponseTextConfigParam = {"format": format_config}

    async with sem:
        t0 = time.perf_counter()
        resp = await client.responses.create(
            model=model,
            input=text_to_send,
            text=text_config,
        )
        dt = time.perf_counter() - t0

    usage = resp.usage
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    # reasoning tokens are billed inside output_tokens on gpt-5-mini;
    # capture if available for transparency.
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


async def measure_variant(name: str, prompt: str, schema: dict,
                          schema_name: str, model: str) -> list[dict]:
    print(f"\n--- {name} ---", flush=True)
    rows: list[dict] = []
    # Launch all (queries × repeats) concurrently so the two variants
    # face similar API conditions.
    tasks = []
    for (qid, query, ref_time) in QUERIES:
        for r in range(N_REPEATS):
            tasks.append((qid, r, one_call(prompt, schema, schema_name, model, query, ref_time)))
    results = await asyncio.gather(*[t[2] for t in tasks])
    for (qid, r, _), res in zip(tasks, results, strict=False):
        res["qid"] = qid
        res["rep"] = r
        rows.append(res)
        print(f"  {qid:18s} rep={r}  lat={res['latency_s']:5.2f}s  "
              f"in={res['in_tok']:5d}  out={res['out_tok']:5d}  "
              f"(reasoning={res['reasoning_tok']})", flush=True)
    return rows


def summarize(name: str, rows: list[dict]) -> dict:
    lat = [r["latency_s"] for r in rows]
    in_t = [r["in_tok"] for r in rows]
    out_t = [r["out_tok"] for r in rows]
    reason_t = [r["reasoning_tok"] for r in rows]
    chars = [r["response_chars"] for r in rows]
    n = len(rows)

    def m_s(xs):
        if not xs:
            return (0.0, 0.0)
        return (mean(xs), stdev(xs) if len(xs) > 1 else 0.0)

    lm, ls = m_s(lat)
    im, _ = m_s(in_t)
    om, os_ = m_s(out_t)
    rm, _ = m_s(reason_t)
    cm, _ = m_s(chars)
    print(f"\n  {name} SUMMARY across {n} calls:", flush=True)
    print(f"    latency:   {lm:6.2f} ± {ls:5.2f}s", flush=True)
    print(f"    input_tok: {im:7.0f}  (prompt size)", flush=True)
    print(f"    output_tok:{om:7.0f} ± {os_:6.1f}  (incl reasoning={rm:.0f})", flush=True)
    print(f"    resp_chars:{cm:7.0f}", flush=True)
    return {"name": name, "n": n, "lat_mean": lm, "lat_std": ls,
            "in_mean": im, "out_mean": om, "out_std": os_,
            "reasoning_mean": rm, "chars_mean": cm}


async def main():
    # Static input-token estimate from the prompt template alone
    # (independent of query). Useful to compare prompt body size up-front.
    print("PROMPT TEMPLATE SIZE (characters, before {query}/{ref_time} substitution):", flush=True)
    print(f"  DNF  prompt: {len(DNF_PROMPT):6d} chars", flush=True)
    print(f"  TREE prompt: {len(TREE_PROMPT):6d} chars", flush=True)
    delta_chars = len(TREE_PROMPT) - len(DNF_PROMPT)
    pct = 100.0 * delta_chars / len(DNF_PROMPT)
    print(f"  Δ           : {delta_chars:+6d} chars ({pct:+.1f}%)", flush=True)

    print(f"\nQueries: {len(QUERIES)}  Repeats: {N_REPEATS}  Concurrency: {CONCURRENCY}", flush=True)
    print(f"Total LLM calls per variant: {len(QUERIES) * N_REPEATS}", flush=True)

    dnf_rows = await measure_variant("DNF", DNF_PROMPT, DNF_SCHEMA, "query_plan", DNF_MODEL)
    tree_rows = await measure_variant("TREE", TREE_PROMPT, TREE_SCHEMA, "query_plan_tree", TREE_MODEL)

    s_dnf = summarize("DNF", dnf_rows)
    s_tree = summarize("TREE", tree_rows)

    print("\n" + "=" * 70, flush=True)
    print("COMPARISON", flush=True)
    print("=" * 70, flush=True)

    def delta(d, t, label, unit=""):
        diff = t - d
        pct = 100.0 * diff / d if d else 0.0
        print(f"  {label:14s}  DNF {d:8.2f}{unit}   TREE {t:8.2f}{unit}   "
              f"Δ {diff:+7.2f}{unit} ({pct:+5.1f}%)", flush=True)

    delta(s_dnf["lat_mean"], s_tree["lat_mean"], "latency (s)", "s")
    delta(s_dnf["in_mean"], s_tree["in_mean"], "input_tok")
    delta(s_dnf["out_mean"], s_tree["out_mean"], "output_tok")
    delta(s_dnf["reasoning_mean"], s_tree["reasoning_mean"], "reasoning_tok")
    delta(s_dnf["chars_mean"], s_tree["chars_mean"], "resp_chars")


if __name__ == "__main__":
    asyncio.run(main())
