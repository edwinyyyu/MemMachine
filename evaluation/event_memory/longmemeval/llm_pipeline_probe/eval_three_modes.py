"""Robustness eval comparing F vs U1 vs U2 across model x reasoning.

  F  - structured: JSON schema list of strings, per-string maxLength.
  U1 - unstructured rewrite, LLM unaware of splitter, deterministic
       RecursiveCharacterTextSplitter on the result.
  U2 - unstructured rewrite, LLM told "use \\n\\n between segments";
       trivial paragraph splitter.

All three share the same prompt rules: verbatim quoting, drop generic
scaffolding, preserve order, max segment length.

Metrics (no safety net, exactly as the LLM produced):
  - schema_ok / parse_ok rate
  - verbatim ratio (longest-substring coverage)
  - distinctive-token coverage
  - avg max segment length, over-cap rate
  - preserved-order rate (does each segment appear in source order?)
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import re

import openai
from compare_R_vs_RS import coverage
from dotenv import load_dotenv
from probe_segmenter_F import PROMPT_F, build_schema, verbatim_ratio
from probe_segmenter_U import PROMPT_U1, PROMPT_U2, split_u1, split_u2
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


SCAFFOLD_RE = re.compile(
    r"\b(what a great question|i hope this helps|great question|"
    r"glad I could help|let me know if you have any other questions|"
    r"feel free to (ask|reach out))\b",
    re.IGNORECASE,
)


async def call_f(client, model, prompt, reasoning, schema):
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
        return None, f"api_error: {type(e).__name__}: {e}"
    raw = (resp.output_text or "").strip()
    try:
        import json

        return json.loads(raw), None
    except Exception as e:
        return None, f"parse_error: {e}"


async def call_u(client, model, prompt, reasoning):
    kwargs: dict = {"model": model, "input": prompt}
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    try:
        resp = await client.responses.create(**kwargs)
    except Exception as e:
        return None, f"api_error: {type(e).__name__}: {e}"
    return (resp.output_text or "").strip(), None


async def run_F(client, model, reasoning, src, max_chars):
    schema = build_schema(max_chars)
    prompt = PROMPT_F.format(passage=src, max_chars=max_chars)
    parsed, err = await call_f(client, model, prompt, reasoning, schema)
    if parsed is None:
        return None, err
    return parsed.get("segments", []), None


async def run_U1(client, model, reasoning, src, max_chars):
    prompt = PROMPT_U1.format(passage=src)
    text, err = await call_u(client, model, prompt, reasoning)
    if text is None:
        return None, err
    return split_u1(text, max_chars), None


async def run_U2(client, model, reasoning, src, max_chars):
    prompt = PROMPT_U2.format(passage=src, max_chars=max_chars)
    text, err = await call_u(client, model, prompt, reasoning)
    if text is None:
        return None, err
    return split_u2(text), None


def _stats(segs: list[str], src: str, max_chars: int) -> dict:
    if not segs:
        return {
            "n": 0,
            "max_len": 0,
            "verbatim": 1.0,
            "cov": 0.0,
            "scaffold": 0,
            "over_cap": 0,
            "order_ok": 1.0,
        }
    verb = sum(verbatim_ratio(s, src) for s in segs) / len(segs)
    cov = coverage(" ".join(segs), src)
    scaffold = sum(len(SCAFFOLD_RE.findall(s)) for s in segs)
    over_cap = sum(1 for s in segs if len(s) > max_chars)
    # Preserved-order: do segments' first 60-char anchors appear in source
    # in the same order? (Verbatim segments will, by construction.)
    anchors = []
    for s in segs:
        head = s[:60]
        idx = src.find(head)
        anchors.append(idx if idx >= 0 else -1)
    in_order = [a for a in anchors if a >= 0]
    order_ok = (
        (
            sum(1 for x, y in itertools.pairwise(in_order) if x <= y)
            / max(len(in_order) - 1, 1)
        )
        if len(in_order) > 1
        else 1.0
    )
    return {
        "n": len(segs),
        "max_len": max(len(s) for s in segs),
        "verbatim": verb,
        "cov": cov,
        "scaffold": scaffold,
        "over_cap": over_cap,
        "order_ok": order_ok,
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
    print(f"Test: {len(samples)} chunks  max_chars={args.max_chars}")
    print()

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    async def cell_run(model, reasoning):
        async def one(src):
            async with sem:
                f_segs, f_err = await run_F(
                    client, model, reasoning, src, args.max_chars
                )
                u1_segs, u1_err = await run_U1(
                    client, model, reasoning, src, args.max_chars
                )
                u2_segs, u2_err = await run_U2(
                    client, model, reasoning, src, args.max_chars
                )
            return (f_segs, f_err), (u1_segs, u1_err), (u2_segs, u2_err)

        results = await asyncio.gather(*(one(t) for _, t in samples))
        agg = {"F": [], "U1": [], "U2": []}
        for (bucket, src), ((fs, fe), (u1s, u1e), (u2s, u2e)) in zip(
            samples, results, strict=False
        ):
            if fe is None:
                agg["F"].append(_stats(fs, src, args.max_chars))
            if u1e is None:
                agg["U1"].append(_stats(u1s, src, args.max_chars))
            if u2e is None:
                agg["U2"].append(_stats(u2s, src, args.max_chars))
        return agg

    print(
        f"{'cell':24s} | {'mode':4s} | {'cov':>4s} {'verb':>4s} "
        f"{'maxL':>4s} {'scaf':>4s} {'oc%':>4s} {'ord%':>4s} {'n':>3s}"
    )
    print("-" * 80)

    for model, reasoning in CELLS:
        agg = await cell_run(model, reasoning)
        cell = f"{model} {reasoning or '(default)'}"
        for mode in ("F", "U1", "U2"):
            rs = agg[mode]
            if not rs:
                print(f"{cell:24s} | {mode:4s} | -- all failed --")
                continue
            n = len(rs)
            cov_avg = sum(r["cov"] for r in rs) / n
            verb_avg = sum(r["verbatim"] for r in rs) / n
            ml_avg = sum(r["max_len"] for r in rs) / n
            scaffold_avg = sum(r["scaffold"] for r in rs) / n
            oc_rate = sum(r["over_cap"] for r in rs) / max(sum(r["n"] for r in rs), 1)
            order_avg = sum(r["order_ok"] for r in rs) / n
            avg_n = sum(r["n"] for r in rs) / n
            print(
                f"{cell:24s} | {mode:4s} | "
                f"{cov_avg:>3.0%} {verb_avg:>3.0%} "
                f"{ml_avg:>4.0f} {scaffold_avg:>4.2f} "
                f"{oc_rate:>3.0%} {order_avg:>3.0%} "
                f"{avg_n:>3.0f}"
            )
        print("-" * 80)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
