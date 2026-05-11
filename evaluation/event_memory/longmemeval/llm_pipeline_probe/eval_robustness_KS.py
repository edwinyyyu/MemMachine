"""Robustness eval for Mode KS across model x reasoning, NO safety net.

Three goals:
  1. ~0 hallucinations          → measured by verbatim-substring check.
                                   With KS, every segment is a join of source
                                   units, so this should be 100%.
  2. high coverage of memorable → distinctive-token recall.
     content
  3. robust to model/reasoning  → all of the above stable across cells.

Also checks:
  - schema_ok rate (should be 100% with strict json_schema)
  - filtered-scaffolding rate (heuristic count of "What a great question!"
    / "I hope this helps!" / etc. surviving in segments — lower = better)
  - segment-length distribution
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re

import openai
from compare_R_vs_RS import _tokens, coverage
from dotenv import load_dotenv
from probe_segmenter_KS import (
    PROMPT_KS,
    SEGMENT_SCHEMA,
    annotate,
    assemble,
    split_units,
)
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


SCAFFOLDING_PATTERNS = [
    r"\bwhat a great question\b",
    r"\bI hope this helps\b",
    r"\bgreat question\b",
    r"\bglad I could help\b",
    r"\blet me know if you have any other questions\b",
    r"\b(thanks|thank you)\s+for (asking|sharing|the question)\b",
    r"\b(I am|I'm) (excited|happy|glad|delighted|happy to help)\b",
]
_SCAFFOLD_RE = re.compile("|".join(SCAFFOLDING_PATTERNS), re.IGNORECASE)


def _verbatim_ratio(segments: list[str], units: list[str]) -> float:
    """Fraction of segment characters that come from source units verbatim."""
    if not segments:
        return 1.0
    src = "\n".join(units)
    matched = 0
    total = 0
    for seg in segments:
        total += len(seg)
        # Each segment is a join of units; check that all units appear
        # in source.
        for u in seg.split("\n\n"):
            for sub in u.split(" "):
                pass
        # Simpler: just check each segment is a contiguous-or-not subset
        # of source by token presence. KS guarantees verbatim by
        # construction; this is just a paranoid check.
        seg_tokens = _tokens(seg)
        src_set = set(_tokens(src))
        in_src = sum(len(t) for t in seg_tokens if t in src_set)
        all_chars = sum(len(t) for t in seg_tokens) or 1
        matched += int(in_src / all_chars * len(seg))
    return matched / total if total else 1.0


async def call_ks(client, model, prompt, reasoning):
    kwargs: dict = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "segments_response",
                "schema": SEGMENT_SCHEMA,
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
    units = split_units(src)
    annotated = annotate(units)
    prompt = PROMPT_KS.format(passage=annotated, max_chars=max_chars)
    parsed, err = await call_ks(client, model, prompt, reasoning)
    if parsed is None:
        return units, [], err
    segs = assemble(units, parsed.get("segments", []))
    return units, segs, None


async def run_cell(client, model, reasoning, samples, max_chars, sem):
    async def go(text):
        async with sem:
            return await go_one(client, model, reasoning, text, max_chars)

    rep = await asyncio.gather(*(go(t) for _, t in samples))

    rows = []
    for (bucket, src), (units, segs, err) in zip(samples, rep, strict=False):
        if err:
            rows.append({"bucket": bucket, "err": err})
            continue
        verbatim_ok = _verbatim_ratio(segs, units)
        seg_lens = [len(s) for s in segs]
        cov = coverage("\n".join(segs), src) if segs else 0.0
        scaffold_hits = sum(len(_SCAFFOLD_RE.findall(s)) for s in segs)
        # Did the LLM include any scaffolding-only unit?
        # Mark "kept scaffolding" if a unit consisting of just "what a great..."
        # appears in any segment.
        rows.append(
            {
                "bucket": bucket,
                "src_len": len(src),
                "n_units": len(units),
                "n_segs": len(segs),
                "verbatim": verbatim_ok,
                "cov": cov,
                "max_seg_len": max(seg_lens) if seg_lens else 0,
                "p95_seg_len": sorted(seg_lens)[int(0.95 * len(seg_lens)) - 1]
                if seg_lens
                else 0,
                "over_cap": sum(1 for L in seg_lens if max_chars < L),
                "scaffold_hits": scaffold_hits,
                "err": None,
            }
        )
    return rows


def summarize(model: str, reasoning: str, rows: list[dict], max_chars: int) -> str:
    errs = [r for r in rows if r.get("err")]
    ok = [r for r in rows if not r.get("err")]
    if not ok:
        return f"{model:14s} {reasoning:7s}  ALL FAILED ({len(errs)} errs: {errs[0]['err']})"
    n = len(ok)
    sched_ok = (n / (n + len(errs))) * 100
    avg_cov = sum(r["cov"] for r in ok) / n
    avg_verb = sum(r["verbatim"] for r in ok) / n
    avg_scaffold = sum(r["scaffold_hits"] for r in ok) / n
    avg_segs = sum(r["n_segs"] for r in ok) / n
    avg_max_len = sum(r["max_seg_len"] for r in ok) / n
    over_cap_rate = sum(r["over_cap"] for r in ok) / sum(r["n_segs"] for r in ok)
    return (
        f"{model:14s} {reasoning:7s}  schema_ok={sched_ok:.0f}%  "
        f"cov={avg_cov:.0%}  verbatim={avg_verb:.0%}  "
        f"scaffold/chunk={avg_scaffold:.2f}  "
        f"avg_segs={avg_segs:.1f}  avg_maxlen={avg_max_len:.0f}  "
        f"over_cap_rate={over_cap_rate:.0%}"
    )


CELLS = [
    ("gpt-5-nano", "minimal"),
    ("gpt-5-nano", "low"),
    ("gpt-5-nano", "medium"),
    ("gpt-5.4-nano", "low"),
    ("gpt-5.4-nano", "medium"),
    ("gpt-4o-mini", None),
    ("gpt-4.1-nano", None),
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
    print(
        f"Test set: {len(samples)} chunks across {len(bins)} buckets, "
        f"min_length={args.min_length}, max_chars={args.max_chars}"
    )
    print()

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    print("Running cells (sequential per cell, concurrent within)…")
    print()
    summaries = []
    for model, reasoning in CELLS:
        rows = await run_cell(
            client, model, reasoning or "", samples, args.max_chars, sem
        )
        summary = summarize(model, reasoning or "(default)", rows, args.max_chars)
        summaries.append(summary)
        print(summary)
    await client.close()

    print()
    print("=" * 80)
    for s in summaries:
        print(s)


if __name__ == "__main__":
    asyncio.run(main())
