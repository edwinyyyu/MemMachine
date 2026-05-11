"""Side-by-side R vs RS on the same inputs.

Mode R  - LLM rewrites only; we split with RecursiveCharacterTextSplitter.
Mode RS - LLM rewrites + splits in one step (JSON list).

We measure:
  - Faithfulness: does every word in the segment appear (case-insensitive,
    after light normalization) in the source? Reports: OOV-token ratio.
    Low is good; high suggests hallucination or aggressive paraphrase.
  - Coverage: fraction of distinctive content tokens (≥4-char words that
    appear ≤3 times in the source) recovered in the output.
  - Length: did any segment exceed the cap?

Eyeball-grade comparison only; this is a feasibility probe, not a benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re

import openai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_v2 import PROMPT_R, PROMPT_RS, parse_rs
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


_TOKEN_RE = re.compile(r"[a-zA-Z0-9\$\.\-/]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
    "from",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "will",
    "would",
    "could",
    "should",
    "can",
    "may",
    "i",
    "you",
    "we",
    "they",
    "it",
    "this",
    "that",
    "these",
    "those",
    "one",
    "two",
    "three",
}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _distinctive(src: str) -> set[str]:
    counts: dict[str, int] = {}
    for t in _tokens(src):
        if len(t) >= 4 and t not in _STOPWORDS:
            counts[t] = counts.get(t, 0) + 1
    return {t for t, n in counts.items() if n <= 3}


def faithfulness_oov(seg: str, source: str) -> tuple[int, int]:
    src_set = set(_tokens(source))
    out_tokens = _tokens(seg)
    out_tokens = [t for t in out_tokens if len(t) >= 3]
    if not out_tokens:
        return 0, 0
    oov = sum(1 for t in out_tokens if t not in src_set)
    return oov, len(out_tokens)


def coverage(output_text: str, source: str) -> float:
    distinctive = _distinctive(source)
    if not distinctive:
        return 1.0
    out_tokens = set(_tokens(output_text))
    hit = sum(1 for d in distinctive if d in out_tokens)
    return hit / len(distinctive)


async def call(client, model, prompt, reasoning):
    kwargs: dict = {"model": model, "input": prompt}
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    return (resp.output_text or "").strip()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--reasoning", default="low")
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

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.max_chars,
        chunk_overlap=0,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",
        ],
        keep_separator="end",
    )

    async def go(text: str) -> tuple[str, str]:
        async with sem:
            r = await call(
                client, args.model, PROMPT_R.format(passage=text), args.reasoning
            )
            rs = await call(
                client,
                args.model,
                PROMPT_RS.format(passage=text, max_chars=args.max_chars),
                args.reasoning,
            )
        return r, rs

    rep = await asyncio.gather(*(go(t) for _, t in samples))
    await client.close()

    # Aggregate
    rows = []
    for (bucket, src), (r_reply, rs_reply) in zip(samples, rep, strict=False):
        # R: rewrite then deterministic split
        r_segs = splitter.split_text(r_reply)
        # RS: parse JSON
        rs_segs = parse_rs(rs_reply) or [rs_reply]  # fallback if parse fails
        rs_parsed = parse_rs(rs_reply) is not None

        r_oov_n, r_oov_d = 0, 0
        rs_oov_n, rs_oov_d = 0, 0
        for s in r_segs:
            n, d = faithfulness_oov(s, src)
            r_oov_n += n
            r_oov_d += d
        for s in rs_segs:
            n, d = faithfulness_oov(s, src)
            rs_oov_n += n
            rs_oov_d += d

        rows.append(
            {
                "bucket": bucket,
                "src_len": len(src),
                "r_segs": len(r_segs),
                "r_max": max(len(s) for s in r_segs) if r_segs else 0,
                "r_oov_pct": (r_oov_n / r_oov_d * 100) if r_oov_d else 0.0,
                "r_cov": coverage(r_reply, src),
                "rs_segs": len(rs_segs),
                "rs_max": max(len(s) for s in rs_segs) if rs_segs else 0,
                "rs_oov_pct": (rs_oov_n / rs_oov_d * 100) if rs_oov_d else 0.0,
                "rs_cov": coverage(rs_reply, src),
                "rs_parsed": rs_parsed,
            }
        )

    print(f"# model={args.model} reasoning={args.reasoning} max_chars={args.max_chars}")
    print()
    print(
        f"{'bucket':9s} {'src':>5s} | "
        f"{'R segs':>6s} {'R max':>5s} {'R oov%':>6s} {'R cov':>5s} | "
        f"{'RS segs':>7s} {'RS max':>6s} {'RS oov%':>7s} {'RS cov':>6s} {'parsed':>7s}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r['bucket']:9s} {r['src_len']:>5d} | "
            f"{r['r_segs']:>6d} {r['r_max']:>5d} {r['r_oov_pct']:>5.1f}% {r['r_cov']:>4.0%} | "
            f"{r['rs_segs']:>7d} {r['rs_max']:>6d} {r['rs_oov_pct']:>6.1f}% {r['rs_cov']:>5.0%} "
            f"{'yes' if r['rs_parsed'] else 'NO':>7s}"
        )
    print("-" * 100)
    n = len(rows)
    print(
        f"{'avg':9s} {'':>5s} | "
        f"{sum(r['r_segs'] for r in rows) / n:>6.1f} {sum(r['r_max'] for r in rows) / n:>5.0f} "
        f"{sum(r['r_oov_pct'] for r in rows) / n:>5.1f}% "
        f"{sum(r['r_cov'] for r in rows) / n:>4.0%} | "
        f"{sum(r['rs_segs'] for r in rows) / n:>7.1f} "
        f"{sum(r['rs_max'] for r in rows) / n:>6.0f} "
        f"{sum(r['rs_oov_pct'] for r in rows) / n:>6.1f}% "
        f"{sum(r['rs_cov'] for r in rows) / n:>5.0%}"
    )


if __name__ == "__main__":
    asyncio.run(main())
