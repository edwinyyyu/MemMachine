"""Three-way: R (rewrite + deterministic split), RS (LLM JSON), H (LLM marker).

Adds H: LLM emits compressed text with explicit <<<SPLIT>>> sentinels.
A trivial splitter cuts only on the sentinel — no markdown / newline
prioritization that could turn "4." into a tiny segment.
"""

from __future__ import annotations

import argparse
import asyncio
import os

import openai
from compare_R_vs_RS import coverage, faithfulness_oov
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_v2 import PROMPT_H, PROMPT_R, PROMPT_RS, parse_h, parse_rs
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


async def call(client, model, prompt, reasoning):
    kwargs: dict = {"model": model, "input": prompt}
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    return (resp.output_text or "").strip()


def post_split_oversize(segs: list[str], cap: int) -> list[str]:
    """Safety net: if LLM produced an oversized segment, deterministic-split it."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cap,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator="end",
    )
    out: list[str] = []
    for s in segs:
        if len(s) <= int(cap * 1.2):
            out.append(s)
        else:
            out.extend(splitter.split_text(s))
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
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
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator="end",
    )

    async def go(text: str):
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
            h = await call(
                client,
                args.model,
                PROMPT_H.format(passage=text, max_chars=args.max_chars),
                args.reasoning,
            )
        return r, rs, h

    rep = await asyncio.gather(*(go(t) for _, t in samples))
    await client.close()

    rows = []
    for (bucket, src), (r_reply, rs_reply, h_reply) in zip(samples, rep, strict=False):
        r_segs = splitter.split_text(r_reply)
        rs_segs = parse_rs(rs_reply) or [rs_reply]
        rs_parsed = parse_rs(rs_reply) is not None
        h_segs = parse_h(h_reply)
        # Apply post-hoc safety net for over-cap segments
        rs_segs_safe = post_split_oversize(rs_segs, args.max_chars)
        h_segs_safe = post_split_oversize(h_segs, args.max_chars)

        def stats(segs: list[str]):
            if not segs:
                return 0, 0, 0.0, 0.0
            n_oov, n_total = 0, 0
            for s in segs:
                a, b = faithfulness_oov(s, src)
                n_oov += a
                n_total += b
            return (
                len(segs),
                max(len(s) for s in segs),
                (n_oov / n_total * 100) if n_total else 0.0,
                coverage("\n".join(segs), src),
            )

        rN, rMax, rOov, rCov = stats(r_segs)
        rsN, rsMax, rsOov, rsCov = stats(rs_segs_safe)
        hN, hMax, hOov, hCov = stats(h_segs_safe)

        rows.append(
            {
                "bucket": bucket,
                "src_len": len(src),
                "rN": rN,
                "rMax": rMax,
                "rOov": rOov,
                "rCov": rCov,
                "rsN": rsN,
                "rsMax": rsMax,
                "rsOov": rsOov,
                "rsCov": rsCov,
                "rs_parsed": rs_parsed,
                "hN": hN,
                "hMax": hMax,
                "hOov": hOov,
                "hCov": hCov,
            }
        )

    print(f"# model={args.model} reasoning={args.reasoning} max_chars={args.max_chars}")
    print()
    hdr = (
        f"{'bucket':9s} {'src':>5s} | "
        f"{'R N':>4s} {'R max':>5s} {'R oov':>5s} {'R cov':>5s} | "
        f"{'RSN':>4s} {'RSmax':>5s} {'RSoov':>5s} {'RScov':>5s} {'P':>2s} | "
        f"{'HN':>3s} {'Hmax':>4s} {'Hoov':>4s} {'Hcov':>4s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['bucket']:9s} {r['src_len']:>5d} | "
            f"{r['rN']:>4d} {r['rMax']:>5d} {r['rOov']:>4.1f}% {r['rCov']:>4.0%} | "
            f"{r['rsN']:>4d} {r['rsMax']:>5d} {r['rsOov']:>4.1f}% {r['rsCov']:>4.0%} "
            f"{'Y' if r['rs_parsed'] else 'N':>2s} | "
            f"{r['hN']:>3d} {r['hMax']:>4d} {r['hOov']:>3.1f}% {r['hCov']:>3.0%}"
        )
    print("-" * len(hdr))
    n = len(rows)
    print(
        f"{'avg':9s} {'':>5s} | "
        f"{sum(r['rN'] for r in rows) / n:>4.1f} "
        f"{sum(r['rMax'] for r in rows) / n:>5.0f} "
        f"{sum(r['rOov'] for r in rows) / n:>4.1f}% "
        f"{sum(r['rCov'] for r in rows) / n:>4.0%} | "
        f"{sum(r['rsN'] for r in rows) / n:>4.1f} "
        f"{sum(r['rsMax'] for r in rows) / n:>5.0f} "
        f"{sum(r['rsOov'] for r in rows) / n:>4.1f}% "
        f"{sum(r['rsCov'] for r in rows) / n:>4.0%}    | "
        f"{sum(r['hN'] for r in rows) / n:>3.1f} "
        f"{sum(r['hMax'] for r in rows) / n:>4.0f} "
        f"{sum(r['hOov'] for r in rows) / n:>3.1f}% "
        f"{sum(r['hCov'] for r in rows) / n:>3.0%}"
    )


if __name__ == "__main__":
    asyncio.run(main())
