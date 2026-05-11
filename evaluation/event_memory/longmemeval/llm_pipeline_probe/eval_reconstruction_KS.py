"""How well does Σ(segments) reconstruct the original passage?

Three reconstruction metrics, plus an eyeball pass:
  - dup_rate:    fraction of unit-uses that are repeats. Caused by the
                 LLM re-including a referent unit in multiple segments
                 for standalone-ness. Some duplication is healthy.
  - skip_rate:   fraction of source units that no segment used. These
                 are either scaffolding (good) or true content drops (bad).
  - order_breaks: number of unit-id descents across the full ordered
                 unit-id sequence (concat all segments in segment order,
                 then list their unit ids). 0 = perfectly monotonic; >0
                 means the LLM produced segments in a non-natural order
                 OR re-used a unit from earlier in a later segment.
                 Repeats (re-includes) ARE counted here — they go to a
                 lower id from a higher one.

Plus we print the side-by-side: original passage | concatenated segments
(with separators) so the human can read both and judge coherence.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os

import openai
from dotenv import load_dotenv
from probe_segmenter_KS import PROMPT_KS, SEGMENT_SCHEMA, annotate, split_units
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


async def call(client, model, prompt, reasoning):
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
    resp = await client.responses.create(**kwargs)
    import json

    return json.loads(resp.output_text or "{}")


def reconstruction_stats(units: list[str], segs_indices: list[list[int]]) -> dict:
    flat = [i for seg in segs_indices for i in seg]
    used = set(flat)
    n = len(units)
    skipped = sorted(set(range(1, n + 1)) - used)
    dup_uses = len(flat) - len(used)
    order_breaks = sum(1 for a, b in itertools.pairwise(flat) if b <= a)
    return {
        "n_units": n,
        "n_unique_used": len(used),
        "n_skipped": len(skipped),
        "skipped_indices": skipped,
        "dup_uses": dup_uses,
        "dup_rate": dup_uses / max(len(flat), 1),
        "skip_rate": len(skipped) / max(n, 1),
        "order_breaks": order_breaks,
        "concat_order": flat,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--per-bucket", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=600)
    parser.add_argument(
        "--show", type=int, default=2, help="Number of full side-by-side prints."
    )
    args = parser.parse_args()

    bins = collect()
    samples = []
    for b, ts in bins.items():
        for t in [x for x in ts if len(x) >= args.min_length][: args.per_bucket]:
            samples.append((b, t))

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    summaries = []
    for bucket, src in samples:
        units = split_units(src)
        annotated = annotate(units)
        prompt = PROMPT_KS.format(passage=annotated, max_chars=args.max_chars)
        parsed = await call(client, args.model, prompt, args.reasoning)
        segs_idx = [s.get("include", []) for s in parsed.get("segments", [])]
        seg_texts = [
            "\n\n".join(units[i - 1] for i in idx if 1 <= i <= len(units))
            if any("\n" in units[i - 1] for i in idx if 1 <= i <= len(units))
            else " ".join(units[i - 1] for i in idx if 1 <= i <= len(units))
            for idx in segs_idx
        ]
        stats = reconstruction_stats(units, segs_idx)
        summaries.append((bucket, src, stats, seg_texts, units))

    await client.close()

    # Aggregate stats
    print(f"# model={args.model} reasoning={args.reasoning} max_chars={args.max_chars}")
    print()
    print(
        f"{'bucket':9s} {'units':>5s} {'used':>4s} {'skip':>4s} "
        f"{'dups':>4s} {'order_brk':>9s} {'segs':>4s}"
    )
    for bucket, _, st, segs, _ in summaries:
        print(
            f"{bucket:9s} {st['n_units']:>5d} {st['n_unique_used']:>4d} "
            f"{st['n_skipped']:>4d} {st['dup_uses']:>4d} {st['order_breaks']:>9d} "
            f"{len(segs):>4d}"
        )

    n = len(summaries)
    avg_dup = sum(st["dup_rate"] for _, _, st, _, _ in summaries) / n
    avg_skip = sum(st["skip_rate"] for _, _, st, _, _ in summaries) / n
    avg_breaks = sum(st["order_breaks"] for _, _, st, _, _ in summaries) / n
    print(
        f"AVG dup_rate={avg_dup:.0%}  skip_rate={avg_skip:.0%}  "
        f"order_breaks/chunk={avg_breaks:.1f}"
    )

    # Show full side-by-side for the first N
    for bucket, src, stats, seg_texts, units in summaries[: args.show]:
        print()
        print("=" * 80)
        print(
            f"BUCKET=[{bucket}] units={stats['n_units']} used={stats['n_unique_used']} "
            f"skipped={stats['n_skipped']} dups={stats['dup_uses']} "
            f"order_breaks={stats['order_breaks']}"
        )
        print(f"concat_unit_ids={stats['concat_order']}")
        print(f"skipped_unit_ids={stats['skipped_indices']}")
        print()
        print("--- ORIGINAL ---")
        print(src)
        print()
        print("--- CONCATENATED SEGMENTS ---")
        for i, st in enumerate(seg_texts):
            print(f"[seg {i}]")
            print(st)
            print()
        print("--- SKIPPED UNITS ---")
        for idx in stats["skipped_indices"]:
            print(f"  [{idx}] {units[idx - 1]}")


if __name__ == "__main__":
    asyncio.run(main())
