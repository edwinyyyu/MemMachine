"""Run the classifier on a stratified sample of the LongMemEval corpus.

Reports: % rejected at each length bucket, plus example REJECTs at the
long-text end (sanity: should be near zero) and example KEEPs at the
short-text end (sanity: should be specific entities/facts).
"""

from __future__ import annotations

import asyncio
import random
import sys

from dotenv import load_dotenv

sys.path.insert(
    0, "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval"
)
from classifier import classify_many
from longmemeval_models import load_longmemeval_dataset

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


def collect_unique_messages() -> list[str]:
    qs = load_longmemeval_dataset(
        "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/data/longmemeval_s_cleaned.json"
    )
    seen: set[str] = set()
    for q in qs:
        for sid in q.session_id_map:
            for turn in q.get_session(sid):
                t = turn.content.strip()
                if t:
                    seen.add(t)
    return sorted(seen)


def stratified_sample(
    texts: list[str], n_per_bucket: int = 60, seed: int = 0
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    buckets = {
        "tiny (≤15)": [t for t in texts if 0 < len(t) <= 15],
        "short (16-40)": [t for t in texts if 16 <= len(t) <= 40],
        "medium (41-120)": [t for t in texts if 41 <= len(t) <= 120],
        "long (121-500)": [t for t in texts if 121 <= len(t) <= 500],
        "xlong (>500)": [t for t in texts if len(t) > 500],
    }
    return {k: rng.sample(v, min(n_per_bucket, len(v))) for k, v in buckets.items()}


def main() -> None:
    texts = collect_unique_messages()
    print(f"Unique stripped messages in corpus: {len(texts):,}")
    sample = stratified_sample(texts, n_per_bucket=80)
    flat = [(b, t) for b, ts in sample.items() for t in ts]

    results = asyncio.run(
        classify_many(
            [t for _, t in flat],
            model="gpt-5-mini",
            prompt="v1",
            reasoning_effort="low",
            concurrency=24,
        )
    )

    by_bucket: dict[str, list] = {b: [] for b in sample}
    for (b, _), r in zip(flat, results, strict=False):
        by_bucket[b].append(r)

    print()
    print(f"{'bucket':18s}  {'n':>4s}  {'reject%':>8s}")
    for b, rs in by_bucket.items():
        n = len(rs)
        n_rej = sum(1 for r in rs if r.label == "REJECT")
        print(f"{b:18s}  {n:>4d}  {n_rej / max(n, 1):>7.1%}")

    print()
    print(
        "Sample LONG-text rejects (should be ~empty — false-rejects on long text are bad):"
    )
    long_rej = [
        r
        for b, rs in by_bucket.items()
        for r in rs
        if b in ("medium (41-120)", "long (121-500)", "xlong (>500)")
        and r.label == "REJECT"
    ]
    for r in long_rej[:5]:
        print(f"  REJECT [{len(r.text)}]: {r.text[:160]!r}")
    if not long_rej:
        print("  (none)")

    print()
    print("Sample TINY-text keeps (should be specific entities/facts):")
    tiny_keep = [r for r in by_bucket["tiny (≤15)"] if r.label == "KEEP"]
    for r in tiny_keep[:25]:
        print(f"  KEEP  [{len(r.text)}]: {r.text!r}")

    print()
    print("Sample TINY-text rejects (these are the wins):")
    tiny_rej = [r for r in by_bucket["tiny (≤15)"] if r.label == "REJECT"]
    for r in tiny_rej[:25]:
        print(f"  REJECT[{len(r.text)}]: {r.text!r}")


if __name__ == "__main__":
    main()
