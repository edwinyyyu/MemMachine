"""Behavioral characterization of probe_deriver_v2_single_call on real corpus.

Samples 30 segments (50-500 chars) from LongMemEval, runs the deriver,
and answers Q1-Q5 from the research brief.

Run:
    uv run python eval_deriver_real_corpus.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Any

import openai
from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")

sys.path.insert(
    0, "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval"
)
from longmemeval_models import load_longmemeval_dataset  # noqa: E402
from probe_deriver_v2_single_call import derive  # noqa: E402

LME = (
    "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/data/longmemeval_s_cleaned.json"
)

SEED = 42
N_SAMPLE = 30


# --------------------------------------------------------------------------
# SAMPLING
# --------------------------------------------------------------------------


def collect_turns() -> list[tuple[str, str]]:
    """Return list of (role, content) for all turns in the corpus."""
    qs = load_longmemeval_dataset(LME)
    seen: set[tuple[str, str]] = set()
    for q in qs:
        for sid in q.session_id_map:
            for turn in q.get_session(sid):
                role = getattr(turn, "role", "?") or "?"
                content = (turn.content or "").strip()
                if content:
                    seen.add((role, content))
    return list(seen)


def categorize(text: str) -> str:
    """Lightweight content-type bucketing for diversification."""
    if "|" in text and text.count("|") >= 4 and "---" in text:
        return "table"
    if (
        "```" in text
        or re.search(r"\bdef \w+\(", text)
        or re.search(r"\bfunction \w+\(", text)
    ):
        return "code"
    if re.search(r"\[\d{4}-\d{2}-\d{2}", text):
        return "log"
    if len(re.findall(r"\$?\d[\d,]*(?:\.\d+)?", text)) >= 4:
        return "numeric"
    if re.search(r"^\s*[-*]\s", text, re.MULTILINE) or re.search(
        r"^\s*\d+\.", text, re.MULTILINE
    ):
        return "list"
    return "prose"


def sample_segments(seed: int = SEED, n: int = N_SAMPLE) -> list[dict[str, Any]]:
    """Sample N segments diversified across length buckets and content types.

    Length buckets: short (50-149), medium (150-299), long (300-500).
    Aim for diversity across role (user/assistant) and content category.
    """
    rng = random.Random(seed)
    turns = collect_turns()

    # restrict to 50-500 char range
    eligible = [(r, c) for r, c in turns if 50 <= len(c) <= 500]
    rng.shuffle(eligible)

    short, medium, long_ = [], [], []
    for r, c in eligible:
        L = len(c)
        if L < 150:
            short.append((r, c))
        elif L < 300:
            medium.append((r, c))
        else:
            long_.append((r, c))

    # Aim 10/10/10 per bucket; diversify by category within each bucket
    def pick_diverse(pool: list[tuple[str, str]], k: int) -> list[tuple[str, str]]:
        # group by (role, category)
        by_key: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for r, c in pool:
            key = (r, categorize(c))
            by_key.setdefault(key, []).append((r, c))
        keys = list(by_key.keys())
        rng.shuffle(keys)
        out: list[tuple[str, str]] = []
        # round-robin over keys
        idx = 0
        used: set[int] = set()
        while len(out) < k and len(used) < len(pool):
            key = keys[idx % len(keys)]
            bucket = by_key[key]
            if bucket:
                pick = bucket.pop(0)
                out.append(pick)
                used.add(id(pick))
            idx += 1
            if idx > 10000:
                break
        # if still short, fill from remaining pool
        if len(out) < k:
            remaining = [x for x in pool if x not in out]
            rng.shuffle(remaining)
            out.extend(remaining[: k - len(out)])
        return out[:k]

    picks = pick_diverse(short, 10) + pick_diverse(medium, 10) + pick_diverse(long_, 10)

    return [
        {
            "idx": i,
            "role": r,
            "len": len(c),
            "category": categorize(c),
            "content": c,
        }
        for i, (r, c) in enumerate(picks)
    ]


# --------------------------------------------------------------------------
# ANALYSIS HELPERS
# --------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(s: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(s) if len(t) >= 4}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Formatting-leakage patterns
LEAK_PATTERNS = {
    "markdown_bold": re.compile(r"\*\*"),
    "pipe_table": re.compile(r"\s\|\s"),
    "code_fence": re.compile(r"```"),
    "bracketed_ts": re.compile(r"\[\d{4}-\d{2}-\d{2}"),
    "inline_code": re.compile(r"`[^`]+`"),
    "html_tag": re.compile(r"<[A-Za-z]+\b[^>]*>"),
}


def scan_leak(s: str) -> list[str]:
    return [name for name, p in LEAK_PATTERNS.items() if p.search(s)]


def is_verbatim_copy(seg: str, deriv: str, tol: float = 0.95) -> bool:
    """Roughly: does the derivative reproduce the segment whole-cloth?"""
    seg_norm = re.sub(r"\s+", " ", seg.strip().lower())
    der_norm = re.sub(r"\s+", " ", deriv.strip().lower())
    if seg_norm == der_norm:
        return True
    # if derivative contains 95%+ of segment tokens AND length comparable
    seg_tokens = tokenize(seg)
    der_tokens = tokenize(deriv)
    if not seg_tokens:
        return False
    coverage = len(seg_tokens & der_tokens) / len(seg_tokens)
    length_ratio = len(deriv) / max(1, len(seg))
    return coverage >= tol and 0.6 <= length_ratio <= 1.5


def looks_encoded(seg: str) -> bool:
    """Does segment contain encoded/formatted content (timestamps, code, fences)?"""
    if "```" in seg:
        return True
    if re.search(r"\[\d{4}-\d{2}-\d{2}", seg):
        return True
    if re.search(r"\bdef \w+\(", seg):
        return True
    if "|" in seg and seg.count("|") >= 4 and "---" in seg:
        return True
    return bool(re.search(r"^\s*[-*]\s", seg, re.MULTILINE))


def looks_artifact_description(d: str) -> bool:
    """Heuristic for an 'artifact-description' derivative: starts with a noun
    phrase like 'A code block...', 'A table...', 'A log line...', 'A markdown
    table...', 'Python function...', etc."""
    return bool(
        re.match(
            r"^(A |An |The )?(python |javascript |markdown |sql |code |error |log |server |benchmark |comparison )?(table|code block|code snippet|function|log line|list|markdown table|csv|json|message|note)\b",
            d.strip(),
            re.IGNORECASE,
        )
    )


def bucket_p4_branch(seg: str, derivs: list[str]) -> str:
    """Bucket each segment into one of:
    - focused-emit-verbatim (single deriv, ~equal to segment)
    - decompose (N>=2 derivs, none full verbatim copy of segment)
    - decode (segment was encoded/structured AND deriv normalizes it
              — i.e. NOT a full verbatim copy and represents the content
              in plain prose)
    - describe-artifact (segment is non-prose AND a deriv looks like
                         an artifact description)
    - mixed/other (fallback)
    """
    encoded = looks_encoded(seg)
    has_artifact_desc = any(looks_artifact_description(d) for d in derivs)
    n = len(derivs)
    any_verbatim_full = any(is_verbatim_copy(seg, d) for d in derivs)

    if encoded:
        # Prefer describe-artifact if there's an explicit artifact-name derivative
        if has_artifact_desc:
            return "describe-artifact"
        return "decode"

    # not encoded
    if n == 1 and any_verbatim_full:
        return "focused-emit-verbatim"
    if n == 1 and not any_verbatim_full:
        # paraphrase of single fact — count as focused (P3 says "stays close")
        return "focused-emit-verbatim"
    if n >= 2 and not any_verbatim_full:
        return "decompose"
    if n >= 2 and any_verbatim_full:
        return "decompose"  # mixed; note in detail
    if n == 0:
        return "empty"
    return "other"


# --------------------------------------------------------------------------
# RUNNER
# --------------------------------------------------------------------------


async def run_all(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(8)

    async def one(s: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            try:
                derivs = await derive(client, s["content"])
            except Exception as e:
                derivs = []
                s["error"] = str(e)
            s["derivatives"] = derivs
            return s

    results = await asyncio.gather(*(one(s) for s in samples))
    await client.close()
    return list(results)


def analyze(results: list[dict[str, Any]]) -> dict[str, Any]:
    # Q1 — bucketing
    buckets = Counter()
    bucket_per_idx: dict[int, str] = {}
    for r in results:
        b = bucket_p4_branch(r["content"], r["derivatives"])
        buckets[b] += 1
        bucket_per_idx[r["idx"]] = b

    # Q2 — redundancy among multi-derivative outputs
    redundancy_pairs = 0
    redundancy_segments = 0
    multi_segments = 0
    worst_pairs: list[tuple[int, str, str, float]] = []
    for r in results:
        ds = r["derivatives"]
        if len(ds) < 2:
            continue
        multi_segments += 1
        flagged = False
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                ti, tj = tokenize(ds[i]), tokenize(ds[j])
                jj = jaccard(ti, tj)
                if jj > 0.7:
                    redundancy_pairs += 1
                    flagged = True
                    worst_pairs.append((r["idx"], ds[i], ds[j], jj))
        if flagged:
            redundancy_segments += 1
    worst_pairs.sort(key=lambda x: -x[3])

    # Q3 — formatting leakage
    leak_counts = Counter()
    leak_examples: list[tuple[int, str, list[str]]] = []
    for r in results:
        for d in r["derivatives"]:
            leaks = scan_leak(d)
            for nm in leaks:
                leak_counts[nm] += 1
            if leaks:
                leak_examples.append((r["idx"], d, leaks))

    # Q5 — token cost
    n_segs = len(results)
    total_segs_chars = sum(len(r["content"]) for r in results)
    total_derivs = sum(len(r["derivatives"]) for r in results)
    total_derivs_chars = sum(sum(len(d) for d in r["derivatives"]) for r in results)
    avg_derivs_per_seg = total_derivs / max(1, n_segs)
    char_ratio = total_derivs_chars / max(1, total_segs_chars)

    return {
        "buckets": buckets,
        "bucket_per_idx": bucket_per_idx,
        "redundancy_pairs": redundancy_pairs,
        "redundancy_segments": redundancy_segments,
        "multi_segments": multi_segments,
        "worst_pairs": worst_pairs[:5],
        "leak_counts": leak_counts,
        "leak_examples": leak_examples[:10],
        "n_segs": n_segs,
        "avg_derivs_per_seg": avg_derivs_per_seg,
        "char_ratio": char_ratio,
        "total_segs_chars": total_segs_chars,
        "total_derivs_chars": total_derivs_chars,
        "total_derivs": total_derivs,
    }


# --------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------


def short(s: str, n: int = 200) -> str:
    s = s.replace("\n", " \\n ")
    return s if len(s) <= n else s[: n - 1] + "…"


def report(results: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("SAMPLE INVENTORY")
    print("=" * 70)
    cat_count = Counter(r["category"] for r in results)
    role_count = Counter(r["role"] for r in results)
    lens = [r["len"] for r in results]
    print(
        f"  N={len(results)}  lengths min/med/max = {min(lens)}/{sorted(lens)[len(lens) // 2]}/{max(lens)}"
    )
    print(f"  categories: {dict(cat_count)}")
    print(f"  roles: {dict(role_count)}")
    print()
    for r in results:
        print(
            f"  [{r['idx']:2d}] role={r['role']:9s} cat={r['category']:7s} len={r['len']:3d}  {short(r['content'], 110)}"
        )

    print("\n" + "=" * 70)
    print("Q1 — CASE DISTRIBUTION")
    print("=" * 70)
    for b, c in stats["buckets"].most_common():
        print(f"  {b:30s} {c:3d}")
    print("\n  per-segment bucket:")
    for r in results:
        print(
            f"   [{r['idx']:2d}] {stats['bucket_per_idx'][r['idx']]:25s} "
            f"n_deriv={len(r['derivatives'])} cat={r['category']:7s}"
        )

    print("\n" + "=" * 70)
    print("Q2 — REDUNDANCY (Jaccard > 0.7 on tokens len>=4)")
    print("=" * 70)
    print(f"  multi-derivative segments: {stats['multi_segments']}")
    print(f"  segments w/ near-dup pair: {stats['redundancy_segments']}")
    print(f"  total flagged pairs: {stats['redundancy_pairs']}")
    if stats["multi_segments"]:
        rate = stats["redundancy_segments"] / stats["multi_segments"]
        print(f"  rate (segments-with-dup / multi-segments): {rate:.2%}")
    print("  worst pairs:")
    for idx, a, b, j in stats["worst_pairs"]:
        print(f"   [{idx}] J={j:.2f}")
        print(f"     A: {short(a, 150)}")
        print(f"     B: {short(b, 150)}")

    print("\n" + "=" * 70)
    print("Q3 — FORMATTING LEAKAGE")
    print("=" * 70)
    if not stats["leak_counts"]:
        print("  (none detected)")
    for nm, c in stats["leak_counts"].most_common():
        print(f"  {nm:20s} {c:3d}")
    print("  examples:")
    for idx, d, leaks in stats["leak_examples"]:
        print(f"   [{idx}] leaks={leaks}")
        print(f"     {short(d, 200)}")

    print("\n" + "=" * 70)
    print("Q4 — SIDE-BY-SIDE (eyeball)")
    print("=" * 70)
    print("  (printing all 30 for selection)")
    for r in results:
        print()
        print(
            f"  --- [{r['idx']:2d}] cat={r['category']} role={r['role']} len={r['len']} bucket={stats['bucket_per_idx'][r['idx']]}"
        )
        print(f"  SEG: {short(r['content'], 300)}")
        for i, d in enumerate(r["derivatives"]):
            print(f"   D{i}: {short(d, 300)}")

    print("\n" + "=" * 70)
    print("Q5 — COST")
    print("=" * 70)
    print(f"  segments: {stats['n_segs']}")
    print(f"  total derivatives: {stats['total_derivs']}")
    print(f"  avg derivatives / segment: {stats['avg_derivs_per_seg']:.2f}")
    print(f"  total segment chars: {stats['total_segs_chars']}")
    print(f"  total derivative chars: {stats['total_derivs_chars']}")
    print(f"  char ratio (deriv / seg): {stats['char_ratio']:.2f}x")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------


async def main() -> None:
    print("# DERIVER REAL-CORPUS BEHAVIORAL CHARACTERIZATION")
    print(f"# seed={SEED} n={N_SAMPLE} model=gpt-5.4-nano reasoning=low")

    samples = sample_segments(seed=SEED, n=N_SAMPLE)
    print(f"\nSampled {len(samples)} segments.")

    results = await run_all(samples)
    stats = analyze(results)
    report(results, stats)

    # also dump JSON for downstream poking
    out_path = "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval/llm_pipeline_probe/eval_deriver_real_corpus.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "samples": results,
                "stats": {
                    k: (dict(v) if isinstance(v, Counter) else v)
                    for k, v in stats.items()
                    if k not in ("worst_pairs", "leak_examples", "bucket_per_idx")
                },
                "bucket_per_idx": {
                    str(k): v for k, v in stats["bucket_per_idx"].items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nWrote JSON dump to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
