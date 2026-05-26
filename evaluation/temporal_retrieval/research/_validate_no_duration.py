"""A/B compare PASS1 with-duration vs without-duration on NON-saturated
benches only.

Background: `flatten_intervals` in `core.py` has no branch for
`kind=="duration"`, so duration extractions never produce intervals and
never affect retrieval. Removing duration from the PASS1 emission set is
a no-op for the retrieval layer in principle. This script tests whether
that's true in practice — does removing duration emission change R@5 on
benches where prompts can move the score?

Saturated benches (R@5=1.0 across all variants in prior ablation) excluded:
- era_refs, conjunctive_temporal, multi_te_doc, mixed_cue.

Tested benches (all have non-trivial baseline R@5 < 1.0 in prior runs):
- composition (R@5 ~0.640)
- adversarial (R@5 ~0.843-0.886)
- realq_v2 (R@5 ~0.941)
- temporal_essential (held-out, ~0.92 historically)
- hard_bench (held-out, ~0.89 historically)

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_no_duration
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval import extractor as ext_mod

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

# Reconstruct the prior prompt (with duration emission) for the A side.
PASS1_WITH_DURATION = ext_mod.PASS1_SYSTEM.replace(
    """- Recurring schedules describe a pattern, not a moment: "every Thursday at
  3pm", "monthly", "each year on Mom's birthday".

# What does NOT count (skip)

The unifying principle: skip phrases that name time without pinning or
bounding a specific calendar location.

- Unanchored durations describe a length, not a time: "for 3 weeks", "two
  hours long". They tell you HOW LONG, not when. EXCEPTION: a duration
  attached to an anchor IS a calendar location — "for 3 weeks starting
  Jan 1" should emit the anchored span, not the bare duration.""",
    """- Recurring schedules describe a pattern, not a moment: "every Thursday at
  3pm", "monthly", "each year on Mom's birthday".
- Durations bound a length even without an anchor: "for 3 weeks", "two
  hours long".

# What does NOT count (skip)

The unifying principle: skip phrases that name time without pinning or
bounding a specific calendar location.""",
).replace(
    """- kind_guess: one of [instant, interval, recurrence].
  - instant: a pinpointed time (even if fuzzy): "yesterday", "2015", "last month".
  - interval: an explicit start-to-end range: "from X to Y", "the first week of April".
  - recurrence: a recurring pattern: "every Thursday".""",
    """- kind_guess: one of [instant, interval, duration, recurrence].
  - instant: a pinpointed time (even if fuzzy): "yesterday", "2015", "last month".
  - interval: an explicit start-to-end range: "from X to Y", "the first week of April".
  - duration: an unanchored length: "for 3 weeks", "two hours long".
  - recurrence: a recurring pattern: "every Thursday".""",
)

assert PASS1_WITH_DURATION != ext_mod.PASS1_SYSTEM, (
    "Replace did not change anything — prompt content drifted; reconstruction logic needs update"
)


# Non-saturated benches (R@5 < 1.0 in prior runs).
NONSAT_BENCHES = {
    "composition": (
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
    ),
    "adversarial": (
        "adversarial_docs.jsonl",
        "adversarial_queries.jsonl",
        "adversarial_gold.jsonl",
    ),
    "realq_v2": (
        "realq_v2_docs.jsonl",
        "realq_v2_queries.jsonl",
        "realq_v2_gold.jsonl",
    ),
    "temporal_essential": (
        "temporal_essential_docs.jsonl",
        "temporal_essential_queries.jsonl",
        "temporal_essential_gold.jsonl",
    ),
    "hard_bench": (
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
    ),
}

INPUT_COST_PER_M = 0.25
OUTPUT_COST_PER_M = 2.00


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * INPUT_COST_PER_M + (
        output_tokens / 1e6
    ) * OUTPUT_COST_PER_M


async def evaluate(bench: str, prompt: str, embed_fn, rerank_fn, label: str) -> dict:
    docs_file, queries_file, gold_file = NONSAT_BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(
        docs_file, queries_file, gold_file
    )
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    extractor = ext_mod.TemporalExtractor(
        cache_subdir=f"nodur_{label}_{bench}", pass1_system=prompt
    )
    retriever = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn, extractor=extractor
    )
    await retriever.index(docs)

    n_eval = 0
    n_r5 = 0
    n_r1 = 0
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        first_gold = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
        if first_gold is not None:
            if first_gold <= 1:
                n_r1 += 1
            if first_gold <= 5:
                n_r5 += 1

    stats = retriever.stats()
    in_t = stats["extractor_usage"]["input"]
    out_t = stats["extractor_usage"]["output"]
    intervals = retriever.doc_intervals()
    total_intervals = sum(len(v) for v in intervals.values())
    return {
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "n_eval": n_eval,
        "input_tokens": in_t,
        "output_tokens": out_t,
        "cost_usd": cost_usd(in_t, out_t),
        "intervals_per_doc": total_intervals / max(1, len(docs)),
    }


async def main() -> None:
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    bench_order = list(NONSAT_BENCHES.keys())
    rows: list[dict] = []
    for bench in bench_order:
        print(f"\n=== bench: {bench} ===", flush=True)
        for label, prompt in (
            ("with_dur", PASS1_WITH_DURATION),
            ("no_dur", ext_mod.PASS1_SYSTEM),
        ):
            print(f"  {label}...", flush=True)
            r = await evaluate(bench, prompt, embed_fn, rerank_fn, label)
            print(
                f"    R@5={r['R@5']:.3f} R@1={r['R@1']:.3f} "
                f"cost=${r['cost_usd']:.4f} "
                f"in={r['input_tokens']} out={r['output_tokens']} "
                f"iv/doc={r['intervals_per_doc']:.2f} "
                f"n_eval={r['n_eval']}",
                flush=True,
            )
            rows.append({"bench": bench, "label": label, **r})

    print("\n" + "=" * 100, flush=True)
    print(
        f"{'bench':22s} {'label':10s} {'R@5':>6s} {'R@1':>6s} "
        f"{'cost_usd':>10s} {'iv/doc':>7s} {'verdict':>14s}",
        flush=True,
    )
    print("-" * 100, flush=True)
    grand_with = 0.0
    grand_without = 0.0
    for bench in bench_order:
        with_dur = next(
            r for r in rows if r["bench"] == bench and r["label"] == "with_dur"
        )
        no_dur = next(r for r in rows if r["bench"] == bench and r["label"] == "no_dur")
        grand_with += with_dur["cost_usd"]
        grand_without += no_dur["cost_usd"]
        for label, r in (("with_dur", with_dur), ("no_dur", no_dur)):
            verdict = ""
            if label == "no_dur":
                d = r["R@5"] - with_dur["R@5"]
                if d <= -0.005:
                    verdict = f"REGRESS({d:+.3f})"
                elif d >= 0.005:
                    verdict = f"BEAT({d:+.3f})"
                else:
                    verdict = "match"
            print(
                f"{bench:22s} {label:10s} {r['R@5']:>6.3f} {r['R@1']:>6.3f} "
                f"${r['cost_usd']:>9.4f} {r['intervals_per_doc']:>7.2f} {verdict:>14s}",
                flush=True,
            )

    print("-" * 100, flush=True)
    delta = grand_with - grand_without
    pct = delta / grand_with * 100 if grand_with else 0
    print(
        f"GRAND TOTAL  with_dur ${grand_with:.4f} | "
        f"no_dur ${grand_without:.4f} | "
        f"DELTA ${delta:+.4f} ({pct:+.1f}%)",
        flush=True,
    )

    out_path = ROOT / "no_duration_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
