"""sim_gap diagnostic restricted to same-topic doc pairs.

The earlier diagnostic mixed two failure modes:
  - Type A: same-topic docs, different dates (recency disambiguates legitimately)
  - Type B: anaphora-anchor doc hijacks (rerank/match shouldn't have lifted it)

This script isolates Type A — pairs where two docs are clearly about
the same thing (high lexical overlap, excluding date tokens) and one
is the recency-correct gold. The sim_gap distribution here is the
ACTIONABLE one for recency tuning.

A pair is "same-topic" when shared content-words / max(|A|, |B|) >= 0.5,
where content-words excludes dates, stop words, and 1-2 char tokens.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._sim_gap_sametopic
"""
from __future__ import annotations

import asyncio
import re
import statistics

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import load_bench, make_cosine_rerank_fn

setup_env()

# Stop words + month names + a few generic temporal words to strip
STOP = frozenset("""
a an the and or but of for in on at to from with by as is was were are be been being
i my me we our us you your he she it they them their this that these those
""".split())
MONTHS = frozenset("""
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec
""".split())
GENERIC = frozenset("""
posted held ran did had wrote drafted made gave took went met conducted
update updates status meeting meetings review reviews
""".split())

_DATE_RE = re.compile(r"\b\d{1,2}\b|\b\d{4}\b|\b20\d\d\b")
_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def content_words(text: str) -> set[str]:
    text = _DATE_RE.sub(" ", text.lower())
    words = _WORD_RE.findall(text)
    return {w for w in words if w not in STOP and w not in MONTHS}


def jaccard_max(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / max(len(a), len(b))


async def run_arm(vd, queries) -> dict[str, list]:
    vd.recency_weight = 0.0
    vd.copeland_bonus = None
    out = {}
    for q in queries:
        r = await vd.query(q["text"], q["ref_time"], k=10)
        out[q["query_id"]] = r
    return out


async def analyze_bench(bench: str, embed_fn, rerank_fn) -> list[dict]:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return []
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    doc_text = {d.id: d.text for d in docs}
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    res = await run_arm(vd, queries)
    ref_us_all = vd._doc_ref_us
    rows = []
    for q in queries:
        qid = q["query_id"]
        golds = gold.get(qid, [])
        pool = res[qid]
        if not pool or not golds:
            continue
        pool_ids = [r.doc_id for r in pool]
        by_id = {r.doc_id: r for r in pool}
        gold_in_pool = [g for g in golds if g in by_id]
        if not gold_in_pool:
            continue
        # For each gold doc in pool, find same-topic same-pool docs
        for gid in gold_in_pool:
            gtext = doc_text[gid]
            gwords = content_words(gtext)
            if not gwords:
                continue
            gsim = by_id[gid].rerank + by_id[gid].match
            for did in pool_ids:
                if did == gid:
                    continue
                dtext = doc_text[did]
                dwords = content_words(dtext)
                j = jaccard_max(gwords, dwords)
                if j < 0.5:
                    continue
                dsim = by_id[did].rerank + by_id[did].match
                gold_more_recent = ref_us_all[gid] > ref_us_all[did]
                # Only meaningful when gold-side is the recency-correct doc
                # AND competitor has higher (or equal) sim
                rows.append({
                    "bench": bench,
                    "qid": qid,
                    "query": q["text"],
                    "gold_id": gid,
                    "compt_id": did,
                    "gold_text": gtext,
                    "compt_text": dtext,
                    "jaccard": j,
                    "sim_gap_compt_minus_gold": dsim - gsim,
                    "gold_more_recent": gold_more_recent,
                    "gold_sim": gsim,
                    "compt_sim": dsim,
                })
    return rows


def summarize(rows: list[dict], label: str) -> None:
    print(f"\n=== {label}: {len(rows)} same-topic pairs ===")
    if not rows:
        return
    # Where the COMPETITOR is sim-stronger AND gold is more-recent:
    # this is the actionable case for recency to overcome.
    actionable = [r for r in rows
                  if r["gold_more_recent"] and r["sim_gap_compt_minus_gold"] > 0]
    print(f"  Actionable (gold more recent, competitor stronger on sim): {len(actionable)}")
    if actionable:
        gaps = [r["sim_gap_compt_minus_gold"] for r in actionable]
        print(f"    sim_gap: min={min(gaps):.3f} med={statistics.median(gaps):.3f} "
              f"max={max(gaps):.3f} mean={statistics.mean(gaps):.3f}")
        bins = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0]
        for lo, hi in zip(bins[:-1], bins[1:]):
            c = sum(1 for g in gaps if lo <= g < hi)
            if c:
                print(f"    [{lo:.2f}, {hi:.2f}): {c}")
    # Print every actionable pair so we can eyeball
    print(f"\n  Actionable pairs (sorted by sim_gap):")
    actionable.sort(key=lambda r: r["sim_gap_compt_minus_gold"])
    for r in actionable:
        print(f"    qid={r['qid']} gap=+{r['sim_gap_compt_minus_gold']:.3f} "
              f"jaccard={r['jaccard']:.2f}")
        print(f"      Q: {r['query']}")
        print(f"      gold  ({r['gold_id']}): {r['gold_text'][:140]}")
        print(f"      compt ({r['compt_id']}): {r['compt_text'][:140]}")


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print("=== Same-topic sim_gap diagnostic ===", flush=True)
    print("Filtering to pairs with Jaccard(content-words) >= 0.5\n", flush=True)
    comp_rows = await analyze_bench("composition", embed_fn, rerank_fn)
    cot_rows = await analyze_bench("cotemporal", embed_fn, rerank_fn)
    summarize(comp_rows, "COMPOSITION")
    summarize(cot_rows, "COTEMPORAL")

    # Joint table: bonus needed
    actionable = [r for r in comp_rows + cot_rows
                  if r["gold_more_recent"] and r["sim_gap_compt_minus_gold"] > 0]
    if actionable:
        print(f"\n=== Combined actionable: {len(actionable)} pairs ===")
        gaps = sorted([r["sim_gap_compt_minus_gold"] for r in actionable])
        print(f"  Bonus needed to flip these pairs:")
        for B in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0]:
            flipped = sum(1 for g in gaps if g <= B)
            print(f"    B={B:.2f}: {flipped}/{len(gaps)} flipped")


if __name__ == "__main__":
    asyncio.run(main())
