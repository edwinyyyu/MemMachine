"""Per-query drill on the composition bench for V7-Direct.

Prints, for each of the 25 queries: comp_type, query text, planner refs
+ extremum, gold doc(s), top-3 retrieved, HIT/MISS. Groups failures by
comp_type so we can see whether A/B/E (non-anaphoric) are failing too,
not just C/D (anaphora-bound).

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._composition_drill
"""
from __future__ import annotations

import asyncio
from collections import Counter

from temporal_retrieval_v7 import Doc as DocV7
from temporal_retrieval_v7 import NEG_INF, POS_INF, TemporalRetrieverV7Direct

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_v7.research._full_ab import _load_bench, make_cosine_rerank_fn

setup_env()


def _fmt_us(t: int) -> str:
    if t <= NEG_INF + 1:
        return "-inf"
    if t >= POS_INF - 1:
        return "+inf"
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t / 1_000_000, tz=timezone.utc).strftime("%Y-%m-%d")


def _fmt_refs(refs) -> str:
    if not refs:
        return "[]"
    parts = []
    for r in refs:
        ivs = ",".join(f"[{_fmt_us(iv.earliest_us)},{_fmt_us(iv.latest_us)})"
                       for iv in r.intervals)
        parts.append("{" + ivs + "}")
    return " ".join(parts)


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    docs_jsonl, queries, gold = _load_bench("composition")
    docs = [DocV7(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    doc_text = {d["doc_id"]: d["text"] for d in docs_jsonl}

    vd = TemporalRetrieverV7Direct(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)

    fails_by_type: Counter = Counter()
    total_by_type: Counter = Counter()
    for q in queries:
        ctype = q.get("comp_type", "?")
        total_by_type[ctype] += 1
        plan = await vd._planner.plan(q["text"], q["ref_time"])  # type: ignore[attr-defined]
        results = await vd.query(q["text"], q["ref_time"], k=10)
        ranked = [r.doc_id for r in results]
        gold_ids = gold.get(q["query_id"], [])
        hit = bool(ranked[:1]) and ranked[0] in gold_ids
        if not hit:
            fails_by_type[ctype] += 1
        mark = "HIT " if hit else "MISS"
        refs_s = "ERR:" + (plan.parse_error or "") if plan.parse_error \
            else _fmt_refs(plan.refs)
        print(f"\n[{mark}] {ctype} {q['query_id']}: {q['text']}")
        print(f"   plan: refs={refs_s} extremum={plan.extremum}")
        for gid in gold_ids:
            print(f"   gold: {gid} = {doc_text.get(gid, '?')[:90]}")
        for i, did in enumerate(ranked[:3]):
            g = "*" if did in gold_ids else " "
            print(f"   top{i+1}{g} {did} = {doc_text.get(did, '?')[:80]}")

    print("\n" + "=" * 60)
    print("Failures by comp_type:")
    for t in sorted(total_by_type):
        print(f"  {t}: {fails_by_type[t]}/{total_by_type[t]} missed")
    total_fail = sum(fails_by_type.values())
    total = sum(total_by_type.values())
    print(f"  R@1 = {(total - total_fail) / total:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
