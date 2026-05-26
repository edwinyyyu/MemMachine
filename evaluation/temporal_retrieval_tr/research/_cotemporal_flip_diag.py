"""Find the specific cotemporal query that flips between W=0 and W=1.5,
and the specific composition queries that additive flips but Copeland misses.

We don't need to guess what the failure pattern is — just find the
queries where R@1 changes and look at the full pool.
"""
from __future__ import annotations

import asyncio

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import load_bench, make_cosine_rerank_fn

setup_env()


async def run_arm(vd, queries, mode: str, value: float) -> dict[str, list]:
    if mode == "additive":
        vd.recency_weight = value
        vd.copeland_bonus = None
    else:
        vd.copeland_bonus = value
    out = {}
    for q in queries:
        r = await vd.query(q["text"], q["ref_time"], k=10)
        out[q["query_id"]] = r
    return out


async def diag_bench(bench: str, embed_fn, rerank_fn) -> None:
    print(f"\n=== {bench} ===", flush=True)
    loaded = load_bench(bench)
    if loaded[0] is None:
        print("  SKIPPED")
        return
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    doc_text = {d.id: d.text for d in docs}
    doc_reftime = {d.id: d.ref_time for d in docs}
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    res_w0   = await run_arm(vd, queries, "additive", 0.0)
    res_w1   = await run_arm(vd, queries, "additive", 1.0)
    res_w15  = await run_arm(vd, queries, "additive", 1.5)
    res_c015 = await run_arm(vd, queries, "copeland", 0.15)
    res_c030 = await run_arm(vd, queries, "copeland", 0.30)
    # Find queries where R@1 changes meaningfully
    q_by_id = {q["query_id"]: q for q in queries}
    for qid in res_w0:
        top_w0   = res_w0[qid][0].doc_id   if res_w0[qid]   else None
        top_w1   = res_w1[qid][0].doc_id   if res_w1[qid]   else None
        top_w15  = res_w15[qid][0].doc_id  if res_w15[qid]  else None
        top_c015 = res_c015[qid][0].doc_id if res_c015[qid] else None
        top_c030 = res_c030[qid][0].doc_id if res_c030[qid] else None
        golds = gold.get(qid, [])
        hit_w0   = top_w0   in golds
        hit_w1   = top_w1   in golds
        hit_w15  = top_w15  in golds
        hit_c015 = top_c015 in golds
        hit_c030 = top_c030 in golds
        # Show queries where ANY arm disagrees with W=0 baseline
        same_set = {top_w0, top_w1, top_w15, top_c015, top_c030}
        if len(same_set) > 1 or not hit_w0:
            print(f"\n  qid={qid}: '{q_by_id[qid]['text']}'")
            print(f"    golds={golds}")
            print(f"    W=0   top={top_w0}  ({'HIT' if hit_w0 else 'miss'})")
            print(f"    W=1.0 top={top_w1}  ({'HIT' if hit_w1 else 'miss'})")
            print(f"    W=1.5 top={top_w15} ({'HIT' if hit_w15 else 'miss'})")
            print(f"    C0.15 top={top_c015} ({'HIT' if hit_c015 else 'miss'})")
            print(f"    C0.30 top={top_c030} ({'HIT' if hit_c030 else 'miss'})")
            print(f"    Pool at W=0 (top 8):")
            for r in res_w0[qid][:8]:
                rt = doc_reftime.get(r.doc_id, "")
                mark = "*GOLD*" if r.doc_id in golds else ""
                txt = doc_text.get(r.doc_id, "")[:120]
                print(f"      {mark:6s} {r.doc_id:25s} sim(rrk+m)={r.rerank+r.match:5.3f}"
                      f" rrk={r.rerank:5.3f} m={r.match:5.3f}"
                      f" ref={rt[:10]} | {txt}")


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    await diag_bench("cotemporal", embed_fn, rerank_fn)
    print("\n\n" + "=" * 60)
    await diag_bench("composition", embed_fn, rerank_fn)


if __name__ == "__main__":
    asyncio.run(main())
