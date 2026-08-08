"""Track A: does conditioning the anchor keep/drop on neighbor context help?

Runs TWO arms on the SAME pools, apples-to-apples:
  A0 bare    : judge each anchor in isolation (module FILTER_PROMPT / v5)
  A1 context : judge each anchor marked within its expand_context neighborhood

Gold is anchor(seed)-level: a candidate is gold iff its seed segment timestamp
matches gold evidence. Metric: recall@10 / nDCG@10 over the filtered ordering
(keepers in embedding order, droppers behind), plus keep-ratio and gold-dropped.

Subsets: 'golddrop' = questions where the prior run dropped >=1 gold (context
should help here); 'sample' = first --limit scored questions (over-keep / no-harm
check). Single sample per call (aggregates over questions).

Usage:
  uv run python context_filter_eval.py --subset golddrop --pool-size 100 \
    --expand-context 6 --limit 0 --out ctxfilter-A-golddrop.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from datetime import timedelta

from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import create_async_engine

from embedder_factory import build_embedder
from locomo_models import datetime_from_locomo_time, load_locomo_dataset
from swiss_rerank_probe import (
    _FORMAT_OPTIONS, FILTER_PROMPT, build_gold_timestamps, recall_at_k,
    ndcg_at_k,
)
from context_filter_test import CTX_PROMPT as CTX_V1
from recover_test import CTX_V2
import os as _os
CTX_PROMPT = CTX_V2 if _os.getenv("CTX_VERSION", "v2") == "v2" else CTX_V1
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory, EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore, SQLAlchemySegmentStoreParams,
)
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLiteVecVectorStore, SQLiteVecVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)


def render_seg(seg) -> str:
    return EventMemory.string_from_segment_context(
        [seg], format_options=_FORMAT_OPTIONS)


def render_ctx(segments, seed_uuid) -> str:
    """Render the neighborhood with the seed (anchor) marked."""
    lines = []
    for s in segments:
        line = EventMemory.string_from_segment_context(
            [s], format_options=_FORMAT_OPTIONS)
        if s.uuid == seed_uuid:
            line = ">>> CANDIDATE >>> " + line
        lines.append(line)
    return "\n".join(lines)


async def judge(client, model, effort, prompt) -> tuple[str, int, int]:
    try:
        r = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            extra_body={"reasoning_effort": effort})
    except Exception:
        r = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}])
    t = (r.choices[0].message.content or "").strip().upper()
    u = r.usage
    v = "DROP" if t.startswith("DROP") else "KEEP"
    return v, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


def order_metrics(is_gold, verdicts, emb_order, k):
    keepers = [i for i in emb_order if verdicts[i] == "KEEP"]
    droppers = [i for i in emb_order if verdicts[i] == "DROP"]
    order = keepers + droppers
    rg = [is_gold[i] for i in order]
    tg = sum(is_gold)
    return {
        "recall@k": recall_at_k(rg, tg, k),
        "ndcg@k": ndcg_at_k(rg, tg, k),
        "n_kept": len(keepers),
        "gold_dropped": sum(1 for i in droppers if is_gold[i]),
    }


async def main() -> None:
    load_dotenv(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/"
        "locomo/.env"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["golddrop", "sample"], default="golddrop")
    ap.add_argument("--pool-size", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--expand-context", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--comparator-model", default="gpt-5-nano")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--http-pool", type=int, default=40)
    ap.add_argument("--out", default="ctxfilter-A.json")
    args = ap.parse_args()

    golddrop_qs = set()
    if args.subset == "golddrop":
        prior = json.load(open("swiss-probe-filter-p100-v3.json"))
        golddrop_qs = {r["question"] for r in prior["records"]
                       if r.get("gold_dropped", 0) > 0}

    data = load_locomo_dataset("../../data/locomo10_c2sub.json")
    seg_engine = create_async_engine(
        "sqlite+aiosqlite:///swiss-textwhole-c2sub.sqlite",
        connect_args={"timeout": 30}, pool_size=20, max_overflow=80)
    ss = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=seg_engine))
    await ss.startup()
    vec_engine = create_async_engine(
        "sqlite+aiosqlite:///swiss-textwhole-c2sub.vec.sqlite",
        connect_args={"timeout": 30}, pool_size=20, max_overflow=80)
    vs = SQLiteVecVectorStore(SQLiteVecVectorStoreParams(engine=vec_engine))
    await vs.startup()
    import httpx
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.AsyncClient(limits=httpx.Limits(
            max_connections=args.http_pool,
            max_keepalive_connections=args.http_pool)),
        max_retries=0)
    embedder = build_embedder("text-embedding-3-small", client)

    records = []
    tok = {"bare_in": 0, "bare_out": 0, "ctx_in": 0, "ctx_out": 0}
    for idx, item in enumerate(data):
        if "conversation" not in item:
            continue
        col = await vs.open_collection(namespace="locomo", name=f"group_{idx}")
        if col is None:
            continue
        pa = await ss.open_or_create_partition(
            f"group_{idx}", SegmentStorePartitionConfig())
        mem = EventMemory(EventMemoryParams(
            vector_store_collection=col, segment_store_partition=pa,
            segmenter=TextSegmenter(), deriver=WholeTextDeriver(),
            embedder=embedder, reranker=None))
        gold_ts = build_gold_timestamps(item["conversation"])
        for qa in item["qa"]:
            if str(qa["category"]) not in {"1", "2", "4"}:
                continue
            q = qa["question"]
            if args.subset == "golddrop" and q not in golddrop_qs:
                continue
            gold_set = {gold_ts[e] for e in qa.get("evidence", [])
                        if e in gold_ts}
            if not gold_set:
                continue
            qr = await mem.query(
                query=q, vector_search_limit=args.pool_size,
                expand_context=args.expand_context, format_options=_FORMAT_OPTIONS,
                bm25_fusion="none")
            pool = qr.scored_segment_contexts
            if len(pool) < 2:
                continue
            # seed segment per candidate (the anchor)
            seeds = []
            for ssc in pool:
                seed = next((s for s in ssc.segments
                             if s.uuid == ssc.seed_segment_uuid),
                            ssc.segments[0])
                seeds.append(seed)
            is_gold = [seed.timestamp in gold_set for seed in seeds]
            if not any(is_gold):
                continue
            n = len(pool)
            emb_order = list(range(n))

            async def do(i):
                bare_doc = render_seg(seeds[i])
                ctx_doc = render_ctx(pool[i].segments, pool[i].seed_segment_uuid)
                bv, bi, bo = await judge(
                    client, args.comparator_model, args.effort,
                    FILTER_PROMPT.format(query=q, doc=bare_doc))
                cv, ci, co = await judge(
                    client, args.comparator_model, args.effort,
                    CTX_PROMPT.format(query=q, conversation=ctx_doc))
                return bv, cv, bi, bo, ci, co

            outs = await asyncio.gather(*(do(i) for i in range(n)))
            bare_v = [o[0] for o in outs]
            ctx_v = [o[1] for o in outs]
            for o in outs:
                tok["bare_in"] += o[2]; tok["bare_out"] += o[3]
                tok["ctx_in"] += o[4]; tok["ctx_out"] += o[5]
            rec = {
                "category": str(qa["category"]), "question": q,
                "gold_in_pool": sum(is_gold), "pool": n,
                "A0_bare": order_metrics(is_gold, bare_v, emb_order, args.top_k),
                "A1_ctx": order_metrics(is_gold, ctx_v, emb_order, args.top_k),
            }
            records.append(rec)
            print(f"[g{idx} c{qa['category']}] gip={sum(is_gold)} "
                  f"bare R@k={rec['A0_bare']['recall@k']:.2f} kept={rec['A0_bare']['n_kept']} gd={rec['A0_bare']['gold_dropped']} | "
                  f"ctx R@k={rec['A1_ctx']['recall@k']:.2f} kept={rec['A1_ctx']['n_kept']} gd={rec['A1_ctx']['gold_dropped']}")
            if args.limit and len(records) >= args.limit:
                break
        if args.limit and len(records) >= args.limit:
            break

    def agg(arm, metric):
        vals = [r[arm][metric] for r in records
                if not (isinstance(r[arm][metric], float) and math.isnan(r[arm][metric]))]
        return round(sum(vals) / len(vals), 4) if vals else float("nan")

    summary = {
        "n": len(records), "subset": args.subset, "expand_context": args.expand_context,
        "pool_size": args.pool_size, "comparator": args.comparator_model, "effort": args.effort,
        "A0_bare": {m: agg("A0_bare", m) for m in
                    ["recall@k", "ndcg@k", "n_kept", "gold_dropped"]},
        "A1_ctx": {m: agg("A1_ctx", m) for m in
                   ["recall@k", "ndcg@k", "n_kept", "gold_dropped"]},
        "tokens": tok,
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)
    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
