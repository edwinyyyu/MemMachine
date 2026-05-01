"""Quick prompt-sanity check: feed a few hard_bench queries through each
prompt template, print model's responses, verify it's reading them right.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

from openai import AsyncOpenAI
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from salience_eval import (
    rank_t as rank_multi_axis_t,
)
from v7l_ts_blind_eval import (
    DIR_PROMPT,
    DIR_WITH_REF_PROMPT,
    MODEL,
    PICK_PROMPT,
    PICK_WITH_REF_PROMPT,
    RetrievalCache,
    _format_one_set,
    _format_sets,
)

N_DIAG = 5  # number of queries to diagnose
ONLY_DIVERSE = True  # only diagnose queries where top-3 differs across weights


async def call(client, prompt, label):
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=64,
        reasoning_effort="minimal",
    )
    out = resp.choices[0].message.content or ""
    return out.strip()


async def main():
    name = "hard_bench"
    docs = [json.loads(l) for l in open(DATA_DIR / "hard_bench_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "hard_bench_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "hard_bench_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"=== {name}: {len(docs)} docs, {len(queries)} queries ===")

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", "v7l-hard_bench")
    q_ext = await run_v2_extract(q_items, f"{name}-queries", "v7l-hard_bench")

    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)
    for d in docs:
        doc_mem.setdefault(
            d["doc_id"],
            {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "multi_tags": set(),
            },
        )

    doc_text = {d["doc_id"]: d["text"] for d in docs}

    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    client = AsyncOpenAI(timeout=30.0, max_retries=1)

    # Find queries where top-3 changes between w=0.0, 0.2, 0.4 (i.e., weight matters)
    diverse = []
    for q in queries:
        qid = q["query_id"]
        per_q_t = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        per_q_s = rank_semantic(qid, q_embs, doc_embs)
        cache = RetrievalCache(per_q_t, per_q_s, doc_text)
        top3_at = {w: tuple(cache.get(w)[0][:3]) for w in [0.0, 0.2, 0.4]}
        if len({top3_at[w] for w in [0.0, 0.2, 0.4]}) > 1:
            diverse.append((q, per_q_t, per_q_s))
        if len(diverse) >= N_DIAG:
            break

    print(f"\nFound {len(diverse)} queries where top-3 differs across w=0.0/0.2/0.4")

    for q, per_q_t, per_q_s in diverse:
        qid = q["query_id"]
        qtext = q["text"]
        rel = gold.get(qid, [])
        cache = RetrievalCache(per_q_t, per_q_s, doc_text)

        # Build candidate sets at three w_T values
        sets_at = {w: cache.get(w)[1] for w in [0.0, 0.2, 0.4, 1.0]}
        ref_S = sets_at[0.0]
        ref_T = sets_at[1.0]

        print(f"\n=========== Query {qid} ===========")
        print(f"Q: {qtext[:200]}")
        print(f"Gold: {rel[:3]}")
        # Print docs at each weight
        for w in [0.0, 0.2, 0.4, 1.0]:
            ranked, _ = cache.get(w)
            top3 = ranked[:3]
            hit_pos = next((i + 1 for i, d in enumerate(ranked[:10]) if d in rel), None)
            print(f"  w={w} top3={top3[:3]} gold@={hit_pos}")

        # Test PICK_PROMPT
        cands = [sets_at[0.0], sets_at[0.2], sets_at[0.4]]
        prompt = PICK_PROMPT.format(
            query=qtext,
            sets=_format_sets(cands),
            choices="1, 2, 3",
        )
        out = await call(client, prompt, "PICK_PROMPT")
        print(f"  PICK            (cands w_T=0.0/0.2/0.4): {out!r}")

        # Test PICK_WITH_REF_PROMPT
        prompt = PICK_WITH_REF_PROMPT.format(
            query=qtext,
            ref_S=_format_one_set(ref_S),
            ref_T=_format_one_set(ref_T),
            sets=_format_sets(cands),
            choices="1, 2, 3",
        )
        out = await call(client, prompt, "PICK_WITH_REF_PROMPT")
        print(f"  PICK_WITH_REF   (cands w_T=0.0/0.2/0.4): {out!r}")

        # Test DIR_PROMPT — prev=0.4, cur=0.2
        prompt = DIR_PROMPT.format(
            query=qtext,
            prev_set=_format_one_set(sets_at[0.4]),
            cur_set=_format_one_set(sets_at[0.2]),
        )
        out = await call(client, prompt, "DIR_PROMPT")
        print(f"  DIR             (prev=0.4, cur=0.2):    {out!r}")

        # Test DIR_WITH_REF_PROMPT — same prev/cur with refs
        prompt = DIR_WITH_REF_PROMPT.format(
            query=qtext,
            ref_S=_format_one_set(ref_S),
            ref_T=_format_one_set(ref_T),
            prev_set=_format_one_set(sets_at[0.4]),
            cur_set=_format_one_set(sets_at[0.2]),
        )
        out = await call(client, prompt, "DIR_WITH_REF_PROMPT")
        print(f"  DIR_WITH_REF    (prev=0.4, cur=0.2):    {out!r}")


if __name__ == "__main__":
    asyncio.run(main())
