"""Compare HNSW vs exact search on broken partitions using Qdrant's built-in exact mode."""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchParams

load_dotenv()


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    collection = (
        "longmemeval__3d2e190dd486181aa7995d909c7ddda5c445b2f94d0e07f528573441e14f0da5"
    )

    with open("raw-v250.json") as f:
        raw_v250 = json.load(f)
    with open("recall-v250.json") as f:
        recall_v250 = json.load(f)
    with open("recall-v200-cons0_85.json") as f:
        recall_cons = json.load(f)

    raw_by_id = {r["question_id"]: r for r in raw_v250}
    ov250 = {q["question_id"]: q for q in recall_v250["overall"]["per_question"]}
    ocons = {q["question_id"]: q for q in recall_cons["overall"]["per_question"]}

    diff_qids = [
        qid
        for qid in ov250
        if ocons[qid].get("recalled", 0) > ov250[qid].get("recalled", 0)
    ]

    # Also a few controls
    ctrl_qids = [
        qid
        for qid in ov250
        if ov250[qid].get("recalled", 0) == ocons[qid].get("recalled", 0)
        and ov250[qid].get("num_answer_turns", 0) > 0
    ][:5]

    all_qids = diff_qids + ctrl_qids
    queries = [f"User: {raw_by_id[qid]['question']}" for qid in all_qids]
    resp = await openai_client.embeddings.create(
        input=queries,
        model="text-embedding-3-small",
        dimensions=1536,
    )
    qembs = {qid: resp.data[i].embedding for i, qid in enumerate(all_qids)}

    print(
        f"{'qid':<20} {'group':>5} {'hnsw_top':>9} {'exact_top':>9} {'match':>6} {'hnsw_ret':>8} {'exact_ret':>9}"
    )
    print("-" * 75)

    for qid in all_qids:
        group = "DIFF" if qid in diff_qids else "CTRL"
        pf = Filter(
            must=[FieldCondition(key="sys-partition_key", match=MatchValue(value=qid))]
        )

        # HNSW search
        hnsw = await client.query_points(
            collection_name=collection,
            query=qembs[qid],
            limit=300,
            with_payload=False,
            with_vectors=False,
            query_filter=pf,
        )

        # Exact search
        exact = await client.query_points(
            collection_name=collection,
            query=qembs[qid],
            limit=300,
            with_payload=False,
            with_vectors=False,
            query_filter=pf,
            search_params=SearchParams(exact=True),
        )

        h_scores = [p.score for p in hnsw.points]
        e_scores = [p.score for p in exact.points]
        h_top = h_scores[0] if h_scores else 0
        e_top = e_scores[0] if e_scores else 0
        match = abs(h_top - e_top) < 0.01

        print(
            f"{qid:<20} {group:>5} {h_top:>9.4f} {e_top:>9.4f} {'OK' if match else 'FAIL':>6} {len(h_scores):>8} {len(e_scores):>9}"
        )

    await client.close()
    await openai_client.close()


asyncio.run(main())
