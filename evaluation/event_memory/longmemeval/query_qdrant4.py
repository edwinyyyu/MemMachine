"""Check HNSW health for all 18 diff questions + some control questions."""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    collection_name = (
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

    # Find the 18 diff questions
    diff_qids = [
        qid
        for qid in ov250
        if ocons[qid].get("recalled", 0) > ov250[qid].get("recalled", 0)
    ]

    # Also pick 18 control questions where both systems perform the same
    control_qids = [
        qid
        for qid in ov250
        if ov250[qid].get("recalled", 0) == ocons[qid].get("recalled", 0)
        and ov250[qid].get("num_answer_turns", 0) > 0
    ][:18]

    # Embed all queries at once
    all_qids = diff_qids + control_qids
    queries = [f"User: {raw_by_id[qid]['question']}" for qid in all_qids]

    resp = await openai_client.embeddings.create(
        input=queries,
        model="text-embedding-3-small",
        dimensions=1536,
    )
    query_embeddings = {qid: resp.data[i].embedding for i, qid in enumerate(all_qids)}

    print(
        f"{'qid':<20} {'group':>8} {'derivs':>7} {'returned':>8} {'top_score':>10} {'tied':>5} {'score_250':>10}"
    )
    print("-" * 80)

    for qid in all_qids:
        group = "DIFF" if qid in diff_qids else "CTRL"
        partition_filter = Filter(
            must=[FieldCondition(key="sys-partition_key", match=MatchValue(value=qid))]
        )

        count = await client.count(
            collection_name=collection_name,
            count_filter=partition_filter,
            exact=True,
        )

        results = await client.query_points(
            collection_name=collection_name,
            query=query_embeddings[qid],
            limit=300,
            with_payload=False,
            with_vectors=False,
            query_filter=partition_filter,
        )

        scores = [p.score for p in results.points]
        tied = sum(1 for s in scores if abs(s - scores[0]) < 0.001) if scores else 0
        score_250 = scores[249] if len(scores) >= 250 else scores[-1] if scores else 0

        print(
            f"{qid:<20} {group:>8} {count.count:>7} {len(scores):>8} {scores[0]:>10.4f} {tied:>5} {score_250:>10.4f}"
        )

    await client.close()
    await openai_client.close()


asyncio.run(main())
