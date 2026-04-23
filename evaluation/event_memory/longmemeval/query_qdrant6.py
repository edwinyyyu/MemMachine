"""Check the remaining 9 diff questions where HNSW seems fine — why does cons still win?"""

import asyncio
import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()

HNSW_BROKEN = {
    "19b5f2b3",
    "1d4e3b97",
    "gpt4_385a5000",
    "gpt4_d31cdae3",
    "41698283",
    "41275add",
    "gpt4_18c2b244",
    "gpt4_45189cb4",
    "4f54b7c9",
}


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

    remaining = [qid for qid in diff_qids if qid not in HNSW_BROKEN]
    print(f"Remaining diff questions (HNSW OK): {remaining}")

    queries = [f"User: {raw_by_id[qid]['question']}" for qid in remaining]
    resp = await openai_client.embeddings.create(
        input=queries,
        model="text-embedding-3-small",
        dimensions=1536,
    )
    qembs = {qid: resp.data[i].embedding for i, qid in enumerate(remaining)}

    for qid in remaining:
        pf = Filter(
            must=[FieldCondition(key="sys-partition_key", match=MatchValue(value=qid))]
        )
        qvec = np.array(qembs[qid])
        qnorm = np.linalg.norm(qvec)

        # HNSW
        hnsw = await client.query_points(
            collection_name=collection,
            query=qembs[qid],
            limit=300,
            with_payload=False,
            with_vectors=False,
            query_filter=pf,
        )
        hnsw_scores = [p.score for p in hnsw.points]

        # Exact
        all_points = []
        offset = None
        while True:
            points, next_offset = await client.scroll(
                collection_name=collection,
                scroll_filter=pf,
                limit=1000,
                with_vectors=True,
                offset=offset,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        sims = []
        for p in all_points:
            vec = np.array(p.vector)
            sim = float(np.dot(qvec, vec) / (qnorm * np.linalg.norm(vec)))
            sims.append(sim)
        sims.sort(reverse=True)

        v250_recalled = ov250[qid].get("recalled", 0)
        v250_total = ov250[qid].get("num_answer_turns", 0)
        cons_recalled = ocons[qid].get("recalled", 0)
        cons_total = ocons[qid].get("num_answer_turns", 0)

        print(
            f"\n{qid}: v250={v250_recalled}/{v250_total} cons={cons_recalled}/{cons_total}"
        )
        print(
            f"  HNSW top={hnsw_scores[0]:.4f} exact top={sims[0]:.4f} match={abs(hnsw_scores[0] - sims[0]) < 0.01}"
        )
        print(f"  HNSW ret={len(hnsw_scores)} total={len(all_points)}")
        print(
            f"  HNSW@250={hnsw_scores[249]:.4f} exact@250={sims[249]:.4f}"
            if len(hnsw_scores) >= 250
            else ""
        )

    await client.close()
    await openai_client.close()


asyncio.run(main())
