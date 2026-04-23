"""Exact search for mode-2 partitions to check if low scores are HNSW failure or genuine."""

import asyncio
import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()

# Mode-2 partitions identified from prior run (returns 250-300 results, top < 0.40)
MODE2_QIDS = [
    "19b5f2b3",
    "1d4e3b97",
    "gpt4_385a5000",
    "gpt4_d31cdae3",
    "41698283",
    "41275add",
]
# A few controls with high top scores
CTRL_QIDS = ["a96c20ee", "6c49646a", "3fdac837"]


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    collection = (
        "longmemeval__3d2e190dd486181aa7995d909c7ddda5c445b2f94d0e07f528573441e14f0da5"
    )

    with open("raw-v250.json") as f:
        raw_v250 = json.load(f)
    raw_by_id = {r["question_id"]: r for r in raw_v250}

    all_qids = MODE2_QIDS + CTRL_QIDS
    queries = [f"User: {raw_by_id[qid]['question']}" for qid in all_qids]
    resp = await openai_client.embeddings.create(
        input=queries,
        model="text-embedding-3-small",
        dimensions=1536,
    )
    qembs = {qid: resp.data[i].embedding for i, qid in enumerate(all_qids)}

    print(
        f"{'qid':<20} {'group':>6} {'total':>6} {'hnsw_top':>9} {'exact_top':>9} {'exact@250':>9} {'hnsw==exact':>11}"
    )
    print("-" * 85)

    for qid in all_qids:
        group = "MODE2" if qid in MODE2_QIDS else "CTRL"
        pf = Filter(
            must=[FieldCondition(key="sys-partition_key", match=MatchValue(value=qid))]
        )
        qvec = np.array(qembs[qid])
        qnorm = np.linalg.norm(qvec)

        # HNSW search
        hnsw_results = await client.query_points(
            collection_name=collection,
            query=qembs[qid],
            limit=10,
            with_payload=False,
            with_vectors=False,
            query_filter=pf,
        )
        hnsw_top = hnsw_results.points[0].score if hnsw_results.points else 0

        # Exact search: scroll all vectors
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

        # Compute exact cosine
        sims = []
        for p in all_points:
            vec = np.array(p.vector)
            sim = float(np.dot(qvec, vec) / (qnorm * np.linalg.norm(vec)))
            sims.append(sim)
        sims.sort(reverse=True)

        exact_top = sims[0] if sims else 0
        exact_250 = sims[249] if len(sims) >= 250 else sims[-1] if sims else 0
        match = abs(hnsw_top - exact_top) < 0.01

        print(
            f"{qid:<20} {group:>6} {len(all_points):>6} {hnsw_top:>9.4f} {exact_top:>9.4f} {exact_250:>9.4f} {'YES' if match else 'NO':>11}"
        )

    await client.close()
    await openai_client.close()


asyncio.run(main())
