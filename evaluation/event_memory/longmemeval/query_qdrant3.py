"""Check multiple partitions and test exact derivative retrieval."""

import asyncio
import json
import os
import sys

sys.path.insert(0, "/Users/eyu/edwinyyyu/mmcc/extra_memory/packages/server/src")

from dotenv import load_dotenv
from memmachine_server.common.utils import extract_sentences
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

    # Load raw data to get queries and answer texts
    with open("raw-v250.json") as f:
        raw_v250 = json.load(f)
    raw_by_id = {r["question_id"]: r for r in raw_v250}

    # Test multiple partitions - some that work well and some that don't
    # The 18 diff questions vs some normal questions
    test_partitions = {
        "gpt4_18c2b244": "broken (0/3 answer turns in v250)",
        "1d4e3b97": "broken (0/2 answer turns in v250)",
        "gpt4_385a5000": "broken (0/2 answer turns in v250)",
        "6c49646a": "works (1/2 answer turns in v250)",
        "3fdac837": "works (1/2 answer turns in v250)",
    }

    query_text = "User: {q}"

    for qid, label in test_partitions.items():
        raw = raw_by_id[qid]
        q = f"User: {raw['question']}"

        partition_filter = Filter(
            must=[FieldCondition(key="sys-partition_key", match=MatchValue(value=qid))]
        )

        count = await client.count(
            collection_name=collection_name,
            count_filter=partition_filter,
            exact=True,
        )

        resp = await openai_client.embeddings.create(
            input=[q],
            model="text-embedding-3-small",
            dimensions=1536,
        )
        qe = resp.data[0].embedding

        results = await client.query_points(
            collection_name=collection_name,
            query=qe,
            limit=300,
            with_payload=True,
            with_vectors=False,
            query_filter=partition_filter,
        )

        scores = [p.score for p in results.points]
        unique_ids = len(set(p.id for p in results.points))
        at_top_score = (
            sum(1 for s in scores if abs(s - scores[0]) < 0.001) if scores else 0
        )

        print(f"\n{qid} [{label}]")
        print(f"  Total derivatives: {count.count}")
        print(f"  Results returned: {len(scores)}")
        print(f"  Unique IDs: {unique_ids}")
        print(f"  Top score: {scores[0]:.4f}, tied at top: {at_top_score}")
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")

    # Now test exact derivative text for gpt4_18c2b244
    print(f"\n{'=' * 60}")
    print("=== Exact derivative text test for gpt4_18c2b244 ===")

    # Get actual answer segment text from cons results
    with open("raw-v200-cons0_85.json") as f:
        raw_cons = json.load(f)
    cons_by_id = {r["question_id"]: r for r in raw_cons}

    raw_c = cons_by_id["gpt4_18c2b244"]
    raw_v = raw_by_id["gpt4_18c2b244"]

    partition_filter = Filter(
        must=[
            FieldCondition(
                key="sys-partition_key", match=MatchValue(value="gpt4_18c2b244")
            )
        ]
    )

    # Extract sentences from the answer turns and non-answer turns in v250
    # to see if they exist as derivatives
    print("\n--- Testing answer turn sentences ---")
    for sc in raw_c["segment_contexts"][:3]:
        seg = sc["segments"][0]
        ctx = seg.get("context")
        source = ctx.get("source", "") if ctx else ""
        full_text = seg.get("text", "")

        sentences = extract_sentences(full_text)
        # Format like the system does
        formatted = [f"{source}: {s}" if source else s for s in sentences]

        # Embed each sentence and search
        if not formatted:
            continue

        resp = await openai_client.embeddings.create(
            input=formatted[:5],  # just first 5 sentences
            model="text-embedding-3-small",
            dimensions=1536,
        )

        print(f"\n  Rank {sc['rank']} segment (from cons):")
        for sent, emb_data in zip(formatted[:5], resp.data):
            results = await client.query_points(
                collection_name=collection_name,
                query=emb_data.embedding,
                limit=1,
                with_vectors=True,
                query_filter=partition_filter,
            )
            if results.points:
                p = results.points[0]
                # compute cosine between query embedding and result
                import numpy as np

                qv = np.array(emb_data.embedding)
                rv = np.array(p.vector)
                cosine = np.dot(qv, rv) / (np.linalg.norm(qv) * np.linalg.norm(rv))
                print(
                    f"    sent='{sent[:80]}' -> top match score={p.score:.4f}, recomputed_cosine={cosine:.4f}, id={p.id}"
                )

    # Also test: embed "Assistant: 3." exactly and see what score it gets
    print("\n--- Testing degenerate derivatives ---")
    degen_texts = ["Assistant: 3.", "Assistant: 1.", "User: 3.", "Assistant: 2."]
    resp = await openai_client.embeddings.create(
        input=degen_texts,
        model="text-embedding-3-small",
        dimensions=1536,
    )
    for text, emb_data in zip(degen_texts, resp.data):
        results = await client.query_points(
            collection_name=collection_name,
            query=emb_data.embedding,
            limit=1,
            with_vectors=True,
            query_filter=partition_filter,
        )
        if results.points:
            p = results.points[0]
            import numpy as np

            qv = np.array(emb_data.embedding)
            rv = np.array(p.vector)
            cosine = np.dot(qv, rv) / (np.linalg.norm(qv) * np.linalg.norm(rv))
            print(f"  '{text}' -> score={p.score:.4f}, cosine={cosine:.4f}, id={p.id}")

    await client.close()
    await openai_client.close()


asyncio.run(main())
