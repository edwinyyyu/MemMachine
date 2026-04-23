"""Check: how many derivatives score 0.215, and do answer-relevant derivatives exist?"""

import asyncio
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
    partition_filter = Filter(
        must=[
            FieldCondition(
                key="sys-partition_key", match=MatchValue(value="gpt4_18c2b244")
            )
        ]
    )

    query = "User: What is the order of the three events: 'I signed up for the rewards program at ShopRite', 'I used a Buy One Get One Free coupon on Luvs diapers at Walmart', and 'I redeemed $12 cashback for a $10 Amazon gift card from Ibotta'?"

    resp = await openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
        dimensions=1536,
    )
    query_embedding = resp.data[0].embedding

    # Get top 300 to see how many are at 0.215
    results = await client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=300,
        with_payload=True,
        with_vectors=False,
        query_filter=partition_filter,
    )

    scores = [p.score for p in results.points]
    at_215 = sum(1 for s in scores if abs(s - 0.215) < 0.001)
    print(f"Top 300: {at_215} at score ~0.215")
    print(f"Score distribution: min={min(scores):.4f}, max={max(scores):.4f}")
    print(f"Unique scores (rounded to 4dp): {len(set(round(s, 4) for s in scores))}")

    # Show score transitions
    prev = None
    for i, s in enumerate(scores):
        if prev is not None and abs(s - prev) > 0.001:
            print(f"  Score changes at rank {i}: {prev:.4f} -> {s:.4f}")
        prev = s

    # Now embed some answer-relevant text and search for it
    answer_texts = [
        "User: I'm planning a trip to Walmart this weekend and I'm looking for some deals on baby essentials.",
        "User: I'm trying to plan my grocery shopping trip for this week. Can you help me find any good deals or sales on diapers and formula?",
        "Assistant: ShopRite Rewards: As a rewards member, you'll earn points on your purchases",
    ]

    resp2 = await openai_client.embeddings.create(
        input=answer_texts,
        model="text-embedding-3-small",
        dimensions=1536,
    )

    print("\n=== Searching for answer-relevant derivatives ===")
    for text, emb_data in zip(answer_texts, resp2.data):
        results = await client.query_points(
            collection_name=collection_name,
            query=emb_data.embedding,
            limit=5,
            with_payload=True,
            with_vectors=False,
            query_filter=partition_filter,
        )
        print(f"\nQuery: {text[:80]}")
        for p in results.points:
            print(f"  score={p.score:.4f}, id={p.id}")

    # Count total points in this partition
    count = await client.count(
        collection_name=collection_name,
        count_filter=partition_filter,
        exact=True,
    )
    print(f"\nTotal derivatives for gpt4_18c2b244: {count.count}")

    await client.close()
    await openai_client.close()


asyncio.run(main())
