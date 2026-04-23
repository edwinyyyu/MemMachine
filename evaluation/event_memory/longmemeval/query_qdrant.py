"""Query Qdrant directly to see what derivatives are returned for gpt4_18c2b244."""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

load_dotenv()


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # First, find the collection name for gpt4_18c2b244
    collections = await client.get_collections()
    matching = [c.name for c in collections.collections if "gpt4_18c2b244" in c.name]
    print(f"Matching collections: {matching}")

    if not matching:
        all_names = [c.name for c in collections.collections]
        lme = [n for n in all_names if "longmemeval" in n.lower()]
        print(f"Longmemeval collections: {lme[:10]}")
        print(f"Total collections: {len(all_names)}")
        print(f"Sample names: {all_names[:10]}")

    collection_name = (
        "longmemeval__3d2e190dd486181aa7995d909c7ddda5c445b2f94d0e07f528573441e14f0da5"
    )
    print(f"Using collection: {collection_name}")

    info = await client.get_collection(collection_name)
    print(f"Points count: {info.points_count}")
    print(f"Config: {info.config}")

    query = "User: What is the order of the three events: 'I signed up for the rewards program at ShopRite', 'I used a Buy One Get One Free coupon on Luvs diapers at Walmart', and 'I redeemed $12 cashback for a $10 Amazon gift card from Ibotta'?"

    resp = await openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
        dimensions=1536,
    )
    query_embedding = resp.data[0].embedding

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    results = await client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=20,
        with_payload=True,
        with_vectors=True,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="sys-partition_key",
                    match=MatchValue(value="gpt4_18c2b244"),
                )
            ]
        ),
    )

    print("\nTop 20 derivatives from Qdrant:")
    import numpy as np

    query_vec = np.array(query_embedding)
    query_norm = np.linalg.norm(query_vec)

    for i, point in enumerate(results.points):
        vec = np.array(point.vector)
        actual_sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec))
        print(
            f"\n  rank {i}: id={point.id}, qdrant_score={point.score:.4f}, recomputed_cosine={actual_sim:.4f}"
        )
        if point.payload:
            for k, v in point.payload.items():
                val_str = str(v)[:100]
                print(f"    {k}: {val_str}")

    await client.close()
    await openai_client.close()


asyncio.run(main())
