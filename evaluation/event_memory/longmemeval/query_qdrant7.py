"""Check: how many derivatives are exact vector dupes vs near-dupes in problematic partitions?"""

import asyncio
from collections import Counter

import numpy as np
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()

TEST_QIDS = ["gpt4_18c2b244", "031748ae", "61f8c8f8", "a96c20ee"]


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    collection = (
        "longmemeval__3d2e190dd486181aa7995d909c7ddda5c445b2f94d0e07f528573441e14f0da5"
    )

    for qid in TEST_QIDS:
        pf = Filter(
            must=[FieldCondition(key="sys-partition_key", match=MatchValue(value=qid))]
        )

        all_points = []
        offset = None
        while True:
            points, next_offset = await client.scroll(
                collection_name=collection,
                scroll_filter=pf,
                limit=1000,
                with_vectors=True,
                with_payload=False,
                offset=offset,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        # Hash vectors for exact dedup
        vec_hashes = []
        vecs = []
        for p in all_points:
            v = np.array(p.vector)
            vecs.append(v)
            vec_hashes.append(tuple(round(x, 6) for x in p.vector))

        vec_counter = Counter(vec_hashes)
        total = len(vec_hashes)
        unique = len(vec_counter)

        print(f"\n{'=' * 70}")
        print(
            f"{qid}: {total} total, {unique} unique vectors ({total - unique} exact dupes, {100 * (total - unique) / total:.1f}% removable)"
        )

        top_dupes = vec_counter.most_common(20)
        print("  Top duplicated vectors:")
        for i, (vh, count) in enumerate(top_dupes[:10]):
            if count <= 1:
                break
            print(f"    {count}x (vector norm={np.linalg.norm(np.array(vh)):.4f})")

        # After exact dedup, count near-dupe clusters (>0.85 cosine)
        unique_vecs = []
        seen = set()
        for vh, v in zip(vec_hashes, vecs):
            if vh not in seen:
                seen.add(vh)
                unique_vecs.append(v)

        # Sample 1000 unique vecs for pairwise check
        import random

        random.seed(42)
        sample_size = min(1000, len(unique_vecs))
        sample_idx = random.sample(range(len(unique_vecs)), sample_size)
        sample = [unique_vecs[i] for i in sample_idx]
        norms = [np.linalg.norm(v) for v in sample]

        # Count near-dupe pairs
        high_sim = 0
        total_pairs = 0
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                sim = np.dot(sample[i], sample[j]) / (norms[i] * norms[j])
                if sim > 0.85:
                    high_sim += 1
                total_pairs += 1

        print(f"  After exact dedup: {len(unique_vecs)} unique vectors")
        print(
            f"  Near-dupe pairs (>0.85 cos) in sample of {sample_size}: {high_sim}/{total_pairs} ({100 * high_sim / total_pairs:.3f}%)"
        )

        # Estimate cluster sizes: for each sampled vec, count how many others are >0.85
        neighbor_counts = []
        for i in range(min(200, sample_size)):
            neighbors = sum(
                1
                for j in range(sample_size)
                if j != i
                and np.dot(sample[i], sample[j]) / (norms[i] * norms[j]) > 0.85
            )
            neighbor_counts.append(neighbors)

        neighbor_counts.sort(reverse=True)
        print(
            f"  Near-dupe neighbor counts (top 20 of 200 sampled): {neighbor_counts[:20]}"
        )

    await client.close()


asyncio.run(main())
