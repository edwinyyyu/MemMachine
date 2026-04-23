"""Verify: are v250 results ordered by SENTENCE derivative similarity?
The index contains sentence-level derivatives, not full segment text.
Extract sentences from each segment and check if any sentence has high similarity."""

import asyncio
import json
import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

with open("raw-v250.json") as f:
    raw_v250 = json.load(f)

raw_v250_by_id = {r["question_id"]: r for r in raw_v250}


# Use the same sentence extraction as the system
def extract_sentences(text):
    """Simple sentence splitter matching the system's behavior."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


async def check_sentence_similarities():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    import numpy as np

    test_qids = ["1d4e3b97", "gpt4_18c2b244", "gpt4_385a5000"]

    for qid in test_qids:
        raw_v = raw_v250_by_id[qid]
        query_text = f"User: {raw_v['question']}"

        # Get top 5 v250 segments and extract sentences
        texts_to_embed = [query_text]
        labels = ["QUERY"]
        segment_sentence_ranges = []  # (start_idx, end_idx) in texts_to_embed

        for i, sc in enumerate(raw_v["segment_contexts"][:5]):
            seg = sc["segments"][0]
            ctx = seg.get("context")
            source = ctx.get("source", "") if ctx else ""
            full_text = seg.get("text", "")

            # The derivative is formatted as "Source: sentence"
            sentences = extract_sentences(full_text)
            start = len(texts_to_embed)
            for sent in sentences:
                formatted = f"{source}: {sent}" if source else sent
                texts_to_embed.append(formatted)
                labels.append(f"v250_r{i}_sent")
            end = len(texts_to_embed)
            segment_sentence_ranges.append((i, start, end, full_text[:80]))

        # Embed all
        # Batch to avoid token limits
        all_embeddings = []
        batch_size = 100
        for batch_start in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[batch_start : batch_start + batch_size]
            resp = await client.embeddings.create(
                input=batch,
                model="text-embedding-3-small",
                dimensions=1536,
            )
            all_embeddings.extend([d.embedding for d in resp.data])

        query_vec = np.array(all_embeddings[0])
        query_norm = np.linalg.norm(query_vec)

        print(f"\n{'=' * 70}")
        print(f"Question: {qid}")
        print(f"Query: {raw_v['question'][:100]}")

        for rank, start, end, text_preview in segment_sentence_ranges:
            best_sim = -1
            best_sent = ""
            for j in range(start, end):
                vec = np.array(all_embeddings[j])
                sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec))
                if sim > best_sim:
                    best_sim = sim
                    best_sent = texts_to_embed[j][:100]

            all_sims = []
            for j in range(start, end):
                vec = np.array(all_embeddings[j])
                sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec))
                all_sims.append(sim)

            print(f"\n  rank {rank} ({end - start} sentences): segment={text_preview}")
            print(f"    best sentence sim: {best_sim:.4f} - {best_sent}")
            print(f"    all sentence sims: {[f'{s:.3f}' for s in all_sims]}")

    await client.close()


asyncio.run(check_sentence_similarities())
