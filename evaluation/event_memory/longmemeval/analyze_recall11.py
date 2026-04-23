"""Verify ordering using the ACTUAL sentence extractor and context formatting."""

import asyncio
import json
import os
import sys

sys.path.insert(0, "/Users/eyu/edwinyyyu/mmcc/extra_memory/packages/server/src")

from dotenv import load_dotenv
from memmachine_server.common.utils import extract_sentences
from openai import AsyncOpenAI

load_dotenv()

with open("raw-v250.json") as f:
    raw_v250 = json.load(f)

raw_v250_by_id = {r["question_id"]: r for r in raw_v250}


def format_with_context(text, context):
    if context and context.get("source"):
        return f"{context['source']}: {text}"
    return text


async def check():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    import numpy as np

    # gpt4_385a5000 had non-monotonic sentence sims - investigate
    for qid in ["gpt4_385a5000", "gpt4_18c2b244"]:
        raw_v = raw_v250_by_id[qid]
        query_text = f"User: {raw_v['question']}"

        texts_to_embed = [query_text]
        labels = ["QUERY"]
        segment_info = []

        for i, sc in enumerate(raw_v["segment_contexts"][:10]):
            seg = sc["segments"][0]
            ctx = seg.get("context")
            full_text = seg.get("text", "")

            # Use actual sentence extractor
            sentences = extract_sentences(full_text)

            start = len(texts_to_embed)
            formatted_sents = []
            for sent in sentences:
                formatted = format_with_context(sent, ctx)
                texts_to_embed.append(formatted)
                formatted_sents.append(formatted)
            end = len(texts_to_embed)

            segment_info.append((i, start, end, full_text[:60], formatted_sents))

        # Embed
        all_embeddings = []
        batch_size = 100
        for bs in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[bs : bs + batch_size]
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
        print(f"Query embedding prefix (for sanity): {query_text[:50]}")

        for rank, start, end, text_preview, formatted_sents in segment_info:
            if end == start:
                print(f"\n  rank {rank}: NO SENTENCES extracted from: {text_preview}")
                continue

            sims = []
            for j in range(start, end):
                vec = np.array(all_embeddings[j])
                sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec))
                sims.append((sim, texts_to_embed[j]))

            best_sim, best_text = max(sims, key=lambda x: x[0])
            print(f"\n  rank {rank} ({end - start} sentences): {text_preview}")
            print(f"    BEST sim={best_sim:.4f}: {best_text[:100]}")
            if len(sims) > 1:
                sims_sorted = sorted(sims, key=lambda x: -x[0])
                for s, t in sims_sorted[:3]:
                    print(f"      {s:.4f}: {t[:100]}")

    await client.close()


asyncio.run(check())
