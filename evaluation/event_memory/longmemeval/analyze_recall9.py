"""Check if v250 results are reasonably ordered by embedding similarity,
or if they look random. Compute actual embedding similarity for a few questions."""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

with open("raw-v250.json") as f:
    raw_v250 = json.load(f)
with open("raw-v200-cons0_85.json") as f:
    raw_cons = json.load(f)

raw_v250_by_id = {r["question_id"]: r for r in raw_v250}
raw_cons_by_id = {r["question_id"]: r for r in raw_cons}


def turn_key(props):
    return f"{props['longmemeval_session_id']}:{props['turn_id']}"


# Pick a few of the diff questions where v250 looks most "random"
test_qids = ["1d4e3b97", "gpt4_18c2b244", "61f8c8f8", "19b5f2b3", "gpt4_385a5000"]


async def compute_similarities():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for qid in test_qids:
        raw_v = raw_v250_by_id[qid]
        raw_c = raw_cons_by_id[qid]
        query = f"User: {raw_v['question']}"

        # Collect texts: query + v250 top 10 + cons top 10 + answer turns in cons
        texts = [query]
        labels = ["QUERY"]

        # v250 top 10
        for i, sc in enumerate(raw_v["segment_contexts"][:10]):
            seg = sc["segments"][0]
            ctx = seg.get("context")
            source = ctx.get("source", "") if ctx else ""
            text = f"{source}: {seg.get('text', '')}" if source else seg.get("text", "")
            texts.append(text)
            answer_turns = set(raw_v["answer_turn_indices"])
            is_ans = turn_key(seg["properties"]) in answer_turns
            labels.append(f"v250_r{i}{'[ANS]' if is_ans else ''}")

        # cons top 10
        for i, sc in enumerate(raw_c["segment_contexts"][:10]):
            seg = sc["segments"][0]
            ctx = seg.get("context")
            source = ctx.get("source", "") if ctx else ""
            text = f"{source}: {seg.get('text', '')}" if source else seg.get("text", "")
            texts.append(text)
            answer_turns = set(raw_c["answer_turn_indices"])
            is_ans = turn_key(seg["properties"]) in answer_turns
            labels.append(f"cons_r{i}{'[ANS]' if is_ans else ''}")

        # Embed all
        resp = await client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
            dimensions=1536,
        )
        embeddings = [d.embedding for d in resp.data]
        query_emb = embeddings[0]

        # Compute cosine similarity with query
        import numpy as np

        query_vec = np.array(query_emb)
        query_norm = np.linalg.norm(query_vec)

        print(f"\n{'=' * 70}")
        print(f"Question: {qid}")
        print(f"Query: {raw_v['question'][:100]}")
        print(f"\n{'Label':<25} {'Cosine Sim':>10}  Text preview")
        print("-" * 90)

        for i in range(1, len(embeddings)):
            vec = np.array(embeddings[i])
            sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec))
            text_preview = texts[i][:80].replace("\n", " ")
            print(f"{labels[i]:<25} {sim:>10.4f}  {text_preview}")

        # Check if v250 results are monotonically decreasing in similarity
        v250_sims = []
        for i in range(1, 11):
            vec = np.array(embeddings[i])
            sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec))
            v250_sims.append(sim)

        is_monotonic = all(
            v250_sims[i] >= v250_sims[i + 1] - 0.001 for i in range(len(v250_sims) - 1)
        )
        print(f"\nv250 top-10 sims: {[f'{s:.4f}' for s in v250_sims]}")
        print(f"Roughly monotonic: {is_monotonic}")
        print(f"v250 sim range: {min(v250_sims):.4f} - {max(v250_sims):.4f}")

        cons_sims = []
        for i in range(11, 21):
            vec = np.array(embeddings[i])
            sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec))
            cons_sims.append(sim)
        print(f"cons sim range: {min(cons_sims):.4f} - {max(cons_sims):.4f}")

    await client.close()


asyncio.run(compute_similarities())
