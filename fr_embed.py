"""Embed normalized corpus items (documents) and queries (queries) with
embeddinggemma-300m, normalized, save to .npz for reuse."""
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

SYNTH = "/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/synth"
NORM = os.path.join(SYNTH, "normalized_corpus.json")
OUT = "/Users/eyu/edwinyyyu/mmcc/temporal_scoring/fr_embeddings.npz"


def main():
    with open(NORM) as f:
        d = json.load(f)
    items = d["items"]
    queries = d["queries"]

    model = SentenceTransformer("google/embeddinggemma-300m", device="mps")

    item_texts = [it["text"] for it in items]
    query_texts = [q["situation"] for q in queries]

    print(f"embedding {len(item_texts)} documents...")
    item_emb = model.encode(
        item_texts, prompt_name="document",
        normalize_embeddings=True, batch_size=32, show_progress_bar=True,
    ).astype(np.float32)
    print(f"embedding {len(query_texts)} queries...")
    query_emb = model.encode(
        query_texts, prompt_name="query",
        normalize_embeddings=True, batch_size=32, show_progress_bar=True,
    ).astype(np.float32)

    item_ids = np.array([it["id"] for it in items])
    query_ids = np.array([q["id"] for q in queries])
    np.savez(
        OUT,
        item_emb=item_emb, query_emb=query_emb,
        item_ids=item_ids, query_ids=query_ids,
    )
    print("item_emb", item_emb.shape, "query_emb", query_emb.shape)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
