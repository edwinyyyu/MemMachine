"""Probe whether similarity-clustered batching uniquely beats time-contiguous
batching on interleaved multi-fragment facts.

Uses the existing synthetic dataset from evaluate_attribute_memory_clustering.py
(24 facts × 3 fragments, interleaved so each fact's fragments are 24 events apart).
Runs three modes sharing the same synthetic rule-based LLM:

* clustered: similarity_threshold=0.55, same-anchor events cluster together
* time_batched_3: threshold=0.0 pool cluster, external chunks of 3 via repeated
  ingest() calls — the "fixed-size buffer" baseline that time-windows the stream
* non_clustered: per-event extraction

Extraction recall = stored exact (category, attribute, value) triples vs expected.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import openai
from dotenv import load_dotenv
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.vector_store import VectorStoreCollectionConfig
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.semantic_memory.attribute_memory import (
    AttributeMemory,
    ClusteringConfig,
)
from memmachine_server.semantic_memory.attribute_memory.data_types import ClusterParams
from memmachine_server.semantic_memory.attribute_memory.semantic_store.sqlalchemy_semantic_store import (
    SQLAlchemySemanticStore,
    SQLAlchemySemanticStoreParams,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

from evaluation.attribute_memory.evaluate_attribute_memory_clustering import (
    SyntheticRuleLanguageModel,
    build_dataset,
    build_schema,
)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

BASE_DIR = Path("/tmp")
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_GRPC = 6334


async def run_one(
    *,
    mode: str,
    events,
    facts,
    qdrant_client,
    openai_client,
    clustering_enabled: bool,
    similarity_threshold: float,
    trigger_messages: int,
    chunk_size: int | None,
) -> dict:
    run_suffix = uuid4().hex[:8]
    # Namespace/collection must match [a-z0-9_]+ and be ≤32 bytes
    import re as _re

    slug = _re.sub(r"[^a-z0-9_]+", "_", mode.lower()).strip("_")[:20]
    sqlite_path = BASE_DIR / f"synth_{slug}_{run_suffix}.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}")
    semantic_store = SQLAlchemySemanticStore(
        SQLAlchemySemanticStoreParams(engine=engine)
    )
    await semantic_store.startup()

    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    namespace = f"synth_{slug}_{run_suffix}"
    collection_name = f"col_{slug}_{run_suffix}"
    collection = await vector_store.open_or_create_collection(
        namespace=namespace,
        name=collection_name,
        config=VectorStoreCollectionConfig(
            vector_dimensions=EMBEDDING_DIMS,
            similarity_metric=SimilarityMetric.COSINE,
            properties_schema=AttributeMemory.expected_vector_store_collection_schema(),
        ),
    )

    partition_key = f"{slug}_{run_suffix}"
    partition = await semantic_store.open_or_create_partition(partition_key)

    facts_by_anchor = {fact.anchor: fact for fact in facts}
    llm = SyntheticRuleLanguageModel(facts_by_anchor)

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMS,
        )
    )
    memory = AttributeMemory(
        partition=partition,
        vector_collection=collection,
        embedder=embedder,
        language_model=llm,
        schema=build_schema(),
        clustering_config=ClusteringConfig(
            enabled=clustering_enabled,
            cluster_params=ClusterParams(similarity_threshold=similarity_threshold),
            trigger_messages=trigger_messages,
            trigger_age=timedelta(seconds=0),
            idle_ttl=None,
            max_clusters_per_run=1000,
            consolidation_threshold=0,
        ),
    )

    try:
        if chunk_size is None:
            await memory.ingest(events)
            llm_call_count = -1  # single call for non-cluster, N clusters for clustered
        else:
            ordered = sorted(events, key=lambda e: (e.timestamp, e.uuid))
            for i in range(0, len(ordered), chunk_size):
                await memory.ingest(ordered[i : i + chunk_size])
            llm_call_count = (len(ordered) + chunk_size - 1) // chunk_size

        stored = {
            (attr.category, attr.attribute, attr.value)
            async for attr in partition.list_attributes()
        }
        expected = {(f.category, f.attribute, f.expected_value) for f in facts}
        tp = len(stored & expected)
        prec = tp / len(stored) if stored else 0.0
        rec = tp / len(expected) if expected else 0.0
        return {
            "mode": mode,
            "extracted": len(stored),
            "expected": len(expected),
            "true_positives": tp,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "ingest_chunks": llm_call_count,
        }
    finally:
        await semantic_store.shutdown()
        await engine.dispose()
        with contextlib.suppress(Exception):
            await vector_store.delete_collection(
                namespace=namespace, name=collection_name
            )
        with contextlib.suppress(FileNotFoundError):
            sqlite_path.unlink()


async def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    load_dotenv(
        Path(__file__).resolve().parents[2] / "evaluation" / ".env", override=True
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    openai_client = openai.AsyncOpenAI(api_key=api_key)

    dataset = build_dataset(cases=24, seed=7)
    events = list(dataset.events)
    facts = list(dataset.facts)

    print(
        f"dataset: {len(events)} events, {len(facts)} facts "
        f"({sum(1 for f in facts if f.clustered_case)} multi-turn, "
        f"{sum(1 for f in facts if not f.clustered_case)} one-shot)"
    )
    print(f"embedder: {EMBEDDING_MODEL} ({EMBEDDING_DIMS}d)")

    qdrant_client = AsyncQdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT, grpc_port=QDRANT_GRPC
    )

    results = []
    # Sweep similarity thresholds to find where same-anchor events cluster
    # under real text-embedding-3-small.
    for sim in (0.50, 0.65, 0.80, 0.90):
        results.append(
            await run_one(
                mode=f"clustered_sim{sim:.2f}",
                events=events,
                facts=facts,
                qdrant_client=qdrant_client,
                openai_client=openai_client,
                clustering_enabled=True,
                similarity_threshold=sim,
                trigger_messages=1,
                chunk_size=None,
            )
        )
    # time_batched chunk=3: fixed-size buffer; no similarity signal
    results.append(
        await run_one(
            mode="time_batched_3",
            events=events,
            facts=facts,
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            clustering_enabled=True,
            similarity_threshold=0.0,
            trigger_messages=3,
            chunk_size=3,
        )
    )
    # time_batched chunk=6
    results.append(
        await run_one(
            mode="time_batched_6",
            events=events,
            facts=facts,
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            clustering_enabled=True,
            similarity_threshold=0.0,
            trigger_messages=6,
            chunk_size=6,
        )
    )
    # time_batched chunk=24 (exactly one stride)
    results.append(
        await run_one(
            mode="time_batched_24",
            events=events,
            facts=facts,
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            clustering_enabled=True,
            similarity_threshold=0.0,
            trigger_messages=24,
            chunk_size=24,
        )
    )
    # time_batched chunk=all (one big batch — degenerate but tests recovery)
    results.append(
        await run_one(
            mode="time_batched_all",
            events=events,
            facts=facts,
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            clustering_enabled=True,
            similarity_threshold=0.0,
            trigger_messages=len(events),
            chunk_size=len(events),
        )
    )
    # non_clustered
    results.append(
        await run_one(
            mode="non_clustered",
            events=events,
            facts=facts,
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            clustering_enabled=False,
            similarity_threshold=0.55,
            trigger_messages=1,
            chunk_size=None,
        )
    )

    await qdrant_client.close()

    print()
    print(f"{'mode':<22} {'extracted':>10} {'TP':>4} {'prec':>6} {'recall':>7}")
    for r in results:
        print(
            f"{r['mode']:<22} {r['extracted']:>10} {r['true_positives']:>4} "
            f"{r['precision']:>6.3f} {r['recall']:>7.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
