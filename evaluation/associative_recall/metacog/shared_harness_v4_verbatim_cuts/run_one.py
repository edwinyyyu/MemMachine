"""Quick driver to run a single scenario for v4 (used after the main parallel
run hung on world-knowledge-bridge-01). Reuses everything from main.py."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from main import (  # noqa: E402
    RESULTS_DIR,
    SCORE_K_LIST,
    AsyncQdrantClient,
    OpenAIEmbedder,
    OpenAIEmbedderParams,
    QdrantVectorStore,
    QdrantVectorStoreParams,
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
    create_async_engine,
    load_locomo_segments,
    load_scenarios,
    load_speakers,
    openai,
    run_one_scenario,
)


async def main() -> None:
    SID = sys.argv[1] if len(sys.argv) > 1 else "world-knowledge-bridge-01"
    K_list = SCORE_K_LIST
    variants = ["baseline_fifo", "operator_lru"]

    scenarios_all = load_scenarios()
    target = next(s for s in scenarios_all if s["scenario_id"] == SID)
    locomo_segments = load_locomo_segments()
    speakers_map = load_speakers()

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sqlite_path = RESULTS_DIR / f"eventmemory_v4_one_{SID}.sqlite3"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
    engine = create_async_engine(sql_url)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = openai.AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], timeout=120.0
    )
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    try:
        result = await run_one_scenario(
            scenario=target,
            locomo_segments=locomo_segments,
            speakers_map=speakers_map,
            vector_store=vector_store,
            segment_store=segment_store,
            embedder=embedder,
            openai_client=openai_client,
            K_list=K_list,
            variants=variants,
        )
        print("\n=== single-scenario result ===")
        for v in variants:
            agg = result["per_variant"][v]["aggregates"]
            ar = result["per_variant"][v]
            n_compactor = sum(1 for t in ar["trace"] if t.get("compactor_invoked"))
            n_dropped = sum(len(t.get("compactor_dropped", [])) for t in ar["trace"])
            print(
                f"  {v}: cov={agg['coverage_rate']} | full_R@5={agg.get('triggered_recall_full@5')} | "
                f"cond_R@5={agg.get('recall_given_covered@5')} | turns={ar['n_turns']} | "
                f"compactor_calls={n_compactor} | items_dropped={n_dropped}"
            )
    finally:
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()


if __name__ == "__main__":
    asyncio.run(main())
