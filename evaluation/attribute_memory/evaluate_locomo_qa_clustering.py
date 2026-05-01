"""LoCoMo QA accuracy: clustering ON vs OFF with a real gpt-5-mini extractor.

Ingest a small slice of LoCoMo into AttributeMemory twice — once with
``ClusteringConfig.enabled=True`` and once with it off — then answer a
fixed set of QAs whose evidence lies inside the ingested sessions.
Both answer generation and grading use gpt-5-mini.  Reports QA
accuracy, extraction count, ingest wall time, and LLM call counts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import openai
from dotenv import load_dotenv
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
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
from memmachine_server.semantic_memory.attribute_memory.data_types import (
    ClusterParams,
    Content,
    Event,
    MessageContext,
    Text,
)
from memmachine_server.semantic_memory.attribute_memory.semantic_store.sqlalchemy_semantic_store import (
    CategoryDefinition,
    PartitionSchema,
    SQLAlchemySemanticStore,
    SQLAlchemySemanticStoreParams,
    TopicDefinition,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenvs() -> None:
    load_dotenv(_repo_root() / ".env", override=False)
    load_dotenv(_repo_root() / "evaluation" / ".env", override=True)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "qaeval"
    return cleaned[:32]


def _dt_from_locomo(s: str) -> datetime:
    return datetime.strptime(s, "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)


def _format_attachment(message: dict[str, Any]) -> str:
    blip_caption = message.get("blip_caption")
    image_query = message.get("query")
    if blip_caption and image_query:
        return f" [Attached {blip_caption}: {image_query}]"
    if blip_caption:
        return f" [Attached {blip_caption}]"
    if image_query:
        return f" [Attached a photo: {image_query}]"
    return ""


def build_events(item: dict[str, Any], *, max_sessions: int) -> list[Event]:
    conversation = item["conversation"]
    events: list[Event] = []
    for session_index in range(1, max_sessions + 1):
        key = f"session_{session_index}"
        if key not in conversation:
            break
        base = _dt_from_locomo(conversation[f"{key}_date_time"])
        for message_index, message in enumerate(conversation[key]):
            events.append(
                Event(
                    uuid=uuid4(),
                    timestamp=base + timedelta(seconds=message_index),
                    body=Content(
                        context=MessageContext(source=message["speaker"]),
                        items=[
                            Text(text=message["text"] + _format_attachment(message))
                        ],
                    ),
                    properties={
                        "sample_id": item["sample_id"],
                        "dia_id": message["dia_id"],
                        "session_id": key,
                    },
                )
            )
    return events


def filter_qa(item: dict[str, Any], *, max_sessions: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for qa in item["qa"]:
        if "answer" not in qa:
            continue  # adversarial questions (cat 5) have no gold answer
        evid = qa.get("evidence") or []
        sessions: set[int] = set()
        ok = True
        for e in evid:
            if not isinstance(e, str) or ":" not in e:
                ok = False
                break
            d = e.split(":")[0]
            if not (d.startswith("D") and d[1:].isdigit()):
                ok = False
                break
            sessions.add(int(d[1:]))
        if not ok or not sessions:
            continue
        if max(sessions) <= max_sessions:
            out.append(qa)
    return out


def build_schema() -> PartitionSchema:
    return PartitionSchema(
        topics=(
            TopicDefinition(
                name="ParticipantProfile",
                description=(
                    "You are extracting durable facts about named participants "
                    "from a multi-session conversation transcript. Only record "
                    "facts that are explicitly stated in the transcript. Do not "
                    "infer hidden personality traits, diagnoses, or social class. "
                    "Every attribute name must begin with the participant's first "
                    "name in lowercase followed by an underscore, for example "
                    "`caroline_identity` or `melanie_children`. Keep different "
                    "participants in separate attributes. Prefer durable facts, "
                    "preferences, relationships, recurring activities, important "
                    "life events, possessions, values, and future plans over "
                    "small talk."
                ),
                categories=(
                    CategoryDefinition(name="identity"),
                    CategoryDefinition(name="relationships"),
                    CategoryDefinition(name="work_education"),
                    CategoryDefinition(name="activities"),
                    CategoryDefinition(name="preferences"),
                    CategoryDefinition(name="health_wellbeing"),
                    CategoryDefinition(name="possessions"),
                    CategoryDefinition(name="values_goals"),
                    CategoryDefinition(name="plans"),
                ),
            ),
        )
    )


ANSWER_PROMPT = """You are answering a question about a two-person conversation transcript based only on a structured profile extracted from it.

Profile entries (each is category | attribute | value):
{profile}

Question: {question}

Rules:
- Answer in one short phrase, as concise as possible.
- If the profile does not contain enough information, answer "I don't know".
- Do not invent facts.

Answer:"""


JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.

Question: {question}
Gold answer: {gold}
Generated answer: {generated}

Be generous: if the generated answer touches on the same topic, date (any format), or same entity as the gold answer, label CORRECT. For "I don't know" or answers that don't address the topic, label WRONG.

Reply with JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}."""


async def _answer_question(
    client: openai.AsyncOpenAI,
    model: str,
    question: str,
    attributes: list[tuple[str, str, str]],
) -> str:
    profile = "\n".join(f"{cat} | {attr} | {val}" for cat, attr, val in attributes)
    if not profile:
        profile = "(empty)"
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(profile=profile, question=question),
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def _judge(
    client: openai.AsyncOpenAI,
    model: str,
    question: str,
    gold: str,
    generated: str,
) -> bool:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question, gold=gold, generated=generated
                ),
            }
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        label = json.loads(raw).get("label", "").upper()
    except json.JSONDecodeError:
        label = ""
    return label == "CORRECT"


async def evaluate_qa(
    *,
    memory: AttributeMemory,
    qa_items: list[dict[str, Any]],
    client: openai.AsyncOpenAI,
    model: str,
    top_k: int,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[dict[str, Any]] = []

    async def _one(qa: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            retrieved = await memory.retrieve(qa["question"], top_k=top_k)
            attrs = [(a.category, a.attribute, a.value) for a, _ in retrieved]
            generated = await _answer_question(client, model, qa["question"], attrs)
            correct = await _judge(
                client, model, qa["question"], str(qa["answer"]), generated
            )
            return {
                "category": qa.get("category"),
                "question": qa["question"],
                "gold": str(qa["answer"]),
                "generated": generated,
                "correct": correct,
                "retrieved_count": len(attrs),
                "evidence": qa.get("evidence"),
            }

    tasks = [asyncio.create_task(_one(qa)) for qa in qa_items]
    for task in asyncio.as_completed(tasks):
        results.append(await task)
    return results


async def run_mode(
    *,
    mode: str,
    clustering_enabled: bool,
    batch_size: int | None,
    clustered_similarity_threshold: float,
    consolidation_threshold: int,
    events: list[Event],
    qa_items: list[dict[str, Any]],
    qdrant_client: AsyncQdrantClient,
    openai_client: openai.AsyncOpenAI,
    extraction_model: str,
    embedding_model: str,
    embedding_dimensions: int,
    qa_model: str,
    top_k: int,
    max_concurrency: int,
    sqlite_path: Path,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}")
    semantic_store = SQLAlchemySemanticStore(
        SQLAlchemySemanticStoreParams(engine=engine)
    )
    await semantic_store.startup()

    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    namespace = _slug(f"qaeval_{mode}_{uuid4().hex[:8]}")
    collection_name = _slug(f"col_{mode}_{uuid4().hex[:8]}")
    collection = await vector_store.open_or_create_collection(
        namespace=namespace,
        name=collection_name,
        config=VectorStoreCollectionConfig(
            vector_dimensions=embedding_dimensions,
            similarity_metric=SimilarityMetric.COSINE,
            properties_schema=AttributeMemory.expected_vector_store_collection_schema(),
        ),
    )

    partition_key = _slug(f"qaeval_{mode}_{uuid4().hex[:8]}")
    partition = await semantic_store.open_or_create_partition(partition_key)

    language_model: LanguageModel = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=openai_client,
            model=extraction_model,
            reasoning_effort=reasoning_effort,
        )
    )
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
    )

    # similarity_threshold=0.0 effectively forces every event to join
    # the nearest existing cluster (OpenAI embeddings on conversational
    # text essentially never go negative against each other), so all
    # events pool into one growing cluster.  Combined with chunked
    # ingest(), that flushes each chunk as a single time-contiguous
    # batch — the "batched but not embedding-clustered" ablation.
    similarity_threshold = (
        0.0 if batch_size is not None else clustered_similarity_threshold
    )
    trigger_messages = batch_size if batch_size is not None else 2
    clustering_config = ClusteringConfig(
        enabled=clustering_enabled,
        cluster_params=ClusterParams(similarity_threshold=similarity_threshold),
        trigger_messages=trigger_messages,
        trigger_age=timedelta(seconds=0),
        idle_ttl=None,
        max_clusters_per_run=1000,
        consolidation_threshold=consolidation_threshold,
    )

    memory = AttributeMemory(
        partition=partition,
        vector_collection=collection,
        embedder=embedder,
        language_model=language_model,
        schema=build_schema(),
        clustering_config=clustering_config,
    )

    try:
        print(f"[{mode}] ingesting {len(events)} events...", flush=True)
        t0 = time.monotonic()
        if batch_size is None:
            await memory.ingest(events)
        else:
            sorted_events = sorted(events, key=lambda e: (e.timestamp, e.uuid))
            for i in range(0, len(sorted_events), batch_size):
                chunk = sorted_events[i : i + batch_size]
                await memory.ingest(chunk)
                print(
                    f"[{mode}] flushed chunk {i // batch_size + 1} "
                    f"({len(chunk)} events)",
                    flush=True,
                )
        ingest_seconds = time.monotonic() - t0

        stored_count = 0
        async for _ in partition.list_attributes():
            stored_count += 1

        # Cluster state
        cluster_state = await partition.get_cluster_state()
        cluster_sizes: list[int] = []
        if cluster_state is not None:
            cluster_sizes = [c.count for c in cluster_state.clusters.values()]

        print(
            f"[{mode}] ingested in {ingest_seconds:.1f}s; "
            f"{stored_count} attributes stored; "
            f"{len(cluster_sizes)} clusters "
            f"(sizes={sorted(cluster_sizes, reverse=True)})",
            flush=True,
        )

        print(f"[{mode}] answering {len(qa_items)} QAs...", flush=True)
        t1 = time.monotonic()
        results = await evaluate_qa(
            memory=memory,
            qa_items=qa_items,
            client=openai_client,
            model=qa_model,
            top_k=top_k,
            max_concurrency=max_concurrency,
        )
        qa_seconds = time.monotonic() - t1

        correct = sum(1 for r in results if r["correct"])
        by_cat: dict[int, dict[str, int]] = {}
        for r in results:
            cat = r.get("category") or -1
            d = by_cat.setdefault(cat, {"correct": 0, "total": 0})
            d["total"] += 1
            if r["correct"]:
                d["correct"] += 1

        return {
            "mode": mode,
            "clustering_enabled": clustering_enabled,
            "ingest_seconds": ingest_seconds,
            "qa_seconds": qa_seconds,
            "stored_attributes": stored_count,
            "cluster_sizes": sorted(cluster_sizes, reverse=True),
            "cluster_count": len(cluster_sizes),
            "qa_total": len(results),
            "qa_correct": correct,
            "qa_accuracy": correct / len(results) if results else 0.0,
            "accuracy_by_category": {str(k): v for k, v in sorted(by_cat.items())},
            "results": sorted(
                results, key=lambda r: (r.get("category"), r["question"])
            ),
        }
    finally:
        await semantic_store.shutdown()
        await engine.dispose()
        with contextlib.suppress(Exception):
            await vector_store.delete_collection(
                namespace=namespace, name=collection_name
            )


async def async_main(args: argparse.Namespace) -> None:
    _load_dotenvs()
    data_path = Path(args.data_path)
    dataset = json.loads(await asyncio.to_thread(data_path.read_text))
    item = dataset[args.sample_index]

    events = build_events(item, max_sessions=args.max_sessions)
    qa_items = filter_qa(item, max_sessions=args.max_sessions)
    if args.qa_limit and len(qa_items) > args.qa_limit:
        qa_items = qa_items[: args.qa_limit]

    print(
        f"Sample {item['sample_id']}: {len(events)} events across "
        f"{args.max_sessions} sessions, {len(qa_items)} QAs with evidence in them",
        flush=True,
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    openai_client = openai.AsyncOpenAI(api_key=api_key, base_url=args.openai_base_url)
    qdrant_client = AsyncQdrantClient(
        host=args.qdrant_host,
        port=args.qdrant_port,
        grpc_port=args.qdrant_grpc_port,
        prefer_grpc=args.prefer_grpc,
    )

    base_dir = Path(args.sqlite_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = uuid4().hex[:8]

    # (mode_name, clustering_enabled, batch_size)
    all_modes: list[tuple[str, bool, int | None]] = [
        ("clustered", True, None),
        ("non_clustered", False, None),
        ("time_batched", True, args.time_batch_size),
    ]
    selected = set(args.modes.split(",")) if args.modes else {m[0] for m in all_modes}
    reports: list[dict[str, Any]] = []
    for mode, enabled, batch_size in all_modes:
        if mode not in selected:
            continue
        sqlite_path = base_dir / f"qaeval_{run_suffix}_{mode}.db"
        report = await run_mode(
            mode=mode,
            clustering_enabled=enabled,
            batch_size=batch_size,
            clustered_similarity_threshold=args.similarity_threshold,
            consolidation_threshold=args.consolidation_threshold,
            events=events,
            qa_items=qa_items,
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            extraction_model=args.extraction_model,
            embedding_model=args.embedding_model,
            embedding_dimensions=args.embedding_dimensions,
            qa_model=args.qa_model,
            top_k=args.top_k,
            max_concurrency=args.max_concurrency,
            sqlite_path=sqlite_path,
            reasoning_effort=args.reasoning_effort,
        )
        reports.append(report)

    await qdrant_client.close()

    summary = {
        "sample_id": item["sample_id"],
        "max_sessions": args.max_sessions,
        "event_count": len(events),
        "qa_count": len(qa_items),
        "extraction_model": args.extraction_model,
        "qa_model": args.qa_model,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "modes": reports,
    }

    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str, sort_keys=False))
    print("\n=== RESULT ===", flush=True)
    for r in reports:
        print(
            f"{r['mode']:<16} acc={r['qa_accuracy']:.3f} "
            f"({r['qa_correct']}/{r['qa_total']}), "
            f"ingest={r['ingest_seconds']:.1f}s, "
            f"attrs={r['stored_attributes']}, "
            f"clusters={r['cluster_count']}",
            flush=True,
        )
    print(f"report: {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    _load_dotenvs()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        default=str(_repo_root() / "evaluation" / "data" / "locomo10.json"),
    )
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--max-sessions", type=int, default=2)
    p.add_argument("--qa-limit", type=int, default=20)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--extraction-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini")
    )
    p.add_argument("--qa-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--embedding-dimensions", type=int, default=1536)
    p.add_argument("--reasoning-effort", default=os.getenv("OPENAI_REASONING_EFFORT"))
    p.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--qdrant-host", default="127.0.0.1")
    p.add_argument("--qdrant-port", type=int, default=6333)
    p.add_argument("--qdrant-grpc-port", type=int, default=6334)
    p.add_argument("--prefer-grpc", action="store_true")
    p.add_argument("--sqlite-dir", default=str(_repo_root() / "evaluation" / "out"))
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument(
        "--modes",
        help="Comma-separated subset of {clustered,non_clustered,time_batched}",
    )
    p.add_argument(
        "--time-batch-size",
        type=int,
        default=5,
        help="Fixed chunk size (events) for the time_batched ablation",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.55,
        help="Cosine threshold for clustered mode (ClusterParams.similarity_threshold)",
    )
    p.add_argument(
        "--consolidation-threshold",
        type=int,
        default=0,
        help="Auto-consolidate when (topic, category) attr count reaches this",
    )
    p.add_argument(
        "--report-path",
        default=str(
            _repo_root() / "evaluation" / "out" / "qaeval_clustering_report.json"
        ),
    )
    return p.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
