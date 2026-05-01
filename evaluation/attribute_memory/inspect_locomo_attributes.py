"""Ingest LoCoMo conversations into AttributeMemory and dump SQLite rows."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import openai
from dotenv import load_dotenv
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.language_model.openai_chat_completions_language_model import (
    OpenAIChatCompletionsLanguageModel,
    OpenAIChatCompletionsLanguageModelParams,
)
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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

LLM_BACKEND_OPENAI_RESPONSES = "openai-responses"
LLM_BACKEND_OPENAI_CHAT = "openai-chat"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_evaluation_dotenvs() -> None:
    repo_root = _repo_root()
    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(repo_root / "evaluation" / ".env", override=True)


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    if not cleaned:
        cleaned = "inspect"
    return cleaned[:32]


class TokenHashEmbedder(Embedder):
    """Deterministic token-hash embedder with cosine-normalized outputs."""

    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    async def ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        del max_attempts
        return [self._embed(str(item)) for item in inputs]

    async def search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        del max_attempts
        return [self._embed(str(item)) for item in queries]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            vector[index] += 1.0
        if tokens:
            anchor_digest = hashlib.sha256(tokens[0].encode("utf-8")).digest()
            anchor_a = int.from_bytes(anchor_digest[:4], "big") % self._dimensions
            anchor_b = int.from_bytes(anchor_digest[4:8], "big") % self._dimensions
            vector[anchor_a] += 4.0
            vector[anchor_b] += 4.0
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]

    @property
    def model_id(self) -> str:
        return "locomo-token-hash"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


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


def build_events(item: dict[str, Any], *, max_sessions: int | None) -> list[Event]:
    conversation = item["conversation"]
    events: list[Event] = []
    session_index = 1
    while True:
        session_key = f"session_{session_index}"
        date_key = f"{session_key}_date_time"
        if session_key not in conversation:
            break
        if max_sessions is not None and session_index > max_sessions:
            break

        session_datetime = datetime_from_locomo_time(conversation[date_key])
        for message_index, message in enumerate(conversation[session_key]):
            events.append(
                Event(
                    uuid=uuid4(),
                    timestamp=session_datetime + timedelta(seconds=message_index),
                    body=Content(
                        context=MessageContext(source=message["speaker"]),
                        items=[
                            Text(
                                text=message["text"] + _format_attachment(message),
                            )
                        ],
                    ),
                    properties={
                        "sample_id": item["sample_id"],
                        "dia_id": message["dia_id"],
                        "session_id": session_key,
                    },
                    metadata={
                        "speaker_a": conversation["speaker_a"],
                        "speaker_b": conversation["speaker_b"],
                    },
                )
            )
        session_index += 1
    return events


def trim_events(events: list[Event], *, max_messages: int | None) -> list[Event]:
    if max_messages is None:
        return events
    return events[:max_messages]


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
                    CategoryDefinition(
                        name="identity",
                        description=(
                            "Identity, background, demographics, self-description."
                        ),
                    ),
                    CategoryDefinition(
                        name="relationships",
                        description=(
                            "Partner, family, children, friends, support system."
                        ),
                    ),
                    CategoryDefinition(
                        name="work_education",
                        description=(
                            "Work, studies, training, volunteering, career goals."
                        ),
                    ),
                    CategoryDefinition(
                        name="activities",
                        description=(
                            "Hobbies, routines, travel, events, creative activities."
                        ),
                    ),
                    CategoryDefinition(
                        name="preferences",
                        description=("Likes, dislikes, favorites, stable tastes."),
                    ),
                    CategoryDefinition(
                        name="health_wellbeing",
                        description=(
                            "Health, recovery, coping strategies, wellbeing practices."
                        ),
                    ),
                    CategoryDefinition(
                        name="possessions",
                        description=(
                            "Pets, collections, owned items, meaningful objects."
                        ),
                    ),
                    CategoryDefinition(
                        name="values_goals",
                        description=(
                            "Beliefs, motivations, principles, long-term goals."
                        ),
                    ),
                    CategoryDefinition(
                        name="plans",
                        description=(
                            "Explicit upcoming plans or commitments likely to matter "
                            "later."
                        ),
                    ),
                ),
            ),
        )
    )


def build_language_model(
    *,
    llm_backend: str,
    openai_model: str,
    openai_base_url: str | None,
    openai_reasoning_effort: str | None,
) -> LanguageModel:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to the repo-root .env or "
            "evaluation/.env."
        )

    client = openai.AsyncOpenAI(api_key=api_key, base_url=openai_base_url)
    if llm_backend == LLM_BACKEND_OPENAI_RESPONSES:
        return OpenAIResponsesLanguageModel(
            OpenAIResponsesLanguageModelParams(
                client=client,
                model=openai_model,
                reasoning_effort=openai_reasoning_effort,
            )
        )
    if llm_backend == LLM_BACKEND_OPENAI_CHAT:
        return OpenAIChatCompletionsLanguageModel(
            OpenAIChatCompletionsLanguageModelParams(
                client=client,
                model=openai_model,
            )
        )
    raise ValueError(f"Unsupported llm backend: {llm_backend}")


def _serialize_value(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize_value(raw) for key, raw in value.items()}
    if isinstance(value, list):
        return [_serialize_value(raw) for raw in value]
    return value


async def fetch_raw_attribute_rows(
    engine: AsyncEngine,
    *,
    partition_key: str,
) -> list[dict[str, Any]]:
    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT
                    id,
                    topic,
                    category,
                    attribute,
                    value,
                    properties,
                    created_at,
                    updated_at
                FROM semantic_store_at
                WHERE partition_key = :partition_key
                ORDER BY category, attribute, value, id
                """
            ),
            {"partition_key": partition_key},
        )
        return [
            {key: _serialize_value(value) for key, value in row.items()}
            for row in result.mappings()
        ]


async def fetch_cluster_state(
    engine: AsyncEngine,
    *,
    partition_key: str,
) -> dict[str, Any] | None:
    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT state
                FROM semantic_store_cs
                WHERE partition_key = :partition_key
                """
            ),
            {"partition_key": partition_key},
        )
        row = result.mappings().first()
        if row is None:
            return None
        return _serialize_value(row["state"])


def _qa_examples(item: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    return item["qa"][:limit]


def _prepare_sqlite_path(raw_path: str) -> Path:
    sqlite_path = Path(raw_path).resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite_path


async def async_main(args: argparse.Namespace) -> None:
    load_evaluation_dotenvs()

    data_path = Path(args.data_path)
    dataset = json.loads(await asyncio.to_thread(data_path.read_text))
    item = dataset[args.sample_index]
    conversation = item["conversation"]
    events = trim_events(
        build_events(item, max_sessions=args.max_sessions),
        max_messages=args.max_messages,
    )
    print(
        f"Loaded sample {item['sample_id']} with {len(events)} events "
        f"(max_sessions={args.max_sessions}, max_messages={args.max_messages})",
        flush=True,
    )

    run_suffix = uuid4().hex[:8]
    sqlite_path = _prepare_sqlite_path(args.sqlite_path)

    engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}")
    semantic_store = SQLAlchemySemanticStore(
        SQLAlchemySemanticStoreParams(engine=engine)
    )
    await semantic_store.startup()

    client = AsyncQdrantClient(
        host=args.qdrant_host,
        port=args.qdrant_port,
        grpc_port=args.qdrant_grpc_port,
        prefer_grpc=args.prefer_grpc,
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=client))

    namespace = _slug(f"locomo_inspect_{run_suffix}")
    collection_name = _slug(f"{item['sample_id']}_{run_suffix}")
    collection = await vector_store.open_or_create_collection(
        namespace=namespace,
        name=collection_name,
        config=VectorStoreCollectionConfig(
            vector_dimensions=64,
            similarity_metric=SimilarityMetric.COSINE,
            properties_schema=AttributeMemory.expected_vector_store_collection_schema(),
        ),
    )

    partition_key = _slug(f"locomo_{item['sample_id']}_{run_suffix}")
    partition = await semantic_store.open_or_create_partition(partition_key)
    memory = AttributeMemory(
        partition=partition,
        vector_collection=collection,
        embedder=TokenHashEmbedder(),
        language_model=build_language_model(
            llm_backend=args.llm_backend,
            openai_model=args.openai_model,
            openai_base_url=args.openai_base_url,
            openai_reasoning_effort=args.openai_reasoning_effort,
        ),
        schema=build_schema(),
        clustering_config=ClusteringConfig(
            enabled=args.clustering,
            cluster_params=ClusterParams(
                similarity_threshold=args.similarity_threshold,
                max_time_gap=(
                    timedelta(days=args.max_time_gap_days)
                    if args.max_time_gap_days is not None
                    else None
                ),
            ),
            trigger_messages=args.trigger_messages,
            trigger_age=(
                timedelta(days=args.trigger_age_days)
                if args.trigger_age_days is not None
                else None
            ),
            idle_ttl=None,
            max_clusters_per_run=1000,
            consolidation_threshold=0,
        ),
    )

    try:
        print("Starting ingest...", flush=True)
        processed = await memory.ingest(events)
        print(f"Ingest complete, processed {len(processed)} events.", flush=True)
        raw_rows = await fetch_raw_attribute_rows(engine, partition_key=partition_key)
        print(f"Fetched {len(raw_rows)} stored SQLite rows.", flush=True)
        typed_attributes = [
            {
                "id": str(attribute.id),
                "topic": attribute.topic,
                "category": attribute.category,
                "attribute": attribute.attribute,
                "value": attribute.value,
                "properties": attribute.properties,
                "citations": (
                    [str(citation) for citation in attribute.citations]
                    if attribute.citations is not None
                    else None
                ),
            }
            async for attribute in partition.list_attributes(load_citations=True)
        ]
        report = {
            "sample": {
                "sample_index": args.sample_index,
                "sample_id": item["sample_id"],
                "speaker_a": conversation["speaker_a"],
                "speaker_b": conversation["speaker_b"],
                "max_sessions": args.max_sessions,
                "events_ingested": len(events),
                "processed_event_count": len(processed),
                "session_summaries": {
                    f"session_{index}_summary": item["session_summary"][
                        f"session_{index}_summary"
                    ]
                    for index in range(1, (args.max_sessions or 999) + 1)
                    if f"session_{index}_summary" in item["session_summary"]
                },
                "qa_examples": _qa_examples(item, limit=args.qa_limit),
            },
            "run": {
                "clustering": args.clustering,
                "llm_backend": args.llm_backend,
                "openai_model": args.openai_model,
                "sqlite_path": str(sqlite_path),
                "partition_key": partition_key,
                "qdrant_namespace": namespace,
                "qdrant_collection": collection_name,
            },
            "cluster_state": await fetch_cluster_state(
                engine, partition_key=partition_key
            ),
            "attribute_count": len(raw_rows),
            "raw_database_rows": raw_rows,
            "typed_attributes": typed_attributes,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        try:
            if not args.keep_qdrant:
                await vector_store.delete_collection(
                    namespace=namespace, name=collection_name
                )
        finally:
            await client.close()
            await engine.dispose()


def parse_args() -> argparse.Namespace:
    load_evaluation_dotenvs()
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a LoCoMo conversation into AttributeMemory and dump the "
            "stored SQLite attributes for qualitative inspection."
        ),
    )
    parser.add_argument(
        "--data-path",
        default=str(_repo_root() / "evaluation" / "data" / "locomo10.json"),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=3)
    parser.add_argument("--max-messages", type=int)
    parser.add_argument(
        "--llm-backend",
        choices=[LLM_BACKEND_OPENAI_RESPONSES, LLM_BACKEND_OPENAI_CHAT],
        default=LLM_BACKEND_OPENAI_RESPONSES,
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL"),
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        default=os.getenv("OPENAI_REASONING_EFFORT"),
    )
    parser.add_argument("--qdrant-host", default="127.0.0.1")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--qdrant-grpc-port", type=int, default=6334)
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument(
        "--sqlite-path",
        default=str(
            _repo_root() / "evaluation" / "out" / "locomo_attribute_inspect.db"
        ),
    )
    parser.add_argument(
        "--clustering", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--max-time-gap-days", type=float)
    parser.add_argument("--trigger-messages", type=int, default=5)
    parser.add_argument("--trigger-age-days", type=float, default=1.0)
    parser.add_argument("--qa-limit", type=int, default=12)
    parser.add_argument("--keep-qdrant", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
