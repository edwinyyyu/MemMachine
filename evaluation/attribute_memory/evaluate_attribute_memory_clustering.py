"""Synthetic evaluation harness for AttributeMemory clustering.

Uses:
* SQLite via ``SQLAlchemySemanticStore`` for relational storage.
* Qdrant via ``QdrantVectorStore`` for vector retrieval.
* Lazily generated synthetic conversations with exact ground truth.

The harness compares clustered ingestion against a true non-clustered
mode (``ClusteringConfig.enabled=False``) and reports:
* exact attribute extraction precision / recall / F1
* retrieval recall@1 and recall@3 over generated queries
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any
from uuid import uuid4

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
    CommandType,
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
from pydantic import TypeAdapter
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine


@dataclass(frozen=True)
class FactTemplate:
    category: str
    attribute_prefix: str
    anchor_prefix: str
    fields: tuple[str, ...]
    pools: dict[str, tuple[str, ...]]
    fragments: int


@dataclass(frozen=True)
class SyntheticFact:
    anchor: str
    category: str
    attribute: str
    required_fields: tuple[str, ...]
    field_values: dict[str, str]
    expected_value: str
    query: str
    fragments: tuple[str, ...]
    clustered_case: bool


@dataclass(frozen=True)
class SyntheticDataset:
    events: tuple[Event, ...]
    facts: tuple[SyntheticFact, ...]


@dataclass(frozen=True)
class ModeMetrics:
    mode: str
    partition_key: str
    namespace: str
    collection_name: str
    sqlite_path: str
    extracted: int
    expected: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    retrieval_recall_at_1: float
    retrieval_recall_at_3: float
    clustered_case_recall: float
    one_shot_case_recall: float


LLM_BACKEND_SYNTHETIC = "synthetic"
LLM_BACKEND_OPENAI_RESPONSES = "openai-responses"
LLM_BACKEND_OPENAI_CHAT = "openai-chat"


MULTI_TURN_TEMPLATES: tuple[FactTemplate, ...] = (
    FactTemplate(
        category="travel",
        attribute_prefix="trip",
        anchor_prefix="summer_trip",
        fields=("destination", "neighborhood", "flight"),
        pools={
            "destination": ("osaka", "lisbon", "seoul", "montreal"),
            "neighborhood": ("dotonbori", "alfama", "hongdae", "plateau"),
            "flight": ("redeye", "window", "aisle", "morning"),
        },
        fragments=3,
    ),
    FactTemplate(
        category="food",
        attribute_prefix="pizza",
        anchor_prefix="pizza_plan",
        fields=("style", "topping", "oven"),
        pools={
            "style": ("thin_crust", "detroit", "neapolitan", "sicilian"),
            "topping": ("basil", "pepperoni", "mushroom", "anchovy"),
            "oven": ("wood_fire", "cast_iron", "steel", "stone"),
        },
        fragments=3,
    ),
    FactTemplate(
        category="music",
        attribute_prefix="playlist",
        anchor_prefix="vinyl_session",
        fields=("artist", "album", "mood"),
        pools={
            "artist": ("khruangbin", "sault", "phoebe_bridgers", "the_beths"),
            "album": ("mordechai", "untitled", "punisher", "expert_in_a_dying_field"),
            "mood": ("late_night", "focus", "weekend", "sunrise"),
        },
        fragments=3,
    ),
    FactTemplate(
        category="health",
        attribute_prefix="training",
        anchor_prefix="half_marathon",
        fields=("distance", "pace", "recovery"),
        pools={
            "distance": ("18k", "20k", "22k", "24k"),
            "pace": ("steady", "tempo", "easy", "progression"),
            "recovery": ("electrolytes", "stretch", "ice_bath", "sleep"),
        },
        fragments=3,
    ),
)

ONE_SHOT_TEMPLATES: tuple[FactTemplate, ...] = (
    FactTemplate(
        category="work",
        attribute_prefix="project",
        anchor_prefix="launch_pad",
        fields=("deadline", "owner", "status"),
        pools={
            "deadline": ("friday", "monday", "wednesday", "next_week"),
            "owner": ("maya", "julian", "sofia", "owen"),
            "status": ("green", "blocked", "review", "shipping"),
        },
        fragments=1,
    ),
    FactTemplate(
        category="food",
        attribute_prefix="coffee",
        anchor_prefix="coffee_order",
        fields=("drink", "milk", "size"),
        pools={
            "drink": ("latte", "cortado", "americano", "flat_white"),
            "milk": ("oat", "whole", "almond", "none"),
            "size": ("small", "medium", "large", "double"),
        },
        fragments=1,
    ),
)

_PROMPT_OLD_PROFILE_RE = re.compile(
    r"<OLD_PROFILE>\s*(?P<profile>\{.*?\})\s*</OLD_PROFILE>", re.DOTALL
)
_PROMPT_HISTORY_RE = re.compile(r"<HISTORY>\s*(?P<history>.*?)\s*</HISTORY>", re.DOTALL)
_MEMO_RE = re.compile(r"(?P<anchor>[a-z0-9_]+)\s+memo:\s*(?P<body>[^\n]+)")


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
        return "synthetic-token-hash"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class SyntheticRuleLanguageModel(LanguageModel):
    """Deterministic extractor that rewards multi-message coherence."""

    def __init__(self, facts_by_anchor: dict[str, SyntheticFact]) -> None:
        self._facts_by_anchor = facts_by_anchor

    async def generate_parsed_response(
        self,
        output_format: type,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> Any:
        del system_prompt, max_attempts
        profile = self._extract_old_profile(user_prompt or "")
        slots_by_anchor = self._extract_history_slots(user_prompt or "")

        commands: list[dict[str, str]] = []
        for anchor, slots in slots_by_anchor.items():
            fact = self._facts_by_anchor.get(anchor)
            if fact is None:
                continue
            if not all(field in slots for field in fact.required_fields):
                continue

            category_profile = profile.get(fact.category, {})
            existing_value = category_profile.get(fact.attribute)
            if existing_value and existing_value != fact.expected_value:
                commands.append(
                    {
                        "command": CommandType.DELETE.value,
                        "category": fact.category,
                        "attribute": fact.attribute,
                        "value": existing_value,
                    }
                )

            commands.append(
                {
                    "command": CommandType.ADD.value,
                    "category": fact.category,
                    "attribute": fact.attribute,
                    "value": fact.expected_value,
                }
            )

        return TypeAdapter(output_format).validate_python({"commands": commands})

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        raise NotImplementedError

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any, int, int]:
        raise NotImplementedError

    @staticmethod
    def _extract_old_profile(prompt: str) -> dict[str, dict[str, str]]:
        match = _PROMPT_OLD_PROFILE_RE.search(prompt)
        if match is None:
            return {}
        return json.loads(match.group("profile"))

    @staticmethod
    def _extract_history_slots(prompt: str) -> dict[str, dict[str, str]]:
        match = _PROMPT_HISTORY_RE.search(prompt)
        if match is None:
            return {}
        history = match.group("history")
        slots_by_anchor: dict[str, dict[str, str]] = {}
        for memo in _MEMO_RE.finditer(history):
            anchor = memo.group("anchor")
            body = memo.group("body")
            for chunk in body.split("|"):
                chunk = chunk.strip().strip(".")
                if "=" not in chunk:
                    continue
                key, value = chunk.split("=", 1)
                slots_by_anchor.setdefault(anchor, {})[key.strip()] = value.strip()
        return slots_by_anchor


def build_dataset(*, cases: int, seed: int) -> SyntheticDataset:
    rng = Random(seed)
    facts: list[SyntheticFact] = []
    for index in range(cases):
        if index % 5 == 0:
            template = rng.choice(ONE_SHOT_TEMPLATES)
        else:
            template = rng.choice(MULTI_TURN_TEMPLATES)
        facts.append(_materialize_fact(template=template, index=index, rng=rng))

    events: list[Event] = []
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    max_fragments = max(len(fact.fragments) for fact in facts)
    offset = 0
    for fragment_index in range(max_fragments):
        for fact in facts:
            if fragment_index >= len(fact.fragments):
                continue
            events.append(
                Event(
                    uuid=uuid4(),
                    timestamp=base_time + timedelta(seconds=offset * 45),
                    body=Content(
                        context=MessageContext(source="user"),
                        items=[Text(text=fact.fragments[fragment_index])],
                    ),
                )
            )
            offset += 1
    return SyntheticDataset(events=tuple(events), facts=tuple(facts))


def _materialize_fact(
    *,
    template: FactTemplate,
    index: int,
    rng: Random,
) -> SyntheticFact:
    anchor = f"{template.anchor_prefix}_{index:03d}"
    attribute = anchor
    fields = {field: rng.choice(values) for field, values in template.pools.items()}
    expected_value = " ; ".join(f"{field}={fields[field]}" for field in template.fields)
    query = f"{anchor} " + " ".join(fields[field] for field in template.fields)
    if template.fragments == 1:
        fragments = (
            f"{anchor} memo: "
            + " | ".join(f"{field}={fields[field]}" for field in template.fields),
        )
    else:
        fragments = tuple(
            f"{anchor} memo: {field}={fields[field]}" for field in template.fields
        )
    return SyntheticFact(
        anchor=anchor,
        category=template.category,
        attribute=attribute,
        required_fields=template.fields,
        field_values=fields,
        expected_value=expected_value,
        query=query,
        fragments=fragments,
        clustered_case=template.fragments > 1,
    )


def build_schema() -> PartitionSchema:
    categories = sorted(
        {template.category for template in (*MULTI_TURN_TEMPLATES, *ONE_SHOT_TEMPLATES)}
    )
    return PartitionSchema(
        topics=(
            TopicDefinition(
                name="Profile",
                description=(
                    "You are processing synthetic benchmark conversations. "
                    "Only extract explicit facts from lines that contain "
                    "'memo:'. Do not infer anything. For each memo thread, use "
                    "the memo anchor (the token before 'memo:') as the attribute "
                    "name, and use the value field to store the exact semicolon-"
                    "separated key=value facts seen across the batch. When later "
                    "messages for the same anchor add more key=value fragments, "
                    "update the existing value so it contains the complete set."
                ),
                categories=tuple(
                    CategoryDefinition(
                        name=name,
                        description=(
                            f"Explicit synthetic memo facts for {name}. "
                            "Only record memo anchors and literal key=value facts."
                        ),
                    )
                    for name in categories
                ),
            ),
        )
    )


async def run_mode(
    *,
    mode: str,
    clustering_enabled: bool,
    dataset: SyntheticDataset,
    client: AsyncQdrantClient,
    namespace: str,
    collection_name: str,
    sqlite_path: Path,
    keep_artifacts: bool,
    language_model: LanguageModel,
) -> ModeMetrics:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}")
    semantic_store = SQLAlchemySemanticStore(
        SQLAlchemySemanticStoreParams(engine=engine)
    )
    await semantic_store.startup()

    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=client))
    config = VectorStoreCollectionConfig(
        vector_dimensions=64,
        similarity_metric=SimilarityMetric.COSINE,
        properties_schema=AttributeMemory.expected_vector_store_collection_schema(),
    )
    collection = await vector_store.open_or_create_collection(
        namespace=namespace,
        name=collection_name,
        config=config,
    )

    partition_key = _slug(f"{mode}_{uuid4().hex[:8]}")
    partition = await semantic_store.open_or_create_partition(partition_key)
    memory = AttributeMemory(
        partition=partition,
        vector_collection=collection,
        embedder=TokenHashEmbedder(),
        language_model=language_model,
        schema=build_schema(),
        clustering_config=ClusteringConfig(
            enabled=clustering_enabled,
            cluster_params=ClusterParams(similarity_threshold=0.55),
            trigger_messages=1,
            trigger_age=timedelta(seconds=0),
            idle_ttl=None,
            max_clusters_per_run=1000,
            consolidation_threshold=0,
        ),
    )

    try:
        await memory.ingest(dataset.events)

        stored = {
            (attr.category, attr.attribute, attr.value)
            async for attr in partition.list_attributes()
        }
        expected = {
            (fact.category, fact.attribute, fact.expected_value)
            for fact in dataset.facts
        }
        true_positives = len(stored & expected)
        precision = _safe_div(true_positives, len(stored))
        recall = _safe_div(true_positives, len(expected))
        f1 = _safe_div(2 * precision * recall, precision + recall)

        hit_at_1 = 0
        hit_at_3 = 0
        clustered_hits = 0
        clustered_total = 0
        one_shot_hits = 0
        one_shot_total = 0
        for fact in dataset.facts:
            retrieved = await memory.retrieve(fact.query, top_k=3)
            triples = {
                (attr.category, attr.attribute, attr.value)
                for attr, _score in retrieved
            }
            target = (fact.category, fact.attribute, fact.expected_value)
            if target in triples:
                hit_at_3 += 1
                if fact.clustered_case:
                    clustered_hits += 1
                else:
                    one_shot_hits += 1
            if retrieved:
                first = retrieved[0][0]
                if (first.category, first.attribute, first.value) == target:
                    hit_at_1 += 1
            if fact.clustered_case:
                clustered_total += 1
            else:
                one_shot_total += 1

        return ModeMetrics(
            mode=mode,
            partition_key=partition_key,
            namespace=namespace,
            collection_name=collection_name,
            sqlite_path=str(sqlite_path),
            extracted=len(stored),
            expected=len(expected),
            true_positives=true_positives,
            precision=precision,
            recall=recall,
            f1=f1,
            retrieval_recall_at_1=_safe_div(hit_at_1, len(dataset.facts)),
            retrieval_recall_at_3=_safe_div(hit_at_3, len(dataset.facts)),
            clustered_case_recall=_safe_div(clustered_hits, clustered_total),
            one_shot_case_recall=_safe_div(one_shot_hits, one_shot_total),
        )
    finally:
        await semantic_store.shutdown()
        await engine.dispose()
        if not keep_artifacts:
            await vector_store.delete_collection(
                namespace=namespace, name=collection_name
            )
            _delete_sqlite(sqlite_path)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _delete_sqlite(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_evaluation_dotenvs() -> None:
    repo_root = _repo_root()
    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(repo_root / "evaluation" / ".env", override=True)


def build_language_model(
    *,
    llm_backend: str,
    facts_by_anchor: dict[str, SyntheticFact],
    openai_model: str,
    openai_base_url: str | None,
    openai_reasoning_effort: str | None,
) -> LanguageModel:
    if llm_backend == LLM_BACKEND_SYNTHETIC:
        return SyntheticRuleLanguageModel(facts_by_anchor)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to the repo-root .env or "
            "evaluation/.env, or pass --llm-backend synthetic."
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


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    if not cleaned:
        cleaned = "eval"
    return cleaned[:32]


async def async_main(args: argparse.Namespace) -> None:
    load_evaluation_dotenvs()
    dataset = build_dataset(cases=args.cases, seed=args.seed)
    facts_by_anchor = {fact.anchor: fact for fact in dataset.facts}
    client = AsyncQdrantClient(
        host=args.qdrant_host,
        port=args.qdrant_port,
        grpc_port=args.qdrant_grpc_port,
        prefer_grpc=args.prefer_grpc,
    )

    run_suffix = uuid4().hex[:8]
    base_dir = Path(args.sqlite_dir)
    clustered = await run_mode(
        mode="clustered",
        clustering_enabled=True,
        dataset=dataset,
        client=client,
        namespace=_slug(f"aeval_{run_suffix}_on"),
        collection_name=_slug(f"run_{run_suffix}_on"),
        sqlite_path=base_dir / f"attribute_memory_{run_suffix}_on.db",
        keep_artifacts=args.keep_artifacts,
        language_model=build_language_model(
            llm_backend=args.llm_backend,
            facts_by_anchor=facts_by_anchor,
            openai_model=args.openai_model,
            openai_base_url=args.openai_base_url,
            openai_reasoning_effort=args.openai_reasoning_effort,
        ),
    )
    non_clustered = await run_mode(
        mode="non_clustered",
        clustering_enabled=False,
        dataset=dataset,
        client=client,
        namespace=_slug(f"aeval_{run_suffix}_off"),
        collection_name=_slug(f"run_{run_suffix}_off"),
        sqlite_path=base_dir / f"attribute_memory_{run_suffix}_off.db",
        keep_artifacts=args.keep_artifacts,
        language_model=build_language_model(
            llm_backend=args.llm_backend,
            facts_by_anchor=facts_by_anchor,
            openai_model=args.openai_model,
            openai_base_url=args.openai_base_url,
            openai_reasoning_effort=args.openai_reasoning_effort,
        ),
    )
    await client.close()

    report = {
        "dataset": {
            "cases": len(dataset.facts),
            "events": len(dataset.events),
            "seed": args.seed,
            "clustered_cases": sum(1 for fact in dataset.facts if fact.clustered_case),
            "one_shot_cases": sum(
                1 for fact in dataset.facts if not fact.clustered_case
            ),
        },
        "llm_backend": args.llm_backend,
        "openai_model": args.openai_model,
        "modes": [
            clustered.__dict__,
            non_clustered.__dict__,
        ],
        "delta": {
            "extraction_recall": clustered.recall - non_clustered.recall,
            "retrieval_recall_at_1": (
                clustered.retrieval_recall_at_1 - non_clustered.retrieval_recall_at_1
            ),
            "retrieval_recall_at_3": (
                clustered.retrieval_recall_at_3 - non_clustered.retrieval_recall_at_3
            ),
            "clustered_case_recall": (
                clustered.clustered_case_recall - non_clustered.clustered_case_recall
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    load_evaluation_dotenvs()
    parser = argparse.ArgumentParser(
        description="Evaluate AttributeMemory with and without clustering.",
    )
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--llm-backend",
        choices=[
            LLM_BACKEND_OPENAI_RESPONSES,
            LLM_BACKEND_OPENAI_CHAT,
            LLM_BACKEND_SYNTHETIC,
        ],
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
    parser.add_argument("--sqlite-dir", default="/tmp")
    parser.add_argument("--keep-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
