import argparse
import asyncio
import json
import logging
import os
import re
import time

import boto3
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    load_longmemeval_dataset,
)
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import QueryResult
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

# Generic v2f cue-gen prompt — corpus-agnostic. Does not prescribe speaker
# roles or cue counts beyond "2". The LLM infers the embedded turn format
# from the primer context and mirrors it in its cues.
V2F_PROMPT = """\
You are generating search text for semantic retrieval over a conversation \
history. Your cues will be embedded and compared via cosine similarity.

Question: {question}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this \
search going? What kind of content is still missing? Should you search for \
similar content or pivot to a different topic?

If the question implies MULTIPLE items or asks "all/every", keep searching \
for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific \
vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would \
actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else."""

_CUE_LINE_RE = re.compile(
    r"^\s*CUE\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_cues(response: str, max_cues: int) -> list[str]:
    cues: list[str] = []
    for m in _CUE_LINE_RE.finditer(response):
        raw = m.group(1).strip().strip('"').strip()
        if raw:
            cues.append(raw)
        if len(cues) >= max_cues:
            break
    return cues


def _format_primer_context(
    segments: list[dict], *, max_items: int = 12, max_len: int = 250
) -> str:
    if not segments:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    sorted_segs = sorted(segments, key=lambda s: s["turn_id"])
    lines = [
        f"[Turn {s['turn_id']}, {s['role']}]: {s['text'][:max_len]}"
        for s in sorted_segs[:max_items]
    ]
    return "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + "\n".join(lines)


def _extract_primer_segments(
    query_result: QueryResult, max_items: int = 10
) -> list[dict]:
    out: list[dict] = []
    seen: set[int] = set()
    for sc in query_result.scored_segment_contexts:
        for seg in sc.segments:
            props = seg.properties or {}
            turn_id_raw = props.get("turn_id", -1)
            try:
                turn_id = int(turn_id_raw)
            except (TypeError, ValueError):
                turn_id = -1
            if turn_id in seen:
                continue
            seen.add(turn_id)
            out.append(
                {
                    "turn_id": turn_id,
                    "role": str(props.get("role", "")),
                    "text": seg.block.text,
                }
            )
            if len(out) >= max_items:
                return out
    return out


RRF_K = 60  # standard reciprocal-rank-fusion constant


def _merge_max(batches: list[QueryResult]) -> list:
    best: dict = {}
    for qr in batches:
        for sc in qr.scored_segment_contexts:
            prev = best.get(sc.seed_segment_uuid)
            if prev is None or sc.score > prev.score:
                best[sc.seed_segment_uuid] = sc
    return sorted(best.values(), key=lambda sc: -sc.score)


def _merge_rrf(batches: list[QueryResult], *, k: int = RRF_K) -> list:
    scores: dict = {}
    repr_sc: dict = {}
    for qr in batches:
        for rank, sc in enumerate(qr.scored_segment_contexts):
            scores[sc.seed_segment_uuid] = scores.get(
                sc.seed_segment_uuid, 0.0
            ) + 1.0 / (k + rank + 1)
            if sc.seed_segment_uuid not in repr_sc:
                repr_sc[sc.seed_segment_uuid] = sc
    ranked_uuids = sorted(scores, key=lambda u: -scores[u])
    # Rebuild ScoredSegmentContext with the RRF score so downstream sees a
    # consistent scoring for the merged list.
    out = []
    for u in ranked_uuids:
        sc = repr_sc[u]
        out.append(sc.model_copy(update={"score": scores[u]}))
    return out


def _merge_sum(batches: list[QueryResult]) -> list:
    scores: dict = {}
    repr_sc: dict = {}
    for qr in batches:
        seen_in_batch: set = set()
        for sc in qr.scored_segment_contexts:
            if sc.seed_segment_uuid in seen_in_batch:
                continue
            seen_in_batch.add(sc.seed_segment_uuid)
            scores[sc.seed_segment_uuid] = (
                scores.get(sc.seed_segment_uuid, 0.0) + sc.score
            )
            if sc.seed_segment_uuid not in repr_sc:
                repr_sc[sc.seed_segment_uuid] = sc
    ranked_uuids = sorted(scores, key=lambda u: -scores[u])
    out = []
    for u in ranked_uuids:
        sc = repr_sc[u]
        out.append(sc.model_copy(update={"score": scores[u]}))
    return out


_MERGE_STRATEGIES = {
    "max": _merge_max,
    "rrf": _merge_rrf,
    "sum": _merge_sum,
}


async def _v2f_cue_gen_query(
    memory: EventMemory,
    question_text: str,
    *,
    vector_search_limit: int,
    expand_context: int,
    openai_client: AsyncOpenAI,
    cue_gen_model: str,
    max_cues: int = 2,
    merge_strategy: str = "rrf",
) -> tuple[QueryResult, dict]:
    """Multi-probe retrieval with generic v2f cue generation.

    1. Primer: query with the raw question, limit=10, expand_context=0.
    2. Format top-10 primer hits as context so the LLM can observe the
       embedded turn format and mirror it.
    3. Generate `max_cues` cues via the cue-gen LLM.
    4. Run primer + cue queries in parallel at vector_search_limit=max(K, 20).
    5. Merge by the chosen strategy (rrf/max/sum), take top-K.

    Merge strategy: RRF is the default. It is more robust at small K than
    max-score (one garbage cue hit can't eject real gold via rank ordering),
    and all four strategies converge at K>=50 per the ensemble study.
    """
    primer_result = await memory.query(
        query=question_text,
        vector_search_limit=10,
        expand_context=0,
    )
    primer_segs = _extract_primer_segments(primer_result, max_items=10)
    context_section = _format_primer_context(primer_segs, max_items=12, max_len=250)

    prompt = V2F_PROMPT.format(question=question_text, context_section=context_section)
    resp = await openai_client.chat.completions.create(
        model=cue_gen_model,
        messages=[{"role": "user", "content": prompt}],
    )
    cue_gen_raw = resp.choices[0].message.content or ""
    cues = _parse_cues(cue_gen_raw, max_cues=max_cues)

    effective_limit = max(vector_search_limit, 20)
    queries = [question_text, *cues]

    batches = await asyncio.gather(
        *[
            memory.query(
                query=q,
                vector_search_limit=effective_limit,
                expand_context=expand_context,
            )
            for q in queries
        ]
    )

    merge_fn = _MERGE_STRATEGIES[merge_strategy]
    merged = merge_fn(batches)[:vector_search_limit]

    meta = {
        "cues": cues,
        "cue_gen_raw": cue_gen_raw,
        "num_queries": len(queries),
        "merge_strategy": merge_strategy,
    }
    return QueryResult(scored_segment_contexts=merged), meta


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    parser.add_argument(
        "--vector-search-limit",
        type=int,
        default=100,
        help="Number of vectors to retrieve (default: 100)",
    )
    parser.add_argument(
        "--expand-context",
        type=int,
        default=0,
        help="Number of context segments to expand (default: 0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent questions (default: 4)",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable the reranker and rank by embedding similarity only",
    )
    parser.add_argument(
        "--v2f-cue-gen",
        action="store_true",
        help=(
            "Use v2f multi-probe cue generation "
            "(corpus-agnostic: primer + LLM-generated cues merged across queries)."
        ),
    )
    parser.add_argument(
        "--cue-gen-model",
        default="gpt-5-mini",
        help="LLM model for cue generation when --v2f-cue-gen is set (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--cue-count",
        type=int,
        default=2,
        help=(
            "Number of LLM-generated cues when --v2f-cue-gen is set (default: 2). "
            "Total vector queries = 1 primer (raw question) + cue_count cues."
        ),
    )
    parser.add_argument(
        "--cue-merge",
        choices=sorted(_MERGE_STRATEGIES.keys()),
        default="rrf",
        help=(
            "Strategy to merge primer + cue query results: rrf (default, "
            "robust at small K), max (best score wins), sum (sum of cosine "
            "scores across batches)."
        ),
    )
    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

    all_questions = load_longmemeval_dataset(data_path)

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    engine = create_async_engine(os.getenv("SQL_URL"))
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
        )
    )

    if args.no_reranker:
        reranker = None
    else:
        region = "us-west-2"
        aws_client = boto3.client(
            "bedrock-agent-runtime",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        reranker = AmazonBedrockReranker(
            AmazonBedrockRerankerParams(
                client=aws_client,
                region=region,
                model_id="cohere.rerank-v3-5:0",
            )
        )

    async def process_question(question: LongMemEvalItem):
        partition_key = question.question_id

        collection = await vector_store.open_collection(
            namespace="longmemeval", name=question.question_id
        )

        segment_store_partition = await segment_store.open_or_create_partition(
            partition_key
        )

        memory = EventMemory(
            EventMemoryParams(
                vector_store_collection=collection,
                segment_store_partition=segment_store_partition,
                embedder=embedder,
                reranker=reranker,
            )
        )

        memory_start = time.monotonic()
        cue_gen_meta: dict | None = None
        if args.v2f_cue_gen:
            query_result, cue_gen_meta = await _v2f_cue_gen_query(
                memory,
                question.question,
                vector_search_limit=args.vector_search_limit,
                expand_context=args.expand_context,
                openai_client=openai_client,
                cue_gen_model=args.cue_gen_model,
                max_cues=args.cue_count,
                merge_strategy=args.cue_merge,
            )
        else:
            search_query = f"User: {question.question}"
            query_result = await memory.query(
                query=search_query,
                vector_search_limit=args.vector_search_limit,
                expand_context=args.expand_context,
            )
        memory_latency = time.monotonic() - memory_start

        print(
            f"Question ID: {question.question_id}\n"
            f"Question: {question.question}\n"
            f"Question Type: {question.question_type}\n"
            f"Memory retrieval time: {memory_latency:.2f} seconds\n"
        )

        result = {
            "question_id": question.question_id,
            "question_date": question.question_date,
            "question": question.question,
            "answer": question.answer,
            "answer_turn_indices": question.answer_turn_indices,
            "question_type": question.question_type.value,
            "abstention": question.abstention_question,
            "memory_latency": memory_latency,
            "query_result": query_result.model_dump(mode="json"),
        }
        if cue_gen_meta is not None:
            result["cue_gen"] = cue_gen_meta

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

        return result

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        async_with(
            semaphore,
            process_question(question),
        )
        for question in all_questions
    ]
    results = await asyncio.gather(*tasks)

    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)

    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig()
    logging.getLogger(
        "memmachine_server.episodic_memory.event_memory.event_memory"
    ).setLevel(logging.DEBUG)
    asyncio.run(main())
