"""Unified evaluation for:

  Part A (LoCoMo-30 topic+summary stacked dual-view):
    - em_cosine_baseline_topicsumm
    - em_v2f_topicsumm
    - em_v2f_topicsumm_sf_spkfilter

  Part B (LME-hard POC turn-summary dual-view):
    - em_cosine_baseline_summ_lme
    - em_v2f_lme_mixed_7030_expand3_summ

Writes dedicated caches:
  cache/compose_topicsumm_v2f_cache.json
  cache/compose_topicsumm_v2f_sf_cache.json
  cache/summlme_v2f_mixed7030_cache.json

Does NOT modify framework files or prior em_*.py / em_setup_*.py files.

Run Part A:
    uv run python evaluation/associative_recall/compose_eval.py \
        --part a \
        --variants em_cosine_baseline_topicsumm,em_v2f_topicsumm,em_v2f_topicsumm_sf_spkfilter

Run Part B POC:
    uv run python evaluation/associative_recall/compose_eval.py \
        --part b \
        --variants em_cosine_baseline_summ_lme,em_v2f_lme_mixed_7030_expand3_summ
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import openai
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)

from em_architectures import (
    V2F_MODEL,
    V2F_PROMPT,
    _MergedLLMCache,
    format_primer_context,
    parse_v2f_cues,
)
from em_retuned_cue_gen import (
    build_v2f_speakerformat_prompt,
    parse_cues as parse_retuned_cues,
)
from em_two_speaker import classify_speaker_side, load_two_speaker_map
from em_lme_tuned_cues import (
    V2F_LME_MIXED_7030_PROMPT,
    parse_speaker_cues,
)


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

ASSOC_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSOC_DIR / "data"
RESULTS_DIR = ASSOC_DIR / "results"
CACHE_DIR = ASSOC_DIR / "cache"

# Part A caches.
TOPICSUMM_V2F_CACHE = CACHE_DIR / "compose_topicsumm_v2f_cache.json"
TOPICSUMM_V2F_SF_CACHE = CACHE_DIR / "compose_topicsumm_v2f_sf_cache.json"
# Part B caches.
LMESUMM_MIXED7030_CACHE = CACHE_DIR / "summlme_v2f_mixed7030_cache.json"

BUDGETS = (20, 50)
OVERSAMPLE = 2  # Dual-view index has 2 entries per turn.

# Part A.
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")
TOPICSUMM_COLLECTIONS_JSON = RESULTS_DIR / "eventmemory_topicsumm_collections.json"

# Part B.
LMESUMM_COLLECTIONS_JSON = RESULTS_DIR / "eventmemory_lmesumm_collections.json"
HARD_QUESTIONS_JSON = DATA_DIR / "questions_longmemeval_hard.json"

# Part B expand_context.
LME_EXPAND_CONTEXT = 3


@dataclass
class DualEMHit:
    turn_id: int
    score: float
    seed_segment_uuid: object
    role: str
    text: str
    view: str  # "raw" | "summary" | "unknown"


# =========================================================================
# Shared helpers (dual-view)
# =========================================================================


async def _query_em_dual(
    memory: EventMemory,
    text: str,
    *,
    vector_search_limit: int,
    expand_context: int = 0,
    property_filter=None,
) -> list[DualEMHit]:
    qr = await memory.query(
        query=text,
        vector_search_limit=vector_search_limit,
        expand_context=expand_context,
        property_filter=property_filter,
    )
    hits: list[DualEMHit] = []
    for sc in qr.scored_segment_contexts:
        for seg in sc.segments:
            hits.append(
                DualEMHit(
                    turn_id=int(seg.properties.get("turn_id", -1)),
                    score=sc.score,
                    seed_segment_uuid=sc.seed_segment_uuid,
                    role=str(seg.properties.get("role", "")),
                    text=seg.block.text,
                    view=str(seg.properties.get("view", "unknown")),
                )
            )
    return hits


def _dedupe_dual_by_turn_id(hits: list[DualEMHit]) -> list[DualEMHit]:
    seen: set[int] = set()
    out: list[DualEMHit] = []
    for h in hits:
        if h.turn_id in seen:
            continue
        seen.add(h.turn_id)
        out.append(h)
    return out


def _merge_dual_by_max_score(
    batches: list[list[DualEMHit]],
) -> list[DualEMHit]:
    best: dict[int, DualEMHit] = {}
    for batch in batches:
        seen_in_batch: set[int] = set()
        for h in batch:
            if h.turn_id in seen_in_batch:
                continue
            seen_in_batch.add(h.turn_id)
            prev = best.get(h.turn_id)
            if prev is None or h.score > prev.score:
                best[h.turn_id] = h
    return sorted(best.values(), key=lambda h: -h.score)


def _view_coverage(
    hits: list[DualEMHit], gold: set[int], K: int
) -> dict:
    topk = hits[:K]
    per_turn: dict[int, str] = {}
    for h in topk:
        if h.turn_id in gold and h.turn_id not in per_turn:
            per_turn[h.turn_id] = h.view
    return {str(tid): per_turn[tid] for tid in sorted(per_turn.keys())}


async def _llm_call(
    openai_client, prompt: str, cache: _MergedLLMCache, model: str = V2F_MODEL
) -> tuple[str, bool]:
    cached = cache.get(model, prompt)
    if cached is not None:
        return cached, True
    resp = await openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    cache.put(model, prompt, text)
    return text, False


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


# =========================================================================
# Part A variants (LoCoMo topic+summary)
# =========================================================================


async def em_cosine_baseline_topicsumm(
    memory: EventMemory, question: str, *, K: int
) -> list[DualEMHit]:
    raw = await _query_em_dual(
        memory, question, vector_search_limit=K * OVERSAMPLE, expand_context=0
    )
    return _dedupe_dual_by_turn_id(raw)[:K]


async def em_v2f_topicsumm(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[DualEMHit], dict]:
    # Primer at K=10.
    primer = _dedupe_dual_by_turn_id(
        await _query_em_dual(
            memory, question, vector_search_limit=10 * OVERSAMPLE, expand_context=0
        )
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer
    ]
    context_section = format_primer_context(primer_segments)
    prompt = V2F_PROMPT.format(question=question, context_section=context_section)
    raw, hit = await _llm_call(openai_client, prompt, cache)
    cues = parse_v2f_cues(raw, max_cues=2)

    batches = [
        await _query_em_dual(
            memory, question, vector_search_limit=K * OVERSAMPLE, expand_context=0
        )
    ]
    for cue in cues[:2]:
        batches.append(
            await _query_em_dual(
                memory, cue, vector_search_limit=K * OVERSAMPLE, expand_context=0
            )
        )
    merged = _merge_dual_by_max_score(batches)[:K]
    return merged, {"cues": cues, "cache_hit": hit}


async def em_v2f_topicsumm_sf(
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[DualEMHit], dict]:
    """v2f_speakerformat variant on dual-view topicsumm index."""
    primer = _dedupe_dual_by_turn_id(
        await _query_em_dual(
            memory, question, vector_search_limit=10 * OVERSAMPLE, expand_context=0
        )
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer
    ]
    context_section = format_primer_context(primer_segments)
    p_user, p_asst = participants
    prompt = build_v2f_speakerformat_prompt(
        question, context_section, p_user, p_asst
    )
    raw, hit = await _llm_call(openai_client, prompt, cache)
    cues = parse_retuned_cues(raw, max_cues=2)

    batches = [
        await _query_em_dual(
            memory, question, vector_search_limit=K * OVERSAMPLE, expand_context=0
        )
    ]
    for cue in cues[:2]:
        batches.append(
            await _query_em_dual(
                memory, cue, vector_search_limit=K * OVERSAMPLE, expand_context=0
            )
        )
    merged = _merge_dual_by_max_score(batches)[:K]
    return merged, {"cues": cues, "cache_hit": hit}


async def em_v2f_topicsumm_sf_spkfilter(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    participants: tuple[str, str],
    speaker_map: dict[str, dict[str, str]],
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[DualEMHit], dict]:
    """v2f_topicsumm_sf + property_filter on speaker when query names one."""
    base_hits, meta = await em_v2f_topicsumm_sf(
        memory, question, participants, K=max(K, 50),
        cache=cache, openai_client=openai_client,
    )
    side, user_name, asst_name, name_tokens = classify_speaker_side(
        question, conversation_id, speaker_map
    )
    meta = {
        **meta,
        "matched_side": side,
        "conv_user_name": user_name,
        "conv_assistant_name": asst_name,
        "query_name_tokens": name_tokens,
        "applied_speaker_filter": False,
    }
    if side not in ("user", "assistant"):
        return base_hits[:K], meta

    matched_name = user_name if side == "user" else asst_name
    matched_role = side
    meta["applied_speaker_filter"] = True
    meta["matched_name"] = matched_name

    prop_filter = Comparison(field="context.source", op="=", value=matched_name)
    filtered_hits = _dedupe_dual_by_turn_id(
        await _query_em_dual(
            memory, question,
            vector_search_limit=(K + 10) * OVERSAMPLE,
            expand_context=0,
            property_filter=prop_filter,
        )
    )

    kept = [h for h in base_hits if h.role == matched_role]
    seen = {h.turn_id for h in kept}
    appended: list[DualEMHit] = []
    for h in filtered_hits:
        if h.turn_id in seen:
            continue
        appended.append(h)
        seen.add(h.turn_id)
        if len(appended) >= 15:
            break
    merged = kept + appended
    meta["appended_turn_ids"] = [h.turn_id for h in appended]
    return merged[:K], meta


# =========================================================================
# Part B variants (LME-hard dual-view)
# =========================================================================


USER_PREFIX = "User: "


def _ensure_user_prefix(text: str) -> str:
    t = text.lstrip()
    low = t.lower()
    if low.startswith("user:") or low.startswith("assistant:"):
        return t
    return USER_PREFIX + t


async def em_cosine_baseline_summ_lme(
    memory: EventMemory, question: str, *, K: int
) -> list[DualEMHit]:
    prefixed = _ensure_user_prefix(question)
    raw = await _query_em_dual(
        memory,
        prefixed,
        vector_search_limit=K * OVERSAMPLE,
        expand_context=LME_EXPAND_CONTEXT,
    )
    return _dedupe_dual_by_turn_id(raw)[:K]


async def em_v2f_lme_mixed_7030_expand3_summ(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[DualEMHit], dict]:
    """LME's best single-shot recipe + dual-view summary index."""
    # Primer with User: prefix, expand_context=3.
    prefixed_q = _ensure_user_prefix(question)
    primer = _dedupe_dual_by_turn_id(
        await _query_em_dual(
            memory,
            prefixed_q,
            vector_search_limit=10 * OVERSAMPLE,
            expand_context=0,
        )
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer
    ]
    context_section = format_primer_context(primer_segments)
    prompt = V2F_LME_MIXED_7030_PROMPT.format(
        question=question, context_section=context_section
    )
    raw, hit = await _llm_call(openai_client, prompt, cache)
    cues = parse_speaker_cues(raw, max_cues=3)

    vsl = max(K, 20) * OVERSAMPLE
    primer_batch = await _query_em_dual(
        memory,
        prefixed_q,
        vector_search_limit=vsl,
        expand_context=LME_EXPAND_CONTEXT,
    )
    cue_batches: list[list[DualEMHit]] = []
    for cue in cues:
        cue_text = _ensure_user_prefix(cue)
        cue_batches.append(
            await _query_em_dual(
                memory,
                cue_text,
                vector_search_limit=vsl,
                expand_context=LME_EXPAND_CONTEXT,
            )
        )
    merged = _merge_dual_by_max_score([primer_batch, *cue_batches])[:K]
    return merged, {"cues": cues, "cache_hit": hit}


# =========================================================================
# Evaluation driver
# =========================================================================


def load_locomo_questions() -> list[dict]:
    with open(DATA_DIR / "questions_extended.json") as f:
        qs = json.load(f)
    return [q for q in qs if q.get("benchmark") == "locomo"][:30]


def load_lme_poc_questions(poc_ids: list[str]) -> list[dict]:
    with open(HARD_QUESTIONS_JSON) as f:
        all_hard = json.load(f)
    by_id = {q["question_id"]: q for q in all_hard}
    return [by_id[qid] for qid in poc_ids if qid in by_id]


async def run_part_a(variants: list[str], args) -> dict:
    """Part A: LoCoMo-30 topicsumm variants."""
    with open(TOPICSUMM_COLLECTIONS_JSON) as f:
        collections_meta = json.load(f)
    questions = load_locomo_questions()
    if args.limit is not None:
        questions = questions[: args.limit]
    conv_to_meta = {r["conversation_id"]: r for r in collections_meta["conversations"]}
    speaker_map = load_two_speaker_map()

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sql_url = collections_meta.get("sql_url") or os.getenv("SQL_URL")
    if sql_url is None:
        raise RuntimeError("No sql_url in topicsumm collections meta")
    if sql_url.startswith("sqlite"):
        engine = create_async_engine(sql_url)
    else:
        engine = create_async_engine(sql_url, pool_size=20, max_overflow=20)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    v2f_cache = _MergedLLMCache(
        reader_paths=[TOPICSUMM_V2F_CACHE], writer_path=TOPICSUMM_V2F_CACHE,
    )
    v2f_sf_cache = _MergedLLMCache(
        reader_paths=[TOPICSUMM_V2F_SF_CACHE], writer_path=TOPICSUMM_V2F_SF_CACHE,
    )

    memories: dict[str, EventMemory] = {}
    participants_by_conv: dict[str, tuple[str, str]] = {}
    opened: list = []
    for conv_id in LOCOMO_CONV_IDS:
        meta = conv_to_meta[conv_id]
        coll = await vector_store.open_collection(
            namespace=meta["namespace"], name=meta["collection_name"]
        )
        part = await segment_store.open_or_create_partition(meta["partition_key"])
        mem = EventMemory(
            EventMemoryParams(
                vector_store_collection=coll,
                segment_store_partition=part,
                embedder=embedder,
                reranker=None,
                derive_sentences=False,
                max_text_chunk_length=500,
            )
        )
        memories[conv_id] = mem
        participants_by_conv[conv_id] = (meta["user_name"], meta["assistant_name"])
        opened.append((coll, part))

    max_K = max(BUDGETS)
    results: dict = {
        "variants": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
        "oversample": OVERSAMPLE,
        "collections": collections_meta,
    }

    try:
        for variant in variants:
            rows: list[dict] = []
            t_v = time.monotonic()
            for q in questions:
                cid = q["conversation_id"]
                mem = memories[cid]
                participants = participants_by_conv[cid]
                q_text = q["question"]
                gold = set(q.get("source_chat_ids", []))

                t0 = time.monotonic()
                meta_out: dict = {}
                if variant == "em_cosine_baseline_topicsumm":
                    hits = await em_cosine_baseline_topicsumm(
                        mem, q_text, K=max_K,
                    )
                elif variant == "em_v2f_topicsumm":
                    hits, meta_out = await em_v2f_topicsumm(
                        mem, q_text, K=max_K,
                        cache=v2f_cache, openai_client=openai_client,
                    )
                elif variant == "em_v2f_topicsumm_sf":
                    hits, meta_out = await em_v2f_topicsumm_sf(
                        mem, q_text, participants, K=max_K,
                        cache=v2f_sf_cache, openai_client=openai_client,
                    )
                elif variant == "em_v2f_topicsumm_sf_spkfilter":
                    hits, meta_out = await em_v2f_topicsumm_sf_spkfilter(
                        mem, q_text, cid, participants, speaker_map,
                        K=max_K, cache=v2f_sf_cache, openai_client=openai_client,
                    )
                else:
                    raise KeyError(variant)
                elapsed = time.monotonic() - t0

                row: dict = {
                    "conversation_id": cid,
                    "category": q.get("category", "unknown"),
                    "question_index": q.get("question_index", -1),
                    "question": q_text,
                    "gold_turn_ids": sorted(gold),
                    "n_hits": len(hits),
                    "time_s": round(elapsed, 3),
                }
                row.update(meta_out)
                for K in BUDGETS:
                    topk = hits[:K]
                    retrieved = {h.turn_id for h in topk}
                    row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
                    row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
                    row[f"view_coverage@{K}"] = _view_coverage(hits, gold, K)
                rows.append(row)
            elapsed = time.monotonic() - t_v
            n = len(rows)
            summary = {"n": n, "time_s": round(elapsed, 1)}
            for K in BUDGETS:
                summary[f"mean_r@{K}"] = round(
                    sum(r[f"r@{K}"] for r in rows) / max(n, 1), 4
                )
            by_cat: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by_cat[r.get("category", "unknown")].append(r)
            cat_summary: dict[str, dict] = {}
            for cat, cat_rows in by_cat.items():
                d = {"n": len(cat_rows)}
                for K in BUDGETS:
                    d[f"mean_r@{K}"] = round(
                        sum(r[f"r@{K}"] for r in cat_rows) / max(len(cat_rows), 1), 4
                    )
                cat_summary[cat] = d
            results["variants"][variant] = {
                "summary": summary,
                "by_category": cat_summary,
                "per_question": rows,
            }
            v2f_cache.save()
            v2f_sf_cache.save()
            print(
                f"[{variant}] n={summary['n']} "
                f"r@20={summary['mean_r@20']:.4f} "
                f"r@50={summary['mean_r@50']:.4f} "
                f"in {summary['time_s']:.1f}s",
                flush=True,
            )
    finally:
        for coll, part in opened:
            await segment_store.close_partition(part)
            await vector_store.close_collection(collection=coll)
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    # View coverage aggregate.
    coverage_stats: dict[str, dict] = {}
    for variant in variants:
        per_q = results["variants"][variant]["per_question"]
        tot_gold = 0
        raw_wins = 0
        summ_wins = 0
        unknown = 0
        for row in per_q:
            cov = row.get("view_coverage@50", {})
            for _tid, view in cov.items():
                tot_gold += 1
                if view == "summary":
                    summ_wins += 1
                elif view == "raw":
                    raw_wins += 1
                else:
                    unknown += 1
        coverage_stats[variant] = {
            "gold_credited_top50": tot_gold,
            "summary_wins_top50": summ_wins,
            "raw_wins_top50": raw_wins,
            "unknown_view_top50": unknown,
            "summary_share_top50": round(summ_wins / max(tot_gold, 1), 4),
        }
    results["view_coverage"] = coverage_stats
    return results


async def run_part_b(variants: list[str], args) -> dict:
    """Part B: LME-hard POC summ variants."""
    with open(LMESUMM_COLLECTIONS_JSON) as f:
        meta_all = json.load(f)
    poc_ids = meta_all["poc_question_ids"]
    questions = load_lme_poc_questions(poc_ids)
    if args.limit is not None:
        questions = questions[: args.limit]

    qid_to_meta = {r["question_id"]: r for r in meta_all["questions"]}

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sql_url = meta_all["sql_url"]
    if sql_url.startswith("sqlite"):
        engine = create_async_engine(sql_url)
    else:
        engine = create_async_engine(sql_url, pool_size=20, max_overflow=20)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mixed7030_cache = _MergedLLMCache(
        reader_paths=[LMESUMM_MIXED7030_CACHE], writer_path=LMESUMM_MIXED7030_CACHE,
    )

    memories: dict[str, EventMemory] = {}
    opened: list = []
    for q in questions:
        qm = qid_to_meta[q["question_id"]]
        coll = await vector_store.open_collection(
            namespace=qm["namespace"], name=qm["collection_name"]
        )
        part = await segment_store.open_or_create_partition(qm["partition_key"])
        mem = EventMemory(
            EventMemoryParams(
                vector_store_collection=coll,
                segment_store_partition=part,
                embedder=embedder,
                reranker=None,
                derive_sentences=False,
                max_text_chunk_length=500,
            )
        )
        memories[q["question_id"]] = mem
        opened.append((coll, part))

    max_K = max(BUDGETS)
    results: dict = {
        "variants": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
        "oversample": OVERSAMPLE,
        "expand_context": LME_EXPAND_CONTEXT,
        "collections": meta_all,
    }

    ARCH_CONCURRENCY = 8
    semaphore = asyncio.Semaphore(ARCH_CONCURRENCY)

    async def run_one_q(variant: str, q: dict) -> dict:
        async with semaphore:
            mem = memories[q["question_id"]]
            q_text = q["question"]
            gold = set(q.get("source_chat_ids", []))
            t0 = time.monotonic()
            meta_out: dict = {}
            if variant == "em_cosine_baseline_summ_lme":
                hits = await em_cosine_baseline_summ_lme(mem, q_text, K=max_K)
            elif variant == "em_v2f_lme_mixed_7030_expand3_summ":
                hits, meta_out = await em_v2f_lme_mixed_7030_expand3_summ(
                    mem, q_text, K=max_K,
                    cache=mixed7030_cache, openai_client=openai_client,
                )
            else:
                raise KeyError(variant)
            elapsed = time.monotonic() - t0
            row: dict = {
                "question_id": q["question_id"],
                "category": q.get("category", "unknown"),
                "question": q_text,
                "num_gold": len(gold),
                "n_hits": len(hits),
                "time_s": round(elapsed, 3),
            }
            row.update(meta_out)
            for K in BUDGETS:
                topk = hits[:K]
                retrieved = {h.turn_id for h in topk}
                row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
                row[f"view_coverage@{K}"] = _view_coverage(hits, gold, K)
            return row

    try:
        for variant in variants:
            t_v = time.monotonic()
            rows = await asyncio.gather(*(run_one_q(variant, q) for q in questions))
            elapsed = time.monotonic() - t_v
            n = len(rows)
            summary = {"n": n, "time_s": round(elapsed, 1)}
            for K in BUDGETS:
                summary[f"mean_r@{K}"] = round(
                    sum(r[f"r@{K}"] for r in rows) / max(n, 1), 4
                )
            by_cat: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by_cat[r.get("category", "unknown")].append(r)
            cat_summary: dict[str, dict] = {}
            for cat, cat_rows in by_cat.items():
                d = {"n": len(cat_rows)}
                for K in BUDGETS:
                    d[f"mean_r@{K}"] = round(
                        sum(r[f"r@{K}"] for r in cat_rows) / max(len(cat_rows), 1), 4
                    )
                cat_summary[cat] = d
            results["variants"][variant] = {
                "summary": summary,
                "by_category": cat_summary,
                "per_question": rows,
            }
            mixed7030_cache.save()
            cat_str = " ".join(
                f"{c}={cat_summary[c].get('mean_r@50', 0):.3f}"
                for c in sorted(cat_summary)
            )
            print(
                f"[{variant}] n={n} "
                f"r@20={summary['mean_r@20']:.4f} "
                f"r@50={summary['mean_r@50']:.4f} ({cat_str}) "
                f"in {summary['time_s']:.1f}s",
                flush=True,
            )
    finally:
        for coll, part in opened:
            await segment_store.close_partition(part)
            await vector_store.close_collection(collection=coll)
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    # View coverage aggregate per category.
    coverage_stats: dict[str, dict] = {}
    for variant in variants:
        per_q = results["variants"][variant]["per_question"]
        tot_gold = 0
        raw_wins = 0
        summ_wins = 0
        unknown = 0
        for row in per_q:
            cov = row.get("view_coverage@50", {})
            for _tid, view in cov.items():
                tot_gold += 1
                if view == "summary":
                    summ_wins += 1
                elif view == "raw":
                    raw_wins += 1
                else:
                    unknown += 1
        coverage_stats[variant] = {
            "gold_credited_top50": tot_gold,
            "summary_wins_top50": summ_wins,
            "raw_wins_top50": raw_wins,
            "unknown_view_top50": unknown,
            "summary_share_top50": round(summ_wins / max(tot_gold, 1), 4),
        }
    results["view_coverage"] = coverage_stats
    return results


# =========================================================================
# Main
# =========================================================================


def _build_md_part_a(results: dict) -> list[str]:
    md = [
        "# LoCoMo-30 topic+summary stacked dual-view (Part A)",
        "",
        "## References",
        "",
        "| Variant | R@20 | R@50 |",
        "| --- | --- | --- |",
        "| em_v2f_topic (topic-only ingest)              | 0.8333 | 0.9333 |",
        "| em_v2f_summ (summary-only dual-view)          | 0.9083 | 0.9167 |",
        "| em_v2f_summ_sf_spkfilter (summary-only + spkf)| 0.8917 | 0.9417 |",
        "| em_topic_plus_speaker_filter (topic-only)     | 0.8667 | 0.9333 |",
        "",
        "## Stacked (topic + summary) recall",
        "",
        "| Variant | R@20 | R@50 | time (s) |",
        "| --- | --- | --- | --- |",
    ]
    for variant, data in results["variants"].items():
        s = data["summary"]
        md.append(
            f"| `{variant}` | {s['mean_r@20']:.4f} | "
            f"{s['mean_r@50']:.4f} | {s['time_s']:.1f} |"
        )

    md += [
        "",
        "## View coverage (top-50, gold-credited)",
        "",
        "| Variant | gold credited | raw wins | summary wins | summary share |",
        "| --- | --- | --- | --- | --- |",
    ]
    for variant, c in results.get("view_coverage", {}).items():
        md.append(
            f"| `{variant}` | {c['gold_credited_top50']} | "
            f"{c['raw_wins_top50']} | {c['summary_wins_top50']} | "
            f"{c['summary_share_top50']:.2%} |"
        )

    md += [
        "",
        "## Decision rules (from plan)",
        "",
        "- em_v2f_topicsumm > em_v2f_summ (0.9083/0.9167) by >=1pp -> stacks additively",
        "- ties em_v2f_summ -> summary captures what topic baking did",
        "- em_v2f_topicsumm_sf_spkfilter breaks 0.9417 (K=50) -> new LoCoMo ceiling",
        "",
        "## Outputs",
        "",
        "- Collections manifest: `results/eventmemory_topicsumm_collections.json`",
        "- SQLite store: `results/eventmemory_topicsumm.sqlite3`",
        "- Sources: `em_setup_topicsumm.py`, `compose_eval.py`",
    ]
    return md


def _build_md_part_b(results: dict) -> list[str]:
    md = [
        "# LME-hard POC turn-summary dual-view (Part B)",
        "",
        "## References",
        "",
        "| Variant (single-view raw-only) | R@20 | R@50 |",
        "| --- | --- | --- |",
        "| em_v2f_lme_mixed_7030 + expand_3 (prev leader) | 0.6368 | 0.8631 |",
        "| reflmem_lme 3round (prev overall leader)       | n/a    | 0.876  |",
        "",
        "## POC (dual-view raw + summary) recall",
        "",
        "| Variant | R@20 | R@50 | time (s) |",
        "| --- | --- | --- | --- |",
    ]
    for variant, data in results["variants"].items():
        s = data["summary"]
        md.append(
            f"| `{variant}` | {s['mean_r@20']:.4f} | "
            f"{s['mean_r@50']:.4f} | {s['time_s']:.1f} |"
        )

    md += ["", "## Per-category R@50", "",
           "| Variant | multi-session | single-session-preference | "
           "temporal-reasoning |",
           "| --- | --- | --- | --- |"]
    for variant, data in results["variants"].items():
        bc = data["by_category"]
        md.append(
            f"| `{variant}` | "
            f"{bc.get('multi-session', {}).get('mean_r@50', 0):.4f} | "
            f"{bc.get('single-session-preference', {}).get('mean_r@50', 0):.4f} | "
            f"{bc.get('temporal-reasoning', {}).get('mean_r@50', 0):.4f} |"
        )

    md += ["", "## View coverage (top-50)", "",
           "| Variant | gold credited | raw wins | summary wins | summary share |",
           "| --- | --- | --- | --- | --- |"]
    for variant, c in results.get("view_coverage", {}).items():
        md.append(
            f"| `{variant}` | {c['gold_credited_top50']} | "
            f"{c['raw_wins_top50']} | {c['summary_wins_top50']} | "
            f"{c['summary_share_top50']:.2%} |"
        )

    md += [
        "",
        "## Decision rules",
        "",
        "- POC R@50 on `em_v2f_lme_mixed_7030_expand3_summ` > 0.873 (+1pp over 0.863) ",
        "  -> summary generalizes; full 90-question run recommended.",
        "- POC temporal-reasoning R@50 > 0.807 -> cracks prior temporal ceiling.",
        "",
        "## Outputs",
        "",
        "- Collections manifest: `results/eventmemory_lmesumm_collections.json`",
        "- SQLite store: `results/eventmemory_lmesumm.sqlite3`",
        "- Summaries audit: `results/lmesumm_summaries.json`",
        "- Sources: `em_setup_lmesumm.py`, `compose_eval.py`",
    ]
    return md


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["a", "b"], required=True)
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variants. Defaults to all for the chosen part.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    PART_A_DEFAULT = [
        "em_cosine_baseline_topicsumm",
        "em_v2f_topicsumm",
        "em_v2f_topicsumm_sf_spkfilter",
    ]
    PART_B_DEFAULT = [
        "em_cosine_baseline_summ_lme",
        "em_v2f_lme_mixed_7030_expand3_summ",
    ]

    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = PART_A_DEFAULT if args.part == "a" else PART_B_DEFAULT

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.part == "a":
        results = await run_part_a(variants, args)
        out_json = RESULTS_DIR / "compose_topicsumm.json"
        out_md = RESULTS_DIR / "compose_topicsumm.md"
        out_md_body = _build_md_part_a(results)
    else:
        results = await run_part_b(variants, args)
        out_json = RESULTS_DIR / "compose_lme_turn_summary.json"
        out_md = RESULTS_DIR / "lme_turn_summary.md"
        out_md_body = _build_md_part_b(results)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}", flush=True)
    out_md.write_text("\n".join(out_md_body))
    print(f"Saved: {out_md}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
