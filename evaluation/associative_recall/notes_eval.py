"""Evaluate model-note-augmented EventMemory (v3) on LoCoMo-30 and on task-sufficiency.

Three retrieval matrices:
  1. em_cosine_notes          raw-cosine, dual-event (messages+notes) index
  2. em_v2f_notes             v2f speakerformat cues, dual-event index
  3. em_v2f_notes_msgs_only   messages-only control (event_type="message" filter)
  4. em_v2f_notes_only        notes-only ablation (event_type="model_note")

Task-sufficiency (THE KEY TEST): reuses data/proactive_tasks.json (20 tasks).
  A. single-shot v2f on standard ingest            (baseline, arc_em_lc30_v1_*)
  B. single-shot v2f on notes ingest               (arc_em_lc30_notes_v3_*)
  C. proactive decomposition on notes ingest       (same notes ingest)

Retrieved hits are labeled "[message]" or "[model_note]" when fed to the
sufficiency judge so it understands what it's seeing.

Usage:
  uv run python evaluation/associative_recall/notes_eval.py --phase retrieval
  uv run python evaluation/associative_recall/notes_eval.py --phase sufficiency
  uv run python evaluation/associative_recall/notes_eval.py   # both
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
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
    EMHit,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _query_em,
)
from em_retuned_cue_gen import (
    V2F_SPEAKERFORMAT_PROMPT,
    parse_cues as parse_speakerformat_cues,
)
from proactive_memory import (
    PROACTIVE_CUEGEN_CACHE,
    PROACTIVE_DECOMPOSE_CACHE,
    PROACTIVE_SUFFICIENCY_CACHE,
    ProactiveResult,
    _llm_call,
    run_proactive,
    run_single_shot,
)


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")
BUDGETS = (20, 50)

# Dedicated caches so we don't collide with other specialists.
NOTES_V2F_CUEGEN_CACHE = CACHE_DIR / "notes_v3_v2f_cuegen_cache.json"
NOTES_JUDGE_CACHE = CACHE_DIR / "notes_v3_judge_cache.json"
NOTES_SINGLESHOT_CUEGEN_CACHE = CACHE_DIR / "notes_v3_singleshot_cuegen_cache.json"


NOTES_COLLECTIONS_META = RESULTS_DIR / "eventmemory_notes_v3_collections.json"
STD_COLLECTIONS_META = RESULTS_DIR / "eventmemory_collections.json"


# ---------------------------------------------------------------------------
# Retrieval architectures
# ---------------------------------------------------------------------------


def _format_primer_context_for_cuegen(hits: list[EMHit]) -> str:
    """Context section for v2f cue-gen prompt. Byte-compatible with
    best_shot's _format_segments so cues cached for standard ingest still
    hit when the retrieved items are all messages. When notes leak into
    the primer their role="model_note" naturally disambiguates them.
    """
    if not hits:
        return (
            "No conversation excerpts retrieved yet. Generate cues based on "
            "what you'd expect to find in a conversation about this topic."
        )
    sorted_hits = sorted(hits, key=lambda h: h.turn_id)[:12]
    lines = []
    for h in sorted_hits:
        txt = (h.text or "")[:250].replace("\n", " ")
        lines.append(f"[Turn {h.turn_id}, {h.role}]: {txt}")
    return "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + "\n".join(lines)


def _event_type_filter(event_type: str):
    """Build a property_filter for EventMemory.query filtering by event_type.

    User-defined properties use the `m.<name>` / `metadata.<name>` query prefix;
    without the prefix the framework treats the field as a system field (and
    translates to `_event_type` which we did NOT store).
    """
    return Comparison(field="m.event_type", op="=", value=event_type)


async def em_cosine_notes(
    memory: EventMemory,
    question: str,
    *,
    K: int,
) -> list[EMHit]:
    hits = await _query_em(
        memory, question, vector_search_limit=K * 2, expand_context=0
    )
    return _dedupe_by_turn_id(hits)[:K]


async def em_v2f_notes(
    memory: EventMemory,
    question: str,
    *,
    K: int,
    llm_cache: _MergedLLMCache,
    openai_client,
    participants: tuple[str, str],
    property_filter=None,
) -> tuple[list[EMHit], dict]:
    """V2F-speakerformat primer + 2 cues, merged, optionally filtered.

    property_filter lets us restrict the retrieval to messages or notes only.
    """
    # Primer: raw-query retrieval (K=10).
    primer_hits = await _query_em_filtered(
        memory, question, K=10, property_filter=property_filter
    )
    primer_hits = _dedupe_by_turn_id(primer_hits)[:10]
    context_section = _format_primer_context_for_cuegen(primer_hits)

    prompt = V2F_SPEAKERFORMAT_PROMPT.format(
        question=question,
        context_section=context_section,
        participant_1=participants[0],
        participant_2=participants[1],
    )
    cached = llm_cache.get(V2F_MODEL, prompt)
    cache_hit = cached is not None
    if cached is None:
        # reasoning_effort="low" keeps latency ~2-3s vs ~10s at default.
        resp = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low",
        )
        cached = resp.choices[0].message.content or ""
        llm_cache.put(V2F_MODEL, prompt, cached)
    cues = parse_speakerformat_cues(cached, max_cues=2)

    # Retrievals: primer (full K) + per-cue
    batches = [
        await _query_em_filtered(memory, question, K=K, property_filter=property_filter)
    ]
    for cue in cues[:2]:
        batches.append(
            await _query_em_filtered(memory, cue, K=K, property_filter=property_filter)
        )

    merged = _merge_by_max_score(batches)
    merged = _dedupe_by_turn_id(merged)[:K]
    return merged, {"cues": cues, "cache_hit": cache_hit}


async def _query_em_filtered(
    memory: EventMemory,
    text: str,
    *,
    K: int,
    property_filter=None,
) -> list[EMHit]:
    qr = await memory.query(
        query=text,
        vector_search_limit=K,
        expand_context=0,
        property_filter=property_filter,
    )
    hits: list[EMHit] = []
    for sc in qr.scored_segment_contexts:
        for seg in sc.segments:
            hits.append(
                EMHit(
                    turn_id=int(seg.properties.get("turn_id", -1)),
                    score=sc.score,
                    seed_segment_uuid=sc.seed_segment_uuid,
                    role=str(seg.properties.get("role", "")),
                    text=seg.block.text if hasattr(seg.block, "text") else "",
                )
            )
    return hits


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


def load_locomo_questions() -> list[dict]:
    with open(DATA_DIR / "questions_extended.json") as f:
        qs = json.load(f)
    locomo = [q for q in qs if q.get("benchmark") == "locomo"]
    return locomo[:30]


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


async def evaluate_retrieval_question(
    arch: str,
    memory: EventMemory,
    question: dict,
    *,
    max_K: int,
    cache: _MergedLLMCache,
    openai_client,
    participants: tuple[str, str],
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()
    meta: dict = {}
    if arch == "em_cosine_notes":
        hits = await em_cosine_notes(memory, q_text, K=max_K)
    elif arch == "em_v2f_notes":
        hits, meta = await em_v2f_notes(
            memory, q_text, K=max_K, llm_cache=cache,
            openai_client=openai_client, participants=participants,
        )
    elif arch == "em_v2f_notes_msgs_only":
        hits, meta = await em_v2f_notes(
            memory, q_text, K=max_K, llm_cache=cache,
            openai_client=openai_client, participants=participants,
            property_filter=_event_type_filter("message"),
        )
    elif arch == "em_v2f_notes_only":
        hits, meta = await em_v2f_notes(
            memory, q_text, K=max_K, llm_cache=cache,
            openai_client=openai_client, participants=participants,
            property_filter=_event_type_filter("model_note"),
        )
    else:
        raise KeyError(arch)
    elapsed = time.monotonic() - t0

    row: dict = {
        "arch": arch,
        "conversation_id": question["conversation_id"],
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "gold_turn_ids": sorted(gold),
        "n_hits": len(hits),
        "time_s": round(elapsed, 3),
    }
    row.update(meta)
    for K in BUDGETS:
        topk = hits[:K]
        retrieved = {h.turn_id for h in topk}
        row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
        row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
    return row


async def run_retrieval_phase(
    *,
    memories: dict[str, EventMemory],
    participants_by_conv: dict[str, tuple[str, str]],
    questions: list[dict],
    archs: list[str],
    openai_client,
    v2f_cache: _MergedLLMCache,
) -> dict:
    max_K = max(BUDGETS)
    out: dict = {"archs": {}, "budgets": list(BUDGETS), "n_questions": len(questions)}

    for arch in archs:
        t_arch = time.monotonic()
        rows: list[dict] = []
        for i, q in enumerate(questions):
            mem = memories[q["conversation_id"]]
            participants = participants_by_conv[q["conversation_id"]]
            t_q = time.monotonic()
            row = await evaluate_retrieval_question(
                arch, mem, q,
                max_K=max_K, cache=v2f_cache,
                openai_client=openai_client, participants=participants,
            )
            rows.append(row)
            if (i + 1) % 5 == 0 or i == len(questions) - 1:
                print(f"  [{arch}] q {i+1}/{len(questions)} r@20={row['r@20']} r@50={row['r@50']} t={time.monotonic()-t_q:.1f}s", flush=True)
        v2f_cache.save()
        arch_elapsed = time.monotonic() - t_arch

        n = len(rows)
        summary = {"n": n, "time_s": round(arch_elapsed, 1)}
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
        out["archs"][arch] = {
            "summary": summary,
            "by_category": cat_summary,
            "per_question": rows,
        }
        print(
            f"[{arch}] n={n} r@20={summary['mean_r@20']:.4f} "
            f"r@50={summary['mean_r@50']:.4f} in {summary['time_s']:.1f}s"
        )
    return out


# ---------------------------------------------------------------------------
# Task sufficiency
# ---------------------------------------------------------------------------


def load_proactive_tasks() -> list[dict]:
    with open(DATA_DIR / "proactive_tasks.json") as f:
        return json.load(f)["tasks"]


JUDGE_PROMPT_LABELED = """\
You are a retrieval sufficiency judge. You are given a user TASK and a \
set of chat-memory items that were retrieved to help an AI complete the \
task. Each item is labeled as either [message] (an actual chat turn) or \
[model_note] (an LLM-generated internal-monologue note summarizing state \
after that turn).

Rate the retrieval on four dimensions (all integers 0-10):

- COVERAGE: Does the retrieved content cover all the information types \
the task requires? (10 = every type present; 0 = nothing relevant.)
- DEPTH: Does it contain enough specific detail to actually complete the \
task? (10 = richly specific; 0 = only superficial snippets.)
- NOISE (reverse-scored): 10 means the retrieved set is tightly on-topic \
(low noise); 0 means almost everything is irrelevant.
- SUFFICIENCY: Overall, could an AI use this retrieval to complete the \
task well? (10 = yes, plenty; 0 = no, would mostly be guessing.)

TASK:
{task_prompt}

RETRIEVED ITEMS ({n_items} total, top-{n_shown} shown):
{retrieved_section}

Output ONLY a JSON object, no prose:
{{"coverage": <0-10>, "depth": <0-10>, "noise": <0-10>, \
"sufficiency": <0-10>, "brief_reasoning": "one or two sentences"}}"""


def _format_hits_for_judge_labeled(
    hits: list[EMHit], max_items: int = 40, max_len: int = 220
) -> str:
    if not hits:
        return "(no retrievals)"
    sorted_hits = sorted(hits, key=lambda h: h.turn_id)[:max_items]
    lines = []
    for h in sorted_hits:
        label = "[model_note]" if h.role == "model_note" else "[message]"
        txt = (h.text or "").replace("\n", " ")
        if len(txt) > max_len:
            txt = txt[:max_len] + "..."
        lines.append(f"{label} [Turn {h.turn_id}]: {txt}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    import re as _re
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    m = _re.search(r"\{.*\}", t, _re.DOTALL)
    if m is None:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


async def judge_retrieval_labeled(
    task_prompt: str,
    hits: list[EMHit],
    *,
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[dict, bool]:
    shown = sorted(hits, key=lambda h: h.turn_id)[:40]
    retrieved_section = _format_hits_for_judge_labeled(shown, max_items=40)
    prompt = JUDGE_PROMPT_LABELED.format(
        task_prompt=task_prompt,
        n_items=len(hits),
        n_shown=len(shown),
        retrieved_section=retrieved_section,
    )
    raw, cache_hit = await _llm_call(openai_client, prompt, cache)
    obj = _extract_json(raw)
    out = {
        "coverage": int(obj.get("coverage", 0) or 0),
        "depth": int(obj.get("depth", 0) or 0),
        "noise": int(obj.get("noise", 0) or 0),
        "sufficiency": int(obj.get("sufficiency", 0) or 0),
        "brief_reasoning": str(obj.get("brief_reasoning") or "").strip(),
    }
    return out, cache_hit


async def run_sufficiency_phase(
    *,
    memories_notes: dict[str, EventMemory],
    memories_std: dict[str, EventMemory],
    participants_by_conv: dict[str, tuple[str, str]],
    tasks: list[dict],
    openai_client,
    caches: dict[str, _MergedLLMCache],
    K: int = 50,
    K_per_need: int = 15,
    max_rounds: int = 2,
) -> list[dict]:
    rows: list[dict] = []
    for task in tasks:
        cid = task["conversation_id"]
        participants = participants_by_conv[cid]
        task_prompt = task["task_prompt"]

        # System A: single-shot v2f on STANDARD ingest (baseline).
        memA = memories_std[cid]
        tA = time.monotonic()
        resA = await run_single_shot(
            memA, task_prompt, participants, K=K,
            cuegen_cache=caches["singleshot_std"],
            openai_client=openai_client,
        )
        tA = time.monotonic() - tA
        judgeA, _ = await judge_retrieval_labeled(
            task_prompt, resA.hits, cache=caches["judge"], openai_client=openai_client
        )

        # System B: single-shot v2f on NOTES ingest.
        memN = memories_notes[cid]
        tB = time.monotonic()
        resB = await run_single_shot(
            memN, task_prompt, participants, K=K,
            cuegen_cache=caches["singleshot_notes"],
            openai_client=openai_client,
        )
        tB = time.monotonic() - tB
        judgeB, _ = await judge_retrieval_labeled(
            task_prompt, resB.hits, cache=caches["judge"], openai_client=openai_client
        )

        # System C: proactive decomposition on NOTES ingest.
        tC = time.monotonic()
        resC = await run_proactive(
            memN, task_prompt, participants,
            K_per_need=K_per_need, K_final=K, max_rounds=max_rounds,
            decompose_cache=caches["decompose"],
            cuegen_cache=caches["proactive_cuegen"],
            sufficiency_cache=caches["sufficiency"],
            openai_client=openai_client,
        )
        tC = time.monotonic() - tC
        judgeC, _ = await judge_retrieval_labeled(
            task_prompt, resC.hits, cache=caches["judge"], openai_client=openai_client
        )

        def _summarize(res: ProactiveResult, judge: dict, t: float) -> dict:
            n_notes_retrieved = sum(1 for h in res.hits if h.role == "model_note")
            n_msgs_retrieved = sum(1 for h in res.hits if h.role != "model_note")
            return {
                "metadata": res.metadata,
                "judge": judge,
                "time_s": round(t, 2),
                "n_llm_calls": res.metadata["n_llm_calls"],
                "n_turns_retrieved": res.metadata["n_turns_retrieved"],
                "n_notes_retrieved": n_notes_retrieved,
                "n_msgs_retrieved": n_msgs_retrieved,
                "hits_turn_ids": [h.turn_id for h in res.hits],
            }

        rows.append({
            "task_id": task["task_id"],
            "conversation_id": cid,
            "task_shape": task.get("task_shape", ""),
            "task_prompt": task_prompt,
            "required_info_categories": task.get("required_info_categories", []),
            "A_std_singleshot": _summarize(resA, judgeA, tA),
            "B_notes_singleshot": _summarize(resB, judgeB, tB),
            "C_notes_proactive": _summarize(resC, judgeC, tC),
        })
        for c in caches.values():
            c.save()
        print(
            f"[{task['task_id']}] cid={cid} "
            f"A_suff={judgeA['sufficiency']} B_suff={judgeB['sufficiency']} "
            f"C_suff={judgeC['sufficiency']} "
            f"(notes_in_B={rows[-1]['B_notes_singleshot']['n_notes_retrieved']}, "
            f"notes_in_C={rows[-1]['C_notes_proactive']['n_notes_retrieved']})"
        )
    return rows


# ---------------------------------------------------------------------------
# Open / close memories
# ---------------------------------------------------------------------------


async def _open_memories_from_meta(
    collections_meta: dict,
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
):
    memories: dict[str, EventMemory] = {}
    participants_by_conv: dict[str, tuple[str, str]] = {}
    opened = []
    conv_to_meta = {r["conversation_id"]: r for r in collections_meta["conversations"]}
    for cid in LOCOMO_CONV_IDS:
        meta = conv_to_meta[cid]
        coll = await vector_store.open_collection(
            namespace=meta["namespace"], name=meta["collection_name"]
        )
        part = await segment_store.open_or_create_partition(meta["partition_key"])
        mem = EventMemory(EventMemoryParams(
            vector_store_collection=coll,
            segment_store_partition=part,
            embedder=embedder,
            reranker=None,
            derive_sentences=False,
            max_text_chunk_length=500,
        ))
        memories[cid] = mem
        participants_by_conv[cid] = (meta["user_name"], meta["assistant_name"])
        opened.append((coll, part))
    return memories, participants_by_conv, opened


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["retrieval", "sufficiency", "both"],
                        default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--K_per_need", type=int, default=15)
    parser.add_argument("--max_rounds", type=int, default=2)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(NOTES_COLLECTIONS_META) as f:
        notes_meta = json.load(f)
    with open(STD_COLLECTIONS_META) as f:
        std_meta = json.load(f)

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True, timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    # Two segment stores: standard SQLite and notes SQLite.
    std_sql_url = std_meta.get("sql_url") or os.getenv("SQL_URL")
    if std_sql_url.startswith("sqlite"):
        std_engine = create_async_engine(std_sql_url)
    else:
        std_engine = create_async_engine(std_sql_url, pool_size=20, max_overflow=20)
    std_segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=std_engine))
    await std_segment_store.startup()

    notes_sql_url = notes_meta["sql_url"]
    notes_engine = create_async_engine(notes_sql_url)
    notes_segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=notes_engine))
    await notes_segment_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedder = OpenAIEmbedder(OpenAIEmbedderParams(
        client=openai_client, model="text-embedding-3-small",
        dimensions=1536, max_input_length=8192,
    ))

    # Warm v2f cache from the existing retuned-speakerformat cache so we don't
    # re-pay for cues that were already generated in prior runs. We still WRITE
    # only to our dedicated cache.
    v2f_cache = _MergedLLMCache(
        reader_paths=[
            CACHE_DIR / "emretune_v2f_speakerformat_cache.json",
            CACHE_DIR / "emts_v2f_speakerformat_cache.json",
            NOTES_V2F_CUEGEN_CACHE,
        ],
        writer_path=NOTES_V2F_CUEGEN_CACHE,
    )

    caches = {
        "singleshot_std": _MergedLLMCache(
            reader_paths=[CACHE_DIR / "proactive_singleshot_cuegen_cache.json"],
            writer_path=CACHE_DIR / "proactive_singleshot_cuegen_cache.json",
        ),
        "singleshot_notes": _MergedLLMCache(
            reader_paths=[NOTES_SINGLESHOT_CUEGEN_CACHE],
            writer_path=NOTES_SINGLESHOT_CUEGEN_CACHE,
        ),
        "decompose": _MergedLLMCache(
            reader_paths=[PROACTIVE_DECOMPOSE_CACHE],
            writer_path=PROACTIVE_DECOMPOSE_CACHE,
        ),
        "proactive_cuegen": _MergedLLMCache(
            reader_paths=[PROACTIVE_CUEGEN_CACHE],
            writer_path=PROACTIVE_CUEGEN_CACHE,
        ),
        "sufficiency": _MergedLLMCache(
            reader_paths=[PROACTIVE_SUFFICIENCY_CACHE],
            writer_path=PROACTIVE_SUFFICIENCY_CACHE,
        ),
        "judge": _MergedLLMCache(
            reader_paths=[NOTES_JUDGE_CACHE],
            writer_path=NOTES_JUDGE_CACHE,
        ),
    }

    # Open memories.
    notes_memories, participants_by_conv, notes_opened = await _open_memories_from_meta(
        notes_meta, vector_store, notes_segment_store, embedder
    )
    std_memories, _std_parts, std_opened = await _open_memories_from_meta(
        std_meta, vector_store, std_segment_store, embedder
    )

    final = {
        "config": {
            "model": V2F_MODEL,
            "notes_namespace": notes_meta["namespace"],
            "std_namespace": std_meta.get("namespace"),
            "K": args.K,
        }
    }

    try:
        if args.phase in ("retrieval", "both"):
            questions = load_locomo_questions()
            if args.limit is not None:
                questions = questions[: args.limit]
            archs = [
                "em_cosine_notes",
                "em_v2f_notes",
                "em_v2f_notes_msgs_only",
                "em_v2f_notes_only",
            ]
            retrieval_results = await run_retrieval_phase(
                memories=notes_memories,
                participants_by_conv=participants_by_conv,
                questions=questions,
                archs=archs,
                openai_client=openai_client,
                v2f_cache=v2f_cache,
            )
            final["retrieval"] = retrieval_results
            v2f_cache.save()

        if args.phase in ("sufficiency", "both"):
            tasks = load_proactive_tasks()
            if args.limit is not None:
                tasks = tasks[: args.limit]
            suff_rows = await run_sufficiency_phase(
                memories_notes=notes_memories,
                memories_std=std_memories,
                participants_by_conv=participants_by_conv,
                tasks=tasks,
                openai_client=openai_client,
                caches=caches,
                K=args.K,
                K_per_need=args.K_per_need,
                max_rounds=args.max_rounds,
            )
            # Aggregate
            def _mean(xs): return sum(xs)/max(len(xs),1)
            def _agg(key):
                return {
                    "mean_sufficiency": round(_mean([r[key]["judge"]["sufficiency"] for r in suff_rows]), 3),
                    "mean_coverage": round(_mean([r[key]["judge"]["coverage"] for r in suff_rows]), 3),
                    "mean_depth": round(_mean([r[key]["judge"]["depth"] for r in suff_rows]), 3),
                    "mean_noise": round(_mean([r[key]["judge"]["noise"] for r in suff_rows]), 3),
                    "mean_llm_calls": round(_mean([r[key]["n_llm_calls"] for r in suff_rows]), 2),
                    "mean_n_notes": round(_mean([r[key]["n_notes_retrieved"] for r in suff_rows]), 2),
                    "mean_n_msgs": round(_mean([r[key]["n_msgs_retrieved"] for r in suff_rows]), 2),
                }
            final["sufficiency"] = {
                "n_tasks": len(suff_rows),
                "A_std_singleshot": _agg("A_std_singleshot"),
                "B_notes_singleshot": _agg("B_notes_singleshot"),
                "C_notes_proactive": _agg("C_notes_proactive"),
                "per_task": suff_rows,
            }
            for c in caches.values():
                c.save()
    finally:
        for coll, part in notes_opened:
            await notes_segment_store.close_partition(part)
            await vector_store.close_collection(collection=coll)
        for coll, part in std_opened:
            await std_segment_store.close_partition(part)
            await vector_store.close_collection(collection=coll)
        await std_segment_store.shutdown()
        await notes_segment_store.shutdown()
        await vector_store.shutdown()
        await std_engine.dispose()
        await notes_engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    out_json = RESULTS_DIR / "model_notes.json"
    with open(out_json, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Saved: {out_json}")

    # Markdown report.
    md = build_report(final, notes_meta)
    out_md = RESULTS_DIR / "model_notes.md"
    out_md.write_text(md)
    print(f"Saved: {out_md}")


def build_report(final: dict, notes_meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Model-note augmented EventMemory (v3)")
    lines.append("")
    lines.append("## Architecture (EM-format context, v3)")
    lines.append("")
    lines.append(
        "- Per turn: ingest message normally (speaker baked via "
        "`MessageContext(source=speaker)`). At `turn_ts + 1us`, generate "
        "a note and ingest it as a second event with "
        "`MessageContext(source=\"ModelNote\")`, `event_type=\"model_note\"`."
    )
    lines.append(
        "- Note-generator (gpt-5-mini, `reasoning_effort=\"low\"`) sees context "
        "formatted as EM-canonical strings `\"<source>: <content>\"`: prior notes "
        "appear as `ModelNote: ...`, recent and retrieved messages as "
        "`<speaker>: ...`. No JSON, no bracket labels at note-gen time."
    )
    lines.append(
        "- Similarity retrieval at note time: V_combined — latest prior note "
        "+ last 3 turns, joined as EM-format strings, top-K=4."
    )
    lines.append(
        "- Output prose is three labeled lines "
        "`current_understanding / open_questions / recent_realization` "
        "concatenated into one paragraph and ingested as `\"ModelNote: <prose>\"`."
    )
    lines.append("")

    # Sample notes
    samples_path = RESULTS_DIR / "notes_v3_samples.json"
    if samples_path.exists():
        with open(samples_path) as f:
            samples = json.load(f).get("samples", [])
        if samples:
            lines.append("## Sample notes (conv-26)")
            lines.append("")
            c26 = [s for s in samples if s["conv_id"] == "locomo_conv-26"]
            for s in c26[:3]:
                lines.append(f"### {s['position']} (turn {s['turn_id']}, {s['speaker']})")
                lines.append("")
                lines.append(f"**Turn**: {s['turn_text']}")
                lines.append("")
                lines.append(f"**Note**: {s['note_prose']}")
                lines.append("")

    if "retrieval" in final:
        lines.append("## Retrieval (LoCoMo-30)")
        lines.append("")
        lines.append("Baseline reference: `em_v2f_speakerformat` (no notes) = R@20=0.8167, R@50=0.8917.")
        lines.append("")
        lines.append("| Architecture | R@20 | R@50 | time (s) |")
        lines.append("| --- | --- | --- | --- |")
        for arch, data in final["retrieval"]["archs"].items():
            s = data["summary"]
            lines.append(
                f"| `{arch}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | {s['time_s']:.1f} |"
            )
        lines.append("")

    if "sufficiency" in final:
        s = final["sufficiency"]
        lines.append("## Task-sufficiency (20 proactive tasks, LLM judge)")
        lines.append("")
        lines.append(
            f"- A = single-shot v2f over STANDARD ingest (messages only)."
        )
        lines.append(
            f"- B = single-shot v2f over NOTES ingest (messages + notes)."
        )
        lines.append(
            f"- C = proactive decomposition over NOTES ingest."
        )
        lines.append("")
        lines.append("| System | Suff | Cov | Depth | Noise | LLM calls | notes in retrieval |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for key, label in [
            ("A_std_singleshot", "A_std_singleshot"),
            ("B_notes_singleshot", "B_notes_singleshot"),
            ("C_notes_proactive", "C_notes_proactive"),
        ]:
            a = s[key]
            lines.append(
                f"| {label} | {a['mean_sufficiency']} | {a['mean_coverage']} | "
                f"{a['mean_depth']} | {a['mean_noise']} | "
                f"{a['mean_llm_calls']} | {a['mean_n_notes']} |"
            )
        lines.append("")
        dBA = s["B_notes_singleshot"]["mean_sufficiency"] - s["A_std_singleshot"]["mean_sufficiency"]
        dCA = s["C_notes_proactive"]["mean_sufficiency"] - s["A_std_singleshot"]["mean_sufficiency"]
        lines.append(f"**Delta vs baseline A**: B-A = {dBA:+.3f}, C-A = {dCA:+.3f}.")
        lines.append("")
        # per-task table
        lines.append("### Per-task sufficiency")
        lines.append("")
        lines.append("| task | conv | shape | A | B | C | notes@B | notes@C |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in s["per_task"]:
            lines.append(
                f"| {r['task_id']} | {r['conversation_id'][-2:]} | {r.get('task_shape','')} | "
                f"{r['A_std_singleshot']['judge']['sufficiency']} | "
                f"{r['B_notes_singleshot']['judge']['sufficiency']} | "
                f"{r['C_notes_proactive']['judge']['sufficiency']} | "
                f"{r['B_notes_singleshot']['n_notes_retrieved']} | "
                f"{r['C_notes_proactive']['n_notes_retrieved']} |"
            )
        lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    if "retrieval" in final:
        em_v2f = final["retrieval"]["archs"].get("em_v2f_notes", {}).get("summary", {})
        if em_v2f:
            lines.append(
                f"- Retrieval: em_v2f_notes R@50 = {em_v2f.get('mean_r@50', 'N/A')} "
                f"vs baseline 0.8917. "
                f"{'Lift' if em_v2f.get('mean_r@50', 0) - 0.8917 >= 0.01 else 'No lift'} "
                f"on retrieval."
            )
    if "sufficiency" in final:
        s = final["sufficiency"]
        dBA = s["B_notes_singleshot"]["mean_sufficiency"] - s["A_std_singleshot"]["mean_sufficiency"]
        dCA = s["C_notes_proactive"]["mean_sufficiency"] - s["A_std_singleshot"]["mean_sufficiency"]
        verdict_line = "Notes help" if max(dBA, dCA) >= 1.0 else ("Tie" if abs(max(dBA, dCA)) < 1.0 else "Notes hurt")
        lines.append(
            f"- Task-sufficiency: best delta over standard-ingest baseline = "
            f"{max(dBA, dCA):+.2f}. Verdict: **{verdict_line}**."
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `results/model_notes.json`")
    lines.append("- `results/model_notes.md`")
    lines.append(f"- Notes collections manifest: `results/eventmemory_notes_v3_collections.json`")
    lines.append("- Source: `em_setup_notes_v3.py`, `notes_eval.py`")
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
