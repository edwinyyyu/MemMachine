"""Task-shape robustness of cue generation on EventMemory (LoCoMo-30).

Question: does cue generation still lift recall on EventMemory for
non-question inputs (commands, drafts, synthesis/meta-queries)?

Architectures (import-only; no modification to framework files):
    em_cosine_baseline     raw query, no cues         (shape-robust reference)
    em_v2f                 vanilla v2f cues
    em_v2f_speakerformat   retuned mini winner
    em_hyde_first_person   single first-person probe  (shape-sensitive)

Shapes:
    ORIGINAL   the 30 base LoCoMo questions (from questions_extended.json)
    CMD        "Find ..."
    DRAFT      "Summarize ..." / "Draft a report covering ..."
    META       "What do we know about ..." / "Tell me about ..."

(CMD/DRAFT/META come from data/questions_locomo_task_shape.json; ORIGINAL
is looked up by orig_row_index from questions_extended.json's first-30
locomo slice.)

Each (arch, shape) cell: per-question R@20 and R@50, mean recall,
W/T/L vs em_cosine_baseline.

Caches are DEDICATED (never poison other specialists' caches):
    cache/emts_<arch>_cache.json

For ORIGINAL shape only, we also read from the existing specialist caches
(emretune_v2f_baseline_cache.json, emretune_v2f_speakerformat_cache.json,
hydeorient_hyde_first_person_cache.json) so cache hits transfer when the
prompt text is byte-identical.

Usage:
    uv run python evaluation/associative_recall/em_task_shape_eval.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from em_architectures import (
    V2F_MODEL,
    V2F_PROMPT,
    EMHit,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _MergedLLMCache,
    _query_em,
    format_primer_context,
    parse_v2f_cues,
)
from em_hyde_orient import (
    HYDE_FIRST_PERSON_PROMPT,
    parse_single_turn,
)
from em_retuned_cue_gen import (
    build_v2f_speakerformat_prompt,
)
from em_retuned_cue_gen import (
    parse_cues as parse_retuned_cues,
)
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
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
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

ARCHES = [
    "em_cosine_baseline",
    "em_v2f",
    "em_v2f_speakerformat",
    "em_hyde_first_person",
]
BASELINE_ARCH = "em_cosine_baseline"

SHAPES = ["ORIGINAL", "CMD", "DRAFT", "META"]

# Dedicated writer caches for this eval. Reader caches also include the
# matching existing specialist caches so ORIGINAL-shape queries hit them.
CACHE_CONFIG: dict[str, dict[str, Path | list[Path]]] = {
    "em_v2f": {
        "writer": CACHE_DIR / "emts_v2f_cache.json",
        "readers": [
            CACHE_DIR / "emts_v2f_cache.json",
            CACHE_DIR / "emretune_v2f_baseline_cache.json",
            CACHE_DIR / "em_v2f_llm_cache.json",
            CACHE_DIR / "bestshot_llm_cache.json",
        ],
    },
    "em_v2f_speakerformat": {
        "writer": CACHE_DIR / "emts_v2f_speakerformat_cache.json",
        "readers": [
            CACHE_DIR / "emts_v2f_speakerformat_cache.json",
            CACHE_DIR / "emretune_v2f_speakerformat_cache.json",
        ],
    },
    "em_hyde_first_person": {
        "writer": CACHE_DIR / "emts_hyde_first_person_cache.json",
        "readers": [
            CACHE_DIR / "emts_hyde_first_person_cache.json",
            CACHE_DIR / "hydeorient_hyde_first_person_cache.json",
        ],
    },
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_questions() -> list[dict]:
    """Load 120 rows: 30 ORIGINAL (from questions_extended.json first 30
    locomo) + 90 task-shape (30 x 3 shapes)."""
    with open(DATA_DIR / "questions_extended.json") as f:
        extended = json.load(f)
    locomo_orig = [q for q in extended if q.get("benchmark") == "locomo"][:30]

    with open(DATA_DIR / "questions_locomo_task_shape.json") as f:
        task_shape = json.load(f)

    rows: list[dict] = []
    # ORIGINAL: the 30 base questions themselves.
    for i, q in enumerate(locomo_orig):
        rows.append(
            {
                "orig_row_index": i,
                "conversation_id": q["conversation_id"],
                "category": q.get("category", "unknown"),
                "question_index": q.get("question_index", -1),
                "shape": "ORIGINAL",
                "original_question": q["question"],
                "question": q["question"],
                "source_chat_ids": q.get("source_chat_ids", []),
                "ideal_response": q.get("ideal_response", ""),
                "benchmark": "locomo",
            }
        )
    rows.extend(task_shape)
    return rows


def load_collections_meta() -> dict:
    with open(RESULTS_DIR / "eventmemory_collections.json") as f:
        return json.load(f)


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


# --------------------------------------------------------------------------
# LLM helper with dedicated caches
# --------------------------------------------------------------------------


async def _llm_call(
    openai_client,
    prompt: str,
    cache: _MergedLLMCache,
) -> tuple[str, bool]:
    cached = cache.get(V2F_MODEL, prompt)
    if cached is not None:
        return cached, True
    resp = await openai_client.chat.completions.create(
        model=V2F_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    cache.put(V2F_MODEL, prompt, text)
    return text, False


# --------------------------------------------------------------------------
# Primer (hop-0 retrieval for cue generation context)
# --------------------------------------------------------------------------


async def _primer_context(
    memory: EventMemory, question: str, max_K: int
) -> tuple[list[EMHit], str]:
    """Returns (primer_full_maxK, context_section)."""
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, question, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text} for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)
    primer_full = await _query_em(
        memory, question, vector_search_limit=max_K, expand_context=0
    )
    return primer_full, context_section


# --------------------------------------------------------------------------
# Per-architecture query runner
# --------------------------------------------------------------------------


async def run_arch(
    arch: str,
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    caches: dict[str, _MergedLLMCache],
    openai_client,
    *,
    max_K: int,
) -> tuple[list[EMHit], dict]:
    p_user, p_asst = participants

    if arch == "em_cosine_baseline":
        hits = _dedupe_by_turn_id(
            await _query_em(
                memory, question, vector_search_limit=max_K, expand_context=0
            )
        )[:max_K]
        return hits, {"cues": [], "cache_hit": True}

    if arch == "em_v2f":
        primer_full, context_section = await _primer_context(memory, question, max_K)
        prompt = V2F_PROMPT.format(question=question, context_section=context_section)
        raw, hit = await _llm_call(openai_client, prompt, caches["em_v2f"])
        cues = parse_v2f_cues(raw, max_cues=2)

        batches = [primer_full]
        for cue in cues:
            batches.append(
                await _query_em(
                    memory, cue, vector_search_limit=max_K, expand_context=0
                )
            )
        merged = _merge_by_max_score(batches)
        return merged[:max_K], {"cues": cues, "cache_hit": hit}

    if arch == "em_v2f_speakerformat":
        primer_full, context_section = await _primer_context(memory, question, max_K)
        prompt = build_v2f_speakerformat_prompt(
            question, context_section, p_user, p_asst
        )
        raw, hit = await _llm_call(
            openai_client, prompt, caches["em_v2f_speakerformat"]
        )
        cues = parse_retuned_cues(raw, max_cues=2)

        batches = [primer_full]
        for cue in cues:
            batches.append(
                await _query_em(
                    memory, cue, vector_search_limit=max_K, expand_context=0
                )
            )
        merged = _merge_by_max_score(batches)
        return merged[:max_K], {"cues": cues, "cache_hit": hit}

    if arch == "em_hyde_first_person":
        primer_full, context_section = await _primer_context(memory, question, max_K)
        prompt = HYDE_FIRST_PERSON_PROMPT.format(
            question=question,
            context_section=context_section,
            participant_1=p_user,
            participant_2=p_asst,
        )
        raw, hit = await _llm_call(
            openai_client, prompt, caches["em_hyde_first_person"]
        )
        turn = parse_single_turn(raw, (p_user, p_asst))

        probe_hits = (
            await _query_em(memory, turn, vector_search_limit=max_K, expand_context=0)
            if turn
            else []
        )
        merged = _merge_by_max_score([primer_full, probe_hits])
        return merged[:max_K], {"cues": [turn] if turn else [], "cache_hit": hit}

    raise KeyError(arch)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


async def main() -> None:
    collections_meta = load_collections_meta()
    questions = load_questions()
    conv_to_meta = {r["conversation_id"]: r for r in collections_meta["conversations"]}

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
        raise RuntimeError("No SQL_URL in collections meta or env")
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

    # Dedicated caches (reader + writer paths).
    caches: dict[str, _MergedLLMCache] = {}
    for arch, cfg in CACHE_CONFIG.items():
        caches[arch] = _MergedLLMCache(
            reader_paths=cfg["readers"], writer_path=cfg["writer"]
        )

    # Open one EventMemory per conversation.
    memories: dict[str, EventMemory] = {}
    participants_by_conv: dict[str, tuple[str, str]] = {}
    opened_resources: list = []
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
        opened_resources.append((coll, part))

    max_K = max(BUDGETS)
    results: dict = {
        "arches": ARCHES,
        "shapes": SHAPES,
        "budgets": list(BUDGETS),
        "n_questions_per_shape": 30,
        "cells": {},  # (arch, shape) -> {summary, per_question}
    }

    total_llm_calls = 0

    for arch in ARCHES:
        for shape in SHAPES:
            rows: list[dict] = []
            t0 = time.monotonic()
            for q in questions:
                if q.get("shape") != shape:
                    continue
                cid = q["conversation_id"]
                mem = memories[cid]
                participants = participants_by_conv[cid]
                q_text = q["question"]
                gold = set(q.get("source_chat_ids", []))

                hits, meta = await run_arch(
                    arch,
                    mem,
                    q_text,
                    participants,
                    caches,
                    openai_client,
                    max_K=max_K,
                )
                if not meta.get("cache_hit", True):
                    total_llm_calls += 1

                row = {
                    "conversation_id": cid,
                    "category": q.get("category", "unknown"),
                    "question_index": q.get("question_index", -1),
                    "orig_row_index": q.get("orig_row_index", -1),
                    "shape": shape,
                    "question": q_text,
                    "original_question": q.get("original_question", q_text),
                    "gold_turn_ids": sorted(gold),
                    "cues": meta.get("cues", []),
                    "cache_hit": meta.get("cache_hit", True),
                }
                for K in BUDGETS:
                    topk = hits[:K]
                    retrieved = {h.turn_id for h in topk}
                    row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
                    row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
                rows.append(row)
            elapsed = time.monotonic() - t0

            n = len(rows)
            summary = {"n": n, "time_s": round(elapsed, 1)}
            for K in BUDGETS:
                summary[f"mean_r@{K}"] = round(
                    sum(r[f"r@{K}"] for r in rows) / max(n, 1), 4
                )
            results["cells"][f"{arch}::{shape}"] = {
                "arch": arch,
                "shape": shape,
                "summary": summary,
                "per_question": rows,
            }
            print(
                f"[{arch} / {shape}] n={summary['n']} "
                f"r@20={summary['mean_r@20']:.4f} "
                f"r@50={summary['mean_r@50']:.4f} in {summary['time_s']:.1f}s"
            )

        # Persist caches after each arch so partial progress is retained.
        if arch in caches:
            caches[arch].save()

    print(f"[cache] live LLM calls (cache misses): {total_llm_calls}")

    # W/T/L per (arch, shape) vs (em_cosine_baseline, SAME shape).
    for arch in ARCHES:
        if arch == BASELINE_ARCH:
            continue
        for shape in SHAPES:
            key = f"{arch}::{shape}"
            base_key = f"{BASELINE_ARCH}::{shape}"
            v_rows = results["cells"][key]["per_question"]
            b_idx = {
                (r["conversation_id"], r["orig_row_index"]): r
                for r in results["cells"][base_key]["per_question"]
            }
            wtl: dict[str, dict[str, int]] = {}
            for K in BUDGETS:
                w = t = l = 0
                for r in v_rows:
                    br = b_idx.get((r["conversation_id"], r["orig_row_index"]))
                    if br is None:
                        continue
                    vr = r[f"r@{K}"]
                    bv = br[f"r@{K}"]
                    if vr > bv:
                        w += 1
                    elif vr < bv:
                        l += 1
                    else:
                        t += 1
                wtl[f"r@{K}"] = {"W": w, "T": t, "L": l}
            results["cells"][key]["wtl_vs_baseline_same_shape"] = wtl

    # Close.
    for coll, part in opened_resources:
        await segment_store.close_partition(part)
        await vector_store.close_collection(collection=coll)
    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "em_task_shape.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    md_lines = build_markdown_report(results)
    out_md = RESULTS_DIR / "em_task_shape.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved: {out_md}")


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------


def _get_mean(results: dict, arch: str, shape: str, K: int) -> float:
    return results["cells"][f"{arch}::{shape}"]["summary"][f"mean_r@{K}"]


def build_markdown_report(results: dict) -> list[str]:
    lines = [
        "# EventMemory Cue-Gen Task-Shape Robustness (LoCoMo-30)",
        "",
        "## Setup",
        "",
        "- n = 30 LoCoMo questions per shape (120 total queries: 30x4 shapes)",
        '- Shapes: ORIGINAL, CMD ("Find ..."), DRAFT ("Summarize/Draft ..."),',
        '  META ("What do we know about ..." / "Tell me about ...")',
        "- Backend: EventMemory (Qdrant + SQLite), speaker-baked embeddings",
        "- Embedder: `text-embedding-3-small`, cue LLM: `gpt-5-mini`",
        "- Dedicated caches: `cache/emts_<arch>_cache.json`",
        "",
        "## Recall matrix (mean R@K across 30 questions per shape)",
        "",
        "### R@20",
        "",
        "| Architecture | ORIGINAL | CMD | DRAFT | META |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arch in ARCHES:
        row = [f"`{arch}`"]
        for shape in SHAPES:
            row.append(f"{_get_mean(results, arch, shape, 20):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "### R@50",
        "",
        "| Architecture | ORIGINAL | CMD | DRAFT | META |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arch in ARCHES:
        row = [f"`{arch}`"]
        for shape in SHAPES:
            row.append(f"{_get_mean(results, arch, shape, 50):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # Cue-gen lift per shape (arch vs em_cosine_baseline, same shape).
    lines += [
        "",
        "## Cue-gen lift vs em_cosine_baseline (same shape)",
        "",
        "Positive = cue gen helps that shape; negative = cue gen hurts.",
        "",
        "### R@20",
        "",
        "| Architecture | ORIGINAL | CMD | DRAFT | META |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arch in ARCHES:
        if arch == BASELINE_ARCH:
            continue
        row = [f"`{arch}`"]
        for shape in SHAPES:
            d = _get_mean(results, arch, shape, 20) - _get_mean(
                results, BASELINE_ARCH, shape, 20
            )
            row.append(f"{d:+.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "### R@50",
        "",
        "| Architecture | ORIGINAL | CMD | DRAFT | META |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arch in ARCHES:
        if arch == BASELINE_ARCH:
            continue
        row = [f"`{arch}`"]
        for shape in SHAPES:
            d = _get_mean(results, arch, shape, 50) - _get_mean(
                results, BASELINE_ARCH, shape, 50
            )
            row.append(f"{d:+.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # Shape-sensitivity per architecture (drop from ORIGINAL to worst shape).
    lines += [
        "",
        "## Shape-sensitivity per architecture",
        "",
        "Drop from ORIGINAL shape to worst-shape (larger = more shape-sensitive).",
        "",
        "| Architecture | R@20 drop | Worst shape@20 | R@50 drop | Worst shape@50 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arch in ARCHES:
        orig_20 = _get_mean(results, arch, "ORIGINAL", 20)
        orig_50 = _get_mean(results, arch, "ORIGINAL", 50)
        worst_shape_20 = min(
            (s for s in SHAPES if s != "ORIGINAL"),
            key=lambda s: _get_mean(results, arch, s, 20),
        )
        worst_shape_50 = min(
            (s for s in SHAPES if s != "ORIGINAL"),
            key=lambda s: _get_mean(results, arch, s, 50),
        )
        drop_20 = orig_20 - _get_mean(results, arch, worst_shape_20, 20)
        drop_50 = orig_50 - _get_mean(results, arch, worst_shape_50, 50)
        lines.append(
            f"| `{arch}` | {drop_20:+.4f} | {worst_shape_20} | "
            f"{drop_50:+.4f} | {worst_shape_50} |"
        )

    # Per-shape ceiling.
    lines += [
        "",
        "## Per-shape ceiling (best architecture per shape)",
        "",
        "| Shape | Best @R@20 | Best @R@50 |",
        "| --- | --- | --- |",
    ]
    for shape in SHAPES:
        best_20_arch = max(ARCHES, key=lambda a: _get_mean(results, a, shape, 20))
        best_50_arch = max(ARCHES, key=lambda a: _get_mean(results, a, shape, 50))
        lines.append(
            f"| {shape} | `{best_20_arch}` "
            f"({_get_mean(results, best_20_arch, shape, 20):.4f}) | "
            f"`{best_50_arch}` "
            f"({_get_mean(results, best_50_arch, shape, 50):.4f}) |"
        )

    # Decision-rule verdict.
    lines += [
        "",
        "## Verdict",
        "",
    ]
    # Rule: (a) cue-gen lift on ORIGINAL but <=+1pp on all task shapes at K=50
    #       -> "collapse to cosine"
    #       (b) shape-invariant lift -> keep cue gen
    #       (c) any arch ACTIVELY hurts some shape -> flag
    v2f_sf_lift_orig_50 = _get_mean(
        results, "em_v2f_speakerformat", "ORIGINAL", 50
    ) - _get_mean(results, BASELINE_ARCH, "ORIGINAL", 50)
    task_shape_lifts_50 = {
        s: _get_mean(results, "em_v2f_speakerformat", s, 50)
        - _get_mean(results, BASELINE_ARCH, s, 50)
        for s in ("CMD", "DRAFT", "META")
    }
    max_task_lift = max(task_shape_lifts_50.values())
    # Detect active-harm cells (>=1pp drop vs cosine).
    active_harm: list[tuple[str, str, int, float]] = []
    for arch in ARCHES:
        if arch == BASELINE_ARCH:
            continue
        for shape in SHAPES:
            for K in BUDGETS:
                d = _get_mean(results, arch, shape, K) - _get_mean(
                    results, BASELINE_ARCH, shape, K
                )
                if d <= -0.01:
                    active_harm.append((arch, shape, K, d))

    lines.append(
        f"- em_v2f_speakerformat ORIGINAL K=50 lift vs cosine: "
        f"{v2f_sf_lift_orig_50:+.4f}"
    )
    lines.append(
        f"- em_v2f_speakerformat K=50 lifts on task shapes: "
        f"CMD {task_shape_lifts_50['CMD']:+.4f}, "
        f"DRAFT {task_shape_lifts_50['DRAFT']:+.4f}, "
        f"META {task_shape_lifts_50['META']:+.4f} "
        f"(max {max_task_lift:+.4f})"
    )
    if active_harm:
        lines.append(f"- Active-harm cells (drop >= 1pp vs cosine): {len(active_harm)}")
        for arch, shape, K, d in active_harm:
            lines.append(f"  - `{arch}` on {shape} at K={K}: {d:+.4f}")
    else:
        lines.append(
            "- No active-harm cells (no arch drops >=1pp vs cosine on any shape)."
        )

    if v2f_sf_lift_orig_50 >= 0.01 and max_task_lift <= 0.01:
        lines.append(
            "\n**Verdict (Rule A): cue gen collapses to cosine on task shapes.** "
            "Cue generation lifts ORIGINAL-question recall but fails to transfer "
            "to task/draft/synthesis inputs. General-case recipe simplifies to "
            "EM cosine baseline."
        )
    elif max_task_lift >= 0.01 and v2f_sf_lift_orig_50 >= 0.01:
        lines.append(
            "\n**Verdict (Rule B): cue gen is shape-robust and pays off in the general case.** "
            "Lift persists across task shapes; keep cue generation in the general recipe."
        )
    else:
        lines.append(
            "\n**Verdict: inconclusive / mixed.** See table above for per-shape behavior."
        )

    # HyDE collapse hypothesis check.
    hyde_drops = []
    for shape in ("CMD", "DRAFT", "META"):
        for K in BUDGETS:
            orig = _get_mean(results, "em_hyde_first_person", "ORIGINAL", K)
            cur = _get_mean(results, "em_hyde_first_person", shape, K)
            hyde_drops.append((shape, K, orig - cur))
    lines += [
        "",
        "## HyDE first-person collapse check",
        "",
        'Hypothesis: em_hyde_first_person\'s "I remember ..." framing hurts '
        "DRAFT/META more than v2f's flexible chat framing.",
        "",
    ]
    for shape, K, drop in hyde_drops:
        lines.append(
            f"- `em_hyde_first_person` {shape} at K={K}: "
            f"drop from ORIGINAL = {drop:+.4f}"
        )
    # Compare v2f_sf vs hyde_fp shape-sensitivity.
    v2f_sf_worst_50 = min(
        _get_mean(results, "em_v2f_speakerformat", s, 50)
        for s in ("CMD", "DRAFT", "META")
    )
    hyde_worst_50 = min(
        _get_mean(results, "em_hyde_first_person", s, 50)
        for s in ("CMD", "DRAFT", "META")
    )
    lines.append(
        f"\nWorst task-shape R@50: v2f_sf={v2f_sf_worst_50:.4f} "
        f"vs hyde_fp={hyde_worst_50:.4f}"
    )

    # Sample cues.
    lines += [
        "",
        "## Sample cues (3 questions x 4 shapes)",
        "",
    ]
    orig_rows = results["cells"]["em_v2f_speakerformat::ORIGINAL"]["per_question"]
    pick = [0, len(orig_rows) // 2, len(orig_rows) - 1]
    for idx in pick:
        orig_row_index = orig_rows[idx]["orig_row_index"]
        lines.append(
            f"### orig_row_index={orig_row_index} "
            f"({orig_rows[idx]['conversation_id']}, "
            f"{orig_rows[idx]['category']})"
        )
        lines.append("")
        for shape in SHAPES:
            rows = results["cells"][f"em_v2f_speakerformat::{shape}"]["per_question"]
            row = next((r for r in rows if r["orig_row_index"] == orig_row_index), None)
            if row is None:
                continue
            lines.append(f"- **{shape}**: {row['question']!r}")
            lines.append(
                f"  - `em_v2f_speakerformat` cues "
                f"(R@20={row['r@20']:.2f}, R@50={row['r@50']:.2f}):"
            )
            for c in row["cues"]:
                lines.append(f"    - `{c}`")
        lines.append("")

    lines += [
        "",
        "## Outputs",
        "",
        "- `results/em_task_shape.json`",
        "- `results/em_task_shape.md`",
        "- Source: `em_task_shape_eval.py`",
        "- Caches: `cache/emts_<arch>_cache.json` (dedicated)",
        "",
    ]
    return lines


if __name__ == "__main__":
    asyncio.run(main())
