"""Evaluate LME-tuned cue variants on LongMemEval-hard EventMemory.

Variants:
  - em_v2f_lme_userformat          (expand_context=3)
  - em_v2f_lme_user_only           (expand_context=3)
  - em_v2f_lme_mixed_7030          (expand_context=3)
  - em_type_enumerated_lme_retuned (expand_context=3)
  - em_ens_2_lme_retuned           (expand_context=3)

Reuses em_lme_setup.py's ingested collections + sqlite.  Imports-only
against em_architectures and em_lme_eval helpers.

Outputs:
  results/em_lme_tuned.json
  results/em_lme_tuned.md
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
    EMHit,
    V2F_MODEL,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _query_em,
    format_primer_context,
)
from em_lme_tuned_cues import (
    V2F_LME_USERFORMAT_PROMPT,
    V2F_LME_USER_ONLY_PROMPT,
    V2F_LME_MIXED_7030_PROMPT,
    TYPE_ENUM_LME_RETUNED_PROMPT,
    parse_speaker_cues,
    LMETUNE_V2F_USERFORMAT_CACHE,
    LMETUNE_V2F_USERONLY_CACHE,
    LMETUNE_V2F_MIXED7030_CACHE,
    LMETUNE_TE_RETUNED_CACHE,
)


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

ASSOC_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSOC_DIR / "data"
RESULTS_DIR = ASSOC_DIR / "results"

HARD_QUESTIONS_JSON = DATA_DIR / "questions_longmemeval_hard.json"
COLLECTIONS_JSON = RESULTS_DIR / "em_lme_hard_collections.json"
RESULTS_JSON = RESULTS_DIR / "em_lme_tuned.json"
RESULTS_MD = RESULTS_DIR / "em_lme_tuned.md"

BUDGETS = (20, 50)
USER_PREFIX = "User: "
EXPAND_CONTEXT = 3
ARCH_CONCURRENCY = 8

# Reference points from prior runs (for report).
REF_EM_V2F_USERPREFIX = 0.780       # em_v2f_userprefix K=50 R overall
REF_EM_V2F_EXPAND_3 = 0.832         # em_v2f_expand_3 K=50 R overall (current leader)
REF_EM_ENS_2_USERPREFIX = 0.735     # em_ens_2_userprefix K=50 (regressed)
REF_SS_V2F = 0.817                  # SegmentStore v2f K=50 overall


def _ensure_user_prefix(text: str) -> str:
    t = text.lstrip()
    low = t.lower()
    if low.startswith("user:") or low.startswith("assistant:"):
        return t
    return USER_PREFIX + t


async def _primer_and_context(memory: EventMemory, question: str) -> str:
    """Build the primer context section (matches em_v2f_userprefix)."""
    prefixed_q = _ensure_user_prefix(question)
    primer_hits = _dedupe_by_turn_id(
        await _query_em(
            memory, prefixed_q, vector_search_limit=10, expand_context=0
        )
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer_hits
    ]
    return format_primer_context(primer_segments)


async def _generate_cues_cached(
    *,
    prompt: str,
    llm_cache: _MergedLLMCache,
    openai_client,
    max_cues: int,
) -> tuple[list[str], bool]:
    """Return (cues, cache_hit)."""
    cached = llm_cache.get(V2F_MODEL, prompt)
    if cached is None:
        if openai_client is None:
            return [], False
        resp = await openai_client.chat.completions.create(
            model=V2F_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cached = resp.choices[0].message.content or ""
        llm_cache.put(V2F_MODEL, prompt, cached)
        cues = parse_speaker_cues(cached, max_cues=max_cues)
        return cues, False
    return parse_speaker_cues(cached, max_cues=max_cues), True


async def _retrieve_with_cues(
    memory: EventMemory,
    *,
    question: str,
    cues: list[str],
    K: int,
    expand_context: int,
    merge: str = "max",
) -> list[EMHit]:
    """Primer (prefixed query) + cue retrieval, merged by max or sum."""
    prefixed_q = _ensure_user_prefix(question)
    vsl = max(K, 20)
    primer_batch = await _query_em(
        memory, prefixed_q, vector_search_limit=vsl, expand_context=expand_context
    )
    cue_batches: list[list[EMHit]] = []
    for cue in cues:
        cue_text = _ensure_user_prefix(cue)
        cue_batches.append(
            await _query_em(
                memory,
                cue_text,
                vector_search_limit=vsl,
                expand_context=expand_context,
            )
        )

    if merge == "max":
        merged = _merge_by_max_score([primer_batch, *cue_batches])
        return merged[:K]
    # sum_cosine merge
    score_sum: dict[int, float] = {}
    repr_hit: dict[int, EMHit] = {}
    for batch in [primer_batch, *cue_batches]:
        seen_in_batch: set[int] = set()
        for h in batch:
            if h.turn_id in seen_in_batch:
                continue
            seen_in_batch.add(h.turn_id)
            score_sum[h.turn_id] = score_sum.get(h.turn_id, 0.0) + h.score
            if h.turn_id not in repr_hit:
                repr_hit[h.turn_id] = h
    ranked = sorted(
        [
            EMHit(
                turn_id=tid,
                score=score_sum[tid],
                seed_segment_uuid=repr_hit[tid].seed_segment_uuid,
                role=repr_hit[tid].role,
                text=repr_hit[tid].text,
            )
            for tid in score_sum
        ],
        key=lambda h: -h.score,
    )
    return ranked[:K]


# -------------------- Variant wrappers --------------------


async def em_v2f_lme_userformat(
    memory, question, *, K, cache, openai_client
):
    context_section = await _primer_and_context(memory, question)
    prompt = V2F_LME_USERFORMAT_PROMPT.format(
        question=question, context_section=context_section
    )
    cues, cache_hit = await _generate_cues_cached(
        prompt=prompt, llm_cache=cache, openai_client=openai_client, max_cues=2
    )
    hits = await _retrieve_with_cues(
        memory,
        question=question,
        cues=cues,
        K=K,
        expand_context=EXPAND_CONTEXT,
        merge="max",
    )
    return hits, {"cues": cues, "cache_hit": cache_hit}


async def em_v2f_lme_user_only(
    memory, question, *, K, cache, openai_client
):
    context_section = await _primer_and_context(memory, question)
    prompt = V2F_LME_USER_ONLY_PROMPT.format(
        question=question, context_section=context_section
    )
    cues, cache_hit = await _generate_cues_cached(
        prompt=prompt, llm_cache=cache, openai_client=openai_client, max_cues=2
    )
    # Force User: prefix normalization (parser already does this if missing).
    hits = await _retrieve_with_cues(
        memory,
        question=question,
        cues=cues,
        K=K,
        expand_context=EXPAND_CONTEXT,
        merge="max",
    )
    return hits, {"cues": cues, "cache_hit": cache_hit}


async def em_v2f_lme_mixed_7030(
    memory, question, *, K, cache, openai_client
):
    context_section = await _primer_and_context(memory, question)
    prompt = V2F_LME_MIXED_7030_PROMPT.format(
        question=question, context_section=context_section
    )
    cues, cache_hit = await _generate_cues_cached(
        prompt=prompt, llm_cache=cache, openai_client=openai_client, max_cues=3
    )
    hits = await _retrieve_with_cues(
        memory,
        question=question,
        cues=cues,
        K=K,
        expand_context=EXPAND_CONTEXT,
        merge="max",
    )
    return hits, {"cues": cues, "cache_hit": cache_hit}


async def em_type_enumerated_lme_retuned(
    memory, question, *, K, cache, openai_client
):
    context_section = await _primer_and_context(memory, question)
    prompt = TYPE_ENUM_LME_RETUNED_PROMPT.format(
        question=question, context_section=context_section
    )
    cues, cache_hit = await _generate_cues_cached(
        prompt=prompt, llm_cache=cache, openai_client=openai_client, max_cues=5
    )
    hits = await _retrieve_with_cues(
        memory,
        question=question,
        cues=cues,
        K=K,
        expand_context=EXPAND_CONTEXT,
        merge="max",
    )
    return hits, {"cues": cues, "cache_hit": cache_hit}


async def em_ens_2_lme_retuned(
    memory, question, *, K, v2f_cache, te_cache, openai_client
):
    context_section = await _primer_and_context(memory, question)

    v2f_prompt = V2F_LME_USERFORMAT_PROMPT.format(
        question=question, context_section=context_section
    )
    v2f_cues, v2f_hit = await _generate_cues_cached(
        prompt=v2f_prompt, llm_cache=v2f_cache, openai_client=openai_client, max_cues=2
    )

    te_prompt = TYPE_ENUM_LME_RETUNED_PROMPT.format(
        question=question, context_section=context_section
    )
    te_cues, te_hit = await _generate_cues_cached(
        prompt=te_prompt, llm_cache=te_cache, openai_client=openai_client, max_cues=5
    )

    all_cues = v2f_cues + te_cues
    hits = await _retrieve_with_cues(
        memory,
        question=question,
        cues=all_cues,
        K=K,
        expand_context=EXPAND_CONTEXT,
        merge="sum",
    )
    return hits, {
        "v2f_cues": v2f_cues,
        "te_cues": te_cues,
        "v2f_cache_hit": v2f_hit,
        "te_cache_hit": te_hit,
    }


# -------------------- Evaluation loop --------------------


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


def load_questions() -> list[dict]:
    with open(HARD_QUESTIONS_JSON) as f:
        return json.load(f)


def load_collections_meta() -> dict:
    with open(COLLECTIONS_JSON) as f:
        return json.load(f)


async def evaluate_one(
    arch_name: str,
    memory: EventMemory,
    question: dict,
    caches: dict[str, _MergedLLMCache],
    openai_client,
    *,
    max_K: int,
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()
    meta: dict = {}
    if arch_name == "em_v2f_lme_userformat":
        hits, meta = await em_v2f_lme_userformat(
            memory, q_text, K=max_K,
            cache=caches["v2f_userformat"], openai_client=openai_client,
        )
    elif arch_name == "em_v2f_lme_user_only":
        hits, meta = await em_v2f_lme_user_only(
            memory, q_text, K=max_K,
            cache=caches["v2f_useronly"], openai_client=openai_client,
        )
    elif arch_name == "em_v2f_lme_mixed_7030":
        hits, meta = await em_v2f_lme_mixed_7030(
            memory, q_text, K=max_K,
            cache=caches["v2f_mixed7030"], openai_client=openai_client,
        )
    elif arch_name == "em_type_enumerated_lme_retuned":
        hits, meta = await em_type_enumerated_lme_retuned(
            memory, q_text, K=max_K,
            cache=caches["te_retuned"], openai_client=openai_client,
        )
    elif arch_name == "em_ens_2_lme_retuned":
        hits, meta = await em_ens_2_lme_retuned(
            memory, q_text, K=max_K,
            v2f_cache=caches["v2f_userformat"],
            te_cache=caches["te_retuned"],
            openai_client=openai_client,
        )
    else:
        raise KeyError(arch_name)
    elapsed = time.monotonic() - t0

    row: dict = {
        "question_id": question["question_id"],
        "category": question.get("category", "unknown"),
        "question": q_text,
        "num_gold": len(gold),
        "n_hits": len(hits),
        "time_s": round(elapsed, 3),
    }
    row.update(meta)

    for K in BUDGETS:
        topk = hits[:K]
        retrieved = {h.turn_id for h in topk}
        row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
    return row


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archs",
        default=(
            "em_v2f_lme_userformat,"
            "em_v2f_lme_user_only,"
            "em_v2f_lme_mixed_7030,"
            "em_type_enumerated_lme_retuned,"
            "em_ens_2_lme_retuned"
        ),
        help="Comma-separated architectures to run.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]

    questions = load_questions()
    if args.limit is not None:
        questions = questions[: args.limit]
    print(f"[em_lme_tuned_eval] n_questions={len(questions)}", flush=True)

    meta = load_collections_meta()
    qid_to_meta = {r["question_id"]: r for r in meta["questions"]}

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sql_url = meta["sql_url"]
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

    caches = {
        "v2f_userformat": _MergedLLMCache(
            reader_paths=[LMETUNE_V2F_USERFORMAT_CACHE],
            writer_path=LMETUNE_V2F_USERFORMAT_CACHE,
        ),
        "v2f_useronly": _MergedLLMCache(
            reader_paths=[LMETUNE_V2F_USERONLY_CACHE],
            writer_path=LMETUNE_V2F_USERONLY_CACHE,
        ),
        "v2f_mixed7030": _MergedLLMCache(
            reader_paths=[LMETUNE_V2F_MIXED7030_CACHE],
            writer_path=LMETUNE_V2F_MIXED7030_CACHE,
        ),
        "te_retuned": _MergedLLMCache(
            reader_paths=[LMETUNE_TE_RETUNED_CACHE],
            writer_path=LMETUNE_TE_RETUNED_CACHE,
        ),
    }

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
    results: dict = {"archs": {}, "budgets": list(BUDGETS), "questions": len(questions)}

    semaphore = asyncio.Semaphore(ARCH_CONCURRENCY)

    async def run_one(arch: str, q: dict) -> dict:
        async with semaphore:
            return await evaluate_one(
                arch,
                memories[q["question_id"]],
                q,
                caches,
                openai_client,
                max_K=max_K,
            )

    try:
        for arch in archs:
            t_arch = time.monotonic()
            tasks = [run_one(arch, q) for q in questions]
            rows = await asyncio.gather(*tasks)
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
            for cat, cr in by_cat.items():
                d = {"n": len(cr)}
                for K in BUDGETS:
                    d[f"mean_r@{K}"] = round(
                        sum(r[f"r@{K}"] for r in cr) / max(len(cr), 1), 4
                    )
                cat_summary[cat] = d
            results["archs"][arch] = {
                "summary": summary,
                "by_category": cat_summary,
                "per_question": rows,
            }

            for c in caches.values():
                c.save()

            cat_str = " ".join(
                f"{c}={cat_summary[c].get('mean_r@50', 0):.3f}"
                for c in sorted(cat_summary)
            )
            print(
                f"[{arch}] n={n} "
                f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
                f"({cat_str}) in {arch_elapsed:.1f}s",
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULTS_JSON}", flush=True)
    write_markdown_report(results)


def write_markdown_report(results: dict) -> None:
    archs = results["archs"]
    md: list[str] = []
    md += [
        "# LME-hard EventMemory: LME-tuned cue variants",
        "",
        "All variants run with `expand_context=3` (the LME winning recipe).",
        "",
        "## References",
        "",
        f"- `em_v2f_userprefix` (expand=0, prior): R@50 = {REF_EM_V2F_USERPREFIX:.3f}",
        f"- `em_v2f_expand_3` (prior leader): R@50 = {REF_EM_V2F_EXPAND_3:.3f}",
        f"- `em_ens_2_userprefix` (regressed, prior): R@50 = {REF_EM_ENS_2_USERPREFIX:.3f}",
        f"- SS v2f reference: R@50 = {REF_SS_V2F:.3f}",
        "",
        "## Per-variant summary",
        "",
        "| Architecture | R@20 | R@50 | Δ vs em_v2f_expand_3 (0.832) | time (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arch, data in archs.items():
        s = data["summary"]
        delta = s["mean_r@50"] - REF_EM_V2F_EXPAND_3
        md.append(
            f"| `{arch}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
            f"{delta:+.4f} | {s['time_s']:.1f} |"
        )

    for K in BUDGETS:
        md += [
            "",
            f"## Recall matrix (R@{K})",
            "",
            "| Architecture | multi-session | single-session-preference | "
            "temporal-reasoning |",
            "| --- | --- | --- | --- |",
        ]
        for arch, data in archs.items():
            bc = data["by_category"]
            md.append(
                f"| `{arch}` | "
                f"{bc.get('multi-session', {}).get(f'mean_r@{K}', 0):.4f} | "
                f"{bc.get('single-session-preference', {}).get(f'mean_r@{K}', 0):.4f} | "
                f"{bc.get('temporal-reasoning', {}).get(f'mean_r@{K}', 0):.4f} |"
            )

    # Sample cues: first 2 questions, one variant.
    md += ["", "## Sample cues (first 2 questions)", ""]
    sample_variants = list(archs.keys())[:2]
    for arch in sample_variants:
        rows = archs[arch]["per_question"][:2]
        md.append(f"### `{arch}`")
        md.append("")
        for r in rows:
            cues = r.get("cues") or r.get("v2f_cues")
            te_cues = r.get("te_cues")
            md.append(f"- Q `{r['question_id']}`: {r['question']}")
            if cues:
                for c in cues:
                    md.append(f"  - CUE: `{c}`")
            if te_cues:
                md.append("  - TE cues:")
                for c in te_cues:
                    md.append(f"    - `{c}`")
            md.append("")

    # Verdict section
    best_name, best_r50 = None, -1.0
    for arch, data in archs.items():
        v = data["summary"]["mean_r@50"]
        if v > best_r50:
            best_name, best_r50 = arch, v
    ens_v = archs.get("em_ens_2_lme_retuned", {}).get("summary", {}).get("mean_r@50", 0.0)
    ens_delta = ens_v - REF_EM_ENS_2_USERPREFIX

    md += [
        "## Verdict",
        "",
        f"- Top variant: `{best_name}` R@50 = {best_r50:.4f} "
        f"(Δ vs em_v2f_expand_3 {REF_EM_V2F_EXPAND_3:.3f} = "
        f"{best_r50 - REF_EM_V2F_EXPAND_3:+.4f})",
        f"- Ensemble recovery: `em_ens_2_lme_retuned` R@50 = {ens_v:.4f} "
        f"(was {REF_EM_ENS_2_USERPREFIX:.3f}, Δ = {ens_delta:+.4f})",
        "",
        "## Outputs",
        "",
        f"- JSON: `{RESULTS_JSON.relative_to(ASSOC_DIR)}`",
        f"- Sources: `em_lme_tuned_cues.py`, `em_lme_tuned_eval.py`",
        "- Caches: `cache/lmetune_v2f_userformat_cache.json`, "
        "`cache/lmetune_v2f_useronly_cache.json`, "
        "`cache/lmetune_v2f_mixed7030_cache.json`, "
        "`cache/lmetune_te_retuned_cache.json`",
    ]

    RESULTS_MD.write_text("\n".join(md))
    print(f"Saved: {RESULTS_MD}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
