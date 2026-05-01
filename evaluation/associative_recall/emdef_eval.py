"""Run the three deferred EventMemory architectures on LoCoMo-30.

Architectures:
  em_two_speaker_filter       : v2f + speaker property_filter topup
  em_two_speaker_query_only   : property_filter on raw query (diagnostic)
  em_alias_expand_v2f         : alias-sibling variants, each -> em_v2f, sum_cosine
  em_gated_no_speaker         : gated-overlay style, 3 channels (no speaker)
  em_meta_router              : route by name-mention (two_speaker vs gated)

Outputs:
  results/em_deferred_archs.json
  results/em_deferred_archs.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv
from em_alias_expand import (
    em_alias_expand_v2f,
    load_alias_groups,
)
from em_architectures import (
    BESTSHOT_LLM_CACHE,
    EM_V2F_LLM_CACHE,
    EMHit,
    _MergedLLMCache,
)
from em_gated_no_speaker import (
    EMDEF_GATED_LLM_CACHE,
    GATED_LLM_CACHE,
    em_gated_no_speaker,
)
from em_two_speaker import (
    classify_speaker_side,
    em_two_speaker_filter,
    em_two_speaker_query_only,
    load_two_speaker_map,
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

EMDEF_V2F_LLM_CACHE = CACHE_DIR / "emdef_v2f_llm_cache.json"
EMDEF_ALIAS_V2F_LLM_CACHE = CACHE_DIR / "emdef_alias_v2f_llm_cache.json"


def load_locomo_questions() -> list[dict]:
    with open(DATA_DIR / "questions_extended.json") as f:
        qs = json.load(f)
    locomo = [q for q in qs if q.get("benchmark") == "locomo"]
    return locomo[:30]


def load_collections_meta() -> dict:
    with open(RESULTS_DIR / "eventmemory_collections.json") as f:
        return json.load(f)


def load_all_segments() -> dict[str, list[tuple[int, str]]]:
    """Load (turn_id, text) for each LoCoMo conv from segments_extended.npz."""
    d = np.load(DATA_DIR / "segments_extended.npz", allow_pickle=True)
    out: dict[str, list[tuple[int, str]]] = {cid: [] for cid in LOCOMO_CONV_IDS}
    for i in range(len(d["texts"])):
        cid = str(d["conversation_ids"][i])
        if cid in out:
            out[cid].append((int(d["turn_ids"][i]), str(d["texts"][i])))
    for cid in out:
        out[cid].sort(key=lambda t: t[0])
    return out


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


async def evaluate_question(
    arch_name: str,
    memory: EventMemory,
    question: dict,
    *,
    speaker_map: dict[str, dict[str, str]],
    alias_groups_by_conv: dict[str, list[list[str]]],
    all_segments_by_conv: dict[str, list[tuple[int, str]]],
    v2f_llm_cache: _MergedLLMCache,
    alias_v2f_llm_cache: _MergedLLMCache,
    router_llm_cache: _MergedLLMCache,
    openai_client,
    max_K: int,
) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()
    meta: dict = {}
    hits: list[EMHit]

    if arch_name == "em_two_speaker_filter":
        r = await em_two_speaker_filter(
            memory,
            q_text,
            conv_id,
            K=max_K,
            speaker_map=speaker_map,
            llm_cache=v2f_llm_cache,
            openai_client=openai_client,
        )
        hits = r.hits
        meta = r.metadata
    elif arch_name == "em_two_speaker_query_only":
        r = await em_two_speaker_query_only(
            memory,
            q_text,
            conv_id,
            K=max_K,
            speaker_map=speaker_map,
        )
        hits = r.hits
        meta = r.metadata
    elif arch_name == "em_alias_expand_v2f":
        ar = await em_alias_expand_v2f(
            memory,
            q_text,
            conv_id,
            K=max_K,
            alias_groups_by_conv=alias_groups_by_conv,
            llm_cache=alias_v2f_llm_cache,
            openai_client=openai_client,
        )
        hits = ar.hits
        meta = ar.metadata
    elif arch_name == "em_gated_no_speaker":
        gr = await em_gated_no_speaker(
            memory,
            q_text,
            conv_id,
            K=max_K,
            alias_groups_by_conv=alias_groups_by_conv,
            all_segments_by_conv=all_segments_by_conv,
            v2f_llm_cache=v2f_llm_cache,
            router_llm_cache=router_llm_cache,
            openai_client=openai_client,
        )
        hits = gr.hits
        meta = gr.metadata
    elif arch_name == "em_meta_router":
        # Dispatch: if query mentions ONE side -> two_speaker_filter; else gated.
        side, _user, _asst, _tokens = classify_speaker_side(
            q_text, conv_id, speaker_map
        )
        if side in ("user", "assistant"):
            r = await em_two_speaker_filter(
                memory,
                q_text,
                conv_id,
                K=max_K,
                speaker_map=speaker_map,
                llm_cache=v2f_llm_cache,
                openai_client=openai_client,
            )
            hits = r.hits
            meta = {"route": "two_speaker_filter", **r.metadata}
        else:
            gr = await em_gated_no_speaker(
                memory,
                q_text,
                conv_id,
                K=max_K,
                alias_groups_by_conv=alias_groups_by_conv,
                all_segments_by_conv=all_segments_by_conv,
                v2f_llm_cache=v2f_llm_cache,
                router_llm_cache=router_llm_cache,
                openai_client=openai_client,
            )
            hits = gr.hits
            meta = {"route": "gated_no_speaker", **gr.metadata}
    else:
        raise KeyError(arch_name)
    elapsed = time.monotonic() - t0

    row: dict = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "gold_turn_ids": sorted(gold),
        "n_hits": len(hits),
        "time_s": round(elapsed, 3),
    }
    # strip big metadata fields for per-question output (keep only useful ones)
    keep_keys = {
        "matched_side",
        "conv_user_name",
        "conv_assistant_name",
        "query_name_tokens",
        "applied_speaker_filter",
        "matched_name",
        "appended_turn_ids",
        "dropped_v2f_turn_ids",
        "v2f_cues",
        "alias_matches",
        "num_variants",
        "fallback",
        "confidences",
        "firing_channels",
        "overlay",
        "route",
    }
    for k, v in meta.items():
        if k in keep_keys:
            row[k] = v
    for K in BUDGETS:
        topk = hits[:K]
        retrieved = {h.turn_id for h in topk}
        row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
        row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
    return row


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archs",
        default=(
            "em_two_speaker_filter,em_two_speaker_query_only,"
            "em_alias_expand_v2f,em_gated_no_speaker,em_meta_router"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]

    collections_meta = load_collections_meta()
    questions = load_locomo_questions()
    if args.limit is not None:
        questions = questions[: args.limit]

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

    # Caches: read shared caches, write to our dedicated emdef_ files.
    v2f_llm_cache = _MergedLLMCache(
        reader_paths=[
            BESTSHOT_LLM_CACHE,
            EM_V2F_LLM_CACHE,
            CACHE_DIR / "meta_llm_cache.json",
            CACHE_DIR / "arch_llm_cache.json",
            EMDEF_V2F_LLM_CACHE,
        ],
        writer_path=EMDEF_V2F_LLM_CACHE,
    )
    alias_v2f_llm_cache = _MergedLLMCache(
        reader_paths=[
            BESTSHOT_LLM_CACHE,
            EM_V2F_LLM_CACHE,
            CACHE_DIR / "meta_llm_cache.json",
            CACHE_DIR / "arch_llm_cache.json",
            CACHE_DIR / "alias_llm_cache.json",
            EMDEF_V2F_LLM_CACHE,
            EMDEF_ALIAS_V2F_LLM_CACHE,
        ],
        writer_path=EMDEF_ALIAS_V2F_LLM_CACHE,
    )
    router_llm_cache = _MergedLLMCache(
        reader_paths=[
            GATED_LLM_CACHE,
            EMDEF_GATED_LLM_CACHE,
        ],
        writer_path=EMDEF_GATED_LLM_CACHE,
    )

    # Open EM per conversation.
    memories: dict[str, EventMemory] = {}
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
        opened_resources.append((coll, part))

    # Load supporting artifacts.
    speaker_map = load_two_speaker_map()
    alias_groups_by_conv = load_alias_groups()
    all_segments_by_conv = load_all_segments()

    max_K = max(BUDGETS)
    results: dict = {
        "archs": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
    }

    for arch in archs:
        rows: list[dict] = []
        t_arch = time.monotonic()
        for q in questions:
            mem = memories[q["conversation_id"]]
            row = await evaluate_question(
                arch,
                mem,
                q,
                speaker_map=speaker_map,
                alias_groups_by_conv=alias_groups_by_conv,
                all_segments_by_conv=all_segments_by_conv,
                v2f_llm_cache=v2f_llm_cache,
                alias_v2f_llm_cache=alias_v2f_llm_cache,
                router_llm_cache=router_llm_cache,
                openai_client=openai_client,
                max_K=max_K,
            )
            rows.append(row)
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

        results["archs"][arch] = {
            "summary": summary,
            "by_category": cat_summary,
            "per_question": rows,
        }

        # Save caches after each arch.
        v2f_llm_cache.save()
        alias_v2f_llm_cache.save()
        router_llm_cache.save()

        print(
            f"[{arch}] n={summary['n']} "
            f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
            f"in {summary['time_s']:.1f}s",
            flush=True,
        )

    # Close.
    for coll, part in opened_resources:
        await segment_store.close_partition(part)
        await vector_store.close_collection(collection=coll)
    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()

    out_json = RESULTS_DIR / "em_deferred_archs.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    # Markdown report.
    em_baseline = {
        "em_cosine_baseline": (0.7333, 0.8833),
        "em_v2f": (0.7417, 0.8833),
        "em_ens_2": (0.7833, 0.8667),
    }
    ss_era = {
        "two_speaker_filter": (0.8917, 0.8917),
        "gated_threshold_0.7": (0.7583, 0.8917),
        "alias_expand_v2f": (0.6944, 0.8806),
        "meta_router": None,
    }

    md_lines = [
        "# EventMemory Deferred Architectures on LoCoMo-30",
        "",
        "## Schema",
        "",
        "- Speaker is baked into embedded text via `MessageContext.source`.",
        "- The filter field name is `context.source` (EM's `property_filter` "
        'API uses `Comparison(field="context.source", op="=", value=<name>)`).',
        "- Per-conversation speaker names (from "
        "`results/conversation_two_speakers.json`):",
        "  - `locomo_conv-26`: user=Caroline, assistant=Melanie",
        "  - `locomo_conv-30`: user=Jon, assistant=Gina",
        "  - `locomo_conv-41`: user=John, assistant=Maria",
        "",
        "## Recall matrix",
        "",
        "| Architecture | R@20 | R@50 |",
        "| --- | --- | --- |",
        f"| em_cosine_baseline (reference) | {em_baseline['em_cosine_baseline'][0]:.4f} | {em_baseline['em_cosine_baseline'][1]:.4f} |",
        f"| em_v2f (reference) | {em_baseline['em_v2f'][0]:.4f} | {em_baseline['em_v2f'][1]:.4f} |",
        f"| em_ens_2 (reference) | {em_baseline['em_ens_2'][0]:.4f} | {em_baseline['em_ens_2'][1]:.4f} |",
    ]
    for arch, data in results["archs"].items():
        s = data["summary"]
        md_lines.append(
            f"| **{arch}** | **{s['mean_r@20']:.4f}** | **{s['mean_r@50']:.4f}** |"
        )
    md_lines += [
        "",
        "## SS-era vs EM-ported",
        "",
        "| SS | SS R@20 | SS R@50 | EM | EM R@20 | EM R@50 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    pair_map = [
        ("two_speaker_filter", "em_two_speaker_filter"),
        ("alias_expand_v2f", "em_alias_expand_v2f"),
        ("gated_threshold_0.7", "em_gated_no_speaker"),
    ]
    for ss_name, em_name in pair_map:
        ss = ss_era.get(ss_name)
        em_s = results["archs"].get(em_name, {}).get("summary")
        if ss and em_s:
            md_lines.append(
                f"| {ss_name} | {ss[0]:.4f} | {ss[1]:.4f} | "
                f"{em_name} | {em_s['mean_r@20']:.4f} | "
                f"{em_s['mean_r@50']:.4f} |"
            )

    # Decision rules
    t_s = results["archs"].get("em_two_speaker_filter", {}).get("summary", {})
    g_s = results["archs"].get("em_gated_no_speaker", {}).get("summary", {})
    m_s = results["archs"].get("em_meta_router", {}).get("summary", {})
    md_lines += [
        "",
        "## Decision rules outcome",
        "",
    ]
    if t_s:
        verdict = (
            "SHIP"
            if t_s.get("mean_r@20", 0) >= 0.88
            else (
                "FOLDED"
                if abs(t_s.get("mean_r@20", 0) - em_baseline["em_cosine_baseline"][0])
                < 0.01
                else "MID"
            )
        )
        md_lines.append(
            f"- `em_two_speaker_filter` R@20 = {t_s['mean_r@20']:.4f} "
            f"(rule: >=0.88 -> SHIP, ~=cosine_baseline -> speaker-baking subsumes) -> **{verdict}**"
        )
    if g_s:
        ship = "SHIP" if g_s.get("mean_r@50", 0) >= 0.90 else "NO"
        md_lines.append(
            f"- `em_gated_no_speaker` R@50 = {g_s['mean_r@50']:.4f} "
            f"(rule: >=0.90 -> multi-channel gating lifts ceiling) -> **{ship}**"
        )
    if m_s:
        best_comp = max(
            t_s.get("mean_r@20", 0) if t_s else 0,
            g_s.get("mean_r@20", 0) if g_s else 0,
        )
        best_50 = max(
            t_s.get("mean_r@50", 0) if t_s else 0,
            g_s.get("mean_r@50", 0) if g_s else 0,
        )
        win20 = m_s.get("mean_r@20", 0) >= best_comp - 0.005
        win50 = m_s.get("mean_r@50", 0) >= best_50 - 0.005
        verdict = "SHIP" if (win20 and win50) else "NO"
        md_lines.append(
            f"- `em_meta_router` R@20/R@50 = "
            f"{m_s['mean_r@20']:.4f}/{m_s['mean_r@50']:.4f} "
            f"(rule: matches or beats both component ceilings) -> **{verdict}**"
        )

    md_lines += [
        "",
        "## Outputs",
        "",
        "- `results/em_deferred_archs.json`",
        "- `results/em_deferred_archs.md`",
        "- Source: `em_two_speaker.py`, `em_alias_expand.py`, "
        "`em_gated_no_speaker.py`, `emdef_eval.py`",
    ]
    out_md = RESULTS_DIR / "em_deferred_archs.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    asyncio.run(main())
