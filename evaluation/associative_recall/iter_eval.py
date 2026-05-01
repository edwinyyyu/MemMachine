"""Evaluate iterative/agentic SS-era architectures re-ported to EventMemory.

Variants (all use V2F_SPEAKERFORMAT_PROMPT-style cues):
  em_hypothesis_driven_sf
  em_v15_conditional_hop2_sf
  em_v15_rerank_sf
  em_working_memory_buffer_sf
  em_hypothesis_driven_sf_filter
  em_v15_conditional_hop2_sf_filter
  em_v15_rerank_sf_filter
  em_working_memory_buffer_sf_filter

Reads/writes dedicated caches (cache/iter_*_cache.json).
Does NOT re-ingest, does NOT modify framework / em_*.py / agent_architectures.py.

Outputs:
  results/em_iterative_archs.json
  results/em_iterative_archs.md
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
from em_architectures import _MergedLLMCache
from em_iterative_archs import (
    ITER_CH2_CACHE,
    ITER_HD_CACHE,
    ITER_RR_CACHE,
    ITER_WMB_CACHE,
    em_hypothesis_driven_sf,
    em_iterative_with_filter,
    em_v15_conditional_hop2_sf,
    em_v15_rerank_sf,
    em_working_memory_buffer_sf,
)
from em_two_speaker import load_two_speaker_map
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
BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

BASE_VARIANTS = [
    "em_hypothesis_driven_sf",
    "em_v15_conditional_hop2_sf",
    "em_v15_rerank_sf",
    "em_working_memory_buffer_sf",
]
FILTER_VARIANTS = [
    "em_hypothesis_driven_sf_filter",
    "em_v15_conditional_hop2_sf_filter",
    "em_v15_rerank_sf_filter",
    "em_working_memory_buffer_sf_filter",
]
ALL_VARIANTS = BASE_VARIANTS + FILTER_VARIANTS

# Each (base) variant gets a dedicated cache; its _filter counterpart shares
# the same cache (the LLM prompts are identical, the filter is post-hoc).
CACHE_FILES = {
    "em_hypothesis_driven_sf": ITER_HD_CACHE,
    "em_v15_conditional_hop2_sf": ITER_CH2_CACHE,
    "em_v15_rerank_sf": ITER_RR_CACHE,
    "em_working_memory_buffer_sf": ITER_WMB_CACHE,
    "em_hypothesis_driven_sf_filter": ITER_HD_CACHE,
    "em_v15_conditional_hop2_sf_filter": ITER_CH2_CACHE,
    "em_v15_rerank_sf_filter": ITER_RR_CACHE,
    "em_working_memory_buffer_sf_filter": ITER_WMB_CACHE,
}

ARCH_FN = {
    "em_hypothesis_driven_sf": em_hypothesis_driven_sf,
    "em_v15_conditional_hop2_sf": em_v15_conditional_hop2_sf,
    "em_v15_rerank_sf": em_v15_rerank_sf,
    "em_working_memory_buffer_sf": em_working_memory_buffer_sf,
}


def load_questions() -> list[dict]:
    with open(DATA_DIR / "questions_extended.json") as f:
        qs = json.load(f)
    return [q for q in qs if q.get("benchmark") == "locomo"][:30]


def load_collections_meta() -> dict:
    with open(RESULTS_DIR / "eventmemory_collections.json") as f:
        return json.load(f)


def compute_recall(retrieved: set[int], gold: set[int]) -> float:
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


async def evaluate_question(
    variant: str,
    memory: EventMemory,
    question: dict,
    participants: tuple[str, str],
    speaker_map: dict[str, dict[str, str]],
    cache: _MergedLLMCache,
    openai_client,
    *,
    max_K: int,
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))
    cid = question["conversation_id"]

    t0 = time.monotonic()
    if variant.endswith("_filter"):
        base_name = variant[: -len("_filter")]
        arch_fn = ARCH_FN[base_name]
        res = await em_iterative_with_filter(
            memory,
            q_text,
            cid,
            K=max_K,
            participants=participants,
            cache=cache,
            openai_client=openai_client,
            speaker_map=speaker_map,
            arch_fn=arch_fn,
        )
    else:
        arch_fn = ARCH_FN[variant]
        res = await arch_fn(
            memory,
            q_text,
            K=max_K,
            participants=participants,
            cache=cache,
            openai_client=openai_client,
        )
    elapsed = time.monotonic() - t0

    hits = res.hits
    row: dict = {
        "conversation_id": cid,
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "gold_turn_ids": sorted(gold),
        "n_hits": len(hits),
        "time_s": round(elapsed, 3),
        "llm_calls": int(res.metadata.get("llm_calls", 0)),
        "total_cues": int(res.metadata.get("total_cues", 0)),
    }
    # Retain filter application flag if present.
    if "filter_matched_side" in res.metadata:
        row["filter_matched_side"] = res.metadata["filter_matched_side"]
        row["filter_applied"] = res.metadata.get("filter_applied_speaker_filter", False)
    for K in BUDGETS:
        topk = hits[:K]
        retrieved = {h.turn_id for h in topk}
        row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
        row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
    return row


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default=",".join(ALL_VARIANTS),
        help="Comma-separated variants to run (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run first N questions (smoke test).",
    )
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    collections_meta = load_collections_meta()
    questions = load_questions()
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

    speaker_map = load_two_speaker_map()

    # Per-variant caches (filter variants share the base's cache).
    caches: dict[str, _MergedLLMCache] = {}
    unique_caches: dict[Path, _MergedLLMCache] = {}
    for variant in ALL_VARIANTS:
        path = CACHE_FILES[variant]
        if path not in unique_caches:
            unique_caches[path] = _MergedLLMCache(reader_paths=[path], writer_path=path)
        caches[variant] = unique_caches[path]

    # Open EM per conversation.
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
        "variants": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
    }

    for variant in variants:
        if variant not in ALL_VARIANTS:
            print(f"[warn] skipping unknown variant {variant}")
            continue
        rows: list[dict] = []
        t_variant = time.monotonic()
        for q in questions:
            cid = q["conversation_id"]
            mem = memories[cid]
            participants = participants_by_conv[cid]
            row = await evaluate_question(
                variant,
                mem,
                q,
                participants,
                speaker_map,
                caches[variant],
                openai_client,
                max_K=max_K,
            )
            rows.append(row)
        elapsed = time.monotonic() - t_variant

        n = len(rows)
        summary = {
            "n": n,
            "time_s": round(elapsed, 1),
            "mean_llm_calls": round(sum(r["llm_calls"] for r in rows) / max(n, 1), 2),
        }
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
                    sum(r[f"r@{K}"] for r in cat_rows) / max(len(cat_rows), 1),
                    4,
                )
            cat_summary[cat] = d

        results["variants"][variant] = {
            "summary": summary,
            "by_category": cat_summary,
            "per_question": rows,
        }
        # Save cache after each variant.
        caches[variant].save()
        print(
            f"[{variant}] n={summary['n']} "
            f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
            f"llm={summary['mean_llm_calls']:.2f} in {summary['time_s']:.1f}s"
        )

    # W/T/L vs em_v2f_speakerformat (baseline) -- pulled from
    # results/em_prompt_retune.json if available. For each variant we also
    # compute W/T/L vs em_two_speaker_filter (0.8417 / 0.9000) when comparing
    # *_filter variants.

    baselines = _load_reference_baselines()

    for variant in variants:
        if variant not in results["variants"]:
            continue
        v_rows = results["variants"][variant]["per_question"]
        # vs v2f speakerformat (if available)
        if "v2f_speakerformat" in baselines:
            base_rows = baselines["v2f_speakerformat"]
            wtl = _wtl_per_question(v_rows, base_rows)
            results["variants"][variant]["wtl_vs_v2f_speakerformat"] = wtl
        # vs two_speaker_filter for the _filter variants
        if variant.endswith("_filter") and "two_speaker_filter" in baselines:
            base_rows = baselines["two_speaker_filter"]
            wtl = _wtl_per_question(v_rows, base_rows)
            results["variants"][variant]["wtl_vs_two_speaker_filter"] = wtl

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
    out_json = RESULTS_DIR / "em_iterative_archs.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    md_lines = build_markdown_report(results, questions, baselines)
    out_md = RESULTS_DIR / "em_iterative_archs.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved: {out_md}")


# --------------------------------------------------------------------------
# W/T/L + markdown report helpers
# --------------------------------------------------------------------------


def _load_reference_baselines() -> dict[str, list[dict]]:
    """Load per-question rows for em_v2f_speakerformat (from em_prompt_retune)
    and em_two_speaker_filter (from em_deferred_archs), keyed by
    (conversation_id, question_index)."""
    out: dict[str, list[dict]] = {}
    retune_path = RESULTS_DIR / "em_prompt_retune.json"
    if retune_path.exists():
        try:
            with open(retune_path) as f:
                data = json.load(f)
            rows = (
                data.get("variants", {})
                .get("v2f_em_speakerformat", {})
                .get("per_question", [])
            )
            if rows:
                out["v2f_speakerformat"] = rows
        except Exception:
            pass
    deferred_path = RESULTS_DIR / "em_deferred_archs.json"
    if deferred_path.exists():
        try:
            with open(deferred_path) as f:
                data = json.load(f)
            rows = (
                data.get("archs", {})
                .get("em_two_speaker_filter", {})
                .get("per_question", [])
            )
            if rows:
                out["two_speaker_filter"] = rows
        except Exception:
            pass
    return out


def _wtl_per_question(v_rows: list[dict], base_rows: list[dict]) -> dict:
    base_idx = {(r["conversation_id"], r["question_index"]): r for r in base_rows}
    out: dict[str, dict[str, int]] = {}
    for K in BUDGETS:
        w = t = l = 0
        for r in v_rows:
            key = (r["conversation_id"], r["question_index"])
            br = base_idx.get(key)
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
        out[f"r@{K}"] = {"W": w, "T": t, "L": l}
    return out


# --- Prior SS (SegmentStore) numbers for side-by-side reference. -----------
SS_NUMBERS = {
    "hypothesis_driven": {"r@20": None, "r@50": 0.842},
    "v15_conditional_hop2": {"r@20": None, "r@50": 0.822},
    "v15_rerank": {"r@20": 0.772, "r@50": 0.772},
    "working_memory_buffer": {"r@20": None, "r@50": 0.717},
}

# Prior EM baselines.
EM_BASELINES = {
    "em_v2f_speakerformat": {"r@20": 0.8167, "r@50": 0.8917},
    "em_two_speaker_filter": {"r@20": 0.8417, "r@50": 0.9000},
    "em_two_speaker_query_only": {"r@20": 0.8000, "r@50": 0.9333},
}


def build_markdown_report(
    results: dict, questions: list[dict], baselines: dict
) -> list[str]:
    lines = [
        "# Iterative/Agentic Architectures on EventMemory (speakerformat)",
        "",
        "## Setup",
        "",
        f"- n_questions = {len(questions)} (benchmark=locomo, first 30)",
        "- Backend: EventMemory (`text-embedding-3-small`, `gpt-5-mini`, "
        "`max_text_chunk_length=500`, `derive_sentences=False`, "
        "`reranker=None`).",
        '- Speaker-baked embeddings: `"{source}: {text}"`. All cue-gen '
        "prompts use the V2F_SPEAKERFORMAT style (cues must start with "
        '`"<speaker_name>: "`).',
        "- `*_filter` variants apply `property_filter(context.source=<speaker>)`"
        " post-hoc when the query mentions one participant (mirrors "
        "`em_two_speaker_filter`).",
        "- Dedicated caches: `cache/iter_{hypothesis_driven,v15_conditional_"
        "hop2,v15_rerank,working_memory_buffer}_sf_cache.json`.",
        "",
        "## Prior SS-era baselines (reference, on LoCoMo K=50)",
        "",
        "| SS arch | SS K=20 | SS K=50 |",
        "| --- | --- | --- |",
        "| hypothesis_driven | n/a | 0.842 |",
        "| v15_conditional_hop2 | n/a | 0.822 |",
        "| v15_rerank | 0.772 | 0.772 |",
        "| working_memory_buffer | n/a | 0.717 |",
        "",
        "## Prior EM baselines (for direct comparison)",
        "",
        "| EM baseline | R@20 | R@50 |",
        "| --- | --- | --- |",
        "| em_v2f_speakerformat | 0.8167 | 0.8917 |",
        "| em_two_speaker_filter (v2f+filter) | 0.8417 | 0.9000 |",
        "| em_two_speaker_query_only | 0.8000 | 0.9333 |",
        "",
        "## Results (this run)",
        "",
        "| Variant | R@20 | R@50 | avg LLM calls/query | time (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for variant in ALL_VARIANTS:
        if variant not in results["variants"]:
            continue
        s = results["variants"][variant]["summary"]
        lines.append(
            f"| `{variant}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
            f"{s['mean_llm_calls']:.2f} | {s['time_s']:.1f} |"
        )

    # W/T/L tables vs the two reference baselines.
    lines += [
        "",
        "## W/T/L vs em_v2f_speakerformat (all variants)",
        "",
        "| Variant | K=20 W/T/L | K=50 W/T/L |",
        "| --- | --- | --- |",
    ]
    for variant in ALL_VARIANTS:
        v = results["variants"].get(variant, {})
        wtl = v.get("wtl_vs_v2f_speakerformat", {})
        if not wtl:
            continue
        w20 = wtl.get("r@20", {"W": 0, "T": 0, "L": 0})
        w50 = wtl.get("r@50", {"W": 0, "T": 0, "L": 0})
        lines.append(
            f"| `{variant}` | "
            f"{w20['W']}/{w20['T']}/{w20['L']} | "
            f"{w50['W']}/{w50['T']}/{w50['L']} |"
        )

    lines += [
        "",
        "## W/T/L vs em_two_speaker_filter (_filter variants)",
        "",
        "| Variant | K=20 W/T/L | K=50 W/T/L |",
        "| --- | --- | --- |",
    ]
    for variant in FILTER_VARIANTS:
        v = results["variants"].get(variant, {})
        wtl = v.get("wtl_vs_two_speaker_filter", {})
        if not wtl:
            continue
        w20 = wtl.get("r@20", {"W": 0, "T": 0, "L": 0})
        w50 = wtl.get("r@50", {"W": 0, "T": 0, "L": 0})
        lines.append(
            f"| `{variant}` | "
            f"{w20['W']}/{w20['T']}/{w20['L']} | "
            f"{w50['W']}/{w50['T']}/{w50['L']} |"
        )

    lines += [
        "",
        "## Deltas vs em_v2f_speakerformat / em_two_speaker_filter / em_two_speaker_query_only",
        "",
        "| Variant | dR@20 vs v2f_sf | dR@50 vs v2f_sf | dR@20 vs two_sf_filter | dR@50 vs two_sf_filter |",
        "| --- | --- | --- | --- | --- |",
    ]
    v2f_sf_20 = EM_BASELINES["em_v2f_speakerformat"]["r@20"]
    v2f_sf_50 = EM_BASELINES["em_v2f_speakerformat"]["r@50"]
    ts_20 = EM_BASELINES["em_two_speaker_filter"]["r@20"]
    ts_50 = EM_BASELINES["em_two_speaker_filter"]["r@50"]
    for variant in ALL_VARIANTS:
        if variant not in results["variants"]:
            continue
        s = results["variants"][variant]["summary"]
        d20_v2f = s["mean_r@20"] - v2f_sf_20
        d50_v2f = s["mean_r@50"] - v2f_sf_50
        d20_ts = s["mean_r@20"] - ts_20
        d50_ts = s["mean_r@50"] - ts_50
        lines.append(
            f"| `{variant}` | {d20_v2f:+.4f} | {d50_v2f:+.4f} | "
            f"{d20_ts:+.4f} | {d50_ts:+.4f} |"
        )

    # Cost efficiency.
    lines += [
        "",
        "## Cost efficiency: pp-gain-per-extra-LLM-call vs em_v2f_speakerformat",
        "",
        "em_v2f_speakerformat uses 1 LLM call/query (v2f prompt). Extra LLM "
        "calls above that are the iterative overhead.",
        "",
        "| Variant | extra LLM calls | dR@50 | dR@50 per extra call |",
        "| --- | --- | --- | --- |",
    ]
    for variant in ALL_VARIANTS:
        if variant not in results["variants"]:
            continue
        s = results["variants"][variant]["summary"]
        extra = max(s["mean_llm_calls"] - 1.0, 0.0)
        d50 = s["mean_r@50"] - v2f_sf_50
        ratio = d50 / extra if extra > 0 else float("inf") if d50 != 0 else 0.0
        ratio_s = "n/a" if extra == 0 else f"{ratio:+.4f}"
        lines.append(f"| `{variant}` | {extra:.2f} | {d50:+.4f} | {ratio_s} |")

    # Verdict block (derived).
    best_variant_r20 = None
    best_variant_r50 = None
    for variant in ALL_VARIANTS:
        if variant not in results["variants"]:
            continue
        s = results["variants"][variant]["summary"]
        if (
            best_variant_r20 is None
            or s["mean_r@20"]
            > results["variants"][best_variant_r20]["summary"]["mean_r@20"]
        ):
            best_variant_r20 = variant
        if (
            best_variant_r50 is None
            or s["mean_r@50"]
            > results["variants"][best_variant_r50]["summary"]["mean_r@50"]
        ):
            best_variant_r50 = variant

    lines += [
        "",
        "## Findings",
        "",
        f"- Best R@20 variant: `{best_variant_r20}` = "
        f"{results['variants'][best_variant_r20]['summary']['mean_r@20']:.4f}"
        if best_variant_r20
        else "- (no variants ran)",
        f"- Best R@50 variant: `{best_variant_r50}` = "
        f"{results['variants'][best_variant_r50]['summary']['mean_r@50']:.4f}"
        if best_variant_r50
        else "",
        "",
        "Ceilings to beat:",
        "  - R@20: em_two_speaker_filter = 0.8417",
        "  - R@50: em_two_speaker_query_only = 0.9333",
        "",
        "## Outputs",
        "",
        "- `results/em_iterative_archs.json`",
        "- `results/em_iterative_archs.md`",
        "- Source: `em_iterative_archs.py`, `iter_eval.py`",
        "- Caches: `cache/iter_{hypothesis_driven,v15_conditional_hop2,"
        "v15_rerank,working_memory_buffer}_sf_cache.json`",
        "",
    ]
    return lines


if __name__ == "__main__":
    asyncio.run(main())
