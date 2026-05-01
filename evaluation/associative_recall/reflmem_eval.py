"""Evaluate reflective-memory EventMemory architectures on LoCoMo-30.

Three variants (no modifications to framework files or prior em_*.py files):

  reflmem_1round         -- reflect once, write scratch, re-probe corpus
                            (ablation: does writing help without iteration?)
  reflmem_2round         -- two full rounds: cue-gen -> reflect -> scratch
                            write -> round-2 cue-gen informed by scratch
  reflmem_2round_filter  -- reflmem_2round + property_filter(context.source)
                            topup (same scheme as em_hyde_first_person +
                            speaker_filter)

Baselines for comparison (from em_hyde_orient and earlier runs):
  em_v2f_speakerformat               R@20=0.8167 R@50=0.8917
  em_hyde_first_person+speaker_filter R@20=0.8500 R@50=0.9417 (current
                                       LoCoMo ceiling)
  em_two_speaker_query_only          R@50=0.9333

Outputs:
  results/reflective_memory.json
  results/reflective_memory.md

Usage:
  uv run python evaluation/associative_recall/reflmem_eval.py
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
from em_architectures import (
    EMHit,
    _dedupe_by_turn_id,
    _MergedLLMCache,
)
from em_two_speaker import classify_speaker_side, load_two_speaker_map
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
from qdrant_client import AsyncQdrantClient
from reflective_memory import (
    REFLMEM_CUEGEN_R1_CACHE,
    REFLMEM_CUEGEN_R2_CACHE,
    REFLMEM_REFLECT_CACHE,
    reflmem_1round,
    reflmem_2round,
)
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")


VARIANTS = [
    "reflmem_1round",
    "reflmem_2round",
]


BASELINES = {
    "em_v2f_speakerformat": {"r@20": 0.8167, "r@50": 0.8917},
    "em_hyde_first_person+speaker_filter": {"r@20": 0.8500, "r@50": 0.9417},
    "em_hypothesis_driven+speaker_filter": {"r@20": 0.8080, "r@50": 0.9330},
    "em_two_speaker_query_only": {"r@20": 0.8000, "r@50": 0.9333},
}
PRIMARY_BASELINE = "em_v2f_speakerformat"
CEILING_BASELINE = "em_hyde_first_person+speaker_filter"


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


async def run_variant(
    variant: str,
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    embedder,
    caches: dict[str, _MergedLLMCache],
    openai_client,
):
    if variant == "reflmem_1round":
        return await reflmem_1round(
            memory,
            question,
            participants,
            K=K,
            embedder=embedder,
            cuegen_r1_cache=caches["reflmem_cuegen_r1"],
            reflect_cache=caches["reflmem_reflect"],
            openai_client=openai_client,
        )
    if variant == "reflmem_2round":
        return await reflmem_2round(
            memory,
            question,
            participants,
            K=K,
            embedder=embedder,
            cuegen_r1_cache=caches["reflmem_cuegen_r1"],
            reflect_cache=caches["reflmem_reflect"],
            cuegen_r2_cache=caches["reflmem_cuegen_r2"],
            openai_client=openai_client,
        )
    raise KeyError(variant)


# --------------------------------------------------------------------------
# Composition with speaker property_filter (same scheme as hydeorient)
# --------------------------------------------------------------------------


async def compose_with_speaker_filter(
    memory: EventMemory,
    question: str,
    conversation_id: str,
    base_hits: list[EMHit],
    *,
    K: int,
    speaker_map: dict[str, dict[str, str]],
    topup_extra: int = 10,
) -> tuple[list[EMHit], dict]:
    side, user_name, asst_name, name_tokens = classify_speaker_side(
        question, conversation_id, speaker_map
    )
    meta: dict = {
        "matched_side": side,
        "applied_speaker_filter": False,
    }
    if side not in ("user", "assistant"):
        return list(base_hits[:K]), meta

    matched_name = user_name if side == "user" else asst_name
    matched_role = side
    meta["applied_speaker_filter"] = True
    meta["matched_name"] = matched_name

    prop_filter = Comparison(field="context.source", op="=", value=matched_name)
    q_resp = await memory.query(
        query=question,
        vector_search_limit=K + topup_extra,
        expand_context=0,
        property_filter=prop_filter,
    )
    filtered: list[EMHit] = []
    for sc in q_resp.scored_segment_contexts:
        for seg in sc.segments:
            filtered.append(
                EMHit(
                    turn_id=int(seg.properties.get("turn_id", -1)),
                    score=sc.score,
                    seed_segment_uuid=sc.seed_segment_uuid,
                    role=str(seg.properties.get("role", "")),
                    text=seg.block.text,
                )
            )
    filtered = _dedupe_by_turn_id(filtered)

    kept = [h for h in base_hits if h.role == matched_role]
    dropped = [h.turn_id for h in base_hits if h.role != matched_role]
    meta["dropped_base_turn_ids"] = dropped
    seen = {h.turn_id for h in kept}
    appended: list[EMHit] = []
    for h in filtered:
        if h.turn_id in seen:
            continue
        appended.append(h)
        seen.add(h.turn_id)
    meta["appended_turn_ids"] = [h.turn_id for h in appended]

    merged = kept + appended
    return merged[:K], meta


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated variants to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run first N questions (smoke test)",
    )
    parser.add_argument(
        "--skip_composition",
        action="store_true",
        help="Skip the speaker_filter composition step",
    )
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    collections_meta = load_collections_meta()
    questions = load_questions()
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

    caches: dict[str, _MergedLLMCache] = {
        "reflmem_cuegen_r1": _MergedLLMCache(
            reader_paths=[REFLMEM_CUEGEN_R1_CACHE],
            writer_path=REFLMEM_CUEGEN_R1_CACHE,
        ),
        "reflmem_reflect": _MergedLLMCache(
            reader_paths=[REFLMEM_REFLECT_CACHE],
            writer_path=REFLMEM_REFLECT_CACHE,
        ),
        "reflmem_cuegen_r2": _MergedLLMCache(
            reader_paths=[REFLMEM_CUEGEN_R2_CACHE],
            writer_path=REFLMEM_CUEGEN_R2_CACHE,
        ),
    }

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
        "compositions": {},
        "budgets": list(BUDGETS),
        "questions": len(questions),
        "baselines": BASELINES,
    }

    try:
        for variant in variants:
            rows: list[dict] = []
            t_variant = time.monotonic()
            for q in questions:
                cid = q["conversation_id"]
                mem = memories[cid]
                participants = participants_by_conv[cid]
                q_text = q["question"]
                gold = set(q.get("source_chat_ids", []))
                t0 = time.monotonic()
                vr = await run_variant(
                    variant,
                    mem,
                    q_text,
                    participants,
                    K=max_K,
                    embedder=embedder,
                    caches=caches,
                    openai_client=openai_client,
                )
                elapsed = time.monotonic() - t0

                row: dict = {
                    "conversation_id": cid,
                    "category": q.get("category", "unknown"),
                    "question_index": q.get("question_index", -1),
                    "question": q_text,
                    "gold_turn_ids": sorted(gold),
                    "time_s": round(elapsed, 3),
                    "metadata": vr.metadata,
                    "hits_turn_ids@50": [h.turn_id for h in vr.hits[:50]],
                }
                for K in BUDGETS:
                    topk = vr.hits[:K]
                    retrieved = {h.turn_id for h in topk}
                    row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
                    row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)

                rows.append(row)

            elapsed_variant = time.monotonic() - t_variant
            n = len(rows)
            summary = {"n": n, "time_s": round(elapsed_variant, 1)}
            for K in BUDGETS:
                summary[f"mean_r@{K}"] = round(
                    sum(r[f"r@{K}"] for r in rows) / max(n, 1), 4
                )
            # Agg meta stats.
            scratch_counts = [r["metadata"].get("scratch_entries", 0) for r in rows]
            summary["avg_scratch_entries"] = round(
                sum(scratch_counts) / max(len(scratch_counts), 1), 2
            )
            rounds = [r["metadata"].get("rounds_executed", 1) for r in rows]
            summary["avg_rounds"] = round(sum(rounds) / max(len(rounds), 1), 2)
            if variant == "reflmem_2round":
                novel_all = [r["metadata"].get("n_novel_turns_round2", 0) for r in rows]
                summary["avg_novel_turns_round2"] = round(
                    sum(novel_all) / max(len(novel_all), 1), 2
                )
            if variant == "reflmem_1round":
                novel_all = [
                    r["metadata"].get("n_novel_turns_from_scratch", 0) for r in rows
                ]
                summary["avg_novel_turns_from_scratch"] = round(
                    sum(novel_all) / max(len(novel_all), 1), 2
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

            # Save caches incrementally.
            for c in caches.values():
                c.save()

            print(
                f"[{variant}] n={summary['n']} "
                f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
                f"scratch={summary.get('avg_scratch_entries', 0):.1f} "
                f"in {summary['time_s']:.1f}s"
            )

        # --- Round-2 gold-novelty: for reflmem_2round, compute per-query
        #     the fraction where round 2 contributed NEW gold over a
        #     round-1-only baseline (we re-run a round-1-only retrieval
        #     using just the primer + round-1 cues; this is cheap: all
        #     LLM calls are cached).
        if "reflmem_2round" in results["variants"]:
            novelty_rows = []
            for q, row in zip(
                questions, results["variants"]["reflmem_2round"]["per_question"]
            ):
                cid = q["conversation_id"]
                mem = memories[cid]
                participants = participants_by_conv[cid]
                q_text = q["question"]
                gold = set(q.get("source_chat_ids", []))
                # Run round-1-only (reflmem_1round without scratch reprobe
                # uses just primer+cues_r1 via its merged, but it ALSO does
                # scratch reprobe. We want a cleaner "round-1 primer+cues"
                # baseline: simulate via running 1round but pretending
                # no reprobe. Simpler: just re-run round-1 using our helper.
                from em_architectures import _merge_by_max_score as _mm
                from reflective_memory import _run_round1_cuegen_and_retrieve

                primer, cues_r1, cue_hits_r1, _ = await _run_round1_cuegen_and_retrieve(
                    mem,
                    q_text,
                    participants,
                    K=50,
                    cache=caches["reflmem_cuegen_r1"],
                    openai_client=openai_client,
                )
                r1_merged = _mm([primer, *cue_hits_r1])
                r1_retrieved_50 = {h.turn_id for h in r1_merged[:50]}
                r2_retrieved_50 = set(row["retrieved_turn_ids@50"])
                r1_gold = r1_retrieved_50 & gold
                r2_gold = r2_retrieved_50 & gold
                novel_gold = r2_gold - r1_gold
                novelty_rows.append(
                    {
                        "conversation_id": cid,
                        "question": q_text,
                        "r1_gold_count@50": len(r1_gold),
                        "r2_gold_count@50": len(r2_gold),
                        "n_novel_gold_round2": len(novel_gold),
                        "novel_gold_turn_ids": sorted(novel_gold),
                        "gold_total": len(gold),
                    }
                )
            total = len(novelty_rows)
            contributed = sum(1 for r in novelty_rows if r["n_novel_gold_round2"] > 0)
            results["round2_gold_novelty"] = {
                "fraction_queries_with_novel_gold": round(
                    contributed / max(total, 1), 4
                ),
                "n_contributed": contributed,
                "n_total": total,
                "per_query": novelty_rows,
            }
            print(
                f"[round2 gold novelty] {contributed}/{total} queries added "
                f"NEW gold at K=50 (frac="
                f"{contributed / max(total, 1):.3f})"
            )

        # --- Composition step: run reflmem_2round + speaker_filter.
        if not args.skip_composition and "reflmem_2round" in results["variants"]:
            variant = "reflmem_2round"
            comp_rows: list[dict] = []
            t_comp = time.monotonic()
            for q in questions:
                cid = q["conversation_id"]
                mem = memories[cid]
                participants = participants_by_conv[cid]
                q_text = q["question"]
                gold = set(q.get("source_chat_ids", []))
                t0 = time.monotonic()
                vr = await run_variant(
                    variant,
                    mem,
                    q_text,
                    participants,
                    K=max_K,
                    embedder=embedder,
                    caches=caches,
                    openai_client=openai_client,
                )
                merged, cmeta = await compose_with_speaker_filter(
                    mem,
                    q_text,
                    cid,
                    vr.hits,
                    K=max_K,
                    speaker_map=speaker_map,
                )
                elapsed = time.monotonic() - t0
                row: dict = {
                    "conversation_id": cid,
                    "category": q.get("category", "unknown"),
                    "question_index": q.get("question_index", -1),
                    "question": q_text,
                    "gold_turn_ids": sorted(gold),
                    "time_s": round(elapsed, 3),
                    "composition_meta": cmeta,
                }
                for K in BUDGETS:
                    topk = merged[:K]
                    retrieved = {h.turn_id for h in topk}
                    row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
                    row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
                comp_rows.append(row)
            elapsed_comp = time.monotonic() - t_comp
            n = len(comp_rows)
            c_summary = {"n": n, "time_s": round(elapsed_comp, 1)}
            for K in BUDGETS:
                c_summary[f"mean_r@{K}"] = round(
                    sum(r[f"r@{K}"] for r in comp_rows) / max(n, 1), 4
                )
            results["compositions"]["reflmem_2round_filter"] = {
                "summary": c_summary,
                "per_question": comp_rows,
            }
            print(
                f"[reflmem_2round_filter] "
                f"r@20={c_summary['mean_r@20']:.4f} "
                f"r@50={c_summary['mean_r@50']:.4f} "
                f"in {c_summary['time_s']:.1f}s"
            )

    finally:
        for c in caches.values():
            c.save()
        for coll, part in opened_resources:
            await segment_store.close_partition(part)
            await vector_store.close_collection(collection=coll)
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "reflective_memory.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    md_lines = build_markdown_report(results, questions, variants)
    out_md = RESULTS_DIR / "reflective_memory.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved: {out_md}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


VARIANT_BLURB = {
    "reflmem_1round": (
        "Single round: LLM cue-gen -> retrieve -> reflect (learned / "
        "still_need) -> write to per-query scratch memory -> top-N "
        "scratch entries re-probe the corpus -> merge."
    ),
    "reflmem_2round": (
        "Two full rounds. Round 1 cue-gen -> retrieve -> reflect -> "
        "write scratch. Round 2 cue-gen is informed by scratch; its "
        "cues + scratch re-probes augment the corpus hits."
    ),
}


def build_markdown_report(
    results: dict, questions: list[dict], variants: list[str]
) -> list[str]:
    base = BASELINES[PRIMARY_BASELINE]
    ceil = BASELINES[CEILING_BASELINE]
    lines = [
        "# Reflective LLM writes-to-memory (LoCoMo-30)",
        "",
        "## Setup",
        "",
        f"- n_questions = {len(questions)} (benchmark=locomo, first 30)",
        "- EventMemory backend (arc_em_lc30_v1_{26,30,41}); "
        'speaker-baked embedded format `"{source}: {text}"`',
        "- Model: text-embedding-3-small + gpt-5-mini (fixed)",
        "- Scratch memory: in-memory numpy cosine; per-query, not persisted",
        "- Caches: `cache/reflmem_{cuegen_r1,reflect,cuegen_r2}_cache.json` "
        "(dedicated)",
        "",
        "## Variants",
        "",
    ]
    for v in variants:
        lines.append(f"- `{v}`: {VARIANT_BLURB.get(v, '')}")
    lines += [
        "",
        "## Recall matrix",
        "",
        "| Variant | R@20 | R@50 | d R@20 vs v2f_sf | d R@50 vs HyDE+sf | avg_rounds | avg_scratch | time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for v in variants:
        s = results["variants"][v]["summary"]
        d20 = s["mean_r@20"] - base["r@20"]
        d50_vs_ceil = s["mean_r@50"] - ceil["r@50"]
        lines.append(
            f"| `{v}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
            f"{d20:+.4f} | {d50_vs_ceil:+.4f} | "
            f"{s.get('avg_rounds', 0):.2f} | "
            f"{s.get('avg_scratch_entries', 0):.2f} | "
            f"{s['time_s']:.1f} |"
        )
    lines += [
        "",
        "Baselines (for reference):",
        "",
    ]
    for k, v in BASELINES.items():
        lines.append(f"- `{k}`: R@20={v['r@20']:.4f}, R@50={v['r@50']:.4f}")

    # Composition section.
    lines += [
        "",
        "## Composition with `property_filter(context.source)`",
        "",
    ]
    if results["compositions"]:
        lines += [
            "| Variant + speaker_filter | R@20 | R@50 | d R@50 vs HyDE+sf |",
            "| --- | --- | --- | --- |",
        ]
        for name, data in results["compositions"].items():
            s = data["summary"]
            d50 = s["mean_r@50"] - ceil["r@50"]
            lines.append(
                f"| `{name}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
                f"{d50:+.4f} |"
            )
        lines.append("")
        lines.append(
            f"Current ceiling: `{CEILING_BASELINE}` "
            f"R@20={ceil['r@20']:.4f}, R@50={ceil['r@50']:.4f}."
        )
    else:
        lines.append("_Composition not run._")

    # Round-2 gold novelty.
    lines += [
        "",
        "## Round 2 gold novelty",
        "",
    ]
    n2 = results.get("round2_gold_novelty")
    if n2:
        lines.append(
            f"- Queries where round 2 added NEW gold over round-1-only: "
            f"**{n2['n_contributed']}/{n2['n_total']}** "
            f"(frac={n2['fraction_queries_with_novel_gold']:.3f})"
        )
        # Show the queries where novelty happened.
        contributors = [r for r in n2["per_query"] if r["n_novel_gold_round2"] > 0]
        if contributors:
            lines.append("")
            lines.append("Queries with novel gold from round 2:")
            lines.append("")
            for r in contributors[:10]:
                lines.append(
                    f"- ({r['conversation_id']}) "
                    f"{r['question']!r} +{r['n_novel_gold_round2']} new gold"
                )
    else:
        lines.append("_No round-2 novelty computed._")

    # Sample outputs: show round-1 retrievals, scratch state, round-2
    # retrievals for two illustrative queries.
    lines += [
        "",
        "## Sample round-trips (reflmem_2round)",
        "",
    ]
    v2 = results["variants"].get("reflmem_2round")
    if v2:
        # Pick two sample questions with varied round-2 novelty.
        rows = v2["per_question"]
        n = len(rows)
        if n >= 2:
            sample_idxs = [0, n // 2]
        else:
            sample_idxs = list(range(n))
        for idx in sample_idxs:
            row = rows[idx]
            meta = row["metadata"]
            lines.append(
                f"### Q{idx} (`{row['conversation_id']}`, "
                f"{row.get('category', '?')}): "
                f"{row['question']!r}"
            )
            lines.append("")
            lines.append(f"Gold turn_ids: {row['gold_turn_ids']}")
            lines.append("")
            lines.append(f"Round-1 cues: {meta.get('cues_r1', [])}")
            lines.append("")
            lines.append("Scratch state after round 1:")
            learned = meta.get("learned", [])
            still_need = meta.get("still_need", [])
            for s in learned:
                lines.append(f"  - [learned] {s}")
            for s in still_need:
                lines.append(f"  - [still_need] {s}")
            lines.append("")
            lines.append(f"Round-2 cues: {meta.get('cues_r2', [])}")
            lines.append(
                f"Scratch reprobe texts (top-3 by cosine): "
                f"{meta.get('reprobe_texts', [])}"
            )
            lines.append(
                f"Scratch reprobe cosine scores: {meta.get('reprobe_scores', [])}"
            )
            lines.append(
                f"R@20={row['r@20']:.2f}, R@50={row['r@50']:.2f}, "
                f"novel_turn_ids_round2={meta.get('n_novel_turns_round2', 0)}"
            )
            lines.append("")

    # Verdict.
    lines += [
        "## Verdict",
        "",
    ]
    verdict_lines: list[str] = []
    # Find the strongest variant + composition.
    comp = results["compositions"].get("reflmem_2round_filter")
    best_r50 = None
    best_name = None
    if comp:
        best_r50 = comp["summary"]["mean_r@50"]
        best_name = "reflmem_2round_filter"
    else:
        for v in variants:
            r50 = results["variants"][v]["summary"]["mean_r@50"]
            if best_r50 is None or r50 > best_r50:
                best_r50 = r50
                best_name = v

    d_vs_ceil = (best_r50 or 0.0) - ceil["r@50"]
    if best_r50 is not None:
        if d_vs_ceil > 1e-4:
            verdict_lines.append(
                f"**New K=50 ceiling**: `{best_name}` R@50={best_r50:.4f} "
                f"> HyDE+speaker_filter {ceil['r@50']:.4f} "
                f"(d={d_vs_ceil:+.4f}). Reflection-as-memory IS real "
                "additive signal."
            )
        elif abs(d_vs_ceil) <= 1e-4:
            verdict_lines.append(
                f"**Ties HyDE+speaker_filter**: `{best_name}` "
                f"R@50={best_r50:.4f} = {ceil['r@50']:.4f}. "
                "Substrate-ceiling hypothesis confirmed: iteration cannot "
                "lift beyond what the corpus embedding geometry allows."
            )
        else:
            verdict_lines.append(
                f"**Loses to HyDE+speaker_filter**: `{best_name}` "
                f"R@50={best_r50:.4f} < {ceil['r@50']:.4f} "
                f"(d={d_vs_ceil:+.4f}). Scratch-memory POLLUTES rather "
                "than guides -- reflections aren't discriminative enough "
                "as probes."
            )

    if n2:
        frac = n2["fraction_queries_with_novel_gold"]
        if frac < 0.10:
            verdict_lines.append(
                f"Round-2 novel-gold rate = {frac:.3f} (<10%): "
                "reflection-as-retrieval-entry adds little incremental "
                "value at the query level."
            )
        else:
            verdict_lines.append(
                f"Round-2 novel-gold rate = {frac:.3f} (>=10%): "
                "at least some queries materially benefit from round 2."
            )
    for ln in verdict_lines:
        lines.append(f"- {ln}")

    lines += [
        "",
        "## Outputs",
        "",
        "- `results/reflective_memory.json`",
        "- `results/reflective_memory.md`",
        "- Source: `reflective_memory.py`, `reflmem_eval.py`",
        "- Caches: `cache/reflmem_{cuegen_r1,reflect,cuegen_r2}_cache.json`",
        "",
    ]
    return lines


if __name__ == "__main__":
    asyncio.run(main())
