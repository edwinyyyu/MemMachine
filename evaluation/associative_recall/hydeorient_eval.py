"""Evaluate HyDE and orient-then-cue EventMemory architectures on LoCoMo-30.

Five mechanism-different variants (no modifications to framework files or
prior em_*.py files; imports only):

  em_hyde_narrative       -- HyDE paragraph, single probe
  em_hyde_turn_format     -- HyDE rendered as speaker-prefixed turns, per-turn probes
  em_hyde_first_person    -- HyDE first-person chat turn, single probe
  em_orient_brief         -- orientation + 2 speakerformat cues
  em_orient_terminology   -- expected vocabulary + 2 speakerformat cues

Also runs a composition step for any variant that beats the
em_v2f_speakerformat baseline at either K=20 or K=50: the variant +
`property_filter(context.source=<matched_speaker>)` topup, same scheme
as em_two_speaker_filter but swapping v2f for the new variant.

Baselines for comparison (from em_prompt_retune and em_deferred_archs):
  em_v2f_speakerformat    R@20=0.8167 R@50=0.8917
  em_two_speaker_filter   R@20=0.8417 R@50=0.9000
  em_two_speaker_query_only R@20=0.8000 R@50=0.9333

Outputs:
  results/em_hyde_orient.json
  results/em_hyde_orient.md

Usage:
  uv run python evaluation/associative_recall/hydeorient_eval.py
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
    EMHit,
    _MergedLLMCache,
    _dedupe_by_turn_id,
)
from em_hyde_orient import (
    HYDE_NARRATIVE_CACHE,
    HYDE_TURN_CACHE,
    HYDE_FIRST_PERSON_CACHE,
    ORIENT_BRIEF_STAGE1_CACHE,
    ORIENT_BRIEF_STAGE2_CACHE,
    ORIENT_TERM_STAGE1_CACHE,
    ORIENT_TERM_STAGE2_CACHE,
    em_hyde_first_person,
    em_hyde_narrative,
    em_hyde_turn_format,
    em_orient_brief,
    em_orient_terminology,
)
from em_two_speaker import classify_speaker_side, load_two_speaker_map


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")


VARIANTS = [
    "em_hyde_narrative",
    "em_hyde_turn_format",
    "em_hyde_first_person",
    "em_orient_brief",
    "em_orient_terminology",
]


# Published baselines referenced in the prompt.
BASELINES = {
    "em_v2f_speakerformat": {"r@20": 0.8167, "r@50": 0.8917},
    "em_two_speaker_filter": {"r@20": 0.8417, "r@50": 0.9000},
    "em_two_speaker_query_only": {"r@20": 0.8000, "r@50": 0.9333},
}
PRIMARY_BASELINE = "em_v2f_speakerformat"


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


# --------------------------------------------------------------------------
# Variant runner
# --------------------------------------------------------------------------


async def run_variant(
    variant: str,
    memory: EventMemory,
    question: str,
    participants: tuple[str, str],
    *,
    K: int,
    caches: dict[str, _MergedLLMCache],
    openai_client,
):
    if variant == "em_hyde_narrative":
        return await em_hyde_narrative(
            memory, question, participants,
            K=K, cache=caches["em_hyde_narrative"],
            openai_client=openai_client,
        )
    if variant == "em_hyde_turn_format":
        return await em_hyde_turn_format(
            memory, question, participants,
            K=K, cache=caches["em_hyde_turn_format"],
            openai_client=openai_client,
        )
    if variant == "em_hyde_first_person":
        return await em_hyde_first_person(
            memory, question, participants,
            K=K, cache=caches["em_hyde_first_person"],
            openai_client=openai_client,
        )
    if variant == "em_orient_brief":
        return await em_orient_brief(
            memory, question, participants,
            K=K,
            stage1_cache=caches["em_orient_brief_stage1"],
            stage2_cache=caches["em_orient_brief_stage2"],
            openai_client=openai_client,
        )
    if variant == "em_orient_terminology":
        return await em_orient_terminology(
            memory, question, participants,
            K=K,
            stage1_cache=caches["em_orient_terminology_stage1"],
            stage2_cache=caches["em_orient_terminology_stage2"],
            openai_client=openai_client,
        )
    raise KeyError(variant)


# --------------------------------------------------------------------------
# Composition with speaker property_filter
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
    """Same scheme as em_two_speaker_filter but using base_hits as the
    v2f replacement.

    - classify side; if 'user' or 'assistant': drop base hits whose role
      mismatches, run EM.query with property_filter=context.source=<name>,
      append novel filtered hits.
    - otherwise return base_hits unchanged.
    """
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

    prop_filter = Comparison(
        field="context.source", op="=", value=matched_name
    )
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
        "--limit", type=int, default=None,
        help="Only run first N questions (smoke test)",
    )
    parser.add_argument(
        "--skip_composition", action="store_true",
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

    # Dedicated caches.
    caches: dict[str, _MergedLLMCache] = {
        "em_hyde_narrative": _MergedLLMCache(
            reader_paths=[HYDE_NARRATIVE_CACHE],
            writer_path=HYDE_NARRATIVE_CACHE,
        ),
        "em_hyde_turn_format": _MergedLLMCache(
            reader_paths=[HYDE_TURN_CACHE],
            writer_path=HYDE_TURN_CACHE,
        ),
        "em_hyde_first_person": _MergedLLMCache(
            reader_paths=[HYDE_FIRST_PERSON_CACHE],
            writer_path=HYDE_FIRST_PERSON_CACHE,
        ),
        "em_orient_brief_stage1": _MergedLLMCache(
            reader_paths=[ORIENT_BRIEF_STAGE1_CACHE],
            writer_path=ORIENT_BRIEF_STAGE1_CACHE,
        ),
        "em_orient_brief_stage2": _MergedLLMCache(
            reader_paths=[ORIENT_BRIEF_STAGE2_CACHE],
            writer_path=ORIENT_BRIEF_STAGE2_CACHE,
        ),
        "em_orient_terminology_stage1": _MergedLLMCache(
            reader_paths=[ORIENT_TERM_STAGE1_CACHE],
            writer_path=ORIENT_TERM_STAGE1_CACHE,
        ),
        "em_orient_terminology_stage2": _MergedLLMCache(
            reader_paths=[ORIENT_TERM_STAGE2_CACHE],
            writer_path=ORIENT_TERM_STAGE2_CACHE,
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
                    variant, mem, q_text, participants,
                    K=max_K, caches=caches, openai_client=openai_client,
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
            by_cat: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by_cat[r.get("category", "unknown")].append(r)
            cat_summary: dict[str, dict] = {}
            for cat, cat_rows in by_cat.items():
                d = {"n": len(cat_rows)}
                for K in BUDGETS:
                    d[f"mean_r@{K}"] = round(
                        sum(r[f"r@{K}"] for r in cat_rows)
                        / max(len(cat_rows), 1), 4
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
                f"in {summary['time_s']:.1f}s"
            )

        # W/T/L vs baseline (em_v2f_speakerformat). The published scalar
        # baseline is from em_prompt_retune; we don't have per-question
        # numbers here, so W/T/L is against the PUBLISHED SCALAR:
        # "won at this K" = variant recall > baseline recall on that
        # question (threshold-like comparison isn't well-defined so we
        # instead report #questions where variant recall >= published
        # baseline scalar for rough signal; also the per-question
        # average lift).
        for variant in variants:
            v_rows = results["variants"][variant]["per_question"]
            for K in BUDGETS:
                base_scalar = BASELINES[PRIMARY_BASELINE][f"r@{K}"]
                above = sum(1 for r in v_rows if r[f"r@{K}"] > base_scalar)
                equal = sum(1 for r in v_rows if r[f"r@{K}"] == base_scalar)
                below = sum(1 for r in v_rows if r[f"r@{K}"] < base_scalar)
                results["variants"][variant].setdefault("per_question_vs_baseline_scalar", {})[f"r@{K}"] = {
                    "above": above, "equal": equal, "below": below,
                    "baseline_scalar": base_scalar,
                }

        # --- Composition step: for any variant that beats v2f_speakerformat
        #     at either K, run with speaker_filter topup and re-score.
        if not args.skip_composition:
            base = BASELINES[PRIMARY_BASELINE]
            lifters = []
            for variant in variants:
                s = results["variants"][variant]["summary"]
                if (s["mean_r@20"] >= base["r@20"]) or (s["mean_r@50"] >= base["r@50"]):
                    lifters.append(variant)
            print(f"Composition candidates: {lifters}")

            for variant in lifters:
                comp_rows: list[dict] = []
                t_comp = time.monotonic()
                for idx, q in enumerate(questions):
                    cid = q["conversation_id"]
                    mem = memories[cid]
                    participants = participants_by_conv[cid]
                    q_text = q["question"]
                    gold = set(q.get("source_chat_ids", []))
                    # Re-run variant (cached) to get ranked hits.
                    t0 = time.monotonic()
                    vr = await run_variant(
                        variant, mem, q_text, participants,
                        K=max_K, caches=caches, openai_client=openai_client,
                    )
                    merged, cmeta = await compose_with_speaker_filter(
                        mem, q_text, cid, vr.hits,
                        K=max_K, speaker_map=speaker_map,
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
                results["compositions"][f"{variant}+speaker_filter"] = {
                    "summary": c_summary,
                    "per_question": comp_rows,
                }
                print(
                    f"[{variant}+speaker_filter] "
                    f"r@20={c_summary['mean_r@20']:.4f} "
                    f"r@50={c_summary['mean_r@50']:.4f} "
                    f"in {c_summary['time_s']:.1f}s"
                )

    finally:
        for c in caches.values():
            c.save()

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
    out_json = RESULTS_DIR / "em_hyde_orient.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    md_lines = build_markdown_report(results, questions, variants)
    out_md = RESULTS_DIR / "em_hyde_orient.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved: {out_md}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


VARIANT_BLURB = {
    "em_hyde_narrative": (
        "HyDE narrative -- LLM writes a 1-2 paragraph narrative retelling "
        "of what the conversation must have contained; embedded as a "
        "single probe."
    ),
    "em_hyde_turn_format": (
        "HyDE turn format -- LLM writes 3-5 speaker-prefixed chat turns; "
        "each is a separate probe unioned by max score."
    ),
    "em_hyde_first_person": (
        "HyDE first person -- LLM writes one first-person \"I remember when "
        "<speaker> said ...\" chat turn; single probe."
    ),
    "em_orient_brief": (
        "Orient brief -- stage 1 writes a 1-sentence orientation describing "
        "what the query is looking for; stage 2 uses the orientation to "
        "generate 2 speakerformat cues."
    ),
    "em_orient_terminology": (
        "Orient terminology -- stage 1 enumerates expected vocabulary "
        "that would appear in the target turns; stage 2 generates 2 "
        "speakerformat cues that include that vocabulary."
    ),
}


def _sample_key(variant: str, row: dict) -> str:
    meta = row.get("metadata", {})
    if variant == "em_hyde_narrative":
        probe = meta.get("probe", "")
        snippet = probe[:280] + ("..." if len(probe) > 280 else "")
        return f"probe: {snippet}"
    if variant == "em_hyde_turn_format":
        turns = meta.get("turns", [])
        rendered = "\n    ".join(t for t in turns[:5])
        return f"turns:\n    {rendered}"
    if variant == "em_hyde_first_person":
        turn = meta.get("turn") or "<none>"
        return f"turn: {turn[:280]}"
    if variant == "em_orient_brief":
        return (
            f"orientation: {meta.get('orientation','')[:200]}\n"
            f"  cues: {meta.get('cues', [])}"
        )
    if variant == "em_orient_terminology":
        return (
            f"vocabulary: {meta.get('vocabulary','')[:200]}\n"
            f"  cues: {meta.get('cues', [])}"
        )
    return str(meta)


def build_markdown_report(
    results: dict, questions: list[dict], variants: list[str]
) -> list[str]:
    base = BASELINES[PRIMARY_BASELINE]
    lines = [
        "# EM HyDE / Orient-then-cue re-test on LoCoMo-30",
        "",
        "## Setup",
        "",
        f"- n_questions = {len(questions)} (benchmark=locomo, first 30)",
        "- EventMemory backend (arc_em_lc30_v1_{26,30,41}); "
        "speaker-baked embedded format `\"{source}: {text}\"`",
        "- Model: text-embedding-3-small + gpt-5-mini (fixed)",
        "- Caches: `cache/hydeorient_<variant>_cache.json` (dedicated)",
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
        "| Variant | R@20 | R@50 | d R@20 vs v2f_sf | d R@50 vs v2f_sf | time (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for v in variants:
        s = results["variants"][v]["summary"]
        d20 = s["mean_r@20"] - base["r@20"]
        d50 = s["mean_r@50"] - base["r@50"]
        lines.append(
            f"| `{v}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
            f"{d20:+.4f} | {d50:+.4f} | {s['time_s']:.1f} |"
        )
    lines += [
        "",
        "Baselines (for reference, from prior runs):",
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
            "| Variant + speaker_filter | R@20 | R@50 |",
            "| --- | --- | --- |",
        ]
        for name, data in results["compositions"].items():
            s = data["summary"]
            lines.append(
                f"| `{name}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} |"
            )
        lines.append("")
        lines.append(
            "Reference: `em_two_speaker_filter` "
            f"R@20={BASELINES['em_two_speaker_filter']['r@20']:.4f}, "
            f"R@50={BASELINES['em_two_speaker_filter']['r@50']:.4f}."
        )
        lines.append("")
        lines.append(
            "Reference: `em_two_speaker_query_only` "
            f"R@50={BASELINES['em_two_speaker_query_only']['r@50']:.4f} "
            "(K=50 leader)."
        )
    else:
        lines.append(
            "_No variant reached the baseline; composition not run._"
        )

    # Sample outputs.
    lines += [
        "",
        "## Sample outputs (2-3 questions, showing mechanism differences)",
        "",
    ]
    # Pick 3 well-spread question indices; require at least one variant has
    # data.
    n = len(questions)
    if n >= 3:
        sample_idxs = [0, n // 2, n - 1]
    else:
        sample_idxs = list(range(n))
    for idx in sample_idxs:
        # Use first variant as anchor for the question text.
        first_rows = results["variants"][variants[0]]["per_question"]
        q_row = first_rows[idx]
        lines.append(
            f"### Q{idx} (`{q_row['conversation_id']}`, "
            f"{q_row.get('category','?')}): "
            f"{q_row['question']!r}"
        )
        lines.append("")
        lines.append(f"Gold turn_ids: {q_row['gold_turn_ids']}")
        lines.append("")
        for v in variants:
            row = results["variants"][v]["per_question"][idx]
            lines.append(
                f"- `{v}` (R@20={row['r@20']:.2f}, R@50={row['r@50']:.2f})"
            )
            lines.append(f"  {_sample_key(v, row)}")
        lines.append("")

    # Verdict.
    lines += [
        "## Verdict",
        "",
    ]
    ship: list[tuple[str, float, float]] = []
    one_side: list[tuple[str, float, float]] = []
    for v in variants:
        s = results["variants"][v]["summary"]
        d20 = s["mean_r@20"] - base["r@20"]
        d50 = s["mean_r@50"] - base["r@50"]
        if d20 >= 0.01 and d50 >= 0.01:
            ship.append((v, d20, d50))
        elif d20 >= 0.01 or d50 >= 0.01:
            one_side.append((v, d20, d50))

    if ship:
        lines.append(
            "**SHIP candidates (>=1pp at both K vs em_v2f_speakerformat):**"
        )
        for v, d20, d50 in ship:
            lines.append(f"- `{v}`: d20={d20:+.4f}, d50={d50:+.4f}")
    elif one_side:
        lines.append(
            "**Narrow (one-sided lift only vs em_v2f_speakerformat):**"
        )
        for v, d20, d50 in one_side:
            lines.append(f"- `{v}`: d20={d20:+.4f}, d50={d50:+.4f}")
    else:
        lines.append(
            "**No variant lifts over em_v2f_speakerformat.** Binning "
            "was correct: on EventMemory where speaker-baked cosine is "
            "already strong, HyDE and orient-then-cue do not add value."
        )

    # Composition ceiling check.
    if results["compositions"]:
        best_name = None
        best_r50 = -1.0
        for name, data in results["compositions"].items():
            r50 = data["summary"]["mean_r@50"]
            if r50 > best_r50:
                best_name = name
                best_r50 = r50
        q_only_r50 = BASELINES["em_two_speaker_query_only"]["r@50"]
        lines.append("")
        if best_name and best_r50 >= q_only_r50:
            lines.append(
                f"**New K=50 ceiling**: `{best_name}` R@50={best_r50:.4f} "
                f">= em_two_speaker_query_only {q_only_r50:.4f}."
            )
        elif best_name:
            lines.append(
                f"Best composition `{best_name}` R@50={best_r50:.4f} "
                f"< em_two_speaker_query_only {q_only_r50:.4f} "
                "(current K=50 leader holds)."
            )

    # HyDE multi-probe vs single-probe.
    lines += [
        "",
        "## HyDE multi-probe vs single-probe pattern",
        "",
    ]
    tf = results["variants"].get("em_hyde_turn_format", {}).get("summary", {})
    nar = results["variants"].get("em_hyde_narrative", {}).get("summary", {})
    fp = results["variants"].get("em_hyde_first_person", {}).get("summary", {})
    if tf and nar and fp:
        lines.append(
            f"- turn_format (multi-probe): "
            f"R@20={tf['mean_r@20']:.4f}, R@50={tf['mean_r@50']:.4f}"
        )
        lines.append(
            f"- narrative (single probe): "
            f"R@20={nar['mean_r@20']:.4f}, R@50={nar['mean_r@50']:.4f}"
        )
        lines.append(
            f"- first_person (single probe): "
            f"R@20={fp['mean_r@20']:.4f}, R@50={fp['mean_r@50']:.4f}"
        )
        if tf["mean_r@50"] > max(nar["mean_r@50"], fp["mean_r@50"]):
            lines.append(
                "Multi-probe wins at K=50 -- consistent with the "
                "session-wide pattern that splitting the probe helps "
                "coverage."
            )
        else:
            lines.append(
                "Multi-probe does NOT win over single-probe HyDE here."
            )

    lines += [
        "",
        "## Outputs",
        "",
        "- `results/em_hyde_orient.json`",
        "- `results/em_hyde_orient.md`",
        "- Source: `em_hyde_orient.py`, `hydeorient_eval.py`",
        "- Caches: `cache/hydeorient_<variant>_cache.json`",
        "",
    ]
    return lines


if __name__ == "__main__":
    asyncio.run(main())
