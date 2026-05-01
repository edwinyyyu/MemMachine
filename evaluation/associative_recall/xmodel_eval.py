"""Cross-model EM evaluation: gpt-5-nano with structural cue-gen prompts.

Hypothesis: structurally-constrained prompts (speakerformat, HyDE first-
person) close the mini-nano gap because they reduce output surface area.

Variants (nano as primary cue-gen model; EM backend/retrieval unchanged):
  nano_v2f                      - vanilla v2f prompt (control)
  nano_v2f_speakerformat        - speakerformat prompt
  nano_hyde_first_person        - HyDE first-person prompt
  nano_hyde_first_person_filter - HyDE first-person + speaker property_filter

Mini controls are PULLED FROM CACHE (no new spend); we trust the
previously-published numbers:
  em_v2f                            0.742 / 0.883
  em_v2f_speakerformat              0.817 / 0.892
  em_hyde_first_person              0.800 / 0.908
  em_hyde_first_person+speaker_filt 0.850 / 0.942

Reports:
  - recall matrix (nano vs mini x variant x K)
  - gap as % of mini for each variant
  - format-compliance rates
  - sample cues (2 per variant)

Outputs:
  results/xmodel_em.json
  results/xmodel_em.md
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
from em_two_speaker import load_two_speaker_map
from hydeorient_eval import compose_with_speaker_filter
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
from xmodel_em import (
    NANO_HYDE_FP_CACHE,
    NANO_SF_CACHE,
    NANO_V2F_CACHE,
    nano_hyde_first_person,
    nano_v2f,
    nano_v2f_speakerformat,
)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

# Variants to run (nano primary).
VARIANTS = [
    "nano_v2f",
    "nano_v2f_speakerformat",
    "nano_hyde_first_person",
    "nano_hyde_first_person_filter",
]

# Mini controls (from em_prompt_retune / em_hyde_orient; NOT re-run).
MINI_REFERENCE = {
    "nano_v2f": {
        "mini_variant": "em_v2f",
        "mini_r@20": 0.742,
        "mini_r@50": 0.883,
    },
    "nano_v2f_speakerformat": {
        "mini_variant": "em_v2f_speakerformat",
        "mini_r@20": 0.817,
        "mini_r@50": 0.892,
    },
    "nano_hyde_first_person": {
        "mini_variant": "em_hyde_first_person",
        "mini_r@20": 0.800,
        "mini_r@50": 0.908,
    },
    "nano_hyde_first_person_filter": {
        "mini_variant": "em_hyde_first_person+speaker_filter",
        "mini_r@20": 0.850,
        "mini_r@50": 0.942,
    },
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


# --------------------------------------------------------------------------
# Variant runner
# --------------------------------------------------------------------------


async def run_variant(
    variant: str,
    memory: EventMemory,
    question: str,
    conversation_id: str,
    participants: tuple[str, str],
    *,
    K: int,
    caches: dict[str, _MergedLLMCache],
    speaker_map: dict[str, dict[str, str]],
    openai_client,
):
    if variant == "nano_v2f":
        return await nano_v2f(
            memory,
            question,
            K=K,
            cache=caches["nano_v2f"],
            openai_client=openai_client,
        )
    if variant == "nano_v2f_speakerformat":
        return await nano_v2f_speakerformat(
            memory,
            question,
            participants,
            K=K,
            cache=caches["nano_v2f_speakerformat"],
            openai_client=openai_client,
        )
    if variant == "nano_hyde_first_person":
        return await nano_hyde_first_person(
            memory,
            question,
            participants,
            K=K,
            cache=caches["nano_hyde_first_person"],
            openai_client=openai_client,
        )
    if variant == "nano_hyde_first_person_filter":
        # Stage 1: nano_hyde_first_person.
        base = await nano_hyde_first_person(
            memory,
            question,
            participants,
            K=K,
            cache=caches["nano_hyde_first_person"],
            openai_client=openai_client,
        )
        # Stage 2: compose with speaker property_filter (same as mini recipe).
        merged, cmeta = await compose_with_speaker_filter(
            memory,
            question,
            conversation_id,
            base.hits,
            K=K,
            speaker_map=speaker_map,
        )
        # Merge metadata; wrap as XModelEMResult-alike.
        from xmodel_em import XModelEMResult

        return XModelEMResult(
            hits=merged,
            metadata={
                **base.metadata,
                "composition_meta": cmeta,
                "variant": "nano_hyde_first_person_filter",
            },
        )
    raise KeyError(variant)


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

    # Dedicated nano caches.
    caches: dict[str, _MergedLLMCache] = {
        "nano_v2f": _MergedLLMCache(
            reader_paths=[NANO_V2F_CACHE],
            writer_path=NANO_V2F_CACHE,
        ),
        "nano_v2f_speakerformat": _MergedLLMCache(
            reader_paths=[NANO_SF_CACHE],
            writer_path=NANO_SF_CACHE,
        ),
        "nano_hyde_first_person": _MergedLLMCache(
            reader_paths=[NANO_HYDE_FP_CACHE],
            writer_path=NANO_HYDE_FP_CACHE,
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
        "budgets": list(BUDGETS),
        "questions": len(questions),
        "mini_reference": MINI_REFERENCE,
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
                    cid,
                    participants,
                    K=max_K,
                    caches=caches,
                    speaker_map=speaker_map,
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

            # Format-compliance stats.
            compliance = _compute_compliance(variant, rows)

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
                "compliance": compliance,
                "by_category": cat_summary,
                "per_question": rows,
            }

            # Save caches incrementally.
            for c in caches.values():
                c.save()

            print(
                f"[{variant}] n={summary['n']} "
                f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
                f"compliance={compliance.get('rate')} "
                f"in {summary['time_s']:.1f}s"
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
    out_json = RESULTS_DIR / "xmodel_em.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    md_lines = build_markdown_report(results, questions, variants)
    out_md = RESULTS_DIR / "xmodel_em.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved: {out_md}")


# --------------------------------------------------------------------------
# Compliance computation
# --------------------------------------------------------------------------


def _compute_compliance(variant: str, rows: list[dict]) -> dict:
    """Return {rate, numer, denom, description} for structural compliance."""
    numer = 0
    denom = 0
    if variant == "nano_v2f":
        return {
            "rate": None,
            "description": "v2f has no structural constraint (N/A)",
        }
    if variant in ("nano_v2f_speakerformat",):
        for r in rows:
            flags = r.get("metadata", {}).get("format_compliant") or []
            for f in flags:
                denom += 1
                if f:
                    numer += 1
        desc = "fraction of cues starting with '<speaker>: '"
    elif variant in ("nano_hyde_first_person", "nano_hyde_first_person_filter"):
        for r in rows:
            md = r.get("metadata", {})
            denom += 1
            if md.get("format_compliant") is True:
                numer += 1
        desc = "fraction of turns parseable as '<speaker>: <content>'"
    else:
        return {"rate": None, "description": "N/A"}
    rate = round(numer / denom, 4) if denom else None
    return {"rate": rate, "numer": numer, "denom": denom, "description": desc}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _sample_key(variant: str, row: dict) -> str:
    meta = row.get("metadata", {})
    if variant == "nano_v2f" or variant == "nano_v2f_speakerformat":
        cues = meta.get("cues", [])
        rendered = "\n    ".join(f"- {c}" for c in cues[:2])
        return f"cues:\n    {rendered}"
    if variant in ("nano_hyde_first_person", "nano_hyde_first_person_filter"):
        turn = meta.get("turn") or "<none>"
        return f"turn: {turn[:280]}"
    return str(meta)


def build_markdown_report(
    results: dict,
    questions: list[dict],
    variants: list[str],
) -> list[str]:
    lines = [
        "# Cross-model EventMemory cue-gen: gpt-5-nano with structural prompts",
        "",
        "## Hypothesis",
        "",
        "Structural-constraint cue-gen prompts (speakerformat, HyDE "
        "first-person) reduce nano's effective output surface area, "
        "closing the mini-nano gap on EventMemory retrieval.",
        "",
        "## Setup",
        "",
        f"- n_questions = {len(questions)} (benchmark=locomo, first 30)",
        "- EventMemory backend (arc_em_lc30_v1_{26,30,41}); "
        'speaker-baked embedded format `"{source}: {text}"`',
        "- Embedder: text-embedding-3-small (fixed)",
        "- Cue-gen model: gpt-5-nano "
        "(`max_completion_tokens=6000`, 1-retry on empty parse)",
        "- Mini controls: numbers pulled from prior runs "
        "(em_prompt_retune, em_hyde_orient); no new mini spend",
        "- Prompts reused VERBATIM from:",
        "  - `em_architectures.V2F_PROMPT` (vanilla)",
        "  - `em_retuned_cue_gen.V2F_SPEAKERFORMAT_PROMPT` (mini-retuned winner)",
        "  - `em_hyde_orient.HYDE_FIRST_PERSON_PROMPT` (mini K=50 ceiling)",
        "- Caches: `cache/xmodel_<variant>_cache.json` (dedicated)",
        "",
        "## Recall matrix (nano vs mini)",
        "",
        "| Variant | nano R@20 | mini R@20 | nano/mini @20 | "
        "nano R@50 | mini R@50 | nano/mini @50 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for v in variants:
        ref = MINI_REFERENCE[v]
        s = results["variants"][v]["summary"]
        mini_20 = ref["mini_r@20"]
        mini_50 = ref["mini_r@50"]
        nano_20 = s["mean_r@20"]
        nano_50 = s["mean_r@50"]
        ratio_20 = nano_20 / mini_20 if mini_20 else 0.0
        ratio_50 = nano_50 / mini_50 if mini_50 else 0.0
        lines.append(
            f"| `{v}` | {nano_20:.4f} | {mini_20:.4f} | "
            f"{ratio_20:.1%} | {nano_50:.4f} | {mini_50:.4f} | "
            f"{ratio_50:.1%} |"
        )

    lines += [
        "",
        "Mini variants mapped to:",
        "",
    ]
    for v in variants:
        lines.append(f"- `{v}` -> mini `{MINI_REFERENCE[v]['mini_variant']}`")

    lines += [
        "",
        "## Format-compliance (nano)",
        "",
        "| Variant | Rate | Fraction | Description |",
        "| --- | --- | --- | --- |",
    ]
    for v in variants:
        c = results["variants"][v]["compliance"]
        rate = c.get("rate")
        rate_str = f"{rate:.2%}" if isinstance(rate, float) else "N/A"
        numer = c.get("numer", "-")
        denom = c.get("denom", "-")
        frac = f"{numer}/{denom}" if denom != "-" else "-"
        lines.append(f"| `{v}` | {rate_str} | {frac} | {c.get('description', '')} |")

    # Sample cues: pick 2 questions spread across the 30.
    lines += [
        "",
        "## Sample cues (2 questions per variant)",
        "",
    ]
    n = len(questions)
    sample_idxs = [0, n - 1] if n >= 2 else list(range(n))
    for idx in sample_idxs:
        # Use first variant as anchor.
        first_rows = results["variants"][variants[0]]["per_question"]
        q_row = first_rows[idx]
        lines.append(
            f"### Q{idx} (`{q_row['conversation_id']}`, "
            f"{q_row.get('category', '?')}): {q_row['question']!r}"
        )
        lines.append("")
        lines.append(f"Gold turn_ids: {q_row['gold_turn_ids']}")
        lines.append("")
        for v in variants:
            row = results["variants"][v]["per_question"][idx]
            lines.append(f"- `{v}` (R@20={row['r@20']:.2f}, R@50={row['r@50']:.2f})")
            lines.append(f"  {_sample_key(v, row)}")
        lines.append("")

    # Verdict.
    lines += [
        "## Verdict",
        "",
    ]

    def _gap_pct(nano, mini):
        return (nano / mini) if mini else 0.0

    ship_candidates = []
    one_side_candidates = []
    for v in variants:
        s = results["variants"][v]["summary"]
        ref = MINI_REFERENCE[v]
        r20 = _gap_pct(s["mean_r@20"], ref["mini_r@20"])
        r50 = _gap_pct(s["mean_r@50"], ref["mini_r@50"])
        if r50 >= 0.90 and r20 >= 0.90:
            ship_candidates.append((v, r20, r50))
        elif r50 >= 0.90 or r20 >= 0.90:
            one_side_candidates.append((v, r20, r50))

    if ship_candidates:
        lines.append(
            "**>= 90% of mini at BOTH K (structural prompts ARE a "
            "cross-model portability lever):**"
        )
        for v, r20, r50 in ship_candidates:
            lines.append(f"- `{v}`: {r20:.1%}@20, {r50:.1%}@50")
        lines.append("")
    if one_side_candidates:
        lines.append("**>= 90% of mini at ONE K (partial close):**")
        for v, r20, r50 in one_side_candidates:
            lines.append(f"- `{v}`: {r20:.1%}@20, {r50:.1%}@50")
        lines.append("")

    # Decision rules per plan.
    nhf = (
        results["variants"].get("nano_hyde_first_person_filter", {}).get("summary", {})
    )
    nsf = results["variants"].get("nano_v2f_speakerformat", {}).get("summary", {})
    nhfp = results["variants"].get("nano_hyde_first_person", {}).get("summary", {})

    lines.append("### Decision-rule evaluation")
    lines.append("")
    if nhf:
        r50 = nhf.get("mean_r@50", 0.0)
        if r50 >= 0.90:
            lines.append(
                f"- `nano_hyde_first_person_filter` R@50={r50:.4f} "
                ">= 0.90 -> EM LoCoMo recipe is MODEL-PORTABLE "
                "(nano matches mini within 4pp)."
            )
        else:
            lines.append(
                f"- `nano_hyde_first_person_filter` R@50={r50:.4f} "
                "< 0.90 -> composition not yet portable to nano."
            )
    if nsf and nhfp:
        nsf50 = nsf.get("mean_r@50", 0.0)
        nhfp50 = nhfp.get("mean_r@50", 0.0)
        if nsf50 >= 0.85 or nhfp50 >= 0.85:
            lines.append(
                f"- Structural prompts lift nano to "
                f">= 0.85 K=50 "
                f"(speakerformat={nsf50:.4f}, hyde_fp={nhfp50:.4f}) "
                "-> structural prompts ARE a cross-model win."
            )
        else:
            lines.append(
                f"- Neither structural prompt reaches 0.85 K=50 on nano "
                f"(speakerformat={nsf50:.4f}, hyde_fp={nhfp50:.4f})."
            )
    # All below 0.75?
    max_r50 = max(results["variants"][v]["summary"]["mean_r@50"] for v in variants)
    if max_r50 < 0.75:
        lines.append(
            f"- All nano variants below 0.75 K=50 "
            f"(max={max_r50:.4f}) -> nano capacity floor is binding."
        )

    # Gradient: does HyDE lift nano more than speakerformat?
    if nsf and nhfp:
        nsf50 = nsf.get("mean_r@50", 0.0)
        nhfp50 = nhfp.get("mean_r@50", 0.0)
        lines.append("")
        if nhfp50 > nsf50:
            lines.append(
                f"- Gradient: HyDE ({nhfp50:.4f}) > speakerformat "
                f"({nsf50:.4f}) at K=50 -> 'more structure = more "
                "portable' gradient CONFIRMED."
            )
        elif nsf50 > nhfp50:
            lines.append(
                f"- Gradient: speakerformat ({nsf50:.4f}) > HyDE "
                f"({nhfp50:.4f}) at K=50 -> 'more structure = more "
                "portable' gradient NOT confirmed; 2-cue speakerformat "
                "beats single-probe HyDE on nano."
            )
        else:
            lines.append("- Gradient: HyDE vs speakerformat tied at K=50.")

    lines += [
        "",
        "## Outputs",
        "",
        "- `results/xmodel_em.json`",
        "- `results/xmodel_em.md`",
        "- Source: `xmodel_em.py`, `xmodel_eval.py`",
        "- Caches: `cache/xmodel_<variant>_cache.json`",
        "",
    ]
    return lines


if __name__ == "__main__":
    asyncio.run(main())
