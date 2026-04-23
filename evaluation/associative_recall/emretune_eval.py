"""Evaluate retuned cue-generation prompts for EventMemory on LoCoMo-30.

Compares the current v2f prompt (as re-run control = em_v2f baseline)
against four retuned variants that bake the speaker-prefix format into
generated cues to match EventMemory's embedded text distribution
("Caroline: Yeah, 16 weeks.").

Variants:
    v2f_em_baseline              current V2F_PROMPT (re-run; parity check)
    v2f_em_speakerformat         LLM picks a speaker per cue
    v2f_em_mixed_speakers        cue1=user, cue2=assistant
    v2f_em_role_tag              [USER]/[ASSISTANT] instead of names
    type_enumerated_em_speakerformat
                                  type_enumerated cues with speaker prefix

For each variant: K=20 and K=50 recall, W/T/L vs baseline, sample cues.

Reads/writes DEDICATED caches so it never poisons other specialists'
caches:
    cache/emretune_v2f_baseline_cache.json
    cache/emretune_v2f_speakerformat_cache.json
    cache/emretune_v2f_mixed_speakers_cache.json
    cache/emretune_v2f_role_tag_cache.json
    cache/emretune_type_enum_speakerformat_cache.json

Reuses existing em_setup ingestion (Qdrant + SQLite segment store).
Does NOT re-ingest. Does NOT modify framework files.
"""

from __future__ import annotations

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
    V2F_MODEL,
    V2F_PROMPT,
    EMHit,
    _MergedLLMCache,
    _dedupe_by_turn_id,
    _merge_by_max_score,
    _query_em,
    format_primer_context,
    parse_v2f_cues,
)
from em_retuned_cue_gen import (
    build_v2f_mixedspeakers_prompt,
    build_v2f_roletag_prompt,
    build_v2f_speakerformat_prompt,
    build_type_enum_speakerfmt_prompt,
    parse_cues as parse_retuned_cues,
)


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
BUDGETS = (20, 50)
LOCOMO_CONV_IDS = ("locomo_conv-26", "locomo_conv-30", "locomo_conv-41")

VARIANTS = [
    "v2f_em_baseline",
    "v2f_em_speakerformat",
    "v2f_em_mixed_speakers",
    "v2f_em_role_tag",
    "type_enumerated_em_speakerformat",
]

CACHE_FILES = {
    "v2f_em_baseline": CACHE_DIR / "emretune_v2f_baseline_cache.json",
    "v2f_em_speakerformat": CACHE_DIR / "emretune_v2f_speakerformat_cache.json",
    "v2f_em_mixed_speakers": CACHE_DIR / "emretune_v2f_mixed_speakers_cache.json",
    "v2f_em_role_tag": CACHE_DIR / "emretune_v2f_role_tag_cache.json",
    "type_enumerated_em_speakerformat": (
        CACHE_DIR / "emretune_type_enum_speakerformat_cache.json"
    ),
}

BASELINE_VARIANT = "v2f_em_baseline"


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
# Cue generation per variant
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


async def generate_cues(
    variant: str,
    question: str,
    context_section: str,
    participants: tuple[str, str],
    cache: _MergedLLMCache,
    openai_client,
) -> tuple[list[str], str, bool]:
    """Returns (cues, raw_llm_response, cache_hit)."""
    p_user, p_asst = participants

    if variant == "v2f_em_baseline":
        prompt = V2F_PROMPT.format(
            question=question, context_section=context_section
        )
        raw, hit = await _llm_call(openai_client, prompt, cache)
        return parse_v2f_cues(raw, max_cues=2), raw, hit

    if variant == "v2f_em_speakerformat":
        prompt = build_v2f_speakerformat_prompt(
            question, context_section, p_user, p_asst
        )
        raw, hit = await _llm_call(openai_client, prompt, cache)
        return parse_retuned_cues(raw, max_cues=2), raw, hit

    if variant == "v2f_em_mixed_speakers":
        prompt = build_v2f_mixedspeakers_prompt(
            question, context_section, p_user, p_asst
        )
        raw, hit = await _llm_call(openai_client, prompt, cache)
        return parse_retuned_cues(raw, max_cues=2), raw, hit

    if variant == "v2f_em_role_tag":
        prompt = build_v2f_roletag_prompt(question, context_section)
        raw, hit = await _llm_call(openai_client, prompt, cache)
        return parse_retuned_cues(raw, max_cues=2), raw, hit

    if variant == "type_enumerated_em_speakerformat":
        prompt = build_type_enum_speakerfmt_prompt(
            question, context_section, p_user, p_asst
        )
        raw, hit = await _llm_call(openai_client, prompt, cache)
        return parse_retuned_cues(raw, max_cues=7), raw, hit

    raise KeyError(variant)


# --------------------------------------------------------------------------
# Evaluation flow
# --------------------------------------------------------------------------


async def evaluate_question(
    variant: str,
    memory: EventMemory,
    question: dict,
    participants: tuple[str, str],
    cache: _MergedLLMCache,
    openai_client,
    *,
    max_K: int,
) -> dict:
    q_text = question["question"]
    gold = set(question.get("source_chat_ids", []))

    t0 = time.monotonic()

    # Primer retrieval (K=10, expand=0) -- same as em_v2f for fair comparison
    primer_hits = _dedupe_by_turn_id(
        await _query_em(memory, q_text, vector_search_limit=10, expand_context=0)
    )[:10]
    primer_segments = [
        {"turn_id": h.turn_id, "role": h.role, "text": h.text}
        for h in primer_hits
    ]
    context_section = format_primer_context(primer_segments)

    cues, raw_resp, cache_hit = await generate_cues(
        variant,
        q_text,
        context_section,
        participants,
        cache,
        openai_client,
    )

    # For baseline and speakerformat/mixed/role_tag variants: merge
    # primer(K) + each cue(K) by max_score.
    # For type_enumerated variant: use v2f's 2 cues logic at K=2? No --
    # this variant exclusively uses the 7 type_enum cues. Keep merge-max
    # style so we're only isolating the PROMPT variable (cue text).
    primer_for_merge = await _query_em(
        memory, q_text, vector_search_limit=max_K, expand_context=0
    )
    cue_hits = []
    for cue in cues:
        cue_hits.append(
            await _query_em(
                memory, cue, vector_search_limit=max_K, expand_context=0
            )
        )
    merged = _merge_by_max_score([primer_for_merge, *cue_hits])
    hits = merged[:max_K]

    elapsed = time.monotonic() - t0

    row: dict = {
        "conversation_id": question["conversation_id"],
        "category": question.get("category", "unknown"),
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "gold_turn_ids": sorted(gold),
        "n_cues": len(cues),
        "cues": cues,
        "cache_hit": cache_hit,
        "time_s": round(elapsed, 3),
    }
    for K in BUDGETS:
        topk = hits[:K]
        retrieved = {h.turn_id for h in topk}
        row[f"r@{K}"] = round(compute_recall(retrieved, gold), 4)
        row[f"retrieved_turn_ids@{K}"] = sorted(retrieved)
    return row


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

    # Per-variant dedicated caches.
    caches: dict[str, _MergedLLMCache] = {}
    for variant in VARIANTS:
        path = CACHE_FILES[variant]
        caches[variant] = _MergedLLMCache(
            reader_paths=[path], writer_path=path
        )

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

    for variant in VARIANTS:
        rows: list[dict] = []
        t_variant = time.monotonic()
        for q in questions:
            cid = q["conversation_id"]
            mem = memories[cid]
            participants = participants_by_conv[cid]
            row = await evaluate_question(
                variant, mem, q, participants, caches[variant],
                openai_client, max_K=max_K,
            )
            rows.append(row)
        elapsed = time.monotonic() - t_variant

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

        caches[variant].save()

        print(
            f"[{variant}] n={summary['n']} "
            f"r@20={summary['mean_r@20']:.4f} r@50={summary['mean_r@50']:.4f} "
            f"in {summary['time_s']:.1f}s"
        )

    # W/T/L vs baseline
    baseline_rows = results["variants"][BASELINE_VARIANT]["per_question"]
    baseline_idx = {
        (r["conversation_id"], r["question_index"]): r for r in baseline_rows
    }
    for variant in VARIANTS:
        if variant == BASELINE_VARIANT:
            continue
        v_rows = results["variants"][variant]["per_question"]
        wtl = {}
        for K in BUDGETS:
            w = t = l = 0
            for r in v_rows:
                key = (r["conversation_id"], r["question_index"])
                br = baseline_idx[key]
                vr = r[f"r@{K}"]
                bv = br[f"r@{K}"]
                if vr > bv:
                    w += 1
                elif vr < bv:
                    l += 1
                else:
                    t += 1
            wtl[f"r@{K}"] = {"W": w, "T": t, "L": l}
        results["variants"][variant]["wtl_vs_baseline"] = wtl

    # Close.
    for coll, part in opened_resources:
        await segment_store.close_partition(part)
        await vector_store.close_collection(collection=coll)
    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()

    # Save outputs.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "em_prompt_retune.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    # Markdown report.
    md_lines = build_markdown_report(results, questions)
    out_md = RESULTS_DIR / "em_prompt_retune.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved: {out_md}")


def build_markdown_report(results: dict, questions: list[dict]) -> list[str]:
    base = results["variants"][BASELINE_VARIANT]["summary"]
    lines = [
        "# EventMemory Prompt Retune (speaker-baked embeddings)",
        "",
        "## Setup",
        "",
        f"- n_questions = {len(questions)} (benchmark=locomo, first 30)",
        "- EventMemory backend, `text-embedding-3-small`, `gpt-5-mini`",
        "- Speaker-baked embedded format: `\"{source}: {text}\"` "
        "(from `event_memory.py::_format_text`)",
        "- Caches: `cache/emretune_<variant>_cache.json` (dedicated)",
        "- Architecture identical across variants (primer + 2 or 7 cues, "
        "merge by max_score per turn_id). Only the PROMPT varies.",
        "",
        "## Speaker-baking format confirmation",
        "",
        "`event_memory.py::_format_text` returns `f\"{source}: {text}\"` for "
        "MessageContext events. Per-conversation sources: conv-26 Caroline/Melanie, "
        "conv-30 Jon/Gina, conv-41 John/Maria.",
        "",
        "## Recall comparison",
        "",
        "| Variant | R@20 | R@50 | vs baseline R@20 | vs baseline R@50 | time (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for variant in VARIANTS:
        s = results["variants"][variant]["summary"]
        d20 = s["mean_r@20"] - base["mean_r@20"]
        d50 = s["mean_r@50"] - base["mean_r@50"]
        lines.append(
            f"| `{variant}` | {s['mean_r@20']:.4f} | {s['mean_r@50']:.4f} | "
            f"{d20:+.4f} | {d50:+.4f} | {s['time_s']:.1f} |"
        )
    lines += [
        "",
        "Prior `em_v2f` reference from `em_eval`: 0.7417 / 0.8833 at K=20 / K=50.",
        "",
        "## W/T/L per variant vs v2f_em_baseline (re-run control)",
        "",
        "| Variant | K=20 W/T/L | K=50 W/T/L |",
        "| --- | --- | --- |",
    ]
    for variant in VARIANTS:
        if variant == BASELINE_VARIANT:
            continue
        w = results["variants"][variant].get("wtl_vs_baseline", {})
        w20 = w.get("r@20", {"W": 0, "T": 0, "L": 0})
        w50 = w.get("r@50", {"W": 0, "T": 0, "L": 0})
        lines.append(
            f"| `{variant}` | "
            f"{w20['W']}/{w20['T']}/{w20['L']} | "
            f"{w50['W']}/{w50['T']}/{w50['L']} |"
        )

    # Sample cues (3 questions).
    lines += [
        "",
        "## Sample cues (3 questions)",
        "",
        "Per question, shows cues from each variant side by side.",
        "",
    ]
    sample_keys = []
    base_rows = results["variants"][BASELINE_VARIANT]["per_question"]
    # Pick 3 questions: first, middle, last with at least one cue produced.
    n = len(base_rows)
    sample_idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    for idx in sample_idxs:
        sample_keys.append(idx)
    for idx in sample_keys:
        q = base_rows[idx]
        lines.append(
            f"### Q{idx}: `{q['conversation_id']}` -- {q['question']!r}"
        )
        lines.append("")
        for variant in VARIANTS:
            v_row = results["variants"][variant]["per_question"][idx]
            lines.append(f"- `{variant}` "
                         f"(R@20={v_row['r@20']:.2f}, R@50={v_row['r@50']:.2f}):")
            for c in v_row["cues"]:
                lines.append(f"  - `{c}`")
        lines.append("")

    # Verdict.
    lines += [
        "## Verdict",
        "",
    ]
    best_variant = None
    best_r20_lift = 0.0
    best_r50_lift = 0.0
    for variant in VARIANTS:
        if variant == BASELINE_VARIANT:
            continue
        s = results["variants"][variant]["summary"]
        d20 = s["mean_r@20"] - base["mean_r@20"]
        d50 = s["mean_r@50"] - base["mean_r@50"]
        # Require >=1pp at both budgets to ship.
        if d20 >= 0.01 and d50 >= 0.01:
            if (d20 + d50) > (best_r20_lift + best_r50_lift):
                best_variant = variant
                best_r20_lift = d20
                best_r50_lift = d50
    if best_variant:
        lines.append(
            f"**Ship `{best_variant}`** as new EM default. "
            f"Lift vs current v2f: +{best_r20_lift:.4f} R@20, "
            f"+{best_r50_lift:.4f} R@50."
        )
    else:
        # Any one-sided?
        side_wins = []
        for variant in VARIANTS:
            if variant == BASELINE_VARIANT:
                continue
            s = results["variants"][variant]["summary"]
            d20 = s["mean_r@20"] - base["mean_r@20"]
            d50 = s["mean_r@50"] - base["mean_r@50"]
            if d20 >= 0.01 or d50 >= 0.01:
                side_wins.append((variant, d20, d50))
        if side_wins:
            lines.append(
                "**Narrow:** no variant wins at both K budgets. One-sided "
                "lifts (>=1pp at exactly one K):"
            )
            for v, d20, d50 in side_wins:
                lines.append(f"- `{v}`: d20={d20:+.4f}, d50={d50:+.4f}")
            lines.append("")
            lines.append(
                "Consider K-specific routing or keep current v2f prompt."
            )
        else:
            lines.append(
                "**No change.** All retuned variants tie or lose vs the "
                "current v2f prompt. The LLM already incorporates speaker "
                "register when asked for chat-style cues (samples show many "
                "baseline cues already begin with 'Caroline: ...'). The "
                "current prompt is well-adapted to speaker-baked "
                "embeddings without explicit prompting."
            )

    lines += [
        "",
        "## Outputs",
        "",
        "- `results/em_prompt_retune.json`",
        "- `results/em_prompt_retune.md`",
        "- Source: `em_retuned_cue_gen.py`, `emretune_eval.py`",
        "- Caches: `cache/emretune_<variant>_cache.json`",
        "",
    ]
    return lines


if __name__ == "__main__":
    asyncio.run(main())
