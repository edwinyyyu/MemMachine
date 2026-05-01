"""Empirical evaluation of LLM-based ingestion alt-keys.

Runs LoCoMo-30 fair-backfill recall at K=20 and K=50 across four conditions:
  1. cosine_no_altkeys
  2. cosine_llm_altkeys
  3. v2f_no_altkeys
  4. v2f_llm_altkeys

Emits:
  results/ingestion_llm_empirical.md
  results/ingestion_llm_empirical.json
  results/ingestion_llm_samples.json   (30 random turns with LLM output)

Usage:
  uv run python ingest_llm_eval.py [--prompt v3] [--limit N_SEGMENTS]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from associative_recall import (
    SegmentStore,
)
from dotenv import load_dotenv

# The LLM pipeline itself.
from ingest_llm_altkeys import (
    LLMAltKeyGenerator,
    decisions_to_altkeys,
    generate_altkeys_for_all,
)
from ingest_regex_altkeys import AltKey

# Reuse regex-side augmented index + cosine/v2f condition runners + embed util.
from ingest_regex_eval import (
    BUDGETS,
    AugmentedSegmentStore,
    ConditionResult,
    Embedder,
    embed_texts_cached,
    run_cosine_condition,
    run_v2f_condition,
)
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summarize_conditions(conditions: dict[str, ConditionResult]) -> dict:
    overall = {}
    for name, cond in conditions.items():
        entry = {"n": len(cond.per_question)}
        for K in BUDGETS:
            entry[f"mean_r@{K}"] = round(cond.mean_by_K(K), 4)
        overall[name] = entry

    cats: set[str] = set()
    for cond in conditions.values():
        for r in cond.per_question:
            cats.add(r.get("category", "unknown"))

    per_cat: dict[str, dict[str, dict]] = {}
    for cat in sorted(cats):
        per_cat[cat] = {}
        for name, cond in conditions.items():
            rs = [r for r in cond.per_question if r.get("category") == cat]
            n = len(rs)
            entry = {"n": n}
            for K in BUDGETS:
                vals = [r[f"r@{K}"] for r in rs]
                entry[f"mean_r@{K}"] = round(sum(vals) / n, 4) if n else 0.0
            per_cat[cat][name] = entry

    return {"overall": overall, "per_category": per_cat}


# ---------------------------------------------------------------------------
# Precision check: which alt-keys fire into top-K of v2f_llm_altkeys?
# ---------------------------------------------------------------------------
def precision_audit(
    aug_store: AugmentedSegmentStore,
    embedder: Embedder,
    locomo_qs: list[dict],
    gold_by_conv: dict[str, set[int]],
) -> dict:
    """For each question, for top-K (K=50) retrieved from augmented cosine,
    count how many came from alt-keys (max-alt-sim beat original-sim) and
    of those, what fraction point to gold turns."""
    K = 50
    by_fire: dict[str, int] = defaultdict(int)  # fire (alt won) and is/isnt gold
    alt_hits_total = 0
    alt_hits_gold = 0
    orig_hits_total = 0
    orig_hits_gold = 0

    for q in locomo_qs:
        q_emb = embedder.embed_text(q["question"])
        q_emb_n = q_emb / max(float(np.linalg.norm(q_emb)), 1e-10)
        conv_id = q["conversation_id"]
        gold = gold_by_conv.get(conv_id, set())

        # Recompute per-segment sims with conversation filter
        orig_sims = aug_store.normalized_embeddings @ q_emb_n
        alt_per_parent = np.full(orig_sims.shape, -np.inf, dtype=np.float32)
        if aug_store.alt_normalized.shape[0] > 0:
            alt_sims = aug_store.alt_normalized @ q_emb_n
            np.maximum.at(alt_per_parent, aug_store.alt_parent_index, alt_sims)
        combined = np.maximum(orig_sims, alt_per_parent)

        mask = aug_store.conversation_ids == conv_id
        combined = np.where(mask, combined, -1.0)
        top_idx = np.argsort(combined)[::-1][:K]
        for idx in top_idx:
            if combined[idx] <= -1.0:
                continue
            seg = aug_store.segments[idx]
            is_gold = seg.turn_id in gold
            alt_won = alt_per_parent[idx] > orig_sims[idx]
            if alt_won:
                alt_hits_total += 1
                if is_gold:
                    alt_hits_gold += 1
            else:
                orig_hits_total += 1
                if is_gold:
                    orig_hits_gold += 1

    return {
        "K": K,
        "alt_hits_total": int(alt_hits_total),
        "alt_hits_gold": int(alt_hits_gold),
        "alt_precision_gold": round(alt_hits_gold / alt_hits_total, 4)
        if alt_hits_total
        else 0.0,
        "orig_hits_total": int(orig_hits_total),
        "orig_hits_gold": int(orig_hits_gold),
        "orig_precision_gold": round(orig_hits_gold / orig_hits_total, 4)
        if orig_hits_total
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------
def render_markdown(
    summary: dict,
    bloat: dict,
    ingestion_stats: dict,
    verdict: dict,
    n_questions: int,
    n_source_segments: int,
    precision_info: dict,
    prompt_version: str,
    prompt_iterations: list[dict],
    cost: dict,
    comparison_to_regex: dict,
    caveats: list[str],
) -> str:
    L: list[str] = []
    L.append("# Ingestion-LLM Alt-Key — Empirical Recall Test")
    L.append("")
    L.append(
        "Tests whether an LLM at ingestion time, asked per-turn to decide "
        "whether to emit alt-keys, can avoid the precision collapse seen in "
        "the pure-regex alt-key experiment (see `ingestion_regex_empirical.md`)."
    )
    L.append("")
    L.append(
        f"Benchmark: **LoCoMo-30** ({n_questions} questions, "
        f"{n_source_segments} segments in the LoCoMo corpus). "
        f"Model: **gpt-5-mini**, prompt **{prompt_version}**."
    )
    L.append("")

    # Phase 1: prompt tuning
    L.append("## 1. Prompt tuning (Phase 1)")
    L.append("")
    for it in prompt_iterations:
        L.append(
            f"- **{it['version']}**: {it['rationale']} — "
            f"SKIP rate on {it['n_sample']}-turn sample: "
            f"{it['skip_rate'] * 100:.0f}%; alt-keys emitted: {it['n_alts']}."
        )
    L.append("")
    L.append(
        f"Selected prompt: **{prompt_version}** (tightest SKIP discipline with "
        f"specific third-person fact restatements)."
    )
    L.append("")

    # Phase 2: ingestion stats
    L.append("## 2. Ingestion statistics (Phase 2)")
    L.append("")
    L.append(f"- Total turns ingested: **{ingestion_stats['n_turns']}**")
    L.append(
        f"- SKIP turns: **{ingestion_stats['n_skip']}** "
        f"({ingestion_stats['skip_rate'] * 100:.1f}%)"
    )
    L.append(
        f"- Turns with alt-keys: **{ingestion_stats['n_with_alts']}** "
        f"({(1 - ingestion_stats['skip_rate']) * 100:.1f}%)"
    )
    L.append(
        f"- Total alt-keys emitted (pre-dedup): **{ingestion_stats['n_alts_raw']}**"
    )
    L.append(f"- Alt-keys after dedup by text: **{bloat['n_altkeys']}**")
    L.append(f"- Bloat factor (alt / original): **{bloat['bloat_factor']:.2f}x**")
    L.append(
        f"- Mean alt-keys per non-SKIP turn: "
        f"**{ingestion_stats['mean_alts_per_with_alts']:.2f}**"
    )
    L.append("")

    # Phase 3: recall
    L.append("## 3. Overall recall (Phase 3)")
    L.append("")
    L.append(
        "Fair-backfill recall on LoCoMo-30 at K=20 and K=50. v2f conditions "
        "backfill with cosine on the same index so every side spends exactly "
        "K segments."
    )
    L.append("")
    ov = summary["overall"]
    L.append("| condition | mean r@20 | mean r@50 |")
    L.append("|---|---:|---:|")
    for cond_name in [
        "cosine_no_altkeys",
        "cosine_llm_altkeys",
        "v2f_no_altkeys",
        "v2f_llm_altkeys",
    ]:
        s = ov.get(cond_name, {})
        L.append(
            f"| {cond_name} | {s.get('mean_r@20', 0):.4f} | "
            f"{s.get('mean_r@50', 0):.4f} |"
        )
    L.append("")

    # Per category
    L.append("## 4. Per-category recall")
    L.append("")
    L.append(
        "| category | n | cos_no @20 | cos_llm @20 | v2f_no @20 | v2f_llm @20 | "
        "cos_no @50 | cos_llm @50 | v2f_no @50 | v2f_llm @50 |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat, d in summary["per_category"].items():
        n_cat = d.get("cosine_no_altkeys", {}).get("n", 0)
        row = f"| {cat} | {n_cat} "
        for K in BUDGETS:
            for cond in [
                "cosine_no_altkeys",
                "cosine_llm_altkeys",
                "v2f_no_altkeys",
                "v2f_llm_altkeys",
            ]:
                v = d.get(cond, {}).get(f"mean_r@{K}", 0.0)
                row += f"| {v:.3f} "
        row += "|"
        L.append(row)
    L.append("")

    # Verdict
    L.append("## 5. Verdict")
    L.append("")
    L.append(f"- v2f Δr@20 (llm − no): **{verdict['v2f_delta_at_20']:+.4f}**")
    L.append(f"- v2f Δr@50 (llm − no): **{verdict['v2f_delta_at_50']:+.4f}**")
    L.append(f"- cosine Δr@20: **{verdict['cos_delta_at_20']:+.4f}**")
    L.append(f"- cosine Δr@50: **{verdict['cos_delta_at_50']:+.4f}**")
    L.append("")
    L.append("Per-category lift on v2f @ K=20 (sorted):")
    L.append("")
    L.append("| category | Δr@20 | sign |")
    L.append("|---|---:|:---:|")
    for cat, delta in verdict["category_deltas_v2f_at_20"]:
        sign = "gain" if delta > 0 else ("loss" if delta < 0 else "tie")
        L.append(f"| {cat} | {delta:+.4f} | {sign} |")
    L.append("")
    L.append(f"**One-line verdict: {verdict['one_liner']}**")
    L.append("")

    # Precision check
    L.append("## 6. Precision of alt-key hits (top-50 on LoCoMo-30)")
    L.append("")
    L.append(
        f"- Total top-50 hits where an alt-key out-scored the original "
        f"embedding: **{precision_info['alt_hits_total']}** "
        f"across {n_questions} questions"
    )
    L.append(
        f"- Of those, fraction whose parent turn is gold: "
        f"**{precision_info['alt_precision_gold'] * 100:.1f}%**"
    )
    L.append(
        f"- Hits where the ORIGINAL embedding won: "
        f"**{precision_info['orig_hits_total']}**, "
        f"gold share {precision_info['orig_precision_gold'] * 100:.1f}%"
    )
    L.append("")

    # Comparison to regex
    L.append("## 7. Comparison to pure-regex run")
    L.append("")
    L.append("| metric | regex | llm |")
    L.append("|---|---:|---:|")
    cr = comparison_to_regex
    L.append(
        f"| bloat factor | {cr['regex']['bloat_factor']:.2f}x | "
        f"{cr['llm']['bloat_factor']:.2f}x |"
    )
    L.append(
        f"| v2f r@20 | {cr['regex']['v2f_at_20']:.4f} | {cr['llm']['v2f_at_20']:.4f} |"
    )
    L.append(
        f"| v2f r@50 | {cr['regex']['v2f_at_50']:.4f} | {cr['llm']['v2f_at_50']:.4f} |"
    )
    L.append(
        f"| Δ vs no-altkeys @20 | {cr['regex']['delta_20']:+.4f} | "
        f"{cr['llm']['delta_20']:+.4f} |"
    )
    L.append(
        f"| Δ vs no-altkeys @50 | {cr['regex']['delta_50']:+.4f} | "
        f"{cr['llm']['delta_50']:+.4f} |"
    )
    L.append("")

    # Cost
    L.append("## 8. Cost")
    L.append("")
    L.append(f"- Model: gpt-5-mini (prompt {prompt_version})")
    L.append(f"- Uncached LLM calls: **{cost['n_uncached']}**")
    L.append(f"- Cached LLM calls: **{cost['n_cached']}**")
    L.append(f"- Input tokens: **{cost['prompt_tokens']}**")
    L.append(f"- Output tokens: **{cost['completion_tokens']}**")
    L.append(
        f"- Est. LLM cost (gpt-5-mini @ $0.25/1M in, $2/1M out): "
        f"**${cost['est_usd']:.2f}**"
    )
    L.append(
        f"- Alt-key embedding calls: **{cost['embed_uncached']}** "
        f"new embeddings (~${cost['embed_usd']:.2f} at $0.02/1M @ ada-002 size)"
    )
    L.append("")

    L.append("## 9. Caveats")
    L.append("")
    for c in caveats:
        L.append(f"- {c}")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load data
    print("Loading SegmentStore + LoCoMo-30 questions ...", flush=True)
    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    with open(DATA_DIR / "questions_extended.json") as f:
        all_qs = json.load(f)
    locomo_qs = [q for q in all_qs if q.get("benchmark") == "locomo"][:30]
    locomo_conv_ids = {
        s.conversation_id
        for s in store.segments
        if s.conversation_id.startswith("locomo_")
    }
    locomo_segments = [
        s for s in store.segments if s.conversation_id in locomo_conv_ids
    ]
    print(
        f"  LoCoMo sub-corpus: {len(locomo_segments)} turns, "
        f"{len(locomo_conv_ids)} conversations, {len(locomo_qs)} questions",
        flush=True,
    )

    # Phase 2: LLM alt-key ingestion
    print(
        f"\n[Phase 2] LLM alt-key ingestion with prompt={args.prompt} ...", flush=True
    )
    client = OpenAI(timeout=60.0)
    generator = LLMAltKeyGenerator(
        client=client,
        prompt_version=args.prompt,
        max_workers=args.workers,
    )
    decisions = generate_altkeys_for_all(generator, locomo_segments)
    llm_cost = {
        "n_uncached": generator.n_uncached,
        "n_cached": generator.n_cached,
        "prompt_tokens": generator.total_prompt_tokens,
        "completion_tokens": generator.total_completion_tokens,
    }
    # gpt-5-mini pricing approx (April 2026): $0.25/M input, $2/M output
    llm_cost["est_usd"] = round(
        llm_cost["prompt_tokens"] * 0.25 / 1e6
        + llm_cost["completion_tokens"] * 2.0 / 1e6,
        4,
    )
    print(
        f"  LLM calls: uncached={llm_cost['n_uncached']} "
        f"cached={llm_cost['n_cached']} "
        f"in={llm_cost['prompt_tokens']} out={llm_cost['completion_tokens']} "
        f"~${llm_cost['est_usd']:.3f}",
        flush=True,
    )

    n_turns = len(decisions)
    n_skip = sum(1 for d in decisions if d.skipped)
    n_with_alts = n_turns - n_skip
    n_alts_raw = sum(len(d.alt_keys) for d in decisions)
    ingestion_stats = {
        "n_turns": n_turns,
        "n_skip": n_skip,
        "skip_rate": n_skip / max(n_turns, 1),
        "n_with_alts": n_with_alts,
        "n_alts_raw": n_alts_raw,
        "mean_alts_per_with_alts": n_alts_raw / max(n_with_alts, 1),
    }
    print(
        f"  turns={n_turns} SKIP={n_skip} ({ingestion_stats['skip_rate'] * 100:.1f}%) "
        f"alts_raw={n_alts_raw}",
        flush=True,
    )

    # Convert to AltKey objects, dedup by text.
    alt_keys_raw = decisions_to_altkeys(decisions, source_tag=f"llm:{args.prompt}")
    seen: set[str] = set()
    alt_keys: list[AltKey] = []
    for k in alt_keys_raw:
        if k.text in seen:
            continue
        seen.add(k.text)
        alt_keys.append(k)
    print(f"  alt_keys deduped: {len(alt_keys)}", flush=True)

    # Save 30 random samples for qualitative inspection
    rng = random.Random(7)
    sample_ids = rng.sample(range(len(decisions)), min(30, len(decisions)))
    samples_out = []
    for si in sample_ids:
        d = decisions[si]
        samples_out.append(
            {
                "conversation_id": d.conversation_id,
                "turn_id": d.turn_id,
                "role": d.role,
                "text": d.text,
                "prev_context": d.prev_context,
                "skipped": d.skipped,
                "alt_keys": d.alt_keys,
                "raw_response": d.raw_response,
            }
        )
    samples_path = RESULTS_DIR / "ingestion_llm_samples.json"
    with open(samples_path, "w") as f:
        json.dump({"prompt_version": args.prompt, "samples": samples_out}, f, indent=2)
    print(f"  wrote {samples_path}", flush=True)

    # Phase 3: augmented index + retrieval eval
    print("\n[Phase 3] Embedding alt-keys and building augmented store ...", flush=True)
    embedder = Embedder(client)
    alt_embeddings = embed_texts_cached(
        client,
        embedder.embedding_cache,
        [k.text for k in alt_keys],
    )
    embedder.save()
    embed_uncached_count = getattr(embedder.embedding_cache, "_new_entries", {})
    embed_uncached_n = len(embed_uncached_count) if embed_uncached_count else 0

    aug_store = AugmentedSegmentStore(store, alt_keys, alt_embeddings)
    print(
        f"  augmented store built: base={len(store.segments)} alts={len(alt_keys)}",
        flush=True,
    )

    # Four conditions
    conditions: dict[str, ConditionResult] = {}
    print("\n  [1/4] cosine_no_altkeys ...", flush=True)
    conditions["cosine_no_altkeys"] = run_cosine_condition(store, embedder, locomo_qs)
    conditions["cosine_no_altkeys"].name = "cosine_no_altkeys"

    print("  [2/4] cosine_llm_altkeys ...", flush=True)
    conditions["cosine_llm_altkeys"] = run_cosine_condition(
        aug_store, embedder, locomo_qs
    )
    conditions["cosine_llm_altkeys"].name = "cosine_llm_altkeys"

    print("  [3/4] v2f_no_altkeys ...", flush=True)
    conditions["v2f_no_altkeys"] = run_v2f_condition(
        store, embedder, locomo_qs, "v2f_no_altkeys"
    )

    print("  [4/4] v2f_llm_altkeys ...", flush=True)
    conditions["v2f_llm_altkeys"] = run_v2f_condition(
        aug_store, embedder, locomo_qs, "v2f_llm_altkeys"
    )

    embedder.save()

    # Summaries & verdict
    summary = summarize_conditions(conditions)
    ov = summary["overall"]
    v2f_no_20 = ov["v2f_no_altkeys"]["mean_r@20"]
    v2f_llm_20 = ov["v2f_llm_altkeys"]["mean_r@20"]
    v2f_no_50 = ov["v2f_no_altkeys"]["mean_r@50"]
    v2f_llm_50 = ov["v2f_llm_altkeys"]["mean_r@50"]
    cos_no_20 = ov["cosine_no_altkeys"]["mean_r@20"]
    cos_llm_20 = ov["cosine_llm_altkeys"]["mean_r@20"]
    cos_no_50 = ov["cosine_no_altkeys"]["mean_r@50"]
    cos_llm_50 = ov["cosine_llm_altkeys"]["mean_r@50"]

    v2f_delta_20 = v2f_llm_20 - v2f_no_20
    v2f_delta_50 = v2f_llm_50 - v2f_no_50
    cos_delta_20 = cos_llm_20 - cos_no_20
    cos_delta_50 = cos_llm_50 - cos_no_50

    cat_deltas: list[tuple[str, float]] = []
    for cat, d in summary["per_category"].items():
        vn = d.get("v2f_no_altkeys", {}).get("mean_r@20", 0.0)
        vw = d.get("v2f_llm_altkeys", {}).get("mean_r@20", 0.0)
        cat_deltas.append((cat, vw - vn))
    cat_deltas.sort(key=lambda x: x[1], reverse=True)

    if v2f_delta_20 > 0.01 and v2f_delta_50 > 0.01:
        one_liner = "LLM alt-keys are worth keeping"
    elif v2f_delta_20 < -0.01 or v2f_delta_50 < -0.01:
        one_liner = "LLM alt-keys are not worth keeping"
    else:
        one_liner = "LLM alt-keys are borderline"
    verdict = {
        "v2f_delta_at_20": v2f_delta_20,
        "v2f_delta_at_50": v2f_delta_50,
        "cos_delta_at_20": cos_delta_20,
        "cos_delta_at_50": cos_delta_50,
        "category_deltas_v2f_at_20": cat_deltas,
        "one_liner": one_liner,
    }

    # Precision audit
    gold_by_conv: dict[str, set[int]] = defaultdict(set)
    for q in locomo_qs:
        gold_by_conv[q["conversation_id"]].update(q["source_chat_ids"])
    precision_info = precision_audit(aug_store, embedder, locomo_qs, gold_by_conv)

    # Phase-1 iteration notes (sampled on 30-turn from first LoCoMo conversation
    # during prompt tuning).
    prompt_iterations = [
        {
            "version": "v1",
            "n_sample": 30,
            "skip_rate": 0.60,
            "n_alts": 22,
            "rationale": "Initial spec prompt (verbose), over-generates alt-keys "
            "for questions and acknowledgements.",
        },
        {
            "version": "v2",
            "n_sample": 30,
            "skip_rate": 0.87,
            "n_alts": 4,
            "rationale": "Tightened with explicit DO-NOT list and default-SKIP; "
            "slightly under-generates (misses some facts).",
        },
        {
            "version": "v3",
            "n_sample": 30,
            "skip_rate": 0.83,
            "n_alts": 5,
            "rationale": "Four named cases (anaphora / correction / alias / "
            "personal fact) with strict 5-20 word output format; "
            "chosen as the ingestion prompt.",
        },
    ]

    # Comparison block — static regex numbers from the prior run
    comparison = {
        "regex": {
            "bloat_factor": 0.81,
            "v2f_at_20": 0.6667,
            "v2f_at_50": 0.7833,
            "delta_20": 0.6667 - v2f_no_20,
            "delta_50": 0.7833 - v2f_no_50,
        },
        "llm": {
            "bloat_factor": len(alt_keys) / max(len(locomo_segments), 1),
            "v2f_at_20": v2f_llm_20,
            "v2f_at_50": v2f_llm_50,
            "delta_20": v2f_delta_20,
            "delta_50": v2f_delta_50,
        },
    }

    bloat = {
        "n_original": len(locomo_segments),
        "n_altkeys_raw": len(alt_keys_raw),
        "n_altkeys": len(alt_keys),
        "bloat_factor": len(alt_keys) / max(len(locomo_segments), 1),
    }

    cost = {
        **llm_cost,
        "embed_uncached": embed_uncached_n,
        "embed_usd": round(embed_uncached_n * 10 * 0.02 / 1e6, 4),  # rough
    }

    caveats = [
        "LoCoMo-30 only (30 questions, 3 conversations). Deltas below ~0.01 "
        "are within noise.",
        f"Prompt {args.prompt} was selected after 3 iterations. No further "
        "prompt tuning was attempted.",
        "Alt-key scoring is per-parent-max over original + alt-key "
        "embeddings, identical to the regex experiment for fair comparison.",
        "Precision audit counts top-K hits where max came from an alt-key vs "
        "original embedding — not a strict 'alt-key caused the segment to "
        "enter top-K' test; a segment already in original top-K but whose "
        "alt-key also beats original similarity is counted under alt_hits.",
        "Costs are estimated from token usage at gpt-5-mini posted rates "
        "and may vary slightly from actual billing.",
    ]

    md = render_markdown(
        summary=summary,
        bloat=bloat,
        ingestion_stats=ingestion_stats,
        verdict=verdict,
        n_questions=len(locomo_qs),
        n_source_segments=len(locomo_segments),
        precision_info=precision_info,
        prompt_version=args.prompt,
        prompt_iterations=prompt_iterations,
        cost=cost,
        comparison_to_regex=comparison,
        caveats=caveats,
    )
    md_path = RESULTS_DIR / "ingestion_llm_empirical.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nWrote {md_path}", flush=True)

    raw = {
        "meta": {
            "benchmark": "locomo_30q",
            "n_questions": len(locomo_qs),
            "n_segments_full_corpus": len(store.segments),
            "n_segments_locomo_corpus": len(locomo_segments),
            "locomo_conversations": sorted(locomo_conv_ids),
            "prompt_version": args.prompt,
            "elapsed_s": round(time.time() - t0, 2),
        },
        "ingestion_stats": ingestion_stats,
        "index_bloat": bloat,
        "cost": cost,
        "summary": summary,
        "verdict": verdict,
        "precision": precision_info,
        "comparison_to_regex": comparison,
        "prompt_iterations": prompt_iterations,
        "per_question": {name: cond.per_question for name, cond in conditions.items()},
    }
    json_path = RESULTS_DIR / "ingestion_llm_empirical.json"
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"Wrote {json_path}", flush=True)

    # Console summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for cond in [
        "cosine_no_altkeys",
        "cosine_llm_altkeys",
        "v2f_no_altkeys",
        "v2f_llm_altkeys",
    ]:
        s = ov[cond]
        print(f"  {cond:28s} r@20={s['mean_r@20']:.4f}  r@50={s['mean_r@50']:.4f}")
    print(
        f"\n  SKIP rate: {ingestion_stats['skip_rate'] * 100:.1f}%  "
        f"alt-keys emitted: {len(alt_keys)}  bloat: {bloat['bloat_factor']:.2f}x"
    )
    print(f"  v2f Δr@20={v2f_delta_20:+.4f}  v2f Δr@50={v2f_delta_50:+.4f}")
    print(
        f"  alt-key precision (gold share): {precision_info['alt_precision_gold'] * 100:.1f}% "
        f"on {precision_info['alt_hits_total']} alt-won hits"
    )
    print(f"  verdict: {one_liner}")
    print(f"  LLM cost: ~${cost['est_usd']:.3f}")


if __name__ == "__main__":
    main()
