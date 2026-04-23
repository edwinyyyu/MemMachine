"""Empirical evaluation of pronoun-resolution ingest index.

Baselines:
  - v2f           (MetaV2f on main index)
  - cosine_baseline (pure cosine on main index)

Variants:
  - pronoun_resolve_stacked
      v2f first, then resolved-index hits fill remaining top-K slots.
  - pronoun_resolve_with_bonus
      same retrieval, but resolved hits get +0.05 score and can displace
      v2f's weakest in-top-K item if boosted score beats it.

Datasets: LoCoMo-30 + synthetic-19. K in {20, 50}. Fair-backfill recall.

Cache: `pronoun_*_cache.json` (dedicated). Reads shared caches for
warm-start.

Usage:
    uv run python pronoun_eval.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import Segment, SegmentStore
from best_shot import MetaV2f
from ingest_regex_eval import (
    BUDGETS,
    Embedder,
    compute_recall,
    fair_backfill_turn_ids,
)
from pronoun_resolution import (
    PronounEmbeddingCache,
    PronounLLMCache,
    PronounResolver,
    PronounTurnDecision,
    ResolvedTurnIndex,
    embed_resolved_texts,
    resolve_turns,
    stacked_merge,
    stacked_merge_with_bonus,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DATASETS = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_questions": 30,
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": None,
        "max_questions": None,
    },
}


# ---------------------------------------------------------------------------
# Main-index ranked list for merge (same pattern as critical_info_eval).
# ---------------------------------------------------------------------------
def v2f_main_ranked(
    arch: MetaV2f,
    store: SegmentStore,
    q: dict,
) -> tuple[list[tuple[Segment, float]], list[Segment], list[Segment]]:
    """Run v2f + cosine backfill. Return:
      - main_ranked: list of (seg, score) in fair-backfill order
        (arch segments first, scored high-to-low; then cosine top-K by score)
      - arch_segs  : v2f's raw output (pre-backfill)
      - cos_segs   : cosine top-maxK on main index
    """
    q_text = q["question"]
    conv_id = q["conversation_id"]

    arch.reset_counters()
    result = arch.retrieve(q_text, conv_id)
    arch_segs = list(result.segments)

    q_emb = arch.embed_text(q_text)
    max_K = max(BUDGETS)
    cos_res = store.search(q_emb, top_k=max_K, conversation_id=conv_id)
    cos_segs = list(cos_res.segments)
    cos_scores = list(cos_res.scores)
    cos_score_by_idx = {s.index: sc for s, sc in zip(cos_segs, cos_scores)}

    main_ranked: list[tuple[Segment, float]] = []
    seen: set[int] = set()
    EPS = 0.001
    for rank, s in enumerate(arch_segs):
        if s.index in seen:
            continue
        main_ranked.append((s, 10.0 - rank * EPS))
        seen.add(s.index)
    for s in cos_segs:
        if s.index in seen:
            continue
        sc = cos_score_by_idx.get(s.index, 0.0)
        main_ranked.append((s, sc))
        seen.add(s.index)

    return main_ranked, arch_segs, cos_segs


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------
def run_cosine_baseline(
    store: SegmentStore,
    embedder: Embedder,
    questions: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for q in questions:
        q_emb = embedder.embed_text(q["question"])
        max_K = max(BUDGETS)
        res = store.search(q_emb, top_k=max_K, conversation_id=q["conversation_id"])
        cos_segs = list(res.segments)
        source_ids = set(q["source_chat_ids"])
        row = {
            "conversation_id": q["conversation_id"],
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q["question"],
            "source_chat_ids": sorted(source_ids),
        }
        for K in BUDGETS:
            ids = {s.turn_id for s in cos_segs[:K]}
            row[f"r@{K}"] = compute_recall(ids, source_ids)
        out.append(row)
    return out


def run_v2f_baseline(
    store: SegmentStore,
    embedder: Embedder,
    questions: list[dict],
) -> list[dict]:
    arch = MetaV2f(store)
    arch.embedding_cache = embedder.embedding_cache
    arch.llm_cache = embedder.llm_cache
    out: list[dict] = []
    for i, q in enumerate(questions):
        main_ranked, arch_segs, cos_segs = v2f_main_ranked(arch, store, q)
        source_ids = set(q["source_chat_ids"])
        row = {
            "conversation_id": q["conversation_id"],
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q["question"],
            "source_chat_ids": sorted(source_ids),
        }
        for K in BUDGETS:
            ids = fair_backfill_turn_ids(arch_segs, cos_segs, K)
            row[f"r@{K}"] = compute_recall(ids, source_ids)
            row[f"retrieved_ids@{K}"] = sorted(ids)
        out.append(row)
        if (i + 1) % 5 == 0:
            try:
                arch.save_caches()
            except Exception as e:
                print(f"  (warn) v2f save_caches: {e}", flush=True)
    try:
        arch.save_caches()
    except Exception as e:
        print(f"  (warn) v2f save_caches: {e}", flush=True)
    return out


def run_resolved_variant(
    variant: str,
    store: SegmentStore,
    resolved_idx: ResolvedTurnIndex,
    embedder: Embedder,
    questions: list[dict],
    top_m: int = 10,
    bonus: float = 0.05,
) -> list[dict]:
    """variant in {'pronoun_resolve_stacked', 'pronoun_resolve_with_bonus'}"""
    arch = MetaV2f(store)
    arch.embedding_cache = embedder.embedding_cache
    arch.llm_cache = embedder.llm_cache

    out: list[dict] = []
    for i, q in enumerate(questions):
        main_ranked, arch_segs, cos_segs = v2f_main_ranked(arch, store, q)
        q_emb = arch.embed_text(q["question"])
        resolved_ranked = resolved_idx.search_per_parent(
            q_emb,
            top_m=top_m,
            conversation_id=q["conversation_id"],
        )
        source_ids = set(q["source_chat_ids"])
        row = {
            "conversation_id": q["conversation_id"],
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q["question"],
            "source_chat_ids": sorted(source_ids),
        }
        for K in BUDGETS:
            if variant == "pronoun_resolve_stacked":
                merged = stacked_merge(main_ranked, resolved_ranked, K)
            elif variant == "pronoun_resolve_with_bonus":
                merged = stacked_merge_with_bonus(
                    main_ranked, resolved_ranked, K, bonus=bonus,
                )
            else:
                raise ValueError(f"unknown variant: {variant}")

            # For stacked only: if merged has fewer than K entries, backfill
            # with cosine segments (arch-fair-backfill style) so we compare
            # against v2f at the SAME K budget.
            if len(merged) < K:
                present = {s.index for s in merged}
                for s in cos_segs:
                    if s.index in present:
                        continue
                    merged.append(s)
                    present.add(s.index)
                    if len(merged) >= K:
                        break

            ids = {s.turn_id for s in merged[:K]}
            row[f"r@{K}"] = compute_recall(ids, source_ids)
            row[f"retrieved_ids@{K}"] = sorted(ids)

            # Orthogonality: did any gold come from resolved-only (not in
            # v2f main_ranked[:K])?
            main_topK = {s.index for s, _ in main_ranked[:K]}
            resolved_gold_only = {
                s.turn_id for s in merged[:K]
                if s.index not in main_topK and s.turn_id in source_ids
            }
            row[f"resolved_gold_contrib@{K}"] = sorted(resolved_gold_only)

        out.append(row)
        if (i + 1) % 5 == 0:
            try:
                arch.save_caches()
            except Exception as e:
                print(f"  (warn) variant save_caches: {e}", flush=True)
    try:
        arch.save_caches()
    except Exception as e:
        print(f"  (warn) variant save_caches: {e}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def summarize(per_q: list[dict]) -> dict:
    n = len(per_q)
    out = {"n": n}
    for K in BUDGETS:
        vals = [r[f"r@{K}"] for r in per_q]
        out[f"mean_r@{K}"] = round(sum(vals) / n, 4) if n else 0.0
    return out


def summarize_by_category(per_q: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in per_q:
        by_cat[r.get("category", "unknown")].append(r)
    out: dict = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry = {"n": n}
        for K in BUDGETS:
            vals = [r[f"r@{K}"] for r in rs]
            entry[f"mean_r@{K}"] = round(sum(vals) / n, 4) if n else 0.0
        out[cat] = entry
    return out


def orthogonality(per_q: list[dict], K: int) -> dict:
    """Fraction of gold not present in v2f's top-K that was surfaced via the
    resolved index. (resolved_gold_contrib vs total gold)."""
    total_gold = 0
    res_gold = 0
    q_with = 0
    for r in per_q:
        gold = set(r["source_chat_ids"])
        contrib = set(r.get(f"resolved_gold_contrib@{K}", []))
        total_gold += len(gold)
        res_gold += len(contrib & gold)
        if contrib & gold:
            q_with += 1
    return {
        "K": K,
        "total_gold": total_gold,
        "resolved_gold": res_gold,
        "frac_gold_via_resolved": round(res_gold / max(total_gold, 1), 4),
        "frac_questions_with_resolved_gold": round(q_with / max(len(per_q), 1), 4),
    }


# ---------------------------------------------------------------------------
# Run a single dataset
# ---------------------------------------------------------------------------
def run_dataset(
    ds_name: str,
    resolver: PronounResolver,
    client: OpenAI,
    embedder: Embedder,
    resolved_cache: PronounEmbeddingCache,
) -> tuple[dict, list[PronounTurnDecision]]:
    cfg = DATASETS[ds_name]
    print(f"\n{'='*70}\nDataset: {ds_name}\n{'='*70}", flush=True)
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        all_qs = json.load(f)
    qs = all_qs
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    conv_ids_in_qs = {q["conversation_id"] for q in qs}
    target_segments = [
        s for s in store.segments if s.conversation_id in conv_ids_in_qs
    ]
    print(
        f"  questions: {len(qs)} | target segments: {len(target_segments)}",
        flush=True,
    )

    # Resolve pronouns for each target segment.
    print(f"  resolving {len(target_segments)} turns ...", flush=True)
    decisions = resolve_turns(resolver, target_segments)

    n_turns = len(decisions)
    n_resolved = sum(1 for d in decisions if d.resolved)
    skip_rate = 1.0 - (n_resolved / max(n_turns, 1))
    print(
        f"  resolved={n_resolved}/{n_turns} ({n_resolved/max(n_turns,1)*100:.1f}%)  "
        f"skip_rate={skip_rate*100:.1f}%",
        flush=True,
    )

    # Embed resolved texts (one per resolved turn).
    resolved_parents: list[int] = []
    resolved_texts: list[str] = []
    for d in decisions:
        if d.resolved and d.resolved_text.strip():
            resolved_parents.append(d.parent_index)
            resolved_texts.append(d.resolved_text.strip())

    if resolved_texts:
        resolved_embs = embed_resolved_texts(client, resolved_cache, resolved_texts)
    else:
        resolved_embs = np.zeros((0, 1536), dtype=np.float32)
    resolved_idx = ResolvedTurnIndex(
        store, resolved_parents, resolved_texts, resolved_embs,
    )

    # ---- Baselines
    print("  [1/4] cosine baseline ...", flush=True)
    cosine_rows = run_cosine_baseline(store, embedder, qs)
    cosine_summary = summarize(cosine_rows)
    cosine_by_cat = summarize_by_category(cosine_rows)

    print("  [2/4] v2f baseline ...", flush=True)
    v2f_rows = run_v2f_baseline(store, embedder, qs)
    v2f_summary = summarize(v2f_rows)
    v2f_by_cat = summarize_by_category(v2f_rows)

    # ---- Variants
    print("  [3/4] pronoun_resolve_stacked ...", flush=True)
    stacked_rows = run_resolved_variant(
        "pronoun_resolve_stacked", store, resolved_idx, embedder, qs,
        top_m=10,
    )
    stacked_summary = summarize(stacked_rows)
    stacked_by_cat = summarize_by_category(stacked_rows)
    stacked_ortho_20 = orthogonality(stacked_rows, K=20)
    stacked_ortho_50 = orthogonality(stacked_rows, K=50)

    print("  [4/4] pronoun_resolve_with_bonus ...", flush=True)
    bonus_rows = run_resolved_variant(
        "pronoun_resolve_with_bonus", store, resolved_idx, embedder, qs,
        top_m=10, bonus=0.05,
    )
    bonus_summary = summarize(bonus_rows)
    bonus_by_cat = summarize_by_category(bonus_rows)
    bonus_ortho_20 = orthogonality(bonus_rows, K=20)
    bonus_ortho_50 = orthogonality(bonus_rows, K=50)

    return (
        {
            "ds_name": ds_name,
            "n_questions": len(qs),
            "n_target_segments": len(target_segments),
            "n_turns_resolved": n_resolved,
            "skip_rate": skip_rate,
            "cosine_baseline": {"summary": cosine_summary, "by_category": cosine_by_cat,
                                "per_question": cosine_rows},
            "v2f": {"summary": v2f_summary, "by_category": v2f_by_cat,
                    "per_question": v2f_rows},
            "pronoun_resolve_stacked": {
                "summary": stacked_summary, "by_category": stacked_by_cat,
                "per_question": stacked_rows,
                "orthogonality@20": stacked_ortho_20,
                "orthogonality@50": stacked_ortho_50,
            },
            "pronoun_resolve_with_bonus": {
                "summary": bonus_summary, "by_category": bonus_by_cat,
                "per_question": bonus_rows,
                "orthogonality@20": bonus_ortho_20,
                "orthogonality@50": bonus_ortho_50,
            },
        },
        decisions,
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_markdown(
    all_results: dict,
    all_decisions: dict[str, list[PronounTurnDecision]],
    cost: dict,
) -> str:
    L: list[str] = []
    L.append("# Pronoun-Resolution Ingest Index -- Empirical Test")
    L.append("")
    L.append(
        "At ingest, an LLM rewrites each turn with pronouns/deictics "
        "(it, they, this, that, these, those, here/there, he/she if "
        "ambiguous) substituted by their specific referents, using the "
        "2-3 preceding turns as context. Turns already self-contained "
        "are SKIP. RESOLVED texts are embedded in a separate vector "
        "index. At query time v2f retrieves as usual; a disjoint cosine "
        "search over the resolved index is STACKED-MERGED (v2f fills "
        "top-K first; resolved hits only fill any remaining slots)."
    )
    L.append("")
    L.append(
        "This differs from prior LLM alt-key tests (-7pp r@20) which "
        "generated alt-keys broadly (49% of turns) and used max-score "
        "merge (competed with v2f). Here the filter is narrow (only "
        "turns with unresolved pronouns) and the merge is stacked "
        "(validated by critical_info_store)."
    )
    L.append("")

    # Flagging
    L.append("## 1. Resolution rate")
    L.append("")
    L.append("| dataset | turns | resolved | SKIP rate |")
    L.append("|---|---:|---:|---:|")
    for ds_name, res in all_results.items():
        n = res["n_target_segments"]
        r = res["n_turns_resolved"]
        sk = 1.0 - (r / max(n, 1))
        L.append(f"| {ds_name} | {n} | {r} | {sk*100:.1f}% |")
    L.append("")

    # Recall
    L.append("## 2. Recall (fair-backfill)")
    L.append("")
    L.append(
        "| dataset | K | cosine | v2f | stacked | Δ vs v2f | with_bonus | "
        "Δ vs v2f |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds_name, res in all_results.items():
        for K in BUDGETS:
            c = res["cosine_baseline"]["summary"][f"mean_r@{K}"]
            b = res["v2f"]["summary"][f"mean_r@{K}"]
            s = res["pronoun_resolve_stacked"]["summary"][f"mean_r@{K}"]
            bo = res["pronoun_resolve_with_bonus"]["summary"][f"mean_r@{K}"]
            L.append(
                f"| {ds_name} | {K} | {c:.4f} | {b:.4f} | {s:.4f} "
                f"| {s-b:+.4f} | {bo:.4f} | {bo-b:+.4f} |"
            )
    L.append("")

    # Per-category (LoCoMo)
    if "locomo_30q" in all_results:
        L.append("## 3. Per-category (LoCoMo-30)")
        L.append("")
        res = all_results["locomo_30q"]
        cats = sorted(res["v2f"]["by_category"].keys())
        L.append(
            "| category | n | v2f @20 | stacked @20 | bonus @20 | "
            "v2f @50 | stacked @50 | bonus @50 |"
        )
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for cat in cats:
            b = res["v2f"]["by_category"][cat]
            s = res["pronoun_resolve_stacked"]["by_category"].get(cat, {})
            bo = res["pronoun_resolve_with_bonus"]["by_category"].get(cat, {})
            L.append(
                f"| {cat} | {b['n']} "
                f"| {b.get('mean_r@20', 0):.3f} "
                f"| {s.get('mean_r@20', 0):.3f} "
                f"| {bo.get('mean_r@20', 0):.3f} "
                f"| {b.get('mean_r@50', 0):.3f} "
                f"| {s.get('mean_r@50', 0):.3f} "
                f"| {bo.get('mean_r@50', 0):.3f} |"
            )
        L.append("")

    # Orthogonality
    L.append("## 4. Orthogonality (gold surfaced via resolved-only)")
    L.append("")
    L.append(
        "| dataset | variant | K | frac gold via resolved | "
        "frac questions with resolved-gold |"
    )
    L.append("|---|---|---:|---:|---:|")
    for ds_name, res in all_results.items():
        for var in ("pronoun_resolve_stacked", "pronoun_resolve_with_bonus"):
            for K in BUDGETS:
                o = res[var].get(f"orthogonality@{K}", {})
                if not o:
                    continue
                L.append(
                    f"| {ds_name} | {var} | {K} "
                    f"| {o['frac_gold_via_resolved']*100:.1f}% "
                    f"| {o['frac_questions_with_resolved_gold']*100:.1f}% |"
                )
    L.append("")

    # Cost
    L.append("## 5. Cost")
    L.append("")
    L.append(
        f"- LLM calls: uncached={cost['n_uncached']} cached={cost['n_cached']}"
    )
    L.append(f"- Input tokens: {cost['prompt_tokens']}")
    L.append(f"- Output tokens: {cost['completion_tokens']}")
    L.append(f"- Est. cost (gpt-5-mini): ${cost['est_usd']:.3f}")
    L.append("")

    # Samples
    L.append("## 6. Sample resolutions (first 6 RESOLVED turns)")
    L.append("")
    shown = 0
    for ds_name, decs in all_decisions.items():
        for d in decs:
            if not d.resolved:
                continue
            L.append(f"- **{ds_name} turn {d.turn_id} ({d.role})**")
            L.append(f"  - original:  {d.text[:200]}")
            L.append(f"  - resolved:  {d.resolved_text[:200]}")
            shown += 1
            if shown >= 6:
                break
        if shown >= 6:
            break
    L.append("")

    # Verdict placeholder -- derived from table
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v1", choices=list(["v1"]))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--datasets", default="locomo_30q,synthetic_19q")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    client = OpenAI(timeout=60.0)

    resolver = PronounResolver(
        client=client,
        prompt_version=args.prompt,
        max_workers=args.workers,
        cache=PronounLLMCache(),
    )
    # Use the Embedder (shared caches) for main-index cosine/v2f; use a
    # dedicated PronounEmbeddingCache for the resolved-text embeddings so
    # we don't pollute bestshot_embedding_cache.
    embedder = Embedder(client)
    resolved_cache = PronounEmbeddingCache()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results: dict = {}
    all_decisions: dict[str, list[PronounTurnDecision]] = {}

    for ds_name in datasets:
        res, decisions = run_dataset(
            ds_name, resolver, client, embedder, resolved_cache,
        )
        all_results[ds_name] = res
        all_decisions[ds_name] = decisions

    # Cost
    cost = {
        "n_uncached": resolver.n_uncached,
        "n_cached": resolver.n_cached,
        "prompt_tokens": resolver.total_prompt_tokens,
        "completion_tokens": resolver.total_completion_tokens,
    }
    cost["est_usd"] = round(
        cost["prompt_tokens"] * 0.25 / 1e6
        + cost["completion_tokens"] * 2.0 / 1e6,
        4,
    )

    try:
        resolver.save()
    except Exception as e:
        print(f"  (warn) resolver.save: {e}", flush=True)
    try:
        resolved_cache.save()
    except Exception as e:
        print(f"  (warn) resolved_cache.save: {e}", flush=True)
    try:
        embedder.save()
    except Exception as e:
        print(f"  (warn) embedder.save: {e}", flush=True)

    # Outputs
    md = render_markdown(all_results, all_decisions, cost)
    md_path = RESULTS_DIR / "pronoun_resolution.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nWrote {md_path}", flush=True)

    def strip_large(res: dict) -> dict:
        out = {k: v for k, v in res.items()}
        for key in (
            "cosine_baseline", "v2f", "pronoun_resolve_stacked",
            "pronoun_resolve_with_bonus",
        ):
            if key in out and isinstance(out[key], dict):
                per_q = out[key].get("per_question", [])
                pruned = []
                for r in per_q:
                    pruned.append({
                        k: v for k, v in r.items()
                        if not k.startswith("retrieved_ids")
                    })
                out[key] = {**out[key], "per_question": pruned}
        return out

    json_path = RESULTS_DIR / "pronoun_resolution.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "prompt_version": args.prompt,
                "elapsed_s": round(time.time() - t0, 2),
                "cost": cost,
                "results": {
                    ds: strip_large(res) for ds, res in all_results.items()
                },
            },
            f, indent=2, default=str,
        )
    print(f"Wrote {json_path}", flush=True)

    # Resolved texts (reusable)
    resolved_path = RESULTS_DIR / "resolved_turns.json"
    resolved_payload = {"prompt_version": args.prompt, "datasets": {}}
    for ds_name, decs in all_decisions.items():
        resolved_payload["datasets"][ds_name] = [
            {
                "parent_index": d.parent_index,
                "conversation_id": d.conversation_id,
                "turn_id": d.turn_id,
                "role": d.role,
                "original_text": d.text,
                "resolved_text": d.resolved_text,
                "resolved": d.resolved,
            }
            for d in decs
        ]
    with open(resolved_path, "w") as f:
        json.dump(resolved_payload, f, indent=2)
    print(f"Wrote {resolved_path}", flush=True)

    # Console summary
    print("\n" + "=" * 70)
    print("PRONOUN-RESOLUTION RESULTS")
    print("=" * 70)
    for ds_name, res in all_results.items():
        print(f"\n{ds_name}:")
        n = res["n_target_segments"]
        r = res["n_turns_resolved"]
        print(
            f"  resolved={r}/{n} ({r/max(n,1)*100:.1f}%)  "
            f"skip_rate={(1.0 - r/max(n,1))*100:.1f}%"
        )
        for K in BUDGETS:
            c = res["cosine_baseline"]["summary"][f"mean_r@{K}"]
            b = res["v2f"]["summary"][f"mean_r@{K}"]
            s = res["pronoun_resolve_stacked"]["summary"][f"mean_r@{K}"]
            bo = res["pronoun_resolve_with_bonus"]["summary"][f"mean_r@{K}"]
            print(
                f"  K={K}: cosine={c:.4f} v2f={b:.4f} "
                f"stacked={s:.4f} (Δ{s-b:+.4f}) bonus={bo:.4f} (Δ{bo-b:+.4f})"
            )
    print(f"\nLLM cost: ~${cost['est_usd']:.3f}")
    print(f"Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
