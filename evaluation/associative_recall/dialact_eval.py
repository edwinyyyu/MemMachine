"""Empirical evaluation of dialogue-act tagging + act-routed retrieval.

Each turn is tagged by an LLM (gpt-5-mini) with a speech-act label
(DECISION, COMMITMENT, RETRACTION, UNRESOLVED, CLARIFICATION, STATEMENT).
Non-STATEMENT turns populate SEPARATE per-act vector stores.

At query time, keyword rules (and optionally a per-query LLM call) decide
which acts are relevant. Top-M hits from those act-indices are merged with
the main v2f retrieval via the always-top-M or additive-bonus pattern that
critical_info_store validated.

Variants:
  - dialact_keyword_route       : keyword-based query -> act routing
  - dialact_llm_route           : LLM classifies query -> act set
  - dialact_plus_v2f            : alias for keyword-route + v2f (ship candidate)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import (
    Segment,
    SegmentStore,
)

# Patch BestshotEmbeddingCache before import of best_shot users (to tolerate
# concurrent corrupt cache files written by other agents).
import best_shot as _best_shot_module  # noqa: E402


_ORIG_BEC_INIT = _best_shot_module.BestshotEmbeddingCache.__init__


def _safe_bec_init(self):
    from associative_recall import CACHE_DIR as _CACHE_DIR
    self.cache_dir = _CACHE_DIR
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    self._cache: dict = {}
    for name in (
        "embedding_cache.json",
        "arch_embedding_cache.json",
        "agent_embedding_cache.json",
        "frontier_embedding_cache.json",
        "meta_embedding_cache.json",
        "optim_embedding_cache.json",
        "synth_test_embedding_cache.json",
        "bestshot_embedding_cache.json",
    ):
        p = self.cache_dir / name
        if not p.exists():
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            self._cache.update(data)
        except Exception:
            try:
                with open(p, "rb") as f:
                    raw = f.read().decode("utf-8", errors="replace")
                obj, _ = json.JSONDecoder().raw_decode(raw)
                self._cache.update(obj)
                print(f"  (warn) cache {name} corrupt; recovered "
                      f"{len(obj)} entries via raw_decode", flush=True)
            except Exception as e:
                print(f"  (warn) skipping corrupt cache {name}: {e}",
                      flush=True)
    self.cache_file = self.cache_dir / "bestshot_embedding_cache.json"
    self._new_entries = {}


_best_shot_module.BestshotEmbeddingCache.__init__ = _safe_bec_init  # type: ignore[method-assign]


_ORIG_BEC_SAVE = _best_shot_module.BestshotEmbeddingCache.save


def _safe_bec_save(self):
    if not self._new_entries:
        return
    try:
        existing = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    existing = json.load(f)
            except Exception:
                try:
                    with open(self.cache_file, "rb") as f:
                        raw = f.read().decode("utf-8", errors="replace")
                    existing, _ = json.JSONDecoder().raw_decode(raw)
                except Exception:
                    existing = {}
        existing.update(self._new_entries)
        tmp = self.cache_file.with_suffix(f".json.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self.cache_file)
    except Exception as e:
        print(f"  (warn) BEC save race: {e}", flush=True)


_best_shot_module.BestshotEmbeddingCache.save = _safe_bec_save  # type: ignore[method-assign]


from best_shot import MetaV2f  # noqa: E402

from dialogue_act import (  # noqa: E402
    ACT_LABELS,
    ActIndex,
    DialactLLMCache,
    DialogueActTagger,
    TurnActLabel,
    act_distribution,
    build_act_indices,
    combine_act_hits,
    merge_additive_bonus,
    merge_always_top_m,
    route_query_keywords,
    tag_turns,
)
from ingest_regex_eval import (  # noqa: E402
    BUDGETS,
    Embedder,
    compute_recall,
    fair_backfill_turn_ids,
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

TARGET_ACTS = ("DECISION", "COMMITMENT", "RETRACTION", "UNRESOLVED")


# ---------------------------------------------------------------------------
# Cache-save safety
# ---------------------------------------------------------------------------
def _safe_save_caches(arch) -> None:
    try:
        arch.save_caches()
    except Exception as e:
        print(f"  (warn) arch.save_caches failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# v2f ranked main list (arch + cosine backfill) — mirror critical_info_eval
# ---------------------------------------------------------------------------
def v2f_main_ranked(
    arch: MetaV2f,
    store: SegmentStore,
    q: dict,
) -> tuple[list[tuple[Segment, float]], list[Segment]]:
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

    seen: set[int] = set()
    main_ranked: list[tuple[Segment, float]] = []
    EPS = 0.001
    for rank, s in enumerate(arch_segs):
        if s.index in seen:
            continue
        main_ranked.append((s, 10.0 - rank * EPS))
        seen.add(s.index)
    for s in cos_segs:
        if s.index in seen:
            continue
        main_ranked.append((s, cos_score_by_idx.get(s.index, 0.0)))
        seen.add(s.index)
    return main_ranked, arch_segs


# ---------------------------------------------------------------------------
# Baseline v2f (mirrors critical_info_eval.run_baseline_v2f)
# ---------------------------------------------------------------------------
def run_baseline_v2f(
    store: SegmentStore,
    embedder: Embedder,
    questions: list[dict],
) -> list[dict]:
    arch = MetaV2f(store)
    arch.embedding_cache = embedder.embedding_cache
    arch.llm_cache = embedder.llm_cache

    out: list[dict] = []
    for i, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])

        arch.reset_counters()
        result = arch.retrieve(q_text, conv_id)
        arch_segs = list(result.segments)

        q_emb = arch.embed_text(q_text)
        max_K = max(BUDGETS)
        cos_res = store.search(q_emb, top_k=max_K, conversation_id=conv_id)
        cos_segs = list(cos_res.segments)

        row = {
            "conversation_id": conv_id,
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q_text,
            "source_chat_ids": sorted(source_ids),
        }
        for K in BUDGETS:
            ids = fair_backfill_turn_ids(arch_segs, cos_segs, K)
            row[f"r@{K}"] = compute_recall(ids, source_ids)
        out.append(row)
        if (i + 1) % 5 == 0:
            _safe_save_caches(arch)
    _safe_save_caches(arch)
    return out


# ---------------------------------------------------------------------------
# Variant runner (act-routed)
# ---------------------------------------------------------------------------
def run_variant(
    variant: str,
    store: SegmentStore,
    act_indices: dict[str, ActIndex],
    embedder: Embedder,
    questions: list[dict],
    tagger: DialogueActTagger | None,
    top_m_act: int = 5,
    bonus: float = 0.1,
    min_score: float = 0.2,
    merge_mode: str = "always_top_m",
) -> list[dict]:
    """Run an act-routed variant.

    variant ∈ {
        "dialact_keyword_route",   # keyword-only routing
        "dialact_llm_route",       # LLM routing (requires tagger)
        "dialact_plus_v2f",        # keyword-route merged with v2f (ship candidate)
    }
    merge_mode ∈ {"always_top_m", "additive_bonus"}
    """
    arch = MetaV2f(store)
    arch.embedding_cache = embedder.embedding_cache
    arch.llm_cache = embedder.llm_cache

    out: list[dict] = []
    route_counts = defaultdict(int)
    for i, q in enumerate(questions):
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])
        q_text = q["question"]

        main_ranked, _ = v2f_main_ranked(arch, store, q)
        q_emb = arch.embed_text(q_text)

        # Query -> act routing
        if variant == "dialact_llm_route":
            assert tagger is not None, "LLM route requires tagger"
            routed, _raw = tagger.route_query(q_text)
        else:
            routed = route_query_keywords(q_text)

        # Per-act retrieval
        max_K = max(BUDGETS)
        act_hits: dict[str, list[tuple[int, float, Segment]]] = {}
        for act in routed:
            idx = act_indices.get(act)
            if idx is None:
                continue
            hits = idx.search_per_parent(
                q_emb,
                top_m=max_K,
                conversation_id=conv_id,
                min_score=-1.0,
            )
            if hits:
                act_hits[act] = hits

        combined = combine_act_hits(act_hits, top_m=max_K)

        row = {
            "conversation_id": conv_id,
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q_text,
            "source_chat_ids": sorted(source_ids),
            "routed_acts": sorted(routed),
            "n_act_hits": sum(len(h) for h in act_hits.values()),
        }
        for a in sorted(routed):
            route_counts[a] += 1
        if not routed:
            route_counts["NONE"] += 1

        for K in BUDGETS:
            if merge_mode == "additive_bonus":
                merged = merge_additive_bonus(
                    main_ranked, combined, K, bonus=bonus,
                )
            else:
                merged = merge_always_top_m(
                    main_ranked, combined, K,
                    top_m=top_m_act, min_score=min_score,
                )
            retrieved_ids = {s.turn_id for s in merged}
            row[f"r@{K}"] = compute_recall(retrieved_ids, source_ids)

            main_topK_ids = {s.turn_id for s, _ in main_ranked[:K]}
            act_gold = retrieved_ids & source_ids - main_topK_ids
            row[f"act_gold_contrib@{K}"] = sorted(act_gold)

        out.append(row)
        if (i + 1) % 5 == 0:
            _safe_save_caches(arch)
    _safe_save_caches(arch)
    return out, dict(route_counts)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(per_q: list[dict]) -> dict:
    n = len(per_q)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for K in BUDGETS:
        vals = [r[f"r@{K}"] for r in per_q]
        out[f"mean_r@{K}"] = round(sum(vals) / n, 4) if vals else 0.0
    return out


def summarize_by_category(per_q: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in per_q:
        by_cat[r.get("category", "unknown")].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry = {"n": n}
        for K in BUDGETS:
            vals = [r[f"r@{K}"] for r in rs]
            entry[f"mean_r@{K}"] = round(sum(vals) / n, 4) if n else 0.0
        out[cat] = entry
    return out


def act_contribution(per_q: list[dict], K: int) -> dict:
    n = len(per_q)
    if n == 0:
        return {"n": 0}
    q_with = 0
    total_gold = 0
    act_gold = 0
    for r in per_q:
        gold = set(r["source_chat_ids"])
        contrib = set(r.get(f"act_gold_contrib@{K}", []))
        total_gold += len(gold)
        act_gold += len(contrib & gold)
        if contrib & gold:
            q_with += 1
    return {
        "n": n, "K": K,
        "frac_questions_with_act_gold": round(q_with / n, 4),
        "frac_gold_via_act": round(act_gold / max(total_gold, 1), 4),
    }


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------
def run_dataset(
    ds_name: str,
    tagger: DialogueActTagger,
    client: OpenAI,
    embedder: Embedder,
    use_llm_route: bool,
) -> tuple[dict, list[TurnActLabel]]:
    cfg = DATASETS[ds_name]
    print(f"\n{'=' * 70}\nDataset: {ds_name}\n{'=' * 70}", flush=True)
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        all_qs = json.load(f)
    qs = all_qs
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    if cfg["max_questions"]:
        qs = qs[: cfg["max_questions"]]
    print(f"  questions: {len(qs)} | segments: {len(store.segments)}", flush=True)

    # Subset to question conversations
    conv_ids = {q["conversation_id"] for q in qs}
    target_segments = [s for s in store.segments if s.conversation_id in conv_ids]
    print(f"  target segments: {len(target_segments)}", flush=True)

    # Tag
    print(f"  tagging {len(target_segments)} turns ...", flush=True)
    labels = tag_turns(tagger, target_segments)
    dist = act_distribution(labels)
    total = sum(dist.values())
    print(f"  act distribution (n={total}):", flush=True)
    for a in ACT_LABELS + ("UNKNOWN",):
        c = dist.get(a, 0)
        pct = c / max(total, 1) * 100
        print(f"    {a:>15} : {c:>5d}  ({pct:5.1f}%)", flush=True)

    # Act indices
    act_indices = build_act_indices(store, labels, target_acts=TARGET_ACTS)
    for a, idx in act_indices.items():
        print(f"  act_index[{a}]: {idx.act_normalized.shape[0]} entries", flush=True)

    # Baseline
    print("  [1/4] baseline v2f ...", flush=True)
    baseline_rows = run_baseline_v2f(store, embedder, qs)
    baseline_summary = summarize(baseline_rows)
    baseline_by_cat = summarize_by_category(baseline_rows)

    # Variant: keyword route
    print("  [2/4] dialact_keyword_route ...", flush=True)
    kw_rows, kw_route_counts = run_variant(
        "dialact_keyword_route", store, act_indices, embedder, qs,
        tagger=None, top_m_act=5, min_score=0.2, merge_mode="always_top_m",
    )
    kw_summary = summarize(kw_rows)
    kw_by_cat = summarize_by_category(kw_rows)
    kw_contrib = {K: act_contribution(kw_rows, K) for K in BUDGETS}

    # Variant: LLM route (optional)
    if use_llm_route:
        print("  [3/4] dialact_llm_route ...", flush=True)
        llm_rows, llm_route_counts = run_variant(
            "dialact_llm_route", store, act_indices, embedder, qs,
            tagger=tagger, top_m_act=5, min_score=0.2,
            merge_mode="always_top_m",
        )
        llm_summary = summarize(llm_rows)
        llm_by_cat = summarize_by_category(llm_rows)
        llm_contrib = {K: act_contribution(llm_rows, K) for K in BUDGETS}
    else:
        llm_rows = []
        llm_route_counts = {}
        llm_summary = {}
        llm_by_cat = {}
        llm_contrib = {}

    # Variant: plus_v2f with additive bonus (keyword routing, bonus merge)
    print("  [4/4] dialact_plus_v2f (additive_bonus) ...", flush=True)
    plus_rows, plus_route_counts = run_variant(
        "dialact_keyword_route", store, act_indices, embedder, qs,
        tagger=None, top_m_act=5, bonus=0.1, min_score=0.2,
        merge_mode="additive_bonus",
    )
    plus_summary = summarize(plus_rows)
    plus_by_cat = summarize_by_category(plus_rows)
    plus_contrib = {K: act_contribution(plus_rows, K) for K in BUDGETS}

    return {
        "ds_name": ds_name,
        "n_questions": len(qs),
        "n_target_segments": len(target_segments),
        "act_distribution": dist,
        "act_index_sizes": {a: int(idx.act_normalized.shape[0])
                            for a, idx in act_indices.items()},
        "baseline": {
            "summary": baseline_summary,
            "by_category": baseline_by_cat,
            "per_question": baseline_rows,
        },
        "dialact_keyword_route": {
            "summary": kw_summary,
            "by_category": kw_by_cat,
            "route_counts": kw_route_counts,
            "act_contribution": kw_contrib,
            "per_question": kw_rows,
        },
        "dialact_llm_route": {
            "summary": llm_summary,
            "by_category": llm_by_cat,
            "route_counts": llm_route_counts,
            "act_contribution": llm_contrib,
            "per_question": llm_rows,
        },
        "dialact_plus_v2f": {
            "summary": plus_summary,
            "by_category": plus_by_cat,
            "route_counts": plus_route_counts,
            "act_contribution": plus_contrib,
            "per_question": plus_rows,
        },
    }, labels


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_markdown(results: dict, cost: dict, use_llm_route: bool) -> str:
    L: list[str] = []
    L.append("# Dialogue-Act Tagging + Act-Routed Retrieval — Empirical Recall Test")
    L.append("")
    L.append(
        "At ingestion time an LLM (gpt-5-mini) tags each turn with a speech-act "
        "label. Non-STATEMENT turns populate separate per-act vector stores. "
        "At query time, the query is routed to relevant acts via keyword rules "
        "(and optionally a per-query LLM call). Top-M hits from each act-index "
        "are merged with the v2f main retrieval using the same always-top-M / "
        "additive-bonus pattern that critical_info_store validated."
    )
    L.append("")

    # ---- Act distribution ----
    L.append("## 1. Act distribution (tagged turns)")
    L.append("")
    L.append("| dataset | n_turns | STATEMENT | DECISION | COMMITMENT | RETRACTION | UNRESOLVED | CLARIFICATION | UNKNOWN |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds, res in results.items():
        d = res["act_distribution"]
        n = res["n_target_segments"]
        def pct(x): return f"{x} ({x/max(n,1)*100:.1f}%)"
        L.append(
            f"| {ds} | {n} | {pct(d.get('STATEMENT',0))} "
            f"| {pct(d.get('DECISION',0))} | {pct(d.get('COMMITMENT',0))} "
            f"| {pct(d.get('RETRACTION',0))} | {pct(d.get('UNRESOLVED',0))} "
            f"| {pct(d.get('CLARIFICATION',0))} | {pct(d.get('UNKNOWN',0))} |"
        )
    L.append("")

    # ---- Recall ----
    L.append("## 2. Recall (fair-backfill)")
    L.append("")
    variants = ["dialact_keyword_route", "dialact_plus_v2f"]
    if use_llm_route:
        variants.append("dialact_llm_route")

    header = "| dataset | K | baseline v2f |"
    sep = "|---|---:|---:|"
    for v in variants:
        header += f" {v} | Δ |"
        sep += "---:|---:|"
    L.append(header)
    L.append(sep)
    for ds, res in results.items():
        b_sum = res["baseline"]["summary"]
        for K in BUDGETS:
            b = b_sum.get(f"mean_r@{K}", 0.0)
            row = f"| {ds} | {K} | {b:.4f} |"
            for v in variants:
                vv = res[v]["summary"].get(f"mean_r@{K}", 0.0)
                row += f" {vv:.4f} | {vv-b:+.4f} |"
            L.append(row)
    L.append("")

    # ---- Per-category (LoCoMo + synthetic — both) ----
    for ds in results:
        res = results[ds]
        cats = sorted(res["baseline"]["by_category"].keys())
        if not cats:
            continue
        L.append(f"## 3. Per-category — {ds}")
        L.append("")
        L.append("| category | n | base@20 | kw@20 | plus@20 | base@50 | kw@50 | plus@50 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for cat in cats:
            bc = res["baseline"]["by_category"][cat]
            kw = res["dialact_keyword_route"]["by_category"].get(cat, {})
            pl = res["dialact_plus_v2f"]["by_category"].get(cat, {})
            L.append(
                f"| {cat} | {bc['n']} "
                f"| {bc.get('mean_r@20', 0):.3f} | {kw.get('mean_r@20', 0):.3f} "
                f"| {pl.get('mean_r@20', 0):.3f} "
                f"| {bc.get('mean_r@50', 0):.3f} | {kw.get('mean_r@50', 0):.3f} "
                f"| {pl.get('mean_r@50', 0):.3f} |"
            )
        L.append("")

    # ---- Act contribution ----
    L.append("## 4. Act-contribution rate")
    L.append("")
    L.append(
        "Fraction of questions where the act-routed store surfaced gold "
        "outside the main top-K."
    )
    L.append("")
    L.append("| dataset | variant | K | frac Q with act-gold | frac gold via act |")
    L.append("|---|---|---:|---:|---:|")
    for ds, res in results.items():
        for v in variants:
            for K in BUDGETS:
                cc = res[v].get("act_contribution", {}).get(K)
                if cc is None or cc.get("n", 0) == 0:
                    continue
                L.append(
                    f"| {ds} | {v} | {K} "
                    f"| {cc['frac_questions_with_act_gold']*100:.1f}% "
                    f"| {cc['frac_gold_via_act']*100:.1f}% |"
                )
    L.append("")

    # ---- Route counts ----
    L.append("## 5. Query-routing distribution (keyword rules)")
    L.append("")
    for ds, res in results.items():
        rc = res["dialact_keyword_route"]["route_counts"]
        L.append(f"- {ds}: " + ", ".join(
            f"{k}={v}" for k, v in sorted(rc.items())))
    L.append("")

    # ---- Cost ----
    L.append("## 6. Cost")
    L.append("")
    L.append(f"- Uncached calls: {cost['n_uncached']}")
    L.append(f"- Cached calls: {cost['n_cached']}")
    L.append(f"- Prompt tokens: {cost['prompt_tokens']}")
    L.append(f"- Completion tokens: {cost['completion_tokens']}")
    L.append(f"- Est. cost (gpt-5-mini): ${cost['est_usd']:.3f}")
    L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--datasets", default="locomo_30q,synthetic_19q",
        help="comma-separated list",
    )
    parser.add_argument("--use_llm_route", action="store_true",
                        help="also run dialact_llm_route variant (adds LLM calls)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    client = OpenAI(timeout=60.0)
    tagger = DialogueActTagger(
        client=client,
        max_workers=args.workers,
        cache=DialactLLMCache(),
    )
    embedder = Embedder(client)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results: dict = {}
    all_labels: list[TurnActLabel] = []

    for ds_name in datasets:
        res, labels = run_dataset(
            ds_name, tagger, client, embedder,
            use_llm_route=args.use_llm_route,
        )
        all_results[ds_name] = res
        all_labels.extend(labels)

    # Cost
    cost = {
        "n_uncached": tagger.n_uncached,
        "n_cached": tagger.n_cached,
        "prompt_tokens": tagger.total_prompt_tokens,
        "completion_tokens": tagger.total_completion_tokens,
    }
    cost["est_usd"] = round(
        cost["prompt_tokens"] * 0.25 / 1e6
        + cost["completion_tokens"] * 2.0 / 1e6,
        4,
    )

    try:
        tagger.save()
    except Exception as e:
        print(f"  (warn) tagger.save failed: {e}", flush=True)
    try:
        embedder.save()
    except Exception as e:
        print(f"  (warn) embedder.save failed: {e}", flush=True)

    # Markdown
    md = render_markdown(all_results, cost, args.use_llm_route)
    md_path = RESULTS_DIR / "dialogue_act_study.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nWrote {md_path}", flush=True)

    # JSON — strip retrieved_ids lists
    def strip_large(res: dict) -> dict:
        out = dict(res)
        for key in [
            "baseline", "dialact_keyword_route", "dialact_llm_route",
            "dialact_plus_v2f",
        ]:
            if key in out and isinstance(out[key], dict):
                per_q = out[key].get("per_question", [])
                pruned = []
                for r in per_q:
                    pruned.append({
                        k: v for k, v in r.items()
                        if not (k.startswith("retrieved_ids")
                                or k.startswith("act_gold_contrib"))
                    })
                out[key] = {**out[key], "per_question": pruned}
        return out

    json_path = RESULTS_DIR / "dialogue_act_study.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "elapsed_s": round(time.time() - t0, 2),
                "cost": cost,
                "use_llm_route": args.use_llm_route,
                "results": {ds: strip_large(r) for ds, r in all_results.items()},
            },
            f, indent=2, default=str,
        )
    print(f"Wrote {json_path}", flush=True)

    # Turn labels — for reuse
    labels_path = RESULTS_DIR / "turn_act_labels.json"
    with open(labels_path, "w") as f:
        json.dump(
            [
                {
                    "conversation_id": lab.conversation_id,
                    "turn_id": lab.turn_id,
                    "parent_index": lab.parent_index,
                    "role": lab.role,
                    "label": lab.label,
                    "text": lab.text[:500],
                }
                for lab in all_labels
            ],
            f, indent=2,
        )
    print(f"Wrote {labels_path}", flush=True)

    # Console summary
    print("\n" + "=" * 70)
    print("DIALOGUE-ACT STUDY RESULTS")
    print("=" * 70)
    for ds, res in all_results.items():
        b = res["baseline"]["summary"]
        kw = res["dialact_keyword_route"]["summary"]
        pl = res["dialact_plus_v2f"]["summary"]
        print(f"\n{ds}:")
        for K in BUDGETS:
            b_k = b.get(f"mean_r@{K}", 0)
            kw_k = kw.get(f"mean_r@{K}", 0)
            pl_k = pl.get(f"mean_r@{K}", 0)
            print(f"  K={K}: baseline={b_k:.4f}  kw_route={kw_k:.4f} "
                  f"(Δ{kw_k-b_k:+.4f})  plus_v2f={pl_k:.4f} "
                  f"(Δ{pl_k-b_k:+.4f})")
        if args.use_llm_route:
            llm = res["dialact_llm_route"]["summary"]
            for K in BUDGETS:
                llm_k = llm.get(f"mean_r@{K}", 0)
                b_k = b.get(f"mean_r@{K}", 0)
                print(f"  K={K}: llm_route={llm_k:.4f} "
                      f"(Δ{llm_k-b_k:+.4f})")

    print(f"\nLLM cost: ~${cost['est_usd']:.3f}")
    print(f"Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
