"""Empirical evaluation of answer-type aware reranking.

Runs baseline v2f and reranked variants (additive-bonus alpha ∈ {0.05, 0.1, 0.2},
hard filter) on LoCoMo-30 and synthetic-19 at K=20 and K=50 under fair-backfill.

Outputs:
  results/answer_type_rerank.md
  results/answer_type_rerank.json

Usage: uv run python atr_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

# Safe patching of BestshotEmbeddingCache before importing anything that uses it.
import best_shot as _best_shot_module
from associative_recall import (
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from openai import OpenAI

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
            except Exception as e:
                print(f"  (warn) skipping corrupt cache {name}: {e}", flush=True)
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


from answer_type_rerank import (  # noqa: E402
    ANSWER_TYPES,
    AnswerTypeDecisionCache,
    AnswerTypeLLMCache,
    classify_answer_type,
    rerank_additive_bonus,
    rerank_hard_filter,
    turn_has_answer_type_tokens,
)
from best_shot import MetaV2f  # noqa: E402
from ingest_regex_eval import BUDGETS, Embedder, compute_recall  # noqa: E402

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

ALPHAS = [0.05, 0.1, 0.2]


def _safe_save_caches(arch) -> None:
    try:
        arch.save_caches()
    except Exception as e:
        print(f"  (warn) arch.save_caches failed: {e}", flush=True)


def v2f_main_ranked(
    arch: MetaV2f,
    store: SegmentStore,
    q: dict,
) -> list[tuple[Segment, float]]:
    """Retrieve v2f candidates + cosine backfill, returning (seg, score) ranked.

    Arch segments get high decreasing scores (10.0, 10.0-eps, ...), cosine
    segments get their cosine scores. This means arch's order is preserved as
    the primary ranking, with cosine backfill for the tail.
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

    seen: set[int] = set()
    main_ranked: list[tuple[Segment, float]] = []
    EPS = 0.001
    # Arch segments first (tops of the list); give them big scores so that
    # additive-bonus rerank stays stable among them by default.
    for rank, s in enumerate(arch_segs):
        if s.index in seen:
            continue
        main_ranked.append((s, 10.0 - rank * EPS))
        seen.add(s.index)
    # Cosine backfill with real cosine scores (for the tail)
    for s in cos_segs:
        if s.index in seen:
            continue
        main_ranked.append((s, cos_score_by_idx.get(s.index, 0.0)))
        seen.add(s.index)
    return main_ranked


def truncate_to_K(ranked: list[tuple[Segment, float]], K: int) -> set[int]:
    return {s.turn_id for s, _ in ranked[:K]}


def run_variant(
    variant: str,
    store: SegmentStore,
    embedder: Embedder,
    questions: list[dict],
    answer_types: dict[str, str],
    alpha: float | None = None,
) -> list[dict]:
    """Run a rerank variant.

    variant ∈ {"baseline", "atr_bonus", "atr_hard_filter"}
    If variant == "atr_bonus", alpha must be set.
    """
    arch = MetaV2f(store)
    arch.embedding_cache = embedder.embedding_cache
    arch.llm_cache = embedder.llm_cache

    out = []
    for i, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])

        main_ranked = v2f_main_ranked(arch, store, q)

        if variant == "baseline":
            ranked = main_ranked
            at = answer_types.get(q_text, "DESCRIPTION")
        else:
            at = answer_types.get(q_text, "DESCRIPTION")
            if at == "DESCRIPTION":
                # No rerank for DESCRIPTION (every turn passes)
                ranked = main_ranked
            elif variant == "atr_bonus":
                assert alpha is not None
                ranked = rerank_additive_bonus(main_ranked, at, alpha)
            elif variant == "atr_hard_filter":
                ranked = rerank_hard_filter(main_ranked, at)
            else:
                raise ValueError(f"Unknown variant: {variant}")

        row = {
            "conversation_id": conv_id,
            "category": q.get("category", "unknown"),
            "question_index": q.get("question_index", -1),
            "question": q_text,
            "answer_type": at,
            "source_chat_ids": sorted(source_ids),
        }
        for K in BUDGETS:
            ids = truncate_to_K(ranked, K)
            row[f"r@{K}"] = compute_recall(ids, source_ids)
        out.append(row)

        if (i + 1) % 5 == 0:
            _safe_save_caches(arch)
    _safe_save_caches(arch)
    return out


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


def wtl_vs_baseline(
    variant_rows: list[dict],
    baseline_rows: list[dict],
    K: int,
) -> tuple[int, int, int]:
    """Win/Tie/Loss count per question against baseline."""
    by_key = {(r["conversation_id"], r["question_index"]): r for r in baseline_rows}
    w = t = l = 0
    for r in variant_rows:
        k = (r["conversation_id"], r["question_index"])
        b = by_key.get(k)
        if b is None:
            continue
        va = r[f"r@{K}"]
        vb = b[f"r@{K}"]
        if va > vb + 1e-6:
            w += 1
        elif vb > va + 1e-6:
            l += 1
        else:
            t += 1
    return w, t, l


def run_dataset(
    ds_name: str,
    client: OpenAI,
    embedder: Embedder,
    llm_cache: AnswerTypeLLMCache,
    decision_cache: AnswerTypeDecisionCache,
) -> dict:
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

    # Classify
    print("  classifying answer types ...", flush=True)
    answer_types: dict[str, str] = {}
    for q in qs:
        at = classify_answer_type(
            q["question"],
            llm_cache,
            decision_cache,
            client=client,
            model="gpt-5-mini",
        )
        answer_types[q["question"]] = at
    llm_cache.save()
    decision_cache.save()

    at_dist = Counter(answer_types.values())
    for k in ANSWER_TYPES:
        if k not in at_dist:
            at_dist[k] = 0
    print(f"  answer-type distribution: {dict(at_dist)}", flush=True)

    # Baseline
    print("  [1/5] baseline v2f ...", flush=True)
    baseline_rows = run_variant("baseline", store, embedder, qs, answer_types)
    baseline_summary = summarize(baseline_rows)
    baseline_by_cat = summarize_by_category(baseline_rows)

    variant_results: dict[str, dict] = {}

    for i, alpha in enumerate(ALPHAS, start=2):
        key = f"atr_bonus_{alpha}"
        print(f"  [{i}/5] {key} ...", flush=True)
        rows = run_variant(
            "atr_bonus",
            store,
            embedder,
            qs,
            answer_types,
            alpha=alpha,
        )
        variant_results[key] = {
            "summary": summarize(rows),
            "by_category": summarize_by_category(rows),
            "wtl": {f"r@{K}": wtl_vs_baseline(rows, baseline_rows, K) for K in BUDGETS},
            "per_question": rows,
        }

    print("  [5/5] atr_hard_filter ...", flush=True)
    hf_rows = run_variant("atr_hard_filter", store, embedder, qs, answer_types)
    variant_results["atr_hard_filter"] = {
        "summary": summarize(hf_rows),
        "by_category": summarize_by_category(hf_rows),
        "wtl": {f"r@{K}": wtl_vs_baseline(hf_rows, baseline_rows, K) for K in BUDGETS},
        "per_question": hf_rows,
    }

    # Compute a sample rerank example from LoCoMo (top-5 before/after)
    sample_example = None
    if ds_name == "locomo_30q" and qs:
        arch_sample = MetaV2f(store)
        arch_sample.embedding_cache = embedder.embedding_cache
        arch_sample.llm_cache = embedder.llm_cache
        # Pick the first DATE-type question with a non-trivial rerank effect
        for q in qs:
            q_text = q["question"]
            at = answer_types.get(q_text, "DESCRIPTION")
            if at == "DESCRIPTION":
                continue
            mr = v2f_main_ranked(arch_sample, store, q)
            if not mr:
                continue
            before = mr[:5]
            reranked = rerank_additive_bonus(mr, at, 0.1)[:5]
            before_ids = [s.turn_id for s, _ in before]
            after_ids = [s.turn_id for s, _ in reranked]
            if before_ids != after_ids:
                sample_example = {
                    "question": q_text,
                    "answer_type": at,
                    "source_chat_ids": sorted(q["source_chat_ids"]),
                    "before_top5": [
                        {
                            "turn_id": s.turn_id,
                            "role": s.role,
                            "text": s.text[:200],
                            "score": round(float(sc), 4),
                            "has_at_token": turn_has_answer_type_tokens(s.text, at),
                        }
                        for s, sc in before
                    ],
                    "after_top5": [
                        {
                            "turn_id": s.turn_id,
                            "role": s.role,
                            "text": s.text[:200],
                            "score": round(float(sc), 4),
                            "has_at_token": turn_has_answer_type_tokens(s.text, at),
                        }
                        for s, sc in reranked
                    ],
                }
                break
        _safe_save_caches(arch_sample)

    return {
        "ds_name": ds_name,
        "n_questions": len(qs),
        "answer_type_distribution": dict(at_dist),
        "baseline": {
            "summary": baseline_summary,
            "by_category": baseline_by_cat,
            "per_question": baseline_rows,
        },
        "variants": variant_results,
        "sample_example": sample_example,
    }


def render_markdown(results: dict) -> str:
    L: list[str] = []
    L.append("# Answer-Type Aware Reranking — Empirical Recall Test")
    L.append("")
    L.append(
        "Classify each query's expected answer type (DATE, PERSON, NUMBER, "
        "LOCATION, REASON, DESCRIPTION) via rules (gpt-5-mini fallback for "
        "ambiguous what/which). Rerank v2f's already-retrieved candidates by "
        "adding an alpha bonus to turns containing answer-type tokens, or (in "
        "hard-filter mode) promoting matches to the front. DESCRIPTION queries "
        "are left un-reranked (no informative filter)."
    )
    L.append("")

    # Answer-type distribution
    L.append("## 1. Answer-type distribution")
    L.append("")
    L.append("| dataset | " + " | ".join(ANSWER_TYPES) + " |")
    L.append("|---|" + "---:|" * len(ANSWER_TYPES))
    for ds, res in results.items():
        d = res["answer_type_distribution"]
        n = res["n_questions"]
        cells = []
        for t in ANSWER_TYPES:
            c = d.get(t, 0)
            pct = c / max(n, 1) * 100
            cells.append(f"{c} ({pct:.0f}%)")
        L.append(f"| {ds} | " + " | ".join(cells) + " |")
    L.append("")

    # Recall table
    L.append("## 2. Recall (fair-backfill)")
    L.append("")
    variant_names = [f"atr_bonus_{a}" for a in ALPHAS] + ["atr_hard_filter"]
    header = "| dataset | K | baseline v2f |"
    sep = "|---|---:|---:|"
    for v in variant_names:
        header += f" {v} | Δ | W/T/L |"
        sep += "---:|---:|---:|"
    L.append(header)
    L.append(sep)
    for ds, res in results.items():
        b_sum = res["baseline"]["summary"]
        for K in BUDGETS:
            b = b_sum.get(f"mean_r@{K}", 0.0)
            row = f"| {ds} | {K} | {b:.4f} |"
            for v in variant_names:
                vv = res["variants"][v]["summary"].get(f"mean_r@{K}", 0.0)
                wtl = res["variants"][v]["wtl"].get(f"r@{K}", (0, 0, 0))
                row += f" {vv:.4f} | {vv - b:+.4f} | {wtl[0]}/{wtl[1]}/{wtl[2]} |"
            L.append(row)
    L.append("")

    # Per-category (each dataset)
    for ds, res in results.items():
        cats = sorted(res["baseline"]["by_category"].keys())
        if not cats:
            continue
        L.append(f"## 3. Per-category — {ds}")
        L.append("")
        hdr = (
            "| category | n | base@20 |"
            + "".join(f" {v}@20 |" for v in variant_names)
            + " base@50 |"
            + "".join(f" {v}@50 |" for v in variant_names)
        )
        sep_ = (
            "|---|---:|---:|"
            + "---:|" * len(variant_names)
            + "---:|"
            + "---:|" * len(variant_names)
        )
        L.append(hdr)
        L.append(sep_)
        for cat in cats:
            bc = res["baseline"]["by_category"][cat]
            row = f"| {cat} | {bc['n']} | {bc.get('mean_r@20', 0):.3f} |"
            for v in variant_names:
                vc = res["variants"][v]["by_category"].get(cat, {})
                row += f" {vc.get('mean_r@20', 0):.3f} |"
            row += f" {bc.get('mean_r@50', 0):.3f} |"
            for v in variant_names:
                vc = res["variants"][v]["by_category"].get(cat, {})
                row += f" {vc.get('mean_r@50', 0):.3f} |"
            L.append(row)
        L.append("")

    # Sample
    for ds, res in results.items():
        sx = res.get("sample_example")
        if not sx:
            continue
        L.append(f"## 4. Sample rerank effect — {ds}")
        L.append("")
        L.append(f"**Question:** {sx['question']}")
        L.append("")
        L.append(f"**Answer type:** {sx['answer_type']}  ")
        L.append(f"**Source turn_ids:** {sx['source_chat_ids']}")
        L.append("")
        L.append("### Before (v2f top-5)")
        L.append("")
        L.append("| rank | turn_id | role | has_at_token | score | text |")
        L.append("|---:|---:|---|---:|---:|---|")
        for i, item in enumerate(sx["before_top5"], start=1):
            txt = item["text"].replace("|", "\\|").replace("\n", " ")[:160]
            L.append(
                f"| {i} | {item['turn_id']} | {item['role']} | "
                f"{'yes' if item['has_at_token'] else 'no'} | "
                f"{item['score']:.3f} | {txt} |"
            )
        L.append("")
        L.append("### After (atr_bonus_0.1 top-5)")
        L.append("")
        L.append("| rank | turn_id | role | has_at_token | score | text |")
        L.append("|---:|---:|---|---:|---:|---|")
        for i, item in enumerate(sx["after_top5"], start=1):
            txt = item["text"].replace("|", "\\|").replace("\n", " ")[:160]
            L.append(
                f"| {i} | {item['turn_id']} | {item['role']} | "
                f"{'yes' if item['has_at_token'] else 'no'} | "
                f"{item['score']:.3f} | {txt} |"
            )
        L.append("")
        break  # only one sample example overall

    # Verdict
    L.append("## 5. Verdict")
    L.append("")
    verdict_lines = []
    for ds, res in results.items():
        b_sum = res["baseline"]["summary"]
        any_win = False
        best_delta = 0.0
        best_label = ""
        for v in variant_names:
            for K in BUDGETS:
                d = res["variants"][v]["summary"].get(f"mean_r@{K}", 0.0) - b_sum.get(
                    f"mean_r@{K}", 0.0
                )
                if d > best_delta + 1e-6:
                    best_delta = d
                    best_label = f"{v}@K={K}"
                if d > 1e-4:
                    any_win = True
        if any_win:
            verdict_lines.append(
                f"- {ds}: best variant = {best_label} "
                f"(+{best_delta:.4f}). {'Ship/narrow-use' if best_delta > 0.01 else 'Marginal gain'}."
            )
        else:
            verdict_lines.append(
                f"- {ds}: no variant beats baseline. Abandon on this dataset."
            )
    L.extend(verdict_lines)
    L.append("")

    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="locomo_30q,synthetic_19q",
        help="comma-separated list",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    client = OpenAI(timeout=60.0)
    embedder = Embedder(client)
    llm_cache = AnswerTypeLLMCache()
    decision_cache = AnswerTypeDecisionCache()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results: dict = {}

    for ds_name in datasets:
        res = run_dataset(
            ds_name,
            client,
            embedder,
            llm_cache,
            decision_cache,
        )
        all_results[ds_name] = res

    try:
        embedder.save()
    except Exception as e:
        print(f"  (warn) embedder.save failed: {e}", flush=True)
    try:
        llm_cache.save()
        decision_cache.save()
    except Exception as e:
        print(f"  (warn) cache save failed: {e}", flush=True)

    # Markdown
    md = render_markdown(all_results)
    md_path = RESULTS_DIR / "answer_type_rerank.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nWrote {md_path}", flush=True)

    # JSON
    json_path = RESULTS_DIR / "answer_type_rerank.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "elapsed_s": round(time.time() - t0, 2),
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Wrote {json_path}", flush=True)

    # Console summary
    print("\n" + "=" * 70)
    print("ANSWER-TYPE RERANK RESULTS")
    print("=" * 70)
    variant_names = [f"atr_bonus_{a}" for a in ALPHAS] + ["atr_hard_filter"]
    for ds, res in all_results.items():
        print(f"\n{ds}:")
        b = res["baseline"]["summary"]
        for K in BUDGETS:
            b_k = b.get(f"mean_r@{K}", 0.0)
            line = f"  K={K}: baseline={b_k:.4f}"
            for v in variant_names:
                vk = res["variants"][v]["summary"].get(f"mean_r@{K}", 0.0)
                line += f"  {v}={vk:.4f} (Δ{vk - b_k:+.4f})"
            print(line)

    print(f"\nElapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
