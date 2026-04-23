"""Fair-backfill evaluation for query-projection variants.

Pipeline:
  1. Learn question-ness direction from ~100 question and ~100 statement
     templates. Save direction to results/question_direction.npy.
  2. For alpha in {0.5, 1.0, 1.5}: run `qproj_{alpha}` zero-LLM variant on
     LoCoMo-30 and synthetic-19 at K=20 and K=50 using fair-backfill.
  3. Baselines: cosine_baseline (no projection) and meta_v2f.
  4. If best qproj alpha beats cosine, also run `qproj_{alpha}_v2f` at that
     alpha.
  5. Sanity checks: cosine between random question embedding and q_dir;
     cosine between statement embedding and q_dir.
  6. Dump markdown report + JSON dump + direction file.

Usage:
    uv run python qproj_eval.py
    uv run python qproj_eval.py --alphas 0.5,1.0,1.5
    uv run python qproj_eval.py --datasets locomo_30q
    uv run python qproj_eval.py --skip-v2f   # skip qproj_v2f even if qproj wins
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from associative_recall import Segment, SegmentStore
from best_shot import BestshotResult
from fair_backfill_eval import (
    BUDGETS,
    DATASETS,
    RESULTS_DIR,
    fair_backfill_evaluate,
    load_dataset,
    summarize,
    summarize_by_category,
)
from antipara_cue_gen import MetaV2fDedicated
from query_projection import (
    QUESTION_TEMPLATES,
    STATEMENT_TEMPLATES,
    QProjEmbeddingCache,
    QProjLLMCache,
    QProjOnly,
    QProjV2f,
    learn_question_direction,
    project_away,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

EVAL_DATASETS = ("locomo_30q", "synthetic_19q")


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class CosineBaseline:
    """Pure cosine top-K baseline with the qproj caches for fair comparisons."""

    arch_name = "cosine_baseline"

    def __init__(self, store: SegmentStore):
        from best_shot import BestshotBase

        self._base = BestshotBase(store)
        self._base.embedding_cache = QProjEmbeddingCache()
        self._base.llm_cache = QProjLLMCache()
        self.store = store

    @property
    def embed_calls(self) -> int:
        return self._base.embed_calls

    @property
    def llm_calls(self) -> int:
        return self._base.llm_calls

    def reset_counters(self) -> None:
        self._base.reset_counters()

    def embed_text(self, text: str) -> np.ndarray:
        return self._base.embed_text(text)

    def save_caches(self) -> None:
        self._base.save_caches()

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        emb = self.embed_text(question)
        result = self.store.search(
            emb, top_k=50, conversation_id=conversation_id
        )
        return BestshotResult(
            segments=list(result.segments),
            metadata={"name": self.arch_name},
        )


# ---------------------------------------------------------------------------
# Per-question eval (adapted from inverse_query_eval)
# ---------------------------------------------------------------------------


def evaluate_question(arch, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])
    category = question.get("category", "unknown")

    arch.reset_counters()
    t0 = time.time()
    result = arch.retrieve(q_text, conv_id)
    elapsed = time.time() - t0

    seen: set[int] = set()
    arch_segments: list[Segment] = []
    for seg in result.segments:
        if seg.index not in seen:
            arch_segments.append(seg)
            seen.add(seg.index)

    # For fair-backfill, cosine side must be RAW (no projection) cosine top-K.
    # Use the underlying embed_text (which returns the raw query embedding).
    query_emb = arch.embed_text(q_text)
    max_K = max(BUDGETS)
    cosine_result = arch.store.search(
        query_emb, top_k=max_K, conversation_id=conv_id
    )
    cosine_segments = list(cosine_result.segments)

    row = {
        "conversation_id": conv_id,
        "category": category,
        "question_index": question.get("question_index", -1),
        "question": q_text,
        "source_chat_ids": sorted(source_ids),
        "num_source_turns": len(source_ids),
        "total_arch_retrieved": len(arch_segments),
        "embed_calls": arch.embed_calls,
        "llm_calls": arch.llm_calls,
        "time_s": round(elapsed, 2),
        "fair_backfill": {},
    }

    for K in BUDGETS:
        b_rec, a_rec, _ = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)

    return row


def run_one(
    arch_name: str,
    arch,
    dataset: str,
    questions: list[dict],
) -> tuple[list[dict], dict, dict]:
    print(f"\n{'=' * 70}")
    print(f"{arch_name} | {dataset} | {len(questions)} questions")
    print(f"{'=' * 70}")

    results = []
    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i+1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
            flush=True,
        )
        try:
            row = evaluate_question(arch, q)
            results.append(row)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()
        sys.stdout.flush()
        if (i + 1) % 5 == 0:
            arch.save_caches()

    arch.save_caches()
    summary = summarize(results, arch_name, dataset)
    by_cat = summarize_by_category(results)

    print(f"\n--- {arch_name} on {dataset} ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(
        f"  avg total_retrieved={summary['avg_total_retrieved']:.0f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"embed={summary['avg_embed_calls']:.1f}"
    )
    for cat, c in by_cat.items():
        print(
            f"    {cat:28s} (n={c['n']}): "
            f"r@20 d={c['delta_r@20']:+.3f} r@50 d={c['delta_r@50']:+.3f} "
            f"W/T/L@50={c['W/T/L_r@50']}"
        )

    return results, summary, by_cat


# ---------------------------------------------------------------------------
# Sanity check: how projection changes cosine on sample queries
# ---------------------------------------------------------------------------


def sanity_check_direction(
    direction: np.ndarray,
    sample_questions: list[str],
    sample_statements: list[str],
    client: OpenAI,
) -> dict:
    """Measure cosine between direction and held-out questions/statements."""
    cache = QProjEmbeddingCache()

    def _embed(t: str) -> np.ndarray:
        from associative_recall import EMBED_MODEL
        c = cache.get(t)
        if c is not None:
            return c
        r = client.embeddings.create(model=EMBED_MODEL, input=[t])
        e = np.array(r.data[0].embedding, dtype=np.float32)
        cache.put(t, e)
        return e

    d = direction / max(np.linalg.norm(direction), 1e-10)

    q_cosines = []
    for q in sample_questions:
        e = _embed(q)
        e_n = e / max(np.linalg.norm(e), 1e-10)
        q_cosines.append(float(np.dot(e_n, d)))

    s_cosines = []
    for s in sample_statements:
        e = _embed(s)
        e_n = e / max(np.linalg.norm(e), 1e-10)
        s_cosines.append(float(np.dot(e_n, d)))

    cache.save()

    return {
        "question_mean_cos_with_dir": float(np.mean(q_cosines)),
        "question_min_cos_with_dir": float(np.min(q_cosines)),
        "question_max_cos_with_dir": float(np.max(q_cosines)),
        "statement_mean_cos_with_dir": float(np.mean(s_cosines)),
        "statement_min_cos_with_dir": float(np.min(s_cosines)),
        "statement_max_cos_with_dir": float(np.max(s_cosines)),
        "separation": float(np.mean(q_cosines) - np.mean(s_cosines)),
        "n_question_samples": len(sample_questions),
        "n_statement_samples": len(sample_statements),
    }


def visualize_projection_effect(
    direction: np.ndarray,
    store: SegmentStore,
    sample_queries: list[tuple[str, str]],  # (question, conversation_id)
    alpha: float,
    client: OpenAI,
    top_k: int = 5,
) -> list[dict]:
    """For each sample (question, conv_id), show top-K turn_ids from raw cosine
    vs projected cosine — text-based 'visualization' of what projection does.
    """
    cache = QProjEmbeddingCache()

    def _embed(t: str) -> np.ndarray:
        from associative_recall import EMBED_MODEL
        c = cache.get(t)
        if c is not None:
            return c
        r = client.embeddings.create(model=EMBED_MODEL, input=[t])
        e = np.array(r.data[0].embedding, dtype=np.float32)
        cache.put(t, e)
        return e

    out = []
    for q_text, cid in sample_queries:
        raw = _embed(q_text)
        proj = project_away(raw, direction, alpha)
        raw_res = store.search(raw, top_k=top_k, conversation_id=cid)
        proj_res = store.search(proj, top_k=top_k, conversation_id=cid)
        raw_n = raw / max(np.linalg.norm(raw), 1e-10)
        cos_with_dir = float(np.dot(raw_n, direction))
        out.append(
            {
                "question": q_text,
                "conversation_id": cid,
                "cosine_with_q_direction": round(cos_with_dir, 4),
                "raw_top_turn_ids": [
                    {"turn_id": s.turn_id, "score": round(sc, 4),
                     "text_head": s.text[:80]}
                    for s, sc in zip(raw_res.segments, raw_res.scores)
                ],
                "proj_top_turn_ids": [
                    {"turn_id": s.turn_id, "score": round(sc, 4),
                     "text_head": s.text[:80]}
                    for s, sc in zip(proj_res.segments, proj_res.scores)
                ],
            }
        )
    cache.save()
    return out


# ---------------------------------------------------------------------------
# Top categories
# ---------------------------------------------------------------------------


def top_categories_delta(by_cat: dict, K: int = 50) -> tuple[list, list]:
    rows = []
    for cat, c in by_cat.items():
        rows.append((cat, c[f"delta_r@{K}"], c[f"W/T/L_r@{K}"], c["n"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    gaining = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows[:2]
        if d > 0
    ]
    losing = [
        {"category": cat, "delta": d, "W/T/L": wtl, "n": n}
        for cat, d, wtl, n in rows[-2:]
        if d < 0
    ]
    return gaining, losing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fmt_alpha(a: float) -> str:
    return f"{a:.1f}"


def qproj_arch_name(alpha: float, v2f: bool) -> str:
    tag = _fmt_alpha(alpha)
    return f"qproj_{tag}_v2f" if v2f else f"qproj_{tag}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--alphas", default="0.5,1.0,1.5",
        help="Comma-separated alpha values for projection strength."
    )
    p.add_argument(
        "--datasets", default=",".join(EVAL_DATASETS),
        help="Comma-separated dataset names."
    )
    p.add_argument(
        "--skip-v2f", action="store_true",
        help="Skip qproj_v2f run even if best qproj alpha beats cosine."
    )
    p.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip cosine_baseline and meta_v2f runs (use cached summaries)."
    )
    args = p.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    ds_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in ds_names:
        if d not in DATASETS:
            raise SystemExit(f"Unknown dataset: {d}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(timeout=60.0)

    # ---- Step 1: Learn the direction ----
    direction_path = RESULTS_DIR / "question_direction.npy"
    print(f"Learning question-ness direction "
          f"(n_q={len(QUESTION_TEMPLATES)}, n_s={len(STATEMENT_TEMPLATES)})")
    direction, dir_stats = learn_question_direction(
        client=client, save_path=direction_path
    )
    print(f"  saved direction -> {direction_path}")
    print(f"  stats: {dir_stats}")

    # ---- Sanity check ----
    # Use held-out samples (last few from each list as light sanity)
    heldout_q = QUESTION_TEMPLATES[-10:]
    heldout_s = STATEMENT_TEMPLATES[-10:]
    sanity = sanity_check_direction(
        direction, heldout_q, heldout_s, client
    )
    print(f"\nSanity check:")
    print(f"  question samples mean cos with q_dir = "
          f"{sanity['question_mean_cos_with_dir']:.4f}")
    print(f"  statement samples mean cos with q_dir = "
          f"{sanity['statement_mean_cos_with_dir']:.4f}")
    print(f"  separation = {sanity['separation']:.4f}")

    # ---- Step 2: Run all qproj_only variants + baselines ----
    all_results: dict[str, dict] = defaultdict(dict)

    for ds_name in ds_names:
        store, questions = load_dataset(ds_name)
        print(
            f"\nLoaded {ds_name}: {len(questions)} questions, "
            f"{len(store.segments)} segments"
        )

        # Baselines
        if not args.skip_baselines:
            for b_name, b_factory in [
                ("cosine_baseline", lambda s: CosineBaseline(s)),
                ("meta_v2f", lambda s: MetaV2fDedicated(s)),
            ]:
                arch = b_factory(store)
                results, summary, by_cat = run_one(
                    b_name, arch, ds_name, questions
                )
                all_results[b_name][ds_name] = {
                    "summary": summary,
                    "category_breakdown": by_cat,
                    "results": results,
                }

        # qproj_only for each alpha
        for alpha in alphas:
            name = qproj_arch_name(alpha, v2f=False)
            arch = QProjOnly(store, direction=direction, alpha=alpha)
            results, summary, by_cat = run_one(
                name, arch, ds_name, questions
            )
            all_results[name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            }

    # ---- Step 3: Decide best alpha vs cosine ----
    # Best alpha = one with highest combined delta@50 vs cosine_baseline on
    # locomo_30q + synthetic_19q.
    best_alpha = None
    best_score = -1e9
    cosine_ds_summaries = all_results.get("cosine_baseline", {})
    for alpha in alphas:
        name = qproj_arch_name(alpha, v2f=False)
        qds = all_results.get(name, {})
        score = 0.0
        wins = 0
        for ds in ds_names:
            if ds in qds and ds in cosine_ds_summaries:
                qs = qds[ds]["summary"]["arch_r@50"]
                cs = cosine_ds_summaries[ds]["summary"]["arch_r@50"]
                score += qs - cs
                if qs > cs + 0.001:
                    wins += 1
        if score > best_score:
            best_score = score
            best_alpha = alpha

    print(
        f"\nBest alpha (by sum delta_r@50 vs cosine baseline): "
        f"{best_alpha} (score={best_score:+.4f})"
    )

    # ---- Step 4: Run qproj_v2f at best alpha if it beat cosine ----
    if (
        not args.skip_v2f
        and best_alpha is not None
        and best_score > 0.0
    ):
        for ds_name in ds_names:
            store, questions = load_dataset(ds_name)
            name = qproj_arch_name(best_alpha, v2f=True)
            arch = QProjV2f(store, direction=direction, alpha=best_alpha)
            results, summary, by_cat = run_one(
                name, arch, ds_name, questions
            )
            all_results[name][ds_name] = {
                "summary": summary,
                "category_breakdown": by_cat,
                "results": results,
            }

    # ---- Step 5: Visualization / qualitative ----
    # Pull a few LoCoMo questions as sample_queries for visualization
    sample_queries: list[tuple[str, str]] = []
    try:
        _, locomo_qs = load_dataset("locomo_30q")
        store_locomo, _ = load_dataset("locomo_30q")
        for q in locomo_qs[:5]:
            sample_queries.append((q["question"], q["conversation_id"]))
        viz = visualize_projection_effect(
            direction,
            store_locomo,
            sample_queries,
            alpha=(best_alpha if best_alpha is not None else 1.0),
            client=client,
        )
    except Exception as e:
        print(f"Viz step failed (continuing): {e}")
        viz = []

    # ---- Step 6: Save outputs ----

    # JSON dump (exclude per-question results from the top-level for readability)
    raw: dict = {
        "alphas": alphas,
        "datasets": ds_names,
        "direction_stats": dir_stats,
        "direction_sanity": sanity,
        "best_alpha_by_sum_delta_r50_vs_cosine": best_alpha,
        "best_score_sum_delta_r50": round(best_score, 4),
        "summaries": {
            a: {
                d: {
                    "summary": all_results[a][d]["summary"],
                    "category_breakdown": all_results[a][d]["category_breakdown"],
                }
                for d in all_results[a]
            }
            for a in all_results
        },
        "projection_viz_samples": viz,
    }

    raw_path = RESULTS_DIR / "query_projection.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"\nSaved: {raw_path}")

    # Per-arch per-dataset full results
    for a in all_results:
        for d in all_results[a]:
            out_path = RESULTS_DIR / f"qproj_{a}_{d}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "arch": a,
                        "dataset": d,
                        "summary": all_results[a][d]["summary"],
                        "category_breakdown": all_results[a][d][
                            "category_breakdown"
                        ],
                        "results": all_results[a][d]["results"],
                    },
                    f,
                    indent=2,
                    default=str,
                )

    # Markdown report
    md: list[str] = []
    md.append("# Query-Direction Embedding Projection\n")
    md.append(
        "Zero-LLM architectural variant. Learn a unit direction separating "
        "question-form from statement-form prose, then project queries away "
        "from that direction before cosine retrieval.\n"
    )
    md.append(
        f"- `q_dir = normalize(mean(question_embs) - mean(statement_embs))`\n"
        f"- `q_emb' = normalize(q_emb - alpha * (q_emb . q_dir) * q_dir)`\n"
    )
    md.append(
        f"**Direction stats**: "
        f"n_q={dir_stats['n_question_templates']}, "
        f"n_s={dir_stats['n_statement_templates']}, "
        f"diff_norm={dir_stats['diff_norm']:.4f}, "
        f"q_mean_norm={dir_stats['q_mean_norm']:.4f}, "
        f"s_mean_norm={dir_stats['s_mean_norm']:.4f}.\n"
    )

    md.append("## Sanity check — held-out samples cosine with q_dir\n")
    md.append(
        f"- Question samples (n={sanity['n_question_samples']}): "
        f"mean={sanity['question_mean_cos_with_dir']:+.4f} "
        f"(min={sanity['question_min_cos_with_dir']:+.4f}, "
        f"max={sanity['question_max_cos_with_dir']:+.4f})\n"
        f"- Statement samples (n={sanity['n_statement_samples']}): "
        f"mean={sanity['statement_mean_cos_with_dir']:+.4f} "
        f"(min={sanity['statement_min_cos_with_dir']:+.4f}, "
        f"max={sanity['statement_max_cos_with_dir']:+.4f})\n"
        f"- Separation (Q-S): {sanity['separation']:+.4f}\n"
    )

    md.append("\n## Fair-backfill recall\n")
    md.append(
        "| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    order = []
    if "cosine_baseline" in all_results:
        order.append("cosine_baseline")
    for alpha in alphas:
        n = qproj_arch_name(alpha, v2f=False)
        if n in all_results:
            order.append(n)
    if "meta_v2f" in all_results:
        order.append("meta_v2f")
    for alpha in alphas:
        n = qproj_arch_name(alpha, v2f=True)
        if n in all_results:
            order.append(n)
    for a in order:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            md.append(
                f"| {a} | {d} | "
                f"{s['baseline_r@20']:.3f} | {s['arch_r@20']:.3f} | "
                f"{s['delta_r@20']:+.3f} | "
                f"{s['baseline_r@50']:.3f} | {s['arch_r@50']:.3f} | "
                f"{s['delta_r@50']:+.3f} | "
                f"{s['avg_llm_calls']:.1f} |"
            )
    md.append(
        "\nNote: `base@K` in all rows is raw cosine top-K (same reference).\n"
    )

    # Best alpha
    md.append("\n## Best alpha (sum of delta_r@50 vs cosine across datasets)\n")
    md.append(f"- **Best alpha**: {best_alpha}\n")
    md.append(f"- **Score**: {best_score:+.4f}\n")

    # Top categories for the best qproj_only on locomo
    if best_alpha is not None:
        best_name = qproj_arch_name(best_alpha, v2f=False)
        if (
            best_name in all_results
            and "locomo_30q" in all_results[best_name]
        ):
            g, l = top_categories_delta(
                all_results[best_name]["locomo_30q"]["category_breakdown"],
                K=50,
            )
            md.append(
                f"\n## Top categories by Δr@50 for {best_name} on LoCoMo-30\n"
            )
            md.append("Gaining:")
            for x in g:
                md.append(
                    f"  - {x['category']} (n={x['n']}): Δ={x['delta']:+.3f} "
                    f"W/T/L={x['W/T/L']}"
                )
            md.append("Losing:")
            for x in l:
                md.append(
                    f"  - {x['category']} (n={x['n']}): Δ={x['delta']:+.3f} "
                    f"W/T/L={x['W/T/L']}"
                )

    # Projection effect visualization
    if viz:
        md.append("\n## Projection-effect visualization\n")
        md.append(
            "For each sample query: cosine-with-direction, raw top-5 vs "
            "projected top-5 retrieved turn_ids and cosine scores.\n"
        )
        for v in viz:
            md.append(
                f"\n### Q: {v['question']}\n"
                f"- cos(q_emb, q_dir) = {v['cosine_with_q_direction']:+.4f}\n"
                f"\n**Raw top-5:**\n"
            )
            for r in v["raw_top_turn_ids"]:
                md.append(
                    f"  - tid={r['turn_id']} score={r['score']:+.3f} :: "
                    f"{r['text_head']!r}"
                )
            md.append("\n**Projected top-5:**\n")
            for r in v["proj_top_turn_ids"]:
                md.append(
                    f"  - tid={r['turn_id']} score={r['score']:+.3f} :: "
                    f"{r['text_head']!r}"
                )

    # Verdict
    md.append("\n## Verdict\n")
    verdict = "(see numbers above)"
    cos_ok = "cosine_baseline" in all_results
    v2f_ok = "meta_v2f" in all_results
    qproj_v2f_name = (
        qproj_arch_name(best_alpha, v2f=True) if best_alpha is not None else None
    )
    qproj_v2f_ok = (
        qproj_v2f_name is not None and qproj_v2f_name in all_results
    )

    # Aggregate deltas on locomo_30q (primary) for verdict
    if cos_ok and "locomo_30q" in all_results["cosine_baseline"]:
        cos_loc50 = all_results["cosine_baseline"]["locomo_30q"][
            "summary"
        ]["arch_r@50"]
        best_qproj_loc50 = None
        if best_alpha is not None:
            bn = qproj_arch_name(best_alpha, v2f=False)
            if bn in all_results and "locomo_30q" in all_results[bn]:
                best_qproj_loc50 = all_results[bn]["locomo_30q"]["summary"][
                    "arch_r@50"
                ]

        v2f_loc50 = (
            all_results["meta_v2f"]["locomo_30q"]["summary"]["arch_r@50"]
            if v2f_ok and "locomo_30q" in all_results["meta_v2f"] else None
        )
        qproj_v2f_loc50 = (
            all_results[qproj_v2f_name]["locomo_30q"]["summary"]["arch_r@50"]
            if qproj_v2f_ok and "locomo_30q" in all_results[qproj_v2f_name]
            else None
        )

        lines = [
            f"LoCoMo-30 @K=50:",
            f"  cosine        = {cos_loc50:.3f}",
        ]
        if best_qproj_loc50 is not None:
            lines.append(
                f"  qproj(α={best_alpha}) = {best_qproj_loc50:.3f} "
                f"(Δ vs cosine = {best_qproj_loc50 - cos_loc50:+.3f})"
            )
        if v2f_loc50 is not None:
            lines.append(f"  meta_v2f      = {v2f_loc50:.3f}")
        if qproj_v2f_loc50 is not None and v2f_loc50 is not None:
            lines.append(
                f"  qproj_v2f(α={best_alpha}) = {qproj_v2f_loc50:.3f} "
                f"(Δ vs v2f = {qproj_v2f_loc50 - v2f_loc50:+.3f})"
            )

        md.append("\n".join(lines) + "\n")

        # Decision
        if (
            best_qproj_loc50 is not None
            and best_qproj_loc50 > cos_loc50 + 0.005
        ):
            if (
                qproj_v2f_loc50 is not None
                and v2f_loc50 is not None
                and qproj_v2f_loc50 > v2f_loc50 + 0.005
            ):
                verdict = (
                    f"**SHIP**: qproj_v2f(α={best_alpha}) beats meta_v2f on "
                    f"LoCoMo @K=50 "
                    f"({qproj_v2f_loc50:.3f} vs {v2f_loc50:.3f})."
                )
            else:
                verdict = (
                    f"**BORDERLINE**: qproj_only(α={best_alpha}) beats cosine "
                    f"baseline, but qproj_v2f does not beat meta_v2f."
                )
        else:
            verdict = (
                f"**ABANDON**: no qproj_only alpha beats cosine baseline on "
                f"LoCoMo @K=50."
            )
    md.append("\n" + verdict + "\n")

    md_path = RESULTS_DIR / "query_projection.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {md_path}")

    # Final table summary
    print("\n" + "=" * 100)
    print("QUERY PROJECTION SUMMARY")
    print("=" * 100)
    for a in order:
        for d in ds_names:
            if d not in all_results.get(a, {}):
                continue
            s = all_results[a][d]["summary"]
            print(
                f"{a:22s} {d:14s} "
                f"b@20={s['baseline_r@20']:.3f} a@20={s['arch_r@20']:.3f} "
                f"d@20={s['delta_r@20']:+.3f}  "
                f"b@50={s['baseline_r@50']:.3f} a@50={s['arch_r@50']:.3f} "
                f"d@50={s['delta_r@50']:+.3f}  "
                f"llm={s['avg_llm_calls']:.1f}"
            )


if __name__ == "__main__":
    main()
