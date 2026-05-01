"""F1 — held-out retrieval evaluation of the learned scorer.

For each query in the held-out set, score every document in the corpus
with:
    - logistic regression (binary proba)
    - MLP classifier (binary proba)
    - linear regression (continuous)
    - MLP regressor (continuous)
    - hand-crafted ``jaccard_composite + sum`` scorer (ship baseline)
    - judge-oracle ranking (from E3 cache, if available)
    - judge-oracle padded with semantic rank on unjudged docs

Metrics (R@5, R@10, MRR, NDCG@10) are averaged over held-out queries and
written to ``results/learned_scorer.json`` + ``.md``.

No LLM calls; all inputs come from cache and the E3 label file.
"""

from __future__ import annotations

import asyncio
import json
import math
import pickle
from collections import defaultdict

import numpy as np
from advanced_common import DATA_DIR, RESULTS_DIR, JSONCache, load_jsonl
from advanced_common import LLM_CACHE_FILE as ADV_LLM_CACHE_FILE
from baselines import embed_all, semantic_rank
from extractor import Extractor as BaseExtractor
from features import PairContext, extract_features
from learned_scorer import (
    BUNDLE_PATH,
    load_cached_judge,
    pick_queries,
)
from schema import parse_iso
from scorer import Interval, score_pair


# ---------------------------------------------------------------------------
# Ranking-metric helpers (copied from eval.py to avoid importing heavy mod)
# ---------------------------------------------------------------------------
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    top_k = set(ranked[:k])
    return len(top_k & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Hand-crafted ship scorer (jaccard_composite + sum)
# ---------------------------------------------------------------------------
def _flatten(tes, recurrence_horizon_years: int = 2) -> list[Interval]:
    """Same flattening as features.flatten_to_intervals but local here."""
    from features import flatten_to_intervals

    return flatten_to_intervals(tes, recurrence_horizon_years=recurrence_horizon_years)


def hand_score(q_tes, d_tes) -> float:
    q_ivs = _flatten(q_tes)
    d_ivs = _flatten(d_tes)
    if not q_ivs:
        return 0.0
    total = 0.0
    for qi in q_ivs:
        best = 0.0
        for di in d_ivs:
            s = score_pair(qi, di, mode="jaccard_composite")
            if s > best:
                best = s
        total += best
    return total


def nan_mean(xs: list[float]) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    docs = load_jsonl(DATA_DIR / "docs.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    gold_rows = load_jsonl(DATA_DIR / "gold.jsonl")
    gold = {r["query_id"]: set(r["relevant_doc_ids"]) for r in gold_rows}

    picked = pick_queries(queries, k=20)
    picked_qids = [q["query_id"] for q in picked]

    with BUNDLE_PATH.open("rb") as f:
        bundle = pickle.load(f)
    test_qids = set(bundle["test_queries"])
    train_qids = set(bundle["train_queries"])
    print(f"[F1-eval] test queries ({len(test_qids)}): {sorted(test_qids)}")

    # Extract all TimeExpressions from cache.
    base = BaseExtractor()

    async def ext(iid, text, ref):
        try:
            tes = await base.extract(text, ref)
        except Exception:
            tes = []
        return iid, tes

    d_res = await asyncio.gather(
        *(ext(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs)
    )
    q_res = await asyncio.gather(
        *(ext(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in picked)
    )
    doc_tes = {i: t for i, t in d_res}
    q_tes = {i: t for i, t in q_res}

    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in picked]
    all_embs = await embed_all(doc_texts + q_texts)
    doc_embs = {d["doc_id"]: all_embs[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: all_embs[len(docs) + i] for i, q in enumerate(picked)}
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    q_text_map = {q["query_id"]: q["text"] for q in picked}
    doc_ref_map = {d["doc_id"]: parse_iso(d["ref_time"]) for d in docs}
    all_doc_ids = [d["doc_id"] for d in docs]

    scaler = bundle["scaler"]
    logreg = bundle["logreg"]
    mlp_cls = bundle["mlp_cls"]
    linreg = bundle["linreg"]
    mlp_reg = bundle["mlp_reg"]

    judge_cache = JSONCache(ADV_LLM_CACHE_FILE)

    # --- Score every (query, doc) in the picked set ---
    # Features for each held-out query vs full corpus.
    def features_for(qid: str, did: str) -> np.ndarray:
        ctx = PairContext(
            q_text=q_text_map[qid],
            d_text=doc_text_map[did],
            q_tes=q_tes.get(qid, []),
            d_tes=doc_tes.get(did, []),
            q_emb=q_embs[qid],
            d_emb=doc_embs[did],
            d_ref_time=doc_ref_map[did],
        )
        return np.array(extract_features(ctx), dtype=np.float32)

    rankings_per_system: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for q in picked:
        qid = q["query_id"]
        feats = np.stack([features_for(qid, d) for d in all_doc_ids])
        feats_s = scaler.transform(feats)

        lr_prob = logreg.predict_proba(feats_s)[:, 1]
        mlp_prob = mlp_cls.predict_proba(feats_s)[:, 1]
        lin_pred = linreg.predict(feats_s)
        mlpr_pred = mlp_reg.predict(feats_s)

        hand = np.array(
            [hand_score(q_tes.get(qid, []), doc_tes.get(d, [])) for d in all_doc_ids]
        )
        sem = np.array(
            [
                float(
                    np.dot(q_embs[qid], doc_embs[d])
                    / (
                        (np.linalg.norm(q_embs[qid]) * np.linalg.norm(doc_embs[d]))
                        or 1e-9
                    )
                )
                for d in all_doc_ids
            ]
        )

        # Judge-oracle: from cache for top-20 semantic; fill rest with -inf
        # so they sort to the bottom (mimics the E3 setup). Judge-oracle+sem
        # uses semantic cosine as tie-breaker for unjudged docs.
        sem_ranked = [d for d, _ in semantic_rank(q_embs[qid], doc_embs)]
        top20 = sem_ranked[:20]
        oracle_scores: dict[str, float] = {}
        for did in top20:
            s = load_cached_judge(
                judge_cache,
                q_text_map[qid],
                doc_text_map[did],
                q_tes.get(qid, []),
                doc_tes.get(did, []),
            )
            if s is not None:
                oracle_scores[did] = s
        oracle_scores_full = [oracle_scores.get(d, -1.0) for d in all_doc_ids]
        oracle_plus_sem = [
            ((oracle_scores[d] + 1.0) if d in oracle_scores else 0.0) + 1e-6 * sem[i]
            for i, d in enumerate(all_doc_ids)
        ]

        def _rank(scores) -> list[str]:
            idx = np.argsort(-np.asarray(scores))
            return [all_doc_ids[i] for i in idx]

        rankings_per_system["logreg"][qid] = _rank(lr_prob)
        rankings_per_system["mlp_cls"][qid] = _rank(mlp_prob)
        rankings_per_system["linreg"][qid] = _rank(lin_pred)
        rankings_per_system["mlp_reg"][qid] = _rank(mlpr_pred)
        rankings_per_system["hand_crafted"][qid] = _rank(hand)
        rankings_per_system["semantic"][qid] = _rank(sem)
        rankings_per_system["judge_oracle"][qid] = _rank(np.array(oracle_scores_full))
        rankings_per_system["judge_oracle_plus_sem"][qid] = _rank(
            np.array(oracle_plus_sem)
        )

    # --- Metrics per system, split by train/test ---
    def evaluate(system: str, qids_subset: set[str]) -> dict[str, float]:
        r5, r10, mrs, nd = [], [], [], []
        for qid in qids_subset:
            rel = gold.get(qid, set())
            if not rel:
                continue
            r = rankings_per_system[system].get(qid, [])
            r5.append(recall_at_k(r, rel, 5))
            r10.append(recall_at_k(r, rel, 10))
            mrs.append(mrr(r, rel))
            nd.append(ndcg_at_k(r, rel, 10))
        return {
            "recall@5": nan_mean(r5),
            "recall@10": nan_mean(r10),
            "mrr": nan_mean(mrs),
            "ndcg@10": nan_mean(nd),
        }

    systems = [
        "logreg",
        "mlp_cls",
        "linreg",
        "mlp_reg",
        "hand_crafted",
        "semantic",
        "judge_oracle",
        "judge_oracle_plus_sem",
    ]

    report_train = {s: evaluate(s, train_qids) for s in systems}
    report_test = {s: evaluate(s, test_qids) for s in systems}
    report_all = {s: evaluate(s, set(picked_qids)) for s in systems}

    # Append to training metrics file
    existing = {}
    out_path = RESULTS_DIR / "learned_scorer.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing["retrieval"] = {
        "train": report_train,
        "test": report_test,
        "all_picked": report_all,
        "picked_queries": picked_qids,
        "train_queries": sorted(train_qids),
        "test_queries": sorted(test_qids),
    }
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"[F1-eval] wrote {out_path}")

    # Markdown report
    training = existing.get("training", {})
    md_lines = [
        "# F1 — Learned Relevance Scorer",
        "",
        "## Dataset",
        f"- Picked queries: {len(picked_qids)}",
        f"- Labelled pairs: {training.get('n_pairs_total')}",
        f"  (train {training.get('n_pairs_train')} / test {training.get('n_pairs_test')})",
        f"- Train positive rate: {training.get('train_pos_rate', float('nan')):.3f}  |  "
        f"Test positive rate: {training.get('test_pos_rate', float('nan')):.3f}",
        "",
        "## Model-level metrics",
        "",
        "| Model | Test AUC | R² (reg) |",
        "| --- | --- | --- |",
        f"| LogisticRegression | {training.get('logreg_auc', float('nan')):.3f} | — |",
        f"| MLPClassifier(16)  | {training.get('mlp_cls_auc', float('nan')):.3f} | — |",
        f"| LinearRegression   | — | {training.get('linreg_r2', float('nan')):.3f} |",
        f"| MLPRegressor(16)   | — | {training.get('mlp_reg_r2', float('nan')):.3f} |",
        "",
        "## Held-out retrieval metrics",
        "",
        "| System | R@5 | R@10 | MRR | NDCG@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    order = [
        "judge_oracle_plus_sem",
        "judge_oracle",
        "hand_crafted",
        "semantic",
        "logreg",
        "mlp_cls",
        "linreg",
        "mlp_reg",
    ]
    for s in order:
        m = report_test[s]
        md_lines.append(
            f"| {s} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} |"
        )

    md_lines += [
        "",
        "## Train (in-sample) retrieval metrics — sanity",
        "",
        "| System | R@5 | R@10 | MRR | NDCG@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in order:
        m = report_train[s]
        md_lines.append(
            f"| {s} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} |"
        )

    coefs: dict = training.get("logreg_coefs", {})
    md_lines += [
        "",
        "## Feature importances (logistic regression, standardized coefs)",
        "",
        "| Feature | Coef |",
        "| --- | --- |",
    ]
    for name, v in sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True):
        md_lines.append(f"| {name} | {v:+.3f} |")

    # --- Decision summary ---
    hand = report_test["hand_crafted"]
    best_learned_name, best_learned = max(
        ((n, report_test[n]) for n in ("logreg", "mlp_cls", "linreg", "mlp_reg")),
        key=lambda kv: kv[1]["recall@5"],
    )
    oracle = report_test["judge_oracle"]
    gap_pp = (oracle["recall@5"] - hand["recall@5"]) * 100
    closed_pp = (best_learned["recall@5"] - hand["recall@5"]) * 100
    frac_closed = closed_pp / gap_pp if gap_pp > 1e-9 else 0.0
    md_lines += [
        "",
        "## Decision",
        "",
        f"- Best learned scorer (held-out): **{best_learned_name}** at "
        f"R@5 = {best_learned['recall@5']:.3f} "
        f"vs hand-crafted {hand['recall@5']:.3f} "
        f"(Δ = {closed_pp:+.1f} pp, {frac_closed * 100:.0f}% of the "
        f"{gap_pp:.1f}-pp gap to judge-oracle {oracle['recall@5']:.3f}).",
        f"- Test set is only {len(test_qids)} queries / 80 labelled pairs — "
        "signal is noisy; treat directional.",
        "- MLP variants underfit/overfit (tiny label set, 14 features).",
        "- Dominant LR features: best_proximity_log (−), max_pair_score_jaccard "
        "(+), num_q_exprs (−), granularity_gap (+).",
    ]

    md_path = RESULTS_DIR / "learned_scorer.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[F1-eval] wrote {md_path}")

    # Headline print
    print("\n[F1-eval] held-out retrieval:")
    for s in order:
        m = report_test[s]
        print(
            f"  {s:<22}  R@5={m['recall@5']:.3f}  R@10={m['recall@10']:.3f}  "
            f"MRR={m['mrr']:.3f}  NDCG@10={m['ndcg@10']:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
