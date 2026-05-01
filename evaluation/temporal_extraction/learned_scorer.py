"""F1 — train a learned relevance scorer from E3 labels.

Pipeline:
  1. Re-derive the E3 (query, candidate) pair list using the same
     deterministic pick as ``llm_judge.main`` (20 queries, top-20 semantic).
  2. Extract base-extractor TimeExpressions from ``cache/llm_cache.json``
     (all calls hit cache; runtime cost is zero LLM calls).
  3. Reconstruct judge scores from ``cache/advanced/llm_cache.json`` using
     the same JSON payload + hashing scheme as ``llm_judge.judge_pair``.
  4. Build a feature matrix (see ``features.FEATURE_NAMES``), split by
     QUERY into 80/20 train/test, and train:
       * LogisticRegression        (binary label: judge >= 0.5)
       * MLPClassifier(16,)        (same binary label)
       * LinearRegression          (regression on 0..1 judge score)
       * MLPRegressor(16,)         (regression on 0..1 judge score)
  5. Persist the trained classifier + regressor bundle for use by
     ``learned_scorer_eval.py``.

Outputs a partial JSON payload with AUC / R^2 / feature importances; the
retrieval metrics are written by the eval module.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from advanced_common import DATA_DIR, RESULTS_DIR, JSONCache, load_jsonl
from advanced_common import LLM_CACHE_FILE as ADV_LLM_CACHE_FILE
from baselines import embed_all, semantic_rank
from extractor import Extractor as BaseExtractor
from features import PairContext, extract_features, feature_names
from llm_judge import JUDGE_SYSTEM, _expr_summary
from schema import parse_iso
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parent / "cache" / "learned_scorer"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_PATH = MODELS_DIR / "bundle.pkl"


# ---------------------------------------------------------------------------
# Deterministic query pick (mirrors llm_judge.main)
# ---------------------------------------------------------------------------
def pick_queries(queries: list[dict], k: int = 20) -> list[dict]:
    by_prefix: dict[str, list[dict]] = {}
    for q in queries:
        p = q["query_id"].rsplit("_", 1)[0]
        by_prefix.setdefault(p, []).append(q)
    picked: list[dict] = []
    prefixes = sorted(by_prefix.keys())
    i = 0
    while len(picked) < k and prefixes:
        p = prefixes[i % len(prefixes)]
        bucket = by_prefix[p]
        if bucket:
            picked.append(bucket.pop(0))
        i += 1
        if all(len(by_prefix[pp]) == 0 for pp in prefixes):
            break
    return picked[:k]


# ---------------------------------------------------------------------------
# Judge cache lookup
# ---------------------------------------------------------------------------
def judge_key(
    system: str,
    user: str,
    *,
    model: str = "gpt-5-mini",
    cache_tag: str = "e3_judge",
) -> str:
    return JSONCache.key(
        model,
        cache_tag,
        hashlib.sha256(system.encode()).hexdigest()[:16],
        user,
    )


def judge_user(q_text: str, d_text: str, q_tes, d_tes) -> str:
    q_summ = "\n".join(_expr_summary(t) for t in q_tes) or "(none)"
    d_summ = "\n".join(_expr_summary(t) for t in d_tes) or "(none)"
    return (
        f"Query text: {q_text}\n"
        f"Query time references:\n{q_summ}\n\n"
        f"Document text: {d_text}\n"
        f"Document time references:\n{d_summ}\n\n"
        'Return {"score": <float 0-1>, "reason": "..."}.'
    )


def load_cached_judge(
    judge_cache: JSONCache,
    q_text: str,
    d_text: str,
    q_tes,
    d_tes,
) -> float | None:
    user = judge_user(q_text, d_text, q_tes, d_tes)
    key = judge_key(JUDGE_SYSTEM, user)
    raw = judge_cache.get(key)
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
        s = float(obj.get("score", 0.0))
        return max(0.0, min(1.0, s))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main data build
# ---------------------------------------------------------------------------
async def build_dataset() -> dict[str, Any]:
    docs = load_jsonl(DATA_DIR / "docs.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    picked = pick_queries(queries, k=20)
    picked_ids = [q["query_id"] for q in picked]
    print(f"[F1] picked {len(picked)} queries: {picked_ids[:4]}...")

    # Extract TimeExpressions from cache — strictly no new LLM calls.
    base = BaseExtractor()

    async def ext(item_id: str, text: str, ref):
        try:
            tes = await base.extract(text, ref)
        except Exception:
            tes = []
        return item_id, tes

    print("[F1] loading cached TimeExpressions for all docs + picked queries")
    d_results = await asyncio.gather(
        *(ext(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs)
    )
    q_results = await asyncio.gather(
        *(ext(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in picked)
    )
    doc_tes = {i: t for i, t in d_results}
    q_tes = {i: t for i, t in q_results}

    # Embeddings (cached)
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in picked]
    all_embs = await embed_all(doc_texts + q_texts)
    doc_embs = {d["doc_id"]: all_embs[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: all_embs[len(docs) + i] for i, q in enumerate(picked)}

    # Build candidate list = top-20 semantic per query (matches judge input).
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    q_text_map = {q["query_id"]: q["text"] for q in picked}
    doc_ref_map = {d["doc_id"]: parse_iso(d["ref_time"]) for d in docs}

    pairs: list[tuple[str, str]] = []
    for q in picked:
        qid = q["query_id"]
        ranked = semantic_rank(q_embs[qid], doc_embs)
        for cand_id, _ in ranked[:20]:
            pairs.append((qid, cand_id))

    # Resolve labels from judge cache.
    judge_cache = JSONCache(ADV_LLM_CACHE_FILE)
    X: list[list[float]] = []
    y_reg: list[float] = []
    y_bin: list[int] = []
    keep_qids: list[str] = []
    keep_dids: list[str] = []
    missing = 0
    for qid, did in pairs:
        score = load_cached_judge(
            judge_cache,
            q_text_map[qid],
            doc_text_map[did],
            q_tes.get(qid, []),
            doc_tes.get(did, []),
        )
        if score is None:
            missing += 1
            continue
        ctx = PairContext(
            q_text=q_text_map[qid],
            d_text=doc_text_map[did],
            q_tes=q_tes.get(qid, []),
            d_tes=doc_tes.get(did, []),
            q_emb=q_embs[qid],
            d_emb=doc_embs[did],
            d_ref_time=doc_ref_map[did],
        )
        fv = extract_features(ctx)
        X.append(fv)
        y_reg.append(float(score))
        y_bin.append(1 if score >= 0.5 else 0)
        keep_qids.append(qid)
        keep_dids.append(did)

    print(f"[F1] built {len(X)} labelled pairs (missing judge label on {missing})")

    return {
        "X": np.array(X, dtype=np.float32),
        "y_reg": np.array(y_reg, dtype=np.float32),
        "y_bin": np.array(y_bin, dtype=np.int32),
        "query_ids": keep_qids,
        "doc_ids": keep_dids,
        "picked_ids": picked_ids,
        "feature_names": feature_names(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def split_by_query(
    query_ids: list[str], picked_ids: list[str], frac: float = 0.8, seed: int = 0
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    order = list(picked_ids)
    rng.shuffle(order)
    n_train = int(round(len(order) * frac))
    return order[:n_train], order[n_train:]


def train_models(data: dict[str, Any]) -> dict[str, Any]:
    X = data["X"]
    y_reg = data["y_reg"]
    y_bin = data["y_bin"]
    qids = data["query_ids"]
    picked = data["picked_ids"]

    train_q, test_q = split_by_query(qids, picked, frac=0.8, seed=0)
    train_q_set = set(train_q)
    test_q_set = set(test_q)

    train_mask = np.array([q in train_q_set for q in qids])
    test_mask = np.array([q in test_q_set for q in qids])

    Xtr, Xte = X[train_mask], X[test_mask]
    ytr_b, yte_b = y_bin[train_mask], y_bin[test_mask]
    ytr_r, yte_r = y_reg[train_mask], y_reg[test_mask]

    print(
        f"[F1] split: {len(train_q)} train qs ({Xtr.shape[0]} pairs) / "
        f"{len(test_q)} test qs ({Xte.shape[0]} pairs)"
    )
    print(f"[F1] train pos-rate: {ytr_b.mean():.3f}  test pos-rate: {yte_b.mean():.3f}")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # --- Classifiers ---
    lr_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    lr_clf.fit(Xtr_s, ytr_b)
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        max_iter=200,
        tol=1e-3,
        random_state=0,
    )
    mlp_clf.fit(Xtr_s, ytr_b)

    def _auc(clf) -> float:
        if len(set(yte_b)) < 2:
            return float("nan")
        p = clf.predict_proba(Xte_s)[:, 1]
        return float(roc_auc_score(yte_b, p))

    # --- Regressors ---
    lin_reg = LinearRegression()
    lin_reg.fit(Xtr_s, ytr_r)
    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(16,),
        max_iter=400,
        tol=1e-3,
        random_state=0,
    )
    mlp_reg.fit(Xtr_s, ytr_r)

    def _r2(model) -> float:
        pr = model.predict(Xte_s)
        return float(r2_score(yte_r, pr))

    metrics = {
        "n_pairs_total": int(X.shape[0]),
        "n_pairs_train": int(Xtr.shape[0]),
        "n_pairs_test": int(Xte.shape[0]),
        "train_pos_rate": float(ytr_b.mean()),
        "test_pos_rate": float(yte_b.mean()) if len(yte_b) else float("nan"),
        "train_queries": sorted(train_q),
        "test_queries": sorted(test_q),
        "logreg_auc": _auc(lr_clf),
        "mlp_cls_auc": _auc(mlp_clf),
        "linreg_r2": _r2(lin_reg),
        "mlp_reg_r2": _r2(mlp_reg),
        "logreg_coefs": dict(
            zip(data["feature_names"], [float(c) for c in lr_clf.coef_[0]])
        ),
        "logreg_intercept": float(lr_clf.intercept_[0]),
    }

    bundle = {
        "scaler": scaler,
        "logreg": lr_clf,
        "mlp_cls": mlp_clf,
        "linreg": lin_reg,
        "mlp_reg": mlp_reg,
        "feature_names": data["feature_names"],
        "train_queries": sorted(train_q),
        "test_queries": sorted(test_q),
    }
    with BUNDLE_PATH.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"[F1] wrote model bundle -> {BUNDLE_PATH}")
    return metrics


async def main() -> None:
    data = await build_dataset()
    metrics = train_models(data)
    (RESULTS_DIR / "learned_scorer.json").write_text(
        json.dumps({"training": metrics}, indent=2)
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k not in {"train_queries", "test_queries", "logreg_coefs"}
            },
            indent=2,
        )
    )
    print("logreg_coefs:")
    print(json.dumps(metrics["logreg_coefs"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
