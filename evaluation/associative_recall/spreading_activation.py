"""Spreading activation retrieval — Phase A geometry + Phase B retrieval.

Phase A: Geometry test. For each LoCoMo-30 multi-gold question compute:
  inter_gold       = mean pairwise cosine(gold_i, gold_j) for i!=j
  gold_to_query    = mean cosine(gold_i, query)
  random_to_query  = mean cosine(random_same_conv_turn, query) (baseline)
Plus: for each missed turn in results/error_analysis_details.json, the max
cosine(missed_turn, retrieved_gold) to see if misses are near retrieved gold.

Phase B (if geometry justifies): kNN-graph spreading activation retrieval on
LoCoMo. Evaluated with fair-backfill methodology at K=20,50 alongside cosine
and v2f.

Usage:
    uv run python spreading_activation.py [--phase a|b|all]
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
    EMBED_MODEL,
    EmbeddingCache,
    Segment,
    SegmentStore,
)
from dotenv import load_dotenv
from fair_backfill_eval import (
    BUDGETS,
    DATASETS,
    compute_recall,
    fair_backfill_evaluate,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"


# =============================================================================
# Helpers: embedding access shared across phases
# =============================================================================


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / max(n, 1e-10)


def _embed_question(
    text: str,
    cache: EmbeddingCache,
    client,
) -> np.ndarray:
    cached = cache.get(text)
    if cached is not None:
        return _l2_normalize(cached.astype(np.float32))
    if client is None:
        raise RuntimeError(
            f"Question embedding missing from cache and no client provided: "
            f"{text[:60]!r}"
        )
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    cache.put(text, emb)
    return _l2_normalize(emb)


def _segments_by_turn_id(
    store: SegmentStore, conversation_id: str
) -> dict[int, Segment]:
    return {
        seg.turn_id: seg
        for seg in store.segments
        if seg.conversation_id == conversation_id
    }


def _turn_emb(
    store: SegmentStore, conversation_id: str, turn_id: int
) -> np.ndarray | None:
    """Return normalized embedding for a turn (or None if not present)."""
    idx_map = store._turn_index.get(conversation_id, {})
    idx = idx_map.get(turn_id)
    if idx is None:
        return None
    return store.normalized_embeddings[idx].astype(np.float32)


# =============================================================================
# PHASE A — GEOMETRY
# =============================================================================


def run_phase_a() -> dict:
    """Compute the geometry statistics for LoCoMo-30 multi-gold questions.

    Also compute stats over all LoCoMo multi-gold questions (n=85) for
    robustness; decision uses LoCoMo-30.
    Also compute missed-to-retrieved-gold stats using error_analysis_details.
    """
    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    with open(DATA_DIR / "questions_extended.json") as f:
        all_questions = json.load(f)

    locomo = [q for q in all_questions if q.get("benchmark") == "locomo"]
    locomo30 = locomo[:30]

    emb_cache = EmbeddingCache()

    # Lazy client: only constructed if a question is missing from cache.
    client = None

    def analyze_questions(qs: list[dict], label: str) -> dict:
        rng = random.Random(1234)
        rows = []
        skipped_missing_emb = 0

        nonlocal client
        for q in qs:
            gold_ids = list(q.get("source_chat_ids", []))
            if len(gold_ids) < 2:
                continue
            conv_id = q["conversation_id"]

            # Gather gold embeddings
            gold_embs = []
            for tid in gold_ids:
                emb = _turn_emb(store, conv_id, tid)
                if emb is not None:
                    gold_embs.append(emb)
            if len(gold_embs) < 2:
                continue
            gold_mat = np.stack(gold_embs)  # (g, D), normalized

            # Question embedding — pull from cache; construct client only if
            # something is missing. All q text should be cached from prior runs.
            q_text = q["question"]
            cached = emb_cache.get(q_text)
            if cached is None:
                if client is None:
                    from openai import OpenAI

                    client = OpenAI(timeout=60.0)
                q_emb = _embed_question(q_text, emb_cache, client)
            else:
                q_emb = _l2_normalize(cached.astype(np.float32))

            # inter_gold: mean pairwise cosine, i!=j
            sim_mat = gold_mat @ gold_mat.T  # (g, g)
            g = gold_mat.shape[0]
            inter_gold = (sim_mat.sum() - np.trace(sim_mat)) / (g * (g - 1))

            # gold_to_query: mean cosine(gold_i, query)
            gold_to_query = float(np.mean(gold_mat @ q_emb))

            # random_to_query: sample up to 20 random same-conversation turns
            #                  that are not gold.
            gold_set = set(gold_ids)
            conv_turn_ids = [
                seg.turn_id
                for seg in store.segments
                if seg.conversation_id == conv_id and seg.turn_id not in gold_set
            ]
            if not conv_turn_ids:
                random_to_query = float("nan")
            else:
                sample_k = min(20, len(conv_turn_ids))
                sampled_tids = rng.sample(conv_turn_ids, sample_k)
                sim_sum = 0.0
                for tid in sampled_tids:
                    e = _turn_emb(store, conv_id, tid)
                    if e is None:
                        continue
                    sim_sum += float(e @ q_emb)
                random_to_query = sim_sum / sample_k

            gap = float(inter_gold - gold_to_query)

            rows.append(
                {
                    "conversation_id": conv_id,
                    "question_index": q.get("question_index"),
                    "category": q.get("category"),
                    "num_gold": len(gold_embs),
                    "inter_gold": float(inter_gold),
                    "gold_to_query": gold_to_query,
                    "random_to_query": random_to_query,
                    "gap_inter_minus_q": gap,
                }
            )

        if not rows:
            return {
                "label": label,
                "n_multi_gold": 0,
                "skipped_missing_emb": skipped_missing_emb,
                "rows": [],
            }

        frac_off_center = sum(1 for r in rows if r["gap_inter_minus_q"] > 0) / len(rows)
        mean_gap = float(np.mean([r["gap_inter_minus_q"] for r in rows]))
        median_gap = float(np.median([r["gap_inter_minus_q"] for r in rows]))
        mean_inter = float(np.mean([r["inter_gold"] for r in rows]))
        mean_gold_q = float(np.mean([r["gold_to_query"] for r in rows]))
        mean_rand_q = float(
            np.mean(
                [
                    r["random_to_query"]
                    for r in rows
                    if not np.isnan(r["random_to_query"])
                ]
            )
        )

        # gap histogram: buckets from -0.30 to +0.30 step 0.05
        edges = np.arange(-0.30, 0.35, 0.05)
        gaps = np.array([r["gap_inter_minus_q"] for r in rows])
        hist, _ = np.histogram(gaps, bins=edges)
        histogram = [
            {"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int(hist[i])}
            for i in range(len(hist))
        ]

        return {
            "label": label,
            "n_multi_gold": len(rows),
            "fraction_gold_off_center": frac_off_center,
            "mean_gap_inter_minus_q": mean_gap,
            "median_gap_inter_minus_q": median_gap,
            "mean_inter_gold": mean_inter,
            "mean_gold_to_query": mean_gold_q,
            "mean_random_to_query": mean_rand_q,
            "mean_query_lift_over_random": mean_gold_q - mean_rand_q,
            "gap_histogram": histogram,
            "rows": rows,
        }

    results_30 = analyze_questions(locomo30, "locomo_30q")
    results_full = analyze_questions(locomo, "locomo_full_182q")

    # Missed-turn vs retrieved-gold analysis
    err_path = RESULTS_DIR / "error_analysis_details.json"
    missed_stats = {"available": False}
    if err_path.exists():
        err = json.load(open(err_path))
        per_q = [
            p for p in err.get("per_question", []) if p.get("dataset") == "locomo_30q"
        ]
        pair_records = []
        for p in per_q:
            conv_id = p["conversation_id"]
            retrieved_gold = p.get("retrieved_source_ids", []) or []
            missed = p.get("missed_source_ids", []) or []
            if not retrieved_gold or not missed:
                continue
            ret_embs = []
            for tid in retrieved_gold:
                e = _turn_emb(store, conv_id, tid)
                if e is not None:
                    ret_embs.append(e)
            if not ret_embs:
                continue
            ret_mat = np.stack(ret_embs)
            for mtid in missed:
                m_e = _turn_emb(store, conv_id, mtid)
                if m_e is None:
                    continue
                sims = ret_mat @ m_e
                pair_records.append(
                    {
                        "conversation_id": conv_id,
                        "missed_turn_id": mtid,
                        "retrieved_gold_ids": retrieved_gold,
                        "max_cos_to_retrieved_gold": float(sims.max()),
                        "mean_cos_to_retrieved_gold": float(sims.mean()),
                        "question_index": p.get("question_index"),
                    }
                )
        if pair_records:
            missed_stats = {
                "available": True,
                "n_pairs": len(pair_records),
                "mean_max_cos_missed_to_retrieved_gold": float(
                    np.mean([r["max_cos_to_retrieved_gold"] for r in pair_records])
                ),
                "median_max_cos_missed_to_retrieved_gold": float(
                    np.median([r["max_cos_to_retrieved_gold"] for r in pair_records])
                ),
                "mean_mean_cos_missed_to_retrieved_gold": float(
                    np.mean([r["mean_cos_to_retrieved_gold"] for r in pair_records])
                ),
                "frac_missed_max_cos_ge_0_6": float(
                    np.mean(
                        [r["max_cos_to_retrieved_gold"] >= 0.6 for r in pair_records]
                    )
                ),
                "frac_missed_max_cos_ge_0_5": float(
                    np.mean(
                        [r["max_cos_to_retrieved_gold"] >= 0.5 for r in pair_records]
                    )
                ),
                "records_preview": pair_records[:15],
            }
        else:
            missed_stats = {"available": False, "reason": "no eligible pairs"}

    # Save
    emb_cache.save()
    out = {
        "locomo_30q": results_30,
        "locomo_full_182q": results_full,
        "missed_to_retrieved_gold": missed_stats,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "spreading_activation_geometry.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Markdown report
    md_path = RESULTS_DIR / "spreading_activation_geometry.md"
    md_lines: list[str] = []
    md_lines.append("# Spreading Activation — Phase A Geometry\n")
    md_lines.append("Question: do gold turns cluster tightly with each other ")
    md_lines.append("relative to the query? (Inter-gold > gold-to-query => ")
    md_lines.append("gold is off-center from the query, so spreading may help.)\n\n")

    for label, res in [
        ("LoCoMo-30 (n multi-gold)", results_30),
        ("LoCoMo full 182Q (n multi-gold)", results_full),
    ]:
        md_lines.append(f"## {label} = {res['n_multi_gold']}\n\n")
        if res["n_multi_gold"] == 0:
            md_lines.append("No multi-gold questions.\n\n")
            continue
        md_lines.append("| Stat | Value |\n|---|---|\n")
        md_lines.append(
            f"| fraction gold off-center (gap>0) | "
            f"{res['fraction_gold_off_center']:.3f} |\n"
        )
        md_lines.append(
            f"| mean gap (inter-gold − gold-to-query) | "
            f"{res['mean_gap_inter_minus_q']:+.4f} |\n"
        )
        md_lines.append(f"| median gap | {res['median_gap_inter_minus_q']:+.4f} |\n")
        md_lines.append(f"| mean inter-gold cos | {res['mean_inter_gold']:.4f} |\n")
        md_lines.append(
            f"| mean gold-to-query cos | {res['mean_gold_to_query']:.4f} |\n"
        )
        md_lines.append(
            f"| mean random-to-query cos (baseline) | "
            f"{res['mean_random_to_query']:.4f} |\n"
        )
        md_lines.append(
            f"| mean query lift over random | "
            f"{res['mean_query_lift_over_random']:+.4f} |\n\n"
        )
        md_lines.append("### Gap histogram (inter-gold − gold-to-query)\n\n")
        md_lines.append("| bin | count |\n|---|---|\n")
        for h in res["gap_histogram"]:
            md_lines.append(f"| [{h['lo']:+.2f},{h['hi']:+.2f}) | {h['count']} |\n")
        md_lines.append("\n")

    # Missed-to-retrieved-gold
    md_lines.append("## Missed turns vs retrieved-gold neighbors (LoCoMo-30)\n\n")
    if not missed_stats.get("available"):
        md_lines.append(f"Not available: {missed_stats.get('reason', 'no data')}\n\n")
    else:
        md_lines.append(
            f"n_pairs (missed-turn × its question's retrieved-gold set) = "
            f"{missed_stats['n_pairs']}\n\n"
        )
        md_lines.append("| Stat | Value |\n|---|---|\n")
        md_lines.append(
            f"| mean max cos(missed, retrieved-gold) | "
            f"{missed_stats['mean_max_cos_missed_to_retrieved_gold']:.4f} |\n"
        )
        md_lines.append(
            f"| median max cos | "
            f"{missed_stats['median_max_cos_missed_to_retrieved_gold']:.4f} |\n"
        )
        md_lines.append(
            f"| mean mean cos | "
            f"{missed_stats['mean_mean_cos_missed_to_retrieved_gold']:.4f} |\n"
        )
        md_lines.append(
            f"| frac missed with max cos >= 0.5 | "
            f"{missed_stats['frac_missed_max_cos_ge_0_5']:.3f} |\n"
        )
        md_lines.append(
            f"| frac missed with max cos >= 0.6 | "
            f"{missed_stats['frac_missed_max_cos_ge_0_6']:.3f} |\n\n"
        )

    # Decision gate verdict based on locomo_30q
    r30 = results_30
    md_lines.append("## Decision gate\n\n")
    if r30["n_multi_gold"] == 0:
        md_lines.append("No multi-gold LoCoMo-30 questions — cannot test.\n")
        verdict = "skip"
    else:
        frac = r30["fraction_gold_off_center"]
        gap = r30["mean_gap_inter_minus_q"]
        md_lines.append(f"Fraction off-center = {frac:.3f}; mean gap = {gap:+.4f}.\n\n")
        if frac >= 0.60 and gap >= 0.05:
            md_lines.append("=> PROCEED to Phase B (clear positive signal).\n")
            verdict = "proceed"
        elif frac >= 0.40:
            md_lines.append("=> BORDERLINE. Proceed to Phase B with K0=10 only.\n")
            verdict = "borderline"
        else:
            md_lines.append(
                "=> SKIP Phase B. Gold clusters near the query, "
                "bigger-K cosine would capture it.\n"
            )
            verdict = "skip"

    md_lines.append("\n")
    md_path.write_text("".join(md_lines))
    print(f"Phase A results:\n  {md_path}\n  {json_path}\n")
    print("".join(md_lines))

    out["verdict"] = verdict
    return out


# =============================================================================
# PHASE B — SPREADING ACTIVATION RETRIEVAL
# =============================================================================


def build_knn_graph(
    store: SegmentStore,
    conversation_ids: list[str] | None = None,
    k: int = 10,
) -> dict[int, list[tuple[int, float]]]:
    """For each turn (by store index), list of (neighbor_store_index, cos_sim)
    within the SAME conversation. Excludes self. Top-k by cosine."""
    adjacency: dict[int, list[tuple[int, float]]] = {}
    conv_set = set(conversation_ids) if conversation_ids else None

    by_conv: dict[str, list[int]] = defaultdict(list)
    for seg in store.segments:
        if conv_set is not None and seg.conversation_id not in conv_set:
            continue
        by_conv[seg.conversation_id].append(seg.index)

    for cid, idx_list in by_conv.items():
        idx_arr = np.array(idx_list)
        embs = store.normalized_embeddings[idx_arr]  # (n, D)
        sim = embs @ embs.T  # (n, n)
        np.fill_diagonal(sim, -1.0)
        # top-k per row
        top_k = min(k, sim.shape[1] - 1) if sim.shape[1] > 1 else 0
        if top_k == 0:
            for i, gi in enumerate(idx_list):
                adjacency[int(gi)] = []
            continue
        top_idx = np.argpartition(-sim, kth=top_k - 1, axis=1)[:, :top_k]
        for row_i in range(sim.shape[0]):
            picks = top_idx[row_i]
            # sort these picks by score desc
            picks_sorted = picks[np.argsort(-sim[row_i, picks])]
            neighbors = [
                (int(idx_list[int(p)]), float(sim[row_i, int(p)])) for p in picks_sorted
            ]
            adjacency[int(idx_list[row_i])] = neighbors

    return adjacency


def save_adjacency_json(adj: dict[int, list[tuple[int, float]]], path: Path) -> None:
    serial = {
        str(k): [[int(n), round(float(s), 6)] for n, s in v] for k, v in adj.items()
    }
    with open(path, "w") as f:
        json.dump(serial, f)


def spread_from_seeds(
    seeds: list[tuple[int, float]],  # (store_index, cosine_activation)
    adjacency: dict[int, list[tuple[int, float]]],
    alpha: float,
    extra_seed_boost: float = 1.0,
) -> dict[int, float]:
    """Single-step spreading: each seed gives alpha * a(seed) * sim(seed,nbr)
    to each neighbor. Seeds also keep their original activation
    (extra_seed_boost multiplies seed's retained score)."""
    activation: dict[int, float] = {}
    for sidx, score in seeds:
        activation[sidx] = activation.get(sidx, 0.0) + extra_seed_boost * score
    for sidx, score in seeds:
        for nidx, sim in adjacency.get(sidx, []):
            activation[nidx] = activation.get(nidx, 0.0) + alpha * score * sim
    return activation


def spreading_retrieve(
    store: SegmentStore,
    query_emb: np.ndarray,
    conversation_id: str,
    adjacency: dict[int, list[tuple[int, float]]],
    K_final: int,
    K0: int = 10,
    alpha: float = 0.5,
) -> list[Segment]:
    """Cosine top-K0 seeds -> spread activation along kNN graph ->
    return top K_final by total activation."""
    # cosine top-K0 seeds
    seed_result = store.search(query_emb, top_k=K0, conversation_id=conversation_id)
    seeds = [
        (seg.index, float(score))
        for seg, score in zip(seed_result.segments, seed_result.scores)
    ]
    activation = spread_from_seeds(seeds, adjacency, alpha=alpha)
    # rank
    ranked = sorted(activation.items(), key=lambda kv: -kv[1])
    out: list[Segment] = []
    for idx, _score in ranked:
        seg = store.segments[idx]
        if seg.conversation_id != conversation_id:
            continue
        out.append(seg)
        if len(out) >= K_final:
            break
    return out


def spread_over_initial(
    store: SegmentStore,
    initial_segments: list[Segment],
    initial_scores: list[float],
    conversation_id: str,
    adjacency: dict[int, list[tuple[int, float]]],
    K_final: int,
    alpha: float = 0.5,
) -> list[Segment]:
    """Spread over an arbitrary list of initial seed segments (e.g., v2f output)."""
    seeds = [(s.index, float(sc)) for s, sc in zip(initial_segments, initial_scores)]
    if not seeds:
        return []
    activation = spread_from_seeds(seeds, adjacency, alpha=alpha)
    ranked = sorted(activation.items(), key=lambda kv: -kv[1])
    out: list[Segment] = []
    for idx, _ in ranked:
        seg = store.segments[idx]
        if seg.conversation_id != conversation_id:
            continue
        out.append(seg)
        if len(out) >= K_final:
            break
    return out


# =============================================================================
# Phase B evaluation
# =============================================================================


def run_phase_b(verdict: str) -> dict:
    """Evaluate spread_plain and spread_v2f on LoCoMo-30 using fair-backfill."""
    from best_shot import MetaV2f
    from openai import OpenAI

    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    cfg = DATASETS["locomo_30q"]
    with open(DATA_DIR / cfg["questions"]) as f:
        questions = [q for q in json.load(f) if cfg["filter"](q)]
    questions = questions[: cfg["max_questions"]]

    conv_ids = list({q["conversation_id"] for q in questions})
    print(f"Building kNN graph over {len(conv_ids)} conversations, k=10 ...")
    adj = build_knn_graph(store, conversation_ids=conv_ids, k=10)
    print(
        f"  Graph size: {len(adj)} nodes, "
        f"mean degree {np.mean([len(v) for v in adj.values()]):.1f}"
    )

    # Save adjacency
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adj_path = RESULTS_DIR / "knn_graph_locomo.json"
    save_adjacency_json(adj, adj_path)
    print(f"  Saved adjacency: {adj_path}")

    emb_cache = EmbeddingCache()
    client = OpenAI(timeout=60.0)

    # K0 values to try — if borderline, only K0=10
    K0_values = [10] if verdict == "borderline" else [10, 20]
    alpha = 0.5

    # v2f architecture
    v2f = MetaV2f(store)

    # Per-question records — track every variant
    per_q_records: list[dict] = []

    for i, q in enumerate(questions):
        q_text = q["question"]
        conv_id = q["conversation_id"]
        source_ids = set(q["source_chat_ids"])

        # Question embedding
        q_emb = _embed_question(q_text, emb_cache, client)

        # Cosine top-max(BUDGETS) (baseline)
        max_K = max(BUDGETS)
        cos_res = store.search(q_emb, top_k=max_K, conversation_id=conv_id)
        cosine_segments = list(cos_res.segments)

        # --- v2f retrieval (for the v2f baseline AND for spread_v2f seeds)
        v2f.reset_counters()
        t0 = time.time()
        v2f_out = v2f.retrieve(q_text, conv_id)
        v2f_time = time.time() - t0
        # dedupe
        seen: set[int] = set()
        v2f_segments: list[Segment] = []
        for s in v2f_out.segments:
            if s.index not in seen:
                v2f_segments.append(s)
                seen.add(s.index)

        # Seed scores for v2f spreading: use cosine(q, seg)
        v2f_seed_scores = []
        for seg in v2f_segments:
            v2f_seed_scores.append(
                float(store.normalized_embeddings[seg.index] @ q_emb)
            )

        record = {
            "conversation_id": conv_id,
            "question_index": q.get("question_index"),
            "category": q.get("category", "locomo"),
            "question": q_text,
            "source_chat_ids": sorted(source_ids),
            "num_source_turns": len(source_ids),
            "time_s": round(v2f_time, 2),
            "llm_calls": v2f.llm_calls,
            "embed_calls": v2f.embed_calls,
            "v2f_total_retrieved": len(v2f_segments),
            "fair_backfill": {},
        }

        # ---- Evaluate at each K_final (BUDGETS)
        for K in BUDGETS:
            # cosine baseline: top-K
            cos_at_K = cosine_segments[:K]
            cos_ids = {s.turn_id for s in cos_at_K}
            record["fair_backfill"][f"cosine_r@{K}"] = round(
                compute_recall(cos_ids, source_ids), 4
            )

            # v2f: fair backfill (arch first, then cosine backfill)
            _b, v2f_r, _ = fair_backfill_evaluate(
                v2f_segments, cosine_segments, source_ids, K
            )
            record["fair_backfill"][f"v2f_r@{K}"] = round(v2f_r, 4)

            # spread_plain: cosine top-K0 seeds -> spread -> top-K
            # If spread yields < K, backfill with cosine top-K.
            for K0 in K0_values:
                sp_segs = spreading_retrieve(
                    store,
                    q_emb,
                    conv_id,
                    adj,
                    K_final=K,
                    K0=K0,
                    alpha=alpha,
                )
                _b, sp_r, _ = fair_backfill_evaluate(
                    sp_segs, cosine_segments, source_ids, K
                )
                record["fair_backfill"][f"spread_plain_K0={K0}_r@{K}"] = round(sp_r, 4)

            # spread_v2f: spread from v2f's retrieved segments, seed scores =
            # cosine(q, seg); then top-K + backfill
            spv2f_segs = spread_over_initial(
                store,
                v2f_segments,
                v2f_seed_scores,
                conv_id,
                adj,
                K_final=K,
                alpha=alpha,
            )
            _b, spv2f_r, _ = fair_backfill_evaluate(
                spv2f_segs, cosine_segments, source_ids, K
            )
            record["fair_backfill"][f"spread_v2f_r@{K}"] = round(spv2f_r, 4)

        per_q_records.append(record)
        print(
            f"[{i + 1}/{len(questions)}] "
            f"cos@20={record['fair_backfill']['cosine_r@20']:.3f} "
            f"v2f@20={record['fair_backfill']['v2f_r@20']:.3f} "
            f"spK0=10@20={record['fair_backfill']['spread_plain_K0=10_r@20']:.3f} "
            f"spv2f@20={record['fair_backfill']['spread_v2f_r@20']:.3f} "
            f"cos@50={record['fair_backfill']['cosine_r@50']:.3f} "
            f"sp@50={record['fair_backfill']['spread_plain_K0=10_r@50']:.3f}"
        )
        if (i + 1) % 5 == 0:
            v2f.save_caches()
            emb_cache.save()
    v2f.save_caches()
    emb_cache.save()

    # Aggregate
    variants = (
        ["cosine", "v2f"]
        + [f"spread_plain_K0={K0}" for K0 in K0_values]
        + ["spread_v2f"]
    )

    n = len(per_q_records)
    summaries: dict[str, dict] = {}
    for v in variants:
        summaries[v] = {"n": n}
        for K in BUDGETS:
            vals = [r["fair_backfill"][f"{v}_r@{K}"] for r in per_q_records]
            summaries[v][f"r@{K}"] = round(float(np.mean(vals)), 4)

    # Per-category
    per_cat: dict[str, dict] = {}
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in per_q_records:
        by_cat[r["category"]].append(r)
    for cat, rs in sorted(by_cat.items()):
        per_cat[cat] = {"n": len(rs)}
        for v in variants:
            for K in BUDGETS:
                vals = [r["fair_backfill"][f"{v}_r@{K}"] for r in rs]
                per_cat[cat][f"{v}_r@{K}"] = round(float(np.mean(vals)), 4)

    out = {
        "variants": variants,
        "summary": summaries,
        "per_category": per_cat,
        "records": per_q_records,
        "params": {
            "alpha": alpha,
            "K0_values": K0_values,
            "graph_k": 10,
        },
    }

    json_path = RESULTS_DIR / "spreading_activation_eval.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    md = []
    md.append("# Spreading Activation — Phase B Eval (LoCoMo-30)\n\n")
    md.append(f"Params: α={alpha}, kNN k=10, K0∈{K0_values}.\n\n")
    md.append("## Summary (recall averaged over n=30)\n\n")
    md.append("| variant | r@20 | r@50 |\n|---|---|---|\n")
    for v in variants:
        md.append(
            f"| {v} | {summaries[v]['r@20']:.4f} | {summaries[v]['r@50']:.4f} |\n"
        )
    md.append("\n## Per-category\n\n")
    md.append(
        "| category | n | "
        + " | ".join(f"{v} r@{K}" for v in variants for K in BUDGETS)
        + " |\n"
    )
    md.append("|---|---|" + "|".join("---" for _ in variants for _ in BUDGETS) + "|\n")
    for cat, entry in per_cat.items():
        row = f"| {cat} | {entry['n']} |"
        for v in variants:
            for K in BUDGETS:
                row += f" {entry[f'{v}_r@{K}']:.3f} |"
        md.append(row + "\n")
    md.append("\n")

    # Verdict
    cos20 = summaries["cosine"]["r@20"]
    cos50 = summaries["cosine"]["r@50"]
    v2f20 = summaries["v2f"]["r@20"]
    v2f50 = summaries["v2f"]["r@50"]
    sp20 = summaries["spread_plain_K0=10"]["r@20"]
    sp50 = summaries["spread_plain_K0=10"]["r@50"]
    spv2f20 = summaries["spread_v2f"]["r@20"]
    spv2f50 = summaries["spread_v2f"]["r@50"]

    md.append("## Verdict\n\n")
    md.append(
        f"- spread_plain vs cosine: r@20 {sp20 - cos20:+.4f}, "
        f"r@50 {sp50 - cos50:+.4f}\n"
    )
    md.append(
        f"- spread_v2f vs v2f:     r@20 {spv2f20 - v2f20:+.4f}, "
        f"r@50 {spv2f50 - v2f50:+.4f}\n\n"
    )
    if sp20 > cos20 and sp50 > cos50 and spv2f20 > v2f20 and spv2f50 > v2f50:
        v = "Ship spreading (wins both standalone and on top of v2f)."
    elif sp20 > cos20 and sp50 > cos50 and spv2f20 <= v2f20:
        v = (
            "Narrow-use: spread_plain beats plain cosine, but does NOT "
            "compose with v2f. Good as a cheap retrieval but skip stacking."
        )
    elif spv2f20 > v2f20 and spv2f50 > v2f50 and sp20 <= cos20:
        v = "Use only stacked on v2f."
    else:
        v = "Abandon: spreading activation does not help here."
    md.append(f"**{v}**\n")

    md_path = RESULTS_DIR / "spreading_activation_eval.md"
    md_path.write_text("".join(md))
    print(f"\nPhase B results:\n  {md_path}\n  {json_path}\n")
    print("".join(md))
    out["verdict_text"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b", "all"], default="all")
    args = ap.parse_args()

    if args.phase in ("a", "all"):
        res_a = run_phase_a()
        verdict = res_a["verdict"]
    else:
        verdict = "proceed"
    if args.phase in ("b", "all"):
        if verdict == "skip":
            print("\nPhase A verdict = skip; not running Phase B.")
        else:
            run_phase_b(verdict)


if __name__ == "__main__":
    main()
