"""Mechanical (zero-extra-LLM) enhancements over v2f.

Two ideas, no additional LLM calls beyond reused v2f cues:

Idea 1: Cluster-then-sample
  v2f retrieves a pool of ~30 segments. Cluster in embedding space and
  sample one representative per cluster to form the top-K output.

Idea 2: Mechanical cue expansion
  v2f retrieves a pool. Use some of those retrieved segments' TEXT as
  secondary queries (no extra LLM calls) to fetch adjacent content.

Combined:
  D. cluster_then_expand: cluster first, then expand with survivors
  E. expand_then_cluster: expand first, cluster the bigger pool, sample

All variants share the bestshot caches (no new LLM calls for v2f step).

Usage:
    uv run python mechanical_enhance.py --quick
    uv run python mechanical_enhance.py --full
    uv run python mechanical_enhance.py --variant cluster_sample_20 --quick
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from associative_recall import Segment, SegmentStore
from best_shot import (
    V2F_PROMPT,
    BestshotBase,
    BestshotResult,
    _format_segments,
    _parse_cues,
)
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50]

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
    "puzzle_16q": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "filter": None,
        "max_questions": None,
    },
    "advanced_23q": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "filter": None,
        "max_questions": None,
    },
}


# ---------------------------------------------------------------------------
# V2f pool retrieval — shared by all variants (1 LLM call via cache)
# ---------------------------------------------------------------------------
def run_v2f_pool(
    arch: BestshotBase,
    question: str,
    conv_id: str,
    top_k_per_cue: int = 10,
) -> tuple[list[Segment], list[str]]:
    """Replicate MetaV2f retrieval, return pool + cues.

    Returns (pool_segments_deduped_in_order, cues_used).
    The pool has: top-10 question, top-10 cue1, top-10 cue2.
    """
    query_emb = arch.embed_text(question)
    hop0 = arch.store.search(query_emb, top_k=top_k_per_cue, conversation_id=conv_id)
    all_segments = list(hop0.segments)
    exclude = {s.index for s in all_segments}

    context_section = "RETRIEVED CONVERSATION EXCERPTS SO FAR:\n" + _format_segments(
        all_segments
    )
    prompt = V2F_PROMPT.format(question=question, context_section=context_section)
    output = arch.llm_call(prompt)
    cues = _parse_cues(output)[:2]

    for cue in cues:
        cue_emb = arch.embed_text(cue)
        result = arch.store.search(
            cue_emb,
            top_k=top_k_per_cue,
            conversation_id=conv_id,
            exclude_indices=exclude,
        )
        for seg in result.segments:
            if seg.index not in exclude:
                all_segments.append(seg)
                exclude.add(seg.index)

    return all_segments, cues


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def get_segment_embeddings(store: SegmentStore, segments: list[Segment]) -> np.ndarray:
    """Return normalized embeddings for the given segments (shape: n x d)."""
    if not segments:
        return np.zeros((0, store.normalized_embeddings.shape[1]), dtype=np.float32)
    idxs = [s.index for s in segments]
    return store.normalized_embeddings[idxs]


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def cluster_sample(
    pool: list[Segment],
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> list[Segment]:
    """K-means cluster pool's embeddings into n_clusters; pick rep closest
    to centroid per cluster. Preserves pool order for ties.

    Returns list of representatives (up to n_clusters), ordered by cluster
    density/size descending to surface larger clusters first.
    """
    from sklearn.cluster import KMeans

    n = len(pool)
    if n == 0:
        return []
    if n <= n_clusters:
        return list(pool)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_

    # Normalize centroids for cosine-like distances (embeddings already L2-normed)
    reps: list[tuple[int, Segment]] = []  # (cluster_size_desc, segment)
    cluster_sizes: list[tuple[int, int]] = []  # (cluster_id, size)
    for c in range(n_clusters):
        mask = labels == c
        idxs_in_cluster = np.where(mask)[0]
        if len(idxs_in_cluster) == 0:
            continue
        cluster_sizes.append((c, len(idxs_in_cluster)))
        cluster_embs = embeddings[mask]
        centroid = centroids[c]
        # Pick closest to centroid (max cosine similarity since unit vectors
        # approx; use negative L2 distance which is monotone for unit vecs)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        best = int(idxs_in_cluster[int(np.argmin(dists))])
        reps.append((len(idxs_in_cluster), pool[best]))

    # Order reps by cluster size desc, then by original pool order
    reps.sort(key=lambda r: (-r[0], pool.index(r[1])))
    return [r[1] for r in reps]


def adaptive_cluster_sample(
    pool: list[Segment],
    embeddings: np.ndarray,
    min_k: int = 10,
    max_k: int = 25,
    seed: int = 42,
) -> list[Segment]:
    """Pick best k via silhouette score in [min_k, max_k], then cluster_sample."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(pool)
    if n == 0:
        return []
    if n <= min_k + 1:
        return list(pool)

    upper = min(max_k, n - 1)
    if upper < min_k:
        return cluster_sample(pool, embeddings, min_k, seed)

    best_k = min_k
    best_score = -2.0
    for k in range(min_k, upper + 1, 2):  # step 2 for speed
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(embeddings, labels, metric="cosine")
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_k = k

    return cluster_sample(pool, embeddings, best_k, seed)


def hierarchical_cluster_sample(
    pool: list[Segment],
    embeddings: np.ndarray,
    distance_threshold: float = 0.35,
) -> list[Segment]:
    """Agglomerative clustering with cosine distance, cut at threshold."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(pool)
    if n == 0:
        return []
    if n < 2:
        return list(pool)

    ac = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = ac.fit_predict(embeddings)

    # Representative per cluster: closest to centroid
    reps: list[tuple[int, Segment]] = []
    for c in sorted(set(labels)):
        mask = labels == c
        idxs_in_cluster = np.where(mask)[0]
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-10)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        best = int(idxs_in_cluster[int(np.argmin(dists))])
        reps.append((len(idxs_in_cluster), pool[best]))

    reps.sort(key=lambda r: (-r[0], pool.index(r[1])))
    return [r[1] for r in reps]


def expand_cue_segments(
    arch: BestshotBase,
    pool: list[Segment],
    cue_retrieved_start_idx: int,
    conv_id: str,
    n_seed: int = 5,
    top_k_per_expansion: int = 3,
) -> list[Segment]:
    """Expand pool by using top-n_seed cue-retrieved segments as queries.

    pool[:cue_retrieved_start_idx] are question-retrieved (skip).
    pool[cue_retrieved_start_idx:] are cue-retrieved — use top n_seed.
    For each, embed its text (cached after first time) and fetch top-K
    excluding the existing pool. Returns expanded pool (pool + new segs).
    """
    expanded = list(pool)
    exclude = {s.index for s in expanded}

    cue_segs = pool[cue_retrieved_start_idx : cue_retrieved_start_idx + n_seed]
    for seg in cue_segs:
        # Use segment text as a query (no new LLM calls; may use cache for embed)
        text = seg.text.strip()
        if not text:
            continue
        q_emb = arch.embed_text(text)
        result = arch.store.search(
            q_emb,
            top_k=top_k_per_expansion + 5,  # slop; dedup below
            conversation_id=conv_id,
            exclude_indices=exclude,
        )
        added = 0
        for s in result.segments:
            if s.index in exclude:
                continue
            expanded.append(s)
            exclude.add(s.index)
            added += 1
            if added >= top_k_per_expansion:
                break

    return expanded


# ---------------------------------------------------------------------------
# Variant runners — each returns an ordered list of segments (truncated at K
# via fair_backfill at eval time). The arch returns its best-ordered pool;
# eval takes pool[:K] and backfills with cosine if short.
# ---------------------------------------------------------------------------
class MechanicalBase(BestshotBase):
    """Shares bestshot caches; subclass overrides retrieve()."""

    variant_name: str = "base"

    def run_v2f(self, question: str, conversation_id: str):
        return run_v2f_pool(self, question, conversation_id)


class V2fBaselineMech(MechanicalBase):
    """Plain v2f: no enhancement. For sanity comparison."""

    variant_name = "v2f_baseline"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        pool, cues = self.run_v2f(question, conversation_id)
        return BestshotResult(
            segments=pool, metadata={"name": self.variant_name, "cues": cues}
        )


class ClusterSample20(MechanicalBase):
    variant_name = "cluster_sample_20"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        pool, cues = self.run_v2f(question, conversation_id)
        embs = get_segment_embeddings(self.store, pool)
        reps = cluster_sample(pool, embs, n_clusters=20)
        # Append any non-rep segments at the end so eval at K>20 still sees them
        rep_ids = {s.index for s in reps}
        tail = [s for s in pool if s.index not in rep_ids]
        final = reps + tail
        return BestshotResult(
            segments=final,
            metadata={"name": self.variant_name, "cues": cues, "n_reps": len(reps)},
        )


class ClusterSampleAdaptive(MechanicalBase):
    variant_name = "cluster_sample_adaptive"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        pool, cues = self.run_v2f(question, conversation_id)
        embs = get_segment_embeddings(self.store, pool)
        reps = adaptive_cluster_sample(pool, embs, min_k=10, max_k=25)
        rep_ids = {s.index for s in reps}
        tail = [s for s in pool if s.index not in rep_ids]
        final = reps + tail
        return BestshotResult(
            segments=final,
            metadata={"name": self.variant_name, "cues": cues, "n_reps": len(reps)},
        )


class ClusterSampleHierarchical(MechanicalBase):
    variant_name = "cluster_sample_hierarchical"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        pool, cues = self.run_v2f(question, conversation_id)
        embs = get_segment_embeddings(self.store, pool)
        reps = hierarchical_cluster_sample(pool, embs, distance_threshold=0.35)
        rep_ids = {s.index for s in reps}
        tail = [s for s in pool if s.index not in rep_ids]
        final = reps + tail
        return BestshotResult(
            segments=final,
            metadata={"name": self.variant_name, "cues": cues, "n_reps": len(reps)},
        )


class ExpandTopCueSegs(MechanicalBase):
    variant_name = "expand_top_cue_segs"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        pool, cues = self.run_v2f(question, conversation_id)
        # First 10 are question-retrieved; rest are cue-retrieved
        expanded = expand_cue_segments(
            self,
            pool,
            cue_retrieved_start_idx=10,
            conv_id=conversation_id,
            n_seed=5,
            top_k_per_expansion=3,
        )
        return BestshotResult(
            segments=expanded,
            metadata={
                "name": self.variant_name,
                "cues": cues,
                "pool_size": len(pool),
                "expanded_size": len(expanded),
            },
        )


class ExpandAllCueSegs(MechanicalBase):
    variant_name = "expand_all_cue_segs"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        pool, cues = self.run_v2f(question, conversation_id)
        cue_count = len(pool) - 10
        expanded = expand_cue_segments(
            self,
            pool,
            cue_retrieved_start_idx=10,
            conv_id=conversation_id,
            n_seed=max(cue_count, 0),
            top_k_per_expansion=2,
        )
        return BestshotResult(
            segments=expanded,
            metadata={
                "name": self.variant_name,
                "cues": cues,
                "pool_size": len(pool),
                "expanded_size": len(expanded),
            },
        )


class ExpandAndSecondPass(MechanicalBase):
    """Expand, then take top-5 expanded results (by cosine vs question)
    and use them as additional queries."""

    variant_name = "expand_and_v2f_second_pass"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        pool, cues = self.run_v2f(question, conversation_id)
        query_emb = self.embed_text(question)
        expanded = expand_cue_segments(
            self,
            pool,
            cue_retrieved_start_idx=10,
            conv_id=conversation_id,
            n_seed=5,
            top_k_per_expansion=3,
        )
        # Re-rank expansion segments (those beyond original pool) by cosine
        # to the question, pick top-5, use as secondary queries.
        new_segs = expanded[len(pool) :]
        if new_segs:
            new_embs = get_segment_embeddings(self.store, new_segs)
            sims = new_embs @ (query_emb / max(np.linalg.norm(query_emb), 1e-10))
            ordered = sorted(zip(new_segs, sims.tolist()), key=lambda x: -x[1])
            top5 = [s for s, _ in ordered[:5]]

            exclude = {s.index for s in expanded}
            for seg in top5:
                q_emb = self.embed_text(seg.text.strip())
                result = self.store.search(
                    q_emb,
                    top_k=5,
                    conversation_id=conversation_id,
                    exclude_indices=exclude,
                )
                added = 0
                for s in result.segments:
                    if s.index in exclude:
                        continue
                    expanded.append(s)
                    exclude.add(s.index)
                    added += 1
                    if added >= 2:
                        break

        return BestshotResult(
            segments=expanded,
            metadata={
                "name": self.variant_name,
                "cues": cues,
                "pool_size": len(pool),
                "expanded_size": len(expanded),
            },
        )


class ClusterThenExpand(MechanicalBase):
    """Cluster the v2f pool, keep representatives, then expand on
    cue-retrieved survivors."""

    variant_name = "cluster_then_expand"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        pool, cues = self.run_v2f(question, conversation_id)
        embs = get_segment_embeddings(self.store, pool)
        reps = cluster_sample(pool, embs, n_clusters=20)
        rep_ids = {s.index for s in reps}

        # Among reps, identify ones that came from the cue-retrieved segment
        # region (i.e. pool index >= 10). Use those as expansion seeds.
        cue_rep_segs = [s for s in reps if pool.index(s) >= 10][:5]
        expanded = list(reps)
        exclude = {s.index for s in expanded}
        for seg in cue_rep_segs:
            q_emb = self.embed_text(seg.text.strip())
            result = self.store.search(
                q_emb,
                top_k=5,
                conversation_id=conversation_id,
                exclude_indices=exclude,
            )
            added = 0
            for s in result.segments:
                if s.index in exclude:
                    continue
                expanded.append(s)
                exclude.add(s.index)
                added += 1
                if added >= 3:
                    break

        # Append non-rep original pool members at the end
        tail = [s for s in pool if s.index not in exclude]
        final = expanded + tail

        return BestshotResult(
            segments=final,
            metadata={
                "name": self.variant_name,
                "cues": cues,
                "n_reps": len(reps),
                "final_size": len(final),
            },
        )


class ExpandThenCluster(MechanicalBase):
    """Expand first, then cluster the bigger pool into 20 reps."""

    variant_name = "expand_then_cluster"

    def retrieve(self, question: str, conversation_id: str) -> BestshotResult:
        pool, cues = self.run_v2f(question, conversation_id)
        expanded = expand_cue_segments(
            self,
            pool,
            cue_retrieved_start_idx=10,
            conv_id=conversation_id,
            n_seed=5,
            top_k_per_expansion=3,
        )
        embs = get_segment_embeddings(self.store, expanded)
        reps = cluster_sample(expanded, embs, n_clusters=20)
        rep_ids = {s.index for s in reps}
        tail = [s for s in expanded if s.index not in rep_ids]
        final = reps + tail
        return BestshotResult(
            segments=final,
            metadata={
                "name": self.variant_name,
                "cues": cues,
                "pool_size": len(pool),
                "expanded_size": len(expanded),
                "n_reps": len(reps),
            },
        )


VARIANTS = {
    "v2f_baseline": V2fBaselineMech,
    "cluster_sample_20": ClusterSample20,
    "cluster_sample_adaptive": ClusterSampleAdaptive,
    "cluster_sample_hierarchical": ClusterSampleHierarchical,
    "expand_top_cue_segs": ExpandTopCueSegs,
    "expand_all_cue_segs": ExpandAllCueSegs,
    "expand_and_v2f_second_pass": ExpandAndSecondPass,
    "cluster_then_expand": ClusterThenExpand,
    "expand_then_cluster": ExpandThenCluster,
}


# ---------------------------------------------------------------------------
# Evaluation helpers — FAIR K-budget backfill
# ---------------------------------------------------------------------------
def load_dataset(ds_name: str) -> tuple[SegmentStore, list[dict]]:
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        questions = json.load(f)
    if cfg["filter"]:
        questions = [q for q in questions if cfg["filter"](q)]
    if cfg["max_questions"]:
        questions = questions[: cfg["max_questions"]]
    return store, questions


def compute_recall(retrieved_ids: set[int], source_ids: set[int]) -> float:
    if not source_ids:
        return 1.0
    return len(retrieved_ids & source_ids) / len(source_ids)


def fair_backfill_evaluate(
    arch_segments: list[Segment],
    cosine_segments: list[Segment],
    source_ids: set[int],
    budget: int,
) -> tuple[float, float]:
    seen: set[int] = set()
    arch_unique: list[Segment] = []
    for s in arch_segments:
        if s.index not in seen:
            arch_unique.append(s)
            seen.add(s.index)
    arch_at_K = arch_unique[:budget]
    arch_indices = {s.index for s in arch_at_K}
    if len(arch_at_K) < budget:
        backfill = [s for s in cosine_segments if s.index not in arch_indices]
        needed = budget - len(arch_at_K)
        arch_at_K = arch_at_K + backfill[:needed]
    arch_at_K = arch_at_K[:budget]
    baseline_at_K = cosine_segments[:budget]
    arch_ids = {s.turn_id for s in arch_at_K}
    baseline_ids = {s.turn_id for s in baseline_at_K}
    return compute_recall(baseline_ids, source_ids), compute_recall(
        arch_ids, source_ids
    )


def evaluate_question(arch: BestshotBase, question: dict) -> dict:
    q_text = question["question"]
    conv_id = question["conversation_id"]
    source_ids = set(question["source_chat_ids"])

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

    query_emb = arch.embed_text(q_text)
    max_K = max(BUDGETS)
    cosine_result = arch.store.search(query_emb, top_k=max_K, conversation_id=conv_id)
    cosine_segments = list(cosine_result.segments)

    row = {
        "conversation_id": conv_id,
        "category": question.get("category", "unknown"),
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
        b_rec, a_rec = fair_backfill_evaluate(
            arch_segments, cosine_segments, source_ids, K
        )
        row["fair_backfill"][f"baseline_r@{K}"] = round(b_rec, 4)
        row["fair_backfill"][f"arch_r@{K}"] = round(a_rec, 4)
        row["fair_backfill"][f"delta_r@{K}"] = round(a_rec - b_rec, 4)

    return row


def summarize(results: list[dict], arch_name: str, dataset: str) -> dict:
    n = len(results)
    if n == 0:
        return {"arch": arch_name, "dataset": dataset, "n": 0}
    summary: dict = {"arch": arch_name, "dataset": dataset, "n": n}
    for K in BUDGETS:
        b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in results]
        a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in results]
        b_mean = sum(b_vals) / n
        a_mean = sum(a_vals) / n
        wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
        losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
        ties = n - wins - losses
        summary[f"baseline_r@{K}"] = round(b_mean, 4)
        summary[f"arch_r@{K}"] = round(a_mean, 4)
        summary[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
        summary[f"W/T/L_r@{K}"] = f"{wins}/{ties}/{losses}"
    summary["avg_total_retrieved"] = round(
        sum(r["total_arch_retrieved"] for r in results) / n, 1
    )
    summary["avg_llm_calls"] = round(sum(r["llm_calls"] for r in results) / n, 1)
    summary["avg_embed_calls"] = round(sum(r["embed_calls"] for r in results) / n, 1)
    return summary


def summarize_by_category(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    out = {}
    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        entry: dict = {"n": n}
        for K in BUDGETS:
            b_vals = [r["fair_backfill"][f"baseline_r@{K}"] for r in rs]
            a_vals = [r["fair_backfill"][f"arch_r@{K}"] for r in rs]
            b_mean = sum(b_vals) / n
            a_mean = sum(a_vals) / n
            wins = sum(1 for b, a in zip(b_vals, a_vals) if a > b + 0.001)
            losses = sum(1 for b, a in zip(b_vals, a_vals) if b > a + 0.001)
            entry[f"baseline_r@{K}"] = round(b_mean, 4)
            entry[f"arch_r@{K}"] = round(a_mean, 4)
            entry[f"delta_r@{K}"] = round(a_mean - b_mean, 4)
            entry[f"W/T/L_r@{K}"] = f"{wins}/{n - wins - losses}/{losses}"
        out[cat] = entry
    return out


# ---------------------------------------------------------------------------
# Quick-test question selection
# ---------------------------------------------------------------------------
def quick_test_questions(ds_name: str, all_questions: list[dict]) -> list[dict]:
    """Select up to 5 questions for a quick sanity test."""
    if ds_name == "locomo_30q":
        return all_questions[:5]
    # For synthetic/puzzle/advanced datasets, take first 5 covering categories
    seen_cats: set[str] = set()
    picked: list[dict] = []
    for q in all_questions:
        cat = q.get("category", "?")
        if cat not in seen_cats:
            seen_cats.add(cat)
            picked.append(q)
        if len(picked) >= 5:
            break
    while len(picked) < 5 and len(picked) < len(all_questions):
        for q in all_questions:
            if q not in picked:
                picked.append(q)
                if len(picked) >= 5:
                    break
    return picked[:5]


def run_variant_on_dataset(
    variant_name: str,
    ds_name: str,
    store: SegmentStore,
    questions: list[dict],
    quick: bool,
) -> tuple[list[dict], dict, dict]:
    cls = VARIANTS[variant_name]
    arch = cls(store)
    results = []
    label = "quick" if quick else "full"
    print(f"\n{'=' * 70}")
    print(f"{variant_name} | {ds_name} | {len(questions)} questions ({label})")
    print(f"{'=' * 70}", flush=True)

    for i, q in enumerate(questions):
        q_short = q["question"][:55]
        print(
            f"  [{i + 1}/{len(questions)}] {q.get('category', '?')}: {q_short}...",
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

    summary = summarize(results, variant_name, ds_name)
    by_cat = summarize_by_category(results)

    print(f"\n--- {variant_name} on {ds_name} ({label}) ---")
    for K in BUDGETS:
        print(
            f"  r@{K}: baseline={summary[f'baseline_r@{K}']:.3f} "
            f"arch={summary[f'arch_r@{K}']:.3f} "
            f"delta={summary[f'delta_r@{K}']:+.3f} "
            f"W/T/L={summary[f'W/T/L_r@{K}']}"
        )
    print(
        f"  avg retrieved={summary['avg_total_retrieved']:.0f} "
        f"llm={summary['avg_llm_calls']:.1f} "
        f"embed={summary['avg_embed_calls']:.1f}"
    )
    return results, summary, by_cat


# ---------------------------------------------------------------------------
# v2f reference numbers (from fairbackfill_summary.json) for comparison
# ---------------------------------------------------------------------------
V2F_REFERENCE = {
    "locomo_30q": {"arch_r@20": 0.7556, "arch_r@50": 0.8583, "baseline_r@20": 0.3833},
    "synthetic_19q": {
        "arch_r@20": 0.6130,
        "arch_r@50": 0.8513,
        "baseline_r@20": 0.5694,
    },
    "puzzle_16q": {"arch_r@20": 0.4804, "arch_r@50": 0.9169, "baseline_r@20": 0.4316},
    "advanced_23q": {"arch_r@20": 0.5931, "arch_r@50": 0.9021, "baseline_r@20": 0.4866},
}


def compare_to_v2f(summaries_by_ds: dict[str, dict]) -> dict:
    """For each dataset, compute delta vs v2f reference."""
    out = {}
    for ds, s in summaries_by_ds.items():
        ref = V2F_REFERENCE.get(ds, {})
        out[ds] = {
            "variant_r@20": s["arch_r@20"],
            "v2f_r@20": ref.get("arch_r@20"),
            "vs_v2f_r@20": round(s["arch_r@20"] - ref.get("arch_r@20", 0), 4),
            "variant_r@50": s["arch_r@50"],
            "v2f_r@50": ref.get("arch_r@50"),
            "vs_v2f_r@50": round(s["arch_r@50"] - ref.get("arch_r@50", 0), 4),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanical enhancements over v2f")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick sanity test: 5 questions per dataset",
    )
    parser.add_argument(
        "--full", action="store_true", help="Full eval across all 4 datasets"
    )
    parser.add_argument(
        "--variant", type=str, default=None, help="Run a specific variant only"
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated list of variants to run",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to run",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Variants:")
        for name in VARIANTS:
            print(f"  {name}")
        return

    if args.variant:
        variant_names = [args.variant]
    elif args.variants:
        variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variant_names = list(VARIANTS.keys())

    if args.datasets:
        ds_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        ds_names = list(DATASETS.keys())

    quick = args.quick or not args.full

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Cache datasets
    loaded_datasets: dict[str, tuple[SegmentStore, list[dict]]] = {}
    for ds_name in ds_names:
        store, all_qs = load_dataset(ds_name)
        if quick:
            qs = quick_test_questions(ds_name, all_qs)
        else:
            qs = all_qs
        loaded_datasets[ds_name] = (store, qs)
        print(f"Loaded {ds_name}: {len(qs)} questions, {len(store.segments)} segments")

    all_summaries: dict = {}
    for variant_name in variant_names:
        if variant_name not in VARIANTS:
            print(f"Unknown variant: {variant_name}")
            continue
        variant_summaries: dict = {}
        for ds_name in ds_names:
            store, qs = loaded_datasets[ds_name]
            results, summary, by_cat = run_variant_on_dataset(
                variant_name, ds_name, store, qs, quick
            )

            label = "quick" if quick else "full"
            out_path = RESULTS_DIR / f"mech_{variant_name}_{ds_name}_{label}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "variant": variant_name,
                        "dataset": ds_name,
                        "quick": quick,
                        "summary": summary,
                        "category_breakdown": by_cat,
                        "results": results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            print(f"  Saved: {out_path}")
            variant_summaries[ds_name] = summary

        all_summaries[variant_name] = {
            "summaries": variant_summaries,
            "vs_v2f": compare_to_v2f(variant_summaries),
        }

    # Aggregate file
    label = "quick" if quick else "full"
    summary_path = RESULTS_DIR / f"mech_summary_{label}.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    # Final table
    print("\n" + "=" * 110)
    print(f"MECHANICAL ENHANCEMENTS vs V2F ({label})")
    print("=" * 110)
    header = (
        f"{'Variant':<32s} {'Dataset':<14s} "
        f"{'r@20':>7s} {'v2f@20':>8s} {'vs@20':>8s} "
        f"{'r@50':>7s} {'v2f@50':>8s} {'vs@50':>8s}"
    )
    print(header)
    print("-" * len(header))
    for vname, data in all_summaries.items():
        for ds_name in ds_names:
            if ds_name not in data["summaries"]:
                continue
            s = data["summaries"][ds_name]
            ref = V2F_REFERENCE.get(ds_name, {})
            vs20 = s["arch_r@20"] - ref.get("arch_r@20", 0)
            vs50 = s["arch_r@50"] - ref.get("arch_r@50", 0)
            print(
                f"{vname:<32s} {ds_name:<14s} "
                f"{s['arch_r@20']:>7.3f} {ref.get('arch_r@20', 0):>8.3f} "
                f"{vs20:>+8.3f} "
                f"{s['arch_r@50']:>7.3f} {ref.get('arch_r@50', 0):>8.3f} "
                f"{vs50:>+8.3f}"
            )


if __name__ == "__main__":
    main()
