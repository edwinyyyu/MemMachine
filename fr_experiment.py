"""raw->raw recall@k scaling experiment.

Setup:
- Embeddings precomputed (fr_embeddings.npz), cosine == dot (normalized).
- For each query, the candidate pool = gold item + (N-1) random distractors
  sampled (without replacement) from all OTHER items. Rank by cosine.
- recall@k = fraction of queries whose gold is within top-k.
- Average over 3 seeds. Chance recall@k = k/N.

Breakdowns at N=ALL (full pool, deterministic — gold always present, all items
are candidates so no sampling needed): by content_type, difficulty, and gold
cluster-density.
"""
import json
import os
import numpy as np
from collections import defaultdict, Counter

SYNTH = "/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/synth"
NORM = os.path.join(SYNTH, "normalized_corpus.json")
EMB = "/Users/eyu/edwinyyyu/mmcc/temporal_scoring/fr_embeddings.npz"
OUT_JSON = "/Users/eyu/edwinyyyu/mmcc/temporal_scoring/fr_results.json"

KS = [1, 5, 10, 20, 50]
NS = [200, 500, 1000, "ALL"]
SEEDS = [0, 1, 2]


def load():
    with open(NORM) as f:
        d = json.load(f)
    items, queries = d["items"], d["queries"]
    z = np.load(EMB, allow_pickle=True)
    item_emb, query_emb = z["item_emb"], z["query_emb"]
    item_ids = list(z["item_ids"])
    query_ids = list(z["query_ids"])
    # sanity: order matches normalized file
    assert item_ids == [it["id"] for it in items]
    assert query_ids == [q["id"] for q in queries]
    return items, queries, item_emb, query_emb


def main():
    items, queries, item_emb, query_emb = load()
    n_items = len(items)
    id_to_idx = {it["id"]: i for i, it in enumerate(items)}
    cluster_size = Counter(it["cluster"] for it in items)  # by (content_type-shared) cluster label

    # per-query metadata
    qmeta = []
    for q in queries:
        gi = id_to_idx[q["gold_id"]]
        qmeta.append({
            "qidx": None,  # set below
            "gold_idx": gi,
            "content_type": q["content_type"],
            "difficulty": q["difficulty"],
            "gold_cluster": items[gi]["cluster"],
        })
    for i, m in enumerate(qmeta):
        m["qidx"] = i

    # Precompute full similarity matrix: queries x items (cosine == dot)
    sims = query_emb @ item_emb.T  # (Q, n_items)

    # ---- helper: rank of gold within a pool ----
    def gold_rank_in_pool(qidx, pool_idx):
        """pool_idx: array of item indices (includes gold). Return 1-based rank
        of gold by descending cosine (ties: count strictly-greater + 1)."""
        gi = qmeta[qidx]["gold_idx"]
        s = sims[qidx, pool_idx]
        gold_sim = sims[qidx, gi]
        # rank = 1 + number of pool items with strictly greater sim
        return int(np.sum(s > gold_sim)) + 1

    # =========================================================
    # 1. Scaling table: recall@k vs N (avg over seeds)
    # =========================================================
    all_idx = np.arange(n_items)
    scaling = {}  # N -> {k -> recall}
    for N in NS:
        if N == "ALL":
            pool_n = n_items
        else:
            pool_n = N
        # accumulate per-query "gold rank" averaged over seeds, then recall@k
        # For efficiency, store rank per (seed, query).
        ranks = np.zeros((len(SEEDS), len(queries)), dtype=int)
        for si, seed in enumerate(SEEDS):
            if N == "ALL":
                # deterministic: whole set is the pool
                for m in qmeta:
                    ranks[si, m["qidx"]] = gold_rank_in_pool(m["qidx"], all_idx)
            else:
                rng = np.random.default_rng(seed)
                for m in qmeta:
                    qidx = m["qidx"]
                    gi = m["gold_idx"]
                    # sample N-1 distractors from all items except gold
                    others = np.concatenate([all_idx[:gi], all_idx[gi + 1:]])
                    distract = rng.choice(others, size=pool_n - 1, replace=False)
                    pool = np.concatenate([[gi], distract])
                    ranks[si, qidx] = gold_rank_in_pool(qidx, pool)
        # recall@k = mean over (seed,query) of (rank<=k)
        scaling[str(N)] = {}
        for k in KS:
            if k >= pool_n:
                continue
            rec = float(np.mean(ranks <= k))
            scaling[str(N)][str(k)] = rec
    # also store ALL-pool ranks for breakdowns (deterministic, seed-independent)
    all_ranks = np.array([gold_rank_in_pool(m["qidx"], all_idx) for m in qmeta])

    # =========================================================
    # 2. Breakdowns at N=ALL, k in {10, 20}
    # =========================================================
    def recall_by(group_fn, ks=(10, 20)):
        groups = defaultdict(list)
        for m in qmeta:
            groups[group_fn(m)].append(all_ranks[m["qidx"]])
        out = {}
        for g, rks in groups.items():
            rks = np.array(rks)
            out[str(g)] = {"n": int(len(rks))}
            for k in ks:
                out[str(g)][f"recall@{k}"] = float(np.mean(rks <= k))
        return out

    by_content = recall_by(lambda m: m["content_type"])
    by_difficulty = recall_by(lambda m: m["difficulty"])

    def density_bucket(m):
        sz = cluster_size[m["gold_cluster"]]
        if sz <= 2:
            return "1-2"
        if sz <= 6:
            return "3-6"
        return "7+"
    by_density = recall_by(density_bucket)

    # overall at ALL
    overall_all = {}
    for k in KS:
        overall_all[f"recall@{k}"] = float(np.mean(all_ranks <= k))

    results = {
        "meta": {
            "n_items": n_items,
            "n_queries": len(queries),
            "ks": KS,
            "ns": [str(x) for x in NS],
            "seeds": SEEDS,
            "items_by_content_type": dict(Counter(it["content_type"] for it in items)),
            "queries_by_content_type": dict(Counter(q["content_type"] for q in queries)),
            "queries_by_difficulty": dict(Counter(q["difficulty"] for q in queries)),
            "cluster_size_summary": {
                "min": min(cluster_size.values()),
                "max": max(cluster_size.values()),
                "n_clusters": len(cluster_size),
            },
        },
        "scaling": scaling,
        "chance": {str(N): {str(k): (k / (n_items if N == "ALL" else N))
                            for k in KS if k < (n_items if N == "ALL" else N)}
                   for N in NS},
        "overall_at_ALL": overall_all,
        "breakdown_by_content_type_ALL": by_content,
        "breakdown_by_difficulty_ALL": by_difficulty,
        "breakdown_by_cluster_density_ALL": by_density,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary ----
    print("=== SCALING recall@k vs N (avg over 3 seeds) ===")
    hdr = "N".ljust(6) + "".join(f"k={k}".rjust(9) for k in KS)
    print(hdr)
    for N in NS:
        row = str(N).ljust(6)
        for k in KS:
            v = scaling[str(N)].get(str(k))
            row += ("-" if v is None else f"{v:.3f}").rjust(9)
        print(row)
    print("\n=== CHANCE recall@k = k/N ===")
    print(hdr)
    for N in NS:
        pn = n_items if N == "ALL" else N
        row = str(N).ljust(6)
        for k in KS:
            row += ("-" if k >= pn else f"{k/pn:.3f}").rjust(9)
        print(row)

    print("\n=== overall @ ALL ===", {k: round(v, 3) for k, v in overall_all.items()})
    print("\n=== by content_type (N=ALL) ===")
    for g, v in sorted(by_content.items()):
        print(f"  {g:28s} n={v['n']:3d}  r@10={v['recall@10']:.3f}  r@20={v['recall@20']:.3f}")
    print("=== by difficulty (N=ALL) ===")
    for g in ["NEAR", "MED", "FAR"]:
        if g in by_difficulty:
            v = by_difficulty[g]
            print(f"  {g:6s} n={v['n']:3d}  r@10={v['recall@10']:.3f}  r@20={v['recall@20']:.3f}")
    print("=== by cluster density (N=ALL) ===")
    for g in ["1-2", "3-6", "7+"]:
        if g in by_density:
            v = by_density[g]
            print(f"  {g:5s} n={v['n']:3d}  r@10={v['recall@10']:.3f}  r@20={v['recall@20']:.3f}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
