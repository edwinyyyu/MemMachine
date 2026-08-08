"""Multi-route normalization control.

Question: how much of multi-route retrieval's recall@10 lift (0.889 vs raw
0.748 / single-gist 0.754) is just MORE KEYS (shots-on-goal, buyable without an
LLM) vs genuinely-better LLM framings? Give RAW the same multi-key and
query-gist advantages and compare.

6 conditions, recall@10/@1, N=ALL pool (gold + all distractors), 3 seeds.
For an item with multiple keys, item score = MAX cosine over its keys.

  1. raw               q=situation,  item keys={text}            (guardrail ~0.748)
  2. raw+qgist         q=query_gist, item keys={text}
  3. gist              q=query_gist, item keys={gist}            (~0.754)
  4. multi-route       q=query_gist, item keys={gist}+alt_gists  (guardrail ~0.889)
  5. raw-multikey      q=situation,  item keys=sentence chunks of text   (no LLM)
  6. raw-multikey+qgist q=query_gist, item keys=sentence chunks of text  (fairest #key-norm comparison to 4)

Encoding convention (matches existing fr_embed.py / run_retrieval.py):
  - item-side keys (text, text chunks, gist, alt_gists) -> document prompt
  - query-side reps (situation, query_gist)             -> query prompt
  - embeddinggemma-300m, mps, L2-normalized -> cosine = dot.

Embeddings cached to mr_norm_cache.npz (the expensive step).
"""
import json
import os
import re
import hashlib
import numpy as np
from collections import defaultdict, Counter

SYNTH = "/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/synth"
NORM = os.path.join(SYNTH, "normalized_corpus.json")
KEYED_FILES = ["keyed_facts.json", "keyed_procedures.json",
               "keyed_stance.json", "keyed_conventions.json"]
CACHE = "/Users/eyu/edwinyyyu/mmcc/temporal_scoring/mr_norm_cache.npz"
OUT_JSON = "/Users/eyu/edwinyyyu/mmcc/temporal_scoring/mr_norm_results.json"

SEEDS = [0, 1, 2]
KS = (1, 10)

# ---- sentence chunker: split on sentence boundaries, guard common abbrevs ----
from nltk.tokenize import sent_tokenize
_ABBR = ["e.g.", "i.e.", "etc.", "vs.", "approx.", "Fig.", "No.", "cf.", "al.",
         "Dr.", "Mr.", "Ms.", "Mrs.", "Inc.", "Ltd.", "Co.", "U.S.", "a.k.a.",
         "Jr.", "Sr.", "St.", "Eq.", "resp."]


def _protect(t):
    for a in _ABBR:
        t = t.replace(a, a.replace(".", "․"))  # one-dot leader sentinel
    return t


def chunk_text(text):
    """Sentence-boundary chunks of text. If <2 chunks, returns 1 key (whole)."""
    t = _protect(text.strip())
    sents = [s.replace("․", ".").strip() for s in sent_tokenize(t)]
    sents = [s for s in sents if s]
    return sents if sents else [text.strip()]


# =========================================================================
# Load + merge by id
# =========================================================================
def load():
    nc = json.load(open(NORM))
    items, queries = nc["items"], nc["queries"]
    keyed_item, keyed_q = {}, {}
    for f in KEYED_FILES:
        d = json.load(open(os.path.join(SYNTH, f)))
        for it in d["items"]:
            keyed_item[it["id"]] = it
        for q in d["queries"]:
            keyed_q[q["id"]] = q
    # merge
    for it in items:
        k = keyed_item[it["id"]]
        it["gist"] = k["gist"]
        it["alt_gists"] = list(k.get("alt_gists", []))
        it["chunks"] = chunk_text(it["text"])
    for q in queries:
        q["query_gist"] = keyed_q[q["id"]]["query_gist"]
    return items, queries


# =========================================================================
# Embedding (cached). Build a global string pool, embed once with the right
# prompt, look up by (prompt, text) hash.
# =========================================================================
def build_string_pool(items, queries):
    doc_strs, qry_strs = set(), set()
    for it in items:
        doc_strs.add(it["text"])
        doc_strs.add(it["gist"])
        for g in it["alt_gists"]:
            doc_strs.add(g)
        for c in it["chunks"]:
            doc_strs.add(c)
    for q in queries:
        qry_strs.add(q["situation"])
        qry_strs.add(q["query_gist"])
    return sorted(doc_strs), sorted(qry_strs)


def _key(prompt, s):
    return prompt + "\x00" + s


def get_embeddings(items, queries):
    doc_strs, qry_strs = build_string_pool(items, queries)
    sig = hashlib.sha1(
        ("|".join(doc_strs) + "##" + "|".join(qry_strs)).encode()
    ).hexdigest()
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if str(z["sig"]) == sig:
            emb = {}
            keys = list(z["keys"])
            vecs = z["vecs"]
            for i, k in enumerate(keys):
                emb[k] = vecs[i]
            print(f"loaded {len(emb)} cached embeddings")
            return emb
        print("cache signature mismatch -> re-embedding")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("google/embeddinggemma-300m", device="mps")
    print(f"embedding {len(doc_strs)} document strings, {len(qry_strs)} query strings...")
    doc_vecs = model.encode(doc_strs, prompt_name="document",
                            normalize_embeddings=True, batch_size=128,
                            show_progress_bar=True).astype(np.float32)
    qry_vecs = model.encode(qry_strs, prompt_name="query",
                            normalize_embeddings=True, batch_size=128,
                            show_progress_bar=True).astype(np.float32)
    keys, vecs = [], []
    for s, v in zip(doc_strs, doc_vecs):
        keys.append(_key("document", s)); vecs.append(v)
    for s, v in zip(qry_strs, qry_vecs):
        keys.append(_key("query", s)); vecs.append(v)
    vecs = np.array(vecs, dtype=np.float32)
    np.savez(CACHE, sig=sig, keys=np.array(keys, dtype=object), vecs=vecs)
    print(f"cached {len(keys)} embeddings -> {CACHE}")
    return {k: v for k, v in zip(keys, vecs)}


# =========================================================================
# Conditions
# =========================================================================
def item_keys_for_condition(it, cond):
    """Return list of (document-prompt) key strings for this item."""
    if cond in ("raw", "raw+qgist"):
        return [it["text"]]
    if cond == "gist":
        return [it["gist"]]
    if cond == "multi-route":
        return [it["gist"]] + it["alt_gists"]
    if cond in ("raw-multikey", "raw-multikey+qgist"):
        return it["chunks"]
    raise ValueError(cond)


def query_str_for_condition(q, cond):
    if cond in ("raw", "raw-multikey"):
        return ("query", q["situation"])
    return ("query", q["query_gist"])  # qgist conditions + gist + multi-route


CONDS = ["raw", "raw+qgist", "gist", "multi-route",
         "raw-multikey", "raw-multikey+qgist"]


def main():
    items, queries = load()
    n_items = len(items)
    id_to_idx = {it["id"]: i for i, it in enumerate(items)}
    emb = get_embeddings(items, queries)

    def vec(prompt, s):
        return emb[_key(prompt, s)]

    results = {"meta": {}, "conditions": {}, "avg_keys_per_item": {}}
    results["meta"] = {
        "n_items": n_items,
        "n_queries": len(queries),
        "seeds": SEEDS,
        "pool": "N=ALL (gold + all 1891 distractors)",
        "item_score": "MAX cosine over item keys",
        "items_by_content_type": dict(Counter(it["content_type"] for it in items)),
        "queries_by_content_type": dict(Counter(q["content_type"] for q in queries)),
        "encoding": "embeddinggemma-300m mps normalized; item keys=document prompt, query reps=query prompt",
        "chunker": "nltk sent_tokenize with abbreviation guard; <2 sentences -> 1 key",
    }

    content_types = sorted(set(q["content_type"] for q in queries))

    for cond in CONDS:
        # ---- build item-key embedding matrix (flat) ----
        flat_vecs = []
        flat_owner = []
        keys_per_item = np.zeros(n_items, dtype=int)
        for ii, it in enumerate(items):
            ks = item_keys_for_condition(it, cond)
            keys_per_item[ii] = len(ks)
            for kstr in ks:
                flat_vecs.append(vec("document", kstr))
                flat_owner.append(ii)
        flat_vecs = np.array(flat_vecs, dtype=np.float32)  # (num_keys, 768)
        flat_owner = np.array(flat_owner)

        # avg #keys per item, overall + by content_type
        akpi = {"overall": float(keys_per_item.mean())}
        # also: distribution + how many items have exactly 1 key (for multikey RECORD)
        n_single = int(np.sum(keys_per_item == 1))
        ct_keys = defaultdict(list)
        for ii, it in enumerate(items):
            ct_keys[it["content_type"]].append(int(keys_per_item[ii]))
        for ct in content_types:
            arr = np.array(ct_keys[ct])
            akpi[ct] = float(arr.mean())
        akpi["_n_items_single_key"] = n_single
        akpi["_pct_items_single_key"] = round(100 * n_single / n_items, 1)
        results["avg_keys_per_item"][cond] = akpi

        # ---- query reps ----
        q_vecs = np.array([vec(*query_str_for_condition(q, cond)) for q in queries],
                          dtype=np.float32)  # (Q, 768)

        # ---- score: for each query, item_score = MAX over its keys ----
        # sims_flat (Q, num_keys); reduce-max by owner.
        sims_flat = q_vecs @ flat_vecs.T  # (Q, num_keys)
        item_scores = np.full((len(queries), n_items), -np.inf, dtype=np.float32)
        # np.maximum.at over the item axis per query
        for ii_owner in range(0):
            pass
        # vectorized reduce: use np.maximum.at on a (Q, n_items) target
        np.maximum.at(item_scores, (np.arange(len(queries))[:, None],
                                    flat_owner[None, :]), sims_flat)

        # ---- gold rank in N=ALL pool (deterministic; seeds irrelevant for ALL,
        # but we run 3 seeds for identical protocol -> identical numbers) ----
        gold_idx = np.array([id_to_idx[q["gold_id"]] for q in queries])
        gold_scores = item_scores[np.arange(len(queries)), gold_idx]
        # rank = 1 + count of items strictly greater
        ranks = (item_scores > gold_scores[:, None]).sum(axis=1) + 1  # (Q,)

        # recall@k, averaged over seeds (deterministic -> all seeds equal)
        seed_rec = {k: [] for k in KS}
        for _seed in SEEDS:
            for k in KS:
                seed_rec[k].append(float(np.mean(ranks <= k)))
        overall = {f"recall@{k}": float(np.mean(seed_rec[k])) for k in KS}
        overall["recall@k_seed_std"] = {
            f"recall@{k}": float(np.std(seed_rec[k])) for k in KS
        }

        # by content_type
        by_ct = {}
        qct = np.array([q["content_type"] for q in queries])
        for ct in content_types:
            mask = qct == ct
            by_ct[ct] = {"n": int(mask.sum())}
            for k in KS:
                by_ct[ct][f"recall@{k}"] = float(np.mean(ranks[mask] <= k))

        results["conditions"][cond] = {
            "overall": overall,
            "by_content_type": by_ct,
            "avg_keys_overall": akpi["overall"],
        }
        print(f"[{cond:22s}] r@10={overall['recall@10']:.4f} "
              f"r@1={overall['recall@1']:.4f}  avg_keys={akpi['overall']:.2f}")

    # ---- derived comparisons ----
    r10 = {c: results["conditions"][c]["overall"]["recall@10"] for c in CONDS}
    r1 = {c: results["conditions"][c]["overall"]["recall@1"] for c in CONDS}
    short_cts = ["declarative_fact", "stance_feedback"]
    long_cts = ["procedure", "convention_or_principle"]

    def grouped_recall(cond, cts, k=10):
        # micro-average over queries in the listed content types
        qct = np.array([q["content_type"] for q in queries])
        # recompute ranks would need re-scoring; instead aggregate from by_ct with n weights
        by = results["conditions"][cond]["by_content_type"]
        num = sum(by[ct][f"recall@{k}"] * by[ct]["n"] for ct in cts)
        den = sum(by[ct]["n"] for ct in cts)
        return num / den if den else float("nan")

    comparisons = {
        "guardrail_raw_recall@10": r10["raw"],
        "guardrail_multiroute_recall@10": r10["multi-route"],
        "guardrail_gist_recall@10": r10["gist"],
        "4_vs_6_gap_overall_recall@10": r10["multi-route"] - r10["raw-multikey+qgist"],
        "4_vs_6_gap_short_facts_stance": (
            grouped_recall("multi-route", short_cts) - grouped_recall("raw-multikey+qgist", short_cts)
        ),
        "4_vs_6_gap_long_proc_conv": (
            grouped_recall("multi-route", long_cts) - grouped_recall("raw-multikey+qgist", long_cts)
        ),
        "1_vs_5_shots_on_goal_recall@10": r10["raw-multikey"] - r10["raw"],
        "1_vs_2_qgist_benefit_on_raw_recall@10": r10["raw+qgist"] - r10["raw"],
        "2_vs_3_raw-qgist_vs_gist_recall@10": r10["gist"] - r10["raw+qgist"],
        "multiroute_total_lift_over_raw": r10["multi-route"] - r10["raw"],
        "shots_on_goal_recoverable_via_raw-multikey+qgist": r10["raw-multikey+qgist"] - r10["raw"],
        "grouped_short_long": {
            "multi-route_short": grouped_recall("multi-route", short_cts),
            "raw-multikey+qgist_short": grouped_recall("raw-multikey+qgist", short_cts),
            "multi-route_long": grouped_recall("multi-route", long_cts),
            "raw-multikey+qgist_long": grouped_recall("raw-multikey+qgist", long_cts),
        },
    }
    results["comparisons"] = comparisons

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_JSON}")
    print("\n=== KEY COMPARISONS (recall@10) ===")
    for kk, vv in comparisons.items():
        if isinstance(vv, dict):
            continue
        print(f"  {kk:48s} {vv:+.4f}" if "gap" in kk or "vs" in kk or "lift" in kk or "recoverable" in kk
              else f"  {kk:48s} {vv:.4f}")


if __name__ == "__main__":
    main()
