"""STEP 1 Exp C: pre-feedback query gist -> stored post-feedback gist, same-macro LOO P@1.

Query keys = MY (Claude's) pre-feedback gists built ONLY from request+action.
Corpus = the 41 stored post-feedback gists (labels.json).
For each query i, NN among the 41 stored gists EXCLUDING index i; score same-macro.
"""
import json
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

FR = "/Users/eyu/edwinyyyu/mmcc/feedback_retrieval"
TS = "/Users/eyu/edwinyyyu/mmcc/temporal_scoring"
labels = {l["idx"]: l for l in json.load(open(f"{FR}/labels.json"))}
macros = {m["idx"]: m for m in json.load(open(f"{FR}/exp1_macro_assignments.json"))}
pre = json.load(open(f"{TS}/exp_c_prefeedback_gists.json"))

idxs = sorted(macros.keys())
assert set(idxs) == set(int(k) for k in pre.keys()), "idx mismatch pre vs macros"

stored_gists = [labels[i]["gist"] for i in idxs]
query_gists = [pre[str(i)] for i in idxs]
gold = [macros[i]["macro"] for i in idxs]
n = len(idxs)

model = SentenceTransformer("google/embeddinggemma-300m", device="mps")
emb_store = model.encode(stored_gists, prompt_name="document",
                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
emb_query = model.encode(query_gists, prompt_name="document",
                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

# query[i] vs all stored[j], exclude j==i
S = emb_query @ emb_store.T  # (n_query, n_store)
np.fill_diagonal(S, -np.inf)
nn = S.argmax(axis=1)
hits = [gold[i] == gold[nn[i]] for i in range(n)]
p_at_1 = float(np.mean(hits))

# chance: avg over items of (same-macro among other 40)/40
cnt = Counter(gold)
chance = float(np.mean([(cnt[gold[i]] - 1) / (n - 1) for i in range(n)]))

# per-macro breakdown
per = {}
for m in sorted(set(gold)):
    members = [i for i in range(n) if gold[i] == m]
    per[m] = {
        "n": len(members),
        "p_at_1": float(np.mean([hits[i] for i in members])),
        "chance": float(np.mean([(cnt[gold[i]] - 1) / (n - 1) for i in members])),
    }

result = {
    "exp_c_p_at_1": p_at_1,
    "chance": chance,
    "n": n,
    "per_macro": per,
    "per_item": [
        {"idx": idxs[i], "gold_macro": gold[i],
         "nn_idx": idxs[nn[i]], "nn_macro": gold[nn[i]],
         "hit": bool(hits[i]), "sim": float(S[i, nn[i]])}
        for i in range(n)
    ],
}
json.dump(result, open(f"{TS}/exp_c_result.json", "w"), indent=1)

print(f"Exp C pre-feedback-gist->stored-gist LOO P@1 = {p_at_1:.4f}  (chance {chance:.4f})  n={n}")
print("\nPer-macro:")
for m, v in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
    print(f"  {m:28s} n={v['n']:2d}  P@1={v['p_at_1']:.3f}  chance={v['chance']:.3f}")
