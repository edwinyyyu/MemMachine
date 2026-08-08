"""STEP 0 GUARDRAIL: reproduce gist->gist same-macro LOO P@1 (must be ~0.65)."""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

FR = "/Users/eyu/edwinyyyu/mmcc/feedback_retrieval"
labels = {l["idx"]: l for l in json.load(open(f"{FR}/labels.json"))}
macros = {m["idx"]: m for m in json.load(open(f"{FR}/exp1_macro_assignments.json"))}
idxs = sorted(macros.keys())
assert len(idxs) == 41, len(idxs)

gists = [labels[i]["gist"] for i in idxs]
gold = [macros[i]["macro"] for i in idxs]

model = SentenceTransformer("google/embeddinggemma-300m", device="mps")
emb = model.encode(gists, prompt_name="document", normalize_embeddings=True,
                   convert_to_numpy=True)
emb = emb.astype(np.float32)

# cosine sim (normalized -> dot)
S = emb @ emb.T
np.fill_diagonal(S, -np.inf)
nn = S.argmax(axis=1)
n = len(idxs)
p_at_1 = np.mean([gold[i] == gold[nn[i]] for i in range(n)])

# chance: avg over items of (same-macro among other 40)/40
from collections import Counter
cnt = Counter(gold)
chance = np.mean([(cnt[gold[i]] - 1) / (n - 1) for i in range(n)])

print(f"STEP0 gist->gist LOO P@1 = {p_at_1:.4f}  (chance {chance:.4f})  n={n}")
in_range = 0.60 <= p_at_1 <= 0.70
print(f"IN RANGE [0.60,0.70]: {in_range}")
json.dump({"step0_p_at_1": float(p_at_1), "chance": float(chance),
           "in_range": bool(in_range), "n": n},
          open("/Users/eyu/edwinyyyu/mmcc/temporal_scoring/exp_c_step0_result.json", "w"),
          indent=1)
