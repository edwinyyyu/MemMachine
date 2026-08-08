"""
Structural-feature pre-feedback retrieval key.
Tests whether a key computed from (request + ACTION) ONLY -- never the correction --
can beat the 0.244 pre-feedback gist floor, especially for did_more_than_asked.

READ-ONLY on all inputs. Writes nothing except via the caller's report step.
"""
import json, re
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE = "/Users/eyu/edwinyyyu/mmcc/feedback_retrieval"
TS = "/Users/eyu/edwinyyyu/mmcc/temporal_scoring"

macro_assign = json.load(open(f"{BASE}/exp1_macro_assignments.json"))
inputs = json.load(open(f"{TS}/exp_c_prefeedback_inputs.json"))
v2 = json.load(open(f"{BASE}/v2_tags.json"))["tags"]

idxs = [m["idx"] for m in macro_assign]
gold = {m["idx"]: m["macro"] for m in macro_assign}
macros = [gold[i] for i in idxs]
n = len(idxs)

# ---------- chance (LOO) ----------
cnt = Counter(macros)
chance = float(np.mean([(cnt[gold[i]] - 1) / (n - 1) for i in idxs]))
# per-macro chance: for an item of class c, chance = (count[c]-1)/(n-1)
per_macro_chance = {c: (cnt[c] - 1) / (n - 1) for c in cnt}

# ---------- V1 structural features (no LLM, deterministic) ----------
EXPANSION_MARKERS = ["also", "additionally", "in addition", "as well", "furthermore",
                     "for completeness", "while i was at it", "bonus", "along the way",
                     "worth flagging", "one more", "extra"]
HEDGE_MARKERS = ["framework", "architecture", "abstraction", "generic", "extensible",
                 "in general", "reusable", "general-purpose", "scalable", "flexible",
                 "principled"]
MINIMALITY_CUES = ["just", "simple", "only", "minimal", "exactly", "merely", "don't even"]

def count_code_blocks(s):
    # fenced ``` blocks come in pairs; also count tool-call lines (Bash/Read/Edit/Write{...})
    fences = s.count("```")
    return fences // 2

def count_paths(s):
    # file paths / module dotted refs
    paths = len(re.findall(r"/[\w./-]+\.\w+", s))           # /a/b/c.py
    paths += len(re.findall(r"\b[\w]+\.(?:py|rs|md|json|yml|yaml|toml|ini)\b", s))  # foo.py
    return paths

def count_components(s):
    # approximate distinct "components": headings, def/class, tool calls, bullets, table rows
    comps = 0
    comps += len(re.findall(r"^#+\s", s, re.M))                  # markdown headings
    comps += len(re.findall(r"\bdef \w+|\bclass \w+|\bfn \w+", s))  # python/rust defs
    comps += len(re.findall(r"\b(?:Bash|Read|Edit|Write|Grep|Glob|ToolSearch|WebFetch|Agent) \{", s))  # tool calls
    comps += len(re.findall(r"^\s*[-*]\s", s, re.M))            # bullets
    comps += len(re.findall(r"^\s*\d+\.\s", s, re.M))           # numbered
    comps += len(re.findall(r"^\|.*\|", s, re.M))               # table rows
    return comps

def count_markers(s, markers):
    sl = s.lower()
    return sum(sl.count(m) for m in markers)

def has_minimality_cue(s):
    sl = s.lower()
    return int(any(re.search(r"\b" + re.escape(c) + r"\b", sl) for c in MINIMALITY_CUES))

def v1_features(req, act):
    f = {}
    f["act_tok_len"] = len(act.split())
    f["act_char_len"] = len(act)
    f["n_code_blocks"] = count_code_blocks(act)
    f["n_paths"] = count_paths(act)
    f["n_components"] = count_components(act)
    f["n_expansion_markers"] = count_markers(act, EXPANSION_MARKERS)
    f["n_hedge_markers"] = count_markers(act, HEDGE_MARKERS)
    req_tok = max(1, len(req.split()))
    f["act_to_req_len_ratio"] = len(act.split()) / req_tok
    # request-side
    f["req_tok_len"] = len(req.split())
    f["req_has_minimality_cue"] = has_minimality_cue(req)
    return f

V1_NAMES = ["act_tok_len", "act_char_len", "n_code_blocks", "n_paths", "n_components",
            "n_expansion_markers", "n_hedge_markers", "act_to_req_len_ratio",
            "req_tok_len", "req_has_minimality_cue"]
V2_NAMES = ["self_chosen_design", "introduces_new_abstraction", "does_unrequested_work",
            "makes_unverified_claim", "action_much_longer_than_request"]

# build matrices
X1, X2 = [], []
for i in idxs:
    rec = inputs[str(i)]
    f = v1_features(rec["request"], rec["action"])
    X1.append([f[k] for k in V1_NAMES])
    t = v2[str(i)]
    X2.append([t[k] for k in V2_NAMES])
X1 = np.array(X1, float)
X2 = np.array(X2, float)
y = np.array(macros)
y_dmta = (y == "did_more_than_asked").astype(int)

# ---------- TEST 1: LOO nearest-neighbor P@1 ----------
def loo_p1(X, standardize=True):
    """LOO NN by Euclidean over feature vector; returns overall P@1 and per-macro hits."""
    hits = np.zeros(n, int)
    for i in range(n):
        train_mask = np.arange(n) != i
        Xtr = X[train_mask]
        if standardize:
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            Xtr_z = (Xtr - mu) / sd
            xq = (X[i] - mu) / sd
        else:
            Xtr_z, xq = Xtr, X[i]
        d = np.sqrt(((Xtr_z - xq) ** 2).sum(1))
        # break ties deterministically by original order
        nn_local = int(np.argmin(d))
        train_idx = np.where(train_mask)[0]
        nn = train_idx[nn_local]
        hits[i] = int(y[nn] == y[i])
    overall = hits.mean()
    per = {}
    for c in cnt:
        mask = (y == c)
        per[c] = (float(hits[mask].mean()), int(mask.sum()))
    return float(overall), per, hits

p1_v1, per_v1, hits_v1 = loo_p1(X1, standardize=True)   # z-score V1
p1_v2, per_v2, hits_v2 = loo_p1(X2, standardize=False)  # booleans as 0/1 (Hamming==Euclidean here)

# ---------- TEST 2: LOO logistic regression AUC for did_more_than_asked ----------
def loo_logreg_auc(X, ylab, standardize=True):
    """LOO logistic regression; collect held-out predicted prob; AUC over all."""
    probs = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        Xtr, ytr = X[tr], ylab[tr]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr_s = sc.transform(Xtr)
            xq = sc.transform(X[i:i+1])
        else:
            Xtr_s, xq = Xtr, X[i:i+1]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr_s, ytr)
        probs[i] = clf.predict_proba(xq)[0, 1]
    return float(roc_auc_score(ylab, probs)), probs

auc_v1, _ = loo_logreg_auc(X1, y_dmta, standardize=True)
auc_v2, _ = loo_logreg_auc(X2, y_dmta, standardize=False)

# ---------- per-macro P@1 of did_more_than_asked specifically ----------
dmta_p1_v1 = per_v1["did_more_than_asked"][0]
dmta_p1_v2 = per_v2["did_more_than_asked"][0]
dmta_chance = per_macro_chance["did_more_than_asked"]

out = {
    "n": n,
    "chance_overall": chance,
    "baselines": {"chance": 0.171, "raw_to_gist": 0.20, "prefeedback_gist": 0.244, "oracle_gist": 0.65},
    "V1": {"p1_overall": p1_v1, "per_macro": {c: {"p1": v[0], "n": v[1], "chance": per_macro_chance[c]} for c, v in per_v1.items()},
           "dmta_auc": auc_v1},
    "V2": {"p1_overall": p1_v2, "per_macro": {c: {"p1": v[0], "n": v[1], "chance": per_macro_chance[c]} for c, v in per_v2.items()},
           "dmta_auc": auc_v2},
    "dmta": {"chance": dmta_chance, "prefeedback_gist": 0.429,
             "V1_p1": dmta_p1_v1, "V2_p1": dmta_p1_v2, "V1_auc": auc_v1, "V2_auc": auc_v2},
}
print(json.dumps(out, indent=2))

# also print V2 tag prevalence (sanity)
print("\nV2 tag prevalence:")
for j, nm in enumerate(V2_NAMES):
    print(f"  {nm}: {int(X2[:,j].sum())}/{n}")
print("\nV2 tag mean by class (did_more_than_asked vs rest):")
dm = X2[y_dmta == 1].mean(0); rest = X2[y_dmta == 0].mean(0)
for j, nm in enumerate(V2_NAMES):
    print(f"  {nm}: dmta={dm[j]:.2f} rest={rest[j]:.2f}")

json.dump(out, open(f"{BASE}/results_structural_key.json", "w"), indent=2)
print("\nwrote results_structural_key.json")
