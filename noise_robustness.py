"""Noise-robustness of Bayesian concept learning for preference convergence.

Question: real users answer probes inconsistently. The current learner
(bayesian_concept.py) zeros a hypothesis on ANY contradiction (hard-zero
likelihood), so one flipped answer can kill the true rule. We test whether a
soft Bernoulli likelihood (and redundant probing) fixes this.

Reuses from feedback_retrieval/simplicity:
  - latents + 5 hypotheses + simplicity prior + size-principle likelihood
  - hypothesis-averaging prediction + info-gain probe selection
all mirrored from temporal_scoring/bayesian_concept.py. Labels are RECOMPUTED
from latents via the true rule H_and (verified to match ground_truth exactly).

Noise model: each REVEALED answer (2 corrections + every probe answer) is
flipped with prob eps (Bernoulli, i.i.d.). Held-out labels used for scoring are
ALWAYS clean (the learner never sees them).

Likelihood variants:
  1. hard-zero  : P(answers|h)=0 if h misclassifies ANY revealed answer (current).
  2. soft       : P(answer|h) = (1-eps_a) if h agrees, eps_a if not; product.
  3. soft+redund: soft likelihood, each boundary probed R=3 times; repeats count
                  against the same probe budget (cost-fair).

READ-ONLY data. Writes only new result files.
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

DOMAIN = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/simplicity")
RESULT_MD = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_noise_robustness.md")
RESULT_JSON = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_noise_robustness.json")

# ---------------------------------------------------------------- load (read-only)
gt = json.loads((DOMAIN / "ground_truth.json").read_text())
train = json.loads((DOMAIN / "train_labels.json").read_text())

LATENT = {c["id"]: c["latent"] for c in gt["cases"]}
HELDOUT_IDS = list(train["heldout_ids"])
TRAIN_IDS = list(train["train_labels"].keys())
POOL_IDS = TRAIN_IDS + HELDOUT_IDS

# ---------------------------------------------------------------- hypothesis space (same 5)
HYPOTHESES = {
    "H_greenfield": lambda L: (not L["preexisting"]),
    "H_ownership":  lambda L: (L["solely_mine"]),
    "H_and":        lambda L: ((not L["preexisting"]) and L["solely_mine"]),   # TRUE rule
    "H_or":         lambda L: ((not L["preexisting"]) or L["solely_mine"]),
    "H_personal":   lambda L: (L["work_vs_personal"] == "personal"),
}
LITERALS = {"H_greenfield": 1, "H_ownership": 1, "H_personal": 1, "H_and": 2, "H_or": 2}
H_NAMES = list(HYPOTHESES.keys())
TRUE_H = "H_and"


def predicts_mystyle(h, cid):
    return HYPOTHESES[h](LATENT[cid])


# clean oracle label recomputed from latent via the TRUE rule (H_and)
def clean_label(cid):
    return "my_style" if HYPOTHESES[TRUE_H](LATENT[cid]) else "established"


# ---------------------------------------------------------------- prior + size principle (same)
def make_prior():
    w = {h: 2.0 ** (-LITERALS[h]) for h in H_NAMES}
    z = sum(w.values())
    return {h: w[h] / z for h in H_NAMES}


PRIOR = make_prior()
EXT = {h: sum(1 for cid in POOL_IDS if predicts_mystyle(h, cid)) / len(POOL_IDS) for h in H_NAMES}

# held-out clean labels for scoring ONLY
TRUE = {cid: clean_label(cid) for cid in POOL_IDS}


# ---------------------------------------------------------------- noise
def flip(label):
    return "established" if label == "my_style" else "my_style"


def noisy_answer(cid, eps, rng):
    """Reveal a (possibly flipped) answer for case cid at noise eps."""
    lab = clean_label(cid)
    return flip(lab) if rng.random() < eps else lab


# ---------------------------------------------------------------- posteriors (3 variants)
# `obs` is a list of (cid, observed_label) -- one entry PER revealed answer (repeats allowed).

def posterior_hard(obs):
    """Variant 1: hard-zero with size-principle likelihood (mirrors bayesian_concept.py).
    Any single misclassified revealed answer -> likelihood 0 for that h."""
    post = {}
    for h in H_NAMES:
        consistent = all((predicts_mystyle(h, cid)) == (lab == "my_style") for cid, lab in obs)
        if len(obs) == 0:
            like = 1.0
        elif not consistent:
            like = 0.0
        else:
            ext = EXT[h]
            like = (1.0 / ext) ** len(obs) if ext > 0 else 0.0
        post[h] = PRIOR[h] * like
    z = sum(post.values())
    if z == 0:
        return dict(PRIOR), True  # degenerate: all-inconsistent -> fall back to prior
    return {h: post[h] / z for h in H_NAMES}, False


def posterior_soft(obs, eps_a):
    """Variant 2/3: soft Bernoulli likelihood. P(answer|h)=(1-eps_a) agree, eps_a disagree.
    Never zeros. eps_a is the learner's ASSUMED noise rate (floored away from 0)."""
    # floor assumed eps so soft never degenerates to hard-zero / never gives log(0)
    e = min(max(eps_a, 1e-3), 0.49)
    log_post = {}
    for h in H_NAMES:
        lp = math.log(PRIOR[h])
        for cid, lab in obs:
            agree = (predicts_mystyle(h, cid)) == (lab == "my_style")
            lp += math.log(1.0 - e) if agree else math.log(e)
        log_post[h] = lp
    m = max(log_post.values())
    w = {h: math.exp(log_post[h] - m) for h in H_NAMES}
    z = sum(w.values())
    return {h: w[h] / z for h in H_NAMES}, False


# ---------------------------------------------------------------- prediction / scoring
def predict_prob(post, cid):
    return sum(post[h] for h in H_NAMES if predicts_mystyle(h, cid))


def score_heldout(post):
    correct = 0
    for cid in HELDOUT_IDS:
        pred = "my_style" if predict_prob(post, cid) >= 0.5 else "established"
        if pred == TRUE[cid]:
            correct += 1
    return correct / len(HELDOUT_IDS)


def info_gain_score(post, cid):
    p = predict_prob(post, cid)
    return p * (1.0 - p)


# ---------------------------------------------------------------- run loops
# Corrections: the 2 seed corrections c01, c03 (both clean my_style). Under noise
# these can also flip -- they are "revealed answers" too.
CORRECTIONS = ["c01", "c03"]


def make_posterior_fn(variant, eps_a):
    if variant == "hard":
        return lambda obs: posterior_hard(obs)
    return lambda obs: posterior_soft(obs, eps_a)


def run_single(variant, eps, seed, n_probes=8, redundant_R=1, eps_assumed=None):
    """One run. Returns per-step history + flags.
    variant in {hard, soft}. redundant_R>1 -> probe each distinct boundary R times,
    each repeat is an independent noisy answer and counts against the probe budget.
    For soft, the learner's assumed noise = eps_assumed (defaults to true eps =
    matched). Passing a fixed eps_assumed models a learner that does NOT know the
    true noise rate. hard ignores assumed eps.
    """
    rng = random.Random(seed)
    post_fn = make_posterior_fn(variant, eps if eps_assumed is None else eps_assumed)

    # seed observations = the 2 corrections, each possibly flipped
    obs = [(cid, noisy_answer(cid, eps, rng)) for cid in CORRECTIONS]

    history = []
    n_asked = 0  # number of probe ANSWERS consumed (budget unit)
    distinct_probed = set()

    def snapshot():
        post, degen = post_fn(obs)
        return {
            "n_probes": n_asked,
            "acc": score_heldout(post),
            "post": dict(post),
            "top": max(post, key=post.get),
            "max_post": post[max(post, key=post.get)],
            "p_true": post[TRUE_H],
            "degenerate": degen,
            # "true rule zeroed/abandoned": its mass is ~0 (effectively killed)
            "true_zeroed": post[TRUE_H] < 1e-9,
            # abandoned = not the MAP hypothesis
            "true_not_map": max(post, key=post.get) != TRUE_H,
        }

    history.append(snapshot())

    while n_asked < n_probes:
        post, _ = post_fn(obs)
        # choose next DISTINCT boundary by info-gain among unprobed train cases
        candidates = [cid for cid in TRAIN_IDS
                      if cid not in distinct_probed and cid not in CORRECTIONS]
        if not candidates:
            break
        boundary = max(candidates, key=lambda c: info_gain_score(post, c))
        distinct_probed.add(boundary)
        # ask this boundary R times (or fewer if budget runs out), each a fresh noisy answer
        reps = min(redundant_R, n_probes - n_asked)
        for _ in range(reps):
            obs.append((boundary, noisy_answer(boundary, eps, rng)))
            n_asked += 1
        history.append(snapshot())

    # pad history so all runs have a comparable final row at budget
    final = history[-1]
    return {
        "history": history,
        "final": final,
        "n_distinct": len(distinct_probed),
        "n_answers": n_asked,
    }


# ---------------------------------------------------------------- experiment grid
EPS_LIST = [0.0, 0.1, 0.2, 0.3]
N_SEEDS = 30          # >= 10 required; 30 for tighter CIs
SEEDS = list(range(N_SEEDS))
N_PROBES = 8

VARIANTS = [
    ("hard",            "hard", 1),   # hard-zero, single probe each (current)
    ("soft",            "soft", 1),   # soft Bernoulli, single probe each
    ("soft_redundant",  "soft", 3),   # soft Bernoulli, R=3 per boundary, budget-capped
]


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def stderr(xs):
    return stdev(xs) / math.sqrt(len(xs)) if xs else 0.0


results = {}  # results[label][eps] = aggregate dict
for label, variant, R in VARIANTS:
    results[label] = {}
    for eps in EPS_LIST:
        runs = [run_single(variant, eps, s, n_probes=N_PROBES, redundant_R=R) for s in SEEDS]
        finals = [r["final"] for r in runs]
        accs = [f["acc"] for f in finals]
        p_trues = [f["p_true"] for f in finals]
        # "true rule zeroed/abandoned AT ANY POINT during the run"
        ever_zeroed = [any(h["true_zeroed"] for h in r["history"]) for r in runs]
        ever_not_map = [any(h["true_not_map"] for h in r["history"][1:]) for r in runs]  # after first probe-step
        final_zeroed = [f["true_zeroed"] for f in finals]
        final_is_map = [f["top"] == TRUE_H for f in finals]
        degen_ever = [any(h["degenerate"] for h in r["history"]) for r in runs]
        results[label][eps] = {
            "acc_mean": mean(accs), "acc_se": stderr(accs),
            "acc_min": min(accs), "acc_max": max(accs),
            "p_true_mean": mean(p_trues), "p_true_se": stderr(p_trues),
            "ever_zeroed_rate": mean([1.0 if x else 0.0 for x in ever_zeroed]),
            "ever_not_map_rate": mean([1.0 if x else 0.0 for x in ever_not_map]),
            "final_zeroed_rate": mean([1.0 if x else 0.0 for x in final_zeroed]),
            "final_is_map_rate": mean([1.0 if x else 0.0 for x in final_is_map]),
            "degen_ever_rate": mean([1.0 if x else 0.0 for x in degen_ever]),
            "n_distinct_mean": mean([r["n_distinct"] for r in runs]),
            "n_answers_mean": mean([r["n_answers"] for r in runs]),
        }

# ---------------------------------------------------------------- majority-class baseline
# Trivial "always predict the held-out majority class" floor, for interpreting acc.
_maj_lab = Counter(TRUE[c] for c in HELDOUT_IDS).most_common(1)[0][0]
MAJORITY_ACC = sum(1 for c in HELDOUT_IDS if TRUE[c] == _maj_lab) / len(HELDOUT_IDS)

# ---------------------------------------------------------------- mismatched assumed-eps (realistic: learner does NOT know eps)
# Soft + redundant, but the learner ALWAYS assumes eps_a=0.1 regardless of true eps.
MIS_EPS_A = 0.1
mismatched = {}
for eps in EPS_LIST:
    runs = [run_single("soft", eps, s, n_probes=N_PROBES, redundant_R=3, eps_assumed=MIS_EPS_A)
            for s in SEEDS]
    finals = [r["final"] for r in runs]
    accs = [f["acc"] for f in finals]
    mismatched[eps] = {
        "acc_mean": mean(accs), "acc_se": stderr(accs),
        "p_true_mean": mean([f["p_true"] for f in finals]),
        "final_is_map_rate": mean([1.0 if f["top"] == TRUE_H else 0.0 for f in finals]),
    }

# ---------------------------------------------------------------- probes-to-converge (extra-probe cost of noise)
# For soft (single-probe) variant: how many probe answers until posterior puts
# >=0.90 on H_and, averaged over seeds, at each eps. NaN if not reached within a
# larger budget. Compares cost at eps=0 vs eps=0.1, 0.2.
CONV_THRESH = 0.90
CONV_BUDGET = 20  # allow more probes to measure convergence cost


def probes_to_converge(variant, eps, seed, R=1, thresh=CONV_THRESH, budget=CONV_BUDGET):
    r = run_single(variant, eps, seed, n_probes=budget, redundant_R=R)
    for h in r["history"]:
        if h["p_true"] >= thresh:
            return h["n_probes"]
    return None  # never reached


conv = {}
for label, variant, R in VARIANTS:
    conv[label] = {}
    for eps in EPS_LIST:
        vals = [probes_to_converge(variant, eps, s, R=R) for s in SEEDS]
        reached = [v for v in vals if v is not None]
        conv[label][eps] = {
            "reach_rate": len(reached) / len(vals),
            "probes_mean": mean(reached) if reached else None,
            "probes_se": stderr(reached) if len(reached) > 1 else 0.0,
        }

# ---------------------------------------------------------------- console (measure, don't narrate)
print("=== SETUP ===")
print(f"  hypotheses: {H_NAMES} (true={TRUE_H})")
print(f"  heldout n={len(HELDOUT_IDS)}  train n={len(TRAIN_IDS)}  pool={len(POOL_IDS)}")
print(f"  prior: " + ", ".join(f"{h}={PRIOR[h]:.3f}" for h in H_NAMES))
print(f"  seeds={N_SEEDS}  probe budget={N_PROBES}  eps={EPS_LIST}")
print(f"  corrections (seed reveals): {CORRECTIONS} clean={[clean_label(c) for c in CORRECTIONS]}")

print("\n=== HELD-OUT ACCURACY vs eps (after probe budget=%d) ===" % N_PROBES)
hdr = "  variant            | " + " ".join(f"eps={e:<4}" for e in EPS_LIST)
print(hdr)
for label, _, _ in VARIANTS:
    row = "  %-18s |" % label
    for eps in EPS_LIST:
        a = results[label][eps]
        row += f" {a['acc_mean']:.2f}±{a['acc_se']:.2f}"
    print(row)

print("\n=== P(H_and) in final posterior vs eps ===")
print(hdr)
for label, _, _ in VARIANTS:
    row = "  %-18s |" % label
    for eps in EPS_LIST:
        a = results[label][eps]
        row += f"  {a['p_true_mean']:.2f}    "
    print(row)

print("\n=== TRUE-RULE ZEROED RATE (H_and mass -> ~0 at any point) hard vs soft ===")
print(hdr)
for label, _, _ in VARIANTS:
    row = "  %-18s |" % label
    for eps in EPS_LIST:
        row += f"  {results[label][eps]['ever_zeroed_rate']:.2f}    "
    print(row)

print("\n=== TRUE-RULE final-MAP RATE (final posterior's top hypothesis is H_and) ===")
print(hdr)
for label, _, _ in VARIANTS:
    row = "  %-18s |" % label
    for eps in EPS_LIST:
        row += f"  {results[label][eps]['final_is_map_rate']:.2f}    "
    print(row)

print("\n=== DEGENERATE (all-inconsistent fallback to prior) rate ===")
print(hdr)
for label, _, _ in VARIANTS:
    row = "  %-18s |" % label
    for eps in EPS_LIST:
        row += f"  {results[label][eps]['degen_ever_rate']:.2f}    "
    print(row)

print("\n=== PROBES TO CONVERGE (P(H_and)>=%.2f), mean over seeds [reach-rate] ===" % CONV_THRESH)
print(hdr)
for label, _, _ in VARIANTS:
    row = "  %-18s |" % label
    for eps in EPS_LIST:
        c = conv[label][eps]
        pm = f"{c['probes_mean']:.1f}" if c["probes_mean"] is not None else "  -"
        row += f" {pm}[{c['reach_rate']:.2f}]"
    print(row)

print("\n  (redundant variant uses %d distinct boundaries on avg at budget=%d)" %
      (round(results['soft_redundant'][0.1]['n_distinct_mean']), N_PROBES))

print(f"\n=== MAJORITY-CLASS FLOOR (always predict '{_maj_lab}') = {MAJORITY_ACC:.2f} ===")
print("=== MISMATCHED assumed-eps (soft+redundant, learner always assumes eps_a=0.1) ===")
print(hdr)
row = "  acc                |"
for eps in EPS_LIST:
    row += f" {mismatched[eps]['acc_mean']:.2f}±{mismatched[eps]['acc_se']:.2f}"
print(row)
row = "  final_is_map(H_and)|"
for eps in EPS_LIST:
    row += f"  {mismatched[eps]['final_is_map_rate']:.2f}    "
print(row)

# ---------------------------------------------------------------- write JSON
RESULT_JSON.write_text(json.dumps({
    "setup": {
        "hypotheses": H_NAMES, "true": TRUE_H, "prior": PRIOR, "ext": EXT,
        "heldout_n": len(HELDOUT_IDS), "train_n": len(TRAIN_IDS),
        "seeds": N_SEEDS, "probe_budget": N_PROBES, "eps_list": EPS_LIST,
        "corrections": CORRECTIONS, "conv_thresh": CONV_THRESH, "conv_budget": CONV_BUDGET,
        "redundant_R": 3,
    },
    "results": {l: {str(e): results[l][e] for e in EPS_LIST} for l, _, _ in VARIANTS},
    "convergence": {l: {str(e): conv[l][e] for e in EPS_LIST} for l, _, _ in VARIANTS},
    "majority_floor": {"label": _maj_lab, "acc": MAJORITY_ACC},
    "mismatched_assumed_eps": {"eps_a": MIS_EPS_A,
                               "by_eps": {str(e): mismatched[e] for e in EPS_LIST}},
}, indent=2))
print(f"\nWrote {RESULT_JSON}")
