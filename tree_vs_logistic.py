"""
Controller comparison for preference convergence.

Compares three online controllers that try to converge an apply/not-apply
('thorough' / 'minimal') decision to a hidden user preference defined over
LATENTS, while only ever seeing blind LLM-extracted FEATURES:

  1. logistic     - SGD log-loss over one-hot features, LABEL-ONLY updates (baseline).
  2. greedy_list  - LABEL-ONLY decision rules: on each mistake, greedily add the
                    single feature-predicate split that best fixes recent errors.
  3. elicited_list- LABEL + REASON + scope: on each mistake the ORACLE reveals the
                    determining latent(s); map latent->feature; PREPEND an ordered
                    exception rule scoped to that region. First-match-wins.

READ-ONLY inputs; writes only result JSON. Pure stdlib + math (no numpy needed).
"""

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CONV = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/convergence")
OUT_JSON = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_tree_vs_logistic.json")

SEEDS = list(range(12))          # >= 8 seeds
TEST_FRAC = 0.20
LR = 0.15                        # matches the prior smooth-controller logistic LR
GREEDY_WINDOW = None             # None => use ALL labels seen so far for greedy split scoring
RECENT_K = 8                     # also report a windowed greedy variant? keep simple: use all.

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
gt = {s["id"]: s["latent"] for s in json.load(open(CONV / "ground_truth.json"))["situations"]}
ft = {s["id"]: s["features"] for s in json.load(open(CONV / "features.json"))["situations"]}
IDS = list(gt.keys())
N = len(IDS)

# ----------------------------------------------------------------------------
# Oracles (over LATENTS). label True == 'thorough', False == 'minimal'.
#   R0 = is_production OR safety_or_money OR data_external   (same as prior work)
#   R1 = is_production OR safety_or_money OR audience_expert (drift: swaps the
#        data_external disjunct, which maps cleanly, for audience_expert, which
#        has NO clean blind feature -> exercises the FEATURE-GAP mechanism).
# ----------------------------------------------------------------------------
ORACLES = {
    "R0": ["is_production", "safety_or_money", "data_external"],
    "R1": ["is_production", "safety_or_money", "audience_expert"],
}

def make_oracle(rule_name):
    disjuncts = ORACLES[rule_name]
    label = {i: bool(any(gt[i][d] for d in disjuncts)) for i in IDS}

    def reason(i):
        """(label, determining_latents). True -> satisfied disjuncts;
        False -> all disjuncts fail (default-minimal region)."""
        if label[i]:
            return True, [d for d in disjuncts if gt[i][d]]
        return False, []

    return label, reason, disjuncts

# ----------------------------------------------------------------------------
# Latent -> feature correspondence map (for elicited_list rule construction).
# Each latent maps to a feature PREDICATE expressed over blind features, or None
# (a FEATURE-GAP -> rule cannot be expressed precisely).
# Predicate = function(features_dict) -> bool, plus a human-readable string.
# ----------------------------------------------------------------------------
def pred_production(f): return bool(f["production_bound"])
def pred_stakes(f):     return bool(f["stakes_high"])
def pred_external(f):   return bool(f["external_facing"] or f["handles_untrusted_input"])

LATENT_TO_FEATURE = {
    "is_production":   ("production_bound == true",                      pred_production),
    "safety_or_money": ("stakes_high == true",                          pred_stakes),
    "data_external":   ("external_facing OR handles_untrusted_input",   pred_external),
    # The following are NOT in R0 but listed for the correspondence map / gap audit:
    # audience_expert -> NO clean feature (audience_expertise=='expert' agrees only ~56%); FEATURE-GAP.
    "audience_expert": (None, None),
    # time_pressure -> urgency=='high' agrees ~90%; usable but imperfect.
    "time_pressure":   ("urgency == high",                              lambda f: f["urgency"] == "high"),
}

# ----------------------------------------------------------------------------
# Feature encoding for the logistic baseline: one-hot.
# ----------------------------------------------------------------------------
BOOL_FEATS = ["stakes_high", "production_bound", "external_facing",
              "handles_untrusted_input", "wants_explanation", "disposable"]
CAT_FEATS = {
    "audience_expertise": ["novice", "mid", "expert"],
    "urgency": ["none", "low", "high"],
    "reversibility": ["easy", "hard"],
    "complexity": ["low", "mid", "high"],
}

def encode(i):
    f = ft[i]
    x = [1.0]  # bias
    for b in BOOL_FEATS:
        x.append(1.0 if f[b] else 0.0)
    for c, vals in CAT_FEATS.items():
        for v in vals:
            x.append(1.0 if f[c] == v else 0.0)
    return x

DIM = len(encode(IDS[0]))
FEATNAMES = ["bias"] + BOOL_FEATS + [f"{c}={v}" for c, vals in CAT_FEATS.items() for v in vals]

# ----------------------------------------------------------------------------
# Candidate atomic predicates for the GREEDY list (label-only rule learner).
# Each predicate: (name, fn(features)->bool). Greedy picks the predicate+output
# that best fixes recent errors.
# ----------------------------------------------------------------------------
def build_predicates():
    preds = []
    for b in BOOL_FEATS:
        preds.append((f"{b}==true", (lambda bb: (lambda f: bool(f[bb])))(b)))
        preds.append((f"{b}==false", (lambda bb: (lambda f: not f[bb]))(b)))
    for c, vals in CAT_FEATS.items():
        for v in vals:
            preds.append((f"{c}=={v}", (lambda cc, vv: (lambda f: f[cc] == vv))(c, v)))
    return preds

PREDICATES = build_predicates()

# ----------------------------------------------------------------------------
# Controllers. Each exposes:
#   update(i, y)      -> incorporate label y for situation i (online)
#   predict(i)        -> bool decision
# ----------------------------------------------------------------------------
class Logistic:
    name = "logistic"
    def __init__(self, lr=LR):
        self.w = [0.0] * DIM
        self.lr = lr
    def _score(self, i):
        x = encode(i)
        return sum(wj * xj for wj, xj in zip(self.w, x))
    def predict(self, i):
        return self._score(i) >= 0.0
    def update(self, i, y):
        x = encode(i)
        z = sum(wj * xj for wj, xj in zip(self.w, x))
        p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
        g = p - (1.0 if y else 0.0)
        self.w = [wj - self.lr * g * xj for wj, xj in zip(self.w, x)]


class GreedyList:
    """Label-only rule list. On each MISTAKE, greedily prepend the single
    predicate split (+ output) that best reduces error over ALL labels seen so
    far. First-match-wins; default = majority of seen labels."""
    name = "greedy_list"
    def __init__(self):
        self.rules = []          # list of (pred_name, pred_fn, output_bool)
        self.seen = {}           # id -> y
        self.default = False
    def predict(self, i):
        f = ft[i]
        for (_, fn, out) in self.rules:
            if fn(f):
                return out
        return self.default
    def update(self, i, y):
        self.seen[i] = y
        # refresh default = majority of seen
        if self.seen:
            cnt = Counter(self.seen.values())
            self.default = cnt[True] >= cnt[False]
        if self.predict(i) == y:
            return  # no mistake, no rule added
        # mistake: pick the predicate+output that maximizes correctly-classified
        # seen examples under the NEW list (prepend candidate).
        best = None
        best_correct = -1
        for (pname, pfn) in PREDICATES:
            for out in (True, False):
                trial = [(pname, pfn, out)] + self.rules
                correct = 0
                for sid, sy in self.seen.items():
                    f = ft[sid]
                    pred = self.default
                    for (_, fn, o) in trial:
                        if fn(f):
                            pred = o
                            break
                    if pred == sy:
                        correct += 1
                if correct > best_correct:
                    best_correct = correct
                    best = (pname, pfn, out)
        if best is not None:
            self.rules = [best] + self.rules


class ElicitedList:
    """Label + REASON + scope. On each mistake, the ORACLE reveals the
    determining latent(s); map to feature predicate; PREPEND an ordered
    exception rule scoped to that region. First-match-wins. Records feature-gaps
    when the determining latent has no clean feature."""
    name = "elicited_list"
    def __init__(self):
        self.rules = []          # list of (rule_str, pred_fn, output_bool)
        self.rule_keys = set()   # dedupe by (pred_str, output)
        self.default = False     # default-minimal region
        self.feature_gaps = 0
        self.corrections = 0
    def predict(self, i):
        f = ft[i]
        for (_, fn, out) in self.rules:
            if fn(f):
                return out
        return self.default
    def update(self, i, y):
        if self.predict(i) == y:
            return
        self.corrections += 1
        label, det = ORACLE_REASON(i)
        if label:
            # satisfied disjunct(s): add a "thorough" exception for each determining
            # latent that maps to a feature. If none map -> feature-gap.
            mapped_any = False
            for lat in det:
                rstr, pfn = LATENT_TO_FEATURE.get(lat, (None, None))
                if pfn is None:
                    self.feature_gaps += 1
                    continue
                key = (rstr, True)
                if key not in self.rule_keys:
                    # prepend so most-recent reason wins; scoped to its region.
                    self.rules = [(f"IF {rstr} -> thorough", pfn, True)] + self.rules
                    self.rule_keys.add(key)
                mapped_any = True
            if not mapped_any and det:
                # all determining latents were feature-gaps
                pass
            elif not det:
                # shouldn't happen for label True, but guard
                pass
        else:
            # all disjuncts fail -> this region should be minimal. The reason is
            # "none of the active disjuncts' features present". Built from the
            # active oracle's feature-expressible predicates (feature-gap latents
            # can't contribute to this guard -> approximation).
            rstr, pfn = ACTIVE_MINIMAL_RULE
            key = (rstr, False)
            if key not in self.rule_keys:
                self.rules = [(f"IF {rstr}", pfn, False)] + self.rules
                self.rule_keys.add(key)


# ----------------------------------------------------------------------------
# Feature-sufficiency ceiling: batch logistic on ALL train labels (the best a
# feature-based controller can do). Reuse the same encoder. Gradient descent.
# ----------------------------------------------------------------------------
def feature_ceiling(train_ids, test_ids, epochs=400, lr=0.3):
    w = [0.0] * DIM
    X = {i: encode(i) for i in train_ids}
    for _ in range(epochs):
        grad = [0.0] * DIM
        for i in train_ids:
            x = X[i]
            z = sum(wj * xj for wj, xj in zip(w, x))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            g = p - (1.0 if LABEL[i] else 0.0)
            for k in range(DIM):
                grad[k] += g * x[k]
        m = len(train_ids)
        w = [wj - lr * gk / m for wj, gk in zip(w, grad)]
    correct = 0
    for i in test_ids:
        x = encode(i)
        pred = sum(wj * xj for wj, xj in zip(w, x)) >= 0.0
        if pred == LABEL[i]:
            correct += 1
    return correct / len(test_ids)


def accuracy(ctrl, test_ids):
    c = sum(1 for i in test_ids if ctrl.predict(i) == LABEL[i])
    return c / len(test_ids)


# ----------------------------------------------------------------------------
# Run one seed: shared random label stream over the train set; online updates;
# per-step held-out test accuracy + collateral flips.
# ----------------------------------------------------------------------------
def run_seed(seed):
    rng = random.Random(seed)
    ids = IDS[:]
    rng.shuffle(ids)
    n_test = int(round(N * TEST_FRAC))
    test_ids = ids[:n_test]
    train_ids = ids[n_test:]
    stream = train_ids[:]        # same random label order for all controllers
    rng.shuffle(stream)

    ceiling = feature_ceiling(train_ids, test_ids)
    majority = max(
        sum(LABEL[i] for i in train_ids) / len(train_ids),
        1 - sum(LABEL[i] for i in train_ids) / len(train_ids),
    )
    target = majority + 0.90 * (ceiling - majority)

    controllers = {
        "logistic": Logistic(),
        "greedy_list": GreedyList(),
        "elicited_list": ElicitedList(),
    }

    out = {name: {"acc": [], "collateral": [], "cum_collateral": []} for name in controllers}
    labels_to_target = {name: None for name in controllers}

    for step, i in enumerate(stream, start=1):
        y = LABEL[i]
        for name, ctrl in controllers.items():
            # snapshot test predictions BEFORE update (excluding corrected one,
            # which isn't in test set anyway since i is a train id).
            before = {t: ctrl.predict(t) for t in test_ids}
            ctrl.update(i, y)
            after = {t: ctrl.predict(t) for t in test_ids}
            flips = sum(1 for t in test_ids if before[t] != after[t])
            acc = sum(1 for t in test_ids if after[t] == LABEL[t]) / len(test_ids)
            out[name]["acc"].append(acc)
            out[name]["collateral"].append(flips)
            prev = out[name]["cum_collateral"][-1] if out[name]["cum_collateral"] else 0
            out[name]["cum_collateral"].append(prev + flips)
            if labels_to_target[name] is None and acc >= target:
                labels_to_target[name] = step

    # final legibility artifacts
    final = {
        "ceiling": ceiling,
        "majority": majority,
        "target": target,
        "labels_to_target": labels_to_target,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "feature_gaps": controllers["elicited_list"].feature_gaps,
        "elicited_corrections": controllers["elicited_list"].corrections,
        "elicited_rules": [r[0] for r in controllers["elicited_list"].rules],
        "greedy_rules": [(r[0], r[2]) for r in controllers["greedy_list"].rules],
        "logistic_w": list(zip(FEATNAMES, controllers["logistic"].w)),
        "final_acc": {name: out[name]["acc"][-1] for name in controllers},
        "total_collateral": {name: out[name]["cum_collateral"][-1] for name in controllers},
        "mean_collateral_per_update": {name: statistics.mean(out[name]["collateral"]) for name in controllers},
    }
    return out, final


# ----------------------------------------------------------------------------
# Aggregate over seeds for the currently-active oracle.
# ----------------------------------------------------------------------------
def aggregate(per_seed, rule_name):
    max_len = max(len(out["logistic"]["acc"]) for out, _ in per_seed)
    names = ["logistic", "greedy_list", "elicited_list"]

    # averaged accuracy & collateral curves (pad with last value)
    def avg_curve(key):
        curves = {n: [] for n in names}
        for step in range(max_len):
            for n in names:
                vals = []
                for out, _ in per_seed:
                    arr = out[n][key]
                    vals.append(arr[step] if step < len(arr) else arr[-1])
                curves[n].append(statistics.mean(vals))
        return curves

    acc_curves = avg_curve("acc")
    coll_curves = avg_curve("collateral")

    # labels-to-target on the AVERAGED curve + per-seed median
    avg_ceiling = statistics.mean(f["ceiling"] for _, f in per_seed)
    avg_majority = statistics.mean(f["majority"] for _, f in per_seed)
    avg_target = statistics.mean(f["target"] for _, f in per_seed)

    labels_to_target_avgcurve = {}
    for n in names:
        hit = None
        for step in range(max_len):
            if acc_curves[n][step] >= avg_target:
                hit = step + 1
                break
        labels_to_target_avgcurve[n] = hit

    labels_to_target_median = {}
    labels_to_target_mean = {}
    for n in names:
        vals = [f["labels_to_target"][n] for _, f in per_seed if f["labels_to_target"][n] is not None]
        nver = sum(1 for _, f in per_seed if f["labels_to_target"][n] is None)
        labels_to_target_median[n] = (statistics.median(vals) if vals else None, f"{nver} seeds never reached")
        labels_to_target_mean[n] = (statistics.mean(vals) if vals else None)

    # collateral aggregates
    mean_coll_per_update = {}
    total_coll = {}
    for n in names:
        mcs = [statistics.mean(out[n]["collateral"]) for out, _ in per_seed]
        tcs = [out[n]["cum_collateral"][-1] for out, _ in per_seed]
        mean_coll_per_update[n] = (statistics.mean(mcs), statistics.pstdev(mcs))
        total_coll[n] = (statistics.mean(tcs), statistics.pstdev(tcs))

    # feature-gap rate (elicited)
    gaps = sum(f["feature_gaps"] for _, f in per_seed)
    corrs = sum(f["elicited_corrections"] for _, f in per_seed)
    feature_gap_rate = gaps / corrs if corrs else 0.0

    # final acc
    final_acc = {n: (statistics.mean(f["final_acc"][n] for _, f in per_seed),
                     statistics.pstdev([f["final_acc"][n] for _, f in per_seed])) for n in names}

    # pick a representative seed (seed 0) for legibility printout
    _, rep = per_seed[0]

    result = {
        "oracle": rule_name,
        "ceiling": {"feature_sufficiency": avg_ceiling, "majority": avg_majority, "target_90pct": avg_target},
        "labels_to_target": {
            "avg_curve": labels_to_target_avgcurve,
            "per_seed_median": labels_to_target_median,
            "per_seed_mean": labels_to_target_mean,
        },
        "final_acc": final_acc,
        "collateral_per_update": mean_coll_per_update,
        "total_collateral": total_coll,
        "feature_gap": {"gaps": gaps, "corrections": corrs, "rate": feature_gap_rate},
        "correspondence_map": {
            "is_production": "production_bound (95.8% agree)",
            "safety_or_money": "stakes_high (97.5% agree)",
            "data_external": "external_facing OR handles_untrusted_input (96.7% agree)",
            "audience_expert": "NO CLEAN FEATURE -> audience_expertise=='expert' only 55.8% agree (FEATURE-GAP); audience_expertise!='novice' is the real signal",
            "time_pressure": "urgency=='high' (90.0% agree, imperfect)",
        },
        "acc_curves": acc_curves,
        "coll_curves": coll_curves,
        "representative_seed0": {
            "elicited_rules": rep["elicited_rules"],
            "greedy_rules": rep["greedy_rules"],
            "logistic_top_weights": sorted(rep["logistic_w"], key=lambda kv: -abs(kv[1]))[:10],
            "n_elicited_rules": len(rep["elicited_rules"]),
            "n_greedy_rules": len(rep["greedy_rules"]),
        },
        "avg_rule_count": {
            "elicited_list": statistics.mean(len(f["elicited_rules"]) for _, f in per_seed),
            "greedy_list": statistics.mean(len(f["greedy_rules"]) for _, f in per_seed),
        },
    }

    # console summary
    print(f"\n########## ORACLE {rule_name} ##########")
    print("=== CEILING ===")
    print(f"feature-sufficiency: {avg_ceiling:.3f}  majority: {avg_majority:.3f}  target(90% of ceiling): {avg_target:.3f}")
    print("\n=== LABELS-TO-TARGET ===")
    print("controller        avg-curve   per-seed-median   per-seed-mean")
    for n in names:
        med, note = labels_to_target_median[n]
        print(f"{n:16s}  {str(labels_to_target_avgcurve[n]):>9s}   {str(med):>8s} ({note})   {labels_to_target_mean[n]}")
    print("\n=== FINAL TEST ACC ===")
    for n in names:
        m, sd = final_acc[n]
        print(f"{n:16s}  {m:.3f} +/- {sd:.3f}")
    print("\n=== COLLATERAL (flips per update, mean over seeds) ===")
    for n in names:
        m, sd = mean_coll_per_update[n]
        tm, tsd = total_coll[n]
        print(f"{n:16s}  per-update {m:.3f} +/- {sd:.3f}   cumulative {tm:.1f} +/- {tsd:.1f}")
    print(f"\n=== FEATURE-GAP RATE (elicited) === {gaps}/{corrs} = {feature_gap_rate:.3f}")
    print(f"\navg rule count: elicited={result['avg_rule_count']['elicited_list']:.1f}  greedy={result['avg_rule_count']['greedy_list']:.1f}")
    print("\n=== REP SEED elicited rules ===")
    for r in rep["elicited_rules"]:
        print("  ", r)
    print("=== REP SEED logistic top weights ===")
    for fn, wv in sorted(rep["logistic_w"], key=lambda kv: -abs(kv[1]))[:8]:
        print(f"   {fn:30s} {wv:+.3f}")
    return result


# Active-oracle globals (set per-rule before running seeds).
LABEL = {}
ORACLE_REASON = None
ACTIVE_MINIMAL_RULE = None

def set_oracle(rule_name):
    global LABEL, ORACLE_REASON, ACTIVE_MINIMAL_RULE
    label, reason, disjuncts = make_oracle(rule_name)
    LABEL = label
    ORACLE_REASON = reason
    # Build the minimal-region guard from the FEATURE-EXPRESSIBLE disjuncts only.
    parts, fns = [], []
    for d in disjuncts:
        rstr, pfn = LATENT_TO_FEATURE.get(d, (None, None))
        if pfn is not None:
            parts.append(rstr)
            fns.append(pfn)
    guard_str = "NOT(" + " OR ".join(parts) + ") -> minimal"
    def guard(f, _fns=tuple(fns)):
        return not any(fn(f) for fn in _fns)
    ACTIVE_MINIMAL_RULE = (guard_str, guard)


def main():
    results = {"config": {"seeds": SEEDS, "n": N, "test_frac": TEST_FRAC, "lr": LR,
                          "oracles": {k: " OR ".join(v) for k, v in ORACLES.items()}},
               "correspondence_map": {
                   "is_production": "production_bound (95.8% agree)",
                   "safety_or_money": "stakes_high (97.5% agree)",
                   "data_external": "external_facing OR handles_untrusted_input (96.7% agree)",
                   "audience_expert": "NO CLEAN FEATURE -> audience_expertise=='expert' only 55.8% agree (FEATURE-GAP); real signal is audience_expertise!='novice'",
                   "time_pressure": "urgency=='high' (90.0% agree, imperfect)",
               },
               "by_oracle": {}}
    for rule_name in ORACLES:
        set_oracle(rule_name)
        per_seed = [run_seed(s) for s in SEEDS]
        results["by_oracle"][rule_name] = aggregate(per_seed, rule_name)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
