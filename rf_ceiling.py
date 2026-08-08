"""
Random-forest accuracy reference for the controller comparison.

Extends `tree_vs_logistic.py` by adding two RandomForest controllers on the
EXACT same setup (same data, same oracles R0/R1, same one-hot encoder, same 12
seeds, same 80/20 splits, same shared online label stream) so RF lands on the
same axes as logistic / greedy_list / elicited_list:

  1. rf_online  - RandomForestClassifier retrained on the accumulated labeled
                  set after EACH new label (same online stream as the others).
                  Reports labels-to-90%-of-ceiling and collateral-change/update.
  2. rf_batch   - RandomForest trained on ALL train labels -> max test accuracy
                  (the RF ceiling), compared to the logistic-all-labels ceiling
                  and the true-latent reference.

Everything that defines the experiment is imported from the prior module, NOT
re-implemented, so the two studies are guaranteed identical except for the added
controllers. The prior controllers' reference numbers are loaded from
results_tree_vs_logistic.json (re-run upstream if missing).

READ-ONLY on data inputs; writes only new result files.
"""

import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Reuse the EXACT prior setup. Importing runs no experiment (guarded by __main__).
import tree_vs_logistic as base
from tree_vs_logistic import (
    IDS, N, TEST_FRAC, SEEDS, encode, DIM, FEATNAMES,
    feature_ceiling, set_oracle, ORACLES,
)
import random

CONV = base.CONV
PRIOR_JSON = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_tree_vs_logistic.json")
OUT_JSON = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_rf_ceiling.json")
OUT_MD = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_rf_ceiling.md")

# RF hyperparameters: fixed, modest, seeded for determinism. Same encoder/input
# space as logistic so the comparison is on identical axes.
RF_KW = dict(n_estimators=200, max_depth=None, n_jobs=1)

# ----------------------------------------------------------------------------
# Precompute the one-hot design matrix once (the SAME encoder the logistic uses).
# ----------------------------------------------------------------------------
X_ALL = {i: np.asarray(encode(i), dtype=float) for i in IDS}

# True-latent design matrix (one-hot over the *latents*) for the true-latent
# reference: the best an oracle that could SEE the latents would do. Since every
# oracle here is an exact Boolean function of latents, this is 1.0 by
# construction; we still fit/evaluate it empirically to report it honestly.
gt = {s["id"]: s["latent"] for s in json.load(open(CONV / "ground_truth.json"))["situations"]}
LATENT_BOOLS = ["is_production", "data_external", "safety_or_money",
                "audience_expert", "time_pressure"]

def encode_latent(i):
    l = gt[i]
    return np.asarray([1.0] + [1.0 if l[b] else 0.0 for b in LATENT_BOOLS], dtype=float)

XL_ALL = {i: encode_latent(i) for i in IDS}


# ----------------------------------------------------------------------------
# rf_online controller: same online interface as the other controllers
# (update(i, y), predict(i)). Retrains a fresh RF on the accumulated labeled set
# after each label. Falls back to majority/constant before a 2nd class is seen
# (RF cannot fit a single-class set meaningfully). Deterministic per seed.
# ----------------------------------------------------------------------------
class RFOnline:
    name = "rf_online"
    def __init__(self, seed):
        self.seed = seed
        self.seen = {}          # id -> y
        self.clf = None
        self.const = False      # used until >=2 classes present
    def update(self, i, y):
        self.seen[i] = bool(y)
        classes = set(self.seen.values())
        if len(classes) < 2:
            # single-class: predict that class everywhere (matches "majority")
            self.const = next(iter(classes))
            self.clf = None
            return
        Xtr = np.vstack([X_ALL[s] for s in self.seen])
        ytr = np.asarray([self.seen[s] for s in self.seen], dtype=int)
        clf = RandomForestClassifier(random_state=self.seed, **RF_KW)
        clf.fit(Xtr, ytr)
        self.clf = clf
    def predict(self, i):
        if self.clf is None:
            return self.const
        return bool(self.clf.predict(X_ALL[i].reshape(1, -1))[0])
    def predict_many(self, ids):
        if self.clf is None:
            return {t: self.const for t in ids}
        X = np.vstack([X_ALL[t] for t in ids])
        pred = self.clf.predict(X)
        return {t: bool(p) for t, p in zip(ids, pred)}


# ----------------------------------------------------------------------------
# rf_batch ceiling: RF on ALL train labels -> test accuracy.
# ----------------------------------------------------------------------------
def rf_batch_ceiling(train_ids, test_ids, label, seed, X=X_ALL):
    Xtr = np.vstack([X[i] for i in train_ids])
    ytr = np.asarray([1 if label[i] else 0 for i in train_ids], dtype=int)
    clf = RandomForestClassifier(random_state=seed, **RF_KW)
    clf.fit(Xtr, ytr)
    Xte = np.vstack([X[i] for i in test_ids])
    pred = clf.predict(Xte)
    correct = sum(1 for p, t in zip(pred, test_ids) if bool(p) == label[t])
    return correct / len(test_ids), clf


def latent_reference(train_ids, test_ids, label, seed):
    """True-latent reference: classifier with access to the LATENTS. Exact by
    construction for a Boolean oracle; reported empirically."""
    acc, _ = rf_batch_ceiling(train_ids, test_ids, label, seed, X=XL_ALL)
    return acc


# ----------------------------------------------------------------------------
# One seed: reproduce the EXACT split + stream from the prior run_seed, then run
# the two RF controllers on it. (Same rng calls in the same order => identical
# test_ids/train_ids/stream as logistic/greedy/elicited saw.)
# ----------------------------------------------------------------------------
def split_and_stream(seed):
    rng = random.Random(seed)
    ids = IDS[:]
    rng.shuffle(ids)
    n_test = int(round(N * TEST_FRAC))
    test_ids = ids[:n_test]
    train_ids = ids[n_test:]
    stream = train_ids[:]
    rng.shuffle(stream)
    return test_ids, train_ids, stream


def run_seed(seed, label):
    test_ids, train_ids, stream = split_and_stream(seed)

    # ceilings on this split (identical feature_ceiling fn as prior => same axes)
    logistic_ceiling = feature_ceiling(train_ids, test_ids)   # uses base.LABEL (set by set_oracle)
    rf_ceiling, rf_clf = rf_batch_ceiling(train_ids, test_ids, label, seed)
    latent_ref = latent_reference(train_ids, test_ids, label, seed)

    majority = max(
        sum(label[i] for i in train_ids) / len(train_ids),
        1 - sum(label[i] for i in train_ids) / len(train_ids),
    )
    # target consistent with prior study: 90% of the way from majority to the
    # FEATURE-SUFFICIENCY (logistic) ceiling, so labels-to-90% is on the SAME
    # axis as the other controllers.
    target = majority + 0.90 * (logistic_ceiling - majority)

    rf = RFOnline(seed)
    acc_curve, coll_curve = [], []
    cum = 0
    labels_to_target = None
    for step, i in enumerate(stream, start=1):
        y = label[i]
        before = rf.predict_many(test_ids)
        rf.update(i, y)
        after = rf.predict_many(test_ids)
        flips = sum(1 for t in test_ids if before[t] != after[t])
        acc = sum(1 for t in test_ids if after[t] == label[t]) / len(test_ids)
        acc_curve.append(acc)
        coll_curve.append(flips)
        cum += flips
        if labels_to_target is None and acc >= target:
            labels_to_target = step

    importances = sorted(zip(FEATNAMES, rf_clf.feature_importances_),
                         key=lambda kv: -kv[1])

    return {
        "logistic_ceiling": logistic_ceiling,
        "rf_ceiling": rf_ceiling,
        "latent_ref": latent_ref,
        "majority": majority,
        "target": target,
        "acc_curve": acc_curve,
        "coll_curve": coll_curve,
        "cum_collateral": cum,
        "labels_to_target": labels_to_target,
        "final_acc": acc_curve[-1],
        "rf_importances": importances,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
    }


def aggregate(per_seed, rule_name, target_for_curve):
    max_len = max(len(s["acc_curve"]) for s in per_seed)

    def avg_curve(key):
        out = []
        for step in range(max_len):
            vals = [s[key][step] if step < len(s[key]) else s[key][-1] for s in per_seed]
            out.append(statistics.mean(vals))
        return out

    acc_curve = avg_curve("acc_curve")
    coll_curve = avg_curve("coll_curve")

    # labels-to-target on the averaged curve (matches prior aggregate())
    ltt_avgcurve = None
    for step in range(max_len):
        if acc_curve[step] >= target_for_curve:
            ltt_avgcurve = step + 1
            break

    ltt_vals = [s["labels_to_target"] for s in per_seed if s["labels_to_target"] is not None]
    never = sum(1 for s in per_seed if s["labels_to_target"] is None)
    ltt_median = (statistics.median(ltt_vals) if ltt_vals else None, f"{never} seeds never reached")
    ltt_mean = statistics.mean(ltt_vals) if ltt_vals else None

    mcs = [statistics.mean(s["coll_curve"]) for s in per_seed]
    tcs = [s["cum_collateral"] for s in per_seed]

    final_accs = [s["final_acc"] for s in per_seed]
    rf_ceils = [s["rf_ceiling"] for s in per_seed]
    log_ceils = [s["logistic_ceiling"] for s in per_seed]
    lat_refs = [s["latent_ref"] for s in per_seed]

    # aggregate RF feature importances: mean over seeds of the batch RF
    imp_acc = {f: [] for f in FEATNAMES}
    for s in per_seed:
        for f, v in s["rf_importances"]:
            imp_acc[f].append(v)
    mean_imp = sorted(((f, statistics.mean(vs)) for f, vs in imp_acc.items()),
                      key=lambda kv: -kv[1])

    return {
        "oracle": rule_name,
        "rf_batch_ceiling": (statistics.mean(rf_ceils), statistics.pstdev(rf_ceils)),
        "logistic_ceiling": (statistics.mean(log_ceils), statistics.pstdev(log_ceils)),
        "latent_reference": (statistics.mean(lat_refs), statistics.pstdev(lat_refs)),
        "rf_online_labels_to_target": {
            "avg_curve": ltt_avgcurve,
            "per_seed_median": ltt_median,
            "per_seed_mean": ltt_mean,
        },
        "rf_online_final_acc": (statistics.mean(final_accs), statistics.pstdev(final_accs)),
        "rf_online_collateral_per_update": (statistics.mean(mcs), statistics.pstdev(mcs)),
        "rf_online_total_collateral": (statistics.mean(tcs), statistics.pstdev(tcs)),
        "rf_batch_importances": mean_imp,
        "acc_curve": acc_curve,
        "coll_curve": coll_curve,
        "target_used": target_for_curve,
    }


def main():
    prior = json.load(open(PRIOR_JSON))
    out = {"config": prior["config"], "rf_kw": RF_KW, "by_oracle": {}, "prior_ref": {}}

    for rule_name in ORACLES:
        set_oracle(rule_name)         # sets base.LABEL etc. for feature_ceiling
        label = dict(base.LABEL)      # snapshot the oracle label map
        per_seed = [run_seed(s, label) for s in SEEDS]
        # use the seed-averaged target from the prior study's ceiling axis
        target_for_curve = statistics.mean(s["target"] for s in per_seed)
        agg = aggregate(per_seed, rule_name, target_for_curve)
        out["by_oracle"][rule_name] = agg
        out["prior_ref"][rule_name] = {
            "ceiling": prior["by_oracle"][rule_name]["ceiling"],
            "labels_to_target": prior["by_oracle"][rule_name]["labels_to_target"],
            "final_acc": prior["by_oracle"][rule_name]["final_acc"],
            "collateral_per_update": prior["by_oracle"][rule_name]["collateral_per_update"],
            "total_collateral": prior["by_oracle"][rule_name]["total_collateral"],
        }
        # console
        print(f"\n######### ORACLE {rule_name} #########")
        m, sd = agg["rf_batch_ceiling"]; print(f"rf_batch ceiling     {m:.3f} +/- {sd:.3f}")
        m, sd = agg["logistic_ceiling"]; print(f"logistic ceiling     {m:.3f} +/- {sd:.3f}")
        m, sd = agg["latent_reference"]; print(f"true-latent ref      {m:.3f} +/- {sd:.3f}")
        m, sd = agg["rf_online_final_acc"]; print(f"rf_online final acc  {m:.3f} +/- {sd:.3f}")
        med, note = agg["rf_online_labels_to_target"]["per_seed_median"]
        print(f"rf_online labels-to-target: avg-curve={agg['rf_online_labels_to_target']['avg_curve']} "
              f"median={med} ({note}) mean={agg['rf_online_labels_to_target']['per_seed_mean']}")
        m, sd = agg["rf_online_collateral_per_update"]; print(f"rf_online collateral/update {m:.3f} +/- {sd:.3f}")
        print("rf_batch top importances:", [(f, round(v, 3)) for f, v in agg["rf_batch_importances"][:6]])

    OUT_JSON.write_text(json.dumps(out, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o))
    print(f"\nwrote {OUT_JSON}")
    return out


if __name__ == "__main__":
    main()
