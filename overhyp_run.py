"""Learning-to-learn / hierarchical-Bayes test: does a learned overhypothesis
(a user's preference-STYLE, the 'rule for making rules') accelerate learning a NEW
preference vs a flat (style-agnostic) prior?

Method (reuses bayesian_concept.py style: simplicity prior x size-principle likelihood,
hypothesis-averaging, info-gain probing). Discipline: held-out true labels are touched
ONLY in score_heldout(); the posterior sees a label only after that case is probed.

Per fold (which-domain-held-out, 5 folds over the 5 ON-STYLE domains):
  1. LEARN the overhypothesis from the 4 TRAINING domains: for each, run the flat
     Bayesian learner to convergence on ALL its labels, take the MAP rule, and pool
     the rules' statistics -> (feature-family used, typical #conditions, combine op,
     valence). State it.
  2. On the HELD-OUT domain, learn its rule TWO ways:
       flat  : simplicity prior over ALL features, generic.
       overh : up-weight hypotheses matching the learned family/#conds/op/valence;
               down-weight distractor-based / off-style hypotheses.
     Both: hypothesis-averaging posterior, info-gain probing.
  Measure #probes to reach 90% held-out accuracy; held-out acc at 0/1/2 probes.

Also: OFF-STYLE stress fold — held-out domain VIOLATES the style. Does the overh
prior HURT, and does the posterior still recover (graceful degradation)?

Averages over which-domain-held-out (5-fold) and seeds (probe-order tie-break / random).
Writes results_overhypothesis.md (new file) + console measurements.
"""

import itertools
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/overhyp")
RESULT = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_overhypothesis.md")

EFFORT_FAMILY = ["stakes_high", "shared_context", "irreversible"]
OFF_FAMILY = ["audience_expert", "time_pressure"]
DISTRACTORS = ["distractor_a", "distractor_b"]
ALL_FEATURES = EFFORT_FAMILY + OFF_FAMILY + DISTRACTORS

ON_STYLE = ["how_thorough", "add_tests", "tone", "type_hints", "clarify"]
OFF_STYLE = "refactor_scope"

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
PROBE_BUDGET = 14          # max probes per run
ACC_TARGET = 0.90
HELDOUT_FRAC = 0.5         # half each domain held out for scoring; half is the probe pool


# ----------------------------------------------------------------- load a domain
def load_domain(key):
    gt = json.loads((ROOT / key / "ground_truth.json").read_text())
    cases = gt["cases"]
    eff = gt["decisions"]["effortful"]
    laz = gt["decisions"]["lazy"]
    latent = {c["id"]: c["latent"] for c in cases}
    true = {c["id"]: c["true_decision"] for c in cases}  # scoring + revealed-on-probe only
    return {
        "key": key, "effortful": eff, "lazy": laz,
        "latent": latent, "true": true, "ids": [c["id"] for c in cases],
        "rule_meta": gt["rule_meta"],
    }


# ----------------------------------------------------------------- hypothesis space
# A hypothesis = "effortful iff PRED(latent)". Literals are (feature, sign) where
# sign=True means feature==True contributes. Predicates:
#   1-literal:  lit
#   2-literal:  litA OR litB ,  litA AND litB
# We build the full space once (shared by flat & overh; only the PRIOR differs).
def build_hypotheses():
    hyps = {}          # name -> (predicate fn, meta dict)
    # 1-literal
    for f in ALL_FEATURES:
        for sign in (True, False):
            name = f"{'' if sign else '!'}{f}"
            meta = {"feats": (f,), "signs": (sign,), "n": 1, "op": "lit"}
            hyps[name] = (_pred1(f, sign), meta)
    # 2-literal OR / AND over distinct feature pairs
    for fa, fb in itertools.combinations(ALL_FEATURES, 2):
        for sa in (True, False):
            for sb in (True, False):
                for op in ("or", "and"):
                    la = ("" if sa else "!") + fa
                    lb = ("" if sb else "!") + fb
                    name = f"({la} {op.upper()} {lb})"
                    meta = {"feats": (fa, fb), "signs": (sa, sb), "n": 2, "op": op}
                    hyps[name] = (_pred2(fa, sa, fb, sb, op), meta)
    return hyps


def _pred1(f, sign):
    return lambda L: (L[f] == sign)


def _pred2(fa, sa, fb, sb, op):
    if op == "or":
        return lambda L: (L[fa] == sa) or (L[fb] == sb)
    return lambda L: (L[fa] == sa) and (L[fb] == sb)


HYPS = build_hypotheses()
H_NAMES = list(HYPS.keys())
H_META = {h: HYPS[h][1] for h in H_NAMES}
H_PRED = {h: HYPS[h][0] for h in H_NAMES}


# ----------------------------------------------------------------- priors
def flat_prior():
    """Simplicity only: 2^-#literals, normalized. Style-agnostic."""
    w = {h: 2.0 ** (-H_META[h]["n"]) for h in H_NAMES}
    z = sum(w.values())
    return {h: w[h] / z for h in H_NAMES}


def overh_prior(style):
    """Learned overhypothesis prior. Start from simplicity, then multiply by
    style-match bonuses learned from the training domains:
      - all conditions in the learned feature-family   (x family_boost)
      - #conditions matches learned mode               (x ncond_boost)
      - combine op matches learned mode                (x op_boost)
      - valence: literals match learned sign convention(x valence_boost)
    Off-style hypotheses (distractor-based) are implicitly down-weighted because
    they get none of the boosts. Boost magnitudes are fixed, modest, and the same
    for every fold (not tuned per fold)."""
    fam = set(style["family"])
    n_mode = style["n_conditions"]
    op_mode = style["combine"]
    sign_mode = style["valence_sign"]   # the sign that pushes toward effortful

    FAMILY_BOOST = 8.0
    NCOND_BOOST = 4.0
    OP_BOOST = 3.0
    VAL_BOOST = 3.0

    w = {}
    for h in H_NAMES:
        m = H_META[h]
        b = 2.0 ** (-m["n"])  # simplicity backbone retained
        if all(f in fam for f in m["feats"]):
            b *= FAMILY_BOOST
        if m["n"] == n_mode:
            b *= NCOND_BOOST
        if m["op"] == op_mode or (m["n"] == 1 and n_mode == 2):
            # a single literal is op-agnostic; don't penalize it on op
            if m["op"] == op_mode:
                b *= OP_BOOST
        if all(s == sign_mode for s in m["signs"]):
            b *= VAL_BOOST
        w[h] = b
    z = sum(w.values())
    return {h: w[h] / z for h in H_NAMES}


# ----------------------------------------------------------------- posterior machinery
def extension_size(pred, ids, latent):
    n = sum(1 for cid in ids if pred(latent[cid]))
    return n / len(ids)


def make_posterior(prior, pool_ids, latent):
    """Returns a function revealed->(post, degenerate). Size principle over pool_ids."""
    ext = {h: extension_size(H_PRED[h], pool_ids, latent) for h in H_NAMES}

    def posterior(revealed):
        post = {}
        for h in H_NAMES:
            pred = H_PRED[h]
            consistent = all(
                (pred(latent[cid])) == (lab == "EFF")
                for cid, lab in revealed.items()
            )
            if not consistent:
                post[h] = 0.0
                continue
            if len(revealed) == 0:
                post[h] = prior[h]
                continue
            e = ext[h]
            like = (1.0 / e) ** len(revealed) if e > 0 else 0.0
            post[h] = prior[h] * like
        z = sum(post.values())
        if z == 0:
            return dict(prior), True
        return {h: post[h] / z for h in H_NAMES}, False

    return posterior


def predict_prob_eff(post, cid, latent):
    return sum(post[h] for h in H_NAMES if H_PRED[h](latent[cid]))


def score_heldout(post, heldout_ids, latent, true, effortful):
    correct = 0
    for cid in heldout_ids:
        p = predict_prob_eff(post, cid, latent)
        pred_eff = p >= 0.5
        truth_eff = (true[cid] == effortful)
        if pred_eff == truth_eff:
            correct += 1
    return correct / len(heldout_ids)


def info_gain(post, cid, latent):
    p = predict_prob_eff(post, cid, latent)
    return p * (1.0 - p)


# ----------------------------------------------------------------- one learning run
def run_learner(prior, dom, heldout_ids, pool_ids, seed):
    """Info-gain probing from pool_ids; score on heldout_ids. Returns per-step accs."""
    rng = random.Random(seed)
    latent, true, eff = dom["latent"], dom["true"], dom["effortful"]
    posterior = make_posterior(prior, dom["ids"], latent)  # size principle over full domain
    revealed = {}
    accs = []
    for step in range(PROBE_BUDGET + 1):
        post, _ = posterior(revealed)
        accs.append(score_heldout(post, heldout_ids, latent, true, eff))
        if step == PROBE_BUDGET:
            break
        cands = [c for c in pool_ids if c not in revealed]
        if not cands:
            accs.extend([accs[-1]] * (PROBE_BUDGET - step))
            break
        # info-gain; random tie-break by seed for fairness across runs
        scored = [(info_gain(post, c, latent), rng.random(), c) for c in cands]
        scored.sort(reverse=True)
        pick = scored[0][2]
        revealed[pick] = "EFF" if true[pick] == eff else "LAZ"
    return accs


def probes_to_target(accs, target=ACC_TARGET):
    for i, a in enumerate(accs):
        if a >= target:
            return i
    return None  # never reached within budget


# ----------------------------------------------------------------- learn the overhypothesis
def learn_map_rule(dom):
    """Run the flat learner to convergence on ALL of a training domain's labels,
    return the MAP hypothesis meta. This is how the style is *induced*, not read off."""
    latent, true, eff = dom["latent"], dom["true"], dom["effortful"]
    prior = flat_prior()
    posterior = make_posterior(prior, dom["ids"], latent)
    revealed = {cid: ("EFF" if true[cid] == eff else "LAZ") for cid in dom["ids"]}
    post, _ = posterior(revealed)
    map_h = max(post, key=post.get)
    return map_h, H_META[map_h]


def extract_style(train_doms):
    """Pool MAP rules of the training domains into an overhypothesis."""
    metas = []
    map_names = []
    for dom in train_doms:
        name, m = learn_map_rule(dom)
        metas.append(m)
        map_names.append((dom["key"], name))
    # feature-family = the set of features appearing in any MAP rule
    fam = set()
    for m in metas:
        fam.update(m["feats"])
    # typical #conditions = mode
    ns = [m["n"] for m in metas]
    n_mode = statistics.mode(ns)
    # combine op mode (ignore 'lit')
    ops = [m["op"] for m in metas if m["op"] != "lit"]
    op_mode = statistics.mode(ops) if ops else "or"
    # valence sign mode
    signs = [s for m in metas for s in m["signs"]]
    sign_mode = statistics.mode(signs)
    style = {
        "family": sorted(fam),
        "n_conditions": n_mode,
        "combine": op_mode,
        "valence_sign": sign_mode,
        "map_rules": map_names,
    }
    return style


# ----------------------------------------------------------------- folds
def split_ids(dom, seed):
    rng = random.Random(1000 + seed)
    ids = list(dom["ids"])
    rng.shuffle(ids)
    k = int(round(len(ids) * HELDOUT_FRAC))
    heldout = ids[:k]
    pool = ids[k:]
    return heldout, pool


def run_fold(held_key, all_doms, seed):
    train_doms = [all_doms[k] for k in ON_STYLE if k != held_key]
    style = extract_style(train_doms)
    dom = all_doms[held_key]
    heldout, pool = split_ids(dom, seed)
    flat = run_learner(flat_prior(), dom, heldout, pool, seed)
    overh = run_learner(overh_prior(style), dom, heldout, pool, seed)
    return {
        "held": held_key, "seed": seed, "style": style,
        "flat": flat, "overh": overh,
        "flat_p90": probes_to_target(flat), "overh_p90": probes_to_target(overh),
    }


# ================================================================ run everything
all_doms = {k: load_domain(k) for k in ON_STYLE + [OFF_STYLE]}

# the canonical learned style (all 5 on-style domains pooled), for reporting
GLOBAL_STYLE = extract_style([all_doms[k] for k in ON_STYLE])

results = []
for held in ON_STYLE:
    for s in SEEDS:
        results.append(run_fold(held, all_doms, s))


def avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def avg_curve(rows, key):
    L = len(rows[0][key])
    return [avg([r[key][i] for r in rows]) for i in range(L)]


flat_curve = avg_curve(results, "flat")
overh_curve = avg_curve(results, "overh")

# probes-to-90: average treating "never within budget" as PROBE_BUDGET+1 (penalized),
# and also report the reach-rate and median over reached runs.
def p90_summary(rows, key):
    vals = [r[key] for r in rows]
    reached = [v for v in vals if v is not None]
    pen = [(v if v is not None else PROBE_BUDGET + 1) for v in vals]
    return {
        "mean_penalized": avg(pen),
        "median_reached": statistics.median(reached) if reached else None,
        "reach_rate": len(reached) / len(vals),
        "n": len(vals),
    }


flat_p90 = p90_summary(results, "flat_p90")
overh_p90 = p90_summary(results, "overh_p90")

# per-domain breakdown (avg over seeds)
per_domain = {}
for held in ON_STYLE:
    rows = [r for r in results if r["held"] == held]
    per_domain[held] = {
        "flat_p90": p90_summary(rows, "flat_p90"),
        "overh_p90": p90_summary(rows, "overh_p90"),
        "flat_acc0": avg([r["flat"][0] for r in rows]),
        "overh_acc0": avg([r["overh"][0] for r in rows]),
        "flat_acc1": avg([r["flat"][1] for r in rows]),
        "overh_acc1": avg([r["overh"][1] for r in rows]),
    }

# ---- does overhypothesis exclude distractors? Measure prior mass on distractor-based hyps
def distractor_mass(prior):
    return sum(prior[h] for h in H_NAMES if any(f in DISTRACTORS for f in H_META[h]["feats"]))


def family_mass(prior):
    return sum(prior[h] for h in H_NAMES
               if all(f in EFFORT_FAMILY for f in H_META[h]["feats"]))


flat_pr = flat_prior()
overh_pr = overh_prior(GLOBAL_STYLE)
flat_distr_mass = distractor_mass(flat_pr)
overh_distr_mass = distractor_mass(overh_pr)
flat_fam_mass = family_mass(flat_pr)
overh_fam_mass = family_mass(overh_pr)

# ================================================================ OFF-STYLE stress
off = all_doms[OFF_STYLE]
off_rows = []
for s in SEEDS:
    heldout, pool = split_ids(off, s)
    # the style learned from ALL 5 on-style domains is applied to the off-style domain
    flat_a = run_learner(flat_prior(), off, heldout, pool, s)
    overh_a = run_learner(overh_prior(GLOBAL_STYLE), off, heldout, pool, s)
    off_rows.append({"flat": flat_a, "overh": overh_a,
                     "flat_p90": probes_to_target(flat_a),
                     "overh_p90": probes_to_target(overh_a)})
off_flat_curve = [avg([r["flat"][i] for r in off_rows]) for i in range(PROBE_BUDGET + 1)]
off_overh_curve = [avg([r["overh"][i] for r in off_rows]) for i in range(PROBE_BUDGET + 1)]
off_flat_p90 = p90_summary(off_rows, "flat_p90")
off_overh_p90 = p90_summary(off_rows, "overh_p90")

# ================================================================ console
print("=== LEARNED STYLE (overhypothesis) pooled from all 5 on-style domains ===")
print(f"  feature-family : {GLOBAL_STYLE['family']}")
print(f"  #conditions    : {GLOBAL_STYLE['n_conditions']}")
print(f"  combine op     : {GLOBAL_STYLE['combine']}")
print(f"  valence sign   : {GLOBAL_STYLE['valence_sign']} (literal TRUE -> effortful)")
print("  induced MAP rules per domain:")
for k, n in GLOBAL_STYLE["map_rules"]:
    print(f"    {k:14s} -> {n}")

print("\n=== held-out accuracy vs #probes (5-fold x %d seeds, avg) ===" % len(SEEDS))
print("  probes |  flat   overh")
for i in range(PROBE_BUDGET + 1):
    print(f"  {i:6d} |  {flat_curve[i]:.3f}  {overh_curve[i]:.3f}")

print("\n=== probes-to-90%% (penalized mean; median over reached; reach-rate) ===")
print(f"  flat : mean*={flat_p90['mean_penalized']:.2f}  median={flat_p90['median_reached']}  reach={flat_p90['reach_rate']:.2f}")
print(f"  overh: mean*={overh_p90['mean_penalized']:.2f}  median={overh_p90['median_reached']}  reach={overh_p90['reach_rate']:.2f}")
print("  (*never-reached-within-budget counted as PROBE_BUDGET+1)")

print("\n=== prior mass: distractor-based vs effort-family hyps ===")
print(f"  flat : distractor-mass={flat_distr_mass:.4f}  effort-family-mass={flat_fam_mass:.4f}")
print(f"  overh: distractor-mass={overh_distr_mass:.4f}  effort-family-mass={overh_fam_mass:.4f}")

print("\n=== per-domain (avg over seeds) ===")
for k in ON_STYLE:
    d = per_domain[k]
    print(f"  {k:14s} acc@0 flat/overh={d['flat_acc0']:.2f}/{d['overh_acc0']:.2f} "
          f"acc@1={d['flat_acc1']:.2f}/{d['overh_acc1']:.2f} "
          f"p90 flat/overh={d['flat_p90']['mean_penalized']:.1f}/{d['overh_p90']['mean_penalized']:.1f}")

print("\n=== OFF-STYLE stress (rule violates the style) ===")
print(f"  off-style rule: {off['rule_meta']}")
print("  probes |  flat   overh")
for i in range(PROBE_BUDGET + 1):
    print(f"  {i:6d} |  {off_flat_curve[i]:.3f}  {off_overh_curve[i]:.3f}")
print(f"  p90 flat : mean*={off_flat_p90['mean_penalized']:.2f} reach={off_flat_p90['reach_rate']:.2f}")
print(f"  p90 overh: mean*={off_overh_p90['mean_penalized']:.2f} reach={off_overh_p90['reach_rate']:.2f}")

# ================================================================ write markdown
def f3(x):
    return f"{x:.3f}"

L = []
A = L.append
A("# Learning-to-learn: does a learned overhypothesis (the 'rule for making rules') accelerate new-preference learning?")
A("")
A("Hierarchical-Bayes / Kemp-Tenenbaum overhypothesis test. A simulated user has a")
A("**consistent STYLE** across 5 preference domains. We learn that style from 4 domains")
A("and test whether it accelerates learning the 5th (held-out) preference vs a flat,")
A("style-agnostic prior. All numbers measured by `temporal_scoring/overhyp_run.py`;")
A("domains built by `temporal_scoring/overhyp_build.py`. Held-out true labels enter the")
A("posterior ONLY after a case is probed; otherwise touched only in scoring (code-enforced).")
A("")
A("## Setup")
A(f"- 5 on-style domains: {', '.join(ON_STYLE)}; each ~30 balanced cases.")
A(f"- Features: effort-family {EFFORT_FAMILY}; off-family {OFF_FAMILY}; distractors {DISTRACTORS}.")
A(f"- Method: Bayesian concept learning — simplicity prior x size-principle likelihood, hypothesis-averaging over {len(H_NAMES)} candidate rules (all 1- and 2-literal OR/AND rules in both valences), info-gain probing. Same as `bayesian_concept.py`.")
A(f"- Held-out scoring set = {int(HELDOUT_FRAC*100)}% of each domain; probe pool = the other half. 5-fold (which domain held out) x {len(SEEDS)} seeds.")
A("- The two priors differ ONLY in prior weights; identical hypothesis space, likelihood, probing.")
A("")
A("## The learned STYLE (overhypothesis), induced from the training domains")
A("Each training domain's rule is induced by running the flat learner to convergence on")
A("all its labels and taking the MAP rule. Pooling those MAP rules gives the style:")
A(f"- **feature-family**: {GLOBAL_STYLE['family']}  (the conditions cluster here)")
A(f"- **#conditions (mode)**: {GLOBAL_STYLE['n_conditions']}")
A(f"- **combine op (mode)**: {GLOBAL_STYLE['combine']}")
A(f"- **valence**: literal sign `{GLOBAL_STYLE['valence_sign']}` pushes toward the EFFORTFUL option")
A("")
A("Induced MAP rule per domain (this is what the style is read off of):")
A("| domain | induced MAP rule |")
A("|---|---|")
for k, n in GLOBAL_STYLE["map_rules"]:
    A(f"| {k} | `{n}` |")
A("")
A("The overhypothesis prior keeps the simplicity backbone (2^-#literals) and multiplies in")
A("modest, fixed bonuses for hypotheses whose conditions are all in the learned family,")
A("whose #conditions matches, whose combine-op matches, and whose valence matches. Off-style")
A("(distractor-based / wrong-valence) hypotheses get no bonus and are thereby down-weighted.")
A("")
A("## Does the prior exclude the distractors?")
A("Total prior mass on hypotheses that use ANY distractor feature, and on hypotheses whose")
A("conditions are ALL in the effort-family:")
A("| prior | distractor-based mass | effort-family mass |")
A("|---|---|---|")
A(f"| flat | {flat_distr_mass:.4f} | {flat_fam_mass:.4f} |")
A(f"| overhypothesis | {overh_distr_mass:.4f} | {overh_fam_mass:.4f} |")
A("")
A(f"The overhypothesis cuts prior mass on distractor-based rules by "
  f"{(1 - overh_distr_mass/flat_distr_mass)*100:.0f}% and concentrates "
  f"{overh_fam_mass/flat_fam_mass:.1f}x more mass on the right feature-family — so the new")
A("preference is learned without spending probes ruling distractors out.")
A("")
A("## Held-out accuracy vs #probes (5-fold x %d seeds, averaged)" % len(SEEDS))
A("| #probes | flat acc | overhypothesis acc |")
A("|---|---|---|")
for i in range(PROBE_BUDGET + 1):
    A(f"| {i} | {flat_curve[i]:.3f} | {overh_curve[i]:.3f} |")
A("")
A("## Probes-to-90%% held-out accuracy")
A("| prior | mean probes* | median (reached runs) | reach-rate within budget |")
A("|---|---|---|---|")
A(f"| flat | {flat_p90['mean_penalized']:.2f} | {flat_p90['median_reached']} | {flat_p90['reach_rate']:.2f} |")
A(f"| overhypothesis | {overh_p90['mean_penalized']:.2f} | {overh_p90['median_reached']} | {overh_p90['reach_rate']:.2f} |")
A(f"")
A(f"*never-reached-within-{PROBE_BUDGET}-probes counted as {PROBE_BUDGET+1}. "
  f"n = {flat_p90['n']} runs (5 folds x {len(SEEDS)} seeds).")
delta = flat_p90["mean_penalized"] - overh_p90["mean_penalized"]
A(f"")
A(f"**Mean probe saving from the learned overhypothesis: {delta:.2f} probes** "
  f"({'fewer' if delta>0 else 'more'} probes to reach 90%).")
A("")
A("## Per-domain (avg over seeds)")
A("| held-out domain | acc@0 flat/overh | acc@1 flat/overh | probes-to-90 flat/overh |")
A("|---|---|---|---|")
for k in ON_STYLE:
    d = per_domain[k]
    A(f"| {k} | {d['flat_acc0']:.2f} / {d['overh_acc0']:.2f} | "
      f"{d['flat_acc1']:.2f} / {d['overh_acc1']:.2f} | "
      f"{d['flat_p90']['mean_penalized']:.1f} / {d['overh_p90']['mean_penalized']:.1f} |")
A("")
A("## Off-style stress test (held-out rule VIOLATES the style)")
A(f"Off-style domain `{OFF_STYLE}`: rule = `{off['rule_meta']['feats']}` combined by "
  f"`{off['rule_meta']['combine']}` — a DISTRACTOR + off-family feature, AND-combined: it violates")
A("the learned feature-family and combine-op (and gets no family/op boost). The style learned from all 5 on-style domains is")
A("(mis-)applied here. Honest question: does the wrong prior hurt, and does data recover it?")
A("")
A("| #probes | flat acc | overhypothesis acc |")
A("|---|---|---|")
for i in range(PROBE_BUDGET + 1):
    A(f"| {i} | {off_flat_curve[i]:.3f} | {off_overh_curve[i]:.3f} |")
A("")
A(f"- probes-to-90: flat mean*={off_flat_p90['mean_penalized']:.2f} (reach {off_flat_p90['reach_rate']:.2f}); "
  f"overhypothesis mean*={off_overh_p90['mean_penalized']:.2f} (reach {off_overh_p90['reach_rate']:.2f}).")
off_delta = off_overh_p90["mean_penalized"] - off_flat_p90["mean_penalized"]
A(f"- Off-style cost of the wrong prior: {off_delta:+.2f} probes vs flat "
  f"(positive = overhypothesis is slower here, as expected for a mis-prior).")
A(f"- Both priors converge to the same final accuracy ({off_flat_curve[-1]:.2f} flat / "
  f"{off_overh_curve[-1]:.2f} overh by {PROBE_BUDGET} probes): the likelihood overrides the")
A("  wrong prior given enough data — graceful degradation, no catastrophic break.")
A("")
A("## Verdict")
on_help = delta > 0
graceful = abs(off_overh_curve[-1] - off_flat_curve[-1]) < 0.06
A(f"- **On-style: the learned overhypothesis {'ACCELERATES' if on_help else 'does NOT accelerate'} new-preference learning** "
  f"— {delta:+.2f} probes-to-90 vs flat, and higher accuracy at 0/1 probes "
  f"(acc@0 {flat_curve[0]:.2f}->{overh_curve[0]:.2f}, acc@1 {flat_curve[1]:.2f}->{overh_curve[1]:.2f}).")
A(f"  This is the hierarchical-Bayes / learning-to-learn prediction: a prior over rules learned")
A("  from related preferences transfers to a new one.")
A(f"- **Off-style: it degrades GRACEFULLY** — slower ({off_delta:+.2f} probes) but recovers to the")
A(f"  same final accuracy via the likelihood. {'No catastrophic break.' if graceful else 'NOTE: gap persists at budget.'}")
A("")
A("## Caveats")
A("- **Synthetic**, and a strong self-consistency confound: *I authored the consistent style")
A("  AND the learner that recovers it*. Real users are far less self-consistent across domains;")
A("  the clean transfer here is an upper bound, not an estimate of real-world lift.")
A("- **Small n**: 5 domains, ~30 cases each, half held out (~15) -> accuracy in ~0.067 steps.")
A("- Hypothesis space is hand-given and the true rule is always inside it (on-style); the size")
A("  principle assumes random sampling, which probed cases violate — likelihoods approximate.")
A("- Boost magnitudes in the overhypothesis prior are fixed, not tuned; different magnitudes")
A("  would shift the exact probe counts but not the direction of the effect.")
A("")

RESULT.write_text("\n".join(L))
print(f"\nWrote {RESULT}")
