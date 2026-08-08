"""Build a simulated user with a CONSISTENT STYLE across preference domains.

Overhypothesis (the user's "style" = a rule for making rules):
  - each rule uses EXACTLY 2 conditions (#conditions = 2)
  - the conditions are drawn from the EFFORT-FAMILY = {stakes_high, shared_context, irreversible}
    (NOT the distractors, NOT audience_expert / time_pressure)
  - valence: any condition being TRUE pushes toward the EFFORTFUL option.
    Concretely each rule is an OR of 2 effort-family literals:
        effortful  iff  (feat_a OR feat_b)   for feat_a, feat_b in EFFORT_FAMILY
    "high-stakes / shared / irreversible -> the effortful option".

5 on-style domains instantiate this with different 2-feature combos.
A 6th OFF-STYLE domain is built that VIOLATES the style (distractor-based rule,
opposite valence) for the failure / mis-prior stress test.

Features (per case):
  EFFORT_FAMILY  : stakes_high, shared_context, irreversible   (style-relevant)
  OFF_FAMILY     : audience_expert, time_pressure              (real but off-style)
  DISTRACTORS    : distractor_a, distractor_b                  (pure noise)

Each domain has a different "effortful" vs "lazy" decision label pair, e.g.
  how_thorough: thorough / quick
  add_tests:    add_tests / skip_tests
  tone:         formal / casual
  type_hints:   add_hints / no_hints
  clarify:      ask / assume

Writes per-domain {descriptions.json, ground_truth.json} under overhyp/<domain>/.
true rules recorded for scoring only. Deterministic (seeded). New files only.
"""

import itertools
import json
import random
from pathlib import Path

OUT = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/overhyp")
OUT.mkdir(parents=True, exist_ok=True)

EFFORT_FAMILY = ["stakes_high", "shared_context", "irreversible"]
OFF_FAMILY = ["audience_expert", "time_pressure"]
DISTRACTORS = ["distractor_a", "distractor_b"]
ALL_FEATURES = EFFORT_FAMILY + OFF_FAMILY + DISTRACTORS

N_CASES = 30  # per domain

# ---- the 5 on-style domains: (key, effortful_label, lazy_label, 2 effort-family feats) ----
# every rule = (feat_a OR feat_b) -> effortful ; both feats from EFFORT_FAMILY ; #conds = 2.
ON_STYLE = [
    ("how_thorough", "thorough", "quick",      ("stakes_high", "irreversible")),
    ("add_tests",    "add_tests", "skip_tests", ("stakes_high", "shared_context")),
    ("tone",         "formal",   "casual",     ("shared_context", "irreversible")),
    ("type_hints",   "add_hints", "no_hints",  ("stakes_high", "shared_context")),
    ("clarify",      "ask",      "assume",     ("shared_context", "irreversible")),
]

# ---- the off-style stress domain: distractor-based rule, AND not OR, opposite framing ----
# rule = (distractor_a AND time_pressure) -> effortful. Uses a distractor + an off-family
# feature, AND-combination. This deliberately violates EVERY axis of the learned style.
OFF_STYLE = ("refactor_scope", "big_refactor", "small_refactor",
             ("distractor_a", "time_pressure"))


def make_cases(rng, feat_a, feat_b, combine, effortful, lazy):
    """Generate N_CASES feature vectors + true decision under the given rule.

    combine: 'or' -> (feat_a or feat_b); 'and' -> (feat_a and feat_b).
    Balanced-ish: we enumerate enough random vectors then keep a class-balanced subset.
    """
    def decide(L):
        a, b = L[feat_a], L[feat_b]
        hit = (a or b) if combine == "or" else (a and b)
        return effortful if hit else lazy

    # generate a large candidate pool of distinct vectors, then sample class-balanced
    pool = []
    seen = set()
    tries = 0
    while len(pool) < 400 and tries < 5000:
        tries += 1
        L = {f: rng.random() < 0.5 for f in ALL_FEATURES}
        key = tuple(L[f] for f in ALL_FEATURES)
        if key in seen:
            continue
        seen.add(key)
        pool.append(L)
    eff = [L for L in pool if decide(L) == effortful]
    laz = [L for L in pool if decide(L) == lazy]
    rng.shuffle(eff)
    rng.shuffle(laz)
    half = N_CASES // 2
    chosen = eff[:half] + laz[: N_CASES - half]
    rng.shuffle(chosen)
    cases = []
    for i, L in enumerate(chosen, 1):
        cases.append({"id": f"c{i:02d}", "latent": L, "true_decision": decide(L)})
    return cases


def describe(domain_key, effortful, lazy, L):
    """A short NL description per case (the 'observable' surface) — latent stays hidden."""
    parts = []
    if L["stakes_high"]:
        parts.append("the stakes are high")
    if L["shared_context"]:
        parts.append("this is shared/collaborative context")
    if L["irreversible"]:
        parts.append("the action is hard to reverse")
    if L["audience_expert"]:
        parts.append("the audience is expert")
    if L["time_pressure"]:
        parts.append("there is time pressure")
    if L["distractor_a"]:
        parts.append("it's a Tuesday-flavored task")  # pure-noise distractor flavor
    if L["distractor_b"]:
        parts.append("the filename is long")
    ctx = "; ".join(parts) if parts else "no salient factors"
    return f"[{domain_key}] Decide between {effortful} and {lazy}. Context: {ctx}."


def build_domain(key, effortful, lazy, feats, combine, seed):
    rng = random.Random(seed)
    feat_a, feat_b = feats
    cases = make_cases(rng, feat_a, feat_b, combine, effortful, lazy)
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    rule_txt = (
        f"{effortful} iff ({feat_a} {combine.upper()} {feat_b}); else {lazy}. "
        f"Other features are irrelevant."
    )
    gt = {
        "domain": key,
        "decisions": {"effortful": effortful, "lazy": lazy},
        "features": ALL_FEATURES,
        "rule": rule_txt,
        "rule_meta": {
            "feats": list(feats), "combine": combine,
            "n_conditions": 2,
            "family": "effort" if all(f in EFFORT_FAMILY for f in feats) else "off",
        },
        "cases": cases,
    }
    (d / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    descs = [
        {"id": c["id"], "text": describe(key, effortful, lazy, c["latent"])}
        for c in cases
    ]
    (d / "descriptions.json").write_text(json.dumps({"domain": key, "descriptions": descs}, indent=2))
    n_eff = sum(1 for c in cases if c["true_decision"] == effortful)
    return key, len(cases), n_eff, rule_txt


print("=== building on-style domains (style = 2 conds, effort-family, OR->effortful) ===")
summary = []
for i, (key, eff, laz, feats) in enumerate(ON_STYLE):
    r = build_domain(key, eff, laz, feats, "or", seed=100 + i)
    summary.append(r)
    print(f"  {r[0]:14s} n={r[1]} n_effortful={r[2]} rule: {r[3]}")

print("\n=== building OFF-STYLE stress domain (distractor+off-family, AND, violates style) ===")
key, eff, laz, feats = OFF_STYLE
r = build_domain(key, eff, laz, feats, "and", seed=200)
print(f"  {r[0]:14s} n={r[1]} n_effortful={r[2]} rule: {r[3]}")

print(f"\nWrote domains to {OUT}")
print("on-style domains:", [s[0] for s in summary])
print("off-style domain:", key)
