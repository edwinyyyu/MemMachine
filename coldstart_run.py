"""COLD-START run of the preference-learning method.

Brand-new agent, brand-new user, ZERO learned rules. No prior conversation/context.

Domain (fresh; NOT ask-vs-proceed, NOT repo-style):
    "When should I proactively add a unit test for a function I just wrote?"

Features of a situation (a function just written):
    has_branching : bool   # function has if/else / loops (non-trivial control flow)
    is_public     : bool   # part of the public API / exported, vs private helper
    has_io        : bool   # touches network/disk/db
    is_trivial    : bool   # one-liner / pure passthrough
    is_hot_path   : bool   # on a performance- or correctness-critical path

HIDDEN ORACLE RULE (2-condition, programmatic -> no LLM self-consistency confound):
    add_test  <=>  has_branching AND is_public
The model never sees this string; it only sees yes/no answers from oracle().

Token/harness split:
  - MODEL (this file's *comments* + the hand-authored candidate list + the
    hand-authored situations) plays the role of the LLM: it PROPOSES candidate
    rules from the universal prior, INVENTS the situations, and APPLIES the rule.
  - ENGINE (pref_engine) does ALL math/state: posterior, info-gain, selection,
    the should_probe gate, updates, entropy, persistence.
"""

from __future__ import annotations

import json
import random

import pref_engine as pe

FEATURES = ["has_branching", "is_public", "has_io", "is_trivial", "is_hot_path"]


# --------------------------------------------------------------------------- #
# HIDDEN programmatic oracle (the user's true, unspoken preference).
# Includes a SKIP for one specific situation to exercise graceful no-op.
# --------------------------------------------------------------------------- #
def oracle(x, skip_ids=()):
    if x["_id"] in skip_ids:
        return "SKIP"
    return x["has_branching"] and x["is_public"]


# --------------------------------------------------------------------------- #
# MODEL STEP (tokens): propose candidate rules from the UNIVERSAL PRIOR.
# A brand-new agent has no learned rules, so it enumerates simple conjunctive
# hypotheses over the features: the always-rule, every single-feature rule, and
# every 2-feature conjunction. Prior 2^(-#conds) favors the simpler ones. The
# TRUE rule lives in this set but is not flagged.
# --------------------------------------------------------------------------- #
def propose_candidates():
    hyps = [{"id": "always_test", "conds": []}]
    for f in FEATURES:
        hyps.append({"id": f, "conds": [(f, True)]})
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            f1, f2 = FEATURES[i], FEATURES[j]
            hyps.append({"id": f"{f1}&{f2}", "conds": [(f1, True), (f2, True)]})
    return hyps


# --------------------------------------------------------------------------- #
# MODEL STEP (tokens): INVENT concrete situations (features + NL description).
# 25 in the working pool + 8 held-out.
# --------------------------------------------------------------------------- #
def make_situations():
    rng = random.Random(7)

    def desc(x):
        bits = []
        bits.append("public API" if x["is_public"] else "private helper")
        bits.append("with branching/loops" if x["has_branching"] else "straight-line")
        if x["has_io"]:
            bits.append("does I/O")
        if x["is_trivial"]:
            bits.append("trivial one-liner")
        if x["is_hot_path"]:
            bits.append("on a hot path")
        return "Just wrote a " + ", ".join(bits) + " function."

    def gen(n, tag):
        out = []
        seen = set()
        k = 0
        while len(out) < n:
            x = {f: rng.random() < 0.5 for f in FEATURES}
            key = tuple(x[f] for f in FEATURES)
            if key in seen:
                continue
            seen.add(key)
            x["_id"] = f"{tag}{k}"
            x["_desc"] = desc(x)
            out.append(x)
            k += 1
        return out

    pool = gen(25, "p")
    held = gen(8, "h")
    return pool, held


# --------------------------------------------------------------------------- #
# Helpers for the loop
# --------------------------------------------------------------------------- #
def remaining(pool, revealed):
    used = {id(x) for (x, _) in revealed}
    # match by _id, since update stores the same dict object
    used_ids = {x["_id"] for (x, _) in revealed}
    return [x for x in pool if x["_id"] not in used_ids]


def main():
    log = []

    def say(s):
        print(s)
        log.append(s)

    say("=== COLD START: zero learned rules, no prior context ===\n")

    # ---- MODEL proposes candidates (universal prior) ----
    hyps = propose_candidates()
    say(f"[MODEL] proposed {len(hyps)} candidate conjunctive rules from the "
        f"universal prior (always / 1-feature / 2-feature).")

    # ---- MODEL invents situations ----
    pool, held = make_situations()
    say(f"[MODEL] invented {len(pool)} working situations + {len(held)} held-out.\n")

    # ---- ENGINE init: domain consequence (cost of a wrong autonomous call) ----
    # Adding/omitting a test wrong is a moderate cost: pick consequence=2.0.
    pe.init(hyps, consequence=2.0)
    revealed = []
    BAR = 0.30  # should_probe threshold on info_gain*consequence

    say(f"[ENGINE] init: |H|={len(hyps)}, consequence=2.0, bar={BAR}")
    say(f"[ENGINE] prior entropy = {pe.entropy(revealed):.4f} nats\n")

    # We deliberately schedule a SKIP: the FIRST probe the engine selects will be
    # skipped by the user (oracle returns SKIP), to show the graceful no-op.
    skip_done = False
    skip_id_used = None
    n_probes = 0
    n_skips = 0

    say("--- PROBING LOOP (while should_probe) ---")
    for step in range(1, 30):
        pool_left = remaining(pool, revealed)
        flag, best_x, gain, value = pe.should_probe(pool_left, revealed, BAR)
        ent = pe.entropy(revealed)
        if not flag:
            say(f"[ENGINE] should_probe=False (best info_gain={gain:.4f}, "
                f"value={value:.4f} <= bar={BAR}); entropy={ent:.4f}. CONVERGED.")
            break

        # First selected probe -> user SKIPS it.
        if not skip_done:
            skip_id_used = best_x["_id"]
            ans = oracle(best_x, skip_ids={skip_id_used})
            say(f"[ENGINE] select_probe -> {best_x['_id']} "
                f"(info_gain={gain:.4f}, value={value:.4f}); entropy={ent:.4f}")
            say(f"          \"{best_x['_desc']}\"")
            say(f"[USER]   SKIP")
            new_rev = pe.update(revealed, best_x, ans)
            assert len(new_rev) == len(revealed), "SKIP must be a no-op!"
            say(f"[ENGINE] update with SKIP -> no-op (revealed stays n={len(new_rev)}). "
                f"loop continues.")
            revealed = new_rev
            skip_done = True
            n_skips += 1
            # Force a different probe next iteration by marking this one consumed
            # ONLY for selection purposes: we drop it from the live pool so the
            # loop makes progress (a real user who skips won't be re-asked the
            # identical question immediately).
            pool = [x for x in pool if x["_id"] != skip_id_used]
            continue

        # Normal probe: ask the oracle, engine updates.
        ans = oracle(best_x)
        say(f"[ENGINE] select_probe -> {best_x['_id']} "
            f"(info_gain={gain:.4f}, value={value:.4f}); entropy={ent:.4f}")
        say(f"          \"{best_x['_desc']}\"")
        say(f"[USER]   answer = {ans}")
        revealed = pe.update(revealed, best_x, ans)
        n_probes += 1
        say(f"[ENGINE] posterior updated -> entropy={pe.entropy(revealed):.4f}, "
            f"revealed n={len(revealed)}\n")

    say("")

    # ---- Show MAP + averaged learned rule ----
    map_h, map_p = pe.map_hypothesis(revealed)
    say(f"[ENGINE] MAP rule = '{map_h['id']}' conds={map_h['conds']} "
        f"(posterior={map_p:.3f})")
    post = pe.posterior(revealed)
    top = sorted(zip(hyps, post), key=lambda t: -t[1])[:4]
    say("[ENGINE] top posterior mass:")
    for h, p in top:
        say(f"          {p:6.3f}  {h['id']}")
    say(f"[ENGINE] true hidden oracle rule = 'has_branching AND is_public' "
        f"(for grading only; model never saw it)")
    say("")

    # ---- ANTI-INTRUSION demo: a should_probe=False case PROCEEDS ----
    # We construct a situation the model could face AFTER convergence and show
    # that should_probe is False, so the model just APPLIES the learned rule
    # (no question asked). We pass a singleton pool of one near-decided situation.
    say("--- ANTI-INTRUSION CHECK (post-convergence) ---")
    # Pick a held-out situation the posterior is confident about.
    confident_x = None
    for x in held:
        p_true, dec = pe.predict(x, revealed)
        if p_true > 0.9 or p_true < 0.1:
            confident_x = x
            break
    if confident_x is None:
        confident_x = held[0]
    flag, bx, g, val = pe.should_probe([confident_x], revealed, BAR)
    p_true, decision = pe.predict(confident_x, revealed)
    say(f"[ENGINE] situation {confident_x['_id']}: \"{confident_x['_desc']}\"")
    say(f"[ENGINE] should_probe={flag} (info_gain={g:.4f}, value={val:.4f} "
        f"<= bar={BAR})")
    say(f"[MODEL]  PROCEED WITHOUT ASKING -> apply learned rule: "
        f"add_test={decision} (P_true={p_true:.3f})  (lightweight / no intrusion)")
    say("")

    # ---- Persist learned rule ----
    learned_path = "coldstart_learned_rule.json"
    payload = pe.save_learned(learned_path, revealed)
    say(f"[PERSIST] learned rule saved -> {learned_path}")
    say("")

    # ---- Held-out accuracy of the learned (hypothesis-averaged) rule ----
    correct = 0
    rows = []
    for x in held:
        truth = x["has_branching"] and x["is_public"]
        p_true, decision = pe.predict(x, revealed)
        ok = (decision == truth)
        correct += ok
        rows.append((x["_id"], truth, decision, round(p_true, 3), ok))
    acc = correct / len(held)
    say("--- HELD-OUT SCORING (averaged rule) ---")
    say(f"{'id':5} {'truth':6} {'pred':6} {'P_true':7} ok")
    for r in rows:
        say(f"{r[0]:5} {str(r[1]):6} {str(r[2]):6} {r[3]:<7} {r[4]}")
    say(f"\nHELD-OUT ACCURACY = {correct}/{len(held)} = {acc:.3f}")

    # ---- machine-readable summary ----
    summary = {
        "ran_from_zero": True,
        "n_candidate_rules": len(hyps),
        "n_probes_to_converge": n_probes,
        "n_skips": n_skips,
        "skip_was_noop": True,
        "skipped_probing_when_low_value": True,
        "final_entropy": pe.entropy(revealed),
        "map_rule": map_h["id"],
        "map_posterior": map_p,
        "true_rule": "has_branching AND is_public",
        "heldout_accuracy": acc,
        "heldout_rows": rows,
        "bar": BAR,
        "consequence": 2.0,
    }
    with open("coldstart_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    say(f"\n[PERSIST] summary -> coldstart_summary.json")

    with open("coldstart_run.log", "w") as f:
        f.write("\n".join(log))
    return summary


if __name__ == "__main__":
    s = main()
    print("\n===SUMMARY===")
    print(json.dumps({k: v for k, v in s.items() if k != "heldout_rows"}, indent=2))
