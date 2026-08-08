"""Balance test: does Bayesian concept learning balance breadth vs tightness?

Contrast the existing balanced Bayesian learner (simplicity prior x size-principle
likelihood, hypothesis-averaging, info-gain probing) against a PURE-TIGHTEST baseline
(among consistent hypotheses always pick the SMALLEST extension, ignore the prior).

Two oracle worlds, labels RECOMPUTED from latents (the stored true_decision is the
OTHER rule -- "established = preexisting OR NOT solely_mine" -- and is NOT used):
  TIGHT world : true rule = H_and       (my_style = (NOT preexisting) AND solely_mine)
  BROAD world : true rule = H_ownership (my_style = solely_mine)

Seed = 2 ambiguous corrections (c01, c03): both new+solo -> my_style, consistent with
BOTH H_and and H_ownership.

Reuses HYPOTHESES / LITERALS / prior / size principle / posterior / hypothesis-averaging
/ info-gain probing from bayesian_concept.py. Reads are READ-ONLY; only new file written
is the result markdown.
"""

import json
import math
import random
from pathlib import Path

# ---- reuse the existing machinery (no edits to the source file) -----------------
# Importing bayesian_concept runs its module-level analysis, which would rewrite its own
# result markdown. We must not clobber that existing result file ("new result files only"),
# so we neutralize Path.write_text DURING the import only, then restore it.
from pathlib import Path as _Path
_orig_write_text = _Path.write_text
_Path.write_text = lambda self, *a, **k: len(a[0]) if a else 0  # no-op during import
try:
    import bayesian_concept as bc
finally:
    _Path.write_text = _orig_write_text

H_NAMES = bc.H_NAMES
HYPOTHESES = bc.HYPOTHESES
LITERALS = bc.LITERALS
PRIOR = bc.PRIOR                      # simplicity prior, 2^-#literals normalized
EXT = bc.EXT                          # extension size over the full 50-case pool
LATENT = bc.LATENT
TRAIN_IDS = bc.TRAIN_IDS
HELDOUT_IDS = bc.HELDOUT_IDS
POOL_IDS = bc.POOL_IDS
predicts_mystyle = bc.predicts_mystyle
posterior = bc.posterior             # prior x size-principle likelihood, normalized
predict_prob = bc.predict_prob       # hypothesis-averaging P(my_style|case)
entropy = bc.entropy
info_gain_score = bc.info_gain_score  # posterior-weighted disagreement p(1-p)

RESULT = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_balance_test.md")

# ---- two oracle worlds: relabel every case from its latent ----------------------
ORACLES = {
    "TIGHT": HYPOTHESES["H_and"],        # true rule = H_and
    "BROAD": HYPOTHESES["H_ownership"],   # true rule = H_ownership
}


def oracle_label(world, cid):
    return "my_style" if ORACLES[world](LATENT[cid]) else "established"


# seed = the 2 ambiguous corrections (both new+solo). Their oracle label is my_style in
# BOTH worlds (H_and and H_ownership agree here -> genuinely ambiguous seed).
SEED_IDS = ["c01", "c03"]


def seed_revealed(world):
    return {cid: oracle_label(world, cid) for cid in SEED_IDS}


# ---- controller A: balanced Bayesian (reuse posterior + hypothesis-averaging) ----
def balanced_predict_prob(revealed, cid):
    post, _ = posterior(revealed)
    return predict_prob(post, cid)


def balanced_posterior(revealed):
    return posterior(revealed)


# ---- controller B: pure-tightest (consistent + smallest extension, ignore prior) -
def tightest_hypothesis(revealed):
    """Among hypotheses CONSISTENT with revealed examples, pick the SMALLEST extension.
    Ignores the prior. Ties broken by fewer literals then name (deterministic)."""
    consistent = [
        h for h in H_NAMES
        if all((predicts_mystyle(h, cid)) == (lab == "my_style")
               for cid, lab in revealed.items())
    ]
    if not consistent:
        return None
    return min(consistent, key=lambda h: (EXT[h], LITERALS[h], h))


def tightest_predict(revealed, cid):
    h = tightest_hypothesis(revealed)
    if h is None:
        return 0.5  # degenerate; no consistent hypothesis
    return 1.0 if predicts_mystyle(h, cid) else 0.0


# ---- held-out scoring against a given world's oracle (the ONLY label use for score)
def score_heldout(world, predict_prob_fn, revealed):
    correct = 0
    for cid in HELDOUT_IDS:
        p = predict_prob_fn(revealed, cid)
        pred = "my_style" if p >= 0.5 else "established"
        if pred == oracle_label(world, cid):
            correct += 1
    return correct / len(HELDOUT_IDS)


# ---- discriminating corner: old+solo (preexisting AND solely_mine) = h01..h06 -----
# BROAD truth = my_style ; H_and predicts established -> pure-tightest undergeneralizes.
CORNER_IDS = [cid for cid in HELDOUT_IDS
              if LATENT[cid]["preexisting"] and LATENT[cid]["solely_mine"]]


def corner_accuracy(world, predict_prob_fn, revealed):
    correct = 0
    for cid in CORNER_IDS:
        p = predict_prob_fn(revealed, cid)
        pred = "my_style" if p >= 0.5 else "established"
        if pred == oracle_label(world, cid):
            correct += 1
    return correct / len(CORNER_IDS)


# ---- info-gain probing loop (reuse info_gain_score), per world & controller -------
def run_loop(world, controller, n_probes=8, seed=0):
    """controller in {'balanced','tightest'}. Probes asked from UNREVEALED train pool.
    Oracle answers come from the world's recomputed labels (NOT the stored file)."""
    rng = random.Random(seed)
    revealed = dict(seed_revealed(world))

    if controller == "balanced":
        predict_fn = balanced_predict_prob
    else:
        predict_fn = tightest_predict

    history = []
    for step in range(n_probes + 1):
        post, degenerate = posterior(revealed)  # always track Bayesian posterior for reporting
        row = {
            "n_probes": step,
            "acc": score_heldout(world, predict_fn, revealed),
            "corner_acc": corner_accuracy(world, predict_fn, revealed),
            "post": dict(post),
            "entropy": entropy(post),
            "tightest_h": tightest_hypothesis(revealed),
            "degenerate": degenerate,
        }
        history.append(row)
        # choose next probe. Info-gain uses the Bayesian posterior disagreement signal
        # (same probe selector for both controllers, so probe budget is comparable).
        candidates = [c for c in TRAIN_IDS if c not in revealed]
        if not candidates or step == n_probes:
            continue
        best = max(candidates, key=lambda c: info_gain_score(post, c))
        revealed[best] = oracle_label(world, best)
    return history


# ================================================================================
N_PROBES = 8
results = {}
for world in ("TIGHT", "BROAD"):
    results[world] = {
        "balanced": run_loop(world, "balanced", n_probes=N_PROBES),
        "tightest": run_loop(world, "tightest", n_probes=N_PROBES),
    }


# ---- posterior split after the 2 ambiguous corrections (BROAD world) --------------
seed_post_broad, _ = posterior(seed_revealed("BROAD"))
seed_post_tight, _ = posterior(seed_revealed("TIGHT"))  # identical: same seed labels
seed_tightest_broad = tightest_hypothesis(seed_revealed("BROAD"))


def converged_hypothesis(world, controller, target, thresh=0.9):
    """#probes for the controller to commit to `target` hypothesis.
    balanced: posterior mass on target >= thresh. tightest: tightest_h == target."""
    hist = results[world][controller]
    for row in hist:
        if controller == "balanced":
            if row["post"][target] >= thresh:
                return row["n_probes"]
        else:
            if row["tightest_h"] == target:
                return row["n_probes"]
    return None


# probes for balanced to recover the corner (corner_acc == 1.0) in BROAD world
def probes_to_corner(world, controller):
    for row in results[world][controller]:
        if row["corner_acc"] >= 0.999:
            return row["n_probes"]
    return None


# ================================================================================ console
def pp(x):
    return f"{x:.3f}"

print("Corner (old+solo, preexisting AND solely_mine) heldout ids:", CORNER_IDS)
print("  BROAD-world truth for corner:", [oracle_label('BROAD', c) for c in CORNER_IDS])
print("  TIGHT-world truth for corner:", [oracle_label('TIGHT', c) for c in CORNER_IDS])
print()

print("=== Seed posterior after 2 ambiguous corrections (c01,c03 new+solo->my_style) ===")
print("  (seed labels identical in both worlds; posterior is the same)")
for h in H_NAMES:
    print(f"  {h:14s} prior={PRIOR[h]:.3f} ext={EXT[h]:.3f} post={seed_post_broad[h]:.4f}")
print(f"  --> H_ownership (broad,true) = {seed_post_broad['H_ownership']:.4f}  "
      f"H_and (tight) = {seed_post_broad['H_and']:.4f}")
print(f"  --> pure-tightest pick at seed = {seed_tightest_broad}  "
      f"(ext={EXT[seed_tightest_broad]:.3f})")
print()

for world in ("TIGHT", "BROAD"):
    print(f"=== {world} world: held-out accuracy & corner accuracy vs #probes ===")
    print("  probes | bal_acc bal_corner | tight_acc tight_corner | "
          "post(H_own) post(H_and) bal_top  tightest_h")
    for i in range(N_PROBES + 1):
        b = results[world]["balanced"][i]
        t = results[world]["tightest"][i]
        bal_top = max(b["post"], key=b["post"].get)
        print(f"  {i:6d} | {b['acc']:.2f}    {b['corner_acc']:.2f}       | "
              f"{t['acc']:.2f}      {t['corner_acc']:.2f}         | "
              f"{b['post']['H_ownership']:.3f}      {b['post']['H_and']:.3f}      "
              f"{bal_top:12s} {t['tightest_h']}")
    print()

print("=== Adaptation check (does the SAME balanced method track the truth?) ===")
for world, target in (("TIGHT", "H_and"), ("BROAD", "H_ownership")):
    n_bal = converged_hypothesis(world, "balanced", target, thresh=0.9)
    n_tight = converged_hypothesis(world, "tightest", target)
    print(f"  {world} world (true={target}): balanced reaches >=0.9 mass on truth at "
          f"#probes={n_bal}; tightest selects truth at #probes={n_tight}")
print()
print("  BROAD-world #probes for balanced to recover corner (corner_acc=1.0):",
      probes_to_corner("BROAD", "balanced"))
print("  BROAD-world #probes for tightest to recover corner (corner_acc=1.0):",
      probes_to_corner("BROAD", "tightest"))


# ================================================================================ markdown
L = []
A = L.append
A("# Balance test: does Bayesian concept learning balance breadth vs tightness?")
A("")
A("Contrasts the **balanced Bayesian** learner (simplicity prior x size-principle")
A("likelihood, hypothesis-averaging, info-gain probing) against a **pure-tightest**")
A("baseline (among consistent hypotheses, always pick the SMALLEST extension; ignore")
A("the prior). Goal: show the balanced method does not *undergeneralize* when the true")
A("rule is BROAD, where pure-tightest does.")
A("")
A("All numbers measured by `temporal_scoring/balance_test.py`, reusing the hypotheses,")
A("prior, size-principle likelihood, posterior, hypothesis-averaging, and info-gain")
A("probing from `bayesian_concept.py`. Labels are **recomputed from latents** under each")
A("oracle world (the file's stored `true_decision` encodes a different rule and is NOT")
A("used). Held-out labels are touched ONLY for final scoring, never in the posterior.")
A("")
A("## Two oracle worlds")
A("| world | true rule | predicate |")
A("|---|---|---|")
A("| TIGHT | H_and | `my_style = (NOT preexisting) AND solely_mine` |")
A("| BROAD | H_ownership | `my_style = solely_mine` |")
A("")
A("Seed = 2 ambiguous corrections **c01, c03** (both new+solo -> my_style), consistent")
A("with BOTH H_and and H_ownership in either world.")
A("")
A("Hypothesis space (5, reused): " + ", ".join(H_NAMES) + ".")
A("Extension sizes (over the 50-case pool): " +
  ", ".join(f"{h}={EXT[h]:.3f}" for h in H_NAMES) + ".")
A("")
A("## Posterior split after the 2 ambiguous corrections")
A("(Seed labels are identical in both worlds, so the seed posterior is identical too.)")
A("")
A("| hypothesis | #literals | prior | extension | posterior after seed |")
A("|---|---|---|---|---|")
for h in H_NAMES:
    A(f"| {h} | {LITERALS[h]} | {PRIOR[h]:.3f} | {EXT[h]:.3f} | {seed_post_broad[h]:.4f} |")
A("")
A(f"- Mass on **H_ownership** (broad, BROAD-world truth) = **{seed_post_broad['H_ownership']:.4f}**")
A(f"- Mass on **H_and** (tight, TIGHT-world truth) = **{seed_post_broad['H_and']:.4f}**")
A(f"- Pure-tightest commits to **{seed_tightest_broad}** (smallest consistent extension, "
  f"ext={EXT[seed_tightest_broad]:.3f}), ignoring the prior.")
A("")
ratio = (seed_post_broad['H_ownership'] / seed_post_broad['H_and']
         if seed_post_broad['H_and'] > 0 else float('inf'))
A(f"The simplicity prior keeps **meaningful mass on the broad true rule**: the seed")
A(f"posterior puts H_ownership / H_and = {ratio:.2f}x. Both broad and tight stay live")
A(f"under hypothesis-averaging, whereas pure-tightest has already committed to the")
A(f"single tightest rule ({seed_tightest_broad}) and discarded the broad alternative.")
A("")
A("## Held-out accuracy & corner accuracy vs #probes (both worlds, both controllers)")
A("Corner = **old+solo** (preexisting AND solely_mine) held-out cases " + str(CORNER_IDS) + ":")
A("BROAD-world truth there = my_style; H_and predicts established.")
A("`post(H_own)`/`post(H_and)` are the Bayesian posterior masses; `tightest_h` is the")
A("hypothesis the pure-tightest controller currently commits to.")
A("")
for world in ("TIGHT", "BROAD"):
    A(f"### {world} world (true rule = {'H_and' if world=='TIGHT' else 'H_ownership'})")
    A("| #probes | balanced acc | balanced corner | tightest acc | tightest corner | "
      "post(H_own) | post(H_and) | balanced top | tightest_h |")
    A("|---|---|---|---|---|---|---|---|---|")
    for i in range(N_PROBES + 1):
        b = results[world]["balanced"][i]
        t = results[world]["tightest"][i]
        bal_top = max(b["post"], key=b["post"].get)
        A(f"| {i} | {b['acc']:.2f} | {b['corner_acc']:.2f} | {t['acc']:.2f} | "
          f"{t['corner_acc']:.2f} | {b['post']['H_ownership']:.3f} | "
          f"{b['post']['H_and']:.3f} | {bal_top} | {t['tightest_h']} |")
    A("")
A("## The old+solo corner result")
A(f"Corner cases (old+solo): {CORNER_IDS}")
A("")
n_corner_bal = probes_to_corner("BROAD", "balanced")
n_corner_t = probes_to_corner("BROAD", "tightest")
A("- **BROAD world** (truth there = my_style):")
A(f"  - **At 0 probes (prediction from the ambiguous seed alone)** pure-tightest corner "
  f"accuracy = **{results['BROAD']['tightest'][0]['corner_acc']:.2f}** "
  f"(commits to H_and -> calls every old+solo case established when the BROAD truth is "
  f"my_style: classic **undergeneralization**), while balanced corner accuracy = "
  f"**{results['BROAD']['balanced'][0]['corner_acc']:.2f}** "
  f"(hypothesis-averaging keeps H_ownership/H_or/H_personal mass on my_style; it misses "
  f"only the 1 *work* old+solo case, where H_personal does not back my_style).")
A(f"  - Under the **shared info-gain probe selector**, the first probe chosen is the "
  f"disambiguating old+solo case c23 (BROAD label = my_style), which rules out H_and. So "
  f"pure-tightest is *forced* off H_and too: corner acc reaches 1.00 at "
  f"**#probes={n_corner_t}**, and balanced reaches 1.00 at **#probes={n_corner_bal}**. "
  f"The undergeneralization gap therefore lives in the **prediction rule before the "
  f"disambiguating probe arrives**, not in a permanent failure -- here the probe selector "
  f"does the disambiguation work for both. If the disambiguating old+solo example were "
  f"never served, pure-tightest would stay at corner acc 0.00 in BROAD.")
A("- **TIGHT world** (truth there = established for old+solo):")
A(f"  - both controllers are correct on the corner: balanced corner acc = "
  f"{results['TIGHT']['balanced'][N_PROBES]['corner_acc']:.2f}, tightest corner acc = "
  f"{results['TIGHT']['tightest'][N_PROBES]['corner_acc']:.2f} (tight rule says established, "
  f"which is the TIGHT truth).")
A("")
A("## Adaptation: same balanced method, opposite truths")
A("| world | true rule | balanced reaches >=0.9 mass on truth at #probes | "
  "pure-tightest selects truth at #probes |")
A("|---|---|---|---|")
for world, target in (("TIGHT", "H_and"), ("BROAD", "H_ownership")):
    n_bal = converged_hypothesis(world, "balanced", target, thresh=0.9)
    n_tight = converged_hypothesis(world, "tightest", target)
    A(f"| {world} | {target} | {n_bal} | {n_tight if n_tight is not None else 'never'} |")
A("")
A("## Verdict")
broad_bal0 = results['BROAD']['balanced'][0]['acc']
broad_balN = results['BROAD']['balanced'][N_PROBES]['acc']
broad_t0 = results['BROAD']['tightest'][0]['acc']
broad_tN = results['BROAD']['tightest'][N_PROBES]['acc']
tight_balN = results['TIGHT']['balanced'][N_PROBES]['acc']
tight_tN = results['TIGHT']['tightest'][N_PROBES]['acc']
A(f"- TIGHT world: balanced reaches {tight_balN:.2f} held-out acc, pure-tightest "
  f"{tight_tN:.2f} -> both fine; tightest does NOT overgeneralize when truth is tight.")
A(f"- BROAD world, **prediction from the seed (0 probes)**: balanced held-out acc "
  f"{broad_bal0:.2f} (corner {results['BROAD']['balanced'][0]['corner_acc']:.2f}) vs "
  f"pure-tightest {broad_t0:.2f} (corner "
  f"{results['BROAD']['tightest'][0]['corner_acc']:.2f}). Pure-tightest undergeneralizes "
  f"on the old+solo corner; balanced does not. After info-gain probing both reach "
  f"{broad_balN:.2f}, because the shared selector serves the disambiguating old+solo "
  f"probe (c23) -- so the difference is the prediction rule pre-disambiguation, not a "
  f"permanent gap under this probing setup.")
A(f"- The **same** balanced method adapts to whichever rule is true (concentrates on "
  f"H_and in TIGHT, on H_ownership in BROAD); pure-tightest predicts correctly from the "
  f"seed only when the truth happens to be the tightest consistent hypothesis (TIGHT), and "
  f"undergeneralizes from the seed when the truth is broad (BROAD). So the simplicity prior "
  f"+ size principle + hypothesis-averaging is a **genuine balance**: from the same "
  f"ambiguous seed it avoids BOTH overgeneralization (TIGHT: it does not wrongly predict "
  f"my_style for old+solo) and undergeneralization (BROAD: it does predict my_style for "
  f"old+solo), the corner where pure-tightest fails.")
A("")
A("## Caveats")
A("- Synthetic domain; latents are clean and **noise-free** (probe answers are exact).")
A("- Hypothesis space is **hand-given** (5 fixed predicates); both true rules are in it.")
A("- n=20 held-out (acc in steps of 0.05); n=6 corner cases. Point-in-time.")
A("- Probe selector (info-gain) uses the Bayesian posterior for BOTH controllers, so the")
A("  probe budget is shared; pure-tightest's failure is in its prediction rule, not its")
A("  probe choices.")
A("- Size principle assumes random sampling from the concept; corrections/probes are not")
A("  random draws, so likelihoods are an approximation (same caveat as the source model).")
A("")
RESULT.write_text("\n".join(L))
print(f"\nWrote {RESULT}")
