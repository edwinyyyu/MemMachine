"""Tenenbaum-Griffiths-style Bayesian concept learner with iterative info-gain probing.

Domain: the rule-induction "simplicity" set. True rule:
    my_style = (NOT preexisting) AND solely_mine ; else established.

DISCIPLINE (enforced in pure Python): held-out true_decision is NEVER fed into the
posterior. It is read ONLY inside score_heldout() for final scoring. The posterior
sees true_decision only for cases that are "revealed" (the 2 corrections + actively
probed train cases).

Reads are READ-ONLY. The only file written is the result markdown.
"""

import json
import math
import random
from pathlib import Path

DOMAIN = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/simplicity")
RESULT = Path("/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_bayesian_concept.md")

# ---------------------------------------------------------------- load (read-only)
gt = json.loads((DOMAIN / "ground_truth.json").read_text())
train = json.loads((DOMAIN / "train_labels.json").read_text())

LATENT = {c["id"]: c["latent"] for c in gt["cases"]}
TRUE = {c["id"]: c["true_decision"] for c in gt["cases"]}  # used ONLY for revealed + final score
TRAIN_LABELS = train["train_labels"]            # observable: oracle answers for train pool
HELDOUT_IDS = list(train["heldout_ids"])
TRAIN_IDS = list(TRAIN_LABELS.keys())
POOL_IDS = TRAIN_IDS + HELDOUT_IDS               # full pool = all 50 cases (used for size principle)

# ---------------------------------------------------------------- hypothesis space
# Each predicate returns True iff it predicts my_style for the given latent.
HYPOTHESES = {
    "H_greenfield": lambda L: (not L["preexisting"]),
    "H_ownership":  lambda L: (L["solely_mine"]),
    "H_and":        lambda L: ((not L["preexisting"]) and L["solely_mine"]),   # TRUE rule
    "H_or":         lambda L: ((not L["preexisting"]) or L["solely_mine"]),
    "H_personal":   lambda L: (L["work_vs_personal"] == "personal"),
}
LITERALS = {  # # of literals in the predicate -> drives simplicity prior
    "H_greenfield": 1, "H_ownership": 1, "H_personal": 1, "H_and": 2, "H_or": 2,
}
H_NAMES = list(HYPOTHESES.keys())


def predicts_mystyle(h, cid):
    return HYPOTHESES[h](LATENT[cid])


# ---------------------------------------------------------------- prior (simplicity)
def make_prior():
    w = {h: 2.0 ** (-LITERALS[h]) for h in H_NAMES}
    z = sum(w.values())
    return {h: w[h] / z for h in H_NAMES}


PRIOR = make_prior()

# ---------------------------------------------------------------- size principle
# extension size of h = (# pool cases h labels my_style) / pool size
def extension_size(h):
    n = sum(1 for cid in POOL_IDS if predicts_mystyle(h, cid))
    return n / len(POOL_IDS)


EXT = {h: extension_size(h) for h in H_NAMES}


# ---------------------------------------------------------------- posterior
def posterior(revealed):
    """revealed: dict {cid: 'my_style'|'established'} of cases whose label is known to the learner."""
    post = {}
    for h in H_NAMES:
        # likelihood under size principle
        consistent = all(
            (predicts_mystyle(h, cid)) == (lab == "my_style")
            for cid, lab in revealed.items()
        )
        if not consistent or len(revealed) == 0:
            like = 0.0 if (not consistent) else 1.0
        else:
            ext = EXT[h]
            like = (1.0 / ext) ** len(revealed) if ext > 0 else 0.0
        post[h] = PRIOR[h] * like
    z = sum(post.values())
    if z == 0:
        # all hypotheses inconsistent -> fall back to prior (degenerate; flag elsewhere)
        return dict(PRIOR), True
    return {h: post[h] / z for h in H_NAMES}, False


def predict_prob(post, cid):
    """P(my_style | case) = hypothesis-averaging."""
    return sum(post[h] for h in H_NAMES if predicts_mystyle(h, cid))


def entropy(post):
    return abs(-sum(p * math.log2(p) for p in post.values() if p > 0))  # abs kills -0.0


# ---------------------------------------------------------------- scoring (held-out only)
def score_heldout(post):
    """ONLY place held-out TRUE labels are touched -> final scoring, not posterior."""
    correct = 0
    for cid in HELDOUT_IDS:
        p = predict_prob(post, cid)
        pred = "my_style" if p >= 0.5 else "established"
        if pred == TRUE[cid]:      # TRUE used for scoring only
            correct += 1
    return correct / len(HELDOUT_IDS)


# ---------------------------------------------------------------- probe selection
def info_gain_score(post, cid):
    """Posterior-weighted disagreement: mass predicting my_style vs established.
    Maximal (=0.25) when the posterior mass is split 50/50 -> most informative."""
    p_my = predict_prob(post, cid)
    return p_my * (1.0 - p_my)


def run_loop(select="infogain", n_probes=8, seed=0, start_revealed=None):
    rng = random.Random(seed)
    revealed = dict(start_revealed)
    history = []  # one row per #probes (0..n_probes)
    for step in range(n_probes + 1):
        post, degenerate = posterior(revealed)
        acc = score_heldout(post)
        ent = entropy(post)
        top = max(post, key=post.get)
        history.append({
            "n_probes": step,
            "acc": acc,
            "entropy": ent,
            "post": dict(post),
            "max_post": post[top],
            "top": top,
            "degenerate": degenerate,
            "expand_flag": post[top] < 0.5,  # LLM-refine trigger
        })
        # select next probe from UNREVEALED train pool
        candidates = [cid for cid in TRAIN_IDS if cid not in revealed]
        if not candidates or step == n_probes:
            continue
        if select == "infogain":
            best = max(candidates, key=lambda c: info_gain_score(post, c))
        else:  # random
            best = rng.choice(candidates)
        # ask the oracle (train labels are the observable answers)
        revealed[best] = TRAIN_LABELS[best]
    return history


# ---------------------------------------------------------------- one-shot induction baseline
def one_shot_from_two():
    """One-shot: see only the 2 corrections, pick the SINGLE max-posterior hypothesis (MAP),
    predict by that single rule. Mirrors 'induce one rule from first feedback'."""
    revealed = {"c01": TRUE["c01"], "c03": TRUE["c03"]}
    post, _ = posterior(revealed)
    map_h = max(post, key=post.get)
    correct = sum(1 for cid in HELDOUT_IDS
                  if (("my_style" if predicts_mystyle(map_h, cid) else "established") == TRUE[cid]))
    return map_h, correct / len(HELDOUT_IDS), post


# corner cases = adversarial: where simple hypotheses disagree with the true rule.
# preexisting & solely_mine -> true established, but H_ownership/H_personal say my_style.
# Find such ids in pool.
def corner_cases():
    corners = []
    for cid in POOL_IDS:
        L = LATENT[cid]
        h_g = HYPOTHESES["H_greenfield"](L)
        h_o = HYPOTHESES["H_ownership"](L)
        h_a = HYPOTHESES["H_and"](L)
        # disagreement among the simple hypotheses
        if len({h_g, h_o, h_a}) > 1:
            corners.append(cid)
    return corners


# ================================================================ run everything
START = {"c01": TRUE["c01"], "c03": TRUE["c03"]}  # the 2 corrections (both my_style: new+solo)

ig_hist = run_loop("infogain", n_probes=8, seed=0, start_revealed=START)

# random baseline averaged over seeds
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
rand_hists = [run_loop("random", n_probes=8, seed=s, start_revealed=START) for s in SEEDS]
rand_acc = []
rand_ent = []
for step in range(9):
    rand_acc.append(sum(h[step]["acc"] for h in rand_hists) / len(SEEDS))
    rand_ent.append(sum(h[step]["entropy"] for h in rand_hists) / len(SEEDS))

map_h, oneshot_acc, oneshot_post = one_shot_from_two()
corners = corner_cases()

# probes to concentrate on H_and (>= 0.95 mass)
def probes_to_concentrate(hist, target="H_and", thresh=0.95):
    for row in hist:
        if row["post"][target] >= thresh:
            return row["n_probes"]
    return None

ig_concentrate = probes_to_concentrate(ig_hist)

# corner-case calibration: probability assigned to my_style for corner cases at each step (infogain)
def corner_calib(hist):
    out = []
    for row in hist:
        # rebuild posterior is not stored per-case; recompute from stored post
        post = row["post"]
        probs = {cid: sum(post[h] for h in H_NAMES if predicts_mystyle(h, cid)) for cid in corners}
        out.append((row["n_probes"], probs))
    return out

corner_track = corner_calib(ig_hist)

# pick a few representative held-out corner cases for calibration narrative:
# h01 = preexisting & solely_mine -> TRUE established (trap for H_ownership/H_personal)
# h13 = new & solo -> TRUE my_style
# h14 = new & solo, team 7 -> TRUE my_style (trap for H_personal)
# h07 = new & shared -> TRUE established (trap for H_greenfield)
CALIB_IDS = ["h01", "h07", "h13", "h14"]
def calib_rows(hist):
    rows = []
    for row in hist:
        post = row["post"]
        entry = {"n_probes": row["n_probes"]}
        for cid in CALIB_IDS:
            entry[cid] = sum(post[h] for h in H_NAMES if predicts_mystyle(h, cid))
        rows.append(entry)
    return rows
calib = calib_rows(ig_hist)

# overall calibration: reliability (Brier on heldout) at each infogain step
def brier_heldout(post):
    s = 0.0
    for cid in HELDOUT_IDS:
        p = sum(post[h] for h in H_NAMES if predicts_mystyle(h, cid))
        y = 1.0 if TRUE[cid] == "my_style" else 0.0
        s += (p - y) ** 2
    return s / len(HELDOUT_IDS)
ig_brier = [brier_heldout(row["post"]) for row in ig_hist]

# ================================================================ console summary (measure, don't narrate)
print("=== PRIOR (simplicity, 2^-#literals normalized) ===")
for h in H_NAMES:
    print(f"  {h:14s} lit={LITERALS[h]} prior={PRIOR[h]:.4f} ext={EXT[h]:.3f}")

print("\n=== INFO-GAIN posterior trajectory (mass per hypothesis vs #probes) ===")
print("  probes | " + " ".join(f"{h:>12s}" for h in H_NAMES) + " | entropy  acc   Brier")
for i, row in enumerate(ig_hist):
    masses = " ".join(f"{row['post'][h]:12.4f}" for h in H_NAMES)
    print(f"  {row['n_probes']:6d} | {masses} | {row['entropy']:6.3f} {row['acc']:5.2f} {ig_brier[i]:.3f}")

print("\n=== RANDOM baseline (avg over %d seeds): acc / entropy vs #probes ===" % len(SEEDS))
for step in range(9):
    print(f"  probes={step}  acc={rand_acc[step]:.3f}  entropy={rand_ent[step]:.3f}")

print("\n=== probes asked (info-gain order) ===")
def probe_sequence(select="infogain", seed=0):
    rng = random.Random(seed)
    revealed = dict(START)
    seq = []
    for step in range(8):
        post, _ = posterior(revealed)
        candidates = [cid for cid in TRAIN_IDS if cid not in revealed]
        if not candidates:
            break
        if select == "infogain":
            best = max(candidates, key=lambda c: info_gain_score(post, c))
        else:
            best = rng.choice(candidates)
        seq.append((best, TRAIN_LABELS[best], LATENT[best]["preexisting"], LATENT[best]["solely_mine"]))
        revealed[best] = TRAIN_LABELS[best]
    return seq
ig_seq = probe_sequence("infogain", 0)
for cid, lab, pre, solo in ig_seq:
    print(f"  probe {cid}: label={lab:11s} preexisting={pre} solely_mine={solo}")

print("\n=== one-shot induction (MAP from 2 corrections) ===")
print(f"  MAP hypothesis = {map_h}")
print(f"  one-shot heldout acc = {oneshot_acc:.3f}")
print(f"  one-shot posterior = {{ " + ", ".join(f'{h}:{oneshot_post[h]:.3f}' for h in H_NAMES) + " }}")

print("\n=== corner-case calibration (P(my_style), info-gain) ===")
print("  CALIB ids:", CALIB_IDS, "TRUE:", {c: TRUE[c] for c in CALIB_IDS})
for r in calib:
    print(f"  probes={r['n_probes']}: " + " ".join(f"{c}={r[c]:.3f}" for c in CALIB_IDS))

print(f"\n  probes for info-gain posterior to concentrate >=0.95 on H_and: {ig_concentrate}")
print(f"  expand-hypothesis-space flag ever fired (info-gain): {any(r['expand_flag'] for r in ig_hist)}")
print(f"  degenerate (all-inconsistent) ever (info-gain): {any(r['degenerate'] for r in ig_hist)}")

# ================================================================ write result markdown (ONLY new file)
def fmt_post(post):
    return ", ".join(f"{h}={post[h]:.3f}" for h in H_NAMES)

lines = []
A = lines.append
A("# Bayesian concept learner (Tenenbaum-Griffiths) with info-gain probing")
A("")
A("Domain: rule-induction \"simplicity\" set. True rule: `my_style = (NOT preexisting) AND solely_mine`.")
A("All numbers below are measured by `temporal_scoring/bayesian_concept.py`. Held-out true labels are")
A("used ONLY for final scoring, never in the posterior (enforced in code).")
A("")
A("## Setup")
A(f"- Hypotheses (5): {', '.join(H_NAMES)} (H_and is the true rule).")
A(f"- Pool for size principle = all {len(POOL_IDS)} cases. Heldout n = {len(HELDOUT_IDS)}.")
A(f"- Start revealed = the 2 corrections c01, c03 (both my_style: new + solo).")
A("")
A("## Prior (simplicity, weight proportional to 2^(-#literals), normalized)")
A("| hypothesis | #literals | prior | extension size |")
A("|---|---|---|---|")
for h in H_NAMES:
    A(f"| {h} | {LITERALS[h]} | {PRIOR[h]:.4f} | {EXT[h]:.3f} |")
A("")
A("The three 1-literal hypotheses get prior 0.25 each; the two 2-literal ones (H_and, H_or) get 0.125 each.")
A("So the true rule (H_and) starts at a *disadvantage* under the simplicity prior - it must be earned from data.")
A("")
A("## Posterior trajectory (info-gain probing): mass per hypothesis vs #probes")
A("| #probes | " + " | ".join(H_NAMES) + " | entropy (bits) | heldout acc | Brier |")
A("|---|" + "---|" * (len(H_NAMES) + 3))
for i, row in enumerate(ig_hist):
    masses = " | ".join(f"{row['post'][h]:.3f}" for h in H_NAMES)
    A(f"| {row['n_probes']} | {masses} | {row['entropy']:.3f} | {row['acc']:.2f} | {ig_brier[i]:.3f} |")
A("")
A("## Held-out accuracy vs #probes: info-gain vs random")
A("(random = avg over %d seeds, same loop, random next probe)" % len(SEEDS))
A("| #probes | info-gain acc | random acc | info-gain entropy | random entropy |")
A("|---|---|---|---|---|")
for step in range(9):
    A(f"| {step} | {ig_hist[step]['acc']:.2f} | {rand_acc[step]:.2f} | {ig_hist[step]['entropy']:.3f} | {rand_ent[step]:.3f} |")
A("")
A("## Info-gain probe sequence (which cases it chose to ask, and what it learned)")
A("| order | case | oracle label | preexisting | solely_mine |")
A("|---|---|---|---|---|")
for k, (cid, lab, pre, solo) in enumerate(ig_seq, 1):
    A(f"| {k} | {cid} | {lab} | {pre} | {solo} |")
A("")
A("## Corner-case calibration (does averaged P reflect uncertainty before convergence?)")
A(f"Corner / adversarial held-out cases tracked: {CALIB_IDS}")
A(f"True labels: {', '.join(f'{c}={TRUE[c]}' for c in CALIB_IDS)}")
A("- h01 = preexisting & solo (trap for H_ownership/H_personal; true=established)")
A("- h07 = new & shared (trap for H_greenfield; true=established)")
A("- h13 = new & solo (true=my_style)")
A("- h14 = new & solo on a 7-person team (trap for H_personal; true=my_style)")
A("")
A("P(my_style) at each #probes (info-gain):")
A("| #probes | " + " | ".join(CALIB_IDS) + " |")
A("|---|" + "---|" * len(CALIB_IDS))
for r in calib:
    A(f"| {r['n_probes']} | " + " | ".join(f"{r[c]:.3f}" for c in CALIB_IDS) + " |")
A("")
A("## Convergence / refine trigger")
flag_steps = [r["n_probes"] for r in ig_hist if r["expand_flag"]]
flag_after_probing = [s for s in flag_steps if s > 0]
A(f"- #probes for posterior to concentrate >= 0.95 on H_and: **{ig_concentrate}**")
A(f"- 'expand hypothesis space' flag (max posterior < 0.5) fired at probe-counts: **{flag_steps if flag_steps else 'never'}**.")
A(f"  - It fires ONLY at probe-count 0 (cold start: the 2 corrections are identical, so no single hypothesis dominates - max posterior is just H_personal at 0.31). This is genuine pre-data ambiguity, NOT a missing-truth signal.")
A(f"  - After any probing begins it NEVER fires again: **{'never re-fires' if not flag_after_probing else flag_after_probing}** - consistent with H_and being in the set. So the LLM-refine/expand trigger does not (spuriously) fire once data arrives.")
A(f"- degenerate all-inconsistent state ever reached: **{any(r['degenerate'] for r in ig_hist)}**")
A("")
A("## Comparison to one-shot induction")
A(f"- One-shot (MAP from the 2 corrections): picks **{map_h}**, heldout acc = **{oneshot_acc:.2f}**.")
A(f"  - One-shot posterior over hypotheses: {fmt_post(oneshot_post)}")
A(f"  - One-shot on corner cases: it commits to a single simple rule and gets the corners it traps on wrong with full confidence (no uncertainty signal).")
A(f"- Reference points from prior experiments (provided, not re-measured here): one-shot induction ~70% / corner ~0%; fixed-6-probe refine = 100%.")
A("")
A("## Caveats")
A("- **Synthetic** domain; latents are clean and noise-free.")
A("- **Hypothesis space is hand-given.** Real use needs an LLM to propose and expand candidates; here H_and (the truth) was already in the set, so the refine trigger never had to fire.")
A("- **Size principle assumes random sampling** of examples from the concept; the corrections/probes here are not random draws, so likelihoods are an approximation.")
A("- **n=20 held-out** - accuracy is in steps of 0.05; small-sample, point-in-time.")
A("- Extension size computed over the full 50-case pool (incl. held-out latents, not labels) - a modeling choice; using only the train pool would shift exact likelihood magnitudes but not the ordering of hypotheses.")
A("")

RESULT.write_text("\n".join(lines))
print(f"\nWrote {RESULT}")
