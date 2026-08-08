"""Score the rule-induction / over-simplification generalization experiment.

READ-ONLY on inputs in feedback_retrieval/simplicity/.
Writes only new result files:
  - feedback_retrieval/results_simplicity.md
  - temporal_scoring/rf_simplicity_results.json (machine-readable dump)
  - temporal_scoring/rf_simplicity_embcache.npz (embedding cache; new file)
"""
import json
import os
from collections import defaultdict

import numpy as np

SIMP = "/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/simplicity"
GT = os.path.join(SIMP, "ground_truth.json")
DESC = os.path.join(SIMP, "descriptions.json")
TRAIN = os.path.join(SIMP, "train_labels.json")
PRED = os.path.join(SIMP, "predictions.json")

OUT_MD = "/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/results_simplicity.md"
OUT_JSON = "/Users/eyu/edwinyyyu/mmcc/temporal_scoring/rf_simplicity_results.json"
EMB_CACHE = "/Users/eyu/edwinyyyu/mmcc/temporal_scoring/rf_simplicity_embcache.npz"


def stated_rule(latent):
    """established = preexisting OR (NOT solely_mine); else my_style."""
    if latent["preexisting"] or (not latent["solely_mine"]):
        return "established"
    return "my_style"


def main():
    gt = json.load(open(GT))
    desc = json.load(open(DESC))
    train = json.load(open(TRAIN))
    pred = json.load(open(PRED))

    cases = {c["id"]: c for c in gt["cases"]}
    descriptions = {c["id"]: c["description"] for c in desc["cases"]}
    train_labels = train["train_labels"]
    heldout_ids = list(train["heldout_ids"])

    results = {}

    # ---------- STEP 0: INTEGRITY ----------
    rule_mismatches = []  # stated-rule(latent) != true_decision
    for cid, c in cases.items():
        recomputed = stated_rule(c["latent"])
        if recomputed != c["true_decision"]:
            rule_mismatches.append(
                {"id": cid, "latent": c["latent"],
                 "recomputed": recomputed, "true_decision": c["true_decision"]}
            )

    # train_labels vs ground_truth true_decision
    train_mismatches = []
    for cid, lbl in train_labels.items():
        if cid not in cases:
            train_mismatches.append({"id": cid, "issue": "not in ground_truth"})
            continue
        if lbl != cases[cid]["true_decision"]:
            train_mismatches.append(
                {"id": cid, "train_label": lbl,
                 "true_decision": cases[cid]["true_decision"]}
            )

    # also: do train_labels match the stated rule recomputation?
    train_vs_rule = []
    for cid, lbl in train_labels.items():
        if cid in cases:
            rc = stated_rule(cases[cid]["latent"])
            if lbl != rc:
                train_vs_rule.append({"id": cid, "train_label": lbl, "rule": rc})

    results["integrity"] = {
        "n_cases": len(cases),
        "rule_vs_truth_mismatches": rule_mismatches,
        "n_rule_vs_truth_mismatches": len(rule_mismatches),
        "train_vs_truth_mismatches": train_mismatches,
        "n_train_vs_truth_mismatches": len(train_mismatches),
        "train_vs_rule_mismatches": train_vs_rule,
        "n_train_vs_rule_mismatches": len(train_vs_rule),
    }

    # ---------- truth on heldout ----------
    truth = {h: cases[h]["true_decision"] for h in heldout_ids}

    # corner cases (over-simplification probe), restricted to heldout:
    # A) preexisting AND solely_mine  -> true established; ownership-only rule says my_style
    # B) (NOT preexisting) AND (NOT solely_mine) -> true established; greenfield-only rule says my_style
    corner_A = [h for h in heldout_ids
                if cases[h]["latent"]["preexisting"] and cases[h]["latent"]["solely_mine"]]
    corner_B = [h for h in heldout_ids
                if (not cases[h]["latent"]["preexisting"]) and (not cases[h]["latent"]["solely_mine"])]
    corner_ids = set(corner_A) | set(corner_B)
    easy_ids = [h for h in heldout_ids if h not in corner_ids]

    results["corner_definition"] = {
        "corner_A_preexisting_and_solely_mine": corner_A,
        "corner_B_greenfield_and_shared": corner_B,
        "all_corner_ids": sorted(corner_ids),
        "easy_ids": easy_ids,
        "n_corner": len(corner_ids),
        "n_easy": len(easy_ids),
    }

    # ---------- STEP 1 + STEP 2 + STEP 4: per-run ----------
    def classify_rule(text):
        t = text.lower()
        mentions_owner = any(k in t for k in
                             ["solo", "personal", "only i", "others contribute",
                              "shared codebase", "house style", "collaborative",
                              "ever touch", "no one else", "alone"])
        mentions_green = any(k in t for k in
                             ["brand-new", "brand new", "empty repo", "already has code",
                              "already had", "new/empty", "greenfield", "exists yet",
                              "repo already"])
        if mentions_owner and mentions_green:
            return "both"
        if mentions_owner:
            return "ownership_only"
        if mentions_green:
            return "greenfield_only"
        return "other"

    per_run = []
    by_K = defaultdict(list)
    rule_recovery_by_K = defaultdict(lambda: defaultdict(int))

    for run in pred["runs"]:
        K = run["K"]
        preds = run["heldout_predictions"]
        # overall accuracy
        correct = sum(1 for h in heldout_ids if preds.get(h) == truth[h])
        acc = correct / len(heldout_ids)
        # corner vs easy
        c_corr = sum(1 for h in corner_ids if preds.get(h) == truth[h])
        c_acc = c_corr / len(corner_ids) if corner_ids else float("nan")
        e_corr = sum(1 for h in easy_ids if preds.get(h) == truth[h])
        e_acc = e_corr / len(easy_ids) if easy_ids else float("nan")
        # corner A / B separately
        ca_corr = sum(1 for h in corner_A if preds.get(h) == truth[h])
        ca_acc = ca_corr / len(corner_A) if corner_A else float("nan")
        cb_corr = sum(1 for h in corner_B if preds.get(h) == truth[h])
        cb_acc = cb_corr / len(corner_B) if corner_B else float("nan")

        rclass = classify_rule(run["induced_rule"])
        rule_recovery_by_K[K][rclass] += 1

        rec = {
            "K": K, "subset_index": run["subset_index"],
            "subset_ids": run["subset_ids"],
            "induced_rule": run["induced_rule"],
            "rule_class": rclass,
            "accuracy": acc, "n_correct": correct,
            "corner_acc": c_acc, "corner_correct": c_corr, "n_corner": len(corner_ids),
            "easy_acc": e_acc, "easy_correct": e_corr, "n_easy": len(easy_ids),
            "cornerA_acc": ca_acc, "cornerB_acc": cb_acc,
        }
        per_run.append(rec)
        by_K[K].append(rec)

    # aggregate by K
    K_curve = []
    for K in sorted(by_K):
        recs = by_K[K]
        K_curve.append({
            "K": K, "n_subsets": len(recs),
            "mean_acc": float(np.mean([r["accuracy"] for r in recs])),
            "mean_corner_acc": float(np.mean([r["corner_acc"] for r in recs])),
            "mean_easy_acc": float(np.mean([r["easy_acc"] for r in recs])),
            "mean_cornerA_acc": float(np.mean([r["cornerA_acc"] for r in recs])),
            "mean_cornerB_acc": float(np.mean([r["cornerB_acc"] for r in recs])),
        })

    results["per_run"] = per_run
    results["K_curve"] = K_curve
    results["rule_recovery_by_K"] = {
        str(K): dict(rule_recovery_by_K[K]) for K in sorted(rule_recovery_by_K)
    }

    # ---------- STEP 3: baselines ----------
    # majority-class on train labels
    cnt = defaultdict(int)
    for lbl in train_labels.values():
        cnt[lbl] += 1
    majority_label = max(cnt, key=cnt.get)
    maj_correct = sum(1 for h in heldout_ids if majority_label == truth[h])
    maj_acc = maj_correct / len(heldout_ids)
    # majority on corners / easy
    maj_corner = sum(1 for h in corner_ids if majority_label == truth[h]) / len(corner_ids) if corner_ids else float("nan")
    maj_easy = sum(1 for h in easy_ids if majority_label == truth[h]) / len(easy_ids) if easy_ids else float("nan")

    results["baseline_majority"] = {
        "train_label_counts": dict(cnt),
        "majority_label": majority_label,
        "accuracy": maj_acc, "n_correct": maj_correct,
        "corner_acc": maj_corner, "easy_acc": maj_easy,
    }

    # NN-embedding baseline
    train_ids = list(train_labels.keys())
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("google/embeddinggemma-300m", device="mps")
    train_texts = [descriptions[t] for t in train_ids]
    held_texts = [descriptions[h] for h in heldout_ids]
    train_emb = model.encode(train_texts, prompt_name="document",
                             normalize_embeddings=True, batch_size=32).astype(np.float32)
    held_emb = model.encode(held_texts, prompt_name="query",
                            normalize_embeddings=True, batch_size=32).astype(np.float32)
    np.savez(EMB_CACHE, train_emb=train_emb, held_emb=held_emb,
             train_ids=np.array(train_ids), held_ids=np.array(heldout_ids))

    sims = held_emb @ train_emb.T  # (n_held, n_train), cosine since normalized
    nn_idx = sims.argmax(axis=1)
    nn_preds = {}
    nn_detail = []
    for i, h in enumerate(heldout_ids):
        nt = train_ids[nn_idx[i]]
        nn_preds[h] = train_labels[nt]
        nn_detail.append({"heldout": h, "nearest_train": nt,
                          "sim": float(sims[i, nn_idx[i]]),
                          "pred": train_labels[nt], "truth": truth[h]})
    nn_correct = sum(1 for h in heldout_ids if nn_preds[h] == truth[h])
    nn_acc = nn_correct / len(heldout_ids)
    nn_corner = sum(1 for h in corner_ids if nn_preds[h] == truth[h]) / len(corner_ids) if corner_ids else float("nan")
    nn_easy = sum(1 for h in easy_ids if nn_preds[h] == truth[h]) / len(easy_ids) if easy_ids else float("nan")
    nn_cornerA = sum(1 for h in corner_A if nn_preds[h] == truth[h]) / len(corner_A) if corner_A else float("nan")
    nn_cornerB = sum(1 for h in corner_B if nn_preds[h] == truth[h]) / len(corner_B) if corner_B else float("nan")

    results["baseline_nn"] = {
        "accuracy": nn_acc, "n_correct": nn_correct,
        "corner_acc": nn_corner, "easy_acc": nn_easy,
        "cornerA_acc": nn_cornerA, "cornerB_acc": nn_cornerB,
        "detail": nn_detail,
    }

    json.dump(results, open(OUT_JSON, "w"), indent=2)

    # ---------- write markdown ----------
    write_md(results)
    print("WROTE", OUT_JSON, "and", OUT_MD)
    # echo headline numbers to stdout
    print("\n=== HEADLINE ===")
    print("integrity rule-vs-truth mismatches:", results["integrity"]["n_rule_vs_truth_mismatches"])
    print("integrity train-vs-truth mismatches:", results["integrity"]["n_train_vs_truth_mismatches"])
    for kc in K_curve:
        print(f"K={kc['K']:>2}  acc={kc['mean_acc']:.3f}  corner={kc['mean_corner_acc']:.3f}  easy={kc['mean_easy_acc']:.3f}")
    print("majority acc:", round(maj_acc, 3), "label:", majority_label)
    print("NN acc:", round(nn_acc, 3), "corner:", round(nn_corner, 3), "easy:", round(nn_easy, 3))
    print("rule recovery by K:", results["rule_recovery_by_K"])


def fmt_pct(x):
    if x != x:  # nan
        return "n/a"
    return f"{x*100:.1f}%"


def write_md(r):
    L = []
    L.append("# Rule-induction generalization & over-simplification probe")
    L.append("")
    L.append("Scoring of `feedback_retrieval/simplicity/` (rule-induction from few labeled "
             "cases, generalization to 20 held-out cases). Inputs read-only; this file is a new result.")
    L.append("")
    L.append("Stated latent rule: **established = preexisting OR (NOT solely_mine); else my_style.** "
             "`language`, `work_vs_personal`, `team_size` are distractors.")
    L.append("")

    # Integrity
    integ = r["integrity"]
    L.append("## Step 0 — Integrity check")
    L.append("")
    L.append(f"- Recomputed decision from latents via the stated rule for all "
             f"{integ['n_cases']} cases, compared to `true_decision`: "
             f"**{integ['n_rule_vs_truth_mismatches']} mismatches**.")
    L.append(f"- `train_labels` vs ground-truth `true_decision`: "
             f"**{integ['n_train_vs_truth_mismatches']} mismatches**.")
    L.append(f"- `train_labels` vs stated-rule recomputation: "
             f"**{integ['n_train_vs_rule_mismatches']} mismatches**.")
    if integ["n_rule_vs_truth_mismatches"] == 0 and integ["n_train_vs_truth_mismatches"] == 0:
        L.append("")
        L.append("**Data is internally consistent.** Ground-truth `true_decision` is used as truth "
                 "throughout; no inconsistency caveat needed.")
    else:
        L.append("")
        L.append("**INCONSISTENCY FLAGGED.** Ground-truth `true_decision` is used as truth, but the "
                 "stated rule does not reproduce it everywhere; downstream accuracy is relative to "
                 "`true_decision`. Mismatches:")
        for m in integ["rule_vs_truth_mismatches"]:
            L.append(f"  - {m['id']}: latent={m['latent']} rule={m['recomputed']} truth={m['true_decision']}")
        for m in integ["train_vs_truth_mismatches"]:
            L.append(f"  - train {m}")
    L.append("")

    # corner def
    cd = r["corner_definition"]
    L.append("## Held-out structure")
    L.append("")
    L.append(f"- n=20 held-out cases. **Corner cases** (the two disjuncts disagree; true=`established` "
             f"but a single-axis rule predicts `my_style`): {cd['n_corner']} cases.")
    L.append(f"  - Corner A `preexisting AND solely_mine` (an ownership-only rule misses these): "
             f"{cd['corner_A_preexisting_and_solely_mine']}")
    L.append(f"  - Corner B `(NOT preexisting) AND (NOT solely_mine)` (a greenfield-only rule misses these): "
             f"{cd['corner_B_greenfield_and_shared']}")
    L.append(f"- **Easy cases** (both disjuncts agree, or a single axis already separates them): "
             f"{cd['n_easy']} cases: {cd['easy_ids']}")
    L.append("")

    # Step 1 curve + baselines
    L.append("## Step 1 — Held-out accuracy vs K (convergence curve)")
    L.append("")
    maj = r["baseline_majority"]
    nn = r["baseline_nn"]
    L.append(f"Baselines: **majority-class** = `{maj['majority_label']}` "
             f"(train counts {maj['train_label_counts']}) → acc **{fmt_pct(maj['accuracy'])}**; "
             f"**NN-embedding** (embeddinggemma-300m, mps, normalized; nearest train description) → "
             f"acc **{fmt_pct(nn['accuracy'])}**.")
    L.append("")
    L.append("| K | #subsets | induced acc | corner acc | easy acc | majority | NN-embed |")
    L.append("|---|---|---|---|---|---|---|")
    for kc in r["K_curve"]:
        L.append(f"| {kc['K']} | {kc['n_subsets']} | {fmt_pct(kc['mean_acc'])} | "
                 f"{fmt_pct(kc['mean_corner_acc'])} | {fmt_pct(kc['mean_easy_acc'])} | "
                 f"{fmt_pct(maj['accuracy'])} | {fmt_pct(nn['accuracy'])} |")
    L.append("")

    # Step 2 corner
    L.append("## Step 2 — Over-simplification probe (corner vs easy)")
    L.append("")
    L.append("If the induced rule drops a disjunct, it scores ~0 on the corner whose disjunct it dropped. "
             "Corner A is missed by ownership-only rules; Corner B is missed by greenfield-only rules.")
    L.append("")
    L.append("| K | corner A acc (pre∧solo) | corner B acc (green∧shared) | all-corner acc | easy acc |")
    L.append("|---|---|---|---|---|")
    for kc in r["K_curve"]:
        L.append(f"| {kc['K']} | {fmt_pct(kc['mean_cornerA_acc'])} | {fmt_pct(kc['mean_cornerB_acc'])} | "
                 f"{fmt_pct(kc['mean_corner_acc'])} | {fmt_pct(kc['mean_easy_acc'])} |")
    L.append("")
    L.append(f"NN-embedding on corners: corner A {fmt_pct(nn['cornerA_acc'])}, "
             f"corner B {fmt_pct(nn['cornerB_acc'])}, all-corner {fmt_pct(nn['corner_acc'])}, "
             f"easy {fmt_pct(nn['easy_acc'])}. "
             f"Majority on corners {fmt_pct(maj['corner_acc'])}, easy {fmt_pct(maj['easy_acc'])}.")
    L.append("")

    # Step 4 rule recovery
    L.append("## Step 4 — Rule recovery (did induction capture both disjuncts?)")
    L.append("")
    L.append("Classification of each `induced_rule` text: `both` (preexisting OR shared), "
             "`ownership_only`, `greenfield_only`, or `other`.")
    L.append("")
    L.append("| K | both | ownership_only | greenfield_only | other |")
    L.append("|---|---|---|---|---|")
    for K in sorted(r["rule_recovery_by_K"], key=lambda x: int(x)):
        d = r["rule_recovery_by_K"][K]
        L.append(f"| {K} | {d.get('both',0)} | {d.get('ownership_only',0)} | "
                 f"{d.get('greenfield_only',0)} | {d.get('other',0)} |")
    L.append("")
    L.append("Per-run induced rules:")
    L.append("")
    for run in r["per_run"]:
        L.append(f"- K={run['K']} subset {run['subset_index']} [{run['rule_class']}], "
                 f"acc {fmt_pct(run['accuracy'])}: \"{run['induced_rule']}\"")
    L.append("")

    # Caveats
    L.append("## Caveats")
    L.append("")
    L.append("- **Synthetic data, single domain** (coding-style choice), one rule family. No claim "
             "generalizes beyond this construction.")
    L.append(f"- **n=20 held-out**; corner cells are tiny (corner A = "
             f"{len(r['corner_definition']['corner_A_preexisting_and_solely_mine'])} cases, corner B = "
             f"{len(r['corner_definition']['corner_B_greenfield_and_shared'])} cases), so corner "
             "accuracies are coarse (±1 case in a 6-case cell ≈ ±17%).")
    L.append("- The induced rules are pre-supplied in `predictions.json`; this is scoring, not a live "
             "induction. Which single axis the LLM latches onto may reflect a prior over what is more "
             "salient in human experience (ownership vs greenfield), not a property of the data alone.")
    L.append("- NN-embedding uses the natural-language `description`, which leaks the latents lexically; "
             "it is the charitable 'enumerate + nearest match' baseline, not a hard lower bound.")
    L.append("- Subsets per K are few (3 at K=2/4/8, 1 at K=30); the 'curve' is a small-sample estimate.")
    L.append("")

    open(OUT_MD, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
