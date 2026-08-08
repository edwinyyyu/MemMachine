"""Preference-learning HARNESS ENGINE (state + math only).

Division of labor (token/harness split):
  - The MODEL (an LLM, in written tokens) proposes candidate rules, invents probe
    situations, and applies a learned rule. It does NOT do the arithmetic below.
  - This ENGINE holds the hypothesis set + posterior and does all the math:
    prior, soft size-principle likelihood, posterior, hypothesis-averaged
    prediction, expected info-gain probe selection, the should_probe gate,
    Bayesian update, and posterior entropy.

A "hypothesis" h is a dict:
    {
      "id":    str,                      # human label
      "conds": list[(feature, value)],   # conjunctive conditions; matches => decision=True
    }
An hypothesis "matches" situation x (a feature dict) iff ALL its conditions hold.
The hypothesis predicts decision = match(x). (Empty conds => always-True rule.)

Stdlib only; no third-party deps (keeps it lightweight).
"""

from __future__ import annotations

import json
import math

EPS = 0.1  # soft-likelihood floor; never zero


# ----------------------------------------------------------------------------- #
# State container
# ----------------------------------------------------------------------------- #
class Engine:
    def __init__(self, hyps, consequence):
        """hyps: list of hypothesis dicts. consequence: float >=0 (cost of a wrong
        autonomous decision in this domain; scales the should_probe gate)."""
        self.hyps = [self._norm(h) for h in hyps]
        self.consequence = float(consequence)
        self._log_prior = [self._log_prior_of(h) for h in self.hyps]

    @staticmethod
    def _norm(h):
        return {"id": h["id"], "conds": [tuple(c) for c in h["conds"]]}

    # --- core per-hypothesis primitives -------------------------------------- #
    @staticmethod
    def matches(h, x):
        return all(x.get(f) == v for (f, v) in h["conds"])

    @staticmethod
    def predicts(h, x):
        """Binary decision this hypothesis assigns to situation x."""
        return Engine.matches(h, x)

    def _log_prior_of(self, h):
        # prior P(h) proportional to 2^(-#conditions): simpler rules favored.
        return -len(h["conds"]) * math.log(2.0)

    @staticmethod
    def _log_lik(h, x, answer):
        """Soft size-principle likelihood P(answer | h) for one revealed example.
        (1 - EPS) if the hypothesis's decision matches the observed answer, else EPS."""
        return math.log(1.0 - EPS) if (Engine.predicts(h, x) == answer) else math.log(EPS)

    # --- posterior ----------------------------------------------------------- #
    def posterior(self, revealed):
        """revealed: list of (x, answer) with answer in {True, False}. SKIP entries
        must already be filtered out by caller (update() never stores them).
        Returns list of posterior weights aligned with self.hyps (sums to 1)."""
        log_w = list(self._log_prior)
        for (x, ans) in revealed:
            for i, h in enumerate(self.hyps):
                log_w[i] += self._log_lik(h, x, ans)
        m = max(log_w)
        w = [math.exp(lw - m) for lw in log_w]
        z = sum(w)
        return [wi / z for wi in w]

    # --- prediction (hypothesis-averaging) ----------------------------------- #
    def predict(self, x, revealed):
        """Posterior-averaged P(decision=True | x) and the MAP binary decision.
        Returns (p_true, decision_bool)."""
        post = self.posterior(revealed)
        p_true = sum(p for p, h in zip(post, self.hyps) if self.predicts(h, x))
        return p_true, (p_true >= 0.5)

    # --- expected information gain ------------------------------------------- #
    def info_gain(self, x, revealed):
        """Expected reduction in posterior entropy from probing situation x.
        Equivalently the posterior-weighted disagreement: it peaks when the live
        hypotheses split ~50/50 on x. Computed as current entropy minus the
        answer-probability-weighted expected posterior entropy."""
        post = self.posterior(revealed)
        h_now = self._entropy(post)
        p_true = sum(p for p, h in zip(post, self.hyps) if self.predicts(h, x))
        p_false = 1.0 - p_true
        exp_post_ent = 0.0
        for ans, p_ans in ((True, p_true), (False, p_false)):
            if p_ans <= 0.0:
                continue
            post_after = self._posterior_after(post, x, ans)
            exp_post_ent += p_ans * self._entropy(post_after)
        return h_now - exp_post_ent

    def _posterior_after(self, post, x, ans):
        log_w = []
        for p, h in zip(post, self.hyps):
            lw = math.log(p) if p > 0 else -1e300
            lw += self._log_lik(h, x, ans)
            log_w.append(lw)
        m = max(log_w)
        w = [math.exp(lw - m) for lw in log_w]
        z = sum(w)
        return [wi / z for wi in w]

    # --- probe selection ----------------------------------------------------- #
    def select_probe(self, pool, revealed):
        """From pool (list of situations x), pick the one with max expected info-gain.
        Returns (best_x, best_gain). pool empty => (None, 0.0)."""
        best_x, best_g = None, -1.0
        for x in pool:
            g = self.info_gain(x, revealed)
            if g > best_g:
                best_x, best_g = x, g
        return best_x, (best_g if best_x is not None else 0.0)

    def should_probe(self, pool, revealed, bar):
        """Lightweight / anti-intrusion gate. Probe only if
        info_gain(best) * consequence > bar. Returns (flag, best_x, gain, value)."""
        best_x, g = self.select_probe(pool, revealed)
        value = g * self.consequence
        return (value > bar), best_x, g, value

    # --- update -------------------------------------------------------------- #
    def update(self, revealed, x, answer_or_SKIP):
        """Returns a NEW revealed list. If answer is 'SKIP' (or None), this is a
        graceful no-op: nothing is appended, posterior is unchanged. Otherwise the
        (x, bool) pair is appended."""
        if answer_or_SKIP == "SKIP" or answer_or_SKIP is None:
            return list(revealed)  # no-op
        return list(revealed) + [(x, bool(answer_or_SKIP))]

    # --- entropy ------------------------------------------------------------- #
    @staticmethod
    def _entropy(post):
        return -sum(p * math.log(p) for p in post if p > 0.0)

    def entropy(self, revealed):
        """Posterior entropy (nats). Low => converged."""
        return self._entropy(self.posterior(revealed))

    # --- convenience for persistence ---------------------------------------- #
    def map_hypothesis(self, revealed):
        post = self.posterior(revealed)
        i = max(range(len(self.hyps)), key=lambda j: post[j])
        return self.hyps[i], post[i]


# ----------------------------------------------------------------------------- #
# Module-level functional API (spec-required names). Uses a single ambient engine
# so the model can call init() once then the rest without threading the object.
# ----------------------------------------------------------------------------- #
_E: Engine | None = None


def init(hyps, consequence):
    global _E
    _E = Engine(hyps, consequence)
    return _E


def _eng():
    if _E is None:
        raise RuntimeError("call init(hyps, consequence) first")
    return _E


def posterior(revealed):
    return _eng().posterior(revealed)


def predict(x, revealed):
    return _eng().predict(x, revealed)


def info_gain(x, revealed):
    return _eng().info_gain(x, revealed)


def select_probe(pool, revealed):
    return _eng().select_probe(pool, revealed)


def should_probe(pool, revealed, bar):
    return _eng().should_probe(pool, revealed, bar)


def update(revealed, x, answer_or_SKIP):
    return _eng().update(revealed, x, answer_or_SKIP)


def entropy(revealed):
    return _eng().entropy(revealed)


def map_hypothesis(revealed):
    return _eng().map_hypothesis(revealed)


def save_learned(path, revealed):
    """Persist the learned (averaged) rule + MAP rule + full posterior."""
    e = _eng()
    post = e.posterior(revealed)
    map_h, map_p = e.map_hypothesis(revealed)
    payload = {
        "map_rule": {"id": map_h["id"], "conds": map_h["conds"], "posterior": map_p},
        "posterior_distribution": [
            {"id": h["id"], "conds": h["conds"], "weight": p}
            for h, p in sorted(zip(e.hyps, post), key=lambda t: -t[1])
        ],
        "entropy_nats": e.entropy(revealed),
        "n_revealed": len(revealed),
        "consequence": e.consequence,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    # Smoke test: imports + runs.
    hs = [
        {"id": "always", "conds": []},
        {"id": "A", "conds": [("a", True)]},
        {"id": "A&B", "conds": [("a", True), ("b", True)]},
    ]
    init(hs, consequence=1.0)
    rev = []
    print("prior entropy:", round(entropy(rev), 4))
    x = {"a": True, "b": False}
    print("predict pre:", predict(x, rev))
    print("info_gain:", round(info_gain(x, rev), 4))
    rev = update(rev, x, "SKIP")
    print("after SKIP n=", len(rev), "(should be 0)")
    rev = update(rev, x, True)  # a True, b False -> A and always match, A&B doesn't
    print("after real update n=", len(rev), "entropy:", round(entropy(rev), 4))
    print("OK")
