"""Recency-decay scoring channel.

Activates when a query has a "recency cue word" but no explicit date
anchor. Scores docs by exponential decay over (ref_time - doc_time):

    recency_score(d_te, ref_time) = exp(-lambda * (ref_time - d_te.best_us) / DAY)

For docs without a temporal extraction (TE) we fall back to the doc's
``ref_time`` field. For multi-TE docs we take MAX recency across TEs.
Docs whose anchor is in the *future* relative to ``ref_time`` are
penalised symmetrically (we use ``abs(diff)``); the cue words covered
here all imply "as recent as possible up to now", so future docs are
no more useful than equally-distant past docs.

The cue word gate ensures recency is *not* applied to non-recency
queries — without a cue word the channel returns ``None`` so the
caller can fall through to T_v4 / T_lblend / etc.
"""

from __future__ import annotations

import math
import re

# ---- recency cue detection -------------------------------------------------
# Word-boundary regex; case-insensitive. Matches the surface forms named
# in the task spec. We intentionally keep this list short — adding too
# many cue words risks false-positives on non-recency queries.
_CUE_WORDS = [
    r"latest",
    r"most\s+recent",
    r"recently",
    r"newly",
    r"just",  # e.g. "what did I just X?"
    r"current(?:ly)?",
    r"present",  # noun cue ("the present"); verb sense suppressed below
    r"now",
    r"these\s+days",
    r"as\s+of\s+(?:now|today)",
    # "last" and "the last" are ambiguous (last week vs last appointment),
    # but in this benchmark "last X" is the dominant recency form. Gate
    # below excludes "last week", "last month", etc. since those carry
    # their own dates and T_v4 can score them.
    r"last",
]
_CUE_RE = re.compile(r"\b(" + "|".join(_CUE_WORDS) + r")\b", re.IGNORECASE)

# Phrases that *suppress* the cue (because they bring their own explicit
# date anchor and T_v4/T_lblend can score them properly). If the query
# matches these we treat it as non-recency.
#
# Verb-form `present` (presented / presenting / presents) is NOT a cue.
# Hard_bench had several "present at a conference" / "presented in 2023"
# queries that triggered false-positives.
_SUPPRESS_RE = re.compile(
    r"\b(last\s+(?:week|month|year|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"summer|winter|spring|fall|autumn)|"
    r"current(?:ly)?\s+\d{4}|"
    r"now\s+that|"
    # "present at"/"present a"/"presented"/"presenting"/"presents" — verb sense.
    r"present(?:s|ed|ing|ation|er|ation)?\s+(?:at|a\s|an\s|the\s|to\s|on\s|in\s|with\s|some|him|her|them|us|me|you))\b",
    re.IGNORECASE,
)

# Additional suppressor: "present" used as a verb form anywhere
# (presented/presenting/presents/presentation). Keep the noun "the
# present" / "present moment" / "present day" as a recency cue.
_VERB_PRESENT_RE = re.compile(r"\bpresent(?:ed|ing|s|ation|er)\b", re.IGNORECASE)


def has_recency_cue(query_text: str) -> bool:
    """Return True if the query has a recency cue word and no
    suppressing context (explicit date phrase like "last week")."""
    if not query_text:
        return False
    if _SUPPRESS_RE.search(query_text):
        return False
    # Strip verb-form "present" before re-checking the cue regex; if the
    # only cue was that verb, the query becomes non-recency.
    stripped = _VERB_PRESENT_RE.sub("", query_text)
    return _CUE_RE.search(stripped) is not None


# ---- recency score ---------------------------------------------------------
DAY_US = 86_400 * 1_000_000


def lambda_for_half_life(half_life_days: float) -> float:
    """λ such that exp(-λ * h) = 0.5  →  λ = ln 2 / h."""
    return math.log(2.0) / half_life_days


def _doc_anchor_us(
    doc_bundles: list[dict] | None, fallback_ref_us: int | None
) -> int | None:
    """Return the best anchor (in microseconds) for a doc.

    Prefers ``best_us`` of any TE interval; falls back to interval mid-
    point; finally falls back to the doc's ref_time.
    """
    if doc_bundles:
        best_candidates: list[int] = []
        for b in doc_bundles:
            for iv in b.get("intervals", []) or []:
                if iv.best_us is not None:
                    best_candidates.append(iv.best_us)
                else:
                    # Use midpoint as anchor proxy.
                    mid = (iv.earliest_us + iv.latest_us) // 2
                    best_candidates.append(mid)
        if best_candidates:
            # MAX recency = anchor closest to ref_time. Caller selects the
            # max across candidates after computing decay; here we just
            # return all candidates by taking the latest anchor as a
            # conservative single-anchor fallback. To preserve "MAX over
            # all TEs" semantics we expose all candidates via
            # `_doc_anchors_us` below.
            return max(best_candidates)
    return fallback_ref_us


def _doc_anchors_us(
    doc_bundles: list[dict] | None, fallback_ref_us: int | None
) -> list[int]:
    """All candidate anchor microseconds for a doc.

    Used so recency_score can take MAX across TEs.
    """
    out: list[int] = []
    if doc_bundles:
        for b in doc_bundles:
            for iv in b.get("intervals", []) or []:
                if iv.best_us is not None:
                    out.append(iv.best_us)
                else:
                    out.append((iv.earliest_us + iv.latest_us) // 2)
    if not out and fallback_ref_us is not None:
        out.append(fallback_ref_us)
    return out


def recency_score(
    doc_bundles: list[dict] | None,
    doc_ref_us: int | None,
    query_ref_us: int,
    lam: float,
) -> float:
    """exp(-λ * |query_ref - doc_anchor| / DAY).

    Takes MAX recency across all anchors associated with the doc.
    Returns 0.0 if no anchor available.
    """
    anchors = _doc_anchors_us(doc_bundles, doc_ref_us)
    if not anchors:
        return 0.0
    best = 0.0
    for a in anchors:
        diff_us = abs(query_ref_us - a)
        days = diff_us / DAY_US
        s = math.exp(-lam * days)
        if s > best:
            best = s
    return best


def recency_scores_for_docs(
    doc_bundles_map: dict[str, list[dict]],
    doc_ref_us_map: dict[str, int],
    query_ref_us: int,
    lam: float,
) -> dict[str, float]:
    """Per-doc recency scores."""
    out: dict[str, float] = {}
    for did, bundles in doc_bundles_map.items():
        out[did] = recency_score(bundles, doc_ref_us_map.get(did), query_ref_us, lam)
    return out


# ---- combination strategies -----------------------------------------------
def combine_replacement(
    t_scores: dict[str, float],
    rec_scores: dict[str, float],
    cue: bool,
    t_dead_threshold: float = 1e-6,
) -> dict[str, float]:
    """If cue detected AND ALL T scores are dead (≤ threshold), use
    recency-only. Otherwise, return T_scores unchanged.
    """
    if not cue:
        return dict(t_scores)
    max_t = max(t_scores.values()) if t_scores else 0.0
    if max_t <= t_dead_threshold:
        return dict(rec_scores)
    return dict(t_scores)


def combine_additive(
    t_scores: dict[str, float],
    rec_scores: dict[str, float],
    cue: bool,
    alpha: float = 0.5,
) -> dict[str, float]:
    """final = (1-α)*T + α*cue*recency.

    α=0  → recency ignored (T-only).
    α=1  → recency-only when cue, T-only otherwise.
    """
    out = {}
    docs = set(t_scores) | set(rec_scores)
    cue_w = 1.0 if cue else 0.0
    for d in docs:
        t = t_scores.get(d, 0.0)
        r = rec_scores.get(d, 0.0)
        out[d] = (1.0 - alpha) * t + alpha * cue_w * r
    return out


def combine_multiplicative(
    t_scores: dict[str, float],
    rec_scores: dict[str, float],
    cue: bool,
) -> dict[str, float]:
    """final = T * (cue ? recency : 1).

    Only lifts docs that already have temporal relevance — the gate
    fails outright when T is uniformly 0 (the latest_recent regime).
    """
    if not cue:
        return dict(t_scores)
    out = {}
    docs = set(t_scores) | set(rec_scores)
    for d in docs:
        out[d] = t_scores.get(d, 0.0) * rec_scores.get(d, 0.0)
    return out
