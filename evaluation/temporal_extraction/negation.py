"""Negation handling for temporal queries.

Detects exclusion cue words (`not`, `excluding`, `except`, `outside of`,
`without`) and treats the surrounding query as a NEGATIVE temporal
constraint: gold docs lie OUTSIDE the named window, distractors copy
the named-window date words.

Pipeline:
  1. ``has_negation_cue(query)`` — gate (False = don't fire negation logic).
  2. ``parse_negation_query(query)`` — returns ``(positive_query, excluded_phrase)``:
       - positive_query has the cue (and optionally the excluded phrase) stripped
       - excluded_phrase is the substring after the cue word
  3. The caller runs the standard temporal extractor on ``excluded_phrase``
     to obtain its TimeExpression list — those are the EXCLUDED intervals.
  4. Apply one of:
       - **mask**: ``final = positive_T * (1 - in_excluded_window)`` where
         in_excluded_window is the asymmetric containment ``|d ∩ excl| / |d|``.
       - **signed**: ``final = positive_T - λ * negative_T`` where
         negative_T is the same containment score (a continuous penalty).

This file ONLY does string parsing. Score combination lives in the eval.
"""

from __future__ import annotations

import re

# Cue words. Order matters: longer phrases first so re.search picks the
# most specific match. We require `\b` boundaries except for the
# multi-word phrases.
_NEG_CUES = [
    r"outside\s+of",
    r"outside",
    r"excluding",
    r"except\s+for",
    r"except",
    r"without",
    r"not\s+in",
    r"not\s+during",
    r"not\s+on",
    # Bare "not" is intentionally LAST. We require a temporal word after
    # it (year, month, season, "Q1"-"Q4") so phrasings like "did I not
    # already do X" don't fire.
    r"not(?=\s+\S+\s+(?:\d{4}|q[1-4]\b|quarter|january|february|march|april|may|june|"
    r"july|august|september|october|november|december|spring|summer|fall|autumn|winter|holiday))",
    r"not(?=\s+(?:\d{4}|q[1-4]\b|quarter|january|february|march|april|may|june|"
    r"july|august|september|october|november|december|spring|summer|fall|autumn|winter|holiday))",
]

_NEG_RE = re.compile(r"\b(" + "|".join(_NEG_CUES) + r")\b", re.IGNORECASE)


def has_negation_cue(query_text: str) -> bool:
    """True if the query contains a temporal-exclusion cue."""
    if not query_text:
        return False
    return _NEG_RE.search(query_text) is not None


def parse_negation_query(query_text: str) -> tuple[str, str | None]:
    """Parse (positive_query, excluded_phrase).

    - ``positive_query``: the query with the cue word AND excluded phrase
      removed, leaving the topic terms for semantic matching. E.g.
      "What workouts did I do not in January 2025?" →
      "What workouts did I do?".
    - ``excluded_phrase``: the temporal phrase being negated. Strategy:
      take everything from the cue word's end to the next sentence
      boundary or punctuation. The extractor will then parse this for
      its time expressions. E.g. "in January 2025".

    If no cue is found, returns ``(query_text, None)``.
    """
    if not query_text:
        return query_text, None
    m = _NEG_RE.search(query_text)
    if not m:
        return query_text, None
    cue_start, cue_end = m.span()
    # Excluded phrase: from cue_end to next sentence-final punctuation or end.
    rest = query_text[cue_end:]
    # Stop at sentence-terminating punctuation (?, ., !) but NOT at commas
    # (we want phrases like "summer (June–August 2024)" to stay intact)
    stop_match = re.search(r"[?!.]", rest)
    if stop_match:
        excluded_phrase = rest[: stop_match.start()].strip()
    else:
        excluded_phrase = rest.strip()
    excluded_phrase = excluded_phrase.strip(" ,;:")

    # Positive query: replace [cue + excluded_phrase] with empty space.
    if stop_match:
        end_of_excluded = cue_end + stop_match.start()
    else:
        end_of_excluded = len(query_text)
    positive_query = (
        query_text[:cue_start].rstrip() + " " + query_text[end_of_excluded:].lstrip()
    ).strip()
    # Tidy double spaces.
    positive_query = re.sub(r"\s+", " ", positive_query).strip()
    if not excluded_phrase:
        excluded_phrase = None
    return positive_query, excluded_phrase


# ----------------------------------------------------------------------
# Score combination helpers
# ----------------------------------------------------------------------
def excluded_containment(d_ivs, excl_ivs) -> float:
    """How much of the doc's anchor falls inside the excluded window.

    Same primitive as T_v4: ``max over (d_iv, e_iv) of |d ∩ e| / |d|``.
    Returns 0.0 if doc has no anchor or excluded list empty.
    """
    if not d_ivs or not excl_ivs:
        return 0.0
    best = 0.0
    for di in d_ivs:
        d_dur = di.latest_us - di.earliest_us
        if d_dur <= 0:
            d_dur = 1
        for ei in excl_ivs:
            inter_lo = max(di.earliest_us, ei.earliest_us)
            inter_hi = min(di.latest_us, ei.latest_us)
            inter = max(0, inter_hi - inter_lo)
            score = inter / d_dur
            if score > best:
                best = score
                if best >= 1.0:
                    return 1.0
    return best


def apply_mask(positive_scores: dict, excl_containment: dict) -> dict:
    """final = positive * (1 - excluded_containment).

    A doc fully inside the excluded window gets multiplied by 0; a doc
    fully outside gets *1. Partial overlap interpolates.
    """
    out = {}
    for did, t in positive_scores.items():
        c = excl_containment.get(did, 0.0)
        out[did] = t * max(0.0, 1.0 - c)
    return out


def apply_signed(
    positive_scores: dict, excl_containment: dict, lam: float = 1.0
) -> dict:
    """final = positive - lam * excluded_containment.

    Continuous penalty; survives ties. Negative scores are allowed (a
    distractor inside the window with zero positive signal will still
    rank below a doc with even a tiny positive signal but no penalty).
    """
    out = {}
    for did, t in positive_scores.items():
        c = excl_containment.get(did, 0.0)
        out[did] = t - lam * c
    return out
