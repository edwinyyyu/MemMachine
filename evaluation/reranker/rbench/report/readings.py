"""Readings: how a question counts as covered at a reader budget, one definition for every
table, chart and cost calculation.

Event readings (metrics.py) give each question one cost, the reader tokens at which the event
happens: first, every (some / all evidence read in full), first_touch, every_touch (touched).
A question is covered at budget B when its cost is at most B, so coverage is the share of
questions.

Share readings give each question one cost per evidence turn (span) of its answer: the reader
tokens at which that turn is read in full (share) or touched (share_touch). A question's value
at B is the share of its evidence turns within B, each counted once, and coverage is the mean
of that share over questions (the user's decision, 2026-11-09: "add the new measure, sit
alongside"). For a question with one evidence turn the share equals the event reading; touched
is generous on long assistant turns, in full strict. Where a dataset accepts several answers,
the turns are those of the answer read in full first (the one every uses).

A reading's costs are a float array (one per question) for events and an object array of
float arrays (one per evidence turn) for shares; every function here takes either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENTS = ("first", "every", "first_touch", "every_touch")
SHARES = {"share": "share_full", "share_touch": "share_touch"}  # reading -> its per-turn cost column
ALL = (*EVENTS, *SHARES)


def available(df: pd.DataFrame, readings=ALL) -> list[str]:
    """The readings a simulation has columns for (results simulated before the share readings
    have only the events)."""
    return [x for x in readings if (SHARES[x] if x in SHARES else x) in df]


def is_share(reading: str) -> bool:
    return reading in SHARES


def costs(rows: pd.DataFrame, reading: str) -> np.ndarray:
    """A reading's costs for these rows (in row order)."""
    if reading in SHARES:
        out = np.empty(len(rows), dtype=object)
        out[:] = [np.asarray(c, dtype=np.float64) for c in rows[SHARES[reading]]]
        return out
    return rows[reading].to_numpy(dtype=np.float64)


def hit(c: np.ndarray, budget: float) -> np.ndarray:
    """Each question's value at a budget: 1/0 for events, its share of turns for shares."""
    if c.dtype == object:
        return np.asarray([float(np.mean(x <= budget)) if len(x) else 0.0 for x in c])
    return (c <= budget).astype(np.float64)


def coverage(c: np.ndarray, budgets) -> np.ndarray:
    """Coverage at each budget: the mean over questions of their values."""
    budgets = np.asarray(budgets, dtype=np.float64)
    if c.dtype == object:
        flat, weight = _flat(c)
        order = np.argsort(flat, kind="stable")
        cum = np.concatenate([[0.0], np.cumsum(weight[order])])
        return cum[np.searchsorted(flat[order], budgets, "right")] / max(1, len(c))
    s = np.sort(c)
    return np.searchsorted(s, budgets, "right") / max(1, len(s))


def steps(c: np.ndarray) -> np.ndarray:
    """The budgets at which coverage changes (finite costs, sorted)."""
    flat = _flat(c)[0] if c.dtype == object else c
    return np.sort(flat[np.isfinite(flat)])


def min_budgets(c: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """For each target, the smallest budget whose coverage reaches it (inf if none): an order
    statistic of the costs (for shares, of the turn costs weighted by 1 / turns, over the
    questions)."""
    targets = np.asarray(targets, dtype=np.float64)
    if c.dtype == object:
        flat, weight = _flat(c)
        order = np.argsort(flat, kind="stable")
        flat, cum = flat[order], np.cumsum(weight[order]) / max(1, len(c))
        k = np.searchsorted(cum, targets - 1e-9, "left")
        return np.where(k < len(flat), flat[np.minimum(k, len(flat) - 1)], np.inf) if len(flat) else np.full(len(targets), np.inf)
    s = np.sort(c)
    k = np.ceil(targets * len(s) - 1e-9).astype(int) - 1
    return s[np.clip(k, 0, len(s) - 1)] if len(s) else np.full(len(targets), np.inf)


def reachable(c: np.ndarray) -> np.ndarray:
    """Each question's value with no budget limit (all of its pool read)."""
    return hit(c, np.inf)


def _flat(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn costs of every question, and each one's weight (1 / its question's turns)."""
    parts = [x for x in c if len(x)]
    if not parts:
        return np.zeros(0), np.zeros(0)
    flat = np.concatenate(parts)
    weight = np.concatenate([np.full(len(x), 1.0 / len(x)) for x in parts])
    return flat, weight
