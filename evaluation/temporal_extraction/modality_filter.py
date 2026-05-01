"""Retrieval-time modality filter (v2'' addition).

Given a mapping of doc_id -> [TimeExpression], partition docs into
"retrievable" (at least one expression with modality="actual") and
"skipped" (ALL expressions are fictional/hypothetical/quoted_embedded).

Docs with NO extracted expressions fall through — we only skip a doc when
we affirmatively know ALL its expressions are non-actual. This is safe:
non-temporal docs still compete on the semantic channel.

Usage:

    from modality_filter import partition_by_modality

    keep_ids, skip_ids = partition_by_modality(doc_ext)
    ranked = [d for d in ranked if d in keep_ids]
"""

from __future__ import annotations

from typing import Any

from modality_schema import get_modality


def is_doc_actual(tes: list[Any]) -> bool:
    """True if the doc has zero TEs OR at least one TE with modality=actual."""
    if not tes:
        return True
    for te in tes:
        if get_modality(te) == "actual":
            return True
    return False


def partition_by_modality(
    doc_ext: dict[str, list[Any]],
) -> tuple[set[str], set[str]]:
    """Return (keep_ids, skip_ids).

    keep_ids: doc has at least one actual TE, or no TEs at all.
    skip_ids: doc has TEs, all non-actual.
    """
    keep: set[str] = set()
    skip: set[str] = set()
    for did, tes in doc_ext.items():
        if is_doc_actual(tes):
            keep.add(did)
        else:
            skip.add(did)
    return keep, skip


def filter_ranking(
    ranked_ids: list[str],
    doc_ext: dict[str, list[Any]],
    *,
    filter_modality: bool = True,
) -> list[str]:
    """Return ranked_ids with non-actual docs removed, preserving order."""
    if not filter_modality:
        return ranked_ids
    _, skip = partition_by_modality(doc_ext)
    return [d for d in ranked_ids if d not in skip]
