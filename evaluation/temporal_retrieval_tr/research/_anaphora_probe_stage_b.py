"""Anaphora probe — Stage B (doc-side): can we extract anaphoric anchors
from docs that have no explicit dates?

In `allen`, gold docs like "Two weeks before my wedding we tried cake
sampling" carry zero extracted anchors because the extractor only fires
on absolute dates. If we want them rankable on temporal grounds we need
to:
  1. Detect the anaphor pattern ("[offset] [direction] my [event]")
  2. Resolve "my [event]" to the corresponding anchor doc (Stage A)
  3. Compute the resulting interval (apply offset to anchor's interval)

This probe tests stage 2 + 3 on allen gold docs without anchors:
  - List allen docs whose extraction is empty
  - For each, see if a simple regex extracts (offset, direction, event)
  - Run Stage-A semantic search for "my [event]"
  - Report: is the right anchor doc in top-1 / top-3?

A failure here just means the simple-regex approach is too narrow —
the doc-side extraction would need to be LLM-driven, which is a real
prompt-engineering question. The probe still tells us whether the
anchor docs are findable from the anaphor phrase alone.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._anaphora_probe_stage_b
"""
from __future__ import annotations

import asyncio
import json
import re

import numpy as np

from temporal_retrieval.research._common import DATA_DIR, make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import make_cached_embed_fn

setup_env()


# A minimal regex covering the allen doc patterns. Captures
# (offset_phrase, direction, event_phrase).
ANAPHOR_RE = re.compile(
    r"^(?P<offset>[A-Za-z\- ]+?)\s+"
    r"(?P<direction>before|after|during|while|since)\s+"
    r"(?P<event>(?:my|the|our)[^,.]+?)"
    r"(?:[,.]| we | I | my | the |$)",
    re.IGNORECASE,
)

# Hand-annotated event → anchor doc for verification.
EVENT_TO_ANCHOR = {
    "wedding": "a_wedding",
    "graduation": "a_graduation",
    "marathon": "a_marathon",
    "europe trip": "a_europe_trip",
    "europe": "a_europe_trip",
    "honeymoon": "a_honeymoon",
    "conference": "a_conference",
    "move to denver": "a_move_denver",
    "move": "a_move_denver",
    "promotion": "a_promotion",
}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) or 1e-9
    nb = float(np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / (na * nb))


def find_expected_anchor(event_phrase: str) -> str | None:
    """Look up the gold anchor doc for a parsed event phrase."""
    key = event_phrase.lower().strip()
    key = re.sub(r"^(my|the|our)\s+", "", key)
    key = re.sub(r"[^a-z ]", "", key).strip()
    if key in EVENT_TO_ANCHOR:
        return EVENT_TO_ANCHOR[key]
    # try suffix / prefix match
    for k, v in EVENT_TO_ANCHOR.items():
        if k in key or key in k:
            return v
    return None


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    with open(DATA_DIR / "allen_docs.jsonl") as f:
        docs = [json.loads(line) for line in f]
    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    doc_embs = await embed_fn(doc_texts)
    by_id = {d["doc_id"]: d for d in docs}

    # Find docs that DON'T look like anchor docs (don't start with date-pattern).
    # In allen, anchor docs are `a_*`; everything else is `c_*`.
    candidates = [d for d in docs if d["doc_id"].startswith("c_")]
    print(f"\n=== STAGE B probe: allen ({len(candidates)} non-anchor docs) ===\n",
          flush=True)
    n_parse = 0
    n_event_found = 0
    n_top1 = 0
    n_top3 = 0

    for d in candidates:
        text = d["text"]
        m = ANAPHOR_RE.match(text)
        if not m:
            print(f"  ✗ {d['doc_id']:25s}  [no regex match] {text[:80]}", flush=True)
            continue
        n_parse += 1
        offset_phrase = m.group("offset").strip()
        direction = m.group("direction").lower()
        event_phrase = m.group("event").strip().rstrip(",.")
        expected = find_expected_anchor(event_phrase)
        if not expected:
            print(f"  ? {d['doc_id']:25s}  parsed='{offset_phrase}|{direction}|"
                  f"{event_phrase}' [no expected anchor]", flush=True)
            continue
        n_event_found += 1

        # Stage-A semantic search for "my [event]"
        probe_phrase = event_phrase
        probe_emb = (await embed_fn([probe_phrase]))[0]
        scored = [(doc_ids[i], cosine(probe_emb, doc_embs[i]))
                  for i in range(len(doc_ids))]
        # Restrict to anchor docs (a_*) — disambiguation via doc-id heuristic
        # because in this probe we have no real disambiguator yet
        anchor_scored = [(d_, s) for d_, s in scored if d_.startswith("a_")]
        anchor_scored.sort(key=lambda x: -x[1])
        top3 = [d_ for d_, _ in anchor_scored[:3]]
        if not top3:
            continue
        rank = (top3.index(expected) + 1) if expected in top3 else None
        if rank == 1:
            n_top1 += 1
        if rank is not None:
            n_top3 += 1
        mark = "✓" if rank == 1 else ("·" if rank else "✗")
        print(f"  {mark} {d['doc_id']:25s}  '{offset_phrase}|{direction}|{event_phrase}'"
              f"  expected={expected}  top1={top3[0]}", flush=True)

    n_c = len(candidates)
    print(f"\n  parse hit:        {n_parse}/{n_c}  = {n_parse/n_c:.2%}", flush=True)
    print(f"  event resolved:   {n_event_found}/{n_c}  = {n_event_found/n_c:.2%}",
          flush=True)
    print(f"  anchor top-1:     {n_top1}/{n_event_found}  = "
          f"{n_top1/max(1,n_event_found):.2%}", flush=True)
    print(f"  anchor top-3:     {n_top3}/{n_event_found}  = "
          f"{n_top3/max(1,n_event_found):.2%}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
