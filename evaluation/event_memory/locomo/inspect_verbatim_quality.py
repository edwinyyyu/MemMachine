"""Inspect quality of the two verbatim-deictic-resolution variants.

Side-by-side: source raw message vs. xpurp2-verbatim display vs.
deictic-resolved-fresh display, for a curated set of representative
LoCoMo turns covering:
  - "you/your" addressing the other speaker
  - "this/that/here/there" demonstratives
  - third-person pronouns referring to a non-speaker
  - speaker self-reference via third person
  - temporal references that must stay verbatim
  - simple turns with no deictic at all
  - emoji / attached media / quoted wording

Plus a random sample for breadth.

Usage:
  uv run python inspect_verbatim_quality.py
"""

from __future__ import annotations
from artifacts import A  # canonical artifact names

import json
import random
import sqlite3
import sys
from collections import defaultdict

from memmachine_server.episodic_memory.event_memory.data_types import (
    decode_block,
    decode_context,
)


VERBATIM_DB = (
    A("locomo-tslimv3bonaturalsidecarverbatim-54n-l-nb8-c2sub-rep1.sqlite")
)
DEICTIC_DB = (
    A("locomo-deicticresolvedverbatim-54n-l-nb8-c2sub-rep1.sqlite")
)
SRC_BENCH = "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/data/locomo10_c2sub.json"


def _load_segments(db_path: str) -> dict[tuple[str, str], list[dict]]:
    """Return (partition_key, dia_id) -> list of segment dicts with display/embed."""
    con = sqlite3.connect(db_path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(segment_store_sg)")]
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in con.execute("SELECT * FROM segment_store_sg"):
        d = dict(zip(cols, row))
        blk = decode_block(json.loads(d["block"].decode()))
        ctx = decode_context(json.loads(d["context"].decode()))
        props = json.loads(d["properties"])
        dia = props["dia_id"]
        dia = dia["v"] if isinstance(dia, dict) else dia
        out[(d["partition_key"], dia)].append(
            {
                "display": blk.text,
                "embed": getattr(ctx, "text_to_embed", ""),
                "bm25": getattr(ctx, "text_to_score_bm25", ""),
                "producer": ctx.producer,
                "offset": d["offset"],
            }
        )
    con.close()
    for k in out:
        out[k].sort(key=lambda s: s["offset"])
    return out


def _load_source() -> dict[tuple[str, str], dict]:
    """Map (partition_key, dia_id) -> raw turn. The ingest uses group_<idx> as PK."""
    with open(SRC_BENCH) as f:
        bench = json.load(f)
    out: dict[tuple[str, str], dict] = {}
    for idx, conv in enumerate(bench):
        pk = f"group_{idx}"
        c = conv["conversation"]
        for k in list(c):
            if k.startswith("session_") and not k.endswith("_date_time"):
                session = c[k]
                if not isinstance(session, list):
                    continue
                for turn in session:
                    out[(pk, turn["dia_id"])] = {
                        "speaker": turn["speaker"],
                        "text": turn["text"],
                        "session_key": k,
                        "sample_id": conv["sample_id"],
                    }
    return out


# Hand-picked dia_ids exercising specific deictic patterns
# (filled in below after sampling once if needed). Start with C-bucket failing
# turns from previous testing + random for breadth. PK is group_<idx> where
# c2sub idx mapping is conv-42=0, conv-44=1, conv-47=2, conv-50=3.
SEED_TARGETS: list[tuple[str, str, str]] = [
    ("group_0", "D26:19", "C-bucket: plan to share recipes (conv-42)"),
    ("group_0", "D28:32", "C-bucket: plan to watch turtles (conv-42)"),
    ("group_3", "D28:20", "C-bucket: found car for blog (conv-50)"),
    ("group_2", "D30:18", "C-bucket: FIFA 23 advice (conv-47)"),
    ("group_0", "D11:16", "C-bucket: drama after hike (conv-42)"),
]


def _looks_interesting(text: str) -> str | None:
    """Return a label if the text has notable deictic content."""
    t = text.lower()
    labels = []
    if any(w in f" {t} " for w in (" you ", " your ", " yours ", " yourself ")):
        labels.append("you")
    if any(w in f" {t} " for w in (" this ", " that ", " these ", " those ", " here ", " there ")):
        labels.append("dem")
    if any(w in f" {t} " for w in (" he ", " she ", " they ", " his ", " her ", " their ")):
        labels.append("3p")
    if any(w in t for w in ("yesterday", "last week", "tomorrow", "next ", " ago", "today", "tonight", "recently")):
        labels.append("temp")
    return "+".join(labels) if labels else None


def main() -> None:
    src = _load_source()
    verb = _load_segments(VERBATIM_DB)
    deic = _load_segments(DEICTIC_DB)

    keys_verb = set(verb)
    keys_deic = set(deic)
    common = keys_verb & keys_deic & set(src)
    print(f"# c2sub: verbatim segs={sum(len(v) for v in verb.values())}, "
          f"deictic segs={sum(len(v) for v in deic.values())}, "
          f"common dia_ids={len(common)}", file=sys.stderr)

    # Build targets: seed + diverse-deictic sample
    seen = set()
    targets: list[tuple[str, str, str]] = []
    for sid, dia, label in SEED_TARGETS:
        if (sid, dia) in common:
            targets.append((sid, dia, label))
            seen.add((sid, dia))

    # Diverse sample: one from each deictic pattern bucket
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k in common:
        if k in seen:
            continue
        text = src[k]["text"]
        if len(text) < 20 or len(text) > 350:  # mid-length only for inspection
            continue
        label = _looks_interesting(text)
        if label:
            buckets[label].append(k)

    random.seed(42)
    for label, ks in sorted(buckets.items()):
        random.shuffle(ks)
        for k in ks[:2]:
            targets.append((k[0], k[1], label))
            seen.add(k)
            if sum(1 for _, _, lab in targets if lab == label) >= 2:
                break

    # Final: a few completely random for breadth
    leftover = [k for k in common if k not in seen]
    random.shuffle(leftover)
    for k in leftover[:4]:
        text = src[k]["text"]
        if len(text) < 20 or len(text) > 250:
            continue
        targets.append((k[0], k[1], "random"))

    # Emit
    for sid, dia, label in targets:
        msg = src[(sid, dia)]
        print("=" * 78)
        print(f"[{label}] {sid}/{dia} ({msg['speaker']}, {msg['session_key']})")
        print(f"SRC: {msg['text']}")
        print()
        print("-- VERBATIM (sidecar+xpurp2 + verbatim display) --")
        for seg in verb.get((sid, dia), []):
            print(f"  display: {seg['display']}")
            print(f"  embed  : {seg['embed'][:300]}")
            print()
        print("-- DEICTIC-RESOLVED (fresh, single-field) --")
        for seg in deic.get((sid, dia), []):
            print(f"  display: {seg['display']}")
            print(f"  embed  : {seg['embed'][:300]}")
            print()


if __name__ == "__main__":
    main()
