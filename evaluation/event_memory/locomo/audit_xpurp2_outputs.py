"""Audit xpurp2 c2sub outputs for model-bias error patterns.

Samples ~50 segments and dumps side-by-side: source chunk, neighbors
(both directions as the actual ingest saw them, since the c2sub
ingest predated the locomo_ingest default fix), xpurp2 memory text,
xpurp2 terse text, xpurp2 queries.

Reading the output manually categorizes errors:
  - HALLUCINATION: specifics in memory/terse not in source or neighbors
  - WRONG-CONTEXT: substitution where literal match exists but for
    wrong antecedent
  - GENERIC-vs-ADDRESSED: misclassification of "you" / "your"
  - WRONG TEMPORAL: using LATER-turn info to describe earlier event
  - INFO LOSS: substantive content in source dropped from memory/terse

Diverse sampling (10 per conv, mix of short/long messages).
"""

from __future__ import annotations
from artifacts import A  # canonical artifact names

import json
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

from memmachine_server.episodic_memory.event_memory.data_types import (
    decode_block,
    decode_context,
)


XPURP2_DB = (
    A("locomo-tslimv3bonaturalsidecarxpurp2-54n-l-nb8-c2sub-rep2.sqlite")
)
SRC_BENCH = "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/data/locomo10_c2sub.json"
NB = 8
SAMPLE_PER_CONV = 12


def _split_suffix(text: str, marker: str) -> tuple[str, str]:
    idx = text.rfind(marker)
    if idx == -1:
        return text, ""
    return text[:idx], text[idx + len(marker):]


def _parse_xpurp2_fields(producer: str, terse: str, embed_full: str,
                        bm25_full: str) -> dict:
    """Reverse the xpurp2 sidecar packing to recover {memory, queries,
    chunk, dates} from text_to_embed."""
    # bm25 = memory [ + "\nDates: " + dates ]
    memory, dates = _split_suffix(bm25_full, "\nDates: ")
    # embed = memory [ + "\nQueries: " q ] [ + "\n{producer}: " chunk ]
    #               [ + "\nDates: " dates ]
    rest = embed_full
    if not rest.startswith(memory):
        return {"memory": memory, "queries": "", "chunk": "",
                "dates": dates, "terse": terse, "parse_ok": False}
    rest = rest[len(memory):]
    if dates and rest.endswith("\nDates: " + dates):
        rest = rest[: -len("\nDates: " + dates)]
    queries = ""
    chunk = ""
    chunk_marker = f"\n{producer}: "
    if rest.startswith("\nQueries: "):
        r2 = rest[len("\nQueries: "):]
        if chunk_marker in r2:
            queries, chunk = r2.split(chunk_marker, 1)
        else:
            queries = r2
    elif rest.startswith(chunk_marker):
        chunk = rest[len(chunk_marker):]
    return {"memory": memory, "queries": queries, "chunk": chunk,
            "dates": dates, "terse": terse, "parse_ok": True}


def _load_xpurp2_segments(db_path: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(segment_store_sg)")]
    out: list[dict] = []
    for row in con.execute("SELECT * FROM segment_store_sg"):
        d = dict(zip(cols, row))
        blk = decode_block(json.loads(d["block"].decode()))
        ctx = decode_context(json.loads(d["context"].decode()))
        props = json.loads(d["properties"])
        sid = props["locomo_session_id"]
        sid = sid["v"] if isinstance(sid, dict) else sid
        dia = props["dia_id"]
        dia = dia["v"] if isinstance(dia, dict) else dia
        fields = _parse_xpurp2_fields(
            ctx.producer, blk.text,
            ctx.text_to_embed, ctx.text_to_score_bm25,
        )
        out.append({
            "partition_key": d["partition_key"],
            "session_id": sid,
            "dia_id": dia,
            "producer": ctx.producer,
            **fields,
        })
    con.close()
    return out


def _parse_locomo_time(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")


def _build_dia_map(conv: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    c = conv["conversation"]
    for k in c:
        if k.startswith("session_") and not k.endswith("_date_time"):
            n = k.split("_")[1]
            date_str = c.get(f"session_{n}_date_time", "")
            ts = _parse_locomo_time(date_str)
            session = c[k]
            if not isinstance(session, list):
                continue
            for turn in session:
                out[turn["dia_id"]] = {
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "timestamp": ts,
                    "dia_id": turn["dia_id"],
                }
    return out


def _surrounding(dia_map, target_id, nb, both=True):
    session_prefix = target_id.split(":")[0]
    ordered = sorted(
        (k for k in dia_map if k.startswith(session_prefix + ":")),
        key=lambda x: int(x.split(":")[1]),
    )
    if target_id not in ordered:
        return [], []
    idx = ordered.index(target_id)
    before_ids = ordered[max(0, idx - nb):idx]
    after_ids = ordered[idx + 1: idx + 1 + nb] if both else []
    return (
        [(dia_map[i]["speaker"], dia_map[i]["text"]) for i in before_ids],
        [(dia_map[i]["speaker"], dia_map[i]["text"]) for i in after_ids],
    )


def main() -> None:
    print("Loading xpurp2 segments...", file=sys.stderr)
    segs = _load_xpurp2_segments(XPURP2_DB)
    by_pk: dict[str, list[dict]] = defaultdict(list)
    for s in segs:
        by_pk[s["partition_key"]].append(s)
    print(f"# total xpurp2 segments: {len(segs)}; "
          f"partitions: {sorted(by_pk)}", file=sys.stderr)

    with open(SRC_BENCH) as f:
        bench = json.load(f)
    dia_maps_by_pk = {
        f"group_{i}": _build_dia_map(c) for i, c in enumerate(bench)
    }

    random.seed(2026)
    for pk in sorted(by_pk):
        bucket = by_pk[pk]
        # Diverse sample: prefer messages with substantive content
        # (non-empty memory, mid-to-long source). Then sort by dia_id
        # for readability.
        candidates = [s for s in bucket if s.get("memory") and len(s["memory"]) > 40]
        random.shuffle(candidates)
        sample = sorted(
            candidates[:SAMPLE_PER_CONV],
            key=lambda s: (s["session_id"], s["dia_id"]),
        )
        dia_map = dia_maps_by_pk.get(pk, {})
        print(f"\n{'='*92}")
        print(f"=== PARTITION {pk}  -- {len(sample)} sampled segments ===")
        print(f"{'='*92}")
        for s in sample:
            dia = s["dia_id"]
            src_msg = dia_map.get(dia)
            if not src_msg:
                continue
            before, after = _surrounding(dia_map, dia, NB, both=True)
            print(f"\n--- {dia}  ({s['producer']}) ---")
            print(f"SRC: {src_msg['text']}")
            print()
            print(f"MEMORY: {s['memory']}")
            print(f"TERSE : {s['terse']}")
            if s.get("queries"):
                print(f"QUERIES: {s['queries']}")
            print()
            print("PRIOR TURNS:")
            for p, t in before:
                txt = t if len(t) <= 200 else t[:200] + "..."
                print(f"  - {p}: {txt}")
            if after:
                print("LATER TURNS (as actually shown to model):")
                for p, t in after:
                    txt = t if len(t) <= 200 else t[:200] + "..."
                    print(f"  - {p}: {txt}")
            print()


if __name__ == "__main__":
    main()
