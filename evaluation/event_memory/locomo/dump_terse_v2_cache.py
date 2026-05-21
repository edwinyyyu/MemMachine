"""Dump terse-decoupled-v2 segments into a component-level cache.

Reads the terse-decoupled-v2 segment store and parses every segment back
into its raw components -- {memory, terse, queries, chunk, dates} -- so a
re-assembly segmenter can rebuild arbitrary field assignments
(block.text / text_to_embed / text_to_score_bm25) WITHOUT re-running the
segmenter LLM. This is the substrate for the decoupling ablation.

Each segment's three stored texts are:
  block.text          = terse
  text_to_score_bm25  = memory [ + "\\nDates: " + dates ]
  text_to_embed       = memory [ + "\\nQueries: " + q ]
                               [ + "\\n{producer}: " + chunk ]
                               [ + "\\nDates: " + dates ]

The parse is verified by round-tripping text_to_embed; any mismatch is
reported and the record flagged.
"""

from __future__ import annotations

import json
import sqlite3
import sys

from memmachine_server.episodic_memory.event_memory.data_types import (
    decode_block,
    decode_context,
)

SRC_DB = "locomo-terse-decoupled-v2-m54nl-nb8b-fb.sqlite"
OUT = "cache-terse-v2-raw.json"


def _split_suffix(text: str, marker: str) -> tuple[str, str]:
    """Split off a `marker`-prefixed suffix; return (head, suffix_body)."""
    idx = text.rfind(marker)
    if idx == -1:
        return text, ""
    return text[:idx], text[idx + len(marker) :]


def main() -> None:
    con = sqlite3.connect(SRC_DB)
    cols = [r[1] for r in con.execute("PRAGMA table_info(segment_store_sg)")]
    records: list[dict] = []
    mismatches = 0
    for row in con.execute("SELECT * FROM segment_store_sg"):
        d = dict(zip(cols, row))
        blk = decode_block(json.loads(d["block"].decode()))
        ctx = decode_context(json.loads(d["context"].decode()))
        producer = ctx.producer
        terse = blk.text
        embed_full = ctx.text_to_embed
        bm25_full = ctx.text_to_score_bm25

        # bm25_full = memory [ + "\nDates: " + dates ]
        memory, dates = _split_suffix(bm25_full, "\nDates: ")

        # embed_full = memory [ + "\nQueries: " q ] [ + "\n{producer}: " chunk ]
        #                     [ + "\nDates: " dates ]
        rest = embed_full
        if not rest.startswith(memory):
            mismatches += 1
            continue
        rest = rest[len(memory) :]
        if dates and rest.endswith("\nDates: " + dates):
            rest = rest[: -len("\nDates: " + dates)]
        queries = ""
        chunk = ""
        chunk_marker = f"\n{producer}: "
        if rest.startswith("\nQueries: "):
            r2 = rest[len("\nQueries: ") :]
            if chunk_marker in r2:
                queries, chunk = r2.split(chunk_marker, 1)
            else:
                queries = r2
        elif rest.startswith(chunk_marker):
            chunk = rest[len(chunk_marker) :]

        # Round-trip verification.
        rebuilt = memory
        if queries:
            rebuilt += "\nQueries: " + queries
        if chunk:
            rebuilt += chunk_marker + chunk
        if dates:
            rebuilt += "\nDates: " + dates
        if rebuilt != embed_full:
            mismatches += 1
            continue

        props = json.loads(d["properties"])
        sid = props["locomo_session_id"]
        sid = sid["v"] if isinstance(sid, dict) else sid
        dia = props["dia_id"]
        dia = dia["v"] if isinstance(dia, dict) else dia

        records.append(
            {
                "partition_key": d["partition_key"],
                "session_id": sid,
                "dia_id": dia,
                "index": d["index"],
                "offset": d["offset"],
                "producer": producer,
                "terse": terse,
                "memory": memory,
                "queries": queries,
                "chunk": chunk,
                "dates": dates,
            }
        )
    con.close()

    with open(OUT, "w") as f:
        json.dump(records, f)
    print(f"dumped {len(records)} segments to {OUT}; {mismatches} parse mismatches")
    if mismatches:
        print("WARNING: mismatches present -- parse logic is incomplete", file=sys.stderr)


if __name__ == "__main__":
    main()
