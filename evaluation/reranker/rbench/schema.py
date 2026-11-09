"""The prepared-dataset format: loading, validation and writing.

A prepared dataset is three files under data/prepared/<name>/:

  docs.jsonl     {"doc_id", "title", "text"}, one per line. A document's position in the
                 file is its index everywhere else (chunk tables refer to it).
  queries.jsonl  {"qid", "variants", "gold", ["split"], ["doc_id"], ["scope_docs"], ["features"]}
  stats.json     counts and the preparer's decisions (what was dropped and why)

Gold is a list of alternatives; any one alternative fully answers the query. An
alternative is a list of spans [doc_id, start, end] (character offsets, end exclusive)
that are all needed. "First gold" is reached when any span of any alternative has been
read in full; "every gold" when every span of one alternative has.

`split`, where present, is the source dataset's split a query comes from: provenance only.
Nothing is held out (decided 2026-11-06): every sampled query is evaluated.

`variants` holds the same information need phrased in different ways, all taken from the
source data (never written by a model). `doc_id` names the query's own document (needed
for doc scope); `scope_docs` lists the documents a query may retrieve from when its
candidates are a fixed set (SWE-bench's files of one repository snapshot).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

from rbench.paths import PREPARED


@dataclass(frozen=True)
class Corpus:
    doc_ids: tuple[str, ...]
    titles: tuple[str | None, ...]
    texts: tuple[str, ...]

    @cached_property
    def index(self) -> dict[str, int]:
        return {d: i for i, d in enumerate(self.doc_ids)}


@cache
def corpus(dataset: str) -> Corpus:
    rows = read_jsonl(PREPARED / dataset / "docs.jsonl")
    return Corpus(tuple(r["doc_id"] for r in rows), tuple(r.get("title") for r in rows), tuple(r["text"] for r in rows))


@cache
def queries(dataset: str) -> tuple[dict, ...]:
    return tuple(read_jsonl(PREPARED / dataset / "queries.jsonl"))


def stats(dataset: str) -> dict:
    return json.loads((PREPARED / dataset / "stats.json").read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file]


def validate(docs: list[dict], rows: list[dict]) -> list[str]:
    """Problems with a prepared dataset; empty when it follows the format."""
    problems = []
    lengths = {}
    for d in docs:
        if d["doc_id"] in lengths:
            problems.append(f"duplicate doc_id {d['doc_id']}")
        lengths[d["doc_id"]] = len(d["text"])
    seen = set()
    for q in rows:
        where = f"query {q.get('qid')}"
        if q["qid"] in seen:
            problems.append(f"duplicate qid {q['qid']}")
        seen.add(q["qid"])
        if not q.get("variants") or not all(isinstance(v, str) and v.strip() for v in q["variants"].values()):
            problems.append(f"{where}: empty variant")
        if not q.get("gold") or not all(q["gold"]):
            problems.append(f"{where}: empty gold or alternative")
        for alternative in q.get("gold", []):
            for doc_id, start, end in alternative:
                if doc_id not in lengths or not 0 <= start < end <= lengths[doc_id]:
                    problems.append(f"{where}: span {doc_id}:{start}-{end} out of bounds")
        for doc_id in [q.get("doc_id")] * ("doc_id" in q) + q.get("scope_docs", []):
            if doc_id not in lengths:
                problems.append(f"{where}: unknown doc {doc_id}")
    return problems


def write(dataset: str, docs: list[dict], rows: list[dict], stats_: dict) -> Path:
    """Validate and write a prepared dataset; raises if it does not follow the format."""
    problems = validate(docs, rows)
    if problems:
        raise ValueError(f"{dataset}: {len(problems)} problems, first: {problems[:5]}")
    out = PREPARED / dataset
    out.mkdir(parents=True, exist_ok=True)
    for name, items in (("docs.jsonl", docs), ("queries.jsonl", rows)):
        with (out / name).open("w") as file:
            for item in items:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(stats_, indent=1, ensure_ascii=False) + "\n")
    corpus.cache_clear()
    queries.cache_clear()
    return out
