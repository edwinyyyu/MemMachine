"""Natural Questions through DAPR: real Google queries against their Wikipedia article.

Decisions:
- Queries: DAPR's NQ-hard set (479 test queries whose answer needs document context)
  plus a fixed sample of NQ_SAMPLE other test queries, chosen by sha256 of the query id.
- Documents: only the queries' own articles (1,479), with DAPR's passages joined by a
  blank line. Every article is in the corpus, so distractors are other queries' articles.
- Gold: each annotated long-answer passage is its own alternative (any one answers).
- Split: a salted hash of the article id puts 30% in dev.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq

from rbench.datasets import _dapr
from rbench.schema import hash_split

NQ_SAMPLE = 1000  # beside NQ-hard; enough for tight intervals at a bounded embedding cost
DOC_FILES = [f"test-0000{i}-of-00003.parquet" for i in range(3)]


def download(raw: Path) -> None:
    _dapr.download(raw, "NaturalQuestions", "test-*")


def prepare(raw: Path) -> tuple[list[dict], list[dict], dict]:
    base = raw / "dapr" / "NaturalQuestions"
    hard = {r["query_id"] for r in pq.read_table(base / "nq-hard" / "0000.parquet").to_pylist()}
    candidates = _dapr.queries(base, ["test"])
    others = sorted((c for c in candidates if c[1] not in hard),
                    key=lambda c: hashlib.sha256(c[1].encode()).hexdigest())[:NQ_SAMPLE]
    chosen = [c for c in candidates if c[1] in hard] + others
    docs = _dapr.documents(base, DOC_FILES, "\n\n", {qid for _, qid, _, _ in chosen})
    rows, used, missing = [], {}, 0
    for _, qid, text, passages in chosen:
        if qid not in docs or any(p not in docs[qid][2] for p in passages):
            missing += 1
            continue
        used[qid] = docs[qid]
        gold = [[[qid, *docs[qid][2][p]]] for p in passages]
        rows.append({"qid": qid, "split": hash_split(qid), "doc_id": qid, "variants": {"question": text},
                     "gold": gold, "features": {"nq_hard": qid in hard, "n_gold_spans": 1}})
    out_docs = [{"doc_id": d, "title": t, "text": x} for d, (t, x, _) in used.items()]
    return out_docs, rows, {"questions": len(chosen), "kept": len(rows), "missing_doc": missing,
                            "nq_hard": sum(r["features"]["nq_hard"] for r in rows)}
