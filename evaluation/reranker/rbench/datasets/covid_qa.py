"""COVID-QA (deepset): biomedical experts' questions about CORD-19 articles.

Decisions:
- Documents: the article text as given; the title is its first line.
- Gold: each answer span, at its stated offset (or found verbatim if the offset is off),
  is an alternative.
- Split: no official split, so a salted hash of the article id puts 30% in dev.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from rbench.datasets._download import hf_files
from rbench.schema import hash_split

REPO = "deepset/covid_qa_deepset"


def download(raw: Path) -> None:
    if not list(raw.rglob("*.parquet")):
        hf_files(REPO, ["*.parquet"], raw)


def prepare(raw: Path) -> tuple[list[dict], list[dict], dict]:
    table = [row for path in sorted(raw.rglob("*.parquet")) for row in pq.read_table(path).to_pylist()]
    docs: dict[str, dict] = {}
    rows, missing = [], 0
    for row in table:
        doc_id, text = str(row["document_id"]), row["context"]
        title = text.split("\n", 1)[0].strip()
        docs.setdefault(doc_id, {"doc_id": doc_id, "title": title, "text": text})
        gold = []
        for answer, start in zip(row["answers"]["text"], row["answers"]["answer_start"], strict=True):
            if text[start : start + len(answer)] == answer:
                gold.append([[doc_id, start, start + len(answer)]])
            elif (at := text.find(answer.strip())) >= 0 and answer.strip():
                gold.append([[doc_id, at, at + len(answer.strip())]])
        if not gold:
            missing += 1
            continue
        question = row["question"].strip()
        rows.append({"qid": str(row["id"]), "split": hash_split(doc_id), "doc_id": doc_id,
                     "variants": {"question": question, "scoped": f'In the article "{title}": {question}'},
                     "gold": gold, "features": {"n_gold_spans": 1}})
    return list(docs.values()), rows, {"questions": len(table), "kept": len(rows), "spans_missing": missing}
