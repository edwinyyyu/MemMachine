"""Reading DAPR's packaging of a dataset (https://huggingface.co/datasets/UKPLab/dapr).

DAPR stores each document as an ordered list of passages and its qrels name passages.
Joining the passages with a separator gives the document text, and each passage's
offsets in it give the gold spans.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pyarrow.parquet as pq

from rbench.datasets._download import hf_files

REPO = "UKPLab/dapr"


def download(raw: Path, subset: str, doc_files: str) -> Path:
    base = raw / "dapr"
    if not (base / subset / "qrels").exists():
        hf_files(REPO, [f"{subset}/docs/{doc_files}", f"{subset}/queries/*", f"{subset}/qrels/*",
                        f"{subset}/nq-hard/*"], base)
    return base / subset


def documents(base: Path, files: list[str], separator: str, wanted: set[str] | None = None) -> dict:
    """{doc_id: (title, text, {passage_id: (start, end)})}, only `wanted` ids if given."""
    out = {}
    for name in files:
        for batch in pq.ParquetFile(base / "docs" / name).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if wanted is not None and row["doc_id"] not in wanted:
                    continue
                passages, passage_ids = row["passages"], row["passage_ids"]
                if isinstance(passages, str):  # some subsets store the lists as Python literals
                    passages, passage_ids = ast.literal_eval(passages), ast.literal_eval(passage_ids)
                offsets, at = {}, 0
                for passage_id, passage in zip(passage_ids, passages, strict=True):
                    offsets[passage_id] = (at, at + len(passage))
                    at += len(passage) + len(separator)
                out[row["doc_id"]] = (row["title"], separator.join(passages), offsets)
    return out


def queries(base: Path, splits: list[str]) -> list[tuple[str, str, str, list[str]]]:
    """(split, query id, text, relevant passage ids) for queries with a positive qrel."""
    out = []
    for split in splits:
        texts = {r["_id"]: r["text"] for r in pq.read_table(base / "queries" / f"{split}.parquet").to_pylist()}
        relevant: dict[str, list[str]] = {}
        for r in pq.read_table(base / "qrels" / f"{split}.parquet").to_pylist():
            if int(r["score"]) > 0:
                relevant.setdefault(r["query_id"], []).append(r["corpus_id"])
        out.extend((split, qid, texts[qid], relevant[qid]) for qid in texts if qid in relevant)
    return out
