"""ConditionalQA through DAPR, with the scenario and question split from the original release.

Decisions:
- Documents: DAPR's gov.uk pages, passages joined by "\n" (DAPR's own rendering).
- Gold: one alternative holding every relevant passage; the answer depends on all of them
  (the conditions). Queries whose passages span several pages are dropped (none are).
- Variants: DAPR's text is scenario + question; the original ConditionalQA release
  (https://github.com/haitian-sun/ConditionalQA, v1_0) gives each separately, matched by id.
- Split: DAPR's dev and test.
"""

from __future__ import annotations

import json
from pathlib import Path

from rbench.datasets import _dapr
from rbench.datasets._download import fetch

ORIGINAL = "https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0/{}.json"


def download(raw: Path) -> None:
    _dapr.download(raw, "ConditionalQA", "*")
    for name in ("train", "dev"):
        fetch(ORIGINAL.format(name), raw / f"{name}.json")


def prepare(raw: Path) -> tuple[list[dict], list[dict], dict]:
    base = raw / "dapr" / "ConditionalQA"
    docs = _dapr.documents(base, ["test-00000-of-00001.parquet"], "\n")
    passage_doc = {p: d for d, (_, _, offsets) in docs.items() for p in offsets}
    original = {r["id"]: r for name in ("train", "dev") for r in json.loads((raw / f"{name}.json").read_text())}
    rows, multi_doc, unmatched = [], 0, 0
    for split, qid, text, passages in _dapr.queries(base, ["dev", "test"]):
        owners = {passage_doc[p] for p in passages}
        if len(owners) != 1:
            multi_doc += 1
            continue
        doc_id = owners.pop()
        variants = {"scenario_question": text}
        source = original.get(qid)
        if source is None or source["question"].strip() not in text:
            unmatched += 1
        else:
            variants["question"] = source["question"].strip()
            variants["scenario"] = source["scenario"].strip()
        gold = [[[doc_id, *docs[doc_id][2][p]] for p in sorted(passages, key=lambda p: docs[doc_id][2][p])]]
        rows.append({"qid": f"{split}:{qid}", "split": split, "doc_id": doc_id, "variants": variants,
                     "gold": gold, "features": {"n_gold_spans": len(gold[0])}})
    out_docs = [{"doc_id": d, "title": t, "text": x} for d, (t, x, _) in docs.items()]
    return out_docs, rows, {"kept": len(rows), "multi_doc": multi_doc, "question_variant_unmatched": unmatched}
