"""LegalBench-RAG: expert-annotated clauses in contracts and privacy policies.

Decisions:
- Subsets: ContractNLI, CUAD, MAUD and PrivacyQA (all four of the benchmark).
- MAUD's 150 merger agreements are 12.7M of the corpus's 16.5M tokens, so a fixed sample of
  MAUD_DOCS (by sha256 of the path) is kept, with only the queries that fall in it.
- Gold: one alternative with every annotated snippet (all are needed). A query whose
  snippets span several documents, or whose snippet text does not match its offsets, is dropped.
- Variants: the query ("Consider <contract>; <ask>") and its ask alone; the ask does not
  name the contract, so it is only meaningful within the contract (doc scope).
- Split: a salted hash of the document path puts 30% in dev.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rbench.datasets._download import extract, fetch
from rbench.schema import hash_split

# The README's Dropbox folder (https://github.com/zeroentropy-ai/legalbenchrag), as a zip.
URL = ("https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4"
       "?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1")
SUBSETS = ("contractnli", "cuad", "maud", "privacy_qa")
MAUD_DOCS = 30


def download(raw: Path) -> None:
    if not (raw / "benchmarks").exists():
        extract(fetch(URL, raw / "lbr.zip"), raw)


def prepare(raw: Path) -> tuple[list[dict], list[dict], dict]:
    maud = sorted((f"maud/{p.name}" for p in (raw / "corpus" / "maud").iterdir()),
                  key=lambda path: hashlib.sha256(path.encode()).hexdigest())
    maud_kept = set(maud[:MAUD_DOCS])
    texts: dict[str, str] = {}
    rows = []
    stats = {"questions": 0, "kept": 0, "spans_mismatch": 0, "maud_sampled_out": 0}
    for subset in SUBSETS:
        for index, test in enumerate(json.loads((raw / "benchmarks" / f"{subset}.json").read_text())["tests"]):
            stats["questions"] += 1
            paths = {s["file_path"] for s in test["snippets"]}
            if subset == "maud" and not paths <= maud_kept:
                stats["maud_sampled_out"] += 1
                continue
            for path in paths:
                if path not in texts:
                    texts[path] = (raw / "corpus" / path).read_text()
            spans = [[s["file_path"], *s["span"]] for s in test["snippets"]]
            if len(paths) != 1 or any(texts[p][a:b] != s["answer"] for (p, a, b), s in zip(spans, test["snippets"], strict=True)):
                stats["spans_mismatch"] += 1
                continue
            doc_id = paths.pop()
            _, ask = re.fullmatch(r"(Consider [^;]*;)\s*(.*)", test["query"], flags=re.DOTALL).groups()
            rows.append({"qid": f"{subset}-{index}", "split": hash_split(doc_id), "doc_id": doc_id,
                         "variants": {"question": test["query"], "ask": ask}, "gold": [spans],
                         "features": {"subset": subset, "n_gold_spans": len(spans)}})
    stats["kept"] = len(rows)
    docs = [{"doc_id": p, "title": Path(p).stem, "text": t} for p, t in texts.items()]
    return docs, rows, stats
