"""QASPER v0.3: questions about NLP papers by readers who had seen only the title and abstract.

Decisions:
- Documents: title, abstract, then each section's name and paragraphs, joined by blank lines.
- Gold: per annotator, the highlighted evidence sentences (tighter than the evidence
  paragraphs, used when highlights are missing); each annotator is an alternative.
  Evidence that is a figure or table caption ("FLOAT SELECTED") is dropped, as are
  unanswerable answers and evidence not found verbatim in the text.
- Split: the official dev and test sets (train is unused).
"""

from __future__ import annotations

import json
from pathlib import Path

from rbench.datasets._download import extract, fetch

URLS = {
    "traindev.tgz": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test.tgz": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}
FILES = {"dev": "qasper-dev-v0.3.json", "test": "qasper-test-v0.3.json"}


def download(raw: Path) -> None:
    if all((raw / f).exists() for f in FILES.values()):
        return
    for name, url in URLS.items():
        extract(fetch(url, raw / name), raw)


def locate(text: str, needle: str) -> tuple[int, int] | None:
    needle = needle.strip()
    at = text.find(needle) if needle else -1
    return (at, at + len(needle)) if at >= 0 else None


def paper_text(paper: dict) -> str:
    parts = [paper["title"], paper["abstract"]]
    for section in paper["full_text"]:
        if section["section_name"]:
            parts.append(section["section_name"])
        parts.extend(p for p in section["paragraphs"] if p.strip())
    return "\n\n".join(parts)


def prepare(raw: Path) -> tuple[list[dict], list[dict], dict]:
    docs, rows = [], []
    stats = {"questions": 0, "kept": 0, "spans_missing": 0, "float_only": 0, "unanswerable": 0}
    for split, name in FILES.items():
        for paper_id, paper in json.loads((raw / name).read_text()).items():
            text = paper_text(paper)
            docs.append({"doc_id": paper_id, "title": paper["title"], "text": text})
            for qa in paper["qas"]:
                stats["questions"] += 1
                gold = []
                for answer in (a["answer"] for a in qa["answers"]):
                    if answer["unanswerable"]:
                        stats["unanswerable"] += 1
                        continue
                    pieces = [p for p in answer["highlighted_evidence"] or answer["evidence"]
                              if not p.startswith("FLOAT SELECTED")]
                    if not pieces:
                        stats["float_only"] += 1
                        continue
                    spans = [locate(text, p) for p in pieces]
                    if None in spans:
                        stats["spans_missing"] += 1
                        continue
                    gold.append([[paper_id, s, e] for s, e in spans])
                if not gold:
                    continue
                question = qa["question"].strip()
                rows.append({"qid": qa["question_id"], "split": split, "doc_id": paper_id,
                             "variants": {"question": question, "scoped": f'In the paper "{paper["title"]}": {question}'},
                             "gold": gold, "features": {"n_gold_spans": min(len(a) for a in gold)}})
    stats["kept"] = len(rows)
    return docs, rows, stats
