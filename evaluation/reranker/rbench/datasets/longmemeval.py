"""LongMemEval (cleaned release): questions over one user's long chat history each.

Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (MIT). The M file
gives each question its own history of about 500 sessions (~1.05M tokens); the S file the
same 500 questions over ~50 sessions. Only M is evaluated; S serves to test the benchmark.

Decisions:
- A history is a partition: it is processed alone, as MemMachine's LongMemEval evaluation
  ingests each question's history into its own memory. Sessions shared between histories
  carry different dates in each (all 47,077 shared session ids do in M), and EventMemory
  embeds the date, so nothing is shared across partitions.
- Turns are events: producer "User" or "Assistant", content stripped. The files are not in
  date order and date sessions only to the minute, so histories.py sorts sessions by date
  (ties in file order) and orders turns with unrendered microsecond increments, in UTC.
- The 30 abstention questions ("_abs") are excluded: they have no answer in the history,
  and LongMemEval's authors also skip them in retrieval evaluation.
- Gold: every turn marked has_answer, as one acceptable answer (all needed; multi-session
  questions span several sessions). `gold_sessions` lists the evidence sessions for a
  LongMemEval-style session-level view.
- No dev split: every question is evaluated (split "test"), and choices among
  configurations are made and measured on the same questions (decided 2026-11-06).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rbench.datasets._download import hf_files
from rbench.paths import PREPARED

REPO = "xiaowu0162/longmemeval-cleaned"
FILES = {"longmemeval_m": "longmemeval_m_cleaned.json", "longmemeval_s": "longmemeval_s_cleaned.json"}
RAW_DIR = "longmemeval"  # both files share one download folder


def session_datetime(date: str) -> datetime:
    """LongMemEval's "2023/04/10 (Mon) 23:07" as a UTC datetime."""
    return datetime.strptime(date, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=UTC)


def download(raw: Path, name: str) -> Path:
    path = raw.parent / RAW_DIR / FILES[name]
    if not path.exists():
        hf_files(REPO, [FILES[name]], raw.parent / RAW_DIR)
    return path


def prepare(name: str, raw_root: Path) -> dict:
    """Write data/prepared/<name>/{queries.jsonl, histories/<qid>.json, stats.json}; return stats."""
    import ijson

    path = download(raw_root / RAW_DIR, name)
    out = PREPARED / name
    (out / "histories").mkdir(parents=True, exist_ok=True)
    rows, abstention, evidence_turns = [], 0, 0
    with path.open("rb") as file:
        for q in ijson.items(file, "item"):
            if q["question_id"].endswith("_abs"):
                abstention += 1
                continue
            sessions, gold, gold_sessions = [], [], []
            for s, (sid, date, turns) in enumerate(zip(q["haystack_session_ids"], q["haystack_dates"],
                                                        q["haystack_sessions"], strict=True)):
                kept = [{"role": t["role"], "content": t["content"].strip(), "has_answer": bool(t.get("has_answer"))}
                        for t in turns]
                sessions.append({"id": sid, "date": date, "turns": kept})
                marked = [i for i, t in enumerate(kept) if t["has_answer"]]
                gold += [[s, i] for i in marked]
                if sid in q["answer_session_ids"]:
                    gold_sessions.append(s)
            if not gold:  # every answerable question has marked turns; guard anyway
                raise ValueError(f"{q['question_id']}: no has_answer turn")
            evidence_turns += len(gold)
            qid = q["question_id"]
            (out / "histories" / f"{qid}.json").write_text(json.dumps({"qid": qid, "sessions": sessions}))
            rows.append({"qid": qid, "split": "test", "variants": {"question": q["question"].strip()},
                         "gold_turns": [gold], "gold_sessions": gold_sessions,
                         "features": {"question_type": q["question_type"], "question_date": q["question_date"],
                                      "n_sessions": len(sessions), "n_evidence_turns": len(gold),
                                      "n_evidence_sessions": len(gold_sessions)}})
    with (out / "queries.jsonl").open("w") as file:
        for r in rows:
            file.write(json.dumps(r) + "\n")
    stats = {"name": name, "source": f"https://huggingface.co/datasets/{REPO} ({FILES[name]})",
             "queries": {s: sum(r["split"] == s for r in rows) for s in ("dev", "test")},
             "abstention_excluded": abstention, "evidence_turns": evidence_turns,
             "default_variant": "question", "scope": "history", "partitioned": True}
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats


def history(name: str, qid: str) -> dict:
    return json.loads((PREPARED / name / "histories" / f"{qid}.json").read_text())
