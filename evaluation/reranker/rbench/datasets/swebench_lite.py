"""SWE-bench Lite as file localization: find the code a GitHub issue's fix edits.

Query: the issue (SWE-bench Lite `problem_statement`). Candidates: the repository's files at the
issue's base commit. Gold: the original lines the reference patch edits.

Sources (Hugging Face parquet, no repository clones):
  SWE-bench_Lite             instances: issue, repo, base_commit, reference patch (dev 23, test 300)
  SWE-bench_Lite_oracle      prompts embedding the edited files' full text
  SWE-bench_Lite_bm25_27K    prompts embedding the files BM25 retrieved, up to a 27k-token budget
The prompts embed each file as "[start of p]\\n1 l1\\n2 l2\\n...\\n[end of p]", one numbered line per
"\\n"-separated piece, so stripping the numbers rebuilds the file exactly (a trailing newline
shows as a last, empty numbered line). Checked byte-exact against
raw.githubusercontent.com at base_commit for 20 docs (the 8 relocated-hunk files below and 12 random
ones). 3 docs lack a trailing newline (sphinx's doc/usage/extensions/example_google.py), and the
hunks that reach end of file (8) agree with the reconstruction about it. No doc contains "\\r".
Every block's numbering runs unbroken to its end tag, every hunk aligns, and every gold file is in
the oracle prompt, so no GitHub fetch is needed.

Decisions, with the counts that justify them:
  * Candidates = gold files ∪ BM25 files, in `scope_docs`. BM25 chose its files for
    their lexical closeness to the issue, which makes them hard distractors. 1,385 docs, 4.29 per issue (median 4, max 17),
    38.1M characters. Only 156 of the 323 gold files (48%) are BM25 hits; the rest are added.
  * README excluded. Every oracle prompt holds exactly two blocks, the README and the edited
    file: the README is part of SWE-bench's prompt template, not a retrieval hit. Its block is the
    only non-.py block in any prompt (README.md or README.rst; 323 removed, one per issue).
  * Gold = for each hunk, the original lines it removes or modifies ('-' lines), contiguous lines
    merged into one span; one alternative holds all spans, since every hunk is part of the fix.
    Lite patches touch exactly one existing file with 1-3 hunks (202 / 81 / 40 issues), with no
    new files, renames or missing-newline markers, so no issue is dropped.
  * 11 of 484 hunks (8 issues) do not match the file at their stated line. These 8 files are
    byte-identical to GitHub at base_commit, so the patch's line numbers are wrong, not the file.
    Each such hunk's old side occurs exactly once in the file, 2 to 41 lines from the stated
    position. As git apply does, the exact match nearest the stated line is used; an issue whose
    hunk matches nowhere would be dropped (none is).
  * A hunk that only adds lines is anchored on the nearest non-blank original line at or before
    the insertion point (after it, if none is before). The line just before is often the blank
    line between two functions, which says nothing about where the code goes: under that rule
    51 anchors were a lone "\\n". 171 spans in 127 issues come from anchors.
  * A run of removed lines that is entirely blank is extended up to the nearest non-blank line
    (down if none is above), for the same reason: 5 runs. One then touches a neighbouring span and
    merges (scikit-learn-25500), giving 587 spans in all, none blank.
  * mwaskom__seaborn-2848: the oracle and BM25 prompts embed an older issue text (5,820 chars,
    about PairGrid hue in map) than Lite's (965 chars, pairplot with hue_order). The query uses
    Lite's text; the prompt is parsed with its own. Its BM25 files were retrieved for the old text.
  * Variants: "issue" is problem_statement unchanged; "title" is its first non-empty line, which in
    SWE-bench's format is the issue title (the first line is non-empty in all 323).
"""

from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from rbench.datasets._download import hf_files

# The three Hugging Face datasets: instances, edited files, and BM25-retrieved files.
SOURCES = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "oracle": "princeton-nlp/SWE-bench_Lite_oracle",
    "bm25": "princeton-nlp/SWE-bench_Lite_bm25_27K",
}
# Lite's official splits, kept as the benchmark's dev and test.
SPLITS = ("dev", "test")
# Each source has one parquet shard per split.
PARQUET = "data/{split}-00000-of-00001.parquet"
# SWE-bench's prompt text before the issue; the issue follows it verbatim.
PROMPT_HEAD = (
    "You will be provided with a partial code base and an issue statement "
    "explaining a problem to resolve.\n<issue>\n"
)
# What follows the issue in the prompt, right before the first file block.
CODE_OPEN = "\n</issue>\n<code>\n"
# The line that ends the file blocks.
CODE_CLOSE = "</code>"
# The header line of a file block.
BLOCK_START = re.compile(r"\[start of (.+)\]")
# A unified-diff hunk header; omitted lengths mean 1.
HUNK_HEADER = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# The prompt template's always-included files, recognised by name.
README_PREFIX = "readme"
# Markers of code pasted in an issue (features.issue_has_code).
CODE_MARKERS = ("```", "Traceback (most recent call last)")


# ---------------------------------------------------------------- download


def download(raw: Path) -> None:
    """Fetch the three sources' parquet shards into raw/<dataset>/ unless they are there."""
    for repo in SOURCES.values():
        into = raw / repo.split("/")[1]
        if not all((into / PARQUET.format(split=split)).exists() for split in SPLITS):
            hf_files(repo, ["data/*.parquet", "README.md"], into)


def load(raw: Path, source: str, split: str) -> list[dict]:
    return pq.read_table(raw / SOURCES[source].split("/")[1] / PARQUET.format(split=split)).to_pylist()


# ---------------------------------------------------------------- file blocks


def parse_blocks(prompt: str, issue: str) -> list[tuple[str, list[str]]]:
    """(path, numbered lines) for each file block of a SWE-bench prompt, in prompt order."""
    head = PROMPT_HEAD + issue + CODE_OPEN
    start = len(head) if prompt.startswith(head) else prompt.index(CODE_OPEN) + len(CODE_OPEN)
    lines = prompt[start:].split("\n")
    blocks, i = [], 0
    while lines[i] != CODE_CLOSE:
        if not lines[i]:
            i += 1
            continue
        match = BLOCK_START.fullmatch(lines[i])
        if not match:
            raise ValueError(f"unexpected line outside a file block: {lines[i][:80]!r}")
        path, end = match.group(1), lines.index(f"[end of {match.group(1)}]", i + 1)
        blocks.append((path, lines[i + 1 : end]))
        i = end + 1
    return blocks


def strip_line_numbers(numbered: list[str]) -> str:
    """Rebuild a file from its "N text" lines; the numbers must run 1, 2, 3, ..."""
    lines = []
    for n, line in enumerate(numbered, start=1):
        prefix = f"{n} "
        if not line.startswith(prefix):
            raise ValueError(f"line {n} lacks its number: {line[:80]!r}")
        lines.append(line[len(prefix) :])
    return "\n".join(lines).replace("\r\n", "\n")


def file_texts(prompt: str, issue: str) -> dict[str, str]:
    return {path: strip_line_numbers(numbered) for path, numbered in parse_blocks(prompt, issue)}


# ---------------------------------------------------------------- the diff


def parse_patch(patch: str) -> list[dict]:
    """Files of a unified diff: {"old", "new", "hunks"}; a path is None for /dev/null."""
    lines = patch.split("\n")
    files, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            files.append({"old": None, "new": None, "hunks": [], "seen": set()})
        elif line[:4] in ("--- ", "+++ ") and files and line[:3] not in files[-1]["seen"]:
            files[-1]["seen"].add(line[:3])
            key, prefix = ("old", "a/") if line[0] == "-" else ("new", "b/")
            path = line[4:].split("\t")[0]
            files[-1][key] = None if path == "/dev/null" else path.removeprefix(prefix)
        elif line.startswith("@@ "):
            hunk, i = parse_hunk(lines, i)
            files[-1]["hunks"].append(hunk)
            continue
        i += 1
    for f in files:
        del f["seen"]
    return files


def parse_hunk(lines: list[str], i: int) -> tuple[dict, int]:
    """The hunk whose header is lines[i], and the index after it.

    The body is consumed by its stated line counts, so content such as "--- x" cannot end it.
    Lines are (tag, text) with tag ' ', '-', '+' or '\\' (a missing-newline marker).
    """
    match = HUNK_HEADER.match(lines[i])
    old_start, old_len, new_len = int(match.group(1)), int(match.group(2) or 1), int(match.group(4) or 1)
    body, n_old, n_new = [], 0, 0
    i += 1
    while n_old < old_len or n_new < new_len or (i < len(lines) and lines[i].startswith("\\")):
        tag, text = (lines[i][0], lines[i][1:]) if lines[i] else (" ", "")
        if tag not in " -+\\":
            raise ValueError(f"bad hunk line {lines[i][:80]!r}")
        n_old += tag in " -"
        n_new += tag in " +"
        body.append((tag, text))
        i += 1
    return {"old_start": old_start, "old_len": old_len, "lines": body}, i


# ---------------------------------------------------------------- alignment


def line_table(text: str) -> list[tuple[int, int, str]]:
    """(start, end including the newline, text) of each line of a file."""
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
    table, pos = [], 0
    for part in parts:
        end = pos + len(part)
        table.append((pos, end + 1 if end < len(text) else end, part))
        pos = end + 1
    return table


def align_hunk(lines: list[str], hunk: dict) -> tuple[int | None, int | None]:
    """(1-based first line, offset from the stated line) where the hunk's old side matches.

    The stated line is tried first; failing that, the exact match nearest to it, as git apply
    does. (None, None) when the old side occurs nowhere.
    """
    old = [text.rstrip("\r") for tag, text in hunk["lines"] if tag in " -"]
    stated = hunk["old_start"] if hunk["old_len"] > 0 else hunk["old_start"] + 1

    def matches(start):
        return start >= 1 and lines[start - 1 : start - 1 + len(old)] == old

    if matches(stated):
        return stated, 0
    hits = [s for s in range(1, len(lines) - len(old) + 2) if matches(s)]
    if not hits:
        return None, None
    best = min(hits, key=lambda s: (abs(s - stated), s))
    return best, best - stated


def eof_newline_errors(hunk: dict, start: int, n_lines: int, ends_with_newline: bool) -> tuple[list[str], bool]:
    """Disagreements between the hunk and the file about a final newline, and whether it was checkable."""
    errors, marker, pos, prev = [], False, start, None
    for tag, _ in hunk["lines"]:
        if tag in " -":
            pos += 1
        elif tag == "\\" and prev in (" ", "-"):
            marker = True
            if ends_with_newline or pos - 1 != n_lines:
                errors.append("patch says no newline at end of file; the file disagrees")
        prev = tag
    reaches_eof = pos > start and pos - 1 == n_lines
    if reaches_eof and not marker and not ends_with_newline:
        errors.append("patch implies a newline at end of file; the file has none")
    return errors, reaches_eof


def removed_lines(hunk: dict, start: int) -> set[int]:
    """1-based original lines the hunk removes or modifies."""
    removed, pos = set(), start
    for tag, _ in hunk["lines"]:
        if tag in " -":
            if tag == "-":
                removed.add(pos)
            pos += 1
    return removed


def insertion_points(hunk: dict, start: int) -> list[int]:
    """For a hunk that only adds lines, the original line before which each run of '+' lines goes."""
    if any(tag == "-" for tag, _ in hunk["lines"]):
        return []
    points, pos, prev = [], start, None
    for tag, _ in hunk["lines"]:
        if tag == "+" and prev != "+":
            points.append(pos)
        if tag == " ":
            pos += 1
        if tag != "\\":
            prev = tag
    return points


# ---------------------------------------------------------------- gold lines


def nearest_nonblank(lines: list[str], before: int, after: int) -> int | None:
    """The nearest non-blank 1-based line at or before `before`, else at or after `after`."""
    for n in range(min(before, len(lines)), 0, -1):
        if lines[n - 1].strip():
            return n
    for n in range(max(after, 1), len(lines) + 1):
        if lines[n - 1].strip():
            return n
    return None


def anchor_insertion(lines: list[str], point: int) -> int | None:
    """The line that locates an insertion made between original lines point-1 and point."""
    return nearest_nonblank(lines, point - 1, point)


def runs(line_numbers) -> list[list[int]]:
    """Contiguous runs [first, last] of sorted line numbers."""
    out = []
    for n in sorted(line_numbers):
        if out and out[-1][1] == n - 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def extend_blank_runs(lines: list[str], removed: set[int]) -> set[int]:
    """Removed lines plus, for each all-blank run, the lines up to the nearest non-blank one."""
    out = set(removed)
    for first, last in runs(removed):
        if any(lines[n - 1].strip() for n in range(first, last + 1)):
            continue
        n = nearest_nonblank(lines, first - 1, last + 1)
        if n is not None:
            out |= set(range(n, first) if n < first else range(last + 1, n + 1))
    return out


def gold_for_file(text: str, hunks: list[dict]) -> dict:
    """Align a file's hunks and collect its gold lines.

    Returns {"lines", "anchors", "offsets", "eof_checks", "errors", "table"}; errors are hunks
    that match nowhere or disagree with the file about its final newline.
    """
    table = line_table(text)
    lines = [t.rstrip("\r") for _, _, t in table]
    removed, anchors, offsets, errors, eof_checks = set(), set(), [], [], 0
    for hunk in hunks:
        start, offset = align_hunk(lines, hunk)
        if start is None:
            errors.append(f"hunk @@ -{hunk['old_start']},{hunk['old_len']} matches nowhere")
            continue
        offsets.append(offset)
        eof_errors, checked = eof_newline_errors(hunk, start, len(lines), text.endswith("\n"))
        errors += eof_errors
        eof_checks += checked
        removed |= removed_lines(hunk, start)
        for point in insertion_points(hunk, start):
            anchor = anchor_insertion(lines, point)
            if anchor is not None:
                anchors.add(anchor)
    gold = extend_blank_runs(lines, removed) | anchors
    return {"lines": sorted(gold), "anchors": anchors, "offsets": offsets, "eof_checks": eof_checks,
            "errors": errors, "table": table}


def build_spans(doc_id: str, line_numbers: list[int], table: list[tuple[int, int, str]]) -> list[tuple[list, int, int]]:
    """One [doc_id, start, end] span per run of contiguous lines, whole lines with their newline,
    each with its first and last line number."""
    return [([doc_id, table[first - 1][0], table[last - 1][1]], first, last) for first, last in runs(line_numbers)]


# ---------------------------------------------------------------- one instance


def title_of(issue: str) -> str:
    return next((line.strip() for line in issue.split("\n") if line.strip()), "")


def has_code(issue: str) -> bool:
    return any(marker in issue for marker in CODE_MARKERS)


def prepare_instance(row: dict, oracle: dict, bm25: dict, split: str, diag: Counter) -> dict:
    """{"docs", "query", "offsets"} for one instance, or {"drop": reason, "errors": [...]}."""
    iid, issue = row["instance_id"], row["problem_statement"]
    assert oracle["base_commit"] == bm25["base_commit"] == row["base_commit"], iid
    diag["oracle_prompt_issue_differs_from_lite"] += oracle["problem_statement"] != issue
    diag["bm25_prompt_issue_differs_from_lite"] += bm25["problem_statement"] != issue
    oracle_files = file_texts(oracle["text"], oracle["problem_statement"])
    bm25_files = file_texts(bm25["text"], bm25["problem_statement"])

    patch = parse_patch(row["patch"])
    edited = [f["old"] for f in patch if f["old"] is not None]
    if not edited:
        return {"drop": "only_new_files", "errors": []}
    readmes = set(oracle_files) - set(edited)
    assert all(Path(p).name.lower().startswith(README_PREFIX) for p in readmes), (iid, readmes)
    for p in readmes & set(bm25_files):
        diag["readme_blocks_excluded"] += 1
        diag["readme_content_differs"] += bm25_files[p] != oracle_files[p]
    hits = [p for p in bm25_files if p not in readmes]

    contents = {p: bm25_files[p] for p in hits}
    golds, offsets = {}, []
    for f in patch:
        p = f["old"]
        if p is None:
            diag["new_file_edits_ignored"] += 1
            continue
        assert p in oracle_files or p in bm25_files, f"{iid}: gold file {p} is in neither prompt"
        diag["gold_file_oracle_bm25_differ"] += p in oracle_files and p in bm25_files and oracle_files[p] != bm25_files[p]
        text = oracle_files.get(p, bm25_files.get(p))
        gold = gold_for_file(text, f["hunks"])
        if gold["errors"]:
            diag["files_unalignable"] += 1
            diag["hunks_matching_nowhere"] += sum("nowhere" in e for e in gold["errors"])
            return {"drop": "hunks_do_not_align", "errors": [(iid, p, gold["errors"][:3])]}
        diag["eof_newline_checks"] += gold["eof_checks"]
        moved = sum(o != 0 for o in gold["offsets"])
        diag["hunks_aligned_by_offset"] += moved
        if moved:
            diag["files_with_offset_hunks"] += 1
            offsets.append((iid, p, gold["offsets"]))
        diag["hunks_validated"] += len(f["hunks"])
        if not gold["lines"]:
            diag["gold_file_without_lines"] += 1
            continue
        contents[p] = text
        golds[p] = gold
    if not golds:
        return {"drop": "no_gold_lines", "errors": []}

    spans, n_anchored = [], 0
    for p, gold in golds.items():
        for span, first, last in build_spans(f"{iid}:{p}", gold["lines"], gold["table"]):
            spans.append(span)
            n_anchored += any(first <= n <= last for n in gold["anchors"])
    scope = sorted(contents)
    docs = [{"doc_id": f"{iid}:{p}", "title": p, "text": contents[p]} for p in scope]
    query = {
        "qid": iid,
        "split": split,
        "variants": {"issue": issue, "title": title_of(issue)},
        "gold": [spans],
        "scope_docs": [d["doc_id"] for d in docs],
        "features": {
            "repo": row["repo"],
            "instance_id": iid,
            "n_files_edited": len(patch),
            "n_hunks": sum(len(f["hunks"]) for f in patch),
            "n_gold_spans": len(spans),
            "gold_chars": sum(e - s for _, s, e in spans),
            "n_blank_gold_spans": sum(not contents[d.split(":", 1)[1]][s:e].strip() for d, s, e in spans),
            "n_insertion_anchored": n_anchored,
            "n_candidate_files": len(scope),
            "candidate_chars": sum(len(contents[p]) for p in scope),
            "gold_files_in_bm25": sum(p in hits for p in golds),
            "issue_chars": len(issue),
            "issue_has_code": has_code(issue),
            "created_at": row.get("created_at"),
        },
    }
    return {"docs": docs, "query": query, "offsets": offsets}


# ---------------------------------------------------------------- the dataset


def prepare(raw: Path) -> tuple[list[dict], list[dict], dict]:
    docs, queries, snapshots = [], [], {}
    dropped, dropped_ids, diag = Counter(), defaultdict(list), Counter()
    mismatches, offsets = [], []
    for split in SPLITS:
        oracle = {r["instance_id"]: r for r in load(raw, "oracle", split)}
        bm25 = {r["instance_id"]: r for r in load(raw, "bm25", split)}
        for row in load(raw, "lite", split):
            iid = row["instance_id"]
            result = prepare_instance(row, oracle[iid], bm25[iid], split, diag)
            if "drop" in result:
                dropped[result["drop"]] += 1
                dropped_ids[result["drop"]].append(iid)
                mismatches += result["errors"]
                continue
            docs += result["docs"]
            queries.append(result["query"])
            offsets += result["offsets"]
            snapshots[iid] = (row["repo"], row["base_commit"])
    stats = summarize(docs, queries, snapshots)
    stats |= {
        "dropped": dict(dropped),
        "dropped_ids": dict(dropped_ids),
        "diagnostics": {k: v for k, v in diag.items() if v},
        "hunk_mismatch_examples": mismatches[:10],
        "hunk_offset_examples": offsets,
        "notes": NOTES,
    }
    return docs, queries, stats


def summarize(docs: list[dict], queries: list[dict], snapshots: dict[str, tuple[str, str]]) -> dict:
    lengths = [len(d["text"]) for d in docs]
    n_spans = [len(q["gold"][0]) for q in queries]
    n_candidates = [q["features"]["n_candidate_files"] for q in queries]
    gold_files = sum(len({s[0] for s in q["gold"][0]}) for q in queries)
    gold_in_bm25 = sum(q["features"]["gold_files_in_bm25"] for q in queries)
    by_content, by_snapshot_path = {}, defaultdict(set)
    for d in docs:
        iid, path = d["doc_id"].split(":", 1)
        digest = hashlib.sha1(d["text"].encode("utf-8")).hexdigest()
        by_content[(path, digest)] = len(d["text"])
        by_snapshot_path[(*snapshots[iid], path)].add((iid, digest))
    snapshot_counts = Counter(snapshots.values())
    return {
        "sources": SOURCES,
        "queries_per_split": dict(Counter(q["split"] for q in queries)),
        "queries_total": len(queries),
        "documents": len(docs),
        "doc_chars_total": sum(lengths),
        "doc_chars_median": statistics.median(lengths),
        "gold_spans_per_query": {"mean": statistics.mean(n_spans), "max": max(n_spans)},
        "variants": {
            "issue": "problem_statement exactly (SWE-bench Lite)",
            "title": "first non-empty line of problem_statement, whitespace-stripped",
        },
        "candidates_per_query": {
            "mean": statistics.mean(n_candidates),
            "median": statistics.median(n_candidates),
            "min": min(n_candidates),
            "max": max(n_candidates),
        },
        "candidate_chars_total": sum(q["features"]["candidate_chars"] for q in queries),
        "gold_files": gold_files,
        "gold_files_in_bm25": gold_in_bm25,
        "gold_files_in_bm25_share": gold_in_bm25 / gold_files,
        "unique_path_content_pairs": len(by_content),
        "unique_path_content_chars": sum(by_content.values()),
        "docs_without_trailing_newline": sum(not d["text"].endswith("\n") for d in docs),
        "blank_line_gold_spans": sum(q["features"]["n_blank_gold_spans"] for q in queries),
        "queries_with_only_blank_gold_spans": sum(
            q["features"]["n_blank_gold_spans"] == len(q["gold"][0]) for q in queries
        ),
        "insertion_anchored_gold_spans": sum(q["features"]["n_insertion_anchored"] for q in queries),
        "queries_issue_has_code": sum(q["features"]["issue_has_code"] for q in queries),
        "snapshots_shared_by_multiple_instances": sum(n > 1 for n in snapshot_counts.values()),
        "duplicate_docs_same_repo_commit_path_same_content": sum(
            len(v) - 1 for v in by_snapshot_path.values() if len(v) > 1 and len({h for _, h in v}) == 1
        ),
        "same_repo_commit_path_different_content": sum(
            len({h for _, h in v}) > 1 for v in by_snapshot_path.values()
        ),
    }


NOTES = [
    "Candidates = files edited by the gold patch + BM25-27K prompt files; README blocks that the "
    "SWE-bench prompt template always includes are not BM25 hits and are excluded.",
    "File text reconstructed from line-numbered blocks; no doc contains \\r, so line endings "
    "needed no normalization.",
    "Gold = original lines removed by each hunk ('-' lines); a run of such lines that is entirely "
    "blank is extended up to the nearest non-blank line (down if none above). For hunks with only "
    "'+' lines, the nearest non-blank original line at or before each insertion point (else the "
    "nearest after; features.n_insertion_anchored). Contiguous lines merged into one span; one "
    "alternative holds all spans.",
    "issue_has_code: problem_statement contains ``` or 'Traceback (most recent call last)'.",
    "Hunks that do not match at their stated line are aligned like git apply: exact match of the "
    "hunk's old side nearest the stated line.",
    "20 docs (the 8 offset-hunk gold files and 12 random docs) were checked byte-exact against "
    "raw.githubusercontent.com at base_commit.",
]


if __name__ == "__main__":
    from rbench.paths import RAW

    _docs, _queries, _stats = prepare(RAW / "swebench_lite")
    print(
        f"swebench_lite: {_stats['queries_per_split']} queries, {len(_docs)} docs, "
        f"{_stats['candidate_chars_total']:,} candidate chars, "
        f"{_stats['gold_files_in_bm25_share']:.1%} gold files in BM25, dropped {_stats['dropped'] or 'none'}"
    )
