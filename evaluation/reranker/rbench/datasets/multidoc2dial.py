"""MultiDoc2Dial: goal-oriented support dialogues grounded in US government service pages.

Source: the May 2022 release zip (multidoc2dial_doc.json plus dialogues for train, validation and
test). Crowdworkers wrote the dialogues; each turn names the document spans (doc_id, id_sp) it uses,
and each document's `spans` dict maps id_sp to start_sp/end_sp character offsets into doc_text.

Documents: all 488 across the four domains (ssa, va, dmv, studentaid). doc_id and title are the
dataset's own; text is doc_text unchanged (it has no CRLF, and every span's text_sp equals
doc_text[start_sp:end_sp]).

Grounding convention. In this release every turn carries references, user turns included. A query is
a user turn immediately followed by an agent turn. Its gold is the spans that agent turn references:
one alternative, with exact duplicates merged and adjacent spans kept separate. The user turn's own
references mark the part of the document the question is about, not the grounding of the reply. The
agent utterance shares 0.66 of its tokens with the agent-turn spans, against 0.33 with the user-turn
spans and 0.07 with a random span (validation and test agree). The official Hugging Face builder does
the same: it predicts the next agent turn and skips respond_no_solution. Every agent turn references
exactly one document, so no query has multi-document gold.

Splits: validation becomes dev (4,201 queries, 661 dialogues) and test stays test (4,094, 661). The
README calls test a dummy, but this release's test file has real references, with the same overlap
profile as validation. Train is unused. Dialogue ids are disjoint across splits.

Dropped user turns (572 of 8,867 in dev and test):
  - 467: the agent reply is respond_no_solution ("no relevant information is found"). Its references
    point at section headings (463 of 468 spans are h1-h4), not at grounding text.
  - 71: the user turn is followed by another user turn, at a segment boundary where MultiDoc2Dial
    joined doc2dial dialogues. The later user turn, which the agent answers, is kept.
  - 34: the user turn is the last turn of its dialogue, so no agent reply follows.
  Agent turns without references and empty user utterances would also be dropped; neither occurs.

Variants, built from the dialogue text only, with each utterance's whitespace collapsed so every turn
fits on one line:
  - turn: the user utterance alone (17.8% are under 6 words: context-dependent follow-ups);
  - recent: the last agent turn before it (if any), any user turns since, then the user turn. Only the
    69 queries after back-to-back user turns have more than two lines;
  - history: every turn so far as "User: ..." / "Agent: ..." lines, ending with the user turn.

Duplicate-copy alternatives. Many documents are versions of one page: doc_ids that share the prefix
before '#' (for example ...#1_0 and ...#2_0). Thirteen documents duplicate another's text exactly, and
43 are contained verbatim in another. The gold text of 2,998 queries appears verbatim in some other
document, and for 2,156 of them that document is a copy of the same page. A retriever that finds the
identical passage in a copy should get credit. So every other document that contains all span texts of
the original alternative verbatim adds one alternative, with the spans placed to best preserve the
original order and spacing. A copy is accepted if:
  (a) same page: it shares the gold document's page prefix; or
  (b) long verbatim: every span text has at least 40 characters (after stripping whitespace), so short
      boilerplate that recurs across unrelated pages ("Go to FAQs") does not add spurious alternatives.
The original alternative stays first. Results: 2,513 queries gain alternatives, 2,088 via same-page
copies only and 425 with at least one other-page copy. 4,065 same-page and 858 long-verbatim
alternatives are added, up to 29 per query (the SSA "Learn what documents..." page has 31 versions).
In 4,912 of the 4,923 added alternatives the spans keep the original spacing exactly.

Extra features beyond the turn metadata: agent_da (the dialogue act of the agent reply),
gold_text_dup_docs and gold_text_dup_same_page_docs (other documents containing the whole gold
stretch, text[min start:max end]), n_gold_alternatives, and gold_alt_reason ("none", "same_page", or
"verbatim_long" when any added copy comes from another page).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from rbench.datasets import _download

URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"  # official release, May 2022
ZIP_NAME = "multidoc2dial.zip"  # name of the downloaded archive inside raw/
SOURCE_DIR = "multidoc2dial"  # the zip's top-level folder, holding the JSON files
DOC_FILE = "multidoc2dial_doc.json"  # documents with doc_text and span offsets
DIALOGUE_FILE = "multidoc2dial_dial_{split}.json"  # dialogues with per-turn references
SPLITS = {"validation": "dev", "test": "test"}  # test has real references; train is unused
NO_SOLUTION_DA = "respond_no_solution"  # agent says nothing was found; its references are headings
MIN_LONG_SPAN_CHARS = 40  # shorter span texts recur as boilerplate across unrelated pages
PAGE_SEPARATOR = "#"  # doc_ids of one page's versions share the prefix before it
ROLE_LABEL = {"user": "User: ", "agent": "Agent: "}  # line prefixes in the recent/history variants
SHORT_TURN_WORDS = 6  # turns under this many words count as context-dependent follow-ups


# ---- download


def download(raw: Path) -> None:
    """Fetch and extract the release zip into raw; a no-op if the JSON files are already there."""
    source = raw / SOURCE_DIR
    needed = [DOC_FILE] + [DIALOGUE_FILE.format(split=s) for s in ("train", *SPLITS)]
    if all((source / name).exists() for name in needed):
        return
    _download.extract(_download.fetch(URL, raw / ZIP_NAME), raw)


# ---- prepare


def prepare(raw: Path) -> tuple[list[dict], list[dict], dict]:
    source = raw / SOURCE_DIR
    doc_data = _load_doc_data(source)
    docs = [_doc_row(d) for d in doc_data.values()]
    doc_text = {d["doc_id"]: d["text"] for d in docs}
    queries, drops, user_turns = [], {}, {}
    for source_split, split in SPLITS.items():
        rows, drops[split], user_turns[split] = _split_queries(_load_dialogues(source, source_split), split, doc_data)
        queries += rows
    _check_dialogues_in_one_split(queries)
    for q in queries:
        q["features"].update(_dup_text_features(q["gold"][0], doc_text))
    added = _add_copy_alternatives(queries, doc_text)
    return docs, queries, _stats(docs, queries, drops, user_turns, added)


# ---- documents


def _load_doc_data(source: Path) -> dict[str, dict]:
    """Raw document records by doc_id, in file order (domain, then document)."""
    by_domain = json.loads((source / DOC_FILE).read_text())["doc_data"]
    return {doc_id: d for docs in by_domain.values() for doc_id, d in docs.items()}


def _doc_row(d: dict) -> dict:
    assert "\r" not in d["doc_text"], "CRLF would need normalizing and shift span offsets"
    return {"doc_id": d["doc_id"], "title": d.get("title") or None, "text": d["doc_text"]}


# ---- queries


def _load_dialogues(source: Path, source_split: str) -> list[tuple[str, dict]]:
    """(domain, dialogue) pairs of one source split, in file order."""
    by_domain = json.loads((source / DIALOGUE_FILE.format(split=source_split)).read_text())["dial_data"]
    return [(domain, dialogue) for domain, dialogues in by_domain.items() for dialogue in dialogues]


def _split_queries(dialogues: list[tuple[str, dict]], split: str, doc_data: dict) -> tuple[list[dict], dict, int]:
    """Queries of one split, with drop counts by reason and the number of user turns seen."""
    rows, drops, user_turns = [], Counter(), 0
    for domain, dialogue in dialogues:
        turns = dialogue["turns"]
        assert [t["turn_id"] for t in turns] == list(range(1, len(turns) + 1))
        utterances = [_normalize(t["utterance"]) for t in turns]
        user_positions = [i for i, t in enumerate(turns) if t["role"] == "user"]
        user_turns += len(user_positions)
        for user_index, i in enumerate(user_positions):
            reason = _drop_reason(turns, utterances, i)
            if reason:
                drops[reason] += 1
                continue
            spans = _gold_spans(turns[i + 1]["references"], doc_data)
            rows.append(_query(dialogue["dial_id"], domain, split, turns, utterances, i, user_index, spans))
    return rows, dict(drops), user_turns


def _normalize(utterance: str) -> str:
    """Collapse all whitespace, including internal newlines, so each turn fits on one line."""
    return " ".join(utterance.split())


def _drop_reason(turns: list[dict], utterances: list[str], i: int) -> str | None:
    """Why the user turn at i yields no query, or None if it does."""
    if i + 1 == len(turns):
        return "last_turn_of_dialogue_no_agent_reply"
    reply = turns[i + 1]
    if reply["role"] != "agent":
        return "followed_by_another_user_turn"
    if reply["da"] == NO_SOLUTION_DA:
        return "agent_reply_is_respond_no_solution"
    if not reply["references"]:
        return "agent_reply_has_no_references"
    if not utterances[i]:
        return "empty_user_utterance"
    return None


def _gold_spans(references: list[dict], doc_data: dict) -> list[tuple[str, int, int]]:
    """The referenced spans as sorted (doc_id, start, end), exact duplicates merged, adjacent kept apart."""
    spans = set()
    for ref in references:
        doc = doc_data[ref["doc_id"]]
        span = doc["spans"][ref["id_sp"]]
        assert doc["doc_text"][span["start_sp"]:span["end_sp"]] == span["text_sp"]
        spans.add((ref["doc_id"], span["start_sp"], span["end_sp"]))
    return sorted(spans)


def _variants(turns: list[dict], utterances: list[str], i: int) -> dict[str, str]:
    """turn / recent / history for the user turn at position i."""
    def line(j: int) -> str:
        return ROLE_LABEL[turns[j]["role"]] + utterances[j]

    last_agent = max((j for j in range(i) if turns[j]["role"] == "agent"), default=i)
    return {
        "turn": utterances[i],
        "recent": "\n".join(line(j) for j in range(last_agent, i + 1)),
        "history": "\n".join(line(j) for j in range(i + 1)),
    }


def _query(dial_id: str, domain: str, split: str, turns: list[dict], utterances: list[str], i: int,
           user_index: int, spans: list[tuple[str, int, int]]) -> dict:
    user, reply = turns[i], turns[i + 1]
    return {
        "qid": f"{dial_id}_{user['turn_id']}",
        "split": split,
        "variants": _variants(turns, utterances, i),
        "gold": [[[d, a, b] for d, a, b in spans]],
        "features": {
            "dialogue_id": dial_id,
            "turn_index": user_index,
            "domain": domain,
            "n_gold_spans": len(spans),
            "gold_chars": sum(b - a for _, a, b in spans),
            "n_gold_docs": len({d for d, _, _ in spans}),
            "turn_tokens_approx": len(utterances[i].split()),
            "da": user.get("da"),
            "agent_da": reply.get("da"),
        },
    }


def _check_dialogues_in_one_split(queries: list[dict]) -> None:
    split_of = {}
    for q in queries:
        assert split_of.setdefault(q["features"]["dialogue_id"], q["split"]) == q["split"]


# ---- duplicated content


def _page(doc_id: str) -> str:
    return doc_id.split(PAGE_SEPARATOR)[0]


def _dup_text_features(alternative: list[list], doc_text: dict[str, str]) -> dict[str, int]:
    """How many other docs (and how many copies of the same page) contain the whole gold stretch."""
    gold_doc = alternative[0][0]
    stretch = doc_text[gold_doc][min(a for _, a, _ in alternative):max(b for _, _, b in alternative)]
    others = [k for k, t in doc_text.items() if k != gold_doc and stretch in t]
    return {"gold_text_dup_docs": len(others),
            "gold_text_dup_same_page_docs": sum(_page(k) == _page(gold_doc) for k in others)}


def _add_copy_alternatives(queries: list[dict], doc_text: dict[str, str]) -> Counter:
    """Append each query's accepted copy alternatives after its original; count them by reason."""
    added = Counter()
    for q in queries:
        reasons = []
        for alternative, reason in _copy_alternatives(q["gold"][0], doc_text):
            q["gold"].append(alternative)
            reasons.append(reason)
        added.update(reasons)
        q["features"]["n_gold_alternatives"] = len(q["gold"])
        q["features"]["gold_alt_reason"] = (
            "none" if not reasons else "verbatim_long" if "verbatim_long" in reasons else "same_page")
    return added


def _copy_alternatives(original: list[list], doc_text: dict[str, str]) -> list[tuple[list[list], str]]:
    """(alternative, reason) for every other doc that holds all span texts verbatim and is accepted."""
    gold_doc = original[0][0]
    texts = [doc_text[d][a:b] for d, a, b in original]
    starts = [a for _, a, _ in original]
    long_enough = all(len(t.strip()) >= MIN_LONG_SPAN_CHARS for t in texts)
    out = []
    for doc_id, text in doc_text.items():
        if doc_id == gold_doc or not all(t in text for t in texts):
            continue
        same_page = _page(doc_id) == _page(gold_doc)
        if not (same_page or long_enough):
            continue
        alternative = [[doc_id, a, b] for a, b in _locate_in_copy(text, texts, starts)]
        assert all(text[a:b] == t for (_, a, b), t in zip(alternative, texts))
        out.append((alternative, "same_page" if same_page else "verbatim_long"))
    return out


def _occurrences(text: str, sub: str) -> list[int]:
    found, i = [], text.find(sub)
    while i != -1:
        found.append(i)
        i = text.find(sub, i + 1)
    return found


def _locate_in_copy(copy_text: str, span_texts: list[str], orig_starts: list[int]) -> list[tuple[int, int]]:
    """Place each span text in copy_text, best preserving the original order and spacing.

    Every occurrence of the first span is tried as an anchor. Each later span takes the occurrence at or
    after the previous match's end that is closest to its expected position (anchor plus its original
    offset), or the closest occurrence anywhere if none follows. The anchor with the smallest total
    deviation from the original spacing wins; ties go to the earliest.
    """
    occurrences = [_occurrences(copy_text, t) for t in span_texts]
    best = None
    for anchor in occurrences[0]:
        placed, deviation = [anchor], 0
        for k in range(1, len(span_texts)):
            expected = anchor + orig_starts[k] - orig_starts[0]
            prev_end = placed[-1] + len(span_texts[k - 1])
            candidates = [o for o in occurrences[k] if o >= prev_end] or occurrences[k]
            chosen = min(candidates, key=lambda o: (abs(o - expected), o))
            placed.append(chosen)
            deviation += abs(chosen - expected)
        if best is None or deviation < best[0]:
            best = (deviation, placed)
    return [(o, o + len(t)) for o, t in zip(best[1], span_texts)]


# ---- stats


def _stats(docs: list[dict], queries: list[dict], drops: dict, user_turns: dict, added: Counter) -> dict:
    lengths = [len(d["text"]) for d in docs]
    def feature(name: str) -> list:
        return [q["features"][name] for q in queries]

    split_counts = Counter(q["split"] for q in queries)
    return {
        "source": URL,
        "queries_per_split": dict(split_counts),
        "dialogues_per_split": {s: len({q["features"]["dialogue_id"] for q in queries if q["split"] == s})
                                for s in split_counts},
        "queries_per_domain": dict(Counter(feature("domain"))),
        "n_docs": len(docs),
        "doc_chars_total": sum(lengths),
        "doc_chars_median": statistics.median(lengths),
        "docs_with_duplicate_text": sum(c - 1 for c in Counter(d["text"] for d in docs).values() if c > 1),
        "docs_whose_text_is_contained_in_another_doc": _count_contained(docs),
        "queries_whose_gold_text_appears_verbatim_in_other_docs": {
            "any_doc": sum(n > 0 for n in feature("gold_text_dup_docs")),
            "same_page_version_doc": sum(n > 0 for n in feature("gold_text_dup_same_page_docs")),
            "note": "gold stretch = text[min start:max end] of the original gold spans "
                    "(features gold_text_dup_docs / gold_text_dup_same_page_docs); see "
                    "duplicate_copy_gold_alternatives for the alternatives actually added.",
        },
        "duplicate_copy_gold_alternatives": {
            "rule": "extra alternative per other doc containing ALL span texts of the original alternative "
                    "verbatim, accepted if same page (doc_id prefix before '#') or every span text is >= 40 "
                    "chars (stripped); spans placed to best preserve original order/spacing; original first",
            "queries_by_gold_alt_reason": dict(Counter(feature("gold_alt_reason"))),
            "alternatives_added_by_reason": dict(added),
            "queries_with_extra_alternatives": sum(len(q["gold"]) > 1 for q in queries),
            "max_alternatives": max(len(q["gold"]) for q in queries),
            "gold_alt_reason_values": "none | same_page (all added copies are same-page) | verbatim_long "
                                      "(at least one added copy is from another page)",
        },
        "gold_spans_per_query": {"mean": round(statistics.mean(feature("n_gold_spans")), 3),
                                 "max": max(feature("n_gold_spans")),
                                 "note": "spans in the original (first) alternative"},
        "gold_chars_per_query": {"mean": round(statistics.mean(feature("gold_chars")), 1),
                                 "median": statistics.median(feature("gold_chars"))},
        "queries_with_multi_doc_gold": sum(n > 1 for n in feature("n_gold_docs")),
        "variants": {
            "turn": "current user utterance (whitespace-collapsed)",
            "recent": "most recent agent turn before the current user turn (if any), any user turns after it, "
                      "and the current user turn; 'Agent: ...\\nUser: ...'",
            "history": "all turns so far, one per line as 'User: ...'/'Agent: ...', ending with current user turn",
        },
        "grounding_convention": (
            "Every turn in this release carries references, user turns included. Query = user turn immediately "
            "followed by an agent turn; gold = the spans referenced by that agent turn (one alternative, exact "
            "duplicates merged, adjacent spans kept separate). User-turn references are ignored: they mark the "
            "part of the doc the user's question is about, not the grounding of the reply (token overlap of the "
            "agent utterance with agent-turn refs ~0.66 vs ~0.33 with user-turn refs). This matches the official "
            "HF MultiDoc2Dial builder (turn_to_predict = next agent turn; skips respond_no_solution)."
        ),
        "splits": "validation -> dev, test -> test (test has real references in this 2022 release; train unused)",
        "user_turns_seen_per_split": user_turns,
        "dropped_user_turns": {
            "total": sum(sum(d.values()) for d in drops.values()),
            "by_reason": dict(sum((Counter(d) for d in drops.values()), Counter())),
            "by_split": drops,
            "notes": {
                "agent_reply_is_respond_no_solution": "agent says no relevant info; its references point to "
                                                      "section headings (h1-h4), not grounding text",
                "followed_by_another_user_turn": "MultiDoc2Dial segment boundaries; the later user turn is kept",
            },
        },
        "turn_word_count": _word_count_stats(feature("turn_tokens_approx")),
        "agent_da_counts": dict(Counter(feature("agent_da"))),
        "user_da_counts": dict(Counter(feature("da"))),
    }


def _count_contained(docs: list[dict]) -> int:
    """Documents whose whole text appears inside another, different document text."""
    return sum(any(a["text"] in b["text"] and a["text"] != b["text"] for b in docs) for a in docs)


def _word_count_stats(counts: list[int]) -> dict:
    q = statistics.quantiles(counts, n=20)
    return {
        "mean": round(statistics.mean(counts), 2),
        "min": min(counts), "p5": q[0], "p25": q[4], "median": statistics.median(counts),
        "p75": q[14], "p95": q[18], "max": max(counts),
        "frac_under_6_words": round(sum(c < SHORT_TURN_WORDS for c in counts) / len(counts), 4),
    }


if __name__ == "__main__":
    from rbench.paths import RAW

    raw = RAW / "multidoc2dial"
    download(raw)
    docs, queries, stats = prepare(raw)
    print(f"multidoc2dial: {len(docs)} docs, {stats['doc_chars_total']:,} chars; "
          f"queries {stats['queries_per_split']}; dropped {stats['dropped_user_turns']['by_reason']}; "
          f"alternatives {stats['duplicate_copy_gold_alternatives']['queries_by_gold_alt_reason']}")
