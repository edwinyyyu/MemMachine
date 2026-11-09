"""Measured profiles of the prepared datasets, written to DATASETS.md with each dataset's
justification from the registry (rbench/datasets/__init__.py).

What is measured and why:
  documents     count, embeddinggemma tokens (p10 / median / p90 / max); paragraph length
                (blank-line blocks) and line structure (heading, list and table lines),
                because the recursive chunker cuts on exactly those boundaries
  queries       tokens (p10 / median / p90) and the rule-based form and context labels
                (features.py): what the reranker must read per pair
  gold          spans per query, span tokens, and how often the longest span exceeds 64 /
                256 / 1,024 tokens (where small chunks must be read together)
  overlap       share of the query's content words that appear in the gold text: low overlap
                is where lexical cues fail and semantic matching must do the work
  candidates    tokens a query retrieves from (corpus, own document, or scope documents)
"""

from __future__ import annotations

import collections
import re

import numpy as np

from rbench.datasets import DATASETS, info
from rbench.features import context, form
from rbench.paths import ROOT
from rbench.schema import corpus, queries
from rbench.text import count

STOPWORDS = set("""a an the of to in on for and or is are was were be been do does did what which who whom whose when
where why how i my me we you your it its this that these those with by from at as can could should would will shall may
might must not no if then than there their they them he she his her our us any all some about into over under after
before so such only also just more most other""".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def _q(x, p) -> float:
    return float(np.percentile(np.asarray(x), p))


def profile(name: str) -> dict:
    ds, docs = info(name), corpus(name)
    doc_tokens = count(list(docs.texts))
    lines = [line for t in docs.texts for line in t.splitlines() if line.strip()]
    paragraphs = [p for t in docs.texts for p in re.split(r"\n\s*\n", t) if p.strip()]
    picked = np.random.default_rng(0).choice(len(paragraphs), min(3000, len(paragraphs)), replace=False)
    para_tokens = count([paragraphs[i] for i in picked])
    rows = [q for q in queries(name) if q["split"] == "test"]
    texts = [q["variants"][ds.default_variant] for q in rows]
    q_tokens = count(texts)
    forms = collections.Counter(form(t) for t in texts)
    spans, longest, overlap, candidates = [], [], [], []
    for q, t in zip(rows, texts, strict=True):
        alternative = q["gold"][0]
        gold_text = [docs.texts[docs.index[d]][s:e] for d, s, e in alternative]
        n = count(gold_text)
        spans += list(n)
        longest.append(n.max())
        qw = _words(t)
        overlap.append(len(qw & _words(" ".join(gold_text))) / max(1, len(qw)))
        if ds.scope == "corpus":
            candidates.append(doc_tokens.sum())
        else:
            names = [q["doc_id"]] if ds.scope == "doc" else q["scope_docs"]
            candidates.append(sum(doc_tokens[docs.index[d]] for d in names))
    longest = np.asarray(longest)
    return {
        "docs": len(docs.doc_ids), "doc_tokens": [_q(doc_tokens, p) for p in (10, 50, 90)] + [int(doc_tokens.max())],
        "corpus_tokens_M": float(doc_tokens.sum() / 1e6), "paragraph_tokens_median": _q(para_tokens, 50),
        "heading_lines": float(np.mean([len(x) < 90 and bool(re.match(r"^\s*(#{1,6}\s|[A-Z][A-Za-z ,'/&()-]{2,80}$)", x))
                                        for x in lines])),
        "list_lines": float(np.mean([bool(re.match(r"^\s*([-*•]|\d+[.)])\s", x)) for x in lines])),
        "table_lines": float(np.mean([x.count("|") >= 2 or x.count("\t") >= 2 for x in lines])),
        "test_queries": len(rows), "query_tokens": [_q(q_tokens, p) for p in (10, 50, 90)],
        "forms": {k: v / len(rows) for k, v in forms.most_common()},
        "contextualized": float(np.mean([context(t) == "contextualized" for t in texts])),
        "multi_span": float(np.mean([min(len(a) for a in q["gold"]) >= 2 for q in rows])),
        "span_tokens": [_q(spans, 50), _q(spans, 90)],
        "longest_over": {n: float((longest > n).mean()) for n in (64, 256, 1024)},
        "overlap": float(np.mean(overlap)), "candidate_tokens_median": _q(candidates, 50),
    }


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def profile_histories(name: str, every: int = 10, size: int = 500) -> dict:
    """Measured profile of a history dataset on every `every`-th history (segments at `size`)."""
    from rbench import histories as hs
    from rbench.datasets.longmemeval import history

    rows = queries(name)
    types = collections.Counter(q["features"]["question_type"] for q in rows)
    segments, tokens, sessions = [], [], []
    for q in rows[::every]:
        seg = hs.offline(name, q["qid"], size)[0]
        segments.append(len(seg))
        tokens.append(int(seg.tokens.sum()))
        sessions.append(q["features"]["n_sessions"])
    ev = {"user": [], "assistant": []}
    for q in rows[::every]:
        h = history(name, q["qid"])
        for s_, t in q["gold_turns"][0]:
            turn = h["sessions"][s_]["turns"][t]
            ev[turn["role"]].append(turn["content"])
    ev_tok = {r: count(v) for r, v in ev.items() if v}
    return {"questions": len(rows), "types": dict(types), "histories_measured": len(segments),
            "sessions_median": _q(sessions, 50), "segments_median": _q(segments, 50), "tokens_median": _q(tokens, 50),
            "evidence_per_question": float(np.mean([len(q["gold_turns"][0]) for q in rows])),
            "evidence_tokens": {r: [_q(v, 50), _q(v, 90), int(v.max())] for r, v in ev_tok.items()},
            "evidence_share": {r: len(v) / sum(map(len, ev.values())) for r, v in ev.items()}}


def write() -> str:
    flat = [name for name in DATASETS if not info(name).partitioned]
    profiles = {name: profile(name) for name in flat}
    out = ["# Datasets\n",
           "Generated by `python -m rbench profile` from the prepared data; the prose comes from "
           "`rbench/datasets/__init__.py`. Inclusion rules are in that module's docstring and in METHODOLOGY.md.\n",
           "Active datasets are listed in `rbench/datasets/__init__.py` (DATASETS); the others prepared earlier are kept "
           "in INACTIVE until they are re-added.\n",
           *(["## At a glance (test split, flat datasets)\n",
              "| dataset | docs | doc tokens p10 / median / p90 (max) | paragraph tokens | headings / lists | queries | query "
              "tokens p10 / median / p90 | forms | contextualized | multi-span | gold span tokens median / p90 | longest gold "
              "> 64 / 256 | query-gold word overlap | candidate tokens per query |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] if flat else [])]
    for name, p in profiles.items():
        d, q = p["doc_tokens"], p["query_tokens"]
        forms = ", ".join(f"{k} {_pct(v)}" for k, v in p["forms"].items() if v >= 0.01)
        out.append(f"| {name} | {p['docs']:,} | {d[0]:,.0f} / {d[1]:,.0f} / {d[2]:,.0f} ({d[3]:,}) | {p['paragraph_tokens_median']:.0f} "
                   f"| {_pct(p['heading_lines'])} / {_pct(p['list_lines'])} | {p['test_queries']:,} | {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} "
                   f"| {forms} | {_pct(p['contextualized'])} | {_pct(p['multi_span'])} | {p['span_tokens'][0]:.0f} / {p['span_tokens'][1]:.0f} "
                   f"| {_pct(p['longest_over'][64])} / {_pct(p['longest_over'][256])} | {p['overlap']:.2f} "
                   f"| {p['candidate_tokens_median']:,.0f} |")
    out.append("")
    for name in [n for n in DATASETS if info(n).partitioned and info(n).evaluated]:
        p = profile_histories(name)
        ev = "; ".join(f"{r} {100 * p['evidence_share'][r]:.0f}% of them, median {t[0]:.0f} / p90 {t[1]:.0f} / max {t[2]} tokens"
                       for r, t in p["evidence_tokens"].items())
        out += [f"## {name}: measured\n",
                f"- {p['questions']} answerable questions: " + ", ".join(f"{k} {v}" for k, v in sorted(p["types"].items())) + ".",
                f"- Per history (every 10th of them, {p['histories_measured']} measured): median {p['sessions_median']:.0f} "
                f"sessions, {p['segments_median']:,.0f} EventMemory segments at 500 characters, {p['tokens_median']:,.0f} tokens.",
                f"- Evidence turns: {p['evidence_per_question']:.2f} per question; {ev}.", ""]
    for name, ds in DATASETS.items():
        out += [f"## {name}\n", f"- **Documents:** {ds.documents}", f"- **Queries:** {ds.queries}",
                f"- **Gold:** {ds.gold}", f"- **Scope:** {ds.scope}, because {ds.scope_reason}",
                "- **Variants:** " + "; ".join(f"`{k}` {v}" for k, v in ds.variants.items()),
                f"- **What it adds:** {ds.adds}", f"- **Caveats:** {ds.caveats}",
                f"- **Source:** {ds.source}", f"- **License:** {ds.license}", ""]
    path = ROOT / "DATASETS.md"
    path.write_text("\n".join(out) + "\n")
    return str(path)
