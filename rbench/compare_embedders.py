"""Which embedding prompt setup ranks LongMemEval-M's evidence best? (No reranker involved.)

The setups (embedding.SETUPS) differ only in the prompts prepended to queries and documents:
both of the model card's, the query prompt alone (MemMachine's sentence-transformers
embedder), none (its OpenAI-compatible embedder), or the document prompt alone. The best one
is the fair baseline for measuring what a reranker adds: a weaker first stage would inflate
the reranker's gain.

Telling setups apart does not need each question's whole ~500-session history, which is what
makes the benchmark slow to embed. Each question searches a smaller haystack drawn from its own
history: its evidence sessions plus EXTRA other sessions chosen at random (seeded), segmented,
ordered, dated and rendered for embedding exactly as histories.py does. Setups are compared on
how they rank the gold segments, paired by question:
  every@k   every segment of every gold turn is in the top k segments
  first@k   some gold turn is complete in the top k
  MRR       1 / rank of the first gold segment
The haystack is smaller than the real one, so absolute numbers are higher than the
benchmark's; the comparison between setups is what this measures.

  python -m rbench compare-embedders [--server gemma300m] [--questions 100]
"""

from __future__ import annotations

import hashlib
import random
from datetime import timedelta

import numpy as np

from rbench import embedding as emb
from rbench import eventmemory as em
from rbench import histories as hs
from rbench.datasets.longmemeval import history, session_datetime
from rbench.paths import FIGURES, SUMMARY, VECTORS
from rbench.report.stats import delta
from rbench.samples import sample
from rbench.schema import queries

EXTRA = 25  # other sessions per haystack, besides the evidence sessions
KS = (5, 10, 20, 50)
LABEL = {"": "both prompts (model card)", "-q": "query prompt only (MemMachine sentence-transformers)",
         "-raw": "no prompts (MemMachine OpenAI-compatible)", "-d": "document prompt only"}


def haystack(dataset: str, q: dict, size: int) -> tuple[list[str], list[list[int]]]:
    """(embedding texts in EventMemory's order, segment ids of each gold turn) for one question's
    reduced haystack."""
    sessions = history(dataset, q["qid"])["sessions"]
    evidence = sorted({s for s, _ in q["gold_turns"][0]} | set(q["gold_sessions"]))
    others = [s for s in range(len(sessions)) if s not in evidence]
    keep = set(evidence) | set(random.Random(f"haystack:{q['qid']}").sample(others, min(EXTRA, len(others))))
    gold = {(s, t) for s, t in q["gold_turns"][0]}
    texts, gold_ids, event = [], {}, 0
    for s in [s for s in hs.chronological(sessions) if s in keep]:
        start = session_datetime(sessions[s]["date"])
        for t, turn in enumerate(sessions[s]["turns"]):
            for o, piece in enumerate(em.segment_text(turn["content"], size)):
                segment = em.Segment(len(texts), event, 0, o, start + event * hs.TICK, hs.PRODUCERS[turn["role"]], piece)
                if (s, t) in gold:
                    gold_ids.setdefault((s, t), []).append(len(texts))
                texts.append(em.derivative_text(segment))
            event += 1
    assert event * hs.TICK < timedelta(seconds=1)
    return texts, list(gold_ids.values())


def _vectors(embedder: str, dataset: str, qid: str, texts: list[str], size: int) -> np.ndarray:
    """Haystack vectors, cached per document setup (resumable across interruptions)."""
    spec = emb.EMBEDDERS[embedder]
    path = VECTORS / "compare_prompts" / spec.documents / dataset / f"seg{size}" / f"{qid}.npy"
    if path.exists():
        vectors = np.load(path)
        if len(vectors) == len(texts):
            return vectors
    vectors = emb.embed(spec, texts, spec.doc_prompt).astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def measure(server: str, dataset: str, size: int, picked: list[int]) -> dict[str, dict[str, np.ndarray]]:
    """Per setup: every@k, first@k (per k) and MRR, one value per question."""
    rows = queries(dataset)
    out: dict[str, dict[str, list[float]]] = {s: {} for s in emb.SETUPS}
    for n, qi in enumerate(picked):
        q = rows[qi]
        texts, gold = haystack(dataset, q, size)
        for suffix in emb.SETUPS:
            embedder = f"{server}{suffix}"
            sims = _vectors(embedder, dataset, q["qid"], texts, size).astype(np.float32) @ \
                hs.query_vectors(embedder, dataset)[qi].astype(np.float32)
            rank = np.empty(len(sims), dtype=int)
            rank[np.argsort(-sims, kind="stable")] = np.arange(1, len(sims) + 1)
            complete = [max(rank[i] for i in turn) for turn in gold]  # a turn is read once its last segment is
            m = out[suffix]
            for k in KS:
                m.setdefault(f"every@{k}", []).append(float(max(complete) <= k))
                m.setdefault(f"first@{k}", []).append(float(min(complete) <= k))
            m.setdefault("MRR", []).append(1.0 / min(rank[i] for turn in gold for i in turn))
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(picked)} questions", flush=True)
    return {s: {k: np.asarray(v) for k, v in m.items()} for s, m in out.items()}


def write(server: str = "gemma300m", dataset: str = "longmemeval_m", size: int = 500, questions: int = 100) -> str:
    rows = queries(dataset)
    order = sorted(sample(dataset, "all"), key=lambda i: hashlib.sha1(f"sample:{rows[i]['qid']}".encode()).hexdigest())
    picked = sorted(order[:questions])  # the samples' hash order (samples.py): the first 100 are the pilot's questions
    results = measure(server, dataset, size, picked)
    clusters = np.asarray([queries(dataset)[i]["qid"] for i in picked])
    ref = results[""]
    metrics = [*(f"every@{k}" for k in KS), *(f"first@{k}" for k in KS), "MRR"]
    lines = [f"# Embedding prompt setups: {dataset} · seg{size} · {len(picked)} questions · {server}\n",
             f"First stage only, no reranker. Each question searches its evidence sessions plus {EXTRA} other sessions "
             "of its own history (EventMemory segments, order, dates and embedding text), so absolute values are higher "
             "than over the whole history; the comparison between setups is the point. every@k: every gold turn "
             "complete within the top k segments; first@k: some gold turn; MRR: of the first gold segment. Δ: minus "
             "the model card's setup, 95% paired bootstrap over questions.\n",
             "| setup | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for suffix, m in results.items():
        lines.append(f"| {LABEL[suffix]} | " + " | ".join(f"{m[x].mean():.3f}" for x in metrics) + " |")
    lines += ["", "| Δ vs model card | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for suffix, m in results.items():
        if suffix:
            lines.append(f"| {LABEL[suffix]} | " + " | ".join(delta(m[x] - ref[x], clusters) for x in metrics) + " |")
    score = {s: np.mean([m[f"every@{k}"] for k in KS], axis=0) for s, m in results.items()}
    lines += ["", f"Summary (every@k averaged over k = {', '.join(map(str, KS))}): "
              + "; ".join(f"{LABEL[s]} {v.mean():.3f}" for s, v in score.items()) + ".\n",
              "**Each prompt's own effect** (the four setups are a 2 × 2 design, so each prompt's effect is the mean of "
              "the setups with it minus the mean of those without it, paired by question):\n",
              "| measure | query prompt | document prompt | document prompt only − no prompts |", "|---|---|---|---|"]
    for label, f in (("every@k averaged", lambda x: score[x]), ("every@5", lambda x: results[x]["every@5"]),
                     ("MRR", lambda x: results[x]["MRR"])):
        both, q, raw, d = f(""), f("-q"), f("-raw"), f("-d")
        lines.append(f"| {label} | {delta((both + q) / 2 - (raw + d) / 2, clusters)} | {delta((both + d) / 2 - (q + raw) / 2, clusters)} "
                     f"| {delta(d - raw, clusters)} |")
    lines.append("\nA prompt whose intervals exclude 0 matters; one whose intervals include 0 cannot be told apart from "
                 "none on this sample. Choosing among setups on the evaluated questions can only flatter the first stage, "
                 "which makes a reranker's measured gain conservative.\n")
    path = SUMMARY / f"embedders__{dataset}__seg{size}__{len(picked)}q__{server}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    _figure(results, dataset, size, len(picked), server)
    return str(path)


def _figure(results: dict[str, dict[str, np.ndarray]], dataset: str, size: int, n: int, server: str) -> None:
    from rbench.report.figures import _log_x, _save, plt  # the report's palette and style

    colors = {"": "#2a78d6", "-q": "#1baf7a", "-raw": "#eda100", "-d": "#8f8e89"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), squeeze=False, sharey=True)
    handles = {}
    for j, event in enumerate(("every", "first")):
        ax = axes[0][j]
        for suffix, m in results.items():
            (handles[suffix],) = ax.plot(KS, [m[f"{event}@{k}"].mean() for k in KS], color=colors[suffix], marker="o",
                                         markersize=4)
        _log_x(ax, 2, list(KS))
        ax.set_title(f"{event}-gold within the top k segments", loc="left")
        ax.set_xlabel("k segments")
        if j == 0:
            ax.set_ylabel("share of questions")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    out = FIGURES / server
    out.mkdir(parents=True, exist_ok=True)
    _save(fig, out / f"embedders__{dataset}__seg{size}__{n}q.png", f"First stage by embedding prompt setup — {dataset}, {n} questions",
          [handles[k] for k in emb.SETUPS], [LABEL[k] for k in emb.SETUPS], ncol=2)
