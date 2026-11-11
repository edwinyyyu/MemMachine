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
benchmark's; the comparison between setups is what this measures. With --embedders, the report
also gives MRR by question type and the share of Assistant turns in each top 5 beside the gold
segments' share (Qwen3-Embedding-8B with the card's instruction ranked Assistant replies above
the user's own statements, 2026-11-11).

  python -m rbench compare-embedders [--server gemma300m] [--questions 100]
  python -m rbench compare-embedders --embedders a,b,c [--questions 200]
"""

from __future__ import annotations

import random

import numpy as np

from rbench.dbs import replace_atomically
from rbench import embedding as emb
from rbench import partitions as pt
from rbench.datasets.longmemeval import history
from rbench.paths import FIGURES, SUMMARY, VECTORS
from rbench.report.stats import delta
from rbench.samples import sample_order
from rbench.schema import queries

EXTRA = 25  # other sessions per haystack, besides the evidence sessions
KS = (5, 10, 20, 50)
LABEL = {"": "both prompts (model card)", "-q": "query prompt only (MemMachine sentence-transformers)",
         "-raw": "no prompts (MemMachine OpenAI-compatible)", "-d": "document prompt only"}


def haystack(dataset: str, q: dict, size: int) -> tuple[list[str], list[list[int]]]:
    """(embedding texts in EventMemory's order, segment ids of each gold turn) for one question's
    reduced haystack: its evidence sessions plus EXTRA others, as MemMachine segments and derives
    the whole history (partitions.offline)."""
    texts, gold, _ = haystack_producers(dataset, q, size)
    return texts, gold


def haystack_producers(dataset: str, q: dict, size: int) -> tuple[list[str], list[list[int]], list[str]]:
    """haystack(), plus each segment's producer (histories.PRODUCERS: User or Assistant), read
    from the turn it came from."""
    from rbench.histories import PRODUCERS

    sessions = history(dataset, q["qid"])["sessions"]
    evidence = sorted({s for s, _ in q["gold_turns"][0]} | set(q["gold_sessions"]))
    others = [s for s in range(len(sessions)) if s not in evidence]
    keep = set(evidence) | set(random.Random(f"haystack:{q['qid']}").sample(others, min(EXTRA, len(others))))
    p = pt.part(dataset, q)
    seg, _, texts = pt.offline(dataset, p.key, f"seg{size}")
    session = [int(p.labels[e].split(",")[0].removeprefix("session ")) for e in seg.event]
    picked = [i for i in range(len(seg)) if session[i] in keep]
    position = {i: k for k, i in enumerate(picked)}
    turns = pt.gold(pt.gold_spans(dataset, q, p), p, seg).alternatives[0]
    gold = [[position[i] for i, _ in span] for span in turns]
    producers = []
    for i in picked:
        s, t = (int(x.split()[-1]) for x in p.labels[seg.event[i]].split(","))  # "session s, turn t"
        producers.append(PRODUCERS[sessions[s]["turns"][t]["role"]])
    return [texts[i] for i in picked], gold, producers


def _vectors(embedder: str, dataset: str, qid: str, texts: list[str], size: int) -> np.ndarray:
    """Haystack vectors, cached per document setup (resumable across interruptions)."""
    spec = emb.EMBEDDERS[embedder]
    path = VECTORS / "compare_prompts" / spec.documents / dataset / f"seg{size}" / f"{qid}.npy"
    if path.exists():
        try:
            vectors = np.load(path)
        except (OSError, ValueError):  # unreadable (an interrupted write before writes were atomic): embedded again
            vectors = None
        if vectors is not None and len(vectors) == len(texts):
            return vectors
    vectors = emb.embed(spec, texts, spec.doc_prompt).astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_atomically(path, lambda temporary: np.save(temporary, vectors))
    return vectors


def measure(server: str, dataset: str, size: int, picked: list[int]) -> dict[str, dict[str, np.ndarray]]:
    """Per prompt setup of one server (keyed by its suffix): every@k, first@k and MRR per question."""
    results = measure_embedders([f"{server}{suffix}" for suffix in emb.SETUPS], dataset, size, picked)
    return {suffix: results[f"{server}{suffix}"] for suffix in emb.SETUPS}


def write(server: str = "gemma300m", dataset: str = "longmemeval_m", size: int = 500, questions: int = 100) -> str:
    picked = sorted(sample_order(dataset)[:questions])  # the samples' order: the first 100 are the pilot's questions
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
    from rbench.report.figures import _figure, _k_x, _key, _plot, _save, _share_y  # the report's palette and style

    colors = {"": "#2a78d6", "-q": "#1baf7a", "-raw": "#eda100", "-d": "#8f8e89"}
    markers = {"": "x", "-q": "1", "-raw": "2", "-d": "+"}
    fig, axes = _figure(1, 2, panel=(4.6, 3.4), sharey=True)
    for j, event in enumerate(("every", "first")):
        ax = axes[0][j]
        for suffix, m in results.items():
            _plot(ax, KS, [m[f"{event}@{k}"].mean() for k in KS], color=colors[suffix], marker=markers[suffix])
        _k_x(ax, max(KS))
        _share_y(ax)
        ax.set_title(f"{event}-gold within the top k segments", loc="left")
        ax.set_xlabel("k segments; dots: measured points")
        if j == 0:
            ax.set_ylabel("share of questions")
    out = FIGURES / server
    out.mkdir(parents=True, exist_ok=True)
    _save(fig, out / f"embedders__{dataset}__seg{size}__{n}q.png", f"First stage by embedding prompt setup — {dataset}, {n} questions",
          [_key(colors[k], marker=markers[k]) for k in emb.SETUPS], [LABEL[k] for k in emb.SETUPS], ncol=2)


def measure_embedders(embedders: list[str], dataset: str, size: int, picked: list[int]) -> dict[str, dict[str, np.ndarray]]:
    """measure() for any embedders (embedding.EMBEDDERS names), not only one server's setups.
    Besides the metrics: "assistant@5", the share of the top 5 segments that are Assistant turns,
    and "gold assistant", that share among the gold segments (the same for every embedder)."""
    rows = queries(dataset)
    out: dict[str, dict[str, list[float]]] = {e: {} for e in embedders}
    for n, qi in enumerate(picked):
        q = rows[qi]
        texts, gold, producers = haystack_producers(dataset, q, size)
        assistant = np.asarray([p == "Assistant" for p in producers])
        gold_segments = sorted({i for turn in gold for i in turn})
        for e in embedders:
            sims = _vectors(e, dataset, q["qid"], texts, size).astype(np.float32) @ \
                emb.query_vectors(e, dataset, "question")[qi].astype(np.float32)
            rank = np.empty(len(sims), dtype=int)
            rank[np.argsort(-sims, kind="stable")] = np.arange(1, len(sims) + 1)
            complete = [max(rank[i] for i in turn) for turn in gold]
            m = out[e]
            for k in KS:
                m.setdefault(f"every@{k}", []).append(float(max(complete) <= k))
                m.setdefault(f"first@{k}", []).append(float(min(complete) <= k))
            m.setdefault("MRR", []).append(1.0 / min(rank[i] for turn in gold for i in turn))
            m.setdefault("assistant@5", []).append(float(assistant[np.argsort(-sims, kind="stable")[:5]].mean()))
            m.setdefault("gold assistant", []).append(float(assistant[gold_segments].mean()))
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(picked)} questions", flush=True)
    return {e: {k: np.asarray(v) for k, v in m.items()} for e, m in out.items()}


def by_type(rows: list[dict], picked: list[int]) -> dict[str, np.ndarray]:
    """Each LongMemEval question type: a mask over the picked questions, types in name order."""
    types = np.asarray([rows[i]["features"]["question_type"] for i in picked])
    return {t: types == t for t in sorted(set(types))}


def summary_name(dataset: str, size: int, n: int, embedders: list[str]) -> str:
    """The report's file name: the embedders joined, or (a long list, past file-name limits)
    their count and a hash of the list."""
    import hashlib

    joined = "_".join(embedders)
    if len(joined) > 120:
        joined = f"{len(embedders)}setups_{hashlib.sha1(','.join(embedders).encode()).hexdigest()[:8]}"
    return f"embedders__{dataset}__seg{size}__{n}q__{joined}"


def write_embedders(embedders: list[str], dataset: str = "longmemeval_m", size: int = 500, questions: int = 100) -> str:
    """Compare first stages on the reduced haystacks (the first `questions` of the samples'
    order): every@k, first@k and MRR per embedder, and each minus the first one listed, with
    95% paired bootstrap intervals over questions. The user's rule picks among them:
    whatever performs best (Qwen3-Embedding-8B with or without its query instruction)."""
    if unknown := [e for e in embedders if e not in emb.EMBEDDERS]:
        raise ValueError(f"unknown embedders {unknown}; known: {sorted(emb.EMBEDDERS)}")
    rows = queries(dataset)
    picked = sorted(sample_order(dataset)[:questions])  # the samples' order: the first 100 are the pilot's questions
    results = measure_embedders(embedders, dataset, size, picked)
    clusters = np.asarray([rows[i]["qid"] for i in picked])
    metrics = [*(f"every@{k}" for k in KS), *(f"first@{k}" for k in KS), "MRR"]
    ref = embedders[0]
    lines = [f"# First stages: {dataset} · seg{size} · {len(picked)} questions\n",
             f"Each question searches its evidence sessions plus {EXTRA} other sessions of its own history, embedded as "
             f"EventMemory embeds them; absolute values are higher than over the whole history. Δ: minus {ref}, 95% "
             "paired bootstrap over questions.\n",
             "| embedder | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    lines += [f"| {e} | " + " | ".join(f"{m[x].mean():.3f}" for x in metrics) + " |" for e, m in results.items()]
    lines += ["", f"| Δ vs {ref} | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    lines += [f"| {e} | " + " | ".join(delta(m[x] - results[ref][x], clusters) for x in metrics) + " |"
              for e, m in results.items() if e != ref]
    types = by_type(rows, picked)
    lines += ["", "**MRR by question type** (n: questions of the type).\n",
              "| embedder | " + " | ".join(f"{t} (n={mask.sum()})" for t, mask in types.items()) + " |",
              "|---|" + "---|" * len(types)]
    lines += [f"| {e} | " + " | ".join(f"{m['MRR'][mask].mean():.3f}" for mask in types.values()) + " |"
              for e, m in results.items()]
    gold = results[ref]["gold assistant"]
    lines += ["", "**Assistant turns in the top 5 segments** (share, averaged over questions), beside their share of the "
              "gold segments: a first stage that ranks Assistant replies above the user's own statements shows a top-5 "
              "share far above the gold's.\n",
              "| | all | " + " | ".join(types) + " |", "|---|---|" + "---|" * len(types),
              "| gold segments | " + " | ".join(f"{gold[mask].mean():.2f}" for mask in [np.ones_like(gold, bool), *types.values()])
              + " |"]
    lines += [f"| {e} | " + " | ".join(f"{m['assistant@5'][mask].mean():.2f}"
                                        for mask in [np.ones_like(gold, bool), *types.values()]) + " |"
              for e, m in results.items()]
    name = summary_name(dataset, size, len(picked), embedders)
    path = SUMMARY / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    _figure_embedders(results, dataset, len(picked), name)
    return str(path)


def _figure_embedders(results: dict[str, dict[str, np.ndarray]], dataset: str, n: int, name: str) -> None:
    """every@k and first@k against k, a line per embedder (the report's palette and markers)."""
    from rbench.report.figures import _figure, _k_x, _key, _plot, _save, _share_y

    palette = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")  # categorical slots, in order
    shapes = ("x", "2", "1", "3", "4", "+")
    # up to six: a color and a marker each, as before; past six the markers shift by one per round of colors,
    # so every (color, marker) pair stays distinct
    style = {e: (palette[i % len(palette)], shapes[(i + i // len(palette)) % len(shapes)]) for i, e in enumerate(results)}
    fig, axes = _figure(1, 2, panel=(4.6, 3.4), sharey=True)
    for j, event in enumerate(("every", "first")):
        ax = axes[0][j]
        for e, m in results.items():
            _plot(ax, KS, [m[f"{event}@{k}"].mean() for k in KS], color=style[e][0], marker=style[e][1])
        _k_x(ax, max(KS))
        _share_y(ax)
        ax.set_title(f"{event}-gold within the top k segments", loc="left")
        ax.set_xlabel("k segments; markers: measured points")
        if j == 0:
            ax.set_ylabel("share of questions")
    out = FIGURES / "embedders"
    out.mkdir(parents=True, exist_ok=True)
    _save(fig, out / f"{name}.png", f"First stage by embedder — {dataset}, {n} questions, reduced haystacks",
          [_key(c, marker=mk) for c, mk in style.values()], list(results), ncol=3,
          note="Held equal: the questions and their haystacks (evidence sessions plus other sessions of the same "
               "history), embedded as EventMemory embeds them. No reranker.")


WIDTHS = (4096, 2048, 1024)


def measure_widths(embedder: str, dataset: str, size: int, picked: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """The same metrics at each width in WIDTHS, from one full-width embedding of each reduced
    haystack: the documents and questions embedded at 4,096 dimensions through MemMachine's own
    path (memory.QueryCachingEmbedder, so the vectors are cached under the full-width setup
    exactly as an ingest would store them), each width the leading dimensions re-normalized (a
    Matryoshka model's prefix is itself an embedding; embed-checks check 3 shows the API's own
    shorter vectors are exactly this)."""
    import asyncio

    from rbench.memory import QueryCachingEmbedder

    full = f"{embedder}-{emb.FULL}"
    rows = queries(dataset)
    out: dict[int, dict[str, list[float]]] = {w: {} for w in WIDTHS}

    async def vectors(texts: list[str], question: str) -> tuple[np.ndarray, np.ndarray]:
        e = QueryCachingEmbedder(full)
        try:
            docs = np.asarray(await e._ingest_embed(texts), dtype=np.float32)
            query = np.asarray((await e._search_embed([question]))[0], dtype=np.float32)
            return docs, query
        finally:
            await e.aclose()

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    for n, qi in enumerate(picked):
        q = rows[qi]
        texts, gold = haystack(dataset, q, size)
        docs, query = asyncio.run(vectors(texts, q["variants"]["question"]))
        for w in WIDTHS:
            sims = unit(docs[:, :w]) @ unit(query[:w])
            rank = np.empty(len(sims), dtype=int)
            rank[np.argsort(-sims, kind="stable")] = np.arange(1, len(sims) + 1)
            complete = [max(rank[i] for i in turn) for turn in gold]
            m = out[w]
            for k in KS:
                m.setdefault(f"every@{k}", []).append(float(max(complete) <= k))
                m.setdefault(f"first@{k}", []).append(float(min(complete) <= k))
            m.setdefault("MRR", []).append(1.0 / min(rank[i] for turn in gold for i in turn))
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(picked)} questions", flush=True)
    return {w: {k: np.asarray(v) for k, v in m.items()} for w, m in out.items()}


def write_widths(embedder: str, dataset: str = "longmemeval_m", size: int = 500, questions: int = 100) -> tuple[str, bool]:
    """Is 1,024 dimensions enough? Metrics at 4,096 / 2,048 / 1,024, Δ vs 4,096 with 95% paired
    intervals, and the decision rule (the user's, 2026-11-10): keep 1,024 unless it is clearly
    worse than 4,096, i.e. the interval of every@k averaged over k or of MRR excludes 0 and the
    loss is at least LOSS. Returns (report path, keep 1,024)."""
    from rbench.report.stats import interval

    rows = queries(dataset)
    picked = sorted(sample_order(dataset)[:questions])
    results = measure_widths(embedder, dataset, size, picked)
    clusters = np.asarray([rows[i]["qid"] for i in picked])
    metrics = [*(f"every@{k}" for k in KS), *(f"first@{k}" for k in KS), "MRR"]
    lines = [f"# Embedding width: {embedder} · {dataset} · seg{size} · {len(picked)} questions\n",
             "- One full-width embedding (4,096 dimensions) per text; each width keeps the leading dimensions, "
             "re-normalized.",
             f"- Reduced haystacks: each question's evidence sessions plus {EXTRA} others. Δ: minus 4,096, with its 95% "
             "paired interval over questions.\n",
             "| width | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    lines += [f"| {w:,} | " + " | ".join(f"{m[x].mean():.3f}" for x in metrics) + " |" for w, m in results.items()]
    lines += ["", "| Δ vs 4,096 | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    lines += [f"| {w:,} | " + " | ".join(delta(m[x] - results[4096][x], clusters) for x in metrics) + " |"
              for w, m in results.items() if w != 4096]
    keep = True
    verdicts = []
    for label, f in (("every@k averaged", lambda m: np.mean([m[f"every@{k}"] for k in KS], axis=0)),
                     ("MRR", lambda m: m["MRR"])):
        d = f(results[1024]) - f(results[4096])
        lo, hi = interval(d, clusters)
        worse = bool(np.isfinite(hi) and hi < 0 and -d.mean() >= LOSS)
        keep = keep and not worse
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else "(no interval: too few questions)"
        verdicts.append(f"{label}: {d.mean():+.3f} {ci}" + (" clearly worse" if worse else ""))
    lines += ["", "**Decision** (keep 1,024 unless clearly worse: the interval excludes 0 and the loss is at least "
              f"{LOSS}): " + "; ".join(verdicts) + f". **{'Keep 1,024' if keep else 'Stop: 1,024 is clearly worse'}.**\n"]
    path = SUMMARY / f"widths__{dataset}__seg{size}__{len(picked)}q__{embedder}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path), keep


LOSS = 0.01  # the loss at 1,024 that counts as clearly worse, with an interval excluding 0
