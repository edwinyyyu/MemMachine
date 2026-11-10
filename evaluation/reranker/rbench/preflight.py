"""Checks to pass before any paid run.

smoke     A few hand-made pairs per DeepInfra model (a fraction of a cent). Confirms what the
          API returns before the benchmark relies on it: scores are probabilities in [0, 1];
          a relevant document outscores an irrelevant one; the `instruction` field changes the
          scores (so it is honored); billed input tokens match our own count of the
          model-card format (so cost estimates hold); the reported cost matches the list price.
fidelity  Scores the same pairs through DeepInfra's 0.6B as were scored locally, and compares:
          correlation of logits, per-query rank agreement, top-1 agreement, and the simulated
          coverage each produces. Local prototyping results transfer only if these agree.
"""

from __future__ import annotations

import numpy as np

from rbench import rerankers
from rbench.embedding import DEFAULT as EMBEDDER
from rbench.rerankers import RERANKERS, ScoreCache, pair_ids
from rbench.schema import queries

# Another instruction, only to check that DeepInfra applies the instruction field (smoke).
OTHER_INSTRUCTION = "Given a user's message to an assistant, retrieve passages that help the assistant respond to it"
SMOKE = [
    ("what is the capital of france",
     ["Paris is the capital and most populous city of France.",
      "The mitochondria is the powerhouse of the cell."]),
    ("How do I renew my driver's license online?",
     ["You can renew your license online if it expires within 6 months and your address is unchanged.",
      "Bananas are rich in potassium and are a popular fruit worldwide."]),
]


def smoke(models: list[str]) -> list[str]:
    """Problems found; nothing is written to the score caches."""
    import httpx

    key = (rerankers.SECRETS / "deepinfra_api_key").read_text().strip()
    problems = []

    def call(model: str, query: str, docs: list[str], instruction: str) -> dict:
        body = {"queries": [query] * len(docs), "documents": docs, "instruction": instruction}
        response = httpx.post(rerankers.DEEPINFRA_URL + model, json=body, timeout=120,
                              headers={"Authorization": f"bearer {key}"})
        response.raise_for_status()
        return response.json()

    for m in models:
        spec = RERANKERS[m]
        for query, docs in SMOKE:
            a = call(spec.hf_model, query, docs, spec.template.instruction)
            b = call(spec.hf_model, query, docs, OTHER_INSTRUCTION)
            scores, other = np.asarray(a["scores"]), np.asarray(b["scores"])
            counted = sum(len(r) for r in pair_ids(query, docs, spec.template))
            status = a.get("inference_status") or {}
            billed = int(a.get("input_tokens") or status.get("tokens_input") or 0)
            cost = float(status.get("cost") or 0.0)
            print(f"{m}: scores {scores.tolist()} / other instruction {np.round(other, 4).tolist()}; "
                  f"billed {billed} vs counted {counted}; cost ${cost:.6f} vs list ${counted / 1e6 * spec.usd_per_mtok:.6f}")
            if not np.all((scores >= 0) & (scores <= 1)):
                problems.append(f"{m}: scores outside [0, 1]; to_logit assumes probabilities")
            if np.any((scores <= rerankers.SATURATED) | (scores >= 1 - rerankers.SATURATED)):
                problems.append(f"{m}: scores saturate at 0 or 1 ({scores.tolist()}): saturated candidates tie, and "
                                "ties fall back to first-stage order")
            if not scores[0] > scores[1]:
                problems.append(f"{m}: an irrelevant document outscored a relevant one")
            if np.allclose(scores, other):
                problems.append(f"{m}: the instruction field did not change any score (ignored?)")
            if not billed:
                problems.append(f"{m}: the response reports no billed tokens; spend tracking would fall back to our count")
            elif abs(billed / counted - 1) > 0.05:
                problems.append(f"{m}: billed tokens differ from our count by {100 * (billed / counted - 1):+.0f}%")
    return problems


def fidelity(dataset: str, chunking_name: str, sample_name: str, local: str = "q06-local", api: str = "q06",
             pool_seeds: int = 50, scope: str | None = None, variant: str | None = None) -> dict:
    """Score through `api` the pairs `local` has scored (what EventMemory.query sends its
    reranker for the sample at vector search limit pool_seeds), then compare the two."""
    from rbench import simulation

    asked = simulation.pool_pairs(EMBEDDER, dataset, chunking_name, sample_name, local, 0, pool_seeds, scope, variant)
    a, b = ScoreCache(local), ScoreCache(api)
    per_query, work = [], []
    for query, items in asked.items():
        docs = dict(items)
        have = a.get(list(docs))
        keys = [k for k in docs if k in have]
        missing = [(k, docs[k]) for k in keys if k not in b.get(keys)]
        if missing:
            work.append((query, missing))
        per_query.append(keys)
    if work:
        rerankers.score(api, work, f"fidelity/{dataset}/{chunking_name}")
    xs, ys, rho, top1 = [], [], [], []
    for keys in per_query:
        if len(keys) < 3:
            continue
        sa, sb = a.get(keys), b.get(keys)
        x = rerankers.to_logit(local, np.asarray([sa[k][0] for k in keys]))
        y = rerankers.to_logit(api, np.asarray([sb[k][0] for k in keys]))
        xs += list(x)
        ys += list(y)
        rx, ry = np.argsort(np.argsort(-x)), np.argsort(np.argsort(-y))
        rho.append(np.corrcoef(rx, ry)[0, 1])
        top1.append(np.argmax(x) == np.argmax(y))
    return {"pairs": len(xs), "pearson_logit": float(np.corrcoef(xs, ys)[0, 1]),
            "mean_abs_logit_diff": float(np.mean(np.abs(np.asarray(xs) - np.asarray(ys)))),
            "median_spearman_per_query": float(np.median(rho)), "top1_agreement": float(np.mean(top1))}


def embed_fidelity(dataset: str = "longmemeval_m", chunking_name: str = "seg500", n: int = 300,
                   a: str = "gemma300m-raw", b: str = "gemma300m-di-raw") -> dict:
    """Embed the same texts and queries with two embedders and compare (a fraction of a cent).

    The texts are what EventMemory embeds (derivative text: for chat histories date, producer
    and JSON text; for documents the JSON text) for segments drawn from up to three partitions. Pass means the hosted model
    can replace the local one: per-text cosine near 1, and the same top texts for the
    dataset's queries over the sampled ones."""
    from rbench.embedding import EMBEDDERS, embed

    from rbench import partitions as pt

    rng = np.random.default_rng(0)
    docs: list[str] = []
    for k in list(dict.fromkeys(pt.key(dataset, q) for q in queries(dataset)))[:3]:
        _, _, texts = pt.offline(dataset, k, chunking_name)
        docs += [texts[int(i)] for i in rng.choice(len(texts), min(len(texts), n // 3), replace=False)]
    qs = [q["variants"][next(iter(q["variants"]))] for q in queries(dataset)[:50]]
    va, vb = (embed(EMBEDDERS[e], docs, EMBEDDERS[e].doc_prompt) for e in (a, b))
    qa, qb = (embed(EMBEDDERS[e], qs, EMBEDDERS[e].query_prompt) for e in (a, b))
    cos = np.concatenate([(va * vb).sum(1), (qa * qb).sum(1)])
    top_a, top_b = np.argsort(-(qa @ va.T), 1)[:, :10], np.argsort(-(qb @ vb.T), 1)[:, :10]
    overlap = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(top_a, top_b, strict=True)])
    return {"texts": len(cos), "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos)),
            "top10_overlap": float(overlap), "top1_agreement": float(np.mean(top_a[:, 0] == top_b[:, 0]))}


# Qwen3-Embedding-8B's model card example (checked 2026-11-09): its scores, and its own vLLM run's.
CARD_QUERIES = ["What is the capital of China?", "Explain gravity"]
CARD_DOCUMENTS = ["The capital of China is Beijing.",
                  "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects "
                  "and is responsible for the movement of planets around the sun."]
CARD_SCORES = np.asarray([[0.7493, 0.0751], [0.0880, 0.6318]])
CARD_TOLERANCE = 0.005


def _raw(spec, texts: list[str], dimensions: int) -> tuple[np.ndarray, dict]:
    """The API's own vectors (float encoding, no re-normalization here) and its usage block."""
    import httpx

    from rbench.paths import SECRETS

    key = (SECRETS / spec.key_file).read_text().strip()
    body = {"model": spec.model, "input": texts, "encoding_format": "float", "dimensions": dimensions}
    data = httpx.post(spec.url + spec.route, json=body, headers={"Authorization": f"bearer {key}"}, timeout=120).json()
    vectors = np.asarray([d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])], dtype=np.float64)
    return vectors, data.get("usage", {})


def embed_checks(embedder: str = "qwen3-8b-di") -> dict:
    """Before embedding anything at scale with a hosted model (cents), on the API's own vectors:
    (1) the model card's example scores reproduce at full size; (2) a text embeds the same
    alone and in a batch with longer ones (padding and pooling); (3) the stored size comes
    back at unit norm and equals the full vector truncated to it and re-normalized; (4) billed
    tokens equal our count (nothing cut silently); (5) the path ingest uses, MemMachine's
    OpenAIEmbedder (base64, dimensions), gives the same vectors. Retrieval against
    embeddinggemma and the instruction choice is compare-embedders."""
    import asyncio

    from rbench.embedding import EMBEDDERS, QWEN3_QUERY
    from rbench.memory import QueryCachingEmbedder
    from rbench.text import count

    spec = EMBEDDERS[embedder]
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)  # noqa: E731
    out = {}
    q, _ = _raw(spec, [QWEN3_QUERY + x for x in CARD_QUERIES], 4096)
    d, _ = _raw(spec, CARD_DOCUMENTS, 4096)
    scores = unit(q) @ unit(d).T
    out["1 card scores"] = {"got": np.round(scores, 4).tolist(), "card": CARD_SCORES.tolist(),
                            "pass": bool(np.abs(scores - CARD_SCORES).max() <= CARD_TOLERANCE)}
    long = " ".join(["Gravity bends light and keeps the planets on their orbits."] * 60)
    alone = unit(_raw(spec, [CARD_DOCUMENTS[0]], 4096)[0])[0]
    batched = unit(_raw(spec, [long, CARD_DOCUMENTS[0], long + " more"], 4096)[0])[1]
    out["2 batch invariance"] = {"cosine": float(alone @ batched), "pass": bool(alone @ batched >= 0.9999)}
    short, _ = _raw(spec, CARD_DOCUMENTS, spec.dimensions)
    norms = np.linalg.norm(short, axis=1)
    cos = (unit(short) * unit(d[:, : spec.dimensions])).sum(1)
    out["3 dimensions"] = {"asked": spec.dimensions, "returned": int(short.shape[1]), "norms": np.round(norms, 6).tolist(),
                           "cosine_to_truncated_full": np.round(cos, 6).tolist(),
                           "pass": bool(short.shape[1] == spec.dimensions and np.allclose(norms, 1, atol=1e-3)
                                        and cos.min() >= 0.9999)}
    texts = [QWEN3_QUERY + CARD_QUERIES[0], *CARD_DOCUMENTS, long]
    usage = _raw(spec, texts, spec.dimensions)[1]
    ours = int(count(texts, spec.tokenizer).sum()) + spec.special_tokens * len(texts)
    out["4 billed tokens"] = {"billed": usage.get("prompt_tokens"), "ours": ours, "pass": usage.get("prompt_tokens") == ours}

    async def through_memmachine() -> np.ndarray:
        embedder_ = QueryCachingEmbedder(embedder)
        try:
            return np.asarray(await embedder_._ingest_embed(CARD_DOCUMENTS), dtype=np.float64)
        finally:
            await embedder_.aclose()

    via = asyncio.run(through_memmachine())
    cos = (unit(via) * unit(short)).sum(1)
    out["5 MemMachine path"] = {"cosine": np.round(cos, 6).tolist(), "pass": bool(cos.min() >= 0.9999)}
    return out
