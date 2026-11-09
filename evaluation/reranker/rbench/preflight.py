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

from rbench import chunking, retrieval, rerankers
from rbench.embedding import DEFAULT as EMBEDDER
from rbench.rerankers import RERANKERS, ScoreCache, pair_ids, prompt_key
from rbench.samples import sample
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


def fidelity(dataset: str, scope: str, variant: str, chunking_name: str, sample_name: str,
             local: str = "q06-local", api: str = "q06", max_usd: float = 0.10) -> dict:
    """Score through `api` the pairs `local` has scored, then compare the two."""
    table = chunking.load(dataset, chunking_name)
    texts = table.texts(dataset)
    pools = retrieval.load(EMBEDDER, dataset, scope, chunking_name, variant)
    a, b = ScoreCache(local), ScoreCache(api)
    per_query, work = [], []
    for qi in sample(dataset, sample_name):
        ids, _ = pools.head(qi)
        query = queries(dataset)[qi]["variants"][variant]
        prompt = prompt_key(RERANKERS[api].template)
        docs = {ScoreCache.key(prompt, "plain", query, texts[c]): texts[c] for c in ids}
        have = a.get(list(docs))
        keys = [k for k in docs if k in have]
        missing = [(k, docs[k]) for k in keys if k not in b.get(keys)]
        if missing:
            work.append((query, missing))
        per_query.append(keys)
    if work:
        cap = rerankers.total_spent_usd() + max_usd
        rerankers.score(api, work, f"fidelity/{dataset}/{chunking_name}", cap)
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

    For a history dataset the texts are what EventMemory embeds (derivative text: date,
    producer, JSON text) for segments drawn from three histories. Pass means the hosted model
    can replace the local one: per-text cosine near 1, and the same top texts for the
    dataset's queries over the sampled ones."""
    from rbench.datasets import info
    from rbench.embedding import EMBEDDERS, embed

    rng = np.random.default_rng(0)
    if info(dataset).partitioned:
        from rbench import histories as hs

        docs = []
        for q in queries(dataset)[:3]:
            _, _, texts = hs.offline(dataset, q["qid"], chunking_name)
            docs += [texts[int(i)] for i in rng.choice(len(texts), n // 3, replace=False)]
    else:
        texts = chunking.load(dataset, chunking_name).texts(dataset)
        docs = [texts[i] for i in rng.choice(len(texts), min(n, len(texts)), replace=False)]
    qs = [q["variants"][next(iter(q["variants"]))] for q in queries(dataset)[:50]]
    va, vb = (embed(EMBEDDERS[e], docs, EMBEDDERS[e].doc_prompt) for e in (a, b))
    qa, qb = (embed(EMBEDDERS[e], qs, EMBEDDERS[e].query_prompt) for e in (a, b))
    cos = np.concatenate([(va * vb).sum(1), (qa * qb).sum(1)])
    top_a, top_b = np.argsort(-(qa @ va.T), 1)[:, :10], np.argsort(-(qb @ vb.T), 1)[:, :10]
    overlap = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(top_a, top_b, strict=True)])
    return {"texts": len(cos), "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos)),
            "top10_overlap": float(overlap), "top1_agreement": float(np.mean(top_a[:, 0] == top_b[:, 0]))}
