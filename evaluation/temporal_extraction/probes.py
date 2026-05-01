"""E2 — Pre-materialized temporal probes at ingest.

For each doc's time references, generate 3-5 paraphrase surfaces via
gpt-5-mini. Embed each paraphrase independently. At query time, cosine-
search the expanded probe index and aggregate to doc_id by MAX (best-
paraphrase match wins per doc).

Reuses the existing cache/llm_cache.json extractions for the base extractor
when available (reads time surfaces from doc text via a one-pass LLM call
ourselves, to avoid dependency on the ablation-running extractor state).

Instead we call pass1 of the extractor directly — this reuses the same
cache keys as the base extractor, so no new LLM cost for the first pass.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import numpy as np
from advanced_common import (
    DATA_DIR,
    RESULTS_DIR,
    Embedder,
    LLMCaller,
    load_jsonl,
    mean,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from extractor import Extractor as BaseExtractor
from schema import parse_iso

PARAPHRASE_SYSTEM = """You generate paraphrase surfaces for a time expression.

Given a single time expression and its reference time, produce 3-5 concise
alternative surface forms that COULD appear in a natural passage and still
refer to the same time window or a closely related one. Include a mix of:
- date-string variants (e.g., "March 15, 2026" -> "3/15/2026", "2026-03-15", "Mar 15 2026")
- granularity-shifted variants (e.g., day -> month: "mid-March 2026"; month -> quarter: "Q1 2026"; year -> decade: "the mid-2020s")
- named-era / natural-language variants (e.g., "2021" -> "the year after COVID started"; "2009-2017" -> "the Obama years")
- short natural-language rephrasings

For a relative expression like "yesterday", use the reference time to
produce an absolute variant plus 2-3 short surface rephrasings.

Output JSON: {"paraphrases": ["...", "...", ...]}. Do not include the
input surface itself. Keep each paraphrase under 8 words.
"""

PARAPHRASE_SCHEMA = {
    "name": "paraphrases",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "paraphrases": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["paraphrases"],
    },
}


async def paraphrase_surface(
    llm: LLMCaller, surface: str, ref_time: datetime
) -> list[str]:
    iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    user = (
        f"Reference time: {iso_ref}\n"
        f'Time expression: "{surface}"\n\n'
        'Return {"paraphrases": [...]}.'
    )
    raw = await llm.chat(
        PARAPHRASE_SYSTEM,
        user,
        json_schema=PARAPHRASE_SCHEMA,
        max_completion_tokens=1000,
        cache_tag="e2_paraphrase",
    )
    if not raw:
        return []
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = [s.strip() for s in d.get("paraphrases", []) if s and s.strip()]
    return out[:5]


async def main() -> None:
    docs = load_jsonl(DATA_DIR / "docs.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "gold.jsonl")
    }
    critical_pairs = json.loads((DATA_DIR / "critical_pairs.json").read_text())
    crit_map = {q_id: doc_id for (doc_id, q_id) in critical_pairs}

    # 1) Reuse base-extractor pass1 on docs to get time surfaces
    base = BaseExtractor()
    llm = LLMCaller(concurrency=10)
    embedder = Embedder(concurrency=10)

    async def pass1_for(text: str, ref: datetime) -> list[str]:
        refs = await base.pass1(text, ref)
        out = []
        for r in refs:
            s = (r.get("surface") or "").strip()
            if s:
                out.append(s)
        return out

    print(f"E2: gathering time surfaces for {len(docs)} docs (reuses base cache)...")
    doc_surfaces: dict[str, list[tuple[str, datetime]]] = {}
    for d in docs:
        ref = parse_iso(d["ref_time"])
        surfaces = await pass1_for(d["text"], ref)
        doc_surfaces[d["doc_id"]] = [(s, ref) for s in surfaces]
    base.cache.save()
    print(f"  {sum(len(v) for v in doc_surfaces.values())} surfaces total")

    # 2) Generate paraphrases (new LLM calls)
    print("E2: generating paraphrases...")

    async def paraphrase_for(doc_id: str, s: str, ref: datetime):
        ps = await paraphrase_surface(llm, s, ref)
        return (doc_id, s, ps)

    coros = []
    for doc_id, sr in doc_surfaces.items():
        for s, ref in sr:
            coros.append(paraphrase_for(doc_id, s, ref))
    results = await asyncio.gather(*coros)
    llm.save()
    # Probe index: list of (doc_id, text, vec)
    # Include both originals and paraphrases.
    probe_texts: list[tuple[str, str]] = []  # (doc_id, text)
    for doc_id, sr in doc_surfaces.items():
        for s, _ref in sr:
            probe_texts.append((doc_id, s))
    for doc_id, s, ps in results:
        for p in ps:
            probe_texts.append((doc_id, p))

    print(f"  {len(probe_texts)} probe rows (originals + paraphrases)")

    # 3) Embed all probes
    print("E2: embedding probes...")
    texts = [t for _, t in probe_texts]
    # Dedupe embeddings
    uniq = list({t for t in texts})
    uniq_vecs = await embedder.embed_batch(uniq)
    embedder.save()
    vec_map = {t: v for t, v in zip(uniq, uniq_vecs)}

    # 4) For queries: use query TEXT embedding directly (matches the probe
    # index natively). This is the key E2 pattern: query text -> cosine
    # over doc-side paraphrase probes.
    q_texts = [q["text"] for q in queries]
    q_vecs_list = await embedder.embed_batch(q_texts)
    embedder.save()
    q_vec_map = {q["query_id"]: v for q, v in zip(queries, q_vecs_list)}

    # Pre-stack probe vectors for batched cosine
    probe_mat = np.stack([vec_map[t] for _, t in probe_texts])
    probe_doc_ids = [d for d, _ in probe_texts]

    # Normalize
    norm = np.linalg.norm(probe_mat, axis=1, keepdims=True) + 1e-9
    probe_norm = probe_mat / norm

    def rank_probes(qid: str) -> list[tuple[str, float]]:
        qv = q_vec_map[qid]
        qn = qv / (np.linalg.norm(qv) + 1e-9)
        sims = probe_norm @ qn  # (N,)
        # Aggregate per doc by MAX
        best: dict[str, float] = {}
        for doc_id, s in zip(probe_doc_ids, sims):
            s = float(s)
            if s > best.get(doc_id, -1.0):
                best[doc_id] = s
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

    # 5) Evaluate
    rec5s, rec10s, mrrs, ndcgs = [], [], [], []
    crit_top1 = 0
    per_query: dict[str, Any] = {}
    for q in queries:
        qid = q["query_id"]
        ranked_pairs = rank_probes(qid)
        ranked = [d for d, _ in ranked_pairs]
        if qid in crit_map and ranked and ranked[0] == crit_map[qid]:
            crit_top1 += 1
        rel = gold.get(qid, set())
        per_query[qid] = ranked[:10]
        if not rel:
            continue
        rec5s.append(recall_at_k(ranked, rel, 5))
        rec10s.append(recall_at_k(ranked, rel, 10))
        mrrs.append(mrr(ranked, rel))
        ndcgs.append(ndcg_at_k(ranked, rel, 10))

    baseline = json.loads((RESULTS_DIR / "retrieval_results.json").read_text())
    report = {
        "recall@5": mean(rec5s),
        "recall@10": mean(rec10s),
        "mrr": mean(mrrs),
        "ndcg@10": mean(ndcgs),
        "critical_top1": crit_top1,
        "critical_total": len(critical_pairs),
        "n_probes": len(probe_texts),
        "usage_llm": llm.usage,
        "cost_usd_llm": llm.cost_usd(),
        "baseline_T_and_S": baseline.get("T_and_S", {}),
        "baseline_S": baseline.get("S", {}),
    }

    out_path = RESULTS_DIR / "advanced_e2_probes.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"E2 wrote {out_path}")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["recall@5", "recall@10", "mrr", "ndcg@10", "critical_top1"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
