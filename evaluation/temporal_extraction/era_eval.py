"""E4 — Named-era retrieval eval.

1) Generate ~15 new era-focused synthetic docs + ~20 queries with gold,
   persist to data/era_{docs,queries,gold}.jsonl.
2) Run base extractor + era_extractor on both sets.
3) Build a separate IntervalStore for each and compare:
   - extraction recall (how many gold eras the extractor correctly
     brackets to within 6-month tolerance of centroid)
   - retrieval recall@5/@10, MRR, NDCG@10 on the 20 era queries
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from advanced_common import (
    DATA_DIR,
    RESULTS_DIR,
    LLMCaller,
    load_jsonl,
    mean,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from baselines import embed_all, semantic_rank
from era_extractor import EraExtractor
from extractor import Extractor as BaseExtractor
from schema import TimeExpression, parse_iso

# ---------------------------------------------------------------------------
# Synthetic era corpus (15 docs, 20 queries)
# ---------------------------------------------------------------------------
REF_TIME = "2026-04-23T12:00:00Z"


def _iso(y: int, m: int = 1, d: int = 1) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}T00:00:00Z"


# Each doc has text + a "gold_window" = (earliest ISO, latest ISO)
# representing the era to which the core event is anchored. We use that
# for extraction correctness + retrieval relevance computation.
DOCS: list[dict[str, Any]] = [
    # --- World-knowledge eras ---
    {
        "doc_id": "era_doc_obama_0",
        "text": (
            "Back during the Obama years we spent nearly every Saturday "
            "hiking the same trail in the state park."
        ),
        "gold_window": (_iso(2009, 1, 20), _iso(2017, 1, 20)),
        "era_kind": "world",
    },
    {
        "doc_id": "era_doc_90s_0",
        "text": (
            "In the 90s my parents ran a record store downtown; I spent "
            "my afternoons rearranging the vinyl."
        ),
        "gold_window": (_iso(1990), _iso(2000)),
        "era_kind": "world",
    },
    {
        "doc_id": "era_doc_80s_0",
        "text": (
            "The 80s were wild at the arcade — Pac-Man tournaments every "
            "Friday and Def Leppard on the radio."
        ),
        "gold_window": (_iso(1980), _iso(1990)),
        "era_kind": "world",
    },
    {
        "doc_id": "era_doc_coldwar_0",
        "text": (
            "My grandfather worked at the embassy during the Cold War "
            "and rarely spoke about those years."
        ),
        "gold_window": (_iso(1947, 3, 12), _iso(1991, 12, 26)),
        "era_kind": "world",
    },
    {
        "doc_id": "era_doc_postwwii_0",
        "text": (
            "Post-WWII, my great-aunt built her own house from salvaged "
            "timber and opened a bakery in town."
        ),
        "gold_window": (_iso(1945, 9, 2), _iso(1960)),
        "era_kind": "world",
    },
    {
        "doc_id": "era_doc_greatrec_0",
        "text": (
            "I lost my job in the Great Recession and ended up learning "
            "to bake sourdough to save money."
        ),
        "gold_window": (_iso(2007, 12), _iso(2009, 6)),
        "era_kind": "world",
    },
    {
        "doc_id": "era_doc_2010s_0",
        "text": (
            "Throughout the 2010s Slack became the center of our work "
            "life; we replaced three inboxes with it."
        ),
        "gold_window": (_iso(2010), _iso(2020)),
        "era_kind": "world",
    },
    {
        "doc_id": "era_doc_covid_0",
        "text": (
            "During COVID I adopted two cats and finally read War and "
            "Peace from cover to cover."
        ),
        "gold_window": (_iso(2020, 3, 11), _iso(2023, 5, 5)),
        "era_kind": "world",
    },
    # --- Personal eras ---
    {
        "doc_id": "era_doc_college_0",
        "text": (
            "During college I worked three nights a week at a campus "
            "cafe and drank far too much espresso."
        ),
        # author implied ~ age 18-22; anchor: recent past
        "gold_window": (_iso(2014), _iso(2018)),
        "era_kind": "personal",
    },
    {
        "doc_id": "era_doc_college_1",
        "text": (
            "I took a semester off during college to backpack through "
            "Southeast Asia, and it changed how I thought about routine."
        ),
        "gold_window": (_iso(2014), _iso(2018)),
        "era_kind": "personal",
    },
    {
        "doc_id": "era_doc_twenties_0",
        "text": (
            "In my 20s I lived in six different cities; I don't recommend "
            "that to anyone who likes stability."
        ),
        "gold_window": (_iso(2010), _iso(2020)),
        "era_kind": "personal",
    },
    {
        "doc_id": "era_doc_teen_0",
        "text": (
            "When I was a teenager my family moved to Albuquerque for my "
            "dad's job at the lab; I hated it at first."
        ),
        "gold_window": (_iso(2005), _iso(2012)),
        "era_kind": "personal",
    },
    {
        "doc_id": "era_doc_before_kids_0",
        "text": (
            "Before the kids were born we used to go camping every other "
            "weekend. Now the tent just collects dust."
        ),
        "gold_window": (_iso(2015), _iso(2020)),
        "era_kind": "personal",
    },
    {
        "doc_id": "era_doc_post_covid_0",
        "text": (
            "Post-COVID the office moved fully remote; I turned my spare "
            "bedroom into a permanent workspace."
        ),
        "gold_window": (_iso(2023, 5, 5), _iso(2026, 4, 23)),
        "era_kind": "world",
    },
    {
        "doc_id": "era_doc_pre_internet_0",
        "text": (
            "Pre-internet we had to mail actual letters to stay in touch "
            "— my grandma still has a shoebox full of them."
        ),
        "gold_window": (_iso(1900), _iso(1995)),
        "era_kind": "world",
    },
]

# Queries: 20 targeting the era references above. Each query is (text,
# relevant_doc_ids). Some queries target multiple docs (overlapping eras).
QUERIES: list[dict[str, Any]] = [
    # Direct era reuse
    {
        "query_id": "era_q_obama_0",
        "text": "What did I do during the Obama years?",
        "relevant": ["era_doc_obama_0"],
    },
    {
        "query_id": "era_q_90s_0",
        "text": "What was the record store like in the 90s?",
        "relevant": ["era_doc_90s_0"],
    },
    {
        "query_id": "era_q_80s_0",
        "text": "What were the 80s arcade nights like?",
        "relevant": ["era_doc_80s_0"],
    },
    {
        "query_id": "era_q_coldwar_0",
        "text": "What did grandfather do during the Cold War?",
        "relevant": ["era_doc_coldwar_0"],
    },
    {
        "query_id": "era_q_postwwii_0",
        "text": "What happened just after WWII in my family?",
        "relevant": ["era_doc_postwwii_0"],
    },
    {
        "query_id": "era_q_greatrec_0",
        "text": "What did I do during the Great Recession?",
        "relevant": ["era_doc_greatrec_0"],
    },
    {
        "query_id": "era_q_2010s_0",
        "text": "What tools changed work for us in the 2010s?",
        "relevant": ["era_doc_2010s_0"],
    },
    {
        "query_id": "era_q_covid_0",
        "text": "What did I do during COVID?",
        "relevant": ["era_doc_covid_0"],
    },
    {
        "query_id": "era_q_college_0",
        "text": "What did I do during college?",
        "relevant": ["era_doc_college_0", "era_doc_college_1"],
    },
    {
        "query_id": "era_q_college_1",
        "text": "Did I travel in college?",
        "relevant": ["era_doc_college_1"],
    },
    {
        "query_id": "era_q_twenties_0",
        "text": "What was life like in my 20s?",
        "relevant": ["era_doc_twenties_0", "era_doc_2010s_0"],
    },
    {
        "query_id": "era_q_teen_0",
        "text": "What happened when I was a teenager?",
        "relevant": ["era_doc_teen_0"],
    },
    {
        "query_id": "era_q_before_kids_0",
        "text": "What did we do before the kids were born?",
        "relevant": ["era_doc_before_kids_0"],
    },
    {
        "query_id": "era_q_post_covid_0",
        "text": "What changed after COVID at work?",
        "relevant": ["era_doc_post_covid_0"],
    },
    {
        "query_id": "era_q_pre_internet_0",
        "text": "How did people stay in touch pre-internet?",
        "relevant": ["era_doc_pre_internet_0"],
    },
    # Paraphrased era queries
    {
        "query_id": "era_q_obama_1",
        "text": "What was a regular weekend in the late 2000s to mid 2010s?",
        "relevant": ["era_doc_obama_0"],
    },
    {
        "query_id": "era_q_nineties_0",
        "text": "Tell me about the nineties at the store.",
        "relevant": ["era_doc_90s_0"],
    },
    {
        "query_id": "era_q_after_wwii_0",
        "text": "What did relatives do in the postwar era?",
        "relevant": ["era_doc_postwwii_0"],
    },
    {
        "query_id": "era_q_twenties_1",
        "text": "What was my life like around age 25?",
        "relevant": ["era_doc_twenties_0", "era_doc_2010s_0"],
    },
    {
        "query_id": "era_q_during_covid_0",
        "text": "What happened around 2020-2022?",
        "relevant": ["era_doc_covid_0", "era_doc_post_covid_0"],
    },
]


def write_corpus() -> None:
    """Persist era corpus to data/era_*.jsonl."""
    docs_out = DATA_DIR / "era_docs.jsonl"
    queries_out = DATA_DIR / "era_queries.jsonl"
    gold_out = DATA_DIR / "era_gold.jsonl"

    with docs_out.open("w") as f:
        for d in DOCS:
            obj = {
                "doc_id": d["doc_id"],
                "text": d["text"],
                "ref_time": REF_TIME,
                "gold_window": d["gold_window"],
                "era_kind": d["era_kind"],
            }
            f.write(json.dumps(obj) + "\n")
    with queries_out.open("w") as f:
        for q in QUERIES:
            obj = {
                "query_id": q["query_id"],
                "text": q["text"],
                "ref_time": REF_TIME,
                "relevant": q["relevant"],
            }
            f.write(json.dumps(obj) + "\n")
    with gold_out.open("w") as f:
        for q in QUERIES:
            f.write(
                json.dumps(
                    {"query_id": q["query_id"], "relevant_doc_ids": q["relevant"]}
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Extraction correctness
# ---------------------------------------------------------------------------
def te_window(te: TimeExpression) -> tuple[datetime, datetime] | None:
    if te.kind == "instant" and te.instant:
        return te.instant.earliest, te.instant.latest
    if te.kind == "interval" and te.interval:
        return te.interval.start.earliest, te.interval.end.latest
    return None


def extraction_hit(
    tes: list[TimeExpression], gold_start: datetime, gold_end: datetime
) -> bool:
    """Any extracted TE whose window overlaps at least 30% of the gold
    window counts as a hit."""
    g_span = (gold_end - gold_start).total_seconds()
    if g_span <= 0:
        return False
    for te in tes:
        w = te_window(te)
        if w is None:
            continue
        e, l = w
        inter = (min(l, gold_end) - max(e, gold_start)).total_seconds()
        if inter <= 0:
            continue
        # Allow hits that cover the gold or are covered by it with >=30%
        # overlap relative to the gold span.
        if inter / g_span >= 0.3:
            return True
    return False


# ---------------------------------------------------------------------------
# Retrieval on era corpus (temporal-only)
# ---------------------------------------------------------------------------
def temporal_score(q_tes: list[TimeExpression], d_tes: list[TimeExpression]) -> float:
    total = 0.0
    for q in q_tes:
        qw = te_window(q)
        if qw is None:
            continue
        qe, ql = qw
        qspan = (ql - qe).total_seconds() or 1.0
        best = 0.0
        for d in d_tes:
            dw = te_window(d)
            if dw is None:
                continue
            de, dl = dw
            inter = (min(ql, dl) - max(qe, de)).total_seconds()
            if inter <= 0:
                continue
            union = (max(ql, dl) - min(qe, de)).total_seconds() or 1.0
            jacc = inter / union
            if jacc > best:
                best = jacc
        total += best
    return total


async def main() -> None:
    write_corpus()
    docs = load_jsonl(DATA_DIR / "era_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "era_queries.jsonl")
    gold = {q["query_id"]: set(q["relevant"]) for q in queries}

    base = BaseExtractor()
    llm = LLMCaller(concurrency=10)
    era = EraExtractor(llm)

    async def run_on(extractor_fn, text, ref):
        try:
            return await extractor_fn(text, ref)
        except Exception as e:
            print(f"  extract failed: {e}")
            return []

    print(
        f"E4: running base + era extractor on {len(docs)} docs, {len(queries)} queries..."
    )
    base_doc_tes = await asyncio.gather(
        *(run_on(base.extract, d["text"], parse_iso(d["ref_time"])) for d in docs)
    )
    base_q_tes = await asyncio.gather(
        *(run_on(base.extract, q["text"], parse_iso(q["ref_time"])) for q in queries)
    )
    era_doc_tes = await asyncio.gather(
        *(run_on(era.extract, d["text"], parse_iso(d["ref_time"])) for d in docs)
    )
    era_q_tes = await asyncio.gather(
        *(run_on(era.extract, q["text"], parse_iso(q["ref_time"])) for q in queries)
    )
    base.cache.save()
    llm.save()

    base_doc_map = {d["doc_id"]: t for d, t in zip(docs, base_doc_tes)}
    era_doc_map = {d["doc_id"]: t for d, t in zip(docs, era_doc_tes)}
    base_q_map = {q["query_id"]: t for q, t in zip(queries, base_q_tes)}
    era_q_map = {q["query_id"]: t for q, t in zip(queries, era_q_tes)}

    # 1) Extraction correctness — gold window coverage
    def eval_extraction(doc_map) -> dict[str, float]:
        hits = 0
        total = len(docs)
        per_kind = {"world": [0, 0], "personal": [0, 0]}
        for d in docs:
            tes = doc_map[d["doc_id"]]
            gs = parse_iso(d["gold_window"][0])
            ge = parse_iso(d["gold_window"][1])
            hit = extraction_hit(tes, gs, ge)
            per_kind[d["era_kind"]][1] += 1
            if hit:
                hits += 1
                per_kind[d["era_kind"]][0] += 1
        return {
            "doc_recall": hits / total,
            "docs_hit": hits,
            "docs_total": total,
            "world_hit": per_kind["world"][0],
            "world_total": per_kind["world"][1],
            "personal_hit": per_kind["personal"][0],
            "personal_total": per_kind["personal"][1],
        }

    base_ext = eval_extraction(base_doc_map)
    era_ext = eval_extraction(era_doc_map)

    # 2) Temporal retrieval over era corpus (doc-side only; full ranking
    # against all 15 era docs)
    def run_retrieval(q_map, doc_map) -> dict[str, float]:
        rec5s, rec10s, mrrs, ndcgs = [], [], [], []
        for q in queries:
            qid = q["query_id"]
            rel = gold.get(qid, set())
            q_tes = q_map[qid]
            scores: list[tuple[str, float]] = []
            for d in docs:
                s = temporal_score(q_tes, doc_map[d["doc_id"]])
                scores.append((d["doc_id"], s))
            scores.sort(key=lambda x: x[1], reverse=True)
            ranked = [did for did, _ in scores]
            if not rel:
                continue
            rec5s.append(recall_at_k(ranked, rel, 5))
            rec10s.append(recall_at_k(ranked, rel, 10))
            mrrs.append(mrr(ranked, rel))
            ndcgs.append(ndcg_at_k(ranked, rel, 10))
        return {
            "recall@5": mean(rec5s),
            "recall@10": mean(rec10s),
            "mrr": mean(mrrs),
            "ndcg@10": mean(ndcgs),
        }

    base_ret = run_retrieval(base_q_map, base_doc_map)
    era_ret = run_retrieval(era_q_map, era_doc_map)

    # 3) Semantic baseline for comparison on era corpus
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    all_embs = await embed_all(doc_texts + q_texts)
    doc_embs = {d["doc_id"]: all_embs[i] for i, d in enumerate(docs)}

    sem_rec5s, sem_rec10s, sem_mrrs, sem_ndcgs = [], [], [], []
    for i, q in enumerate(queries):
        qid = q["query_id"]
        qe = all_embs[len(docs) + i]
        ranked = [d for d, _ in semantic_rank(qe, doc_embs)]
        rel = gold.get(qid, set())
        if not rel:
            continue
        sem_rec5s.append(recall_at_k(ranked, rel, 5))
        sem_rec10s.append(recall_at_k(ranked, rel, 10))
        sem_mrrs.append(mrr(ranked, rel))
        sem_ndcgs.append(ndcg_at_k(ranked, rel, 10))
    sem_ret = {
        "recall@5": mean(sem_rec5s),
        "recall@10": mean(sem_rec10s),
        "mrr": mean(sem_mrrs),
        "ndcg@10": mean(sem_ndcgs),
    }

    report = {
        "extraction": {"base": base_ext, "era": era_ext},
        "retrieval": {"base_T": base_ret, "era_T": era_ret, "S": sem_ret},
        "usage_llm": llm.usage,
        "cost_usd_llm": llm.cost_usd(),
        "n_docs": len(docs),
        "n_queries": len(queries),
    }

    out_path = RESULTS_DIR / "advanced_e4_era.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"E4 wrote {out_path}")
    print(
        json.dumps(
            {"extraction": report["extraction"], "retrieval": report["retrieval"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
