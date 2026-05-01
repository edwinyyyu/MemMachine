"""T_causal: causal_relative 2-step query plan.

Pipeline:
  1. Detect causal cue (after/before/since/until/following/prior to/right
     after) — but only when there is NO open-ended date anchor (so
     "after 2020" is excluded; the open_ended router handles that).
  2. Extract anchor noun phrase right after the cue word.
  3. Resolve anchor in corpus: cosine top-1 over anchor phrase.
  4. Filter / score: mask wrong-direction docs; or signed penalty.
  5. Tail with fuse_T_R + recency_additive baseline ranking.

Compares vs:
  - rerank_only (baseline)
  - fuse_T_R + recency_additive (current best, R@1 = 0.467 on causal_relative)
  - causal_mask (drop wrong-direction docs)
  - causal_signed (subtract λ from wrong-direction scores)

Run on all 11 benchmarks for regression check.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Strip proxy env vars set by sandbox.
for _k in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
):
    os.environ.pop(_k, None)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

import numpy as np
from baselines import cosine
from force_pick_optimizers_eval import (
    RERANK_TOP_K,
    make_t_scores,
    merge_with_tail,
    rerank_topk,
    topk_from_scores,
)
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from rag_fusion import score_blend
from recency import (
    has_recency_cue,
    lambda_for_half_life,
    recency_scores_for_docs,
)
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from schema import to_us
from T_open_ended_router_eval import has_open_ended_cue

# Match multi_channel_eval defaults so the baselines line up.
HALF_LIFE_DAYS = 21.0
CV_REF = 0.20
W_T_FUSE_TR = 0.4
W_R_FUSE_TR = 0.6
ADDITIVE_ALPHA = 0.5

# Causal-signed penalty (subtracted from base score for wrong-direction docs).
CAUSAL_SIGNED_LAMBDA = 0.5

# Anchor resolution: minimum cosine to accept anchor; below this, fall back.
ANCHOR_MIN_SIM = 0.30

# Map causal cue -> direction. "after" / "since" / "following" => doc later
# than anchor. "before" / "until" / "prior to" => doc earlier than anchor.
_AFTER_CUES = ("after", "since", "following", "right after")
_BEFORE_CUES = ("before", "until", "prior to")

# Ordered (longest-first to avoid prefix collisions).
_CUE_PHRASES = [
    "right after",
    "prior to",
    "after",
    "before",
    "since",
    "until",
    "following",
]

# Match cue + the noun phrase that follows it. The anchor phrase ends at
# punctuation, end-of-string, or a clause-ending word like "was/is/had".
_TRAILING_STOP = (
    r"(?=[.,;:!?]|$|\s+(?:was|is|are|were|had|has|have|completed|complete|"
    r"closed|wrapped|ended|finished)\b)"
)


def _build_cue_regex():
    # Longest first.
    alts = "|".join(re.escape(p) for p in _CUE_PHRASES)
    # cue + (optional "the") + greedy noun phrase up to a stop.
    return re.compile(
        r"\b(?P<cue>"
        + alts
        + r")\s+(?P<phrase>(?:the\s+)?[a-zA-Z][\w\s'-]*?)"
        + _TRAILING_STOP,
        re.IGNORECASE,
    )


_CUE_RE = _build_cue_regex()


def detect_causal(query_text: str) -> tuple[str, str] | None:
    """Return (cue, anchor_phrase) if a causal_relative cue fires, else None.

    Suppress when has_open_ended_cue is True (e.g. "after 2020" — handled
    by the open_ended router).
    """
    if not query_text:
        return None
    if has_open_ended_cue(query_text):
        return None
    m = _CUE_RE.search(query_text)
    if not m:
        return None
    cue = m.group("cue").lower()
    phrase = m.group("phrase").strip()
    # Reject empty / pronoun-only phrases.
    if not phrase or phrase.lower() in {"the", "a", "an", "it", "that", "this"}:
        return None
    return cue, phrase


def cue_direction(cue: str) -> str:
    """Return 'after' or 'before' for the cue word."""
    cl = cue.lower()
    if cl in _AFTER_CUES:
        return "after"
    if cl in _BEFORE_CUES:
        return "before"
    # Default to 'after' for unrecognized — should not happen.
    return "after"


def resolve_anchor(
    anchor_phrase: str,
    anchor_emb: np.ndarray,
    doc_embs: dict[str, np.ndarray],
) -> tuple[str, float] | None:
    """Top-1 cosine match in the corpus."""
    best_did = None
    best_sim = -1.0
    for did, emb in doc_embs.items():
        s = cosine(anchor_emb, emb)
        if s > best_sim:
            best_sim = s
            best_did = did
    if best_did is None or best_sim < ANCHOR_MIN_SIM:
        return None
    return best_did, best_sim


def direction_match(doc_us: int | None, anchor_us: int | None, direction: str) -> bool:
    if doc_us is None or anchor_us is None:
        return True  # don't filter when undated
    if direction == "after":
        return doc_us >= anchor_us
    return doc_us <= anchor_us


def causal_mask_scores(
    base_scores: dict[str, float],
    doc_ref_us: dict[str, int],
    anchor_us: int,
    direction: str,
    anchor_did: str,
) -> dict[str, float]:
    """Drop docs on the wrong side of anchor; also drop the anchor itself."""
    out: dict[str, float] = {}
    for did, s in base_scores.items():
        if did == anchor_did:
            out[did] = 0.0
            continue
        if direction_match(doc_ref_us.get(did), anchor_us, direction):
            out[did] = s
        else:
            out[did] = 0.0
    return out


def causal_signed_scores(
    base_scores: dict[str, float],
    doc_ref_us: dict[str, int],
    anchor_us: int,
    direction: str,
    anchor_did: str,
    lam: float = CAUSAL_SIGNED_LAMBDA,
) -> dict[str, float]:
    """Subtract λ from wrong-direction docs; suppress anchor itself."""
    out: dict[str, float] = {}
    for did, s in base_scores.items():
        if did == anchor_did:
            out[did] = s - lam
            continue
        if direction_match(doc_ref_us.get(did), anchor_us, direction):
            out[did] = s
        else:
            out[did] = s - lam
    return out


def fuse_T_R_blend_scores(t_scores, r_scores, w_T=W_T_FUSE_TR):
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": w_T, "R": 1.0 - w_T},
        top_k_per=40,
        dispersion_cv_ref=CV_REF,
    )
    return dict(fused)


def additive_with_recency(base_scores, rec_scores, cue, alpha=ADDITIVE_ALPHA):
    if not cue:
        return dict(base_scores)
    docs = set(base_scores) | set(rec_scores)
    out = {}
    for d in docs:
        out[d] = (1.0 - alpha) * base_scores.get(d, 0.0) + alpha * rec_scores.get(
            d, 0.0
        )
    return out


def rank_from_scores(scores):
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def normalize_rerank_full(rerank_partial, all_doc_ids, tail_score=0.0):
    if not rerank_partial:
        return dict.fromkeys(all_doc_ids, tail_score)
    vals = list(rerank_partial.values())
    rmin, rmax = min(vals), max(vals)
    span = (rmax - rmin) or 1.0
    out = {}
    for did in all_doc_ids:
        if did in rerank_partial:
            out[did] = (rerank_partial[did] - rmin) / span
        else:
            out[did] = tail_score
    return out


# -----------------------------------------------------------------------------
# Bench
# -----------------------------------------------------------------------------
async def run_bench(name, docs_path, queries_path, gold_path, cache_label, reranker):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    q_ref_us = {q["query_id"]: to_us(parse_iso(q["ref_time"])) for q in queries}

    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)
    for d in docs:
        doc_mem.setdefault(
            d["doc_id"],
            {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "multi_tags": set(),
            },
        )

    # Lattice for T_lblend
    lat_db = ROOT / "cache" / "T_causal" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)
    qids = [q["query_id"] for q in queries]
    per_q_l = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }

    # Recency anchor bundles
    doc_bundles_for_rec: dict[str, list[dict]] = {}
    for did, mem in doc_mem.items():
        ivs = mem.get("intervals") or []
        doc_bundles_for_rec[did] = [{"intervals": ivs}] if ivs else []

    # Doc + query embeddings
    print("  embedding...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # Detect causal queries; collect anchor phrases for batch embedding.
    causal_info: dict[str, dict] = {}
    anchor_phrases: list[str] = []
    for q in queries:
        info = detect_causal(q["text"])
        if info is None:
            continue
        cue, phrase = info
        causal_info[q["query_id"]] = {
            "cue": cue,
            "phrase": phrase,
            "direction": cue_direction(cue),
        }
        anchor_phrases.append(phrase)

    # Embed unique anchor phrases.
    unique_phrases = list(dict.fromkeys(anchor_phrases))
    print(
        f"  causal queries: {len(causal_info)}, unique anchor phrases: {len(unique_phrases)}",
        flush=True,
    )
    if unique_phrases:
        phrase_emb_list = await embed_all(unique_phrases)
        phrase_emb = {p: phrase_emb_list[i] for i, p in enumerate(unique_phrases)}
    else:
        phrase_emb = {}

    # Resolve anchors. Track success/failure for accuracy report.
    anchor_resolution: dict[str, dict] = {}
    for qid, info in causal_info.items():
        phrase = info["phrase"]
        emb = phrase_emb.get(phrase)
        if emb is None:
            anchor_resolution[qid] = {"resolved": False, "reason": "no_emb"}
            continue
        res = resolve_anchor(phrase, emb, doc_embs)
        if res is None:
            anchor_resolution[qid] = {"resolved": False, "reason": "low_sim"}
            continue
        did, sim = res
        anchor_us = doc_ref_us.get(did)
        anchor_resolution[qid] = {
            "resolved": True,
            "anchor_did": did,
            "sim": sim,
            "anchor_us": anchor_us,
            "direction": info["direction"],
            "phrase": phrase,
            "cue": info["cue"],
        }

    # T_lblend scores per query
    per_q_t = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }
    for qid in qids:
        for d in docs:
            per_q_t[qid].setdefault(d["doc_id"], 0.0)

    # Rerank: union(top-50 sem, top-50 T_lblend) -> cross-encoder
    print("  reranking...", flush=True)
    per_q_r_full: dict[str, dict[str, float]] = {}
    per_q_r_partial: dict[str, dict[str, float]] = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        rs = await rerank_topk(reranker, q_text[qid], union, doc_text, len(union))
        per_q_r_partial[qid] = rs
        per_q_r_full[qid] = normalize_rerank_full(
            rs, [d["doc_id"] for d in docs], tail_score=0.0
        )

    lam = lambda_for_half_life(HALF_LIFE_DAYS)

    n_causal = 0
    n_resolved = 0
    n_anchor_correct = 0  # anchor doc id matches the actual `_a` doc
    causal_examples: list[dict] = []

    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue

        Recency_active = has_recency_cue(q["text"])
        causal = qid in causal_info
        if causal:
            n_causal += 1
        ar = anchor_resolution.get(qid)
        if ar and ar.get("resolved"):
            n_resolved += 1

        t_scores = per_q_t[qid]
        r_full = per_q_r_full[qid]
        rerank_partial = per_q_r_partial[qid]
        s_scores = per_q_s[qid]

        rec_scores = recency_scores_for_docs(
            doc_bundles_for_rec,
            doc_ref_us,
            q_ref_us[qid],
            lam,
        )

        # Variant 1: rerank_only baseline.
        rerank_only_rank = merge_with_tail(
            [
                d
                for d, _ in sorted(
                    rerank_partial.items(), key=lambda x: x[1], reverse=True
                )
            ],
            s_scores,
        )

        # Variant 2: fuse_T_R + recency_additive (current best).
        fused_TR_scores = fuse_T_R_blend_scores(t_scores, r_full, w_T=W_T_FUSE_TR)
        fused_TR_with_rec = additive_with_recency(
            fused_TR_scores,
            rec_scores,
            cue=Recency_active,
            alpha=ADDITIVE_ALPHA,
        )
        primary_2 = rank_from_scores(fused_TR_with_rec)
        rank_baseline = primary_2 + [
            d for d in rank_from_scores(s_scores) if d not in set(primary_2)
        ]

        # Variant 3 + 4: causal-aware. Apply mask / signed to fused_TR_with_rec
        # when causal cue fires AND anchor resolved. Otherwise fall back to
        # baseline.
        if causal and ar and ar.get("resolved"):
            anchor_us = ar["anchor_us"]
            anchor_did = ar["anchor_did"]
            direction = ar["direction"]
            mask_scores = causal_mask_scores(
                fused_TR_with_rec,
                doc_ref_us,
                anchor_us,
                direction,
                anchor_did,
            )
            signed_scores = causal_signed_scores(
                fused_TR_with_rec,
                doc_ref_us,
                anchor_us,
                direction,
                anchor_did,
                lam=CAUSAL_SIGNED_LAMBDA,
            )
            primary_mask = rank_from_scores(mask_scores)
            rank_mask = primary_mask + [
                d for d in rank_from_scores(s_scores) if d not in set(primary_mask)
            ]
            primary_signed = rank_from_scores(signed_scores)
            rank_signed = primary_signed + [
                d for d in rank_from_scores(s_scores) if d not in set(primary_signed)
            ]

            # Anchor accuracy: by convention, on causal_relative the anchor doc
            # ends with `_a`. Track for diagnostic.
            if name == "causal_relative":
                expected_anchor = qid.replace("cr_q_", "cr_") + "_a"
                if anchor_did == expected_anchor:
                    n_anchor_correct += 1
                if len(causal_examples) < 15:
                    causal_examples.append(
                        {
                            "qid": qid,
                            "query": q["text"],
                            "phrase": ar.get("phrase"),
                            "cue": ar.get("cue"),
                            "direction": ar.get("direction"),
                            "anchor_did": anchor_did,
                            "expected_anchor": expected_anchor,
                            "sim": round(ar["sim"], 3),
                            "anchor_correct": anchor_did == expected_anchor,
                        }
                    )
        else:
            rank_mask = rank_baseline
            rank_signed = rank_baseline
            if causal and len(causal_examples) < 15 and name == "causal_relative":
                causal_examples.append(
                    {
                        "qid": qid,
                        "query": q["text"],
                        "phrase": causal_info[qid]["phrase"],
                        "cue": causal_info[qid]["cue"],
                        "direction": causal_info[qid]["direction"],
                        "resolved": False,
                    }
                )

        results.append(
            {
                "qid": qid,
                "qtext": q.get("text", "")[:200],
                "gold": list(gold_set),
                "causal": causal,
                "anchor_resolved": bool(ar and ar.get("resolved")),
                "rerank_only": hit_rank(rerank_only_rank, gold_set),
                "fuse_T_R_recAdd": hit_rank(rank_baseline, gold_set),
                "causal_mask": hit_rank(rank_mask, gold_set),
                "causal_signed": hit_rank(rank_signed, gold_set),
            }
        )

    return aggregate(
        results, name, n_causal, n_resolved, n_anchor_correct, causal_examples
    )


def aggregate(results, label, n_causal, n_resolved, n_anchor_correct, causal_examples):
    n = len(results)
    out = {
        "label": label,
        "n": n,
        "n_causal": n_causal,
        "n_resolved": n_resolved,
        "n_anchor_correct": n_anchor_correct,
        "causal_examples": causal_examples,
        "per_q": results,
    }

    variants = ["rerank_only", "fuse_T_R_recAdd", "causal_mask", "causal_signed"]
    for var in variants:
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr_v = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[var] = {
            "R@1": r1 / n if n else 0.0,
            "R@5": r5 / n if n else 0.0,
            "MRR": mrr_v,
            "r1_count": r1,
            "r5_count": r5,
        }

    print(
        f"  n={n}  causal={n_causal}  resolved={n_resolved}  anchor_correct={n_anchor_correct}",
        flush=True,
    )
    for var in variants:
        d = out[var]
        print(
            f"  {var:22s}  R@1={d['R@1']:.3f} ({d['r1_count']}/{n})  "
            f"R@5={d['R@5']:.3f} ({d['r5_count']}/{n})  MRR={d['MRR']:.3f}",
            flush=True,
        )
    return out


# -----------------------------------------------------------------------------
# MD writer
# -----------------------------------------------------------------------------
def write_md(report: dict, path: Path):
    benches = report["benches"]
    lines = []
    lines.append("# T_causal — 2-step plan for causal_relative queries\n")
    cr = benches.get("causal_relative", {})
    if cr and "error" not in cr:
        b = cr["fuse_T_R_recAdd"]["R@1"]
        m = cr["causal_mask"]["R@1"]
        s = cr["causal_signed"]["R@1"]
        lines.append("## Headline R@1 on causal_relative\n")
        lines.append(
            f"- baseline (fuse_T_R + recAdd): **{b:.3f}** ({cr['fuse_T_R_recAdd']['r1_count']}/{cr['n']})"
        )
        lines.append(
            f"- causal_mask:                  **{m:.3f}** ({cr['causal_mask']['r1_count']}/{cr['n']})"
        )
        lines.append(
            f"- causal_signed:                **{s:.3f}** ({cr['causal_signed']['r1_count']}/{cr['n']})"
        )
        lines.append("")
        lines.append(
            f"Anchor resolution: {cr['n_anchor_correct']}/{cr['n_causal']} "
            f"causal queries resolved to the correct `_a` doc "
            f"({cr['n_resolved']}/{cr['n_causal']} resolved at all).\n"
        )

    lines.append("## R@1 by benchmark (regression check)\n")
    lines.append(
        "| Benchmark | n | n_causal | n_resolved | rerank_only | fuse_T_R+recAdd | causal_mask | causal_signed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            lines.append(f"| {name} | err | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {b['n']} | {b['n_causal']} | {b['n_resolved']} | "
            f"{b['rerank_only']['R@1']:.3f} | {b['fuse_T_R_recAdd']['R@1']:.3f} | "
            f"{b['causal_mask']['R@1']:.3f} | {b['causal_signed']['R@1']:.3f} |"
        )
    lines.append("")

    lines.append("## R@5 by benchmark\n")
    lines.append(
        "| Benchmark | n | rerank_only | fuse_T_R+recAdd | causal_mask | causal_signed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, b in benches.items():
        if "error" in b:
            continue
        lines.append(
            f"| {name} | {b['n']} | {b['rerank_only']['R@5']:.3f} | "
            f"{b['fuse_T_R_recAdd']['R@5']:.3f} | "
            f"{b['causal_mask']['R@5']:.3f} | {b['causal_signed']['R@5']:.3f} |"
        )
    lines.append("")

    if cr and "causal_examples" in cr:
        lines.append("## Causal anchor resolutions on causal_relative\n")
        lines.append("| qid | cue | phrase | dir | anchor doc | expected | sim | OK |")
        lines.append("|---|---|---|---|---|---|---:|---:|")
        for ex in cr.get("causal_examples", []):
            if not ex.get("resolved", True):
                lines.append(
                    f"| {ex['qid']} | {ex.get('cue', '')} | `{ex.get('phrase', '')}` | {ex.get('direction', '')} | UNRESOLVED | - | - | - |"
                )
                continue
            lines.append(
                f"| {ex['qid']} | {ex['cue']} | `{ex['phrase']}` | {ex['direction']} | "
                f"{ex['anchor_did']} | {ex['expected_anchor']} | {ex['sim']:.3f} | "
                f"{'Y' if ex['anchor_correct'] else 'N'} |"
            )
        lines.append("")

    lines.append("## Per-query breakdown (causal_relative)\n")
    if cr and "per_q" in cr:
        lines.append(
            "| qid | causal | anchor_resolved | baseline | mask | signed | query |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for r in cr["per_q"]:

            def fmt(x):
                return "-" if x is None else str(x)

            lines.append(
                f"| {r['qid']} | {int(r['causal'])} | {int(r['anchor_resolved'])} | "
                f"{fmt(r['fuse_T_R_recAdd'])} | {fmt(r['causal_mask'])} | "
                f"{fmt(r['causal_signed'])} | `{r['qtext']}` |"
            )

    path.write_text("\n".join(lines))


async def main():
    print("Loading cross-encoder...", flush=True)
    from memmachine_server.common.reranker.cross_encoder_reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerParams,
    )
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    reranker = CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=ce,
            max_input_length=512,
        )
    )

    benches_main = [
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
            "v7l-hard_bench",
        ),
        (
            "temporal_essential",
            "temporal_essential_docs.jsonl",
            "temporal_essential_queries.jsonl",
            "temporal_essential_gold.jsonl",
            "v7l-temporal_essential",
        ),
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
            "v7l-tempreason_small",
        ),
        (
            "conjunctive_temporal",
            "edge_conjunctive_temporal_docs.jsonl",
            "edge_conjunctive_temporal_queries.jsonl",
            "edge_conjunctive_temporal_gold.jsonl",
            "edge-conjunctive_temporal",
        ),
        (
            "multi_te_doc",
            "edge_multi_te_doc_docs.jsonl",
            "edge_multi_te_doc_queries.jsonl",
            "edge_multi_te_doc_gold.jsonl",
            "edge-multi_te_doc",
        ),
        (
            "relative_time",
            "edge_relative_time_docs.jsonl",
            "edge_relative_time_queries.jsonl",
            "edge_relative_time_gold.jsonl",
            "edge-relative_time",
        ),
        (
            "era_refs",
            "edge_era_refs_docs.jsonl",
            "edge_era_refs_queries.jsonl",
            "edge_era_refs_gold.jsonl",
            "edge-era_refs",
        ),
        (
            "latest_recent",
            "latest_recent_docs.jsonl",
            "latest_recent_queries.jsonl",
            "latest_recent_gold.jsonl",
            "edge-latest_recent",
        ),
        (
            "open_ended_date",
            "open_ended_date_docs.jsonl",
            "open_ended_date_queries.jsonl",
            "open_ended_date_gold.jsonl",
            "edge-open_ended_date",
        ),
        (
            "causal_relative",
            "causal_relative_docs.jsonl",
            "causal_relative_queries.jsonl",
            "causal_relative_gold.jsonl",
            "edge-causal_relative",
        ),
        (
            "negation_temporal",
            "negation_temporal_docs.jsonl",
            "negation_temporal_queries.jsonl",
            "negation_temporal_gold.jsonl",
            "edge-negation_temporal",
        ),
    ]

    out = {"benches": {}}
    for name, dp, qp, gp, cache_label in benches_main:
        if not (DATA_DIR / dp).exists():
            alt = f"edge_{dp}"
            if (DATA_DIR / alt).exists():
                dp = alt
        if not (DATA_DIR / qp).exists():
            alt = f"edge_{qp}"
            if (DATA_DIR / alt).exists():
                qp = alt
        if not (DATA_DIR / gp).exists():
            alt = f"edge_{gp}"
            if (DATA_DIR / alt).exists():
                gp = alt
        if not (DATA_DIR / dp).exists():
            print(f"  [{name}] missing {dp} - skipping", flush=True)
            continue
        try:
            agg = await run_bench(name, dp, qp, gp, cache_label, reranker)
            out["benches"][name] = agg
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][name] = {"error": str(e), "n": 0}

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_causal.json"
    json_safe = {"benches": {}}
    for k, v in out["benches"].items():
        if "error" in v:
            json_safe["benches"][k] = v
            continue
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        json_safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "T_causal.md"
    write_md(out, md_path)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
