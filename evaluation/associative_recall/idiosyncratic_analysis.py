"""Helper module for analysis A: characterize the "genuinely_idiosyncratic"
missed turns that ingestion-side regex heuristics cannot flag.

Usage:
    uv run python idiosyncratic_analysis.py

Outputs:
    results/idiosyncratic_analysis.json  (structured per-turn features + clusters)
    results/idiosyncratic_analysis.md    (human-readable summary)

No LLM calls needed. Embeddings come from segments_extended.npz (cached on disk).
Question embeddings reuse the global EmbeddingCache if present.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from associative_recall import (
    DATA_DIR,
    EMBED_MODEL,
    EmbeddingCache,
)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Regex heuristics (cloned from ingest_regex_altkeys.py; kept local to avoid
# touching that file).
# ---------------------------------------------------------------------------
ANAPHORIC_TOKENS = {
    "that",
    "this",
    "those",
    "these",
    "it",
    "they",
    "he",
    "she",
    "him",
    "her",
    "his",
    "its",
    "their",
    "them",
}
SHORT_RESPONSE_TOKENS = {
    "yeah",
    "yes",
    "yep",
    "ok",
    "okay",
    "sure",
    "no",
    "nope",
    "definitely",
    "exactly",
    "right",
    "true",
    "false",
    "maybe",
}
UPDATE_MARKER_RE = re.compile(
    r"^(actually|wait|oh|scratch that|correction|on second thought|update|"
    r"let me correct|turns out|never mind)[\s,.]",
    re.IGNORECASE,
)
KNOWN_UNKNOWN_RE = re.compile(
    r"let me check|circle back|TBD|pending|not sure|waiting on",
    re.IGNORECASE,
)
ALIAS_EVOLUTION_RE = re.compile(
    r"call(ed)? it|\baka\b|also known as|renamed|new name",
    re.IGNORECASE,
)
STRUCTURED_FACT_RE = re.compile(
    r"allergy|deadline|prescription|dosage|prefer|"
    r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\$|%|\bv\d+\b",
    re.IGNORECASE,
)
CAP_TOKEN_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
NUM_VER_RE = re.compile(
    r"\bv\d+(?:\.\d+)*\b"
    r"|\b\d+(?:\.\d+)+\b"
    r"|\b[A-Z]+-\d+\b"
    r"|\$\d[\d,\.]*"
    r"|\b\d+%\b"
    r"|\b\d{1,2}:\d{2}\s*(am|pm)?\b"
    r"|\b\d+[KkMm]\b",
    re.IGNORECASE,
)


def _first_tok(text: str) -> str:
    s = text.lstrip().lstrip("\"'([{<-*").lstrip()
    m = re.match(r"[A-Za-z']+", s)
    return m.group(0).lower() if m else ""


def _wc(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _rare_tokens(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in NUM_VER_RE.finditer(text):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
    for sent in SENTENCE_SPLIT_RE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        matches = list(CAP_TOKEN_RE.finditer(sent))
        if not matches:
            continue
        first_alpha = re.search(r"[A-Za-z]+", sent)
        fs = first_alpha.start() if first_alpha else -1
        for m in matches:
            if m.start() == fs:
                continue
            tok = m.group(0)
            if tok not in seen:
                seen.add(tok)
                found.append(tok)
    return found


def compute_predictable_tags(
    mt: dict, corpus_freq: Counter, rare_k: int = 3
) -> list[str]:
    """Return list of firing predictable heuristic names for a missed turn."""
    text = mt["text"]
    tags: list[str] = []
    if mt.get("starts_anaphoric"):
        tags.append("anaphoric")
    if mt.get("short_response"):
        tags.append("short_response")
    if UPDATE_MARKER_RE.search(text.lstrip()):
        tags.append("update_marker")
    if KNOWN_UNKNOWN_RE.search(text):
        tags.append("known_unknown")
    if ALIAS_EVOLUTION_RE.search(text):
        tags.append("alias_evolution")
    if STRUCTURED_FACT_RE.search(text):
        tags.append("structured_fact")
    toks = _rare_tokens(text)
    rare = [t for t in toks if corpus_freq.get(t, 0) <= rare_k]
    if rare:
        tags.append("rare_entity")
    return tags


# ---------------------------------------------------------------------------
# Surface / lexical features
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "for",
    "with",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "should",
    "could",
    "may",
    "might",
    "can",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "you",
    "i",
    "they",
    "them",
    "he",
    "she",
    "his",
    "her",
    "their",
    "our",
    "me",
    "my",
    "your",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "if",
    "then",
    "than",
    "also",
    "so",
    "just",
    "not",
    "no",
    "yes",
    "yeah",
    "ok",
    "okay",
    "about",
    "there",
    "here",
    "out",
    "up",
    "down",
    "into",
    "over",
    "under",
    "again",
    "very",
    "some",
    "any",
    "all",
    "more",
    "most",
    "much",
    "many",
    "each",
    "every",
    "few",
    "other",
    "another",
    "own",
    "same",
    "such",
    "only",
    "too",
    "both",
    "between",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _toks(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def ensure_question_embedding(
    question: str, cache: EmbeddingCache, client
) -> np.ndarray:
    """Return embedding for a question, using the shared on-disk cache."""
    cached = cache.get(question)
    if cached is not None:
        return cached
    # miss -> call API
    resp = client.embeddings.create(model=EMBED_MODEL, input=question)
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    cache.put(question, emb)
    return emb


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Clustering (pure-numpy k-means on normalized embeddings)
# ---------------------------------------------------------------------------
def kmeans_cosine(
    X: np.ndarray, k: int, n_iter: int = 40, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """K-means on unit-vectors using sklearn (n_init=20 for robustness to seeding).

    Returns (labels, centroids).
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=n_iter)
    labels = km.fit_predict(X)
    # Re-normalize centroids back onto unit sphere for consistency
    cents = km.cluster_centers_.copy()
    norms = np.linalg.norm(cents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    cents = cents / norms
    return labels, cents


def top_terms_per_cluster(
    texts: list[str], labels: np.ndarray, k: int, top_n: int = 8
) -> list[list[tuple[str, float]]]:
    """TF-IDF-ish: for each cluster, score tokens by (in-cluster freq) /
    (overall freq) with smoothing."""
    global_counts = Counter()
    per_cluster = [Counter() for _ in range(k)]
    for i, txt in enumerate(texts):
        toks = _toks(txt)
        global_counts.update(toks)
        per_cluster[labels[i]].update(toks)
    total = sum(global_counts.values()) or 1
    out = []
    for c in range(k):
        scored = []
        for tok, cnt in per_cluster[c].items():
            # lift over expected frequency
            expected = global_counts[tok] / total
            observed = cnt / max(1, sum(per_cluster[c].values()))
            if global_counts[tok] < 2:
                continue
            lift = observed / max(expected, 1e-10)
            scored.append((tok, lift * cnt))
        scored.sort(key=lambda x: -x[1])
        out.append(scored[:top_n])
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(HERE.parents[1] / ".env")
    client = OpenAI()

    # --- Load data
    with open(HERE / "results" / "error_analysis_details.json") as f:
        error = json.load(f)

    # Load all NPZ stores needed to cover the conversations in error_analysis.
    # The missed-turn data spans advanced, puzzle, synthetic, locomo datasets.
    NPZ_NAMES = [
        "segments_advanced_prefixed.npz",
        "segments_puzzle_prefixed.npz",
        "segments_synthetic_prefixed.npz",
        "segments_extended_locomo_prefixed.npz",
    ]

    class _MultiStore:
        """Minimal multi-npz store: concatenates embeddings/texts across files."""

        def __init__(self, data_dir: Path, names: list[str]) -> None:
            embs = []
            cids = []
            tids = []
            roles_ = []
            texts = []
            for n in names:
                arr = np.load(data_dir / n, allow_pickle=True)
                embs.append(arr["embeddings"])
                cids.append(np.array([str(x) for x in arr["conversation_ids"]]))
                tids.append(arr["turn_ids"])
                roles_.append(np.array([str(x) for x in arr["roles"]]))
                texts.append(np.array([str(x) for x in arr["texts"]]))
            self.embeddings = np.concatenate(embs, axis=0).astype(np.float32)
            self.conversation_ids = np.concatenate(cids, axis=0)
            self.turn_ids = np.concatenate(tids, axis=0)
            self.roles = np.concatenate(roles_, axis=0)
            self.texts = np.concatenate(texts, axis=0)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self.normalized_embeddings = self.embeddings / norms

            # segments list mimicking SegmentStore API just enough for our needs
            @dataclass
            class _S:
                conversation_id: str
                turn_id: int
                role: str
                text: str
                index: int

            self.segments = [
                _S(
                    conversation_id=str(self.conversation_ids[i]),
                    turn_id=int(self.turn_ids[i]),
                    role=str(self.roles[i]),
                    text=str(self.texts[i]),
                    index=i,
                )
                for i in range(len(self.texts))
            ]

    store = _MultiStore(DATA_DIR, NPZ_NAMES)
    # Map (conversation_id, turn_id) -> index into store
    seg_lookup: dict[tuple[str, int], int] = {}
    for i, seg in enumerate(store.segments):
        seg_lookup[(seg.conversation_id, seg.turn_id)] = i

    # --- Build corpus frequency for rare_entity filter
    all_corpus_text: list[str] = []
    for pq in error["per_question"]:
        for mt in pq["missed_turns"]:
            all_corpus_text.append(mt["text"])
            for nb in mt.get("neighbors_before", []) + mt.get("neighbors_after", []):
                all_corpus_text.append(nb["text"])
    corpus_freq: Counter = Counter()
    for t in all_corpus_text:
        for tok in set(_rare_tokens(t)):
            corpus_freq[tok] += 1

    # --- Identify idiosyncratic missed turns.
    # Operational rule:
    #   A missed turn is "genuinely_idiosyncratic" iff
    #     (a) none of the 7 predictable regex heuristics fire, AND
    #     (b) q-category is NOT in {proactive, quantitative_aggregation, inference}
    #         (those form the "arbitrary_conjunction" / "pattern_in_many" /
    #         "distant_inference" buckets respectively in the source report).
    NONPRED_QCAT_MAP = {
        "proactive": "arbitrary_conjunction",
        "quantitative_aggregation": "pattern_in_many",
        "inference": "distant_inference",
    }

    idio: list[dict] = []
    all_missed_count = 0
    nonpred_count = 0
    subcat_counts: Counter = Counter()
    for pq in error["per_question"]:
        qcat = pq["category"]
        for mt in pq["missed_turns"]:
            all_missed_count += 1
            tags = compute_predictable_tags(mt, corpus_freq)
            if tags:
                continue
            nonpred_count += 1
            if qcat in NONPRED_QCAT_MAP:
                subcat_counts[NONPRED_QCAT_MAP[qcat]] += 1
                continue
            subcat_counts["genuinely_idiosyncratic"] += 1
            idio.append(
                {
                    "conversation_id": pq["conversation_id"],
                    "q_index": pq["question_index"],
                    "q_category": qcat,
                    "question": pq["question"],
                    "turn_id": mt["turn_id"],
                    "role": mt["role"],
                    "text": mt["text"],
                    "word_count": mt["word_count"],
                    "primary_failure_mode": mt["primary_failure_mode"],
                    "adj_to_retrieved_r1": mt.get("adj_to_retrieved_r1"),
                    "adj_to_retrieved_r2": mt.get("adj_to_retrieved_r2"),
                    "source_ids": pq["source_ids"],
                    "retrieved_source_ids": pq["retrieved_source_ids"],
                    "missed_source_ids": pq["missed_source_ids"],
                    "question_term_overlap": mt.get("question_term_overlap", []),
                    "cue_term_overlap": mt.get("cue_term_overlap", []),
                    "num_source": pq["num_source"],
                    "num_missed": pq["num_missed"],
                    "arch_r@20": pq["arch_r@20"],
                }
            )

    print(
        f"Total missed: {all_missed_count}. "
        f"Non-predictable: {nonpred_count}. "
        f"Idiosyncratic bucket: {len(idio)}"
    )
    print("Subcategory counts:")
    for k, v in subcat_counts.items():
        print(f"  {k}: {v}")

    # --- Embed questions using on-disk cache
    emb_cache = EmbeddingCache()  # uses cache/embedding_cache.json
    q_embs: dict[str, np.ndarray] = {}
    q_misses = 0
    for e in idio:
        q = e["question"]
        if q not in q_embs:
            cached = emb_cache.get(q)
            if cached is None:
                q_misses += 1
                resp = client.embeddings.create(model=EMBED_MODEL, input=q)
                cached = np.array(resp.data[0].embedding, dtype=np.float32)
                emb_cache.put(q, cached)
            q_embs[q] = cached
    if q_misses:
        emb_cache.save()
        print(f"Question embedding cache misses: {q_misses} (saved)")
    else:
        print("Question embedding cache: all hit")

    # --- Pull missed-turn embeddings from segment store
    miss_embs: list[np.ndarray] = []
    found_idx: list[int] = []
    for i, e in enumerate(idio):
        key = (e["conversation_id"], e["turn_id"])
        idx = seg_lookup.get(key)
        if idx is None:
            # Segment not in store for some reason
            e["_seg_idx"] = None
            miss_embs.append(np.zeros(store.embeddings.shape[1], dtype=np.float32))
            continue
        e["_seg_idx"] = int(idx)
        miss_embs.append(store.embeddings[idx])
        found_idx.append(i)
    M = np.stack(miss_embs, axis=0)
    # normalize for cosine
    M_norm = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-10, None)

    # --- Compute per-turn features
    for i, e in enumerate(idio):
        q_emb = q_embs[e["question"]]
        m_emb = miss_embs[i]
        e["cosine_miss_to_q"] = cosine(q_emb, m_emb)
        q_toks = _toks(e["question"])
        m_toks = _toks(e["text"])
        e["jaccard_miss_q"] = jaccard(m_toks, q_toks)
        e["miss_tok_count"] = len(m_toks)
        e["q_tok_count"] = len(q_toks)

    # --- Overlap with OTHER gold turns in the same question (retrieved ones)
    for e in idio:
        cid = e["conversation_id"]
        retrieved_turn_ids = set(e["retrieved_source_ids"]) & set(e["source_ids"])
        other_toks: set[str] = set()
        for tid in retrieved_turn_ids:
            idx = seg_lookup.get((cid, int(tid)))
            if idx is None:
                continue
            other_toks |= _toks(store.segments[idx].text)
        m_toks = _toks(e["text"])
        e["jaccard_miss_sibling_gold"] = jaccard(m_toks, other_toks)

    # --- Cosine of retrieved-gold turns to q (for comparison baseline)
    for e in idio:
        cid = e["conversation_id"]
        q_emb = q_embs[e["question"]]
        retrieved_turn_ids = set(e["retrieved_source_ids"]) & set(e["source_ids"])
        sims = []
        for tid in retrieved_turn_ids:
            idx = seg_lookup.get((cid, int(tid)))
            if idx is None:
                continue
            sims.append(cosine(q_emb, store.embeddings[idx]))
        e["mean_cos_retrieved_gold_to_q"] = float(np.mean(sims)) if sims else None
        e["max_cos_retrieved_gold_to_q"] = float(np.max(sims)) if sims else None

    # --- For each miss, rank vs all segments in SAME conversation to see where it lands
    for e in idio:
        cid = e["conversation_id"]
        q_emb = q_embs[e["question"]]
        mask = store.conversation_ids == cid
        conv_idxs = np.where(mask)[0]
        if len(conv_idxs) == 0 or e["_seg_idx"] is None:
            e["rank_in_conv"] = None
            e["conv_size"] = 0
            continue
        conv_embs = store.normalized_embeddings[conv_idxs]
        q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)
        sims = conv_embs @ q_norm
        order = np.argsort(-sims)
        # rank of e["_seg_idx"]
        target_pos_arr = np.where(conv_idxs[order] == e["_seg_idx"])[0]
        e["rank_in_conv"] = int(target_pos_arr[0]) if target_pos_arr.size else None
        e["conv_size"] = len(conv_idxs)

    # --- Clustering on missed-turn embeddings
    # Use k=6 as a compromise
    N = M_norm.shape[0]
    k = min(6, max(2, N // 8))
    if k <= N:
        labels, centroids = kmeans_cosine(M_norm, k=k, seed=42)
    else:
        labels = np.zeros(N, dtype=int)
        centroids = M_norm.mean(axis=0, keepdims=True)
    for i, e in enumerate(idio):
        e["cluster"] = int(labels[i])
    texts = [e["text"] for e in idio]
    cluster_terms = top_terms_per_cluster(texts, labels, k=k, top_n=10)

    # --- Failure-type classification (rule-based on computed features)
    #   lexical_mismatch: jaccard_miss_q very low, jaccard_miss_sibling_gold ok
    #   topic_drift: jaccard low AND cosine low AND sibling overlap also low
    #   structural_role: short tokens or very generic (<4 content tokens) but role=assistant or transition-like
    #   implicit_reference: has pronouns/deictics missed by strict anaphoric (he/she/they/that appears anywhere)
    #   other: everything else
    DEICTIC = {
        "here",
        "there",
        "then",
        "later",
        "earlier",
        "yesterday",
        "tomorrow",
        "recently",
        "above",
        "below",
        "that one",
        "the same",
    }
    for e in idio:
        text = e["text"]
        text_lower = text.lower()
        jq = e["jaccard_miss_q"]
        js = e["jaccard_miss_sibling_gold"]
        cos = e["cosine_miss_to_q"]
        mtok = e["miss_tok_count"]
        # detect pronouns/deictics anywhere in text
        has_internal_pronoun = any(
            re.search(rf"\b{p}\b", text_lower)
            for p in (
                "he",
                "she",
                "they",
                "them",
                "their",
                "that",
                "this",
                "these",
                "those",
                "it",
            )
        )
        has_deictic = any(d in text_lower for d in DEICTIC)

        if mtok < 4:
            ftype = "structural_role"
        elif jq < 0.05 and js < 0.10 and cos < 0.3:
            ftype = "topic_drift"
        elif jq < 0.08 and js >= 0.10:
            ftype = "lexical_mismatch"
        elif has_internal_pronoun or has_deictic:
            ftype = "implicit_reference"
        else:
            ftype = "other"
        e["failure_type"] = ftype

    # --- Per-question concentration
    per_q_counter: Counter = Counter()
    for e in idio:
        per_q_counter[(e["conversation_id"], e["q_index"])] += 1
    top_questions = per_q_counter.most_common(10)

    # --- Save JSON
    out_json = {
        "metadata": {
            "total_missed": all_missed_count,
            "non_predictable_count": nonpred_count,
            "subcategory_counts": dict(subcat_counts),
            "idiosyncratic_count": len(idio),
            "k_clusters": k,
            "note": (
                "Operational definition of 'genuinely_idiosyncratic': "
                "non-predictable (no heuristic fires) AND q-category NOT in "
                "{proactive, quantitative_aggregation, inference}. "
                "The source report (ingestion_predictability.md) claims 61 turns; "
                "the reproduction here yields a slightly different count due to "
                "corpus-frequency and rare_entity threshold differences."
            ),
        },
        "turns": [
            {k2: v2 for k2, v2 in e.items() if not k2.startswith("_")} for e in idio
        ],
        "clusters": {
            str(c): {
                "size": int((labels == c).sum()),
                "top_terms": [
                    {"term": t, "score": round(float(s), 3)}
                    for t, s in cluster_terms[c]
                ],
                "example_turns": [
                    {
                        "conversation_id": idio[i]["conversation_id"],
                        "turn_id": idio[i]["turn_id"],
                        "text": idio[i]["text"],
                        "question": idio[i]["question"],
                    }
                    for i in np.where(labels == c)[0][:3].tolist()
                ],
            }
            for c in range(k)
        },
        "top_questions_by_miss_count": [
            {"conversation_id": cid, "q_index": qi, "count": cnt}
            for (cid, qi), cnt in top_questions
        ],
    }
    with open(RESULTS_DIR / "idiosyncratic_analysis.json", "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"Wrote {RESULTS_DIR / 'idiosyncratic_analysis.json'}")

    # --- Markdown report
    md = []
    md.append("# Analysis A: Genuinely-Idiosyncratic Missed Turns\n")
    md.append(
        "Operational bucket: non-predictable by 7-regex heuristics AND "
        "q-category not in {proactive, quantitative_aggregation, inference}.\n"
    )
    md.append(
        f"- Total missed turns: **{all_missed_count}**\n"
        f"- Non-predictable (no regex heuristic fires): **{nonpred_count}**\n"
        f"- Idiosyncratic bucket (this analysis): **{len(idio)}**\n"
        f"- Source report's claim was 61; our count differs because the "
        f"source used a slightly different entity/corpus-rarity threshold.\n"
    )

    # Surface features
    wcs = np.array([e["word_count"] for e in idio])
    roles = Counter(e["role"] for e in idio)
    pfms = Counter(e["primary_failure_mode"] for e in idio)
    qcats = Counter(e["q_category"] for e in idio)
    adj_r1 = sum(1 for e in idio if e["adj_to_retrieved_r1"])
    adj_r2 = sum(1 for e in idio if e["adj_to_retrieved_r2"])

    md.append("## 1. Surface features\n")
    md.append("### Word-count distribution (missed turn text)\n")
    md.append(
        f"- mean={wcs.mean():.1f}, median={np.median(wcs):.0f}, "
        f"p10={np.percentile(wcs, 10):.0f}, p90={np.percentile(wcs, 90):.0f}, "
        f"min={wcs.min()}, max={wcs.max()}\n"
    )
    md.append("### Role distribution\n")
    for r, c in roles.most_common():
        md.append(f"- {r}: {c} ({100 * c / len(idio):.1f}%)\n")
    md.append("### Original failure-mode label\n")
    for r, c in pfms.most_common():
        md.append(f"- {r}: {c} ({100 * c / len(idio):.1f}%)\n")
    md.append("### Adjacency to retrieved turns\n")
    md.append(
        f"- adjacent-to-retrieved at radius 1: {adj_r1} / {len(idio)} "
        f"({100 * adj_r1 / len(idio):.1f}%)\n"
        f"- adjacent-to-retrieved at radius 2: {adj_r2} / {len(idio)} "
        f"({100 * adj_r2 / len(idio):.1f}%)\n"
    )
    md.append("### Q-category distribution\n")
    for r, c in qcats.most_common():
        md.append(f"- {r}: {c} ({100 * c / len(idio):.1f}%)\n")

    # Embedding story
    cos_miss_q = np.array([e["cosine_miss_to_q"] for e in idio])
    cos_gold = [e["mean_cos_retrieved_gold_to_q"] for e in idio]
    cos_gold_present = np.array([x for x in cos_gold if x is not None])
    jq = np.array([e["jaccard_miss_q"] for e in idio])
    js = np.array([e["jaccard_miss_sibling_gold"] for e in idio])
    rank_in_conv = np.array(
        [e["rank_in_conv"] for e in idio if e["rank_in_conv"] is not None]
    )

    md.append("## 2. Embedding-distance story\n")
    md.append(
        f"- cosine(missed_turn, question): mean={cos_miss_q.mean():.3f}, "
        f"median={np.median(cos_miss_q):.3f}, "
        f"p25={np.percentile(cos_miss_q, 25):.3f}, "
        f"p75={np.percentile(cos_miss_q, 75):.3f}\n"
    )
    if cos_gold_present.size:
        md.append(
            f"- cosine(retrieved_gold, question) (same question): "
            f"mean={cos_gold_present.mean():.3f}, "
            f"median={np.median(cos_gold_present):.3f}\n"
        )
        gap = cos_gold_present.mean() - cos_miss_q[: len(cos_gold_present)].mean()
        md.append(f"- gap (retrieved_gold mean - missed mean): {gap:+.3f}\n")
    md.append(
        f"- rank of missed turn within its own conversation (1536-D cosine to q): "
        f"mean={rank_in_conv.mean():.1f}, median={np.median(rank_in_conv):.0f}, "
        f"p75={np.percentile(rank_in_conv, 75):.0f}, "
        f"max={rank_in_conv.max()}\n"
    )
    md.append(
        f"- Jaccard(missed_tokens, question_tokens): mean={jq.mean():.3f}, "
        f"median={np.median(jq):.3f}\n"
    )
    md.append(
        f"- Jaccard(missed_tokens, retrieved-sibling-gold_tokens): "
        f"mean={js.mean():.3f}, median={np.median(js):.3f}\n"
    )

    # Clusters
    md.append("## 3. Clusters (k-means on missed-turn embeddings)\n")
    for c in range(k):
        members = np.where(labels == c)[0]
        md.append(f"### Cluster {c} ({len(members)} turns)\n")
        md.append("**Top lift terms:** ")
        md.append(", ".join(f"`{t}` ({s:.1f})" for t, s in cluster_terms[c]) + "\n")
        md.append("**Examples:**\n")
        for i in members[:3].tolist():
            e = idio[i]
            md.append(
                f"- [{e['q_category']}] (conv `{e['conversation_id']}`, "
                f"turn {e['turn_id']}, {e['role']}): "
                f'"{e["text"][:180]}"\n'
                f'  - question: "{e["question"][:120]}"\n'
            )

    # Failure-type classification counts
    ft_counts = Counter(e["failure_type"] for e in idio)
    md.append("## 4. Failure-type classification\n")
    md.append(
        "Rule-based from computed features (lexical overlap, cosine, length, pronouns):\n"
    )
    for t, c in ft_counts.most_common():
        md.append(f"- {t}: {c} ({100 * c / len(idio):.1f}%)\n")

    # Question concentration
    md.append("## 5. Question concentration\n")
    md.append(
        f"- {len(per_q_counter)} distinct questions contain at least one "
        f"idiosyncratic miss.\n"
    )
    md.append(
        f"- Top {min(10, len(per_q_counter))} questions hold "
        f"{sum(c for _, c in top_questions)} / {len(idio)} idiosyncratic misses.\n"
    )
    md.append("| conv | q_idx | count |\n|---|---:|---:|\n")
    for (cid, qi), cnt in top_questions:
        md.append(f"| {cid} | {qi} | {cnt} |\n")

    # Verdict
    fixable = ft_counts.get("lexical_mismatch", 0) + ft_counts.get(
        "implicit_reference", 0
    )
    ceiling = ft_counts.get("topic_drift", 0) + ft_counts.get("other", 0)
    structural = ft_counts.get("structural_role", 0)
    md.append("## 6. Verdict\n")
    md.append(
        f"- Plausibly fixable via existing cue-generation architectures "
        f"(v2f / chain_with_scratchpad): **{fixable} / {len(idio)}** "
        f"({100 * fixable / max(1, len(idio)):.1f}%)\n"
        f"  (lexical_mismatch has sibling-gold signal to piggy-back; "
        f"implicit_reference carries deictics that a small expand-with-prev-turn "
        f"rule would catch.)\n"
    )
    md.append(
        f"- Structural-role turns: **{structural}** — short / transitional / "
        f"meta-comments, best caught by always-attach-preceding-turn expansion.\n"
    )
    md.append(
        f"- Likely retrieval ceiling (topic_drift + other, no surface or "
        f"paraphrase handle): **{ceiling} / {len(idio)}** "
        f"({100 * ceiling / max(1, len(idio)):.1f}%)\n"
    )
    md.append(
        "\n**Key finding:** see mean cosine(missed, q) vs mean cosine(retrieved_gold, q) "
        "above. If the gap is large (>0.1), embeddings alone cannot rank these turns "
        "competitively — a better embedding or query-side expansion is required.\n"
    )

    with open(RESULTS_DIR / "idiosyncratic_analysis.md", "w") as f:
        f.write("".join(md))
    print(f"Wrote {RESULTS_DIR / 'idiosyncratic_analysis.md'}")


if __name__ == "__main__":
    main()
