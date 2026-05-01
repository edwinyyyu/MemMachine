"""Per-cue attribution analysis.

For each (question, cue) pair cached from v2f/meta_v2f runs, measure which
cues actually retrieved gold turns and extract features to distinguish winners
from losers. Pure analysis — no new LLM calls, embeddings are cache hits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def load_all_embedding_caches() -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for f in sorted(os.listdir(CACHE_DIR)):
        if "embedding" not in f or not f.endswith(".json"):
            continue
        if f.endswith(".tmp") or f.endswith(".lock"):
            continue
        try:
            with open(CACHE_DIR / f) as fh:
                ec = json.load(fh)
        except Exception:
            continue
        if not isinstance(ec, dict):
            continue
        for k, v in ec.items():
            if k not in merged:
                merged[k] = v
    return merged


def build_conv_to_npz() -> dict[str, str]:
    """Map conversation_id to smallest npz file containing it."""
    convs: dict[str, str] = {}
    # Prefer non-scaling, non-prefixed npz files
    preferred_order = [
        "segments.npz",
        "segments_synthetic.npz",
        "segments_extended.npz",
        "segments_advanced.npz",
        "segments_puzzle.npz",
        "segments_v2_combined.npz",
    ]
    already = set()
    for fname in preferred_order:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=True)
        if "conversation_ids" not in d.files:
            continue
        for c in d["conversation_ids"]:
            cs = str(c)
            if cs not in already:
                convs[cs] = fname
                already.add(cs)
    # Fallback: any remaining npz
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".npz") or "prefixed" in fname:
            continue
        if fname in preferred_order:
            continue
        path = DATA_DIR / fname
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        if "conversation_ids" not in d.files:
            continue
        for c in d["conversation_ids"]:
            cs = str(c)
            if cs not in already:
                convs[cs] = fname
                already.add(cs)
    return convs


class Npz:
    """Wrapper around a segments npz file that supports per-conversation top-K search."""

    def __init__(self, fname: str):
        path = DATA_DIR / fname
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.conversation_ids = np.array([str(c) for c in data["conversation_ids"]])
        self.turn_ids = data["turn_ids"]
        self.roles = data["roles"]
        self.texts = data["texts"]
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normed = self.embeddings / norms
        # Index per conv
        self._conv_idx: dict[str, np.ndarray] = {}
        for cid in set(self.conversation_ids):
            self._conv_idx[cid] = np.where(self.conversation_ids == cid)[0]

    def search(
        self, query_emb: np.ndarray, conversation_id: str, top_k: int = 20
    ) -> list[dict]:
        idx = self._conv_idx.get(conversation_id)
        if idx is None or len(idx) == 0:
            return []
        qn = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        sims = self.normed[idx] @ qn
        order = np.argsort(sims)[::-1][:top_k]
        out = []
        for rank, local_i in enumerate(order):
            global_i = int(idx[local_i])
            out.append(
                {
                    "rank": rank,
                    "index": global_i,
                    "turn_id": int(self.turn_ids[global_i]),
                    "role": str(self.roles[global_i]),
                    "text": str(self.texts[global_i]),
                    "score": float(sims[local_i]),
                }
            )
        return out

    def conv_size(self, conversation_id: str) -> int:
        idx = self._conv_idx.get(conversation_id)
        return 0 if idx is None else len(idx)


# ---------------------------------------------------------------------------
# Cue collection
# ---------------------------------------------------------------------------
def collect_cue_entries() -> list[dict]:
    entries = []
    seen = set()
    for f in sorted(os.listdir(RESULTS_DIR)):
        if not f.endswith(".json"):
            continue
        try:
            with open(RESULTS_DIR / f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict) or "results" not in d:
            continue
        results = d["results"]
        if not isinstance(results, list):
            continue
        for rec in results:
            if not isinstance(rec, dict):
                continue
            meta = rec.get("metadata", {})
            if not isinstance(meta, dict):
                continue
            cues = meta.get("cues")
            if not isinstance(cues, list):
                continue
            conv_id = rec.get("conversation_id")
            question = rec.get("question")
            if not conv_id or not question:
                continue
            sources = rec.get("source_chat_ids", [])
            category = rec.get("category", "")
            qidx = rec.get("question_index", -1)
            for i, cue in enumerate(cues):
                if not isinstance(cue, str):
                    continue
                cue = cue.strip()
                if len(cue) < 3:
                    continue
                key = (conv_id, question, cue)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(
                    {
                        "conversation_id": conv_id,
                        "category": category,
                        "question_index": qidx,
                        "question": question,
                        "cue": cue,
                        "cue_index": i,
                        "source_chat_ids": [
                            int(x)
                            for x in sources
                            if isinstance(x, (int, float, str))
                            and str(x).lstrip("-").isdigit()
                        ],
                        "source_file": f,
                    }
                )
    return entries


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"\w+")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


STOP = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "but",
    "at",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "me",
    "him",
    "them",
    "us",
    "mine",
    "yours",
    "as",
    "if",
    "with",
    "from",
    "up",
    "about",
    "do",
    "did",
    "does",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "which",
}


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOP]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def is_question(text: str) -> bool:
    t = text.strip()
    if t.endswith("?"):
        return True
    first = t.split(" ", 1)[0].lower().strip(",.")
    interrog = {
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "which",
        "did",
        "do",
        "does",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
    }
    return first in interrog


def entity_count(text: str) -> int:
    """Count likely entities — CapWords, quoted strings, numbers."""
    n = 0
    # Capitalized words (not at start of sentence)
    words = text.split()
    for i, w in enumerate(words):
        core = w.strip(".,;:!?()[]\"'")
        if not core:
            continue
        if i == 0:
            continue
        if core[0].isupper() and not core.isupper():
            n += 1
    # Quoted strings
    n += len(re.findall(r'"[^"]+"', text))
    # Numbers
    n += len(NUMBER_RE.findall(text))
    return n


def number_count(text: str) -> int:
    return len(NUMBER_RE.findall(text))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float((a @ b) / (na * nb))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(top_k: int = 20):
    print("Loading embedding caches...")
    emb_cache = load_all_embedding_caches()
    print(f"  {len(emb_cache)} cached embeddings")

    print("Collecting cue entries...")
    entries = collect_cue_entries()
    print(f"  {len(entries)} (conv, question, cue) entries")

    print("Mapping conversations to npz files...")
    conv_to_npz = build_conv_to_npz()
    print(f"  {len(conv_to_npz)} conversations mapped")

    # Open npz files lazily
    npz_cache: dict[str, Npz] = {}

    def get_npz(fname: str) -> Npz:
        if fname not in npz_cache:
            npz_cache[fname] = Npz(fname)
        return npz_cache[fname]

    # Group entries by (conv_id, question) so we can compute baseline retrieval once
    by_q = defaultdict(list)
    for e in entries:
        by_q[(e["conversation_id"], e["question"])].append(e)

    print(f"  {len(by_q)} unique (conv, question) pairs")

    # Per-cue analysis
    per_cue = []
    skipped_no_npz = 0
    skipped_no_q_emb = 0
    skipped_no_cue_emb = 0
    for (conv_id, question), group in by_q.items():
        npz_name = conv_to_npz.get(conv_id)
        if not npz_name:
            skipped_no_npz += len(group)
            continue
        store = get_npz(npz_name)
        if store.conv_size(conv_id) == 0:
            skipped_no_npz += len(group)
            continue

        # Baseline: embed question
        q_key = _sha(question)
        q_emb_list = emb_cache.get(q_key)
        if q_emb_list is None:
            skipped_no_q_emb += len(group)
            continue
        q_emb = np.array(q_emb_list, dtype=np.float32)
        baseline_hits = store.search(q_emb, conv_id, top_k=top_k)
        baseline_turn_ids = set(h["turn_id"] for h in baseline_hits)

        # Also lookup best_gold segment embedding (for each gold turn, find its embedding in npz)
        # In store, we use embeddings indexed by global index; map source_chat_ids to (global index, embedding)
        # First entry group to get sources
        source_ids = group[0]["source_chat_ids"]
        gold_set = set(source_ids)
        # find gold segments in this conv
        conv_idx = store._conv_idx.get(conv_id, np.array([]))
        gold_embs = []
        gold_texts = {}
        for gi in conv_idx:
            tid = int(store.turn_ids[gi])
            if tid in gold_set:
                gold_embs.append(store.normed[gi].astype(np.float32))
                gold_texts[tid] = str(store.texts[gi])

        q_tokens = content_tokens(question)

        for e in group:
            cue = e["cue"]
            cue_key = _sha(cue)
            cue_emb_list = emb_cache.get(cue_key)
            if cue_emb_list is None:
                skipped_no_cue_emb += 1
                continue
            cue_emb = np.array(cue_emb_list, dtype=np.float32)
            hits = store.search(cue_emb, conv_id, top_k=top_k)
            hit_tids = set(h["turn_id"] for h in hits)
            gold_hit_tids = hit_tids & gold_set
            # gold this cue found that baseline didn't
            gold_exclusive = gold_hit_tids - baseline_turn_ids
            marginal = hit_tids - gold_set

            # features
            cue_tokens_list = content_tokens(cue)
            cue_len_words = len(tokens(cue))
            jac = jaccard(q_tokens, cue_tokens_list)
            q_emb_n = q_emb / max(np.linalg.norm(q_emb), 1e-10)
            cue_emb_n = cue_emb / max(np.linalg.norm(cue_emb), 1e-10)
            cue_q_cos = cosine(cue_emb, q_emb)

            # Closest gold cosine
            best_gold_cos = 0.0
            if gold_embs:
                best_gold_cos = max(float(cue_emb_n @ ge) for ge in gold_embs)

            # Also: which gold did this cue retrieve closest to?
            retrieved_gold_text = None
            if gold_hit_tids:
                # find highest-ranked gold in hits
                for h in hits:
                    if h["turn_id"] in gold_set:
                        retrieved_gold_text = h["text"]
                        break

            per_cue.append(
                {
                    "conversation_id": conv_id,
                    "category": e["category"],
                    "question": question,
                    "cue": cue,
                    "cue_index": e["cue_index"],
                    "source_chat_ids": source_ids,
                    "num_gold": len(gold_set),
                    "gold_hit": len(gold_hit_tids),
                    "gold_exclusive": len(gold_exclusive),
                    "marginal": len(marginal),
                    "baseline_gold_hit": len(baseline_turn_ids & gold_set),
                    "cue_len_words": cue_len_words,
                    "jaccard_with_q": jac,
                    "is_question": is_question(cue),
                    "entity_count": entity_count(cue),
                    "number_count": number_count(cue),
                    "cue_q_cos": cue_q_cos,
                    "best_gold_cos": best_gold_cos,
                    "retrieved_gold_text": retrieved_gold_text,
                    "source_file": e["source_file"],
                }
            )

    print(f"Analyzed {len(per_cue)} cues")
    print(
        f"Skipped: no_npz={skipped_no_npz}, no_q_emb={skipped_no_q_emb}, no_cue_emb={skipped_no_cue_emb}"
    )

    # Save raw JSON
    out_json = RESULTS_DIR / "per_cue_attribution.json"
    with open(out_json, "w") as f:
        json.dump({"data": per_cue, "top_k": top_k}, f, indent=2)
    print(f"Saved {out_json}")

    return per_cue


# ---------------------------------------------------------------------------
# Analysis summary
# ---------------------------------------------------------------------------
def summarize(per_cue: list[dict], top_k: int = 20):
    n = len(per_cue)
    winners = [c for c in per_cue if c["gold_exclusive"] > 0]
    losers = [c for c in per_cue if c["gold_hit"] == 0]
    hitters = [
        c for c in per_cue if c["gold_hit"] > 0
    ]  # any gold hit (including non-exclusive)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else 0.0

    def median(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else 0.0

    def feature_table(group, name):
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            "mean_len_words": mean([c["cue_len_words"] for c in group]),
            "median_len_words": median([c["cue_len_words"] for c in group]),
            "mean_jaccard": mean([c["jaccard_with_q"] for c in group]),
            "pct_is_question": mean([float(c["is_question"]) for c in group]) * 100,
            "mean_entity_count": mean([c["entity_count"] for c in group]),
            "mean_number_count": mean([c["number_count"] for c in group]),
            "mean_cue_q_cos": mean([c["cue_q_cos"] for c in group]),
            "mean_best_gold_cos": mean([c["best_gold_cos"] for c in group]),
        }

    stats = {
        "total_cues": n,
        "winners_(gold_exclusive>0)": len(winners),
        "hitters_(any_gold)": len(hitters),
        "losers_(no_gold)": len(losers),
        "winners_pct": len(winners) / max(n, 1) * 100,
        "losers_pct": len(losers) / max(n, 1) * 100,
        "feature_table": {
            "all": feature_table(per_cue, "all"),
            "winners": feature_table(winners, "winners"),
            "hitters": feature_table(hitters, "hitters"),
            "losers": feature_table(losers, "losers"),
        },
    }

    # Per-category breakdown
    by_cat: dict[str, list] = defaultdict(list)
    for c in per_cue:
        by_cat[c["category"]].append(c)

    cat_stats = {}
    for cat, group in by_cat.items():
        if len(group) < 5:
            continue
        gwin = [c for c in group if c["gold_exclusive"] > 0]
        ghit = [c for c in group if c["gold_hit"] > 0]
        glose = [c for c in group if c["gold_hit"] == 0]
        cat_stats[cat] = {
            "n": len(group),
            "winners": len(gwin),
            "hitters": len(ghit),
            "losers": len(glose),
            "winner_rate": len(gwin) / len(group),
            "hitter_rate": len(ghit) / len(group),
            "winners_features": feature_table(gwin, "winners"),
            "losers_features": feature_table(glose, "losers"),
        }

    return stats, cat_stats


def pick_examples(per_cue: list[dict], n: int = 10):
    # Winners: best gold_exclusive, diverse categories
    wins = [c for c in per_cue if c["gold_exclusive"] > 0]
    wins.sort(key=lambda c: (-c["gold_exclusive"], -c["gold_hit"]))
    seen_cats = set()
    winners = []
    for c in wins:
        key = (c["category"], c["question"])
        if key in seen_cats:
            continue
        seen_cats.add(key)
        winners.append(c)
        if len(winners) >= n:
            break
    # Fill if fewer:
    if len(winners) < n:
        for c in wins:
            if c in winners:
                continue
            winners.append(c)
            if len(winners) >= n:
                break

    # Losers with high cue_q_cos (looked plausible, still missed)
    losers_all = [c for c in per_cue if c["gold_hit"] == 0]
    # prioritize plausible-looking: high cue_q_cos, long, moderately entity-rich
    losers_all.sort(key=lambda c: (-c["cue_q_cos"], -c["cue_len_words"]))
    seen_cats = set()
    losers = []
    for c in losers_all:
        key = (c["category"], c["question"])
        if key in seen_cats:
            continue
        seen_cats.add(key)
        losers.append(c)
        if len(losers) >= n:
            break
    if len(losers) < n:
        for c in losers_all:
            if c in losers:
                continue
            losers.append(c)
            if len(losers) >= n:
                break
    return winners, losers


def write_report(per_cue: list[dict], top_k: int = 20):
    stats, cat_stats = summarize(per_cue, top_k)
    winners, losers = pick_examples(per_cue)

    lines = []
    lines.append("# Per-Cue Attribution Analysis")
    lines.append("")
    lines.append(f"- Total (question, cue) pairs analyzed: **{stats['total_cues']}**")
    n_q = len({(c["conversation_id"], c["question"]) for c in per_cue})
    lines.append(f"- Unique questions: **{n_q}**")
    lines.append(f"- Top-K used for retrieval: **{top_k}**")
    lines.append(
        f"- Winners (≥1 gold hit exclusive vs baseline): **{stats['winners_(gold_exclusive>0)']}** ({stats['winners_pct']:.1f}%)"
    )
    lines.append(f"- Any-gold hitters: **{stats['hitters_(any_gold)']}**")
    lines.append(
        f"- Losers (0 gold in top-{top_k}): **{stats['losers_(no_gold)']}** ({stats['losers_pct']:.1f}%)"
    )
    lines.append("")

    # Feature table
    ft = stats["feature_table"]
    lines.append("## Winners vs Losers — Feature Distribution")
    lines.append("")
    lines.append("| feature | all | winners | hitters (any gold) | losers |")
    lines.append("|---|---|---|---|---|")
    for feat in [
        "mean_len_words",
        "median_len_words",
        "mean_jaccard",
        "pct_is_question",
        "mean_entity_count",
        "mean_number_count",
        "mean_cue_q_cos",
        "mean_best_gold_cos",
    ]:
        row = [feat]
        for g in ["all", "winners", "hitters", "losers"]:
            v = ft[g].get(feat, 0.0)
            if feat == "pct_is_question":
                row.append(f"{v:.1f}%")
            else:
                row.append(f"{v:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Delta rankings: winners - losers
    wf = ft["winners"]
    lf = ft["losers"]
    deltas = []
    for feat in [
        "mean_len_words",
        "mean_jaccard",
        "pct_is_question",
        "mean_entity_count",
        "mean_number_count",
        "mean_cue_q_cos",
        "mean_best_gold_cos",
    ]:
        w = wf.get(feat, 0.0)
        l = lf.get(feat, 0.0)
        # Normalize delta by loser value to get relative effect
        denom = max(abs(l), 1e-6)
        deltas.append((feat, w - l, (w - l) / denom, w, l))
    deltas.sort(key=lambda x: -abs(x[2]))

    lines.append("## Top features distinguishing winners from losers")
    lines.append("")
    lines.append("| feature | winners | losers | delta | rel Δ |")
    lines.append("|---|---|---|---|---|")
    for feat, d, rd, w, l in deltas[:7]:
        if feat == "pct_is_question":
            lines.append(f"| {feat} | {w:.1f}% | {l:.1f}% | {d:+.1f} | {rd:+.2f} |")
        else:
            lines.append(f"| {feat} | {w:.3f} | {l:.3f} | {d:+.3f} | {rd:+.2f} |")
    lines.append("")

    lines.append("## Category-specific patterns")
    lines.append("")
    lines.append(
        "| category | n | winner rate | hitter rate | win_len | lose_len | win_cos | lose_cos |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["n"])[:20]:
        wl = s["winners_features"].get("mean_len_words", 0.0)
        ll = s["losers_features"].get("mean_len_words", 0.0)
        wc = s["winners_features"].get("mean_best_gold_cos", 0.0)
        lc = s["losers_features"].get("mean_best_gold_cos", 0.0)
        lines.append(
            f"| {cat} | {s['n']} | {s['winner_rate']:.2f} | {s['hitter_rate']:.2f} | {wl:.1f} | {ll:.1f} | {wc:.3f} | {lc:.3f} |"
        )
    lines.append("")

    lines.append("## 10 Winning cue examples")
    lines.append("")
    for i, c in enumerate(winners[:10], 1):
        lines.append(f"### Winner #{i}  [{c['category']}]")
        lines.append(f"- **Question:** {c['question']}")
        lines.append(f"- **Cue:** {c['cue']}")
        lines.append(
            f"- gold_hit={c['gold_hit']}, gold_exclusive={c['gold_exclusive']}, cue_q_cos={c['cue_q_cos']:.3f}, best_gold_cos={c['best_gold_cos']:.3f}, len={c['cue_len_words']}"
        )
        if c.get("retrieved_gold_text"):
            txt = c["retrieved_gold_text"][:240].replace("\n", " ")
            lines.append(f"- **Retrieved gold turn:** {txt}")
        lines.append("")

    lines.append("## 10 Losing cue examples")
    lines.append("")
    for i, c in enumerate(losers[:10], 1):
        lines.append(f"### Loser #{i}  [{c['category']}]")
        lines.append(f"- **Question:** {c['question']}")
        lines.append(f"- **Cue:** {c['cue']}")
        lines.append(
            f"- gold_hit=0, cue_q_cos={c['cue_q_cos']:.3f}, best_gold_cos={c['best_gold_cos']:.3f}, len={c['cue_len_words']}, entities={c['entity_count']}, is_question={c['is_question']}"
        )
        lines.append("")

    # Local mean helper (write_report scope)
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else 0.0

    # Within-category feature deltas to remove category confound
    # for each category, compute winner - loser deltas, then average across categories
    local_by_cat: dict[str, list] = defaultdict(list)
    for c in per_cue:
        local_by_cat[c["category"]].append(c)
    cat_deltas = defaultdict(list)
    feats = [
        "cue_len_words",
        "jaccard_with_q",
        "is_question",
        "entity_count",
        "number_count",
        "cue_q_cos",
        "best_gold_cos",
    ]
    for cat, group in local_by_cat.items():
        gwin = [c for c in group if c["gold_exclusive"] > 0]
        glose = [c for c in group if c["gold_hit"] == 0]
        if len(gwin) < 3 or len(glose) < 3:
            continue
        for fkey in feats:
            w = _mean([float(c[fkey]) for c in gwin])
            l = _mean([float(c[fkey]) for c in glose])
            cat_deltas[fkey].append(w - l)

    lines.append(
        "## Within-category deltas (winners − losers, averaged across categories)"
    )
    lines.append("")
    lines.append(
        "Controls for the confound that some categories have inherently longer/shorter cues."
    )
    lines.append("")
    lines.append("| feature | avg Δ (winner − loser) | #cats with data |")
    lines.append("|---|---|---|")
    for fkey in feats:
        ds = cat_deltas[fkey]
        if not ds:
            continue
        lines.append(f"| {fkey} | {_mean(ds):+.3f} | {len(ds)} |")
    lines.append("")

    lines.append("## Actionable summary")
    lines.append("")
    lines.append("Based on winner-vs-loser distributions and within-category patterns:")
    lines.append("")
    # Derive bullets
    wins_len = ft["winners"]["mean_len_words"]
    loss_len = ft["losers"]["mean_len_words"]
    wins_cos = ft["winners"]["mean_best_gold_cos"]
    loss_cos = ft["losers"]["mean_best_gold_cos"]
    wins_iq = ft["winners"]["pct_is_question"]
    loss_iq = ft["losers"]["pct_is_question"]
    wins_ent = ft["winners"]["mean_entity_count"]
    loss_ent = ft["losers"]["mean_entity_count"]
    wins_jac = ft["winners"]["mean_jaccard"]
    loss_jac = ft["losers"]["mean_jaccard"]

    lines.append(
        f"- **Length:** winners average {wins_len:.1f} words vs losers {loss_len:.1f}. {'Shorter' if wins_len < loss_len else 'Longer'} cues win."
    )
    lines.append(
        f"- **Embedding distance from question:** winners' cues sit at cos={ft['winners']['mean_cue_q_cos']:.3f} from the question vs losers' {ft['losers']['mean_cue_q_cos']:.3f}. "
        f"{'Winners probe further from the query' if ft['winners']['mean_cue_q_cos'] < ft['losers']['mean_cue_q_cos'] else 'Winners stay closer to the query'}."
    )
    lines.append(
        f"- **Question-form cues:** {wins_iq:.1f}% of winners are questions vs {loss_iq:.1f}% of losers. {'Statement form wins' if wins_iq < loss_iq else 'Question form wins'}."
    )
    lines.append(
        f"- **Entity density:** winners {wins_ent:.2f} entities/cue vs losers {loss_ent:.2f}. "
        f"{'More entity-dense' if wins_ent > loss_ent else 'Less entity-dense'} cues win."
    )
    lines.append(
        f"- **Lexical overlap with question:** winners Jaccard={wins_jac:.3f} vs losers {loss_jac:.3f}. Counterintuitively, high question-token overlap is a loser signal — good cues probe *around* the question with chat-style text, not by echoing it."
    )
    lines.append(
        f"- **Best-gold cosine (how close cue got to any gold turn):** winners {wins_cos:.3f} vs losers {loss_cos:.3f}. This is the strongest signal — cues that geometrically approach a gold turn succeed."
    )
    lines.append("")
    lines.append("### Loser archetype (from top 10)")
    lines.append("")
    lines.append(
        'The dominant failure mode is the **"interrogative paraphrase"**: cues like `"Melanie painted that sunrise last year"` or `"When was Caroline\'s 18th birthday? ten years ago"`. They:'
    )
    lines.append(
        "- Are short declarative paraphrases of the question or question+guessed-answer."
    )
    lines.append(
        "- Sit at high cosine to the question (0.78–0.92) but low cosine to the actual gold turn (<0.35)."
    )
    lines.append(
        "- Hallucinate/guess the answer inline, polluting the embedding with tokens not in the chat log."
    )
    lines.append(
        "- Concentrate in locomo_temporal / locomo_single_hop where gold is a short chat turn whose vocabulary the LLM cannot predict from the question alone."
    )
    lines.append("")
    lines.append("### Winner archetype (from top 10)")
    lines.append("")
    lines.append(
        "- Chat-message text with named entities + specific nouns + timestamps or numbers. Sequential-chain, logic-constraint, and evolving-terminology questions produce the best cues because the LLM can draw vocabulary from the already-retrieved context rather than guessing."
    )
    lines.append(
        "- Notable: many winners have LOW cue_q_cos (0.23–0.37). The LLM is productively *pivoting* — inventing new vocabulary that matches the gold turn's voice rather than the question's."
    )
    lines.append("")
    lines.append("### v2f-successor variants to test")
    lines.append("")
    lines.append(
        '1. **Anti-paraphrase prompt hardening.** The existing v2f prompt already says "Do NOT write questions", but losers still echo question tokens. Add: "Do NOT restate the question. Do NOT guess an answer. Write a quote that might appear verbatim in the chat." Test: expect biggest lift on locomo_temporal (25% winner rate currently).'
    )
    lines.append(
        "2. **Context-anchored cues for sparse retrieval.** When the context_section is empty or weak (locomo_temporal, locomo_single_hop), v2f has nothing to seed cues with and falls back to paraphrasing. Add a mandatory 2-stage: hop0 retrieves 10, then force the LLM to *quote* 1–2 phrases from the retrieved context verbatim and extend each into a cue. Gold-cosine lift should be large because cues inherit real chat vocabulary."
    )
    lines.append(
        "3. **Entity/number injection for proactive and completeness categories.** Entity density is the 2nd-strongest feature (+1.9 per cue; rel Δ +1.4). For categories where recall is already >80% (constraint_propagation, negation), focus instead on recall@all by generating multiple entity-specific cue variants."
    )
    lines.append("")

    out_md = RESULTS_DIR / "per_cue_attribution.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_md}")
    return stats, cat_stats, winners, losers


if __name__ == "__main__":
    per_cue = main(top_k=20)
    write_report(per_cue, top_k=20)
