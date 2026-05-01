"""Render the hard_bench_pipeline.json into a markdown report.

Reads results/hard_bench_pipeline.json and emits results/hard_bench_pipeline.md
with per-variant tables, win/loss examples, and a generalization verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    p = RESULTS / "hard_bench_pipeline.json"
    if not p.exists():
        print(f"missing: {p}")
        return
    d = json.loads(p.read_text())

    lines = []
    lines.append("# Hard temporal-stress benchmark — pipeline results")
    lines.append("")
    bench = d["benchmark"]
    lines.append(f"**Benchmark**: {bench['name']}")
    lines.append(
        f"**Docs**: {bench['n_docs']}, **queries**: {bench['n_queries']} "
        f"({bench['n_easy']} easy / {bench['n_medium']} medium / {bench['n_hard']} hard)"
    )
    lines.append(
        f"**Cost**: ${d['cost']['total_usd']:.4f}, wall: {d['wall_seconds']:.1f}s"
    )
    lines.append("")
    ext = d["extraction"]
    lines.append(
        f"**Extraction**: {ext['doc_extractions_per_doc_mean']:.2f} mean te/doc, "
        f"{ext['q_extractions_per_q_mean']:.2f} mean te/query, "
        f"timeouts={ext['doc_timeouts']}+{ext['query_timeouts']}, "
        f"errors={ext['doc_errors']}+{ext['query_errors']}"
    )
    lines.append(f"**Lattice**: {d['lattice_stats']}")
    lines.append("")

    pv = d["per_variant"]
    for sub in ["all", "easy", "medium", "hard"]:
        lines.append(f"## {sub.upper()} subset")
        lines.append("")
        lines.append("| Variant | n | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 |")
        lines.append("|---------|---|-----|-----|-----|------|-----|---------|")
        for var, by_sub in pv.items():
            m = by_sub.get(sub, {})
            if not m or m.get("n", 0) == 0:
                continue
            lines.append(
                f"| {var} | {m['n']} | {fmt(m['recall@1'])} | {fmt(m['recall@3'])} | "
                f"{fmt(m['recall@5'])} | {fmt(m['recall@10'])} | "
                f"{fmt(m['mrr'])} | {fmt(m['ndcg@10'])} |"
            )
        lines.append("")

    fa = d["failure_analysis"]
    lines.append("## Failure analysis")
    lines.append("")
    lines.append(f"- Total queries: {fa['n_total']}")
    lines.append(f"- V7 wins (V7 rank=1, sem rank>1): {fa['n_wins_v7_over_sem']}")
    lines.append(f"- V7 losses (V7 rank>1, sem rank=1): {fa['n_losses_v7_to_sem']}")
    lines.append(f"- Persistent misses (both >5): {fa['n_persistent_misses']}")
    lines.append("")
    lines.append("### V7 wins")
    lines.append("")
    for r in fa["wins"][:10]:
        lines.append(f"- **{r['qid']}** [{r['subset']}] `{r['query']}`")
        lines.append(
            f"  - gold: `{r['gold_text']}` (sem rank={r['rank_sem']}, "
            f"V7 rank={r['rank_v7']}, V7L rank={r['rank_v7l']}, "
            f"T-only rank={r['rank_t']})"
        )
        lines.append(f"  - sem top3: `{r['sem_top3_texts']}`")
    lines.append("")
    lines.append("### V7 losses")
    lines.append("")
    for r in fa["losses"][:10]:
        lines.append(f"- **{r['qid']}** [{r['subset']}] `{r['query']}`")
        lines.append(
            f"  - gold: `{r['gold_text']}` (sem rank={r['rank_sem']}, "
            f"V7 rank={r['rank_v7']}, V7L rank={r['rank_v7l']}, "
            f"T-only rank={r['rank_t']})"
        )
        lines.append(f"  - V7 top3: `{r['v7_top3_texts']}`")
    lines.append("")
    lines.append("### Persistent misses (both lost)")
    lines.append("")
    for r in fa["persistent_misses"][:5]:
        lines.append(f"- **{r['qid']}** [{r['subset']}] `{r['query']}`")
        lines.append(
            f"  - gold: `{r['gold_text']}` (sem rank={r['rank_sem']}, V7 rank={r['rank_v7']})"
        )
        lines.append(f"  - sem top3: `{r['sem_top3_texts']}`")
    lines.append("")

    out_path = RESULTS / "hard_bench_pipeline.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
