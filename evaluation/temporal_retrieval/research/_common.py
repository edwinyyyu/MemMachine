"""Shared helpers for the research / ablation harnesses.

Centralizes patterns that the harness scripts (`_ablation_proper`,
`_ablation_hard`, `_prompt_optimizer`, `_validate_best_prompt`,
`_sensitivity_curated_bench`, `_smoke_e2e`) all duplicated:

- Proxy env stripping + dotenv loading.
- `make_embed_fn()` (OpenAI text-embedding-3-small).
- `make_rerank_fn()` (cross-encoder/ms-marco-MiniLM-L-6-v2).
- Bench JSONL loading.
- Per-doc interval summary used by extraction-quality diffs.
- Pass-1 prompt builder + bare ref-context for prompt-component ablations.
- The four reusable print-summary helpers shared by ablation scripts.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from temporal_retrieval import extractor_common as ext_mod

# Legacy v1 prompt constants were used by the historical prompt-ablation
# scripts in this directory (default_variants(), make_pass1(...)). Those
# scripts are inert post-v3.1-ship; the constants are stubbed below so
# imports of `default_variants` etc. still succeed for any caller that
# only references the shared helpers.
LEGACY_FEW_SHOT_EXAMPLES = ""
LEGACY_PASS1_SYSTEM = ""
LEGACY_TRIGGER_GAZETTEER = ""

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT.parent
DATA_DIR = EVAL_ROOT / "temporal_extraction" / "data"


# ---------------------------------------------------------------------------
# Env setup
# ---------------------------------------------------------------------------
_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
)


def setup_env() -> None:
    """Strip proxy envs (which interfere with OpenAI client TLS in some
    setups) and load `.env` from the repo root."""
    for k in _PROXY_ENV_VARS:
        os.environ.pop(k, None)
    load_dotenv(EVAL_ROOT.parents[0] / ".env")


# ---------------------------------------------------------------------------
# Embedder + reranker
# ---------------------------------------------------------------------------
async def make_embed_fn():
    """OpenAI text-embedding-3-small. Same as v5.1 benchmarks."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    async def embed(texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        resp = await client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )
        return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]

    return embed


async def make_rerank_fn():
    """ms-marco-MiniLM-L-6-v2 cross-encoder. Same as v5.1 benchmarks."""
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    async def rerank(query: str, doc_texts: list[str]) -> list[float]:
        if not doc_texts:
            return []
        scores = ce.predict([[query, d] for d in doc_texts])
        return [float(s) for s in scores]

    return rerank


# ---------------------------------------------------------------------------
# Bench JSONL loading
# ---------------------------------------------------------------------------
def load_bench_jsonl(
    docs_file: str, queries_file: str, gold_file: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load a bench's three JSONL files relative to DATA_DIR."""
    with open(DATA_DIR / docs_file) as f:
        docs_jsonl = [json.loads(line) for line in f]
    with open(DATA_DIR / queries_file) as f:
        queries = [json.loads(line) for line in f]
    with open(DATA_DIR / gold_file) as f:
        gold_rows = [json.loads(line) for line in f]
    return docs_jsonl, queries, gold_rows


def summarize_intervals(doc_ivs: dict[str, list]) -> dict:
    """Per-doc surface count + interval keys; used for extraction-quality
    diffs across variants."""
    return {
        did: {
            "n_intervals": len(ivs),
            "interval_keys": [f"{iv.earliest_us}-{iv.latest_us}" for iv in ivs],
        }
        for did, ivs in doc_ivs.items()
    }


# ---------------------------------------------------------------------------
# Pass-1 prompt builder for component ablations
# ---------------------------------------------------------------------------
def make_pass1(without: tuple[str, ...]) -> str:
    """Build a pass-1 system prompt with the named components removed.

    `without` may contain "gazetteer" and/or "few_shot". The generated
    prompt is shorter than the production default and so won't bytewise
    equal the production prompt — used as a deliberate ablation variant.
    """
    gaz = LEGACY_TRIGGER_GAZETTEER if "gazetteer" not in without else ""
    fs = LEGACY_FEW_SHOT_EXAMPLES if "few_shot" not in without else ""
    return f"""You are a meticulous temporal-reference extractor.

Your job: identify EVERY temporal reference in a passage. A temporal
reference is any span that refers to a moment, span, duration, or recurring
pattern in time. It can be absolute ("March 5, 2026"), relative
("yesterday", "2 weeks ago"), vague ("around 2010", "a decade ago"), or
recurring ("every Thursday at 3pm").

{gaz}

For each reference, output:
- surface: the exact substring from the passage, verbatim, with no edits
  to casing, spacing, or punctuation. Prefer the LONGEST natural phrase
  that carries the temporal meaning.
- kind_guess: one of [instant, interval, duration, recurrence].
- context_hint: a short (<=12 word) note of what it refers to.

Do NOT emit seasons ("summer") unless the year is specified or strongly
implied by context.

{fs}

Output a single JSON object: {{"refs": [...]}}. If none, output {{"refs": []}}.
"""


def bare_ref_context(ref_time: _dt.datetime) -> str:
    """Minimal deictic context — just the ISO ref time. Used as the
    no-ref-context ablation variant."""
    iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"Reference time: {iso_ref}."


@dataclass
class Variant:
    """Pair of (pass-1 system prompt, ref-context function) describing
    one ablation variant."""

    pass1: str
    ref_ctx: Callable[[_dt.datetime], str]


def default_variants() -> dict[str, Variant]:
    """Six standard ablation variants used by both ablation scripts:
    two baseline runs (variance) + each prompt component dropped + all
    dropped.

    Baseline is the LEGACY (v5.1) PASS1 prompt — the variants ablate its
    components (TRIGGER_GAZETTEER, FEW_SHOT_EXAMPLES, full_ref_context).
    The current production PASS1_SYSTEM is principle-based and does not
    have those interpolatable components; ablating it requires a
    different harness.
    """
    return {
        "baseline_run1": Variant(LEGACY_PASS1_SYSTEM, ext_mod.full_ref_context),
        "baseline_run2": Variant(LEGACY_PASS1_SYSTEM, ext_mod.full_ref_context),
        "no_few_shot": Variant(make_pass1(("few_shot",)), ext_mod.full_ref_context),
        "no_gazetteer": Variant(make_pass1(("gazetteer",)), ext_mod.full_ref_context),
        "no_ref_context": Variant(LEGACY_PASS1_SYSTEM, bare_ref_context),
        "no_all": Variant(make_pass1(("gazetteer", "few_shot")), bare_ref_context),
    }


# ---------------------------------------------------------------------------
# Print helpers shared by ablation scripts
# ---------------------------------------------------------------------------
def print_main_summary(all_rows: list[dict]) -> None:
    print("\n" + "=" * 95)
    print(
        f"{'bench':22s} {'variant':16s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} "
        f"{'all_rec@5':>10s} {'tokens-in':>10s}"
    )
    print("-" * 95)
    for r in all_rows:
        if "error" in r:
            print(f"{r['bench']:22s} {r['variant']:16s} ERROR: {r['error']}")
            continue
        print(
            f"{r['bench']:22s} {r['variant']:16s} "
            f"{r['R@1']:>6.3f} {r['R@5']:>6.3f} {r['R@10']:>6.3f} "
            f"{r['all_recall@5']:>10.3f} {r['extractor_input_tokens']:>10d}"
        )
    print("=" * 95)


def _baseline_rows_for_bench(all_rows: list[dict], bench: str) -> list[dict]:
    return [
        r
        for r in all_rows
        if r.get("bench") == bench and r.get("variant", "").startswith("baseline_run")
    ]


def print_variance(all_rows: list[dict], benches: dict) -> None:
    print("\n--- Variance estimate (2 baseline runs) ---")
    print(
        f"{'bench':22s} {'metric':14s} {'b1':>7s} {'b2':>7s} {'mean':>7s} {'|delta|':>8s}"
    )
    for bench in benches:
        rows = _baseline_rows_for_bench(all_rows, bench)
        if len(rows) < 2:
            continue
        for metric in ("R@1", "R@5", "all_recall@5"):
            v1, v2 = rows[0][metric], rows[1][metric]
            mean = (v1 + v2) / 2
            delta = abs(v1 - v2)
            print(
                f"{bench:22s} {metric:14s} "
                f"{v1:>7.3f} {v2:>7.3f} {mean:>7.3f} {delta:>8.3f}"
            )


def print_variant_deltas(all_rows: list[dict], benches: dict) -> None:
    print("\n--- Variant deltas vs baseline mean (R@1) ---")
    print(
        f"{'bench':22s} {'variant':16s} {'baseline_mean':>14s} "
        f"{'variant_R@1':>13s} {'delta':>8s}"
    )
    for bench in benches:
        baseline_rows = _baseline_rows_for_bench(all_rows, bench)
        if len(baseline_rows) < 2:
            continue
        baseline_mean = sum(r["R@1"] for r in baseline_rows) / len(baseline_rows)
        for r in all_rows:
            if r.get("bench") != bench or r.get("variant", "").startswith(
                "baseline_run"
            ):
                continue
            if "error" in r:
                continue
            d = r["R@1"] - baseline_mean
            print(
                f"{bench:22s} {r['variant']:16s} "
                f"{baseline_mean:>14.3f} {r['R@1']:>13.3f} {d:+8.3f}"
            )


def print_extraction_quality(all_rows: list[dict], benches: dict) -> None:
    print("\n--- Extraction quality vs baseline ---")
    print(
        f"{'bench':22s} {'variant':16s} {'matching':>9s} {'fewer':>7s} "
        f"{'more':>6s} {'jaccard':>9s}"
    )
    for bench in benches:
        baseline_row = next(
            (
                r
                for r in all_rows
                if r.get("bench") == bench
                and r.get("variant") == "baseline_run1"
                and "extraction" in r
            ),
            None,
        )
        if baseline_row is None:
            continue
        baseline_keys = baseline_row["extraction"]
        n_total = len(baseline_keys)
        for r in all_rows:
            if r.get("bench") != bench or r.get("variant") == "baseline_run1":
                continue
            if "extraction" not in r:
                continue
            n_match = n_fewer = n_more = 0
            jaccards: list[float] = []
            for did, base_data in baseline_keys.items():
                base_set = set(base_data["interval_keys"])
                var_set = set(r["extraction"].get(did, {}).get("interval_keys", []))
                if base_set == var_set:
                    n_match += 1
                elif var_set < base_set:
                    n_fewer += 1
                elif var_set > base_set:
                    n_more += 1
                union = base_set | var_set
                jaccards.append(len(base_set & var_set) / len(union) if union else 1.0)
            mean_j = sum(jaccards) / max(1, len(jaccards))
            print(
                f"{bench:22s} {r['variant']:16s} "
                f"{n_match:>4d}/{n_total:<3d} "
                f"{n_fewer:>7d} {n_more:>6d} {mean_j:>9.3f}"
            )
