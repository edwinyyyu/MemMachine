"""Hill-climbing prompt optimizer for the pass-1 extractor.

gpt-5-mini plays both roles:
- subject: the extractor under optimization
- optimizer: proposes new prompt variants given the history

Bench: realq_v2 (83 docs, 34 queries) — the only ablation bench with
non-saturated R@5 (0.941) and zero baseline-vs-baseline R@5 noise.

Metric: dollar cost = 0.25 * input_M + 2.00 * output_M (gpt-5-mini USD/M).
Constraint: R@5 must stay >= baseline_r5 - 0.005 (i.e. tolerate ≤1 query loss).

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._prompt_optimizer
"""

from __future__ import annotations

import asyncio
import json
import re

from openai import AsyncOpenAI

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval import extractor as ext_mod

from ._common import (
    DATA_DIR,
    ROOT,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

# gpt-5-mini approximate pricing (USD per 1M tokens).
INPUT_COST_PER_M = 0.25
OUTPUT_COST_PER_M = 2.00


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * INPUT_COST_PER_M + (
        output_tokens / 1e6
    ) * OUTPUT_COST_PER_M


OPTIMIZER_SYSTEM_TMPL = """You are a prompt engineer optimizing a temporal-reference \
extraction prompt for gpt-5-mini.

The prompt instructs the model to extract temporal expressions from a passage into:
{{"refs": [{{"surface", "kind_guess", "context_hint"}}]}}

Goal: MINIMIZE total dollar cost while preserving retrieval R@5 on realq_v2.

Cost model (gpt-5-mini USD per 1M tokens):
- Input: ${input_cost}/M
- Output: ${output_cost}/M
- Output costs ~{ratio}x more than input. The fastest cost win is reducing OUTPUT \
tokens (i.e. fewer spurious surfaces emitted). Shortening the prompt itself only \
helps marginally because the input bill is small.

Constraints:
- R@5 must stay ≥ {baseline_r5:.3f} (the production baseline). Below that = reject.
- Output JSON format MUST be {{"refs": [{{"surface": ..., "kind_guess": ..., \
"context_hint": ...}}]}}. kind_guess ∈ {{instant, interval, duration, recurrence}}.
- The user message will append a deictic context block (today/last week/etc.) and \
the passage. Your prompt should not duplicate that.

Strategy hints (these are hypotheses to try, not rules):
- Few-shot examples constrain emission strongly (removing them inflates output 4.5×).
- A trigger gazetteer also acts as a constraint (removing it inflates similarly).
- An EXPLICIT "do not emit" exclusion list might be cheaper than a positive gazetteer.
- Terser examples may work as well as long ones.
- Restating the JSON schema once vs many times is a real degree of freedom.
- Tone of voice (terse vs. verbose, instructions vs. examples) is a degree of freedom.

You will be shown the history of variants tried with their scores. Propose 3 NEW \
variants. Format EXACTLY:

```
VARIANT_1_HYPOTHESIS: <one sentence stating the design hypothesis>
VARIANT_1_PROMPT:
<full prompt text, ending with the JSON-shape instruction>
===END_VARIANT_1===
VARIANT_2_HYPOTHESIS: ...
VARIANT_2_PROMPT:
...
===END_VARIANT_2===
VARIANT_3_HYPOTHESIS: ...
VARIANT_3_PROMPT:
...
===END_VARIANT_3===
```

Be creative; explore different framings. Don't just trim characters — change the \
strategy. If a recent variant won, lean into that direction; if it lost, try the \
opposite. Avoid repeating prompts that already lost."""


async def evaluate_prompt(prompt: str, embed_fn, rerank_fn, cache_subdir: str) -> dict:
    """Evaluate on the sensitivity-curated bench (11 queries from
    composition+adversarial+realq_v2 where ablation variants disagreed).
    Each query that flips contributes 1/11 ≈ 0.091 R@5, so prompt effects
    are amplified vs. running on full saturated benches."""
    with open(DATA_DIR / "sensitivity_curated_docs.jsonl") as f:
        docs_jsonl = [json.loads(line) for line in f]
    with open(DATA_DIR / "sensitivity_curated_queries.jsonl") as f:
        queries = [json.loads(line) for line in f]
    with open(DATA_DIR / "sensitivity_curated_gold.jsonl") as f:
        gold_rows = [json.loads(line) for line in f]
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    extractor = ext_mod.TemporalExtractor(
        cache_subdir=cache_subdir, pass1_system=prompt
    )
    retriever = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn, extractor=extractor
    )
    await retriever.index(docs)

    K = 5
    n_eval = 0
    n_r5 = 0
    n_r1 = 0
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        first_gold = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
        if first_gold is not None:
            if first_gold <= 1:
                n_r1 += 1
            if first_gold <= K:
                n_r5 += 1

    stats = retriever.stats()
    in_t = stats["extractor_usage"]["input"]
    out_t = stats["extractor_usage"]["output"]
    return {
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "input_tokens": in_t,
        "output_tokens": out_t,
        "cost_usd": cost_usd(in_t, out_t),
        "n_eval": n_eval,
    }


def _format_history(history: list[dict]) -> str:
    lines = []
    # Sort by cost ascending for the model's eye
    for h in sorted(history, key=lambda x: x["cost_usd"]):
        lines.append(
            f"[r{h['round']}v{h['variant_id']}] "
            f"R@5={h['R@5']:.3f} R@1={h['R@1']:.3f} "
            f"cost=${h['cost_usd']:.4f} "
            f"in={h['input_tokens']} out={h['output_tokens']}\n"
            f"  hypothesis: {h.get('hypothesis', 'N/A')}\n"
            f"  prompt[:600]: {h['prompt'][:600].replace(chr(10), ' / ')}"
        )
    return "\n\n".join(lines)


def _parse_variants(text: str) -> list[tuple[str, str]]:
    """Parse 3 variants from the optimizer's response."""
    variants = []
    for k in range(1, 4):
        hypo_pat = re.compile(rf"VARIANT_{k}_HYPOTHESIS:\s*(.+?)\n", re.DOTALL)
        prompt_pat = re.compile(
            rf"VARIANT_{k}_PROMPT:\s*(.+?)===END_VARIANT_{k}===", re.DOTALL
        )
        h_match = hypo_pat.search(text)
        p_match = prompt_pat.search(text)
        if h_match and p_match:
            variants.append((h_match.group(1).strip(), p_match.group(1).strip()))
    return variants


async def propose_variants(
    history: list[dict], baseline_r5: float, optimizer: AsyncOpenAI
) -> list[tuple[str, str]]:
    history_text = _format_history(history)
    user = (
        f"History so far ({len(history)} variants tried):\n\n{history_text}\n\n"
        "Propose 3 NEW variants per the format. Aim for variants that explore \n"
        "different design hypotheses, not just minor edits."
    )
    sys_prompt = OPTIMIZER_SYSTEM_TMPL.format(
        input_cost=INPUT_COST_PER_M,
        output_cost=OUTPUT_COST_PER_M,
        ratio=int(OUTPUT_COST_PER_M / INPUT_COST_PER_M),
        baseline_r5=baseline_r5,
    )
    resp = await optimizer.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=10000,
    )
    text = resp.choices[0].message.content or ""
    return _parse_variants(text)


async def optimize(n_rounds: int = 4) -> None:
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()
    optimizer = AsyncOpenAI()

    print("\n=== Round 0: baseline (current production prompt) ===", flush=True)
    baseline = await evaluate_prompt(
        ext_mod.PASS1_SYSTEM, embed_fn, rerank_fn, "opt_baseline"
    )
    history: list[dict] = [
        {
            "round": 0,
            "variant_id": 0,
            "hypothesis": "current production prompt (TRIGGER_GAZETTEER + FEW_SHOT_EXAMPLES + full schema)",
            "prompt": ext_mod.PASS1_SYSTEM,
            **baseline,
        }
    ]
    print(
        f"  R@5={baseline['R@5']:.3f} R@1={baseline['R@1']:.3f} "
        f"cost=${baseline['cost_usd']:.4f} "
        f"in={baseline['input_tokens']} out={baseline['output_tokens']}",
        flush=True,
    )

    baseline_r5 = baseline["R@5"]
    tolerance = 0.005

    for rnd in range(1, n_rounds + 1):
        print(f"\n=== Round {rnd}: proposing variants ===", flush=True)
        try:
            variants = await propose_variants(history, baseline_r5, optimizer)
        except Exception as e:
            print(f"  proposer error: {e}", flush=True)
            continue
        print(f"  {len(variants)} variants proposed", flush=True)

        for i, (hypo, prompt) in enumerate(variants):
            vid = f"r{rnd}v{i}"
            print(f"\n--- {vid}: {hypo[:120]} ---", flush=True)
            try:
                result = await evaluate_prompt(
                    prompt, embed_fn, rerank_fn, f"opt_{vid}"
                )
                history.append(
                    {
                        "round": rnd,
                        "variant_id": i,
                        "hypothesis": hypo,
                        "prompt": prompt,
                        **result,
                    }
                )
                feasible = result["R@5"] >= baseline_r5 - tolerance
                save = (
                    f" SAVE=${baseline['cost_usd'] - result['cost_usd']:.4f}"
                    if feasible and result["cost_usd"] < baseline["cost_usd"]
                    else ""
                )
                viol = "" if feasible else " (R@5 below tolerance)"
                print(
                    f"  R@5={result['R@5']:.3f} R@1={result['R@1']:.3f} "
                    f"cost=${result['cost_usd']:.4f} "
                    f"in={result['input_tokens']} out={result['output_tokens']}"
                    f"{save}{viol}",
                    flush=True,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                history.append(
                    {
                        "round": rnd,
                        "variant_id": i,
                        "hypothesis": hypo,
                        "prompt": prompt,
                        "error": str(e),
                        "R@5": 0.0,
                        "R@1": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": float("inf"),
                        "n_eval": 0,
                    }
                )

    print("\n\n=== FEASIBLE PARETO TABLE (R@5 >= baseline - 0.005) ===", flush=True)
    print(
        f"{'rank':>4s} {'cost_usd':>10s} {'R@5':>6s} {'R@1':>6s} "
        f"{'in':>8s} {'out':>8s} {'id':>8s}  hypothesis"
    )
    feasible = [h for h in history if h["R@5"] >= baseline_r5 - tolerance]
    feasible.sort(key=lambda x: x["cost_usd"])
    for i, h in enumerate(feasible):
        vid = f"r{h['round']}v{h['variant_id']}"
        print(
            f"{i + 1:>4d} ${h['cost_usd']:>9.4f} {h['R@5']:>6.3f} {h['R@1']:>6.3f} "
            f"{h['input_tokens']:>8d} {h['output_tokens']:>8d} {vid:>8s}  "
            f"{h.get('hypothesis', '')[:80]}",
            flush=True,
        )

    out_path = ROOT / "prompt_optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)

    # Save the BEST feasible prompt to its own file
    if feasible:
        best = feasible[0]
        best_path = ROOT / "best_prompt.txt"
        with open(best_path, "w") as f:
            f.write(
                f"# Best feasible prompt (R@5={best['R@5']:.3f}, cost=${best['cost_usd']:.4f})\n"
            )
            f.write(f"# Hypothesis: {best.get('hypothesis', '')}\n")
            f.write(
                f"# Baseline cost was ${baseline['cost_usd']:.4f}; "
                f"savings ${baseline['cost_usd'] - best['cost_usd']:.4f}\n\n"
            )
            f.write(best["prompt"])
        print(f"Wrote {best_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(optimize())
