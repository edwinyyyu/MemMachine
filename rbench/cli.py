"""Command line: python -m rbench <command> ...

  prepare [dataset ... | all]        download and prepare datasets (validated on write)
  validate [dataset ...]             check prepared datasets against the format
  profile                            measure the datasets and write DATASETS.md
  examples [-n N]                    sampled queries, documents and gold per dataset -> EXAMPLES.md
  explain <dataset> <chunking> <qid> [--model M] [--pool-seeds N | --pool-tokens N --min-depth N]
                                     a dry run of one query through every measurement
  estimate <stage> ...               reranker tokens, DeepInfra dollars and hours per stage
  run <stage> [--only prepare|rerank|report] [--models m,...] [--max-usd X]
                                     --max-usd caps TOTAL DeepInfra reranker spend, all models and runs so far
  adhoc --datasets a,b --chunkings x,y --models m,n [--variant V] [--scope S] [--sample S]
        [--pool-tokens N] [--min-depth N] [--pool-seeds N] [--instruction I] [--render R] [--expand E] [--only ...]
        [--max-usd X]
                                     any combination on request (same caching and spend cap)
  smoke [--models q06,q4,q8]         preflight: what DeepInfra returns and bills
  fidelity <dataset> <scope> <variant> <chunking> <sample>   preflight: local vs DeepInfra 0.6B
  compare-embedders [--server S] [--questions N]   first stage under each embedding prompt setup (no reranker)
  embed-fidelity [--dataset D] [--chunking C]                preflight: local vs DeepInfra embeddinggemma
                                     (default: LongMemEval-M's EventMemory embedding text at seg500)

run, adhoc, estimate and explain take --embedder (default gemma300m-raw, the local server without prompts;
gemma300m-di-raw is DeepInfra's).
"""

from __future__ import annotations

import argparse
import json

from rbench import costs, grid, pipeline, preflight, profile, schema
from rbench.datasets import DATASETS, prepare
from rbench.embedding import DEFAULT as EMBEDDER, EMBEDDERS


def _estimate(runs: list, embedder: str) -> dict:
    """costs.estimate for flat datasets, costs.estimate_histories for history datasets."""
    from rbench.datasets import info

    flat = [r for r in runs if not info(r.dataset).partitioned]
    result = costs.estimate(flat, embedder) if flat else {"summary": {}, "jobs": []}
    history_runs = [r for r in runs if info(r.dataset).partitioned]
    if history_runs:
        result["summary"]["histories (estimate)"] = costs.estimate_histories(history_runs, embedder)["summary"]
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rbench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("datasets", nargs="*", default=["all"])
    p = sub.add_parser("validate")
    p.add_argument("datasets", nargs="*")
    sub.add_parser("profile")
    p = sub.add_parser("explain")
    p.add_argument("dataset")
    p.add_argument("chunking")
    p.add_argument("qid")
    p.add_argument("--model", default="q06-local")
    p.add_argument("--pool-seeds", type=int, help="history datasets: seeds in the rerank pool (default: the core stage's)")
    p.add_argument("--pool-tokens", type=int, help="other datasets: rerank pool depth in seed tokens")
    p.add_argument("--min-depth", type=int)
    p = sub.add_parser("examples")
    p.add_argument("-n", type=int, default=3)
    p = sub.add_parser("estimate")
    p.add_argument("stages", nargs="+", choices=grid.STAGES)
    p.add_argument("--jobs", action="store_true", help="also list every scoring job")
    p = sub.add_parser("run")
    p.add_argument("stage", choices=grid.STAGES)
    p.add_argument("--only", choices=("prepare", "rerank", "report"))
    p.add_argument("--models")
    p.add_argument("--max-usd", type=float)
    p = sub.add_parser("adhoc")
    for name in ("datasets", "chunkings", "models"):
        p.add_argument(f"--{name}", required=True)
    for name, default in (("variant", None), ("scope", None), ("sample", "large"), ("instruction", "default"),
                          ("render", "plain"), ("only", None)):
        p.add_argument(f"--{name}", default=default)
    p.add_argument("--pool-tokens", type=int, default=8192)
    p.add_argument("--min-depth", type=int, default=16)
    p.add_argument("--expand", type=int, default=0, help="history datasets: score each seed's window (expand_context)")
    p.add_argument("--pool-seeds", type=int, default=50, help="history datasets: seeds scored per question")
    p.add_argument("--max-usd", type=float)
    p = sub.add_parser("compare-embedders")
    p.add_argument("--server", default="gemma300m", help="gemma300m (local llama-server) or gemma300m-di")
    p.add_argument("--dataset", default="longmemeval_m")
    p.add_argument("--size", type=int, default=500)
    p.add_argument("--questions", type=int, default=100, help="the first N in the samples' hash order (100: the pilot's)")
    p = sub.add_parser("embed-fidelity")
    p.add_argument("--dataset", default="longmemeval_m")
    p.add_argument("--chunking", default="seg500")
    for name in ("run", "adhoc", "estimate", "explain"):
        sub.choices[name].add_argument("--embedder", default=EMBEDDER, choices=EMBEDDERS)
    p = sub.add_parser("smoke")
    p.add_argument("--models", default="q06,q4,q8")
    p = sub.add_parser("fidelity")
    for name in ("dataset", "scope", "variant", "chunking", "sample"):
        p.add_argument(name)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        for name in DATASETS if args.datasets == ["all"] else args.datasets:
            prepare(name)
    elif args.command == "validate":
        from rbench.datasets import info

        for name in args.datasets or DATASETS:
            if info(name).partitioned:
                from rbench.datasets.longmemeval import history

                problems = []
                for q in schema.queries(name):
                    h = history(name, q["qid"])
                    for s_, t in q["gold_turns"][0]:
                        if not (s_ < len(h["sessions"]) and t < len(h["sessions"][s_]["turns"])
                                and h["sessions"][s_]["turns"][t]["has_answer"]):
                            problems.append(f"{q['qid']}: gold turn {s_}/{t} missing or unmarked")
                    if any(s_ >= len(h["sessions"]) for s_ in q["gold_sessions"]):
                        problems.append(f"{q['qid']}: gold session out of range")
                print(name, "ok" if not problems else problems[:10])
                continue
            problems = schema.validate([{"doc_id": d, "text": t} for d, t in zip(schema.corpus(name).doc_ids,
                                       schema.corpus(name).texts, strict=True)], list(schema.queries(name)))
            print(name, "ok" if not problems else problems[:10])
    elif args.command == "profile":
        print(profile.write())
    elif args.command == "explain":
        from rbench import explain

        print(explain.run(args.dataset, args.chunking, args.qid, args.model, args.pool_tokens, args.min_depth, args.embedder,
                          args.pool_seeds))
    elif args.command == "examples":
        from rbench import examples

        print(examples.write(args.n))
    elif args.command == "estimate":
        result = _estimate([r for s in args.stages for r in grid.STAGES[s]], args.embedder)
        print(json.dumps(result["summary"], indent=1))
        if args.jobs:
            for job in result["jobs"]:
                print(job)
    elif args.command in ("run", "adhoc"):
        if args.command == "run":
            runs = grid.STAGES[args.stage]
            models = set(args.models.split(",")) if args.models else None
        else:
            runs = grid.adhoc(args.datasets.split(","), args.chunkings.split(","), args.models.split(","),
                              variant=args.variant, scope=args.scope, sample=args.sample, pool_tokens=args.pool_tokens,
                              min_depth=args.min_depth, instruction=args.instruction, render=args.render,
                              expand=args.expand, pool_seeds=args.pool_seeds)
            models = None
            print(json.dumps(_estimate(runs, args.embedder)["summary"], indent=1))
        if args.only in (None, "prepare"):
            for run in runs:
                pipeline.prepare(run, args.embedder)
        if args.only in (None, "rerank"):
            for run in runs:
                for m in run.models:
                    if models is None or m in models:
                        pipeline.rerank(run, m, args.max_usd, args.embedder)
        if args.only in (None, "report"):
            for path in pipeline.simulate_and_report(runs, args.embedder):
                print(path)
    elif args.command == "compare-embedders":
        from rbench import compare_embedders

        print(compare_embedders.write(args.server, args.dataset, args.size, args.questions))
    elif args.command == "embed-fidelity":
        print(json.dumps(preflight.embed_fidelity(args.dataset, args.chunking), indent=1))
    elif args.command == "smoke":
        problems = preflight.smoke(args.models.split(","))
        print("smoke ok" if not problems else "\n".join(problems))
    elif args.command == "fidelity":
        print(json.dumps(preflight.fidelity(args.dataset, args.scope, args.variant, args.chunking, args.sample), indent=1))
