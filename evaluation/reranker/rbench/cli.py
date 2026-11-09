"""Command line: python -m rbench <command> ...

  prepare [dataset ... | all]        download and prepare datasets (validated on write)
  validate [dataset ...]             check prepared datasets against the format
  profile                            measure the datasets and write DATASETS.md
  examples [-n N]                    sampled queries, documents and gold per dataset -> EXAMPLES.md
  explain <dataset> <chunking> <qid> [--model M] [--pool-seeds N]
                                     a dry run of one query through every measurement
  estimate <stage> ...               reranker tokens, DeepInfra dollars and hours per stage
  run <stage> [--only prepare|rerank|report] [--models m,...] [--max-usd X]
                                     --models runs the stage with these registered models instead of its own
                                     --max-usd caps TOTAL DeepInfra reranker spend, all models and runs so far
  adhoc --datasets a,b --chunkings x,y --models m,n [--variant V] [--scope S] [--sample S]
        [--pool-seeds N] [--expand E] [--only ...] [--max-usd X]
                                     chunkings: seg<N> (EventMemory's TextSegmenter) or cut<N> (the simple chunker)
                                     any combination on request (same caching and spend cap)
  smoke [--models q06,q4,q8]         preflight: what DeepInfra returns and bills
  fidelity <dataset> <chunking> <sample>   preflight: local vs DeepInfra 0.6B on the same pairs
  show <dataset> [--sample S] [--chart coverage,rpf,...] [--scope S] [--variant V]
                                     open a run's charts in zoomable windows, from its saved results
  compare-embedders [--server S] [--questions N]   first stage under each embedding prompt setup (no reranker)
  embed-fidelity [--dataset D] [--chunking C]                preflight: local vs DeepInfra embeddinggemma
                                     (default: LongMemEval-M's EventMemory embedding text at seg500)

run, adhoc and show take --budgets, --limits and --readings: the values reports are drawn at
(defaults in rbench/config.py).
run, adhoc, estimate and explain take --embedder (default gemma300m-raw, the local server without prompts;
gemma300m-di-raw is DeepInfra's).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

from rbench import costs, grid, pipeline, preflight, profile, schema
from rbench.datasets import DATASETS, prepare
from rbench.embedding import DEFAULT as EMBEDDER, EMBEDDERS
from rbench.rerankers import RERANKERS


def _estimate(runs: list, embedder: str) -> dict:
    return costs.estimate(runs, embedder)


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
    p.add_argument("--pool-seeds", type=int, default=50, help="seeds in the rerank pool (the core stage's)")
    p.add_argument("--expand", type=int, default=3, help="expand_context of the windows traced")
    p = sub.add_parser("examples")
    p.add_argument("-n", type=int, default=3)
    p = sub.add_parser("estimate")
    p.add_argument("stages", nargs="+", choices=grid.STAGES)
    p = sub.add_parser("run")
    p.add_argument("stage", choices=grid.STAGES)
    p.add_argument("--only", choices=("prepare", "rerank", "report"))
    p.add_argument("--models")
    p.add_argument("--max-usd", type=float)
    p = sub.add_parser("adhoc")
    for name in ("datasets", "chunkings", "models"):
        p.add_argument(f"--{name}", required=True)
    for name, default in (("variant", None), ("scope", None), ("sample", "large"), ("only", None)):
        p.add_argument(f"--{name}", default=default)
    p.add_argument("--expand", type=int, default=0, help="score each seed's window (expand_context)")
    p.add_argument("--pool-seeds", type=int, default=50, help="seeds scored per query")
    p.add_argument("--max-usd", type=float)
    p = sub.add_parser("show")
    p.add_argument("dataset")
    p.add_argument("--sample", default="tiny")
    p.add_argument("--scope", default="history")
    p.add_argument("--variant", default="question")
    p.add_argument("--chart", default="", help="comma-separated: coverage, rpf, rerank_cost, cost_saving, "
                                                "query_types, sizes, expansion, counts (default: all)")
    p.add_argument("--embedder", default=EMBEDDER)
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
    for name in ("dataset", "chunking", "sample"):
        p.add_argument(name)
    ints = lambda text: tuple(int(x) for x in text.split(","))  # noqa: E731
    for name in ("run", "adhoc", "show"):  # the values reports are drawn at (rbench/config.py)
        sub.choices[name].add_argument("--budgets", type=ints, help="reader budgets, e.g. 500,1000,2000,4000")
        sub.choices[name].add_argument("--limits", type=ints, help="vector search limits shown, e.g. 10,20,50")
        sub.choices[name].add_argument("--readings", type=lambda text: tuple(text.split(",")),
                                       help="single-reading charts for these readings, e.g. every,first")
    args = parser.parse_args(argv)
    if args.command in ("run", "adhoc", "show"):
        from rbench import config

        config.configure(budgets=args.budgets, limits=args.limits, readings=args.readings)

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

        print(explain.run(args.dataset, args.chunking, args.qid, args.model, args.pool_seeds, args.expand, args.embedder))
    elif args.command == "examples":
        from rbench import examples

        print(examples.write(args.n))
    elif args.command == "estimate":
        result = _estimate([r for s in args.stages for r in grid.STAGES[s]], args.embedder)
        print(json.dumps(result["summary"], indent=1))
    elif args.command in ("run", "adhoc"):
        if args.command == "run":
            runs = grid.STAGES[args.stage]
            if args.models:  # the stage's settings with these models: any registered reranker, no code change
                unknown = set(args.models.split(",")) - set(RERANKERS)
                if unknown:
                    raise SystemExit(f"unknown model(s) {sorted(unknown)}; registered: {sorted(RERANKERS)}")
                runs = [replace(r, models=tuple(args.models.split(","))) for r in runs]
        else:
            runs = grid.adhoc(args.datasets.split(","), args.chunkings.split(","), args.models.split(","),
                              variant=args.variant, scope=args.scope, sample=args.sample, expand=args.expand,
                              pool_seeds=args.pool_seeds)
            print(json.dumps(_estimate(runs, args.embedder)["summary"], indent=1))
        if args.only in (None, "prepare"):
            for run in runs:
                pipeline.prepare(run, args.embedder)
        if args.only in (None, "rerank"):
            for run in runs:
                for m in run.models:
                    pipeline.rerank(run, m, args.max_usd, args.embedder)
        if args.only in (None, "report"):
            for path in pipeline.simulate_and_report(runs, args.embedder):
                print(path)
    elif args.command == "show":
        from rbench.report import figures
        from rbench.report.data import load

        charts = [c for c in args.chart.split(",") if c]
        unknown = set(charts) - set(figures.CHARTS)
        if unknown:
            raise SystemExit(f"unknown chart(s) {sorted(unknown)}; choose from {sorted(figures.CHARTS)}")
        figures.show(load(args.embedder, args.dataset, args.scope, args.variant, args.sample), charts or None)
    elif args.command == "compare-embedders":
        from rbench import compare_embedders

        print(compare_embedders.write(args.server, args.dataset, args.size, args.questions))
    elif args.command == "embed-fidelity":
        print(json.dumps(preflight.embed_fidelity(args.dataset, args.chunking), indent=1))
    elif args.command == "smoke":
        problems = preflight.smoke(args.models.split(","))
        print("smoke ok" if not problems else "\n".join(problems))
    elif args.command == "fidelity":
        print(json.dumps(preflight.fidelity(args.dataset, args.chunking, args.sample), indent=1))
