#!/usr/bin/env python3
"""Build a run manifest + canonical names for the locomo artifact directory.

A "run" is a tag: the shared stem joining locomo-{tag}.sqlite,
search-{tag}.json, eval-{tag}-{judge}.json and log-*-{tag}.out.

Canonical name, fixed slot order, defaults omitted:

  {id}__{segmenter}__{segmodel}__{embedder}__nb{N}__v{N}e{N}l{N}__{bm25}
      __{answerpath}__{judge}__{group}__rep{N}

Slots are always in this order, so `ls` sorts runs into comparable groups
and any two names differ only where the configs differ.
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode import decode  # noqa: E402

KINDS = ("eval", "search", "locomo", "ingest", "log")


def slug(s, maxlen=28):
    s = re.sub(r"[^A-Za-z0-9]+", "", str(s))
    return s[:maxlen] if s else ""


def tag_hash(tag):
    """Stable 8-hex digest of the original tag: makes canonical names unique
    and derivable for runs that do not exist yet (no counter/state needed)."""
    import hashlib
    return hashlib.sha1(tag.encode()).hexdigest()[:8]


def canonical(rid, d):
    """Deterministic, fixed-slot name. Omits argparse defaults.
    `rid` is the tag hash, not a sequential id, so new runs can be named
    without consulting the manifest."""
    p = [f"{rid}"]
    seg = slug(d.get("segmenter", "")) or "unkseg"
    p.append(seg)
    mdl = d.get("segmenter_model", "")
    if mdl:
        r = (d.get("segmenter_reasoning") or "")[:1]
        p.append(slug(mdl).replace("gpt", "") + (r and f"-{r}"))
    emb = d.get("embedder", "")
    if emb and "default" not in emb:
        short = {"google/embeddinggemma-300m": "gemma", "embed-large": "emblarge",
                 "all-MiniLM-L6-v2": "minilm"}.get(emb.split(" ")[0], "")
        p.append("emb" + (short or slug(emb.split("/")[-1].split()[0], 12)))
    # the QA-vs-default query prompt IS the ablation axis -- must survive
    qp = d.get("embedder_query_prompt", "")
    if "question answering" in qp:
        p.append("qaprompt")
    ea = d.get("embed_text_ablation", "")
    if ea:
        p.append("embabl" + slug(ea, 16))
    if d.get("neighbor_window") not in (None, ""):
        p.append("nb" + slug(d["neighbor_window"], 12))
    v = d.get("vector_search_limit", "")
    e = d.get("expand_context", "")
    l = d.get("max_num_segments", "")
    if v != "" or e != "" or l != "":
        p.append(f"v{v}e{e}l{l}")
    bm = d.get("bm25_fusion", "")
    if bm and "default" not in bm:
        if "additive" in bm:
            w = re.search(r"([\d.]+)$", bm)
            p.append("bm" + (w.group(1).replace("0.", "") if w else "add"))
        elif "BM25-only" in bm:
            p.append("bm25only")
        elif bm.startswith("none"):
            p.append("bmnone")
    if "disabled" in d.get("reranker", ""):
        p.append("rerankoff")
    ap = d.get("answer_path", "")
    if "raw events" in ap:
        p.append("INVALIDrawev")
    j = d.get("judge_model", "")
    if j and "default" not in j:
        p.append("j" + slug(j).replace("gpt", ""))
    jv = d.get("judge_variant", "")
    if jv and "default" not in jv:
        p.append(slug(jv))
    a = d.get("answerer_model", "")
    if a:
        p.append("a" + slug(a).replace("gpt", ""))
    g = d.get("group", "")
    if "conversations 0-3" in g:
        p.append("conv0to3")
    elif "conv " in g:
        p.append(slug(g))
    ts = d.get("timestamp_format", "")
    if ts:
        p.append("ts" + slug(ts, 10))
    pv = d.get("prompt_variant", "")
    if pv:
        p.append("pv" + slug(pv, 12))
    if d.get("prompt_modifier"):
        p.append("tight")
    sv = d.get("segmenter_variant", "")
    if sv:
        p.append(slug(sv.split(" ")[0], 10))
    if d.get("repetition") not in (None, ""):
        p.append(f"rep{d['repetition']}")
    name = "__".join(x for x in p if x)
    # hard guarantee: filesystem-safe, no separators or spaces leak through
    return re.sub(r"[^A-Za-z0-9_.-]", "", name)


# trailing tokens that belong to the EVAL stage, not the run identity.
# Generalises summarize_runs.py's fixed suffix list so miniJ/gpt5A also collapse.
EVAL_SUFFIX = {"mini", "gpt5", "41mini", "4omini", "mb", "mc", "c14",
               "rejudge", "verify", "amem0v2", "miniJ", "gpt5J", "miniA",
               "gpt5A", "seg", "rawev"}


def tag_of(fn):
    """Strip kind prefix, extension and eval-stage suffixes -> run tag."""
    base = fn
    for ext in (".vec.sqlite", ".sqlite", ".json", ".out", ".txt"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    kind = "other"
    for k in KINDS:
        if base.startswith(k + "-"):
            base, kind = base[len(k) + 1:], k
            break
    # log-eval-... / log-search-... carry a second kind prefix. Keep the stage
    # in the kind (logeval / logsearch / logingest) so two logs of the same run
    # never collide -- that collision is what forced __dupN suffixes before.
    if kind == "log":
        for k in ("eval", "search", "ingest"):
            if base.startswith(k + "-"):
                base, kind = base[len(k) + 1:], "log" + k
                break
    toks = base.split("-")
    while len(toks) > 1 and toks[-1] in EVAL_SUFFIX:
        toks.pop()
    return "-".join(toks), kind


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    os.chdir(src)
    files = [f for f in os.listdir(".") if os.path.isfile(f)
             and f.endswith((".json", ".sqlite", ".out", ".txt"))]
    runs = {}
    for f in files:
        tag, kind = tag_of(f)
        runs.setdefault(tag, {"tag": tag, "files": [], "bytes": 0})
        runs[tag]["files"].append(f)
        runs[tag]["bytes"] += os.path.getsize(f)

    rows = []
    for i, tag in enumerate(sorted(runs), start=1):
        r = runs[tag]
        d = decode(tag)
        rid = tag_hash(tag)
        rows.append({
            "run_id": f"R{i:04d}",
            "tag_hash": rid,
            "canonical_name": canonical(rid, d),
            "tag_original": tag,
            "n_files": len(r["files"]),
            "total_mb": round(r["bytes"] / 1048576, 2),
            "files": " | ".join(sorted(r["files"])),
            "segmenter": d.get("segmenter", ""),
            "segmenter_model": d.get("segmenter_model", ""),
            "embedder": d.get("embedder", ""),
            "neighbor_window": d.get("neighbor_window", ""),
            "vector_search_limit": d.get("vector_search_limit", ""),
            "expand_context": d.get("expand_context", ""),
            "max_num_segments": d.get("max_num_segments", ""),
            "bm25_fusion": d.get("bm25_fusion", ""),
            "reranker": d.get("reranker", ""),
            "answer_path": d.get("answer_path", ""),
            "judge_model": d.get("judge_model", ""),
            "judge_variant": d.get("judge_variant", ""),
            "answerer_model": d.get("answerer_model", ""),
            "group": d.get("group", ""),
            "repetition": d.get("repetition", ""),
            "INVALID": d.get("INVALID", ""),
            "undecoded_tokens": d.get("undecoded_tokens", ""),
        })

    os.makedirs(outdir, exist_ok=True)
    cols = list(rows[0])
    with open(os.path.join(outdir, "MANIFEST.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(outdir, "MANIFEST.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"runs: {len(rows)}   files covered: {sum(r['n_files'] for r in rows)}")
    print(f"invalid-methodology runs: {sum(1 for r in rows if r['INVALID'])}")
    print(f"runs with undecoded tokens: {sum(1 for r in rows if r['undecoded_tokens'])}")
    dupes = len(rows) - len({r["canonical_name"] for r in rows})
    print(f"canonical-name collisions: {dupes}")
    return rows


if __name__ == "__main__":
    main()
