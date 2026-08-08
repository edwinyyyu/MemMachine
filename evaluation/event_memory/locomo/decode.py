#!/usr/bin/env python3
"""Decode LoCoMo run filenames into human-readable configuration.

Every mapping below is traced to a source:
  [cli]      = argparse flag in locomo_search_st.py / locomo_evaluate.py /
               locomo_ingest_st.py
  [NAMING]   = NAMING.md
  [HANDOFF]  = HANDOFF.md prose
  [INFER]    = inferred from usage, NOT confirmed -> flagged in output
Tokens that match nothing are returned in `undecoded` so ambiguity stays visible.
"""
import re

# token-pattern -> (field, renderer, source)
JUDGE_MODEL = {"mini": "gpt-5-mini"}
JUDGE_VARIANT = {"mb": "mem0-bench", "mc": "mem0-classic"}
DERIVER = {"dw": "WholeTextDeriver (one derivative = full segment)",
           "ds": "SentenceTextDeriver (one per sentence)",
           "dl": "LLMTextDeriver (one per LLM call)"}
EMBEDDER = {"st-eg": "google/embeddinggemma-300m (768d)",
            "emg": "google/embeddinggemma-300m (768d)",
            "gemma": "google/embeddinggemma-300m (768d)",
            "emgemma": "google/embeddinggemma-300m (768d)",
            "eg": "google/embeddinggemma-300m (768d)",
            "emblg": "embed-large",
            "minilm": "all-MiniLM-L6-v2"}
FUSION = {"rrf": "RRF (Reciprocal Rank Fusion, k=60)",
          "rsf": "RSF (Relative Score Fusion, max-normalized)"}
MODEL_FAM = {"54": "gpt-5.4", "5": "gpt-5"}
MODEL_SIZE = {"n": "nano", "m": "mini"}
REASONING = {"l": "low", "m": "medium", "h": "high"}

LEGEND = [
    ("v{N}", "--vector-search-limit N (candidates pulled from vector index; default 100)", "cli"),
    ("e{M}", "--expand-context M (neighbouring segments added around each hit; default 3)", "cli"),
    ("l{K}", "--max-num-segments K (hard cap on segments to answerer; default 20)", "cli"),
    ("rnull", "--no-reranker (rank by embedding similarity only)", "cli/NAMING"),
    ("bmf", "--bm25-fusion additive --bm25-fusion-weight 0.5", "cli/NAMING"),
    ("bmfa{NN}", "--bm25-fusion additive --bm25-fusion-weight 0.NN", "cli"),
    ("bmfrrf / bmfrsf", "--bm25-fusion rrf / rsf", "cli"),
    ("(no bm25 token)", "--bm25-fusion none (pure vector retrieval)", "cli"),
    ("nobm25", "BM25 explicitly disabled", "cli"),
    ("mini", "--judge-model gpt-5-mini (absent => gpt-5, the argparse default)", "cli"),
    ("mb / mc", "--judge-variant mem0-bench / mem0-classic", "cli"),
    ("c14", "--skip-category-5 (categories 1-4 only; cat5 = adversarial)", "cli"),
    ("m{54|5}{n|m}{l|m|h}", "segmenter model+reasoning, e.g. m54nl = gpt-5.4-nano @ low", "NAMING"),
    ("{54|5}{n|m}-{l|m|h}", "same as above, alternate spelling e.g. 54n-l", "NAMING"),
    ("nb{N}", "neighbour window N (nb0->nb8 sweep; 8 found to be the peak)", "HANDOFF"),
    ("nb{N}b", "neighbour window N, variant 'b' -- suffix meaning NOT documented", "INFER"),
    ("dw / ds / dl", "deriver: whole / sentence / LLM", "NAMING"),
    ("st-eg, emg, gemma", "SentenceTransformer embeddinggemma-300m", "NAMING"),
    ("emblg / minilm", "embed-large / all-MiniLM-L6-v2", "INFER"),
    ("(no embedder token)", "default OpenAIEmbedder text-embedding-3-small (1536d)", "NAMING"),
    ("textseg", "TextSegmenter (recursive char splitter, no LLM)", "NAMING"),
    ("rw-v{N}", "RewriteSegmenter, prompt iteration N", "NAMING"),
    ("tslimv{N}", "slim segmenter variant N", "INFER"),
    ("dchunk{N}", "--max-text-chunk-length N", "cli/INFER"),
    ("g{N} / g{N}to{M}", "conversation group index / range (absent => all 10)", "NAMING"),
    ("tsshort", "timestamp format 'short' (vs 'verbose'); see reanswer_tsfmt.py", "INFER"),
    ("rep{N}", "repetition N of an identical config", "INFER"),
    ("rawev", "--answer-with-raw-events -- INVALID METHODOLOGY per CLAUDE.md", "CLAUDE.md"),
    ("fb", "full bench -- all 10 conversations (same as 'full')", "user"),
    ("c2sub", "conversations 0-3 only (Joanna+Nate, Audrey+Andrew, James+John, Calvin+Dave); 630 q", "ingest logs"),
    ("gemmaqa", "embeddinggemma with QA-task query prompt; search-only re-run of the gemma DBs", "run_gemma_qaprompt.py"),
    ("gemma", "embeddinggemma with its packaged 'task: search result' query prompt", "run_gemma_qaprompt.py"),
    ("emb_nodate / nodate_all", "text_to_embed ablation: no datetime attached to embedded text", "user + run_embed_ablation_batch.py"),
    ("emb_{mcq perm}", "text_to_embed component ORDER (permutation of m/c/q)", "run_embed_ablation_batch.py"),
    ("e_{m|c|q subset}", "text_to_embed additive lattice: which components are included", "run_embed_ablation_batch.py"),
    ("cur", "the 'cur' baseline: terse-decoupled-v2 K=10 mini = 91.17", "run_embed_ablation_batch.py"),
    ("deco / decoupled", "decoupled segmenter/deriver", "run_embed_ablation_batch.py"),
    ("dateinstr", "prompt variant: modified date handling", "user"),
    ("min3p / fp", "segmenter prompt: minimal THIRD-person / FIRST-person rewrite", "locomo_ingest.py registry"),
    ("simporig", "original/simple prompt -- control arm opposite dateinstr", "filename slot analysis"),
    ("tight", "tightened prompt variant, stacks on dateinstr/simporig", "filename slot analysis"),
    ("amem0v2", "Mem0 answering prompt v2 (a = answerer)", "CLAUDE.md + slot"),
    ("41mini / 4omini", "judge model gpt-4.1-mini / gpt-4o-mini", "slot"),
    ("g{5|54}{n|m}{l|m|h}", "same as m-prefixed model spelling, e.g. g5nl = gpt-5-nano @ low", "slot"),
    ("dchunk{N} / wchunk{N} / c{N}", "--max-text-chunk-length N", "cli"),
    ("bm25only", "BM25-only retrieval channel, no vector", "run_bm25only_confirm.py"),
    ("bm{NN}", "bm25 additive weight (bm07=0.7, bm10=1.0)", "cli"),
    ("v{N}p", "vector-search-limit N -- trailing letter is a typo (p is keyboard-adjacent to '-'); only v28p exists", "user + slot"),
    ("{model}A / {model}J", "explicit ANSWERER / JUDGE model, e.g. gpt5A-miniJ = gpt-5 answerer + gpt-5-mini judge", "complementary-pair analysis"),
    ("text-sent", "raw text segmenter + sentence deriver (parallel to text-whole)", "registry"),
    ("droute1", "routed deriver (llm-routed; --routed-threshold)", "registry"),
    ("verify", "verification re-run", "slot"),
    ("rejudge", "judge re-run over existing search output", "slot"),
    ("seg", "answer from segment context (string_from_segment_context) -- the correct path", "HANDOFF/CLAUDE.md"),
]


def decode(stem):
    """stem: filename without kind prefix and extension."""
    toks = stem.split("-")
    out, used = {}, [False] * len(toks)
    notes = []

    def mark(i):
        used[i] = True

    for i, t in enumerate(toks):
        if used[i]:
            continue
        m = re.fullmatch(r"v(\d+)", t)
        if m: out["vector_search_limit"] = int(m.group(1)); mark(i); continue
        m = re.fullmatch(r"e(\d+)", t)
        if m: out["expand_context"] = int(m.group(1)); mark(i); continue
        m = re.fullmatch(r"l(\d+)", t)
        if m: out["max_num_segments"] = int(m.group(1)); mark(i); continue
        m = re.fullmatch(r"(rnull)?bmfa(\d+)", t)
        if m:
            if m.group(1): out["reranker"] = "disabled (embedding similarity only)"
            out["bm25_fusion"] = f"additive, weight 0.{m.group(2)}"
            mark(i); continue
        m = re.fullmatch(r"(rnull)?bmf(rrf|rsf)", t)
        if m:
            if m.group(1): out["reranker"] = "disabled (embedding similarity only)"
            out["bm25_fusion"] = FUSION[m.group(2)]
            mark(i); continue
        m = re.fullmatch(r"(rnull)?bmf", t)
        if m:
            if m.group(1): out["reranker"] = "disabled (embedding similarity only)"
            out["bm25_fusion"] = "additive, weight 0.5"
            mark(i); continue
        if t == "rnull":
            out["reranker"] = "disabled (embedding similarity only)"; mark(i); continue
        if t == "nobm25":
            out["bm25_fusion"] = "none (explicit)"; mark(i); continue
        m = re.fullmatch(r"nb(\d+)(b?)", t)
        if m:
            out["neighbor_window"] = int(m.group(1))
            if m.group(2): out["neighbor_variant"] = "b (undocumented)"
            mark(i); continue
        if t == "st" and i + 1 < len(toks) and toks[i + 1] == "eg":
            out["embedder"] = EMBEDDER["st-eg"]; mark(i); mark(i + 1); continue
        if t == "text" and i + 1 < len(toks) and toks[i + 1] in ("whole", "sent"):
            which = toks[i + 1]
            out["segmenter"] = f"text+{which} (raw text, {'whole' if which == 'whole' else 'sentence'} deriver)"
            out["deriver"] = DERIVER["dw" if which == "whole" else "ds"]
            mark(i); mark(i + 1); continue
        # rawseg-llm-v1 / rawseg-v3-dual: segmenter family from the
        # locomo_ingest.py registry; absorb the version tokens that follow.
        if t == "rawseg":
            parts = [t]
            j = i + 1
            while j < len(toks) and re.fullmatch(r"(llm|dual|v\d+\w*|nodate|parallel|richprompt)", toks[j]):
                parts.append(toks[j]); mark(j); j += 1
            out["segmenter"] = "-".join(parts) + " (raw-segment LLM family)"
            mark(i); continue
        if t in JUDGE_MODEL: out["judge_model"] = JUDGE_MODEL[t]; mark(i); continue
        if t in JUDGE_VARIANT: out["judge_variant"] = JUDGE_VARIANT[t]; mark(i); continue
        if t == "c14": out["category_filter"] = "cats 1-4 (cat5 adversarial skipped)"; mark(i); continue
        if t in DERIVER: out["deriver"] = DERIVER[t]; mark(i); continue
        if t in EMBEDDER: out["embedder"] = EMBEDDER[t]; mark(i); continue
        m = re.fullmatch(r"m(54|5)(n|m)(l|m|h)", t)
        if m:
            out["segmenter_model"] = f"{MODEL_FAM[m.group(1)]}-{MODEL_SIZE[m.group(2)]}"
            out["segmenter_reasoning"] = REASONING[m.group(3)]
            mark(i); continue
        m = re.fullmatch(r"(54|5)(n|m)", t)
        if m and i + 1 < len(toks) and toks[i + 1] in REASONING:
            out["segmenter_model"] = f"{MODEL_FAM[m.group(1)]}-{MODEL_SIZE[m.group(2)]}"
            out["segmenter_reasoning"] = REASONING[toks[i + 1]]
            mark(i); mark(i + 1); continue
        m = re.fullmatch(r"dchunk(\d+)", t)
        if m: out["max_text_chunk_length"] = int(m.group(1)); mark(i); continue
        m = re.fullmatch(r"g(\d+)(?:to(\d+))?", t)
        if m:
            out["group"] = f"conv {m.group(1)}" + (f"-{m.group(2)}" if m.group(2) else "")
            mark(i); continue
        m = re.fullmatch(r"rep(\d+)", t)
        if m: out["repetition"] = int(m.group(1)); mark(i); continue
        # longmemeval repetition spelling: base run, then -again, -again2 ...
        # Verified structurally: every `againN` file has a base file whose config
        # is identical minus the token. base = rep 1, again = rep 2, againN = N+1.
        if t == "again":
            out["repetition"] = 2; mark(i); continue
        m = re.fullmatch(r"again(\d+)", t)
        if m: out["repetition"] = int(m.group(1)) + 1; mark(i); continue
        # elision / context-formatting variants (longmemeval)
        if t in ("dots", "multidot", "elis", "emdash", "emdashout", "gap", "skip", "compact"):
            out["elision_format"] = t; mark(i); continue
        if t == "tsshort": out["timestamp_format"] = "short"; mark(i); continue
        if t == "rawev":
            out["answer_path"] = "raw events -- INVALID methodology (CLAUDE.md)"
            out["INVALID"] = "answer-with-raw-events"; mark(i); continue
        if t == "seg":
            out["answer_path"] = "segment context (string_from_segment_context; correct path)"
            mark(i); continue
        if t == "gpt5": out["judge_model"] = "gpt-5 (explicit)"; mark(i); continue
        if t == "4omini": out["judge_model"] = "gpt-4o-mini"; mark(i); continue
        if t == "rnullbmnone":
            out["reranker"] = "disabled (embedding similarity only)"
            out["bm25_fusion"] = "none"; mark(i); continue
        if t in ("full", "fb"):
            out["group"] = "all 10 conversations (full bench)"; mark(i); continue
        if t == "dateinstr":
            out["prompt_variant"] = "modified date handling in prompt"; mark(i); continue
        if t == "c2sub":
            out["group"] = "conversations 0-3 (4 of 10; 630 questions)"; mark(i); continue
        if t == "gemmaqa":
            # run_gemma_qaprompt.py: re-searches the SAME gemma DBs with the
            # QA-task query prompt. Search-only; documents untouched.
            out["embedder"] = EMBEDDER["gemma"]
            out["embedder_query_prompt"] = "task: question answering (QA-task variant)"
            mark(i); continue
        if t == "gemma":
            out["embedder"] = EMBEDDER["gemma"]
            out["embedder_query_prompt"] = "task: search result (packaged default)"
            mark(i); continue
        if t == "emb_nodate" or t == "nodate_all":
            out["embed_text_ablation"] = "no datetime attached to embedded text"
            mark(i); continue
        m = re.fullmatch(r"emb_([mcq]{2,3})", t)
        if m:
            out["embed_text_ablation"] = f"text_to_embed component order: {'-'.join(m.group(1))}"
            mark(i); continue
        m = re.fullmatch(r"e_([mcq]{1,3})", t)
        if m:
            out["embed_text_ablation"] = f"text_to_embed additive lattice, components: {'-'.join(m.group(1))}"
            mark(i); continue
        if t == "cur":
            out["baseline_ref"] = "current baseline (terse-decoupled-v2, K=10 mini = 91.17)"
            mark(i); continue
        if t in ("deco", "decoupled"):
            out["segmenter_mode"] = "decoupled segmenter/deriver"; mark(i); continue
        # --- explicit answerer (A) / judge (J) suffixes ---
        m = re.fullmatch(r"(gpt5|mini|41mini|4omini)([AJ])", t)
        if m:
            name = {"gpt5": "gpt-5", "mini": "gpt-5-mini",
                    "41mini": "gpt-4.1-mini", "4omini": "gpt-4o-mini"}[m.group(1)]
            out["answerer_model" if m.group(2) == "A" else "judge_model"] = name
            mark(i); continue
        # --- judge/answerer models ---
        if t == "41mini": out["judge_model"] = "gpt-4.1-mini"; mark(i); continue
        if t == "verify": out["rerun"] = "verification re-run"; mark(i); continue
        if t == "droute1":
            out["deriver_variant"] = "routed deriver (llm-routed; see --routed-threshold)"
            mark(i); continue
        if t == "rnullbmnoor":
            out["reranker"] = "disabled (embedding similarity only)"
            out["bm25_fusion"] = "'bmnoor' variant - MEANING UNCONFIRMED"
            mark(i); continue
        # compound segmenter names from the locomo_ingest registry
        if t in ("llm", "dual", "sent"):
            prev = toks[i - 1] if i else ""
            if t == "sent":
                out["segmenter"] = "text+sentence (raw text, sentence deriver)"
                out["deriver"] = DERIVER["ds"]
            else:
                out["segmenter"] = (out.get("segmenter") or prev) + "-" + t
            mark(i); continue
        if t == "amem0v2":
            out["answerer_prompt"] = "Mem0 answering prompt v2 (required when answerer is gpt-5)"
            mark(i); continue
        # g-prefixed model spelling (g5nl == m5nl)
        m = re.fullmatch(r"g(54|5)(n|m)(l|m|h)", t)
        if m:
            out["segmenter_model"] = f"{MODEL_FAM[m.group(1)]}-{MODEL_SIZE[m.group(2)]}"
            out["segmenter_reasoning"] = REASONING[m.group(3)]
            mark(i); continue
        # --- chunk length: dchunk110 / wchunk110 / c250 ---
        m = re.fullmatch(r"[dw]chunk(\d+)", t)
        if m: out["max_text_chunk_length"] = int(m.group(1)); mark(i); continue
        m = re.fullmatch(r"c(\d{3,4})", t)
        if m: out["max_text_chunk_length"] = int(m.group(1)); mark(i); continue
        # --- retrieval channel ---
        if t == "bm25only":
            out["bm25_fusion"] = "BM25-only channel (no vector retrieval)"; mark(i); continue
        if t == "rnullbmfnone":
            out["reranker"] = "disabled (embedding similarity only)"
            out["bm25_fusion"] = "none"; mark(i); continue
        m = re.fullmatch(r"bm(\d{2})", t)
        if m:
            w = int(m.group(1))
            out["bm25_fusion"] = f"additive, weight {w/10:.1f}" if w in (7, 10) else f"additive, weight 0.{m.group(1)}"
            mark(i); continue
        # --- prompt arms (same slot as dateinstr) ---
        if t == "simporig":
            out["prompt_variant"] = "original/simple prompt (control arm vs dateinstr)"; mark(i); continue
        if t == "dateinstrcond":
            out["prompt_variant"] = "conditional date-handling instruction"; mark(i); continue
        if t == "tight":
            out["prompt_modifier"] = "tightened prompt variant"; mark(i); continue
        if t == "rejudge":
            out["rerun"] = "judge re-run over existing search output"; mark(i); continue
        # --- segmenter prompt family (registry: locomo_ingest.py) ---
        if t == "min3p":
            out["segmenter_variant"] = "min3p = minimal THIRD-person rewrite (rewrite-v22-min3p)"
            mark(i); continue
        if t == "fp":
            out["segmenter_variant"] = "fp = FIRST-person rewrite (rewrite-v22-fp)"; mark(i); continue
        if t == "nq":
            out["segmenter_variant"] = "nq variant (rewrite-v22-qkey-min3p-nq)"; mark(i); continue
        if t == "egs":
            out["embedder"] = EMBEDDER["gemma"]; mark(i); continue
        # --- timestamp format variants ---
        if t.startswith("tsiso"):
            out["timestamp_format"] = "ISO date (" + t + ")"; mark(i); continue
        # --- asymmetric neighbour window nb8a1 / nb8a8 ---
        m = re.fullmatch(r"nb(\d+)a(\d+)", t)
        if m:
            out["neighbor_window"] = f"{m.group(1)} / asymmetric arm {m.group(2)}"
            mark(i); continue
        # --- deriver variant family: d<letter(s)>1, cd1 ---
        m = re.fullmatch(r"(c?d[a-z]{0,3})(\d+)", t)
        if m and m.group(1) not in ("d",):
            out["deriver_variant"] = f"{t} (deriver variant; see locomo_ingest deriver registry)"
            mark(i); continue
        if t == "dp":
            out["deriver_variant"] = "dp (deriver variant)"; mark(i); continue
        # --- vector limit with modifier suffix, e.g. v28p ---
        # v28p: stray trailing letter. 'p' is keyboard-adjacent to '-', only ever
        # appears as v28p, and the typo propagated from the search tag into the
        # eval names. Treated as the plain vector-search-limit.
        m = re.fullmatch(r"v(\d+)([a-z])", t)
        if m:
            out["vector_search_limit"] = int(m.group(1))
            out["vector_limit_modifier"] = f"'{m.group(2)}' = likely typo (keyboard-adjacent to '-'); read as v{m.group(1)}"
            mark(i); continue
        if t == "textseg":
            out["segmenter"] = "TextSegmenter (recursive char splitter, no LLM)"; mark(i); continue
        m = re.fullmatch(r"rw", t)
        if m and i + 1 < len(toks) and re.fullmatch(r"v\d+\w*", toks[i + 1]):
            out["segmenter"] = f"RewriteSegmenter prompt {toks[i+1]}"; mark(i); mark(i + 1); continue

    # first unused token is the pipeline/segmenter identity
    rest = [toks[i] for i in range(len(toks)) if not used[i]]
    if "segmenter" not in out and rest:
        out["segmenter"] = rest.pop(0)
    out["undecoded_tokens"] = " ".join(rest)

    # defaults made explicit (argparse defaults, when token absent)
    out.setdefault("bm25_fusion", "none (default)")
    out.setdefault("reranker", "enabled (default)")
    out.setdefault("judge_model", "gpt-5 (default)")
    out.setdefault("judge_variant", "mem0-classic (default)")
    out.setdefault("embedder", "text-embedding-3-small 1536d (default)")
    out.setdefault("group", "all 10 conversations")
    return out


def human(d):
    bits = []
    for k, lbl in (("segmenter", "segmenter"), ("segmenter_model", "seg-model"),
                   ("segmenter_reasoning", "reasoning"), ("deriver", "deriver"),
                   ("embedder", "embedder"), ("neighbor_window", "neighbor-window"),
                   ("vector_search_limit", "vec-limit"), ("expand_context", "expand-ctx"),
                   ("max_num_segments", "max-segments"), ("bm25_fusion", "bm25"),
                   ("reranker", "reranker"), ("judge_model", "judge"),
                   ("judge_variant", "judge-variant"), ("timestamp_format", "ts-format"),
                   ("answer_path", "answer-path"), ("prompt_variant", "prompt-variant"),
                   ("embedder_query_prompt", "embed-query-prompt"),
                   ("embed_text_ablation", "embed-text"), ("segmenter_mode", "seg-mode"),
                   ("segmenter_variant", "seg-variant"), ("deriver_variant", "deriver-variant"),
                   ("prompt_modifier", "prompt-mod"), ("answerer_prompt", "answerer-prompt"),
                   ("answerer_model", "answerer"),
                   ("max_text_chunk_length", "chunk-len"), ("vector_limit_modifier", "vec-limit-mod"),
                   ("rerun", "rerun"),
                   ("group", "group"),
                   ("repetition", "rep"), ("category_filter", "cat-filter")):
        if k in d and d[k] != "":
            bits.append(f"{lbl}={d[k]}")
    return "; ".join(bits)
