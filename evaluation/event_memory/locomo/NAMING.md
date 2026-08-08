# File naming conventions

Every file I write follows the pattern `{kind}-{pipeline}[-{group}][-{search}][-{eval}].{ext}` where each segment is built from the abbreviations below.

## Kind (leading token)

| token   | meaning                                                            |
|---------|--------------------------------------------------------------------|
| `locomo`  | the SQLite DB holding both segment store and vector store         |
| `ingest`  | log file produced by `locomo_ingest_st.py`                        |
| `search`  | retrieval+answerer output from `locomo_search_st.py`              |
| `eval`    | judge output from `locomo_evaluate.py`                            |

## Pipeline = `{segmenter}-{embedder}-{deriver}`

### Segmenter

| token       | meaning                                                                                |
|-------------|----------------------------------------------------------------------------------------|
| `textseg`   | TextSegmenter — recursive char splitter, no LLM                                        |
| `rw-v{N}`   | RewriteSegmenter, prompt iteration N (e.g. `rw-v7`, `rw-v14`); LLM rewrite each chunk |

### Embedder

| token   | meaning                                                                              |
|---------|--------------------------------------------------------------------------------------|
| `st-eg` | SentenceTransformerEmbedder with `google/embeddinggemma-300m` (768-dim, cosine)      |
| *(absent)* | default `OpenAIEmbedder` with `text-embedding-3-small` (1536-dim)                 |

### Deriver

| token | meaning                                                                                       |
|-------|-----------------------------------------------------------------------------------------------|
| `dw`  | WholeTextDeriver — one derivative per segment, equal to the segment's full text               |
| `ds`  | SentenceTextDeriver — one derivative per sentence in the segment                              |
| `dl`  | LLMTextDeriver — one derivative per LLM call (currently the v65 prompt at gpt-5-nano low)     |

LLM-based segmenters that ran with a non-default segmenter model+reasoning carry a model suffix on the pipeline (e.g. `rw-v14-m54nl` = model `gpt-5.4-nano`, reasoning `low`); the bare-bones `textseg` runs do not need this because no LLM is invoked.

## Group (optional)

| token         | meaning                                                                  |
|---------------|--------------------------------------------------------------------------|
| `g{N}`        | a single conversation (e.g. `g3` = Joanna & Nate)                       |
| `g{N}to{M}`   | a contiguous range of groups, used for partial-ingest log files          |
| *(absent)*    | full bench: all 10 conversations                                         |

## Search = `v{N}-e{M}-l{K}[-{modifier}]`

`v`, `e`, `l` map directly to `locomo_search_st.py` flags.

| token  | flag                       | meaning                                               |
|--------|----------------------------|-------------------------------------------------------|
| `v{N}` | `--vector-search-limit N`  | candidates pulled from the vector index              |
| `e{M}` | `--expand-context M`       | neighbouring segments added around each hit          |
| `l{K}` | `--max-num-segments K`     | hard cap on segments returned to the answerer        |

| modifier   | meaning                                                              |
|------------|----------------------------------------------------------------------|
| `rnull`    | `--no-reranker` (rank by embedding similarity only)                  |
| `bmf`      | `--bm25-fusion additive --bm25-fusion-weight 0.5` (BM25 mixed in)    |
| *(absent)* | this eval's default: `--no-reranker`, no BM25 fusion                 |

## Eval = `{judge}[-{filter}]`

| token      | meaning                                                                                   |
|------------|-------------------------------------------------------------------------------------------|
| `mini`     | `--judge-model gpt-5-mini --judge-variant mem0-bench`                                     |
| *(absent)* | default `gpt-5` judge with `mem0-bench` variant                                           |

| filter   | meaning                                                                |
|----------|------------------------------------------------------------------------|
| `c14`    | categories 1-4 only — cat 5 (adversarial) dropped before judging       |

## Extension

| ext      | meaning                                                          |
|----------|------------------------------------------------------------------|
| `.sqlite`| segment store + vector store, one file per pipeline               |
| `.json`  | structured data (search results or judge results)                |
| `.out`   | captured stdout/stderr from the corresponding script             |

## Examples

| filename                                                    | reading                                                                                                  |
|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `locomo-textseg-st-eg-dw.sqlite`                            | DB for textseg + embeddinggemma + whole-deriver pipeline                                                 |
| `ingest-textseg-st-eg-dw-g0.out`                            | ingest log for that pipeline, group 0 only                                                               |
| `ingest-textseg-st-eg-dw-g1to9.out`                         | ingest log for groups 1-9                                                                                |
| `search-textseg-st-eg-dw-v160-e3-l40.json`                  | full-bench search at v=160, e=3, l=40 (Mem0 top-50 budget), no reranker, no BM25                          |
| `eval-textseg-st-eg-dw-v160-e3-l40-mini-c14.json`           | gpt-5-mini judge with mem0-bench variant, cat 1-4 only                                                   |
