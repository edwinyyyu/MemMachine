# Naming v2 — canonical artifact names

Artifacts in this directory were renamed on 2026-08-07. `NAMING.md` (v1)
documents the *historical* names, which are still what the run scripts and the
handoff notes talk about; this file documents what is actually on disk now and
how the two are bridged.

No symlinks are involved. Every file carries its canonical name directly.

## Canonical name

```
{taghash}__{segmenter}__{model}-{reasoning}__emb{embedder}[__qaprompt]
        __nb{N}__v{N}e{N}l{N}__bm{weight}__rerankoff
        __ts{format}__pv{promptvariant}__rep{N}__{stage}[__{evaltags}]{ext}
```

Example:

```
397beb02__tslimv3__54nano-l__embgemma__qaprompt__nb8__v28e0l10__bm50
        __rerankoff__tsshort__rep1__eval__seg-mini-mb-c14.json
```

= terse-decoupled-slim-v3 segmenter, gpt-5.4-nano @ low, embeddinggemma with
the QA-task query prompt, neighbour window 8, vector-limit 28 / expand-context 0
/ max-segments 10, BM25 additive w=0.5, reranker off, short timestamps, rep 1 —
eval stage, answered from segment context, gpt-5-mini judge, mem0-bench, cats 1-4.

Properties:

- **Fixed slot order, defaults omitted** — two runs' names differ only where
  their configs differ, so `ls | sort` puts an ablation's arms side by side.
- **`{taghash}` = `sha1(legacy_tag)[:8]`** — not a sequential id. This is what
  makes the name *derivable*: a script can compute the canonical name of a run
  that does not exist yet, with no counter and no manifest lookup.
- **Ablation axes survive** — `__qaprompt__` and `__tsshort` are visible slots.
  The v1 names buried both mid-string.
- **Invalid runs are loud** — `--answer-with-raw-events` renders as
  `__INVALIDrawev__`, not a quiet `rawev` token.
- **`{stage}`** is `eval` / `search` / `locomo` / `logeval` / `logsearch` /
  `logingest`. Keeping the log stage in the name is what removed the 309
  `__dupN` collisions an earlier revision produced.

## Writing scripts against it

Scripts keep using the legacy tag strings they always did; wrap any artifact
filename in `A()`:

```python
from artifacts import A, sibling, pattern

db   = A(f"locomo-{tag}.sqlite")        # -> canonical path
out  = A(f"search-{stag}.json")
srch = sibling(eval_path, "search", ".json")   # same run, other stage
evals = glob.glob(pattern("eval", ".json"))    # all eval artifacts
```

| function | use |
|---|---|
| `A(name)` | legacy artifact name -> canonical name on disk |
| `legacy(name)` | canonical -> original legacy name (for reading old notes) |
| `sibling(name, kind, ext)` | swap stage within one run |
| `pattern(kind, ext)` | glob pattern for a stage |
| `exists(name)` | does this artifact exist under its canonical name |

Two rules:

- **Never pass a glob to `A()`.** It resolves concrete names only; use
  `pattern()`.
- **Do not do string surgery on filenames.** The old idiom — strip `eval-`,
  strip `-mini-mb-c14`, prepend `search-` — silently produces garbage on
  canonical names. Use `sibling()`.

`A()` resolves via `artifact_map.json` (exact, built at rename time) and falls
back to deriving the name with the same decoder. Derivation was verified to
agree with the map on all 4173 renamed files, so re-running a script finds its
own prior outputs instead of duplicating them.

## Files

| path | what |
|---|---|
| `artifacts.py` | the resolver — import this |
| `artifact_map.json` | legacy -> canonical, 4173 entries |
| `rename_undo.json` | reverse the rename (`from`/`to` pairs) |
| `MANIFEST.csv` / `.json` | one row per run: hash, canonical name, original tag, decoded config, file list, size |
| `decode.py` | token -> config decoder; every mapping traced to a source |
| `manifest.py` | rebuilds the manifest and canonical names |
| `~/Documents/locomo-results.ods` | per-eval metrics (c1234/c124 micro+macro, exact o200k tokens/q) + token legend |

## Known residue

- **26 files keep legacy names**: 24 stale `.sqlite-wal` / `-shm` sidecars whose
  parent DB no longer exists, and 2 malformed names containing spaces
  (`...fb.sqlite rawseg-llm-v1`) from a shell-quoting accident at creation.
- **2 tokens still undecoded**, affecting 5 runs — listed on the `legend` sheet
  of the ODS marked `NEEDS USER INPUT`.
- Handoff notes and `CLAUDE.md` still cite legacy names. `legacy()` translates
  in that direction when you need it.

## Reverting

```sh
python3 -c "
import json,os
for e in reversed(json.load(open('rename_undo.json'))):
    if os.path.exists(e['to']): os.rename(e['to'], e['from'])
"
```

That restores filenames only. The scripts' `A()` calls are harmless either way
— `A` falls back to returning the name unchanged if the map is absent.
