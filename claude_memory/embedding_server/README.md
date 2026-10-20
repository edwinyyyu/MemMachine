# embeddinggemma-300m embedding server

claude-memory's only embedding model, google/embeddinggemma-300m, runs in a local
[llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (MIT license) on the
Metal GPU, behind an OpenAI-compatible `POST /v1/embeddings` on `127.0.0.1:9931`. The
client is `claude_memory/embed_server.py`.

It replaced sentence-transformers inside the daemon, where torch's GPU allocator kept the
memory of the largest batch it had ever run: the daemon measured 8.3 GB before the switch
and ~330 MB after, plus ~200 MB for the server. The vectors are the same (below), so
existing stores need no re-embedding.

## Install

```bash
claude_memory/embedding_server/setup.sh                # ~35s; macOS on Apple Silicon only
uv pip install sentence-transformers                   # only for the verification below
uv run python -m claude_memory.embedding_server.verify # ~2 min, ~1.5 GB peak
```

`setup.sh` downloads llama.cpp's official release (build b11669, v0.7.0) and its source
from github.com/ggml-org, and converts the model to an F32 GGUF from the local Hugging Face
cache with `convert_local.py`. It writes only inside this directory (all git-ignored: the
files are large, and the model is under the Gemma license). The model is gated: accept its
license on huggingface.co and download it once while logged in, for example by loading it
with sentence-transformers. No third-party copy of the model or tokenizer is used; the
converter would want Google's `tokenizer.model`, which sentence-transformers never
downloads, so `convert_local.py` rebuilds the same tokenizer tables from `tokenizer.json`
(see its docstring). Two fresh installs produced byte-identical model files.

## How it starts and stops

Nothing is installed system-wide. The claude-memory daemon starts `run.sh` on its first
embedding call, detached so it outlives the daemon, and starts it again if it has died.
Override the script with `embed_server` in `<home>/config.json` or
`CLAUDE_MEMORY_EMBED_SERVER`. An experiment can run `run.sh` itself, or use the server
while claude-memory's is up.

- After 30 idle minutes (`--sleep-idle-seconds 1800`) the server unloads the model, down to
  ~97 MB from ~165 MB. The next request waits ~0.6s for the reload and gets the same vector.
  `/health` and `/v1/models` answer without waking it.
- Binding the port is the single-instance guard: a second server exits with code 1.
- With `LOG` set, as the client sets it, warnings and errors go to `logs/server.log` for
  the current run and `logs/server.log.1` for the previous one. `PORT` picks the port.
- A crash drops in-flight connections at once; the client respawns the server and resends
  (verified with SIGKILL: same vectors). SIGTERM is different: llama-server stops
  processing but keeps in-flight connections open until the client gives up (the client's
  timeout is 300s). Stop it while idle, or with SIGKILL, which is safe since it keeps no
  state.
- A server that exits before ever answering (for example, not installed) makes the client
  fail at once with the log's path.

## Limits

Measured on 2026-10-19 on an M3 Pro with 18 GB.

- One input: at most 2,048 tokens including BOS and EOS. Longer input is rejected with
  HTTP 500, so the client truncates it (rule 2 below).
- Concurrency: 4 inputs at a time (`-np 4`); further requests queue inside the server,
  where the wait counts against the client's timeout. The client keeps at most 4 requests
  in flight, each of at most 32 inputs.
- Speed: ~0.4s per 2,048-token input, since the 2,048-token batch fits one such input per
  step; 32 of them take ~12s. Short inputs share a step: 400 short documents took 5.1s
  including a cold start, and 519 mixed texts took 10-11s where sentence-transformers took
  63-68s. One search query took 44 ms (median) where sentence-transformers took 120 ms.
- Memory: 140 MB after load, 154-225 MB under 8 concurrent clients sending 32 long texts
  each. The weights are a memory-mapped file the OS can evict without swapping.

## Fidelity against sentence-transformers

Reference: sentence-transformers 5.6.0, set up as claude-memory used it.

- On 519 real texts (103 queries, 416 documents): minimum cosine 0.999997, the same
  top-10 documents for every query. sentence-transformers on the GPU versus the CPU, as a
  noise floor: 0.99999976. ggml-org's published GGUF, which exists only as 8-bit Q8_0:
  0.99959, with 3 of 103 queries getting a different top result; hence the F32 conversion.
- Through the live daemon: the same 30 searches returned the same hits before and after
  the switch; two swapped the order of near-tied hits, and scores moved by at most 0.00043.
- `verify.py` re-checks this on texts from this repository, plus over-length documents,
  special-token strings and the failure paths above.

## Client rules

`claude_memory/embed_server.py` implements both; any other client must too, or its
vectors will not match.

1. Prepend embeddinggemma's prompt yourself, because the server applies none:
   `task: search result | query: ` for queries, `title: none | text: ` for documents.
2. Truncate to sentence-transformers' 2,048 tokens: tokenize with the server's
   `POST /tokenize` (`add_special: false`) and send token IDs `[2] + ids[:2046] + [1]`.
   Token-ID input gets no BOS (2) or EOS (1) from the server; without them the cosine
   drops to ~0.99. Text of at most 2,046 UTF-8 bytes can be sent as is, since every token
   covers at least one byte.
