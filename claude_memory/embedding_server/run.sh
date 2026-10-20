#!/bin/sh
# Serve embeddinggemma-300m over an OpenAI-compatible /v1/embeddings endpoint on the Metal GPU.
# Installed by setup.sh; started on demand by claude_memory/embed_server.py.
# All buffers are reserved at startup from -c/-b/-ub/-np, so memory does not grow with traffic.
# --cache-ram 0 disables the host-memory prompt cache, which otherwise defaults to 8 GB.
# -lv 2 logs warnings and errors only; the default logs a line per request.
# --sleep-idle-seconds unloads the model after 30 idle minutes and reloads it on the next request.
DIR=$(cd "$(dirname "$0")" && pwd)
SERVER="$DIR/bin/llama.cpp/llama-server"
MODEL="$DIR/models/embeddinggemma-300M-F32.gguf"
# With LOG set (the client sets it), keep only this run's and the previous run's log,
# so a restart loop cannot grow it without bound.
if [ -n "${LOG:-}" ]; then
	mkdir -p "$(dirname "$LOG")"
	if [ -f "$LOG" ]; then mv "$LOG" "$LOG.1"; fi
	exec >"$LOG" 2>&1
fi
if [ ! -x "$SERVER" ] || [ ! -f "$MODEL" ]; then
	echo "embedding server is not installed: run $DIR/setup.sh" >&2
	exit 1
fi
exec "$SERVER" -m "$MODEL" \
	--embeddings -c 8192 -b 2048 -ub 2048 -np 4 --cache-ram 0 \
	--no-webui --offline -lv 2 --sleep-idle-seconds 1800 \
	--host 127.0.0.1 --port "${PORT:-9931}" "$@"
