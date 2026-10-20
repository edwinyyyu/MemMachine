#!/bin/sh
# Install the local embedding server into this directory:
#   bin/llama.cpp/       llama.cpp's official macOS arm64 release (ggml-org, MIT license)
#   src/llama.cpp/       the matching source, used only for its GGUF converter
#   .venv-convert/       an isolated Python environment for the converter
#   models/              embeddinggemma-300m as an F32 GGUF, converted from the local
#                        Hugging Face cache (no third-party copy of the model is used)
# Nothing outside this directory is changed apart from uv's package cache. Re-running skips
# finished steps. Needs curl, uv, and google/embeddinggemma-300m in the Hugging Face cache:
# it is gated, so accept its license on huggingface.co and download it once while logged in
# (for example by loading it with sentence-transformers).
set -eu
BUILD=b11669 # llama.cpp v0.7.0
DIR=$(cd "$(dirname "$0")" && pwd)
HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}

if [ "$(uname -sm)" != "Darwin arm64" ]; then
	echo "setup.sh installs the macOS arm64 (Apple Silicon) build only" >&2
	exit 1
fi
SNAPSHOT=$(ls -d "$HUB"/models--google--embeddinggemma-300m/snapshots/* 2>/dev/null | head -n 1)
if [ -z "$SNAPSHOT" ]; then
	echo "google/embeddinggemma-300m is not in the Hugging Face cache at $HUB" >&2
	exit 1
fi

fetch() { # url, directory to extract into; the archive holds one top-level directory
	tmp=$(mktemp -d)
	curl -fsSL -o "$tmp/archive.tar.gz" "$1"
	tar -xzf "$tmp/archive.tar.gz" -C "$tmp"
	rm "$tmp/archive.tar.gz"
	mkdir -p "$(dirname "$2")"
	rm -rf "$2"
	mv "$tmp"/* "$2"
	rm -rf "$tmp"
}

if [ ! -x "$DIR/bin/llama.cpp/llama-server" ]; then
	fetch "https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/llama-$BUILD-bin-macos-arm64.tar.gz" "$DIR/bin/llama.cpp"
fi
if [ ! -f "$DIR/src/llama.cpp/convert_hf_to_gguf.py" ]; then
	fetch "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$BUILD.tar.gz" "$DIR/src/llama.cpp"
fi
if [ ! -x "$DIR/.venv-convert/bin/python" ]; then
	uv venv -q --python 3.12 "$DIR/.venv-convert"
	uv pip install -q --python "$DIR/.venv-convert/bin/python" \
		torch==2.14.1 transformers==5.20.2 numpy==2.5.4 safetensors==0.8.0
fi
mkdir -p "$DIR/models"
"$DIR/.venv-convert/bin/python" -I "$DIR/convert_local.py" "$DIR/src/llama.cpp" "$SNAPSHOT" \
	--outtype f32 --sentence-transformers-dense-modules \
	--outfile "$DIR/models/embeddinggemma-300M-F32.gguf"
echo "installed; check it with: uv run python -m claude_memory.embedding_server.verify"
