"""Mock OpenAI API: /v1/embeddings (1536-dim) + /v1/chat/completions.

DELAY_MS env adds fixed latency to embeddings (to mimic a real remote
embedder); default 0.
"""

import base64
import json
import os
import struct
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8791
DIM = int(__import__("os").environ.get("MOCK_DIM", "1536"))
DELAY = float(os.environ.get("DELAY_MS", "0")) / 1000.0

vec = struct.pack(f"<{DIM}f", *([0.03125] * DIM))
B64 = base64.b64encode(vec).decode()
FLOATS = [0.03125] * DIM


def emb_body(n, fmt):
    e = B64 if fmt == "base64" else FLOATS
    return json.dumps({
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": e} for i in range(n)],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 8 * n, "total_tokens": 8 * n},
    }).encode()


CHAT = json.dumps({
    "id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "{}"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}).encode()


import threading as _th
INFLIGHT = 0
PEAK = 0
TOTAL = 0
_LK = _th.Lock()

def _gauge():
    import time as _t
    global PEAK
    while True:
        _t.sleep(1)
        with _LK:
            print(f"inflight={INFLIGHT} peak={PEAK} total={TOTAL}", flush=True)
            PEAK = INFLIGHT

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        data = b'{"keys":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        global INFLIGHT, PEAK, TOTAL
        with _LK:
            INFLIGHT += 1
            TOTAL += 1
            PEAK = max(PEAK, INFLIGHT)
        try:
            self._do()
        finally:
            with _LK:
                INFLIGHT -= 1

    def _do(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n)) if n else {}
        if self.path.endswith("/embeddings"):
            if DELAY:
                time.sleep(DELAY)
            inp = req.get("input", "")
            count = len(inp) if isinstance(inp, list) else 1
            data = emb_body(count, req.get("encoding_format", "float"))
        else:
            data = CHAT
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


print(f"mock openai on :{PORT} delay={DELAY * 1000:.0f}ms", flush=True)
class _Srv(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True

_th.Thread(target=_gauge, daemon=True).start()
_Srv(("127.0.0.1", PORT), H).serve_forever()
