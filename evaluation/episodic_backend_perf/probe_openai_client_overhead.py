"""Client-side CPU cost of the OpenAI embeddings client (no network, no API).

Points the real `openai` AsyncOpenAI client at a local mock that returns a
canned 1536-dim embedding (base64 float32 or float-list JSON per the
request's encoding_format), so what is measured is purely the client's own
work: request build, HTTP via httpx, response parse, embedding decode.

Run:  .venv/bin/python probe_openai_client_overhead.py server &   # subprocess
      .venv/bin/python probe_openai_client_overhead.py client
"""

import asyncio
import base64
import json
import struct
import sys
import time

PORT = 8791
DIM = 1536

if sys.argv[1:] == ["server"]:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    vec = struct.pack(f"<{DIM}f", *([0.03125] * DIM))
    b64 = base64.b64encode(vec).decode()
    floats = [0.03125] * DIM

    def body(fmt):
        emb = b64 if fmt == "base64" else floats
        return json.dumps({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": emb}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
        }).encode()

    bodies = {"base64": body("base64"), "float": body("float")}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            fmt = req.get("encoding_format", "float")
            data = bodies["base64" if fmt == "base64" else "float"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
    sys.exit(0)

from openai import AsyncOpenAI


async def bench(label, **kwargs):
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="sk-x")

    async def one():
        r = await client.embeddings.create(
            model="text-embedding-3-small",
            input="what did alice say about the quarterly latency report?",
            **kwargs,
        )
        assert len(r.data[0].embedding) == DIM

    await one()
    n = 400
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n):
        await one()
    seq_wall = time.perf_counter() - w0
    seq_cpu = time.process_time() - c0

    m = 1000
    w0 = time.perf_counter()
    for i in range(0, m, 16):
        await asyncio.gather(*(one() for _ in range(16)))
    conc_wall = time.perf_counter() - w0
    await client.close()
    print(
        f"{label}: client CPU {seq_cpu / n * 1000:.2f} core-ms/call  "
        f"sequential {n / seq_wall:.0f} calls/s  "
        f"c16 {m / conc_wall:.0f} calls/s"
    )


async def main():
    await bench("default (client picks base64, auto-decodes)")
    # note: passing encoding_format="base64" EXPLICITLY makes the client
    # return the raw base64 string undecoded; only the default path decodes.
    await bench("encoding_format=float (JSON float list)", encoding_format="float")


if sys.argv[1:] == ["client"]:
    asyncio.run(main())
