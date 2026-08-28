"""A minimal keep-alive HTTP/1.1 POST client for hot-path JSON calls.

Deliberately tiny: one verb, one content type, no redirects, no cookies, no
TLS. Every error raises; callers are expected to fall back to a full HTTP
client (httpx) or an SDK, which owns retries and canonical error behavior.
RawHTTPPool exists because the general clients charge more per call in
framework machinery than the wire exchange itself costs.
"""

import asyncio
from urllib.parse import urlsplit


def _raise_closed() -> None:
    raise ConnectionError("connection closed by peer")


def _raise_unframed() -> None:
    raise ConnectionError("response without framing")


def _raise_status(status: int) -> None:
    raise ConnectionError(f"HTTP {status}")


class RawHTTPPool:
    """Keep-alive connection pool for plain-HTTP POSTs to one origin."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        max_connections: int = 16,
        timeout: float = 60.0,
    ) -> None:
        """Split the origin out of base_url; reject non-http schemes."""
        parts = urlsplit(base_url)
        if parts.scheme != "http" or parts.hostname is None:
            message = f"RawHTTPPool supports plain http origins only: {base_url}"
            raise ValueError(message)
        self._host = parts.hostname
        self._port = parts.port or 80
        self._path_prefix = parts.path.rstrip("/")
        self._timeout = timeout
        self._free: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
        self._semaphore = asyncio.Semaphore(max_connections)
        extra = "".join(
            f"{name}: {value}\r\n" for name, value in (headers or {}).items()
        )
        self._head_template = (
            "POST {path} HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            "Content-Type: application/json\r\n"
            f"{extra}"
            "Content-Length: {length}\r\n"
            "Connection: keep-alive\r\n\r\n"
        )

    async def post(self, path: str, body: bytes) -> bytes:
        """POST body to path; return the response body of a 200.

        Raises on any transport problem or non-200 status. A connection that
        errors is closed rather than returned to the pool.
        """
        async with self._semaphore:
            return await asyncio.wait_for(
                self._post_once(path, body), timeout=self._timeout
            )

    async def _post_once(self, path: str, body: bytes) -> bytes:
        if self._free:
            reader, writer = self._free.pop()
        else:
            reader, writer = await asyncio.open_connection(self._host, self._port)
        try:
            head = self._head_template.format(
                path=f"{self._path_prefix}{path}", length=len(body)
            )
            writer.write(head.encode() + body)
            await writer.drain()
            status, content_length, chunked, keep_alive = await _read_head(reader)
            response_body = await _read_body(reader, content_length, chunked)
            if status != 200:
                _raise_status(status)
        except BaseException:
            writer.close()
            raise
        if keep_alive:
            self._free.append((reader, writer))
        else:
            writer.close()
        return response_body


async def _read_head(
    reader: asyncio.StreamReader,
) -> tuple[int, int | None, bool, bool]:
    """Read the status line and headers; return framing facts."""
    status_line = await reader.readline()
    if not status_line:
        _raise_closed()
    status = int(status_line.split(b" ", 2)[1])
    content_length: int | None = None
    chunked = False
    keep_alive = True
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b""):
            break
        lowered = line.lower()
        if lowered.startswith(b"content-length:"):
            content_length = int(line.split(b":", 1)[1])
        elif lowered.startswith(b"transfer-encoding:") and b"chunked" in lowered:
            chunked = True
        elif lowered.startswith(b"connection:") and b"close" in lowered:
            keep_alive = False
    return status, content_length, chunked, keep_alive


async def _read_body(
    reader: asyncio.StreamReader,
    content_length: int | None,
    chunked: bool,
) -> bytes:
    """Read one framed response body."""
    if chunked:
        chunks: list[bytes] = []
        while True:
            size = int((await reader.readline()).strip(), 16)
            if size == 0:
                await reader.readline()
                break
            chunks.append(await reader.readexactly(size))
            await reader.readline()
        return b"".join(chunks)
    if content_length is None:
        _raise_unframed()
    return await reader.readexactly(content_length)
