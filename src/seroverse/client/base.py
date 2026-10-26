"""Unified asynchronous HTTP client and msgspec serialization helpers for SeroVerse."""

import asyncio
import hashlib
import os
import time
from collections.abc import AsyncIterator, Coroutine
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import aiohttp
import msgspec

from seroverse.progress import current_progress


class Client:
    """High-performance asynchronous HTTP client powered by `aiohttp` and `msgspec`.

    Args:
        base_url: Base endpoint URL.
        api_key_header: Optional HTTP header key for API authorization.
        api_key: Optional API key string.
        user_agent: User-Agent header value.
        timeout: Total request timeout in seconds.
        max_concurrency: Maximum concurrent in-flight requests per event loop.
        polite_delay: Optional minimum delay (in seconds) after live network requests.
        recorder: Optional `cassetter` recorder instance.
    """

    def __init__(
        self,
        base_url: str,
        api_key_header: str | None = None,
        api_key: str | None = None,
        user_agent: str = "seroverse-client/0.1",
        timeout: float = 30.0,
        *,
        max_concurrency: int = 5,
        polite_delay: float = 0.0,
        recorder: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_header = api_key_header
        self._api_key: str | None = api_key
        self.user_agent = user_agent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrency = max(1, int(max_concurrency))
        self.polite_delay = float(polite_delay)
        self.recorder = recorder
        self._session: aiohttp.ClientSession | None = None
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None

    @property
    def api_key(self) -> str | None:
        """Return the configured API key string, or `None`."""
        return self._api_key

    @staticmethod
    def encode_msgpack(obj: Any) -> bytes:
        """Serialize an object or dataclass tree into compact MessagePack bytes."""
        return msgspec.msgpack.encode(obj)

    @staticmethod
    def decode_msgpack[T](data: bytes, type_: type[T]) -> T:
        """Deserialize MessagePack bytes directly into a typed `@dataclass(frozen=True, slots=True)` structure."""
        return msgspec.msgpack.decode(data, type=type_)

    @staticmethod
    def save_msgpack(path: Path, obj: Any) -> None:
        """Atomically persist `obj` to `path` as binary MessagePack."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(msgspec.msgpack.encode(obj))
        os.replace(tmp, path)

    @staticmethod
    def load_msgpack[T](path: Path, type_: type[T]) -> T:
        """Load and decode a binary MessagePack file into `type_`."""
        return msgspec.msgpack.decode(path.read_bytes(), type=type_)

    def __repr__(self) -> str:
        key_repr = "'**********'" if self._api_key else "None"
        return f"{self.__class__.__name__}(base_url='{self.base_url}', api_key={key_repr})"

    def __cache_fingerprint__(self) -> tuple[str, str, str]:
        """Returns a deterministic, credential-safe fingerprint tuple for content-addressable caches."""
        key_digest = hashlib.sha256((self._api_key or "").encode("utf-8")).hexdigest()[:16] if self._api_key else ""
        return (type(self).__qualname__, self.base_url, key_digest)

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if self.api_key_header and self._api_key:
            headers[self.api_key_header] = self._api_key
        return headers

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.max_concurrency)
            self._sem_loop = loop
        return self._sem

    async def polite_sleep(self, t_req: float, delay: float | None = None) -> None:
        """Sleep politely after a live network request, skipping sleep on fast cassette replays."""
        wait_s = self.polite_delay if delay is None else delay
        if wait_s <= 0.0:
            return
        elapsed = time.perf_counter() - t_req
        if not (self.recorder is not None and elapsed < 0.25) and elapsed > 0.02:
            await asyncio.sleep(wait_s)

    async def get_session(self) -> aiohttp.ClientSession:
        """Retrieve or instantiate the underlying `aiohttp.ClientSession` for the active event loop."""
        current_loop = asyncio.get_running_loop()
        if (
            self._session is None
            or self._session.closed
            or getattr(self._session, "_loop", None) is not current_loop
            or getattr(self._session, "_loop", current_loop).is_closed()
        ):
            self._session = aiohttp.ClientSession(
                headers=self._headers(),
                timeout=self.timeout,
            )
        return self._session

    async def __aenter__(self) -> Self:
        await self.get_session()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close_sync()

    async def close(self) -> None:
        """Close the underlying `aiohttp.ClientSession` if open."""
        if self._session is not None:
            if not self._session.closed:
                loop = getattr(self._session, "_loop", None)
                if loop is None or not loop.is_closed():
                    await self._session.close()
            self._session = None

    def close_sync(self) -> None:
        """Close the underlying `aiohttp.ClientSession` synchronously."""
        if self._session is not None and not self._session.closed:
            loop = getattr(self._session, "_loop", None)
            if loop is not None and not loop.is_closed():
                self.run_sync(self.close())
            else:
                self._session = None

    def _url(self, endpoint: str) -> str:
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        return f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url

    async def request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        """Send an async HTTP request and return decoded JSON (via `msgspec`) or text."""
        session = await self.get_session()
        url = self._url(endpoint)
        async with self._get_semaphore():
            t_req = time.perf_counter()
            try:
                async with session.request(method, url, **kwargs) as resp:
                    raw = await resp.read()
                    if resp.status >= 400:
                        body = raw.decode("utf-8", errors="replace").strip()
                        reason = resp.reason or "HTTP Error"
                        detail = f"{reason}: {body[:300]}" if body else reason
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=detail,
                            headers=resp.headers,
                        )
                    ctype = (resp.content_type or "").lower()
                    if "json" in ctype:
                        return msgspec.json.decode(raw)
                    return raw.decode("utf-8", errors="replace")
            finally:
                await self.polite_sleep(t_req)

    async def get(self, endpoint: str, **kwargs: Any) -> Any:
        """Perform an async HTTP GET request."""
        return await self.request("GET", endpoint, **kwargs)

    async def post(self, endpoint: str, **kwargs: Any) -> Any:
        """Perform an async HTTP POST request."""
        return await self.request("POST", endpoint, **kwargs)

    async def delete(self, endpoint: str, **kwargs: Any) -> Any:
        """Perform an async HTTP DELETE request."""
        return await self.request("DELETE", endpoint, **kwargs)

    async def stream_bytes(self, method: str, endpoint: str, **kwargs: Any) -> bytes:
        """Send an async HTTP request and read raw response bytes."""
        session = await self.get_session()
        url = self._url(endpoint)
        async with self._get_semaphore():
            t_req = time.perf_counter()
            try:
                async with session.request(method, url, **kwargs) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            finally:
                await self.polite_sleep(t_req)

    async def get_json[T](self, endpoint: str, *, type_: type[T], **kwargs: Any) -> T:
        """Fetch `endpoint` via GET and decode JSON directly into `type_` using `msgspec`."""
        raw = await self.stream_bytes("GET", endpoint, **kwargs)
        return msgspec.json.decode(raw, type=type_)

    async def post_json[T](self, endpoint: str, *, type_: type[T], **kwargs: Any) -> T:
        """Fetch `endpoint` via POST and decode JSON directly into `type_` using `msgspec`."""
        raw = await self.stream_bytes("POST", endpoint, **kwargs)
        return msgspec.json.decode(raw, type=type_)

    async def stream_buffer(self, method: str, endpoint: str, **kwargs: Any) -> BytesIO:
        """Send an async HTTP request and return an in-memory `BytesIO` stream."""
        b = await self.stream_bytes(method, endpoint, **kwargs)
        return BytesIO(b)

    async def stream_chunks(
        self,
        method: str,
        endpoint: str,
        chunk_size: int = 65536,
        **kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Stream response bytes in chunks as an async generator, emitting progress when `Content-Length` is known."""
        session = await self.get_session()
        url = self._url(endpoint)
        prog = current_progress()
        async with session.request(method, url, **kwargs) as resp:
            resp.raise_for_status()
            total_bytes = resp.content_length if resp.content_length and resp.content_length > 0 else None
            received = 0
            async for chunk in resp.content.iter_chunked(chunk_size):
                received += len(chunk)
                if total_bytes is not None and prog.is_active:
                    prog.report(
                        min(1.0, received / total_bytes),
                        detail=f"Streaming {endpoint} ({received:,} / {total_bytes:,} B)",
                    )
                yield chunk

    def run_sync[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Execute a coroutine synchronously and close any session bound to the temporary event loop."""

        async def _runner() -> T:
            try:
                return await coro
            finally:
                await self.close()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: asyncio.run(_runner()))
                return future.result()
        return asyncio.run(_runner())
