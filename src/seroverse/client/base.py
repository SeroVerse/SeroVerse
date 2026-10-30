"""Unified HTTP client powered by `pyreqwest` with transparent async proxy support."""

import asyncio
import functools
import hashlib
import inspect
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from io import BytesIO
from types import TracebackType
from typing import Any, Self

from pyreqwest.client import SyncClient, SyncClientBuilder
from pyreqwest.exceptions import StatusError
from pyreqwest.request import SyncRequestBuilder

from seroverse.client.progress import current_progress


class AsyncClientProxy[C: "Client"]:
    """Lightweight asynchronous proxy that dispatches synchronous [`Client`][seroverse.client.base.Client] methods via `asyncio.to_thread`."""

    __slots__ = ("_client",)

    def __init__(self, client: C) -> None:
        self._client = client

    @property
    def sync_client(self) -> C:
        """Return the underlying synchronous [`Client`][seroverse.client.base.Client] instance."""
        return self._client

    def __repr__(self) -> str:
        return f"AsyncClientProxy({self._client!r})"

    def __cache_fingerprint__(self) -> tuple[str, str, str]:
        return self._client.__cache_fingerprint__()

    async def aclose(self) -> None:
        """Close the underlying synchronous client on a worker thread."""
        await asyncio.to_thread(self._client.close)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if not callable(attr) or inspect.iscoroutinefunction(attr):
            return attr

        @functools.wraps(attr)
        async def _async_method(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(attr, *args, **kwargs)

        _async_method.__dict__["__self__"] = self._client
        return _async_method


class Client:
    """Synchronous HTTP client wrapping [`pyreqwest.client.SyncClient`][pyreqwest.client.SyncClient] with built-in `async with` / `.as_async()` support.

    Write domain methods once as standard synchronous methods; callers can use
    ``with SomeClient(...) as client:`` for synchronous execution or
    ``async with SomeClient(...) as client:`` (or ``client.as_async()``) to
    await any method asynchronously via [`AsyncClientProxy`][seroverse.client.base.AsyncClientProxy].
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
        follow_redirects: bool = True,
        recorder: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        clean_base = base_url.rstrip("/")
        self.base_url: str = clean_base
        self.api_key_header = api_key_header
        self._api_key: str | None = api_key
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self.follow_redirects = bool(follow_redirects)
        self.max_concurrency = max(1, int(max_concurrency))
        self.polite_delay = float(polite_delay)
        self.recorder = recorder

        req_headers: dict[str, str] = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if self.api_key_header and self._api_key:
            req_headers[self.api_key_header] = self._api_key
        if headers:
            req_headers.update(headers)
        self.headers: dict[str, str] = req_headers

        self._http: SyncClient = (
            SyncClientBuilder()
            .base_url(f"{clean_base}/")
            .user_agent(self.user_agent)
            .default_headers(req_headers)
            .timeout(timedelta(seconds=self.timeout))
            .follow_redirects(self.follow_redirects)
            .max_connections(self.max_concurrency)
            .pool_max_idle_per_host(self.max_concurrency)
            .error_for_status(True)
            .build()
        )

    @property
    def api_key(self) -> str | None:
        """Return the configured API key string, or `None`."""
        return self._api_key

    def __repr__(self) -> str:
        key_repr = "'**********'" if self._api_key else "None"
        return f"{self.__class__.__name__}(base_url='{self.base_url}', api_key={key_repr})"

    def __cache_fingerprint__(self) -> tuple[str, str, str]:
        """Return a deterministic, credential-safe fingerprint tuple for content-addressable caches."""
        key_digest = hashlib.sha256((self._api_key or "").encode("utf-8")).hexdigest()[:16] if self._api_key else ""
        return (type(self).__qualname__, self.base_url, key_digest)

    def close(self) -> None:
        """Close the underlying `pyreqwest` synchronous client."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def as_async(self) -> AsyncClientProxy[Self]:
        """Return an [`AsyncClientProxy`][seroverse.client.base.AsyncClientProxy] wrapping this client."""
        return AsyncClientProxy(self)

    async def __aenter__(self) -> AsyncClientProxy[Self]:
        return AsyncClientProxy(self)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await asyncio.to_thread(self.close)

    def polite_sleep(self, t_req: float, delay: float | None = None) -> None:
        """Sleep politely after a live network request, skipping sleep on fast cassette replays."""
        wait_s = self.polite_delay if delay is None else delay
        if wait_s <= 0.0:
            return
        elapsed = time.perf_counter() - t_req
        if not (self.recorder is not None and elapsed < 0.25) and elapsed > 0.02:
            time.sleep(wait_s)

    def _prepare_request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
        headers: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
        json: Any = None,
        data: Mapping[str, str] | Sequence[tuple[str, str]] | bytes | str | None = None,
        content: bytes | str | None = None,
    ) -> SyncRequestBuilder:
        """Build a configured `SyncRequestBuilder` for `method` and `endpoint`."""
        target = endpoint if endpoint.startswith(("http://", "https://")) else endpoint.lstrip("/")
        builder = self._http.request(method.upper(), target).error_for_status(True)
        if params:
            builder = builder.query(params)
        if headers:
            builder = builder.headers(headers)
        if json is not None:
            builder = builder.body_json(json)
        elif data is not None:
            if isinstance(data, (bytes, bytearray, memoryview)):
                builder = builder.body_bytes(bytes(data))
            elif isinstance(data, str):
                builder = builder.body_text(data)
            else:
                builder = builder.form(data)
        elif content is not None:
            if isinstance(content, str):
                builder = builder.body_text(content)
            else:
                builder = builder.body_bytes(content)
        return builder

    def request_data(
        self,
        method: str,
        endpoint: str,
        *,
        allow_404: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send an HTTP request, raise on HTTP error status, and return decoded JSON or text (`None` on 404 when `allow_404=True`)."""
        t_req = time.perf_counter()
        try:
            resp = self._prepare_request(method, endpoint, **kwargs).build().send()
            resp.error_for_status()
            ctype = (resp.get_header("content-type") or "").lower()
            if "json" in ctype:
                return resp.json()
            return resp.text()
        except StatusError as exc:
            if allow_404 and exc.details.get("status") == 404:
                return None
            raise
        finally:
            self.polite_sleep(t_req)

    def get_data(self, endpoint: str, *, allow_404: bool = False, **kwargs: Any) -> Any:
        """Perform an HTTP GET request and return decoded JSON or text."""
        return self.request_data("GET", endpoint, allow_404=allow_404, **kwargs)

    def post_data(self, endpoint: str, *, allow_404: bool = False, **kwargs: Any) -> Any:
        """Perform an HTTP POST request and return decoded JSON or text."""
        return self.request_data("POST", endpoint, allow_404=allow_404, **kwargs)

    def delete_data(self, endpoint: str, *, allow_404: bool = False, **kwargs: Any) -> Any:
        """Perform an HTTP DELETE request and return decoded JSON or text."""
        return self.request_data("DELETE", endpoint, allow_404=allow_404, **kwargs)

    def stream_bytes(
        self,
        method: str,
        endpoint: str,
        *,
        allow_404: bool = False,
        **kwargs: Any,
    ) -> bytes:
        """Send an HTTP request, raise on HTTP error status, and return raw response bytes (`b""` on 404 when `allow_404=True`)."""
        t_req = time.perf_counter()
        try:
            resp = self._prepare_request(method, endpoint, **kwargs).build().send()
            resp.error_for_status()
            return bytes(resp.bytes())
        except StatusError as exc:
            if allow_404 and exc.details.get("status") == 404:
                return b""
            raise
        finally:
            self.polite_sleep(t_req)

    def fetch_external_bytes(self, url: str) -> bytes:
        """Fetch raw bytes from an external pre-signed URL without attaching this client's API key headers."""
        if url.startswith(self.base_url):
            return self.stream_bytes("GET", url)
        with (
            SyncClientBuilder()
            .user_agent(self.user_agent)
            .default_headers({"Accept": "*/*"})
            .timeout(timedelta(seconds=self.timeout))
            .follow_redirects(self.follow_redirects)
            .error_for_status(True)
            .build()
        ) as ext_client:
            resp = ext_client.get(url).error_for_status(True).build().send()
            resp.error_for_status()
            return bytes(resp.bytes())

    def stream_buffer(self, method: str, endpoint: str, *, allow_404: bool = False, **kwargs: Any) -> BytesIO:
        """Send an HTTP request and return an in-memory `BytesIO` stream."""
        return BytesIO(self.stream_bytes(method, endpoint, allow_404=allow_404, **kwargs))

    def stream_chunks(
        self,
        method: str,
        endpoint: str,
        chunk_size: int = 65536,
        *,
        allow_404: bool = False,
        **kwargs: Any,
    ) -> Iterator[bytes]:
        """Stream response bytes in chunks, emitting progress when `Content-Length` is known."""
        prog = current_progress()
        req = self._prepare_request(method, endpoint, **kwargs).build_streamed()
        try:
            with req as resp:
                resp.error_for_status()
                raw_len = resp.get_header("content-length")
                total_bytes = int(raw_len) if raw_len and raw_len.isdigit() and int(raw_len) > 0 else None
                received = 0
                reader = resp.body_reader
                while (chunk := reader.read(chunk_size)) is not None:
                    raw_chunk = bytes(chunk)
                    if not raw_chunk:
                        break
                    received += len(raw_chunk)
                    if total_bytes is not None and prog.is_active:
                        prog.report(
                            min(1.0, received / total_bytes),
                            detail=f"Streaming {endpoint} ({received:,} / {total_bytes:,} B)",
                        )
                    yield raw_chunk
        except StatusError as exc:
            if allow_404 and exc.details.get("status") == 404:
                return
            raise

    def map_concurrent[T, R](self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Execute `fn` concurrently across `items` using a thread pool bounded by `max_concurrency`."""
        item_list = list(items)
        if not item_list:
            return []
        if len(item_list) == 1 or self.max_concurrency <= 1:
            return [fn(x) for x in item_list]
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(item_list))) as pool:
            return list(pool.map(fn, item_list))
