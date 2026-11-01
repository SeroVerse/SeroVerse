"""Abstract base classes for streaming and Structure-of-Arrays batch I/O."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from compression import bz2, gzip, lzma, zstd
from io import BufferedIOBase, BytesIO, RawIOBase
from pathlib import Path
from re import IGNORECASE, Pattern
from re import compile as re_compile
from re import escape as re_escape
from typing import IO, Any, ClassVar, Self, TypeIs, get_args, get_origin

from seroverse.core.collections import Batch, View


class _BaseIO:
    r"""Shared file-extension validation, transparent compression openers, and generic type introspection."""

    _EXTENSIONS: ClassVar[tuple[str, ...]] = ()
    _FILE_REGEX: ClassVar[Pattern[str] | None] = None
    _scalar_cls: ClassVar[Any] = None
    _batch_cls: ClassVar[Any] = None

    _OPENERS: ClassVar[dict[str, Callable[..., Any]]] = {
        "gz": gzip.open,
        "bz2": bz2.open,
        "xz": lzma.open,
        "zst": zstd.open,
        "zstd": zstd.open,
    }

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls._EXTENSIONS and "_FILE_REGEX" not in cls.__dict__:
            ext_pat = "|".join(re_escape(ext.lstrip(".")) for ext in cls._EXTENSIONS)
            cls._FILE_REGEX = re_compile(
                rf"\.(?P<ext>{ext_pat})(\.(?P<compression>gz|bz2|xz|zst|zstd))?$",
                IGNORECASE,
            )

        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            if origin is not None and isinstance(origin, type) and issubclass(origin, _BaseIO):
                args = get_args(base)
                if len(args) >= 2:
                    scalar_candidate, batch_candidate = args[0], args[1]
                    if isinstance(scalar_candidate, type) and issubclass(scalar_candidate, View):
                        cls._scalar_cls = scalar_candidate
                    if isinstance(batch_candidate, type) and issubclass(batch_candidate, Batch):
                        cls._batch_cls = batch_candidate
                break

    @classmethod
    def _open_path(cls, filepath: str | Path, mode: str) -> IO[bytes]:
        r"""Validate `filepath` against `_FILE_REGEX` and open a plain or compressed binary handle."""
        path = Path(filepath)
        if cls._FILE_REGEX is None or not (m := cls._FILE_REGEX.search(path.name)):
            raise NotImplementedError(f"Unsupported format: {path}")
        comp = m.group("compression")
        opener = cls._OPENERS.get(comp.lower() if comp else "", open)
        return opener(path, mode=mode)


class BaseReader[T: View, B: Batch[Any]](_BaseIO, Iterator[T], ABC):
    r"""Abstract base class for chunked binary stream readers and SoA batch loaders.

    Subclasses declare `_EXTENSIONS` and implement [`_iter_records`][seroverse.core.io.base.BaseReader._iter_records]
    (and optionally override [`load`][seroverse.core.io.base.BaseReader.load] for vectorized batch loading).

    Args:
        handle: Open binary stream (`IO[bytes]`).
        chunk_size: Read buffer size in bytes for streaming iteration (default 4 MiB).
    """

    __slots__ = (
        "_handle",
        "_chunk_size",
        "_stream",
    )

    def __init__(
        self,
        handle: IO[bytes] | BufferedIOBase | RawIOBase,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> None:
        """Initialize the stateful stream reader."""
        self._handle = handle
        self._chunk_size = chunk_size
        self._stream: Iterator[T] = self._iter_records()

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        **kwargs: Any,
    ) -> Self:
        """Construct a reader from an in-memory binary buffer (`bytes`, `bytearray`, or `memoryview`)."""
        return cls(BytesIO(data), **kwargs)

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        **kwargs: Any,
    ) -> Self:
        """Open a plain or compressed (`.gz`, `.bz2`, `.xz`, `.zst`, `.zstd`) file for reading."""
        return cls(cls._open_path(filepath, mode="rb"), **kwargs)

    def __iter__(self) -> Self:
        """Return the stateful iterator object itself."""
        return self

    def __next__(self) -> T:
        """Fetch the next scalar record from the stream."""
        return next(self._stream)

    def __enter__(self) -> Self:
        """Enter the runtime context for the reader."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the runtime context and close the underlying binary stream."""
        self.close()

    def close(self) -> None:
        """Close the underlying binary stream."""
        self._handle.close()

    @abstractmethod
    def _iter_records(self) -> Iterator[T]:
        """Stateful generator yielding scalar view records across chunk boundaries."""

    def load(self) -> B:
        """Load the entire stream into memory as a contiguous Structure-of-Arrays `Batch`."""
        batch_cls = type(self)._batch_cls
        if batch_cls is None:
            raise TypeError(f"{type(self).__name__} must parameterize BaseReader[T, B] or override load().")
        return batch_cls.from_list(list(self))


class BaseWriter[T: View, B: Batch[Any]](_BaseIO, ABC):
    r"""Abstract base class for binary stream writers and one-shot byte/file serializers.

    Subclasses declare `_EXTENSIONS` and implement [`_format_scalar`][seroverse.core.io.base.BaseWriter._format_scalar]
    (plus optional [`_header`][seroverse.core.io.base.BaseWriter._header] and vectorized
    [`_format_batch`][seroverse.core.io.base.BaseWriter._format_batch]).

    Args:
        handle: Open writable binary stream (`IO[bytes]`).
    """

    __slots__ = (
        "_handle",
        "_wrote_header",
    )

    def __init__(
        self,
        handle: IO[bytes] | BufferedIOBase | RawIOBase,
    ) -> None:
        """Initialize the stateful stream writer."""
        self._handle = handle
        self._wrote_header = False

    @classmethod
    def from_bytes(
        cls,
        buf: BytesIO | None = None,
        **kwargs: Any,
    ) -> Self:
        """Construct a writer backed by an in-memory `BytesIO` buffer."""
        return cls(buf if buf is not None else BytesIO(), **kwargs)

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        **kwargs: Any,
    ) -> Self:
        """Open a plain or compressed (`.gz`, `.bz2`, `.xz`, `.zst`, `.zstd`) file for writing."""
        return cls(cls._open_path(filepath, mode="wb"), **kwargs)

    def __enter__(self) -> Self:
        """Enter the runtime context for the writer."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the runtime context and close the underlying binary stream."""
        self.close()

    def flush(self) -> None:
        """Flush buffered bytes to the underlying stream."""
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying binary stream."""
        self._handle.close()

    def getvalue(self) -> bytes:
        """Return the written bytes when backed by an in-memory `BytesIO` buffer."""
        if isinstance(self._handle, BytesIO):
            return self._handle.getvalue()
        raise TypeError(f"{type(self).__name__} is not backed by an in-memory BytesIO buffer.")

    def _header(self) -> bytes:
        """Optional format header written once before the first record or batch."""
        return b""

    @abstractmethod
    def _format_scalar(self, item: T) -> bytes:
        """Format a single scalar view record as `bytes` (including trailing newline)."""

    def _format_batch(self, batch: B) -> bytes:
        """Format a Structure-of-Arrays `Batch` of records as `bytes`."""
        return b"".join(self._format_scalar(item) for item in batch)

    def _is_batch(self, data: object) -> TypeIs[B]:
        return isinstance(data, Batch)

    def _is_scalar(self, data: object) -> TypeIs[T]:
        return isinstance(data, View)

    def write(self, data: T | B | Iterable[T]) -> int:
        """Write a scalar record, a `Batch`, or an iterable of records to the stream.

        Returns:
            int: Total number of bytes written in this call.
        """
        written = 0
        if not self._wrote_header:
            self._wrote_header = True
            hdr = self._header()
            if hdr:
                written += self._handle.write(hdr)

        if self._is_batch(data):
            payload = self._format_batch(data)
        elif self._is_scalar(data):
            payload = self._format_scalar(data)
        else:
            payload = b"".join(self._format_scalar(item) for item in data)

        if payload:
            written += self._handle.write(payload)
        return written

    @classmethod
    def to_bytes(
        cls,
        data: T | B | Iterable[T],
        **kwargs: Any,
    ) -> bytes:
        """Serialize a scalar record, `Batch`, or iterable of records directly to `bytes`."""
        buf = BytesIO()
        writer = cls(buf, **kwargs)
        writer.write(data)
        return buf.getvalue()

    @classmethod
    def to_file(
        cls,
        filepath: str | Path,
        data: T | B | Iterable[T],
        **kwargs: Any,
    ) -> Path:
        """Serialize a scalar record, `Batch`, or iterable of records to a plain or compressed file."""
        path = Path(filepath)
        with cls.from_file(path, **kwargs) as writer:
            writer.write(data)
        return path
