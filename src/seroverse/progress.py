"""Zero-dependency, context-propagated hierarchical progress reporting."""

import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

type ProgressCallback = Callable[[float, str | None, str | None], None]

_SHARED_VAR_ATTR = "__active_progress_contextvar__"


def _get_shared_contextvar() -> ContextVar[Any]:
    var = getattr(sys, _SHARED_VAR_ATTR, None)
    if not isinstance(var, ContextVar):
        var = ContextVar(_SHARED_VAR_ATTR, default=None)
        setattr(sys, _SHARED_VAR_ATTR, var)
    return var


@dataclass(slots=True)
class _SharedProgressState:
    """Shared root state across a hierarchy of scoped `ProgressReporter` instances."""

    callback: ProgressCallback | None
    min_interval_sec: float = 0.0
    last_emit_time: float = 0.0
    max_fraction: float = 0.0


@dataclass(slots=True)
class ProgressReporter:
    """Hierarchical, rate-throttled progress reporter with sub-range composition.

    Maps local progress fractions in `[0.0, 1.0]` into `[start, end]` of the root task,
    ensuring nested operations advance monotonically from 0% to 100%.

    Attributes:
        start (float): Lower bound of this reporter's progress interval within `[0.0, 1.0]`.
        end (float): Upper bound of this reporter's progress interval within `[0.0, 1.0]`.
        default_message (str | None): Default primary status message when `report()` omits `message`.
        detail_prefix (str | None): Optional hierarchical prefix prepended to `detail` strings.
    """

    _shared: _SharedProgressState = field(default_factory=lambda: _SharedProgressState(callback=None))
    start: float = 0.0
    end: float = 1.0
    default_message: str | None = None
    detail_prefix: str | None = None

    @property
    def is_active(self) -> bool:
        """Whether an active listener callback is bound to this reporter."""
        return self._shared.callback is not None

    active = is_active

    def report(
        self,
        fraction: float,
        message: str | None = None,
        detail: str | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Emits a progress update mapped into `[self.start, self.end]`.

        Args:
            fraction: Local completion fraction in `[0.0, 1.0]`.
            message: Optional primary status heading.
            detail: Optional secondary detail string.
            force: If `True`, bypasses rate-limiting `min_interval_sec` throttling.
        """
        cb = self._shared.callback
        if cb is None:
            return

        if message is not None and self.detail_prefix is None:
            self.default_message = message

        clamped = min(1.0, max(0.0, float(fraction)))
        span = max(0.0, self.end - self.start)
        mapped = min(1.0, max(self._shared.max_fraction, self.start + clamped * span))

        now = time.monotonic()
        if (
            not force
            and self._shared.min_interval_sec > 0.0
            and clamped < 1.0
            and (now - self._shared.last_emit_time) < self._shared.min_interval_sec
        ):
            return

        self._shared.last_emit_time = now
        self._shared.max_fraction = mapped

        if self.detail_prefix is not None:
            eff_message = self.default_message if self.default_message is not None else message
        else:
            eff_message = message if message is not None else self.default_message
        if self.detail_prefix and detail:
            eff_detail: str | None = f"{self.detail_prefix} · {detail}"
        elif self.detail_prefix:
            eff_detail = self.detail_prefix
        else:
            eff_detail = detail

        cb(mapped, eff_message, eff_detail)

    @contextmanager
    def subrange(
        self,
        start: float,
        end: float,
        *,
        message: str | None = None,
        detail_prefix: str | None = None,
    ) -> Generator[ProgressReporter]:
        """Scopes nested `current_progress()` calls into `[start, end]` of the current range.

        Args:
            start: Relative start fraction in `[0.0, 1.0]` within the parent reporter's range.
            end: Relative end fraction in `[start, 1.0]` within the parent reporter's range.
            message: Optional override for `default_message` inside the subrange.
            detail_prefix: Optional prefix prepended to nested `detail` strings.

        Yields:
            Scoped child `ProgressReporter` bound to `_ACTIVE_PROGRESS`.
        """
        if not self.is_active:
            yield self
            return

        s_clamped = min(1.0, max(0.0, float(start)))
        e_clamped = min(1.0, max(s_clamped, float(end)))
        span = max(0.0, self.end - self.start)
        child_start = self.start + s_clamped * span
        child_end = self.start + e_clamped * span

        if self.detail_prefix and detail_prefix:
            combined_prefix: str | None = f"{self.detail_prefix} · {detail_prefix}"
        else:
            combined_prefix = detail_prefix or self.detail_prefix

        child = ProgressReporter(
            _shared=self._shared,
            start=child_start,
            end=child_end,
            default_message=message if message is not None else self.default_message,
            detail_prefix=combined_prefix,
        )
        token = _ACTIVE_PROGRESS.set(child)
        try:
            yield child
        finally:
            _ACTIVE_PROGRESS.reset(token)


_NOOP_REPORTER = ProgressReporter()
_ACTIVE_PROGRESS: ContextVar[ProgressReporter | None] = _get_shared_contextvar()


def current_progress() -> ProgressReporter:
    """Returns the context-local `ProgressReporter` (or a zero-cost no-op reporter if unbound)."""
    active = _ACTIVE_PROGRESS.get()
    return active if active is not None else _NOOP_REPORTER


@contextmanager
def progress_context(
    callback: ProgressCallback,
    *,
    message: str | None = None,
    min_interval_sec: float = 0.0,
) -> Generator[ProgressReporter]:
    """Binds a root `ProgressReporter` to the current execution context (propagates into `asyncio.to_thread`).

    Args:
        callback: Listener callable `(fraction, message, detail) -> None` invoked on progress updates.
        message: Optional initial default status message.
        min_interval_sec: Minimum elapsed seconds between consecutive non-final progress emissions.

    Yields:
        Root `ProgressReporter` bound to the current `contextvars` context.
    """
    shared = _SharedProgressState(callback=callback, min_interval_sec=min_interval_sec)
    reporter = ProgressReporter(_shared=shared, start=0.0, end=1.0, default_message=message)
    token = _ACTIVE_PROGRESS.set(reporter)
    try:
        yield reporter
    finally:
        _ACTIVE_PROGRESS.reset(token)


__all__ = [
    "ProgressCallback",
    "ProgressReporter",
    "current_progress",
    "progress_context",
]
