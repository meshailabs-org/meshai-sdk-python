"""Buffered batcher for telemetry events."""

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger("meshai")


class Batcher:
    """Accumulates events and flushes in batches.

    Flushes when batch_size is reached or flush_interval elapses.
    Thread-safe. Never crashes the host application.
    """

    def __init__(
        self,
        flush_fn: Callable[[list[dict[str, Any]]], None],
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        self._flush_fn = flush_fn
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._started = False

    def add(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._buffer.append(event)
            if not self._started:
                self._start_timer()
                self._started = True
            if len(self._buffer) >= self._batch_size:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        self._cancel_timer()
        try:
            self._flush_fn(batch)
        except Exception:
            logger.exception("Failed to flush batch of %d events", len(batch))
        if self._started:
            self._start_timer()

    def _start_timer(self) -> None:
        self._timer = threading.Timer(self._flush_interval, self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_timer(self) -> None:
        self.flush()

    def shutdown(self) -> None:
        self._cancel_timer()
        self.flush()
        self._started = False
