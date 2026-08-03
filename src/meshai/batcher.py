"""Buffered batcher for telemetry events."""

import logging
import threading
from collections.abc import Callable
from typing import Any

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
        batch: list[dict[str, Any]] | None = None
        with self._lock:
            self._buffer.append(event)
            if not self._started:
                self._start_timer()
                self._started = True
            if len(self._buffer) >= self._batch_size:
                batch = self._drain_locked()
        # Send OUTSIDE the lock so a slow network flush never blocks host threads
        # calling add()/track_usage()/heartbeat().
        if batch is not None:
            self._send(batch)

    def flush(self) -> None:
        with self._lock:
            batch = self._drain_locked()
        if batch is not None:
            self._send(batch)

    def _drain_locked(self) -> list[dict[str, Any]] | None:
        """Snapshot and clear the buffer under the lock. Does no I/O."""
        if not self._buffer:
            return None
        batch = self._buffer[:]
        self._buffer.clear()
        self._cancel_timer()
        return batch

    def _send(self, batch: list[dict[str, Any]]) -> None:
        """Run the network flush function with the lock released."""
        try:
            self._flush_fn(batch)
        except Exception:
            logger.exception("Failed to flush batch of %d events", len(batch))
        with self._lock:
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
        with self._lock:
            batch = self._drain_locked()
            self._started = False
        if batch is not None:
            self._send(batch)
