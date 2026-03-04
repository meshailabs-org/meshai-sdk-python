"""Tests for the Batcher."""

import time

from meshai.batcher import Batcher


def test_flush_on_batch_size():
    flushed = []

    def on_flush(batch):
        flushed.append(batch[:])

    batcher = Batcher(flush_fn=on_flush, batch_size=3, flush_interval=60.0)

    batcher.add({"a": 1})
    batcher.add({"a": 2})
    assert len(flushed) == 0

    batcher.add({"a": 3})  # triggers flush
    assert len(flushed) == 1
    assert len(flushed[0]) == 3
    batcher.shutdown()


def test_flush_on_interval():
    flushed = []

    def on_flush(batch):
        flushed.append(batch[:])

    batcher = Batcher(flush_fn=on_flush, batch_size=100, flush_interval=0.2)

    batcher.add({"a": 1})
    time.sleep(0.4)

    assert len(flushed) >= 1
    batcher.shutdown()


def test_manual_flush():
    flushed = []

    def on_flush(batch):
        flushed.append(batch[:])

    batcher = Batcher(flush_fn=on_flush, batch_size=100, flush_interval=60.0)

    batcher.add({"a": 1})
    batcher.add({"a": 2})
    batcher.flush()

    assert len(flushed) == 1
    assert len(flushed[0]) == 2
    batcher.shutdown()


def test_shutdown_flushes_remaining():
    flushed = []

    def on_flush(batch):
        flushed.append(batch[:])

    batcher = Batcher(flush_fn=on_flush, batch_size=100, flush_interval=60.0)

    batcher.add({"a": 1})
    batcher.shutdown()

    assert len(flushed) == 1
