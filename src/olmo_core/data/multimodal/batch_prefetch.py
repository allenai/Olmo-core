"""Background prefetching of whole collated batches for the mixture data loader.

:func:`~olmo_core.data.multimodal.prefetch.prefetch_map` overlaps *example* preparation with
the training step, but packing and collation still run on the trainer's thread, and when the
head-of-line example is slow (an olmOCR page render, say) the trainer waits for it while the
other in-flight slots sit finished. :class:`BatchPrefetcher` moves the whole batch iterator onto
one background thread feeding a bounded queue, so the trainer's fetch returns as soon as a
finished batch is queued.

Resume safety is the point of the design: every batch the producer finishes is paired with a
snapshot of the loader state *right after it*, taken on the producer thread before the next
batch starts. The consumer adopts that snapshot when it takes the batch, so the loader's
:meth:`~olmo_core.data.multimodal.mixture_data_loader.MixtureDataLoader.state_dict` describes
the batches the caller has consumed, never the ones sitting in the queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterable, Iterator, Optional, Tuple

log = logging.getLogger(__name__)

__all__ = ["BatchPrefetchStats", "BatchPrefetcher"]

_END = object()
_RECENT = 512


@dataclass
class BatchPrefetchStats:
    """Timing counters of a :class:`BatchPrefetcher`, kept on the owning loader.

    Producer-side numbers measure building a batch (example loads, packing, collation) on the
    background thread; consumer-side numbers measure how long the caller waited for a queued
    batch, which is what the trainer's data-loading time sees.
    """

    batches_produced: int = 0
    produce_s_total: float = 0.0
    produce_s_max: float = 0.0
    produce_s_recent: Deque[float] = field(default_factory=lambda: deque(maxlen=_RECENT))
    """Build time of the most recent batches, newest last."""
    batches_consumed: int = 0
    wait_s_total: float = 0.0
    wait_s_max: float = 0.0
    wait_s_recent: Deque[float] = field(default_factory=lambda: deque(maxlen=_RECENT))
    """Consumer wait of the most recent fetches, newest last."""

    def record_produce(self, seconds: float) -> None:
        """Record one batch built on the producer thread."""
        self.batches_produced += 1
        self.produce_s_total += seconds
        self.produce_s_max = max(self.produce_s_max, seconds)
        self.produce_s_recent.append(seconds)

    def record_wait(self, seconds: float) -> None:
        """Record one consumer fetch."""
        self.batches_consumed += 1
        self.wait_s_total += seconds
        self.wait_s_max = max(self.wait_s_max, seconds)
        self.wait_s_recent.append(seconds)


class BatchPrefetcher(Iterator[Tuple[Any, Any]]):
    """Run a batch iterable on a daemon thread, queueing ``(batch, state_after_it)`` pairs.

    :param make_batches: Called once on the background thread to obtain the batch iterable.
    :param snapshot_state: Called on the background thread right after each batch is produced
        and before the next one starts; its result travels with the batch.
    :param depth: Maximum finished batches held in the queue (``> 0``).
    :param stats: Counters to update; a fresh :class:`BatchPrefetchStats` when omitted.

    Iterating yields the pairs in production order. An exception on the producer thread is
    raised from :meth:`__next__`. :meth:`close` stops the producer (closing its iterator, which
    runs the iterator's own cleanup on that thread) and joins it; closing is idempotent and the
    thread never outlives the interpreter.
    """

    def __init__(
        self,
        make_batches: Callable[[], Iterable[Any]],
        snapshot_state: Callable[[], Any],
        *,
        depth: int,
        stats: Optional[BatchPrefetchStats] = None,
    ):
        if depth <= 0:
            raise ValueError("BatchPrefetcher depth must be positive")
        self.stats = stats if stats is not None else BatchPrefetchStats()
        self._make_batches = make_batches
        self._snapshot_state = snapshot_state
        self._queue: "queue.Queue[Tuple[Any, Any, Optional[BaseException]]]" = queue.Queue(
            maxsize=depth
        )
        self._stop = threading.Event()
        self._finished = False
        self._thread = threading.Thread(
            target=self._run, name="mixture-batch-prefetch", daemon=True
        )
        self._thread.start()

    def _put(self, item: Tuple[Any, Any, Optional[BaseException]]) -> bool:
        """Queue ``item`` unless the consumer has asked us to stop; returns whether it was put."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        batches: Optional[Iterator[Any]] = None
        try:
            batches = iter(self._make_batches())
            while not self._stop.is_set():
                started = time.perf_counter()
                try:
                    batch = next(batches)
                except StopIteration:
                    break
                state = self._snapshot_state()
                self.stats.record_produce(time.perf_counter() - started)
                if not self._put((batch, state, None)):
                    return
            self._put((_END, None, None))
        except BaseException as error:  # noqa: BLE001 - everything must reach the consumer
            if not self._put((None, None, error)):
                log.warning("Batch prefetch thread failed after its consumer went away: %r", error)
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()

    def __iter__(self) -> "BatchPrefetcher":
        return self

    def __next__(self) -> Tuple[Any, Any]:
        if self._finished:
            raise StopIteration
        started = time.perf_counter()
        batch, state, error = self._queue.get()
        self.stats.record_wait(time.perf_counter() - started)
        if error is not None:
            self._finished = True
            raise error
        if batch is _END:
            self._finished = True
            raise StopIteration
        return batch, state

    def close(self) -> None:
        """Stop the producer, discard queued batches and join the thread."""
        self._stop.set()
        self._finished = True
        self._thread.join()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
