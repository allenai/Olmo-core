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

A producer thread still shares the trainer's interpreter lock. :class:`BatchProcessPrefetcher`
runs the same producer in a forked child process instead, for rank batches whose Python work
(hundreds of image examples) would otherwise slow the training loop.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterable, Iterator, Optional, Tuple

log = logging.getLogger(__name__)

__all__ = ["BatchPrefetchStats", "BatchPrefetcher", "BatchProcessPrefetcher"]

_END = object()
_RECENT = 512
# Batch tensors from this size up are copied to pinned memory when received from a child process.
_PIN_MIN_BYTES = 16 * 2**20


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


class BatchProcessPrefetcher(Iterator[Tuple[Any, Any]]):
    """
    :class:`BatchPrefetcher` with the producer in a child process forked from the caller's.

    Batches cross a ``torch.multiprocessing`` queue, so their tensors travel through shared
    memory, each paired with the loader state right after it, as with :class:`BatchPrefetcher`.
    A receiver thread takes them off that queue ahead of the consumer and copies their large
    tensors to pinned memory (when CUDA is available), so the host-to-device copies run at full
    speed and the consumer's fetch returns a ready batch.

    :param make_batches: Called once in the child to obtain the batch iterable.
    :param snapshot_state: Called in the child right after each batch is produced and before the
        next one starts; its result travels with the batch.
    :param depth: Maximum finished batches queued between the child and the receiver (``> 0``).
    :param stats: Counters to update; a fresh :class:`BatchPrefetchStats` when omitted. A
        batch's build time is recorded when the consumer takes the batch.

    The child is forked, as :class:`torch.utils.data.DataLoader` workers are, and must not use
    CUDA; it runs torch's CPU ops on one thread (OpenMP thread pools do not survive ``fork``),
    and the queued batches need room in ``/dev/shm``. An exception in the
    child, or its death, is raised from :meth:`__next__` as a :class:`RuntimeError`.
    :meth:`close` stops the child and the receiver and joins both; closing is idempotent. The
    child stays alive until then, because the tensors it queued are handed over through its file
    descriptors.
    """

    def __init__(
        self,
        make_batches: Callable[[], Iterable[Any]],
        snapshot_state: Callable[[], Any],
        *,
        depth: int,
        stats: Optional[BatchPrefetchStats] = None,
    ):
        import torch.multiprocessing as mp

        if depth <= 0:
            raise ValueError("BatchProcessPrefetcher depth must be positive")
        self.stats = stats if stats is not None else BatchPrefetchStats()
        context = mp.get_context("fork")
        self._child_queue = context.Queue(maxsize=depth)
        self._stop = context.Event()
        self._finished = False
        self._process = context.Process(
            target=_produce_in_child,
            args=(make_batches, snapshot_state, self._child_queue, self._stop),
            name="mixture-batch-prefetch",
            daemon=True,
        )
        self._process.start()
        self._queue: "queue.Queue[Tuple[str, Any, Any, float]]" = queue.Queue(maxsize=1)
        self._receiver = threading.Thread(
            target=self._receive, name="mixture-batch-prefetch-receiver", daemon=True
        )
        self._receiver.start()

    def _put(self, item: Tuple[str, Any, Any, float]) -> bool:
        """Queue ``item`` for the consumer unless it has asked us to stop."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _receive(self) -> None:
        while not self._stop.is_set():
            try:
                kind, batch, state, produce_s = self._child_queue.get(timeout=1.0)
                if kind == "batch":
                    batch = _pin_large_tensors(batch)
            except queue.Empty:
                if self._process.is_alive():
                    continue
                kind, batch, state, produce_s = "error", self._exit_message(), None, 0.0
            except BaseException:  # noqa: BLE001 - everything must reach the consumer
                kind, batch, state, produce_s = "error", traceback.format_exc(), None, 0.0
            if not self._put((kind, batch, state, produce_s)) or kind != "batch":
                return

    def _exit_message(self) -> str:
        return f"the batch prefetch process exited with code {self._process.exitcode}"

    def __iter__(self) -> "BatchProcessPrefetcher":
        return self

    def __next__(self) -> Tuple[Any, Any]:
        if self._finished:
            raise StopIteration
        started = time.perf_counter()
        kind, batch, state, produce_s = self._queue.get()
        self.stats.record_wait(time.perf_counter() - started)
        if kind == "error":
            self._finished = True
            raise RuntimeError(f"Batch prefetch process failed:\n{batch}")
        if kind == "end":
            self._finished = True
            raise StopIteration
        self.stats.record_produce(produce_s)
        return batch, state

    def close(self) -> None:
        """Stop the child and the receiver, discard queued batches and join both."""
        self._stop.set()
        self._finished = True
        self._receiver.join()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._process.join(timeout=30)
        if self._process.is_alive():
            log.warning("The batch prefetch process did not exit; terminating it")
            self._process.terminate()
            self._process.join()


def _produce_in_child(make_batches, snapshot_state, out, stop) -> None:
    """The body of a :class:`BatchProcessPrefetcher` child: produce and queue until stopped."""
    import multiprocessing

    import torch

    torch.set_num_threads(1)
    _disable_numpy_hugepages()
    # The loader may fork example workers here (``prefetch_processes``), which multiprocessing
    # refuses in a daemonic process; this process shuts them down itself when it is closed.
    multiprocessing.current_process()._config["daemon"] = False  # type: ignore[attr-defined]

    def put(item) -> bool:
        while not stop.is_set():
            try:
                out.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    batches: Optional[Iterator[Any]] = None
    try:
        iterator = batches = iter(make_batches())
        while not stop.is_set():
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            state = snapshot_state()
            if not put(("batch", batch, state, time.perf_counter() - started)):
                break
        put(("end", None, None, 0.0))
    except BaseException:  # noqa: BLE001 - everything must reach the consumer
        put(("error", traceback.format_exc(), None, 0.0))
    finally:
        try:
            close = getattr(batches, "close", None)
            if close is not None:
                close()
        finally:
            # The consumer receives queued tensors through this process's file descriptors.
            stop.wait()
            # Batches still in the queue are discarded; exit without flushing them.
            out.cancel_join_thread()


def _pin_large_tensors(batch: Any) -> Any:
    """A dict batch with its tensors of :data:`_PIN_MIN_BYTES` and more copied to pinned memory."""
    import torch

    if not isinstance(batch, dict) or not torch.cuda.is_available():
        return batch
    return {
        key: (
            value.pin_memory()
            if isinstance(value, torch.Tensor)
            and value.numel() * value.element_size() >= _PIN_MIN_BYTES
            else value
        )
        for key, value in batch.items()
    }


def _disable_numpy_hugepages() -> None:
    """
    Stop NumPy from asking the kernel for transparent huge pages, as it does for arrays of 4 MiB
    and more.

    With THP ``defrag`` set to ``madvise``, a page fault in such an array may compact memory
    synchronously. Once a node's page cache had fragmented its free memory, that stalled batch
    building for minutes in the large fresh arrays of packing and collation.
    """
    try:
        from numpy._core.multiarray import (  # type: ignore[attr-defined]
            _set_madvise_hugepage,
        )
    except ImportError:  # NumPy 1
        try:
            from numpy.core.multiarray import (  # type: ignore[attr-defined,no-redef]
                _set_madvise_hugepage,
            )
        except ImportError:
            return
    _set_madvise_hugepage(False)
