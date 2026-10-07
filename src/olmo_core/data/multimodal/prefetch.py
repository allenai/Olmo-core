"""Background prefetching for the multimodal data pipeline.

Building a Molmo2 example is CPU-heavy (image decode → resize → patchify, plus
tokenization), and with sequence packing each step consumes several examples. Done
synchronously in the training loop that work stalls the GPU (~14% idle in benchmarks).

:func:`prefetch_map` applies a (slow) function over an iterable on a thread pool, yielding
results **in input order** while keeping a bounded number of items in flight, so the
preprocessing of upcoming examples overlaps the current GPU step. Threads (not processes)
are used deliberately: the heavy steps (PyTorch image ops, the Rust tokenizer, Arrow
mmap reads) release the GIL, and the per-example payload (megabytes of image patches) would
be expensive to ship over process IPC.

Order preservation keeps downstream greedy packing deterministic regardless of worker count.

Two scheduling policies exist. The default submits a new item only when the head result is
consumed, so a slow head (an olmOCR PDF page that takes a second to render) lets the rest of the
window finish and then idles the pool. With ``keep_full=True`` a new item is submitted whenever
*any* in-flight item completes, keeping ``max_in_flight`` **incomplete** items in flight at all
times; completed results are buffered (bounded by ``max_ready``) until their turn. Both policies
yield the same results in the same order.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["PrefetchStats", "prefetch_map"]


@dataclass
class PrefetchStats:
    """Cheap counters :func:`prefetch_map` maintains about how its consumer was served.

    The consumer (the data loader) reads them between rank batches to tell "the head result
    was ready" from "we blocked on the head", which is the loader's own build-time breakdown.
    """

    yielded: int = 0
    """Results handed to the consumer."""
    ready_hits: int = 0
    """Results whose computation had already finished when the consumer asked for them."""
    head_wait_s: float = 0.0
    """Seconds the consumer spent blocked waiting for the head (in-order next) result."""
    submitted: int = 0
    """Items submitted to the pool."""
    max_unconsumed: int = 0
    """Largest number of submitted-but-unconsumed items seen (window depth actually used)."""

    def snapshot(self) -> Dict[str, float]:
        """The counters as a plain dict (for logging deltas between rank batches)."""
        return {
            "yielded": self.yielded,
            "ready_hits": self.ready_hits,
            "head_wait_s": self.head_wait_s,
            "submitted": self.submitted,
            "max_unconsumed": self.max_unconsumed,
        }


def prefetch_map(
    fn: Callable[[T], R],
    iterable: Iterable[T],
    *,
    num_workers: int,
    max_in_flight: Optional[int] = None,
    keep_full: bool = False,
    max_ready: Optional[int] = None,
    stats: Optional[PrefetchStats] = None,
) -> Iterator[R]:
    """Lazily apply ``fn`` over ``iterable`` on a thread pool, yielding results in order.

    :param fn: the (expensive) per-item function, e.g. ``dataset.__getitem__``.
    :param iterable: input items (may be infinite, e.g. a cycled ref stream).
    :param num_workers: thread-pool size. ``<= 0`` runs synchronously (no threads).
    :param max_in_flight: cap on submitted-but-unconsumed items (bounds memory / read-ahead).
        Defaults to ``max(2 * num_workers, 4)``. With ``keep_full`` it is instead the cap on
        **incomplete** items in flight; see ``max_ready``.
    :param keep_full: submit a new item whenever any in-flight item completes (not only when
        the head result is consumed), so a slow head does not drain the pool. Results are still
        yielded strictly in input order. Off by default.
    :param max_ready: with ``keep_full``, the cap on completed-but-unconsumed results buffered
        beyond the ``max_in_flight`` incomplete items, so memory stays bounded when the head is
        very slow. Defaults to ``max_in_flight``. Submitted-but-unconsumed items never exceed
        ``max_in_flight + max_ready``.
    :param stats: optional :class:`PrefetchStats` updated in place (under either policy).
    """
    if num_workers <= 0:
        for item in iterable:
            result = fn(item)
            if stats is not None:
                stats.submitted += 1
                stats.yielded += 1
            yield result
        return

    if max_in_flight is None:
        max_in_flight = max(2 * num_workers, 4)
    if keep_full:
        yield from _prefetch_keep_full(
            fn,
            iterable,
            num_workers=num_workers,
            max_in_flight=max_in_flight,
            max_ready=max_in_flight if max_ready is None else max_ready,
            stats=stats,
        )
        return

    it = iter(iterable)
    executor = ThreadPoolExecutor(max_workers=num_workers)
    futures: Deque[Future] = deque()
    try:
        for _ in range(max_in_flight):
            try:
                futures.append(executor.submit(fn, next(it)))
            except StopIteration:
                break
        if stats is not None:
            stats.submitted += len(futures)
            stats.max_unconsumed = max(stats.max_unconsumed, len(futures))
        while futures:
            head = futures.popleft()
            result = head.result() if stats is None else _timed_result(head, stats)
            try:
                futures.append(executor.submit(fn, next(it)))
                if stats is not None:
                    stats.submitted += 1
            except StopIteration:
                pass
            yield result
    finally:
        # Runs on normal exhaustion and on GeneratorExit (loader stops / epoch ends).
        executor.shutdown(wait=False, cancel_futures=True)


def _timed_result(future: Future, stats: PrefetchStats) -> object:
    """``future.result()`` that charges the time blocked on it to ``stats``."""
    stats.yielded += 1
    if future.done():
        stats.ready_hits += 1
        return future.result()
    t0 = time.perf_counter()
    try:
        return future.result()
    finally:
        stats.head_wait_s += time.perf_counter() - t0


def _prefetch_keep_full(
    fn: Callable[[T], R],
    iterable: Iterable[T],
    *,
    num_workers: int,
    max_in_flight: int,
    max_ready: int,
    stats: Optional[PrefetchStats],
) -> Iterator[R]:
    if max_in_flight <= 0 or max_ready < 0:
        raise ValueError("max_in_flight must be positive and max_ready non-negative")
    it = iter(iterable)
    exhausted = False
    total_cap = max_in_flight + max_ready
    executor = ThreadPoolExecutor(max_workers=num_workers)
    pending: Deque[Future] = deque()  # input order; incomplete and completed items alike
    cond = threading.Condition()
    incomplete = 0  # guarded by ``cond``

    def on_done(_: Future) -> None:
        nonlocal incomplete
        with cond:
            incomplete -= 1
            cond.notify()

    def top_up() -> None:
        """Submit until ``max_in_flight`` items are incomplete or ``total_cap`` are pending."""
        nonlocal exhausted, incomplete
        while not exhausted and len(pending) < total_cap:
            with cond:
                if incomplete >= max_in_flight:
                    return
                # Counted before the submit: the done callback may run before submit returns.
                incomplete += 1
            try:
                item = next(it)
            except StopIteration:
                exhausted = True
                with cond:
                    incomplete -= 1
                return
            future = executor.submit(fn, item)
            pending.append(future)
            if stats is not None:
                stats.submitted += 1
                stats.max_unconsumed = max(stats.max_unconsumed, len(pending))
            future.add_done_callback(on_done)

    def wait_for_head(head: Future) -> None:
        """Block until ``head`` is done, topping the pool up on every other completion."""
        while True:
            with cond:
                while not head.done():
                    if not exhausted and len(pending) < total_cap and incomplete < max_in_flight:
                        break  # a slot opened: refill it, then wait again
                    cond.wait()
                if head.done():
                    return
            top_up()

    try:
        top_up()
        while pending:
            head = pending[0]
            if head.done():
                if stats is not None:
                    stats.ready_hits += 1
            else:
                t0 = time.perf_counter()
                wait_for_head(head)
                if stats is not None:
                    stats.head_wait_s += time.perf_counter() - t0
            pending.popleft()
            if stats is not None:
                stats.yielded += 1
            result = head.result()
            top_up()
            yield result
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
