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
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["prefetch_map"]


def prefetch_map(
    fn: Callable[[T], R],
    iterable: Iterable[T],
    *,
    num_workers: int,
    max_in_flight: Optional[int] = None,
) -> Iterator[R]:
    """Lazily apply ``fn`` over ``iterable`` on a thread pool, yielding results in order.

    :param fn: the (expensive) per-item function, e.g. ``dataset.__getitem__``.
    :param iterable: input items (may be infinite, e.g. a cycled ref stream).
    :param num_workers: thread-pool size. ``<= 0`` runs synchronously (no threads).
    :param max_in_flight: cap on submitted-but-unconsumed items (bounds memory / read-ahead).
        Defaults to ``max(2 * num_workers, 4)``.
    """
    if num_workers <= 0:
        for item in iterable:
            yield fn(item)
        return

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    if max_in_flight is None:
        max_in_flight = max(2 * num_workers, 4)

    it = iter(iterable)
    executor = ThreadPoolExecutor(max_workers=num_workers)
    futures: deque = deque()
    try:
        for _ in range(max_in_flight):
            try:
                futures.append(executor.submit(fn, next(it)))
            except StopIteration:
                break
        while futures:
            result = futures.popleft().result()
            try:
                futures.append(executor.submit(fn, next(it)))
            except StopIteration:
                pass
            yield result
    finally:
        # Runs on normal exhaustion and on GeneratorExit (loader stops / epoch ends).
        executor.shutdown(wait=False, cancel_futures=True)
