"""Background prefetching for the multimodal data pipeline.

Building a Molmo2 example is CPU-heavy (image decode → resize → patchify, plus
tokenization), and with sequence packing each step consumes several examples. Done
synchronously in the training loop that work stalls the GPU (~14% idle in benchmarks).

:func:`prefetch_map` applies a (slow) function over an iterable on a pool of threads or
processes, yielding results **in input order** while keeping a bounded number of items in
flight, so the preprocessing of upcoming examples overlaps the current GPU step. Threads are the
default: the heavy steps (PyTorch image ops, the Rust tokenizer, file reads) release the GIL,
and results need no IPC. A page fault on a memory-mapped file (Arrow tables, numpy memmaps)
does not: the faulting thread keeps the GIL until the page arrives, so on a network file system
one slow read stalls every thread of the pool. Processes (``processes=True``, forked like
:class:`torch.utils.data.DataLoader` workers) confine such a stall to one worker, at the cost of
pickling each result back to the caller.

Order preservation keeps downstream greedy packing deterministic regardless of worker count.
"""

from __future__ import annotations

import functools
import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["prefetch_map"]

# Functions run by forked worker processes, by key: the workers inherit this table when they
# are forked, so the function (often a bound method of a large object) is never pickled.
_PROCESS_FNS: Dict[int, Callable[[Any], Any]] = {}
_PROCESS_FN_KEYS = itertools.count()


def _init_process_worker() -> None:
    import torch

    # OpenMP thread pools do not survive fork, and the pool's processes run side by side.
    torch.set_num_threads(1)


def _call_in_process(key: int, item: Any) -> Any:
    return _PROCESS_FNS[key](item)


def prefetch_map(
    fn: Callable[[T], R],
    iterable: Iterable[T],
    *,
    num_workers: int,
    max_in_flight: Optional[int] = None,
    processes: bool = False,
) -> Iterator[R]:
    """Lazily apply ``fn`` over ``iterable`` on a worker pool, yielding results in order.

    :param fn: the (expensive) per-item function, e.g. ``dataset.__getitem__``.
    :param iterable: input items (may be infinite, e.g. a cycled ref stream).
    :param num_workers: pool size. ``<= 0`` runs synchronously (no workers).
    :param max_in_flight: cap on submitted-but-unconsumed items (bounds memory / read-ahead).
        Defaults to ``max(2 * num_workers, 4)``.
    :param processes: Run ``fn`` in processes forked from this one instead of threads. Items
        and results must pickle; ``fn`` itself is inherited, not pickled, and must not use CUDA.
        The processes are stopped when the iterator is closed or exhausted.
    """
    if num_workers <= 0:
        for item in iterable:
            yield fn(item)
        return

    from collections import deque
    from concurrent.futures import Executor, ThreadPoolExecutor

    if max_in_flight is None:
        max_in_flight = max(2 * num_workers, 4)

    it = iter(iterable)
    executor: Executor
    if processes:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        key = next(_PROCESS_FN_KEYS)
        _PROCESS_FNS[key] = fn
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_process_worker,
        )
        submit = functools.partial(executor.submit, _call_in_process, key)
    else:
        executor = ThreadPoolExecutor(max_workers=num_workers)
        submit = functools.partial(executor.submit, fn)

    futures: deque = deque()
    try:
        for _ in range(max_in_flight):
            try:
                futures.append(submit(next(it)))
            except StopIteration:
                break
        while futures:
            result = futures.popleft().result()
            try:
                futures.append(submit(next(it)))
            except StopIteration:
                pass
            yield result
    finally:
        # Runs on normal exhaustion and on GeneratorExit (loader stops / epoch ends). The worker
        # processes, if any, are listed first: shutting the pool down forgets them.
        workers = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        if processes:
            # The pool stops its workers from a background thread, which a forked caller's exit
            # (a batch prefetch process) does not wait for: stop them here, mid-item if need be.
            for worker in workers:
                worker.terminate()
            for worker in workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
            _PROCESS_FNS.pop(key, None)
