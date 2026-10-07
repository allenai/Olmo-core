"""Background prefetching for the multimodal data pipeline.

Building a Molmo2 example is CPU-heavy (image decode → resize → patchify, plus
tokenization), and with sequence packing each step consumes several examples. Done
synchronously in the training loop that work stalls the GPU (~14% idle in benchmarks).

:func:`prefetch_map` applies a (slow) function over an iterable on a worker pool, yielding
results **in input order** while keeping a bounded number of items in flight, so the
preprocessing of upcoming examples overlaps the current GPU step. Two backends:

* ``"thread"``: a thread pool inside the training process. The heavy steps (PyTorch image
  ops, the Rust tokenizer, Arrow mmap reads) release the GIL, and results need no copying.
  The threads still share the training thread's GIL and every lock, so the Python glue of
  example construction competes with the training loop.
* ``"process"``: forked worker processes. The workers are forked lazily on the first item,
  after the datasets are built, so they inherit the dataset objects; per-process resources
  (HDF5 handles, PDF renderers) are reopened by the readers on first use in each worker.
  Results (dicts of numpy arrays, several MB each) come back by pickling over a pipe.

Order preservation keeps downstream greedy packing deterministic regardless of worker count.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, Literal, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

PrefetchBackend = Literal["thread", "process"]
PREFETCH_BACKENDS = ("thread", "process")

__all__ = ["prefetch_map", "PrefetchBackend", "PREFETCH_BACKENDS"]

# Functions handed to forked workers. The pool is forked *after* its function is registered
# here, so the children inherit the entry and tasks ship only a small token plus the item,
# never the function (whose closure, e.g. a bound method of the data loader, would otherwise
# be pickled, datasets and all, with every task).
_FORKED_FUNCTIONS: Dict[int, Callable[[Any], Any]] = {}
_FUNCTION_TOKENS = itertools.count()


def _run_forked(token: int, item: Any) -> Any:
    return _FORKED_FUNCTIONS[token](item)


def _init_forked_worker() -> None:
    # Each worker builds one example at a time; PyTorch's CPU ops in the image pipeline must
    # not fan out over an intra-op pool per worker (``torch.utils.data`` does the same).
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - torch is always installed here
        pass


def prefetch_map(
    fn: Callable[[T], R],
    iterable: Iterable[T],
    *,
    num_workers: int,
    max_in_flight: Optional[int] = None,
    backend: str = "thread",
) -> Iterator[R]:
    """Lazily apply ``fn`` over ``iterable`` on a worker pool, yielding results in order.

    :param fn: the (expensive) per-item function, e.g. ``dataset.__getitem__``.
    :param iterable: input items (may be infinite, e.g. a cycled ref stream).
    :param num_workers: pool size. ``<= 0`` runs synchronously (no workers).
    :param max_in_flight: cap on submitted-but-unconsumed items (bounds memory / read-ahead).
        Defaults to ``max(2 * num_workers, 4)``.
    :param backend: ``"thread"`` (a thread pool in this process) or ``"process"`` (forked
        worker processes; items and results must be picklable, ``fn`` need not be).
    :raises ValueError: for an unknown ``backend``.
    """
    if backend not in PREFETCH_BACKENDS:
        raise ValueError(f"backend must be one of {PREFETCH_BACKENDS}, got {backend!r}")
    if num_workers <= 0:
        for item in iterable:
            yield fn(item)
        return

    from collections import deque

    if max_in_flight is None:
        max_in_flight = max(2 * num_workers, 4)

    token: Optional[int] = None
    if backend == "process":
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        token = next(_FUNCTION_TOKENS)
        _FORKED_FUNCTIONS[token] = fn
        # Workers are forked on the first submit (below), i.e. on the first iteration.
        executor: Any = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_forked_worker,
        )

        def submit(item: T):
            return executor.submit(_run_forked, token, item)

    else:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=num_workers)

        def submit(item: T):
            return executor.submit(fn, item)

    it = iter(iterable)
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
        # Runs on normal exhaustion and on GeneratorExit (loader stops / epoch ends).
        executor.shutdown(wait=False, cancel_futures=True)
        if token is not None:
            _FORKED_FUNCTIONS.pop(token, None)
