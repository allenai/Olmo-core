"""CPU tests for :func:`~olmo_core.data.multimodal.prefetch.prefetch_map` on threads and on forked
worker processes: input order, unpickled functions, error propagation, shutdown and isolation of
a worker that holds the interpreter lock."""

import ctypes
import itertools
import multiprocessing
import os
import threading
import time

import numpy as np
import pytest

from olmo_core.data.multimodal.prefetch import prefetch_map


class _Source:
    """A function owner that cannot be pickled (it holds a lock), as datasets often cannot."""

    def __init__(self):
        self.lock = threading.Lock()

    def load(self, item):
        return item, np.full(3, item, dtype=np.int64), os.getpid()


def _no_worker_processes(timeout=10.0):
    deadline = time.monotonic() + timeout
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.05)
    return not multiprocessing.active_children()


@pytest.mark.parametrize("processes", [False, True])
@pytest.mark.parametrize("workers", [0, 1, 4])
def test_results_arrive_in_input_order(processes, workers):
    source = _Source()
    results = list(prefetch_map(source.load, range(40), num_workers=workers, processes=processes))
    assert [item for item, _, _ in results] == list(range(40))
    for item, array, _ in results:
        np.testing.assert_array_equal(array, np.full(3, item, dtype=np.int64))
    pids = {pid for _, _, pid in results}
    if processes and workers > 0:
        assert os.getpid() not in pids and 1 <= len(pids) <= workers
    else:
        assert pids == {os.getpid()}
    assert _no_worker_processes()


@pytest.mark.parametrize("processes", [False, True])
def test_closing_early_stops_the_workers(processes):
    stream = prefetch_map(
        _Source().load, itertools.count(), num_workers=3, max_in_flight=6, processes=processes
    )
    assert [next(stream)[0] for _ in range(5)] == list(range(5))
    stream.close()
    assert _no_worker_processes()


def _pids_of_worker_processes_in_a_child(out):
    stream = prefetch_map(_Source().load, itertools.count(), num_workers=3, processes=True)
    out.put(sorted({next(stream)[2] for _ in range(12)}))
    stream.close()


def _gone(pid, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_workers_of_a_forked_caller_do_not_outlive_it():
    """A caller in a forked child process (as the batch prefetch process is) stops its workers
    when the stream closes, though the child's exit skips the pool's own shutdown hook."""
    context = multiprocessing.get_context("fork")
    out = context.Queue()
    child = context.Process(target=_pids_of_worker_processes_in_a_child, args=(out,))
    child.start()
    pids = out.get(timeout=60)
    child.join(timeout=60)
    assert child.exitcode == 0 and pids
    assert all(_gone(pid) for pid in pids)


def _fail_on_two(item):
    if item == 2:
        raise ValueError("synthetic failure on 2")
    return item


@pytest.mark.parametrize("processes", [False, True])
def test_exceptions_reach_the_caller_in_order(processes):
    stream = prefetch_map(_fail_on_two, range(10), num_workers=2, processes=processes)
    assert next(stream) == 0 and next(stream) == 1
    with pytest.raises(ValueError, match="synthetic failure on 2"):
        next(stream)
    stream.close()
    assert _no_worker_processes()


def _hold_the_interpreter_lock_on_last(item):
    if item == 16:
        # A call through PyDLL keeps the GIL, as a page fault on a memory-mapped file does.
        ctypes.PyDLL(None).usleep(3_000_000)
    else:
        time.sleep(0.05)
    return item


def test_a_worker_process_holding_the_interpreter_lock_stalls_only_itself():
    """The item that holds the GIL for 3 s starts while the first 16 are still loading; with
    worker processes those 16 still arrive in well under 3 s."""
    stream = prefetch_map(
        _hold_the_interpreter_lock_on_last, range(17), num_workers=8, processes=True
    )
    started = time.monotonic()
    assert [next(stream) for _ in range(16)] == list(range(16))
    assert time.monotonic() - started < 1.5
    assert next(stream) == 16
    stream.close()
    assert _no_worker_processes()
