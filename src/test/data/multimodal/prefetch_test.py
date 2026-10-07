"""CPU tests for :func:`olmo_core.data.multimodal.prefetch.prefetch_map`'s scheduling policies."""

import random
import threading
import time

import pytest

from olmo_core.data.multimodal.prefetch import PrefetchStats, prefetch_map


def _settle(predicate, timeout=5.0, quiet=0.3):
    """Wait until ``predicate()`` is true or its value has been stable for ``quiet`` seconds."""
    deadline = time.monotonic() + timeout
    last, last_change = predicate(), time.monotonic()
    while time.monotonic() < deadline:
        current = predicate()
        if current is True:
            return
        if current != last:
            last, last_change = current, time.monotonic()
        elif time.monotonic() - last_change > quiet:
            return
        time.sleep(0.01)


@pytest.mark.parametrize("keep_full", [False, True])
@pytest.mark.parametrize("workers,depth,ready", [(4, 8, None), (8, 4, 2), (3, 3, 0)])
def test_order_preserved_under_random_delays(keep_full, workers, depth, ready):
    rng = random.Random(1234 + depth)
    items = list(range(60))
    delays = {i: rng.uniform(0.0, 0.01) for i in items}

    def slow(i):
        time.sleep(delays[i])
        return i * 10

    stats = PrefetchStats()
    out = list(
        prefetch_map(
            slow,
            iter(items),
            num_workers=workers,
            max_in_flight=depth,
            keep_full=keep_full,
            max_ready=ready,
            stats=stats,
        )
    )
    assert out == [i * 10 for i in items]  # the synchronous order
    assert stats.yielded == stats.submitted == len(items)
    assert stats.max_unconsumed <= depth + (depth if ready is None else ready)


def test_keep_full_bounds_incomplete_and_unconsumed_items():
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def work(i):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.003)
        with lock:
            state["active"] -= 1
        return i

    stats = PrefetchStats()
    out = list(
        prefetch_map(
            work,
            range(100),
            num_workers=8,
            max_in_flight=3,
            keep_full=True,
            max_ready=5,
            stats=stats,
        )
    )
    assert out == list(range(100))
    # At most ``max_in_flight`` incomplete items exist, so no more than that many ever execute
    # at once even with a wider pool; the window never exceeds max_in_flight + max_ready.
    assert state["max_active"] <= 3
    assert stats.max_unconsumed <= 8


@pytest.mark.parametrize("depth,ready", [(8, None), (8, 4)])
def test_keep_full_keeps_pool_busy_behind_a_slow_head(depth, ready):
    """One pending head: the default scheme finishes the rest of its window and idles; the
    keep-full scheme keeps completing items up to the buffer cap."""
    total_cap = depth + (depth if ready is None else ready)

    def run(keep_full):
        release = threading.Event()
        lock = threading.Lock()
        completed = []

        def work(i):
            if i == 0:
                release.wait(10)
            with lock:
                completed.append(i)
            return i

        stats = PrefetchStats()
        gen = prefetch_map(
            work,
            range(200),
            num_workers=4,
            max_in_flight=depth,
            keep_full=keep_full,
            max_ready=ready,
            stats=stats,
        )
        results: list = []
        first_t = threading.Thread(target=lambda: results.append(next(gen)))
        first_t.start()
        # Let the pool run with the head pending until completions stop arriving.
        _settle(lambda: len(completed) >= total_cap - 1 or len(completed))
        with lock:
            done_while_head_pending = len(completed)
        release.set()
        first_t.join(10)
        results.extend(gen)
        assert results == list(range(200))
        return done_while_head_pending, stats

    old_done, _ = run(False)
    new_done, stats = run(True)
    assert old_done == depth - 1
    assert new_done == total_cap - 1
    assert new_done > old_done
    assert stats.max_unconsumed == total_cap
    assert stats.head_wait_s > 0
    assert stats.yielded == 200


def test_keep_full_matches_default_with_exceptions_and_finite_input():
    def work(i):
        if i % 7 == 3:
            raise ValueError(i)
        return i

    def safe(i):
        try:
            return ("ok", work(i))
        except ValueError as e:
            return ("err", e.args[0])

    expected = [safe(i) for i in range(30)]
    for keep_full in (False, True):
        got = list(prefetch_map(safe, range(30), num_workers=3, keep_full=keep_full))
        assert got == expected
    # An exception escaping ``fn`` surfaces to the consumer under either policy.
    for keep_full in (False, True):
        with pytest.raises(ValueError):
            list(prefetch_map(work, range(30), num_workers=3, keep_full=keep_full))


def test_keep_full_generator_close_stops_submitting():
    submitted = []

    def work(i):
        submitted.append(i)
        time.sleep(0.002)
        return i

    gen = prefetch_map(work, iter(range(10_000)), num_workers=2, max_in_flight=4, keep_full=True)
    assert next(gen) == 0
    gen.close()  # type: ignore[attr-defined]
    n = len(submitted)
    time.sleep(0.05)
    assert len(submitted) == n  # nothing started after close
    assert n <= 8 + 1


def test_stats_counted_in_synchronous_mode():
    stats = PrefetchStats()
    assert list(prefetch_map(lambda i: i, range(5), num_workers=0, stats=stats)) == list(range(5))
    assert stats.snapshot() == {
        "yielded": 5,
        "ready_hits": 0,
        "head_wait_s": 0.0,
        "submitted": 5,
        "max_unconsumed": 0,
    }
