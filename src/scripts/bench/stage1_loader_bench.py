"""Stage-1 data-loader benchmark for one data-parallel rank, built exactly as ``recipe.phase=stage1``
builds it (run-branch tooling, not for review).

usage: python src/scripts/bench/stage1_loader_bench.py <n_refs> <n_batches> <world_size> -- <Vision-Align overrides...>

1. Node facts: hostname, CPU model, load average, pressure-stall info.
2. Per-source single-thread latency of this rank's first ``n_refs`` example refs in the job's order,
   twice: cold, then the same refs again (warm page cache). Storage-bound sources speed up on the
   warm pass; CPU-bound ones (decode, PDF render, resize) do not.
3. Seconds per rank batch (one optimizer step's sequences) with the configured prefetch workers,
   from a fresh cursor position, for ``n_batches`` batches after one warm-up batch.
"""

import os
import socket
import statistics
import sys
import time
from collections import defaultdict


import olmo_core.data.multimodal.mixture_data_loader as mdl
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.internal.vision_alignment import build_config


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        return f"<{e.__class__.__name__}>"


def node_facts() -> None:
    cpu = next(
        (l.split(":", 1)[1].strip() for l in _read("/proc/cpuinfo").splitlines() if "model name" in l),
        "?",
    )
    print(f"host={socket.gethostname()} cpus={os.cpu_count()} model={cpu}")
    print(f"loadavg={_read('/proc/loadavg')}")
    for kind in ("cpu", "io", "memory"):
        print(f"pressure/{kind}: {_read(f'/proc/pressure/{kind}').replace(chr(10), ' | ')}")
    weka = [l for l in _read("/proc/mounts").splitlines() if "weka" in l][:3]
    print("weka mounts:", *weka, sep="\n  ")
    sys.stdout.flush()


def latency_pass(loader, refs, label: str) -> dict:
    times = defaultdict(list)
    errors = defaultdict(int)
    for ref in refs:
        name = loader.dataset_names[ref[0]]
        t = time.perf_counter()
        try:
            loader._try_load_example(ref)
        except Exception:  # noqa: BLE001
            errors[name] += 1
        times[name].append(time.perf_counter() - t)
    total = sum(sum(v) for v in times.values())
    print(f"\n[{label}] {len(refs)} refs in {total:.1f}s single-thread = {total / len(refs) * 1000:.0f} ms/example")
    print(f"{'source':34s} {'n':>5s} {'share':>6s} {'median':>8s} {'p90':>8s} {'max':>8s} {'err':>4s}")
    for name, v in sorted(times.items(), key=lambda kv: -sum(kv[1])):
        s = sorted(v)
        p90 = s[min(len(s) - 1, int(0.9 * len(s)))]
        print(
            f"{name[:34]:34s} {len(v):5d} {100 * sum(v) / total:5.1f}% {statistics.median(v) * 1000:7.0f}ms "
            f"{p90 * 1000:7.0f}ms {max(v) * 1000:7.0f}ms {errors[name]:4d}"
        )
    sys.stdout.flush()
    return times


def main() -> None:
    n_refs, n_batches, world = (int(a) for a in sys.argv[1:4])
    overrides = sys.argv[sys.argv.index("--") + 1 :]
    mdl.get_world_size = lambda group=None: world
    mdl.get_rank = lambda group=None: 0
    node_facts()
    cli = CliContext("src/scripts/train/Vision-Align.py", SubCmd.dry_run, "stage1-loader-bench", "ai2/holmes", overrides)
    config = build_config(cli)
    t0 = time.time()
    loader = config.data_loader.build(config.dataset.build(), dp_process_group=None)
    loader.reshuffle(epoch=1)
    print(
        f"built in {time.time() - t0:.0f}s; prefetch_workers={config.data_loader.prefetch_workers}; "
        f"rank batch = {loader.rank_batch_size // config.data_loader.sequence_length} sequences",
        flush=True,
    )
    it = loader._rank_refs_from_cursor(0)
    refs = [next(it) for _ in range(n_refs)]
    cold = latency_pass(loader, refs, "cold")
    warm = latency_pass(loader, refs, "warm (same refs)")
    print("\ncold/warm median ratio per source:")
    for name in sorted(cold, key=lambda n: -sum(cold[n])):
        print(f"  {name[:34]:34s} {statistics.median(cold[name]) / max(statistics.median(warm[name]), 1e-6):6.2f}x")
    # Batch timing from a fresh position, so these examples are cold too.
    loader.reshuffle(epoch=2)
    batches = iter(loader)
    next(batches)
    times = []
    for _ in range(n_batches):
        t = time.perf_counter()
        next(batches)
        times.append(time.perf_counter() - t)
    s = sorted(times)
    print(
        f"\nrank batch: median {statistics.median(times):.2f}s p90 {s[int(0.9 * (len(s) - 1))]:.2f}s "
        f"max {s[-1]:.2f}s over {len(times)} batches (prefetch_workers={config.data_loader.prefetch_workers})"
    )
    node_facts()


if __name__ == "__main__":
    main()
