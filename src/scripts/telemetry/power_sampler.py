"""
Per-node telemetry sidecar: samples GPU power, energy, clocks, throttling, GPM activity, and
host CPU/memory/network/disk counters at sub-second rates while a training job runs.

It runs as a separate, low-priority process pinned to a couple of CPUs (started from the Beaker
``post_setup`` hook before ``torchrun``), so it never runs inside the training processes::

    python src/scripts/telemetry/power_sampler.py --out-dir /weka/.../<run>/telemetry/<hostname>

Three sampling loops run in their own processes:

* ``fast``: instantaneous power, 1 s average power, and the energy counter for every GPU,
  polled at ``--fast-hz``. A row is stored only when a value changes, together with the time
  of the previous poll, so each sensor update is localized to one polling interval.
* ``slow``: throttle-time counters, clocks, clock-event reasons, temperatures, memory,
  utilization, and GPM metrics (SM/tensor/DRAM activity, NVLink/PCIe throughput) at
  ``--slow-hz``.
* ``host``: ``/proc/stat``, ``/proc/meminfo``, ``/proc/net/dev``, ``/proc/diskstats``,
  InfiniBand port counters, and RAPL energy counters (if readable) at ``--host-hz``.

Every row carries ``t_mono_ns`` (``CLOCK_MONOTONIC``, shared with the training processes on
the same host) and ``t_wall_ns``. Rows are flushed to Parquet chunks every ``--flush-s``
seconds. Sampling stops ``--cooldown-s`` seconds after ``--stop-file`` appears (written by the
training job when it finishes), or on SIGTERM, after which a ``DONE`` file is written.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import platform
import resource
import signal
import subprocess
import time
from pathlib import Path

import nvml_util as U


def log(msg: str):
    print(f"[power_sampler {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ChunkWriter:
    """Accumulates rows as columns and writes them to numbered Parquet files."""

    def __init__(self, out_dir: Path, name: str, flush_s: float):
        self.out_dir = out_dir
        self.name = name
        self.flush_s = flush_s
        self.cols: dict[str, list] = {}
        self.n_rows = 0
        self.chunk = 0
        self.last_flush = time.monotonic()

    def add(self, row: dict):
        if not self.cols:
            self.cols = {k: [] for k in row}
        for k, col in self.cols.items():
            col.append(row.get(k))
        self.n_rows += 1

    def maybe_flush(self, force: bool = False):
        if not force and time.monotonic() - self.last_flush < self.flush_s:
            return
        self.last_flush = time.monotonic()
        if self.n_rows == 0:
            return
        path = self.out_dir / f"{self.name}-{self.chunk:05d}.parquet"
        tmp = path.with_suffix(".tmp")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.table(self.cols), tmp, compression="zstd")
        except ImportError:
            path = path.with_suffix(".jsonl")
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for i in range(self.n_rows):
                    f.write(json.dumps({k: v[i] for k, v in self.cols.items()}) + "\n")
        os.replace(tmp, path)
        self.chunk += 1
        self.cols = {}
        self.n_rows = 0


class LoopStats:
    """Tracks how well a sampling loop keeps up with its target rate."""

    def __init__(self, period_ns: int):
        self.period_ns = period_ns
        self.n = 0
        self.overruns = 0
        self.max_tick_ns = 0
        self.sum_tick_ns = 0

    def tick(self, duration_ns: int):
        self.n += 1
        self.sum_tick_ns += duration_ns
        self.max_tick_ns = max(self.max_tick_ns, duration_ns)
        if duration_ns > self.period_ns:
            self.overruns += 1

    def row(self, name: str) -> dict:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        row = {
            "t_mono_ns": time.monotonic_ns(),
            "t_wall_ns": time.time_ns(),
            "loop": name,
            "ticks": self.n,
            "overruns": self.overruns,
            "max_tick_us": self.max_tick_ns / 1e3,
            "mean_tick_us": self.sum_tick_ns / max(self.n, 1) / 1e3,
            "cpu_user_s": ru.ru_utime,
            "cpu_sys_s": ru.ru_stime,
        }
        self.n = self.overruns = self.max_tick_ns = self.sum_tick_ns = 0
        return row


def _run_loop(name: str, hz: float, out_dir: Path, flush_s: float, stop, setup, sample):
    """Generic fixed-rate loop: ``sample(state, writer, t_mono, t_wall)`` is called each tick."""
    period_ns = int(1e9 / hz)
    writer = ChunkWriter(out_dir, name, flush_s)
    stats_writer = ChunkWriter(out_dir, f"stats_{name}", flush_s)
    stats = LoopStats(period_ns)
    state = setup()
    next_t = time.monotonic_ns()
    last_stats = time.monotonic()
    try:
        while not stop.is_set():
            t0 = time.monotonic_ns()
            try:
                sample(state, writer, t0, time.time_ns())
            except Exception as e:  # noqa: BLE001
                log(f"{name}: sample error: {e!r}")
                time.sleep(0.5)
            stats.tick(time.monotonic_ns() - t0)
            if time.monotonic() - last_stats >= flush_s:
                stats_writer.add(stats.row(name))
                last_stats = time.monotonic()
            writer.maybe_flush()
            stats_writer.maybe_flush()
            next_t += period_ns
            delay = next_t - time.monotonic_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
            else:
                next_t = time.monotonic_ns()  # fell behind; don't try to catch up
    finally:
        stats_writer.add(stats.row(name))
        writer.maybe_flush(force=True)
        stats_writer.maybe_flush(force=True)


# ----------------------------------------------------------------------------------------------
# fast loop: power + energy, stored on change
# ----------------------------------------------------------------------------------------------


def fast_loop(hz: float, out_dir: str, flush_s: float, stop):
    import pynvml as N

    fields = [U.FIELD_POWER_INSTANT, U.FIELD_POWER_AVERAGE, U.FIELD_ENERGY]
    ids = U.field_ids(fields)

    def setup():
        N.nvmlInit()
        handles = [N.nvmlDeviceGetHandleByIndex(i) for i in range(N.nvmlDeviceGetCount())]
        return {"handles": handles, "last": [None] * len(handles), "prev_t": [0] * len(handles)}

    def sample(state, writer, t_mono, t_wall):
        for gpu, h in enumerate(state["handles"]):
            t = time.monotonic_ns()
            fvs = N.nvmlDeviceGetFieldValues(h, ids)
            vals = tuple(U.decode(fv) for fv in fvs)
            if vals != state["last"][gpu]:
                writer.add(
                    {
                        "t_mono_ns": t,
                        "t_wall_ns": t_wall + (t - t_mono),
                        "t_prev_poll_mono_ns": state["prev_t"][gpu],
                        "gpu": gpu,
                        "power_instant_mw": vals[0],
                        "power_average_mw": vals[1],
                        "energy_mj": vals[2],
                        "nvml_ts_us": int(fvs[0].timestamp),
                    }
                )
                state["last"][gpu] = vals
            state["prev_t"][gpu] = t

    _run_loop("fast", hz, Path(out_dir), flush_s, stop, setup, sample)


# ----------------------------------------------------------------------------------------------
# slow loop: throttling, clocks, temps, memory, utilization, GPM
# ----------------------------------------------------------------------------------------------


def slow_loop(hz: float, out_dir: str, flush_s: float, stop):
    import pynvml as N

    fields = U.THROTTLE_FIELDS + U.SLOW_FIELDS
    ids = U.field_ids(fields)

    def setup():
        N.nvmlInit()
        handles = [N.nvmlDeviceGetHandleByIndex(i) for i in range(N.nvmlDeviceGetCount())]
        gpm = []
        for h in handles:
            try:
                gpm.append(U.GpmReader(h))
            except N.NVMLError as e:
                log(f"slow: GPM unavailable: {e}")
                gpm.append(None)
        return {"handles": handles, "gpm": gpm}

    def _safe(fn, *args):
        try:
            return fn(*args)
        except N.NVMLError:
            return None

    def sample(state, writer, t_mono, t_wall):
        for gpu, h in enumerate(state["handles"]):
            t = time.monotonic_ns()
            row = {"t_mono_ns": t, "t_wall_ns": t_wall + (t - t_mono), "gpu": gpu}
            for (name, _, _), fv in zip(fields, N.nvmlDeviceGetFieldValues(h, ids)):
                row[name] = U.decode(fv)
            row["sm_clock_mhz"] = _safe(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_SM)
            row["mem_clock_mhz"] = _safe(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_MEM)
            row["clock_event_reasons"] = _safe(N.nvmlDeviceGetCurrentClocksEventReasons, h)
            row["gpu_temp_c"] = _safe(N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU)
            mem = _safe(N.nvmlDeviceGetMemoryInfo, h)
            row["mem_used_bytes"] = None if mem is None else mem.used
            util = _safe(N.nvmlDeviceGetUtilizationRates, h)
            row["util_gpu_pct"] = None if util is None else util.gpu
            row["util_mem_pct"] = None if util is None else util.memory
            reader = state["gpm"][gpu]
            vals = None
            if reader is not None:
                try:
                    vals = reader.sample()
                except N.NVMLError:
                    vals = None
            for i, (name, _) in enumerate(U.GPM_METRICS):
                row[name] = None if vals is None else vals[i]
            writer.add(row)

    _run_loop("slow", hz, Path(out_dir), flush_s, stop, setup, sample)


# ----------------------------------------------------------------------------------------------
# host loop: /proc and sysfs counters, stored in long format
# ----------------------------------------------------------------------------------------------


def _host_sources() -> dict[str, str]:
    sources = {}
    for p in sorted(glob.glob("/sys/class/infiniband/*/ports/*/counters/port_*_data")):
        parts = p.split("/")
        sources[f"ib:{parts[4]}:{parts[6]}:{parts[-1]}"] = p
    for p in sorted(glob.glob("/sys/class/powercap/*/energy_uj")):
        try:
            with open(p) as f:
                f.read()
            sources[f"rapl:{p.split('/')[-2]}"] = p
        except OSError:
            pass
    return sources


def host_loop(hz: float, out_dir: str, flush_s: float, stop):
    mem_keys = {"MemTotal", "MemAvailable", "Cached", "Dirty", "Writeback", "Shmem"}

    def setup():
        return {"sysfs": _host_sources()}

    def sample(state, writer, t_mono, t_wall):
        def add(key, value):
            writer.add({"t_mono_ns": t_mono, "t_wall_ns": t_wall, "key": key, "value": int(value)})

        with open("/proc/stat") as f:
            fields = f.readline().split()[1:9]
        for name, v in zip(
            ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"], fields
        ):
            add(f"cpu:{name}_jiffies", v)
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in mem_keys:
                    add(f"mem:{k}_kb", v.split()[0])
        with open("/proc/net/dev") as f:
            for line in f.readlines()[2:]:
                iface, data = line.split(":", 1)
                iface = iface.strip()
                if iface == "lo":
                    continue
                cols = data.split()
                if cols[0] == "0" and cols[8] == "0":
                    continue
                add(f"net:{iface}:rx_bytes", cols[0])
                add(f"net:{iface}:tx_bytes", cols[8])
        with open("/proc/diskstats") as f:
            for line in f:
                cols = line.split()
                dev = cols[2]
                if dev.startswith(("loop", "ram")) or (cols[5] == "0" and cols[9] == "0"):
                    continue
                add(f"disk:{dev}:sectors_read", cols[5])
                add(f"disk:{dev}:sectors_written", cols[9])
        for key, path in state["sysfs"].items():
            try:
                with open(path) as f:
                    add(key, f.read().strip())
            except (OSError, ValueError):
                pass

    _run_loop("host", hz, Path(out_dir), flush_s, stop, setup, sample)


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e!r}"


def write_metadata(out_dir: Path, args: argparse.Namespace):
    import pynvml as N

    N.nvmlInit()
    gpus = []
    for i in range(N.nvmlDeviceGetCount()):
        h = N.nvmlDeviceGetHandleByIndex(i)
        g = {"index": i}
        for key, fn in [
            ("name", N.nvmlDeviceGetName),
            ("uuid", N.nvmlDeviceGetUUID),
            ("serial", N.nvmlDeviceGetSerial),
            ("power_limit_mw", N.nvmlDeviceGetPowerManagementLimit),
            ("enforced_power_limit_mw", N.nvmlDeviceGetEnforcedPowerLimit),
        ]:
            try:
                g[key] = fn(h)
            except N.NVMLError as e:
                g[key] = f"ERROR: {e}"
        try:
            g["pci_bus_id"] = N.nvmlDeviceGetPciInfo(h).busId
        except N.NVMLError as e:
            g["pci_bus_id"] = f"ERROR: {e}"
        try:
            g["power_smoothing_enabled"] = U.read_fields(h, [U.SLOW_FIELDS[1]])[0]
        except N.NVMLError as e:
            g["power_smoothing_enabled"] = f"ERROR: {e}"
        gpus.append(g)
    meta = {
        "hostname": platform.node(),
        "start_mono_ns": time.monotonic_ns(),
        "start_wall_ns": time.time_ns(),
        "driver": N.nvmlSystemGetDriverVersion(),
        "args": vars(args),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("BEAKER_", "CUDA_"))},
        "gpus": gpus,
    }
    N.nvmlShutdown()
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    (out_dir / "nvidia-smi-q.txt").write_text(_run_cmd(["nvidia-smi", "-q"]))
    (out_dir / "chrony.txt").write_text(_run_cmd(["chronyc", "tracking"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--fast-hz", type=float, default=200.0)
    parser.add_argument("--slow-hz", type=float, default=10.0)
    parser.add_argument("--host-hz", type=float, default=20.0)
    parser.add_argument("--flush-s", type=float, default=10.0)
    parser.add_argument("--cooldown-s", type=float, default=30.0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--num-cpus", type=int, default=2, help="Pin to the last N allowed CPUs.")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_file: Path = args.stop_file or (out_dir / "STOP")

    # Stay out of the training job's way: lowest-numbered CPUs are left to training processes.
    cpus = sorted(os.sched_getaffinity(0))[-args.num_cpus :]
    os.sched_setaffinity(0, cpus)
    os.nice(10)

    write_metadata(out_dir, args)
    log(f"writing to {out_dir}, pinned to CPUs {cpus}, stop file {stop_file}")

    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    procs = [
        ctx.Process(target=fast_loop, args=(args.fast_hz, str(out_dir), args.flush_s, stop)),
        ctx.Process(target=slow_loop, args=(args.slow_hz, str(out_dir), args.flush_s, stop)),
        ctx.Process(target=host_loop, args=(args.host_hz, str(out_dir), args.flush_s, stop)),
    ]
    for p in procs:
        p.start()

    terminated = {"flag": False}

    def _on_term(signum, frame):
        del signum, frame
        terminated["flag"] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    deadline = time.monotonic() + args.max_hours * 3600
    reason = "max_hours"
    while time.monotonic() < deadline:
        if terminated["flag"]:
            reason = "signal"
            break
        if stop_file.exists():
            reason = "stop_file"
            log(f"stop file found, sampling {args.cooldown_s:.0f}s more for cool-down")
            t_end = time.monotonic() + args.cooldown_s
            while time.monotonic() < t_end and not terminated["flag"]:
                time.sleep(0.5)
            break
        if not all(p.is_alive() for p in procs):
            reason = "loop_died"
            log("a sampling loop died; stopping")
            break
        time.sleep(0.5)

    stop.set()
    for p in procs:
        p.join(timeout=30)
    (out_dir / "DONE").write_text(json.dumps({"reason": reason, "t_wall_ns": time.time_ns()}))
    log(f"done ({reason})")


if __name__ == "__main__":
    main()
