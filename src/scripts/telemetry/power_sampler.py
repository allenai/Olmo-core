"""
Per-node telemetry sidecar: samples GPU power, energy, clocks, throttling, GPM activity, and
host CPU/memory/network/disk counters at sub-second rates while a training job runs.

It runs as separate, low-priority processes pinned to a few CPUs (started from the Beaker
``post_setup`` hook before ``torchrun``), so it never runs inside the training processes::

    python src/scripts/telemetry/power_sampler.py --out-dir /weka/.../<run>/telemetry/<hostname>

Sampling loops, each in its own process (NVML calls block per GPU, so GPU loops are per GPU):

* ``fast-g<N>`` (``--fast-hz``): instantaneous power, 1 s average power, and the energy counter.
  A row is stored only when a value changes, with the previous poll time and the call's end
  time, so each sensor update is bracketed.
* ``mid-g<N>`` (``--mid-hz``): power/thermal/board-limit/reliability throttle-time counters, SM
  clock, and clock-event reasons.
* ``slow-g<N>`` (``--slow-hz``): GPM metrics (SM/tensor/DRAM activity, NVLink/PCIe throughput),
  utilization, memory, temperatures, PCIe/NVLink byte counters, other throttle counters, and the
  time spent in each group of NVML calls.
* ``host`` (``--host-hz``): InfiniBand/RAPL sysfs counters (when visible), ``/proc/stat``,
  ``/proc/meminfo``, ``/proc/net/dev``, ``/proc/diskstats``.
* ``host_slow`` (``--host-slow-hz``): summed ``/proc/<pid>/io`` of the training processes, and
  hwmon temperatures at 1 Hz.

Every row carries ``t_mono_ns`` (``CLOCK_MONOTONIC``, shared with the training processes on
the same host) and ``t_wall_ns``. Rows are flushed to Parquet chunks every ``--flush-s``
seconds by a background thread per writer, so slow storage delays files, not samples. Sampling
stops ``--cooldown-s`` seconds after ``--stop-file`` appears (written by the training job when it
finishes), or on SIGTERM, after which a ``DONE`` file is written.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import platform
import queue
import resource
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import nvml_util as U

# Version of the output layout described in the telemetry data dictionary. 1: single slow GPU
# loop and single host loop; 2: GPU loops split into fast/mid/slow and host into host/host_slow;
# 3: chunk files written off the sampling loop, loop stats add write time and backlog.
SCHEMA_VERSION = 3


def log(msg: str):
    print(f"[power_sampler {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ChunkWriter:
    """
    Accumulates rows as columns and writes them to numbered Parquet files on a background thread,
    so a slow filesystem (e.g. network storage busy with a checkpoint write) never stalls sampling.
    """

    def __init__(self, out_dir: Path, name: str, flush_s: float):
        self.out_dir = out_dir
        self.name = name
        self.flush_s = flush_s
        self.cols: dict[str, list] = {}
        self.n_rows = 0
        self.chunk = 0
        self.last_flush = time.monotonic()
        self.max_write_s = 0.0  # longest chunk write since the last stats row (writer thread)
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._drain, name=f"write-{name}", daemon=True)
        self._thread.start()

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
        self._queue.put(
            (self.out_dir / f"{self.name}-{self.chunk:05d}.parquet", self.cols, self.n_rows)
        )
        self.chunk += 1
        self.cols = {}
        self.n_rows = 0

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def close(self):
        """Queue the remaining rows and wait until every chunk is written."""
        self.maybe_flush(force=True)
        self._queue.put(None)
        self._thread.join()

    def _drain(self):
        while (item := self._queue.get()) is not None:
            t0 = time.monotonic()
            try:
                self._write(*item)
            except Exception as e:  # noqa: BLE001
                log(f"{self.name}: write error: {e!r}")
            self.max_write_s = max(self.max_write_s, time.monotonic() - t0)

    @staticmethod
    def _write(path: Path, cols: dict[str, list], n_rows: int):
        tmp = path.with_suffix(".tmp")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.table(cols), tmp, compression="zstd")
        except ImportError:
            path = path.with_suffix(".jsonl")
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for i in range(n_rows):
                    f.write(json.dumps({k: v[i] for k, v in cols.items()}) + "\n")
        os.replace(tmp, path)


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

    def row(self, name: str, writer: ChunkWriter) -> dict:
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
            "max_write_ms": writer.max_write_s * 1e3,
            "pending_chunks": writer.pending,
        }
        self.n = self.overruns = self.max_tick_ns = self.sum_tick_ns = 0
        writer.max_write_s = 0.0
        return row


def _loop_name(kind: str, gpus: list[int]) -> str:
    return f"{kind}-g{gpus[0]}" if len(gpus) == 1 else kind


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
                stats_writer.add(stats.row(name, writer))
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
        stats_writer.add(stats.row(name, writer))
        writer.close()
        stats_writer.close()


# ----------------------------------------------------------------------------------------------
# fast loop: power + energy, stored on change
# ----------------------------------------------------------------------------------------------


def fast_loop(gpus: list[int], hz: float, out_dir: str, flush_s: float, stop):
    import pynvml as N

    fields = [U.FIELD_POWER_INSTANT, U.FIELD_POWER_AVERAGE, U.FIELD_ENERGY]
    ids = U.field_ids(fields)

    def setup():
        N.nvmlInit()
        handles = {g: N.nvmlDeviceGetHandleByIndex(g) for g in gpus}
        return {"handles": handles, "last": dict.fromkeys(gpus), "prev_t": dict.fromkeys(gpus, 0)}

    def sample(state, writer, t_mono, t_wall):
        for gpu, h in state["handles"].items():
            t = time.monotonic_ns()
            fvs = N.nvmlDeviceGetFieldValues(h, ids)
            # Under training load a call can take tens of ms; the values were read somewhere
            # between t and t_end, so record both to bracket each sensor update.
            t_end = time.monotonic_ns()
            vals = tuple(U.decode(fv) for fv in fvs)
            if vals != state["last"][gpu]:
                writer.add(
                    {
                        "t_mono_ns": t,
                        "t_call_end_mono_ns": t_end,
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

    _run_loop(_loop_name("fast", gpus), hz, Path(out_dir), flush_s, stop, setup, sample)


# ----------------------------------------------------------------------------------------------
# mid loop: throttle-time counters and clocks
# ----------------------------------------------------------------------------------------------

# Throttle counters that matter for power/thermal analysis are read at the mid rate; the rest
# (sync boost, low utilization, app/base clock policies) at the slow rate.
_MID_THROTTLE = {
    "throttle_power_ns",
    "throttle_thermal_ns",
    "throttle_board_limit_ns",
    "throttle_reliability_ns",
}
MID_FIELDS = [f for f in U.THROTTLE_FIELDS if f[0] in _MID_THROTTLE]
SLOW_FIELDS = [f for f in U.THROTTLE_FIELDS if f[0] not in _MID_THROTTLE] + U.SLOW_FIELDS


def _safe(N, fn, *args):
    try:
        return fn(*args)
    except N.NVMLError:
        return None


def mid_loop(gpus: list[int], hz: float, out_dir: str, flush_s: float, stop):
    import pynvml as N

    ids = U.field_ids(MID_FIELDS)

    def setup():
        N.nvmlInit()
        return {"handles": {g: N.nvmlDeviceGetHandleByIndex(g) for g in gpus}}

    def sample(state, writer, t_mono, t_wall):
        for gpu, h in state["handles"].items():
            t = time.monotonic_ns()
            row = {"t_mono_ns": t, "t_wall_ns": t_wall + (t - t_mono), "gpu": gpu}
            for (name, _, _), fv in zip(MID_FIELDS, N.nvmlDeviceGetFieldValues(h, ids)):
                row[name] = U.decode(fv)
            row["sm_clock_mhz"] = _safe(N, N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_SM)
            row["clock_event_reasons"] = _safe(N, N.nvmlDeviceGetCurrentClocksEventReasons, h)
            row["t_call_end_mono_ns"] = time.monotonic_ns()
            writer.add(row)

    _run_loop(_loop_name("mid", gpus), hz, Path(out_dir), flush_s, stop, setup, sample)


# ----------------------------------------------------------------------------------------------
# slow loop: GPM activity, utilization, memory, temperatures, byte counters, other throttling
# ----------------------------------------------------------------------------------------------


def slow_loop(gpus: list[int], hz: float, out_dir: str, flush_s: float, stop):
    import pynvml as N

    ids = U.field_ids(SLOW_FIELDS)

    def setup():
        N.nvmlInit()
        handles = {g: N.nvmlDeviceGetHandleByIndex(g) for g in gpus}
        gpm = {}
        for g, h in handles.items():
            try:
                gpm[g] = U.GpmReader(h)
            except N.NVMLError as e:
                log(f"slow: GPM unavailable on GPU {g}: {e}")
                gpm[g] = None
        return {"handles": handles, "gpm": gpm}

    def sample(state, writer, t_mono, t_wall):
        for gpu, h in state["handles"].items():
            t = time.monotonic_ns()
            row = {"t_mono_ns": t, "t_wall_ns": t_wall + (t - t_mono), "gpu": gpu}
            # Per-group call durations, to see which NVML queries are slow under load.
            t0 = time.monotonic_ns()
            for (name, _, _), fv in zip(SLOW_FIELDS, N.nvmlDeviceGetFieldValues(h, ids)):
                row[name] = U.decode(fv)
            t1 = time.monotonic_ns()
            row["mem_clock_mhz"] = _safe(N, N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_MEM)
            row["gpu_temp_c"] = _safe(N, N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU)
            mem = _safe(N, N.nvmlDeviceGetMemoryInfo, h)
            row["mem_used_bytes"] = None if mem is None else mem.used
            util = _safe(N, N.nvmlDeviceGetUtilizationRates, h)
            row["util_gpu_pct"] = None if util is None else util.gpu
            row["util_mem_pct"] = None if util is None else util.memory
            t2 = time.monotonic_ns()
            reader = state["gpm"][gpu]
            vals = None
            if reader is not None:
                try:
                    vals = reader.sample()
                except N.NVMLError:
                    vals = None
            for i, (name, _) in enumerate(U.GPM_METRICS):
                row[name] = None if vals is None else vals[i]
            t3 = time.monotonic_ns()
            row["fields_call_ms"] = (t1 - t0) / 1e6
            row["device_calls_ms"] = (t2 - t1) / 1e6
            row["gpm_calls_ms"] = (t3 - t2) / 1e6
            writer.add(row)

    _run_loop(_loop_name("slow", gpus), hz, Path(out_dir), flush_s, stop, setup, sample)


# ----------------------------------------------------------------------------------------------
# host loop: /proc and sysfs counters, stored in long format
# ----------------------------------------------------------------------------------------------


def _readable(path: str) -> bool:
    try:
        with open(path) as f:
            f.read()
        return True
    except OSError:
        return False


def _host_sources() -> dict[str, str]:
    """Counters sampled every host tick: InfiniBand port data and RAPL energy, when visible."""
    sources = {}
    for p in sorted(glob.glob("/sys/class/infiniband/*/ports/*/counters/port_*_data")):
        parts = p.split("/")
        sources[f"ib:{parts[4]}:{parts[6]}:{parts[-1]}"] = p
    for p in sorted(glob.glob("/sys/class/powercap/*/energy_uj")):
        if _readable(p):
            sources[f"rapl:{p.split('/')[-2]}"] = p
    return sources


def _temp_sources() -> dict[str, str]:
    """hwmon temperature sensors (CPU packages/cores, NICs, NVMe), in millidegrees C."""
    sources = {}
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(f"{d}/name") as f:
                name = f.read().strip()
        except OSError:
            continue
        for p in sorted(glob.glob(f"{d}/temp*_input")):
            label_path = p.replace("_input", "_label")
            label = p.split("/")[-1].replace("_input", "")
            try:
                with open(label_path) as f:
                    label = f.read().strip().replace(" ", "_")
            except OSError:
                pass
            if _readable(p):
                sources[f"temp:{d.split('/')[-1]}:{name}:{label}"] = p
    return sources


def _workload_pids() -> list[int]:
    """All processes in the container except this sampler (training ranks, data workers, ...)."""
    pids = []
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(f"{d}/cmdline", "rb") as f:
                cmd = f.read()
        except OSError:
            continue
        if cmd and b"power_sampler.py" not in cmd:
            pids.append(int(d.split("/")[-1]))
    return pids


def host_inventory() -> dict:
    """One-time record of which host-side sources are visible, to diagnose gaps."""
    ib_ports = {}
    for d in sorted(glob.glob("/sys/class/infiniband/*/ports/*")):
        try:
            ib_ports[d] = sorted(os.listdir(d))
        except OSError as e:
            ib_ports[d] = f"ERROR: {e}"
    return {
        "host_sources": sorted(_host_sources()),
        "temp_sources": sorted(_temp_sources()),
        "powercap_entries": sorted(glob.glob("/sys/class/powercap/*")),
        "ib_port_entries": ib_ports,
        "ipmi_devices": sorted(glob.glob("/dev/ipmi*")),
        "perfquery": shutil.which("perfquery"),
        "perfquery_output": _run_cmd(["perfquery"]) if shutil.which("perfquery") else None,
        "proc_io_readable": _readable(f"/proc/{os.getppid()}/io"),
        "mounts_masking_sys": _masked_sys_mounts(),
    }


def _masked_sys_mounts() -> list[str]:
    """tmpfs mounts over parts of /sys, which is how container runtimes hide e.g. RAPL."""
    with open("/proc/mounts") as f:
        return [line.strip() for line in f if " /sys/" in line and "tmpfs" in line]


def host_loop(hz: float, out_dir: str, flush_s: float, stop):
    """
    Fast host counters: InfiniBand/RAPL sysfs counters (read first), CPU jiffies, memory,
    network and disk. Each row carries its own read time (``t_read_mono_ns``) for computing rates.
    """
    mem_keys = {"MemTotal", "MemAvailable", "Cached", "Dirty", "Writeback", "Shmem"}

    def setup():
        return {"sysfs": _host_sources()}

    def sample(state, writer, t_mono, t_wall):
        def add(key, value):
            writer.add(
                {
                    "t_mono_ns": t_mono,
                    "t_wall_ns": t_wall,
                    "t_read_mono_ns": time.monotonic_ns(),
                    "key": key,
                    "value": int(value),
                }
            )

        for key, path in state["sysfs"].items():
            try:
                with open(path) as f:
                    add(key, f.read().strip())
            except (OSError, ValueError):
                pass
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

    _run_loop("host", hz, Path(out_dir), flush_s, stop, setup, sample)


def host_slow_loop(hz: float, out_dir: str, flush_s: float, stop):
    """
    Slower host sources, kept out of the fast loop so they can't delay it: summed
    ``/proc/<pid>/io`` of the training processes (every tick) and hwmon temperatures (every
    ``hz`` ticks, i.e. 1 Hz).
    """
    io_keys = ("rchar", "wchar", "read_bytes", "write_bytes")
    temp_every, pid_refresh_every = max(1, int(hz)), max(1, 2 * int(hz))

    def setup():
        return {"temps": _temp_sources(), "pids": [], "tick": 0}

    def sample(state, writer, t_mono, t_wall):
        def add(key, value):
            writer.add(
                {
                    "t_mono_ns": t_mono,
                    "t_wall_ns": t_wall,
                    "t_read_mono_ns": time.monotonic_ns(),
                    "key": key,
                    "value": int(value),
                }
            )

        tick = state["tick"]
        state["tick"] += 1
        if tick % pid_refresh_every == 0:
            state["pids"] = _workload_pids()
        # Cumulative I/O of the training processes, including checkpoint writes to network
        # filesystems that the container's net/disk counters don't see. Processes that exit drop
        # out of the sum; take per-interval deltas with that in mind.
        totals = dict.fromkeys(io_keys, 0)
        for pid in state["pids"]:
            try:
                with open(f"/proc/{pid}/io") as f:
                    for line in f:
                        k, v = line.split(":", 1)
                        if k in totals:
                            totals[k] += int(v)
            except (OSError, ValueError):
                pass
        for k, v in totals.items():
            add(f"procio:{k}", v)
        add("procio:num_pids", len(state["pids"]))
        if tick % temp_every == 0:
            for key, path in state["temps"].items():
                try:
                    with open(path) as f:
                        add(key, f.read().strip())
                except (OSError, ValueError):
                    pass

    _run_loop("host_slow", hz, Path(out_dir), flush_s, stop, setup, sample)


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
        "schema_version": SCHEMA_VERSION,
        "hostname": platform.node(),
        "node": os.environ.get("BEAKER_NODE_HOSTNAME"),
        "replica_rank": os.environ.get("BEAKER_REPLICA_RANK"),
        "host_inventory": host_inventory(),
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
    # NVML calls on B300 cost ~3-5 ms of CPU each and block per GPU, so GPU loops run in one
    # process per GPU. The power/energy sensors refresh every ~100 ms; 50 Hz localizes each
    # refresh to within 20 ms.
    parser.add_argument("--fast-hz", type=float, default=50.0)
    parser.add_argument("--mid-hz", type=float, default=10.0)
    parser.add_argument("--slow-hz", type=float, default=2.0)
    parser.add_argument("--host-hz", type=float, default=20.0)
    parser.add_argument("--host-slow-hz", type=float, default=4.0)
    parser.add_argument("--flush-s", type=float, default=10.0)
    parser.add_argument("--cooldown-s", type=float, default=30.0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--num-cpus", type=int, default=8, help="Pin to the last N allowed CPUs.")
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
    import pynvml as N

    N.nvmlInit()
    num_gpus = N.nvmlDeviceGetCount()
    N.nvmlShutdown()
    procs = [
        ctx.Process(target=host_loop, args=(args.host_hz, str(out_dir), args.flush_s, stop)),
        ctx.Process(
            target=host_slow_loop, args=(args.host_slow_hz, str(out_dir), args.flush_s, stop)
        ),
    ]
    for g in range(num_gpus):
        for target, hz in (
            (fast_loop, args.fast_hz),
            (mid_loop, args.mid_hz),
            (slow_loop, args.slow_hz),
        ):
            procs.append(
                ctx.Process(target=target, args=([g], hz, str(out_dir), args.flush_s, stop))
            )
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
