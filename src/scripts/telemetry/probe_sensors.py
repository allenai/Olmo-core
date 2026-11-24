"""
Characterize the GPU power/telemetry sensors on a node, to decide how fast (and how) the
``power_sampler.py`` sidecar should poll them.

Run on a single node with all GPUs visible (no torchrun)::

    python src/scripts/telemetry/probe_sensors.py --out-dir /path/to/probe-output

Phases (each phase's failures are recorded and do not stop the probe):

1. ``inventory``: driver, GPUs, power limits, which NVML fields/GPM metrics are readable,
   and which host sources (``/proc``, InfiniBand sysfs counters, RAPL) can be read.
2. ``idle``: poll power fields as fast as possible with the GPUs idle.
3. ``square_wave``: on every GPU, run GEMM bursts with a known on/off schedule (a step
   response plus square waves of several periods) while polling power, energy, clocks, and
   throttle counters as fast as possible. The schedule is saved alongside the samples so the
   update period, averaging window, and latency of each field can be fitted offline.
4. ``gpm``: GPM metrics sampled at several intervals under load, to find the shortest
   usable interval.
5. ``overhead``: GEMM throughput with no poller vs. pollers at several rates.
6. ``host``: cost of reading the host-side sources.

Everything is written to ``--out-dir`` as ``.npz``/``.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import platform
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

GEMM_N = 8192
# (label, on_ms, off_ms, duration_s); "step" is a single 8 s burst between idle periods.
SQUARE_WAVES = [
    ("sq20", 10, 10, 12.0),
    ("sq50", 25, 25, 12.0),
    ("sq100", 50, 50, 12.0),
    ("sq130", 65, 65, 12.0),
    ("sq200", 100, 100, 12.0),
    ("sq1000", 500, 500, 15.0),
]
STEP_IDLE_S = 5.0
STEP_ON_S = 8.0
OVERHEAD_RATES_HZ = [0, 10, 200, -1]  # -1 = as fast as possible
OVERHEAD_SECONDS = 20.0
GPM_INTERVALS_MS = [10, 25, 50, 100, 250, 1000]


def log(msg: str):
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# Phase 1: inventory
# ---------------------------------------------------------------------------------------------


def _try(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"


def _read_text(path: str, limit: int = 4096):
    try:
        with open(path) as f:
            return f.read(limit)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"


def _run(cmd: list[str], timeout: float = 60.0) -> str:
    try:
        return subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=timeout
        ).stdout
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"


def inventory(out_dir: Path):
    import nvml_util as U
    import pynvml as N

    N.nvmlInit()
    inv: dict = {
        "hostname": platform.node(),
        "time": time.time(),
        "driver": _try(N.nvmlSystemGetDriverVersion),
        "nvml": _try(N.nvmlSystemGetNVMLVersion),
        "cuda_driver": _try(N.nvmlSystemGetCudaDriverVersion_v2),
        "cpu_count": os.cpu_count(),
        "sched_affinity": sorted(os.sched_getaffinity(0)),
        "env": {
            k: v for k, v in os.environ.items() if k.startswith(("BEAKER_", "CUDA_", "NVIDIA_"))
        },
        "gpus": [],
    }
    all_fields = (
        [U.FIELD_POWER_INSTANT, U.FIELD_POWER_AVERAGE, U.FIELD_ENERGY]
        + U.THROTTLE_FIELDS
        + U.SLOW_FIELDS
        + [
            ("power_instant_module_mw", N.NVML_FI_DEV_POWER_INSTANT, 1),
            ("power_instant_memory_mw", N.NVML_FI_DEV_POWER_INSTANT, 2),
            ("power_average_module_mw", N.NVML_FI_DEV_POWER_AVERAGE, 1),
            ("power_average_memory_mw", N.NVML_FI_DEV_POWER_AVERAGE, 2),
        ]
    )
    for i in range(N.nvmlDeviceGetCount()):
        h = N.nvmlDeviceGetHandleByIndex(i)
        g: dict = {
            "index": i,
            "name": _try(N.nvmlDeviceGetName, h),
            "uuid": _try(N.nvmlDeviceGetUUID, h),
            "serial": _try(N.nvmlDeviceGetSerial, h),
            "pci_bus_id": _try(lambda d: N.nvmlDeviceGetPciInfo(d).busId, h),
            "power_limit_mw": _try(N.nvmlDeviceGetPowerManagementLimit, h),
            "enforced_power_limit_mw": _try(N.nvmlDeviceGetEnforcedPowerLimit, h),
            "power_limit_constraints_mw": _try(N.nvmlDeviceGetPowerManagementLimitConstraints, h),
            "power_usage_mw": _try(N.nvmlDeviceGetPowerUsage, h),
            "total_energy_mj": _try(N.nvmlDeviceGetTotalEnergyConsumption, h),
            "max_sm_clock": _try(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_SM),
            "max_mem_clock": _try(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_MEM),
            "clock_event_reasons": _try(N.nvmlDeviceGetCurrentClocksEventReasons, h),
            "perf_state": _try(N.nvmlDeviceGetPerformanceState, h),
            "temp_c": _try(N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU),
            "gpm_supported": _try(
                lambda d: bool(N.nvmlGpmQueryDeviceSupport(d).isSupportedDevice), h
            ),
            "fields": {},
        }
        for name, fid, scope in all_fields:
            try:
                fv = N.nvmlDeviceGetFieldValues(h, [(fid, scope)])[0]
                g["fields"][name] = {
                    "field_id": fid,
                    "scope": scope,
                    "return": int(fv.nvmlReturn),
                    "value": U.decode(fv),
                    "timestamp_us": int(fv.timestamp),
                    "latency_us": int(fv.latencyUsec),
                }
            except Exception as e:  # noqa: BLE001
                g["fields"][name] = f"ERROR: {type(e).__name__}: {e}"
        inv["gpus"].append(g)
    N.nvmlShutdown()

    # Host-side sources.
    ib = {}
    for p in (
        sorted(glob.glob("/sys/class/infiniband/*/ports/*/counters/port_xmit_data"))
        + sorted(glob.glob("/sys/class/infiniband/*/ports/*/hw_counters/*"))[:5]
    ):
        ib[p] = _read_text(p, 64).strip()
    inv["infiniband"] = ib
    inv["rapl"] = {
        p: _read_text(p, 64).strip() for p in sorted(glob.glob("/sys/class/powercap/*/energy_uj"))
    }
    inv["hwmon_power"] = {
        p: _read_text(p, 64).strip()
        for p in sorted(glob.glob("/sys/class/hwmon/hwmon*/power*_input"))
    }
    inv["proc_stat_head"] = _read_text("/proc/stat", 400)
    inv["proc_net_dev"] = _read_text("/proc/net/dev", 4096)
    (out_dir / "inventory.json").write_text(json.dumps(inv, indent=2, default=str))

    for name, cmd in [
        ("nvidia-smi-q.txt", ["nvidia-smi", "-q"]),
        ("nvidia-smi-power.txt", ["nvidia-smi", "-q", "-d", "POWER,CLOCK,PERFORMANCE"]),
        ("nvidia-smi-topo.txt", ["nvidia-smi", "topo", "-m"]),
        ("lscpu.txt", ["lscpu"]),
        ("chrony.txt", ["chronyc", "tracking"]),
        ("timedatectl.txt", ["timedatectl"]),
        ("ibstat.txt", ["ibstat"]),
    ]:
        (out_dir / name).write_text(_run(cmd))
    log(f"inventory: {len(inv['gpus'])} GPUs, {len(ib)} IB counters, {len(inv['rapl'])} RAPL files")


# ---------------------------------------------------------------------------------------------
# Pollers and loaders (run in their own processes)
# ---------------------------------------------------------------------------------------------


def poller(gpu: int, out_path: str, stop, rate_hz: float, with_clocks: bool, ready):
    """Poll power/energy (and optionally clocks/throttle counters) until ``stop`` is set."""
    import nvml_util as U
    import pynvml as N

    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByIndex(gpu)
    fast = [U.FIELD_POWER_INSTANT, U.FIELD_POWER_AVERAGE, U.FIELD_ENERGY]
    if with_clocks:
        fast = fast + [U.THROTTLE_FIELDS[0], U.THROTTLE_FIELDS[3]]
    ids = U.field_ids(fast)
    t_list, nvml_ts, vals, clocks = [], [], [], []
    period_ns = int(1e9 / rate_hz) if rate_hz > 0 else 0
    ready.set()
    next_t = time.monotonic_ns()
    while not stop.is_set():
        t0 = time.monotonic_ns()
        fvs = N.nvmlDeviceGetFieldValues(h, ids)
        t_list.append((t0 + time.monotonic_ns()) // 2)
        nvml_ts.append(int(fvs[0].timestamp))
        decoded = [U.decode(fv) for fv in fvs]
        vals.append([np.nan if v is None else v for v in decoded])
        if with_clocks:
            clocks.append(
                (
                    N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
                    N.nvmlDeviceGetCurrentClocksEventReasons(h),
                )
            )
        if period_ns:
            next_t += period_ns
            delay = next_t - time.monotonic_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    np.savez(
        out_path,
        t_mono_ns=np.asarray(t_list, dtype=np.int64),
        nvml_timestamp_us=np.asarray(nvml_ts, dtype=np.int64),
        values=np.asarray(vals, dtype=np.float64),
        names=np.asarray([f[0] for f in fast]),
        clocks=np.asarray(clocks, dtype=np.int64) if clocks else np.zeros((0, 2), dtype=np.int64),
        cpu_user_s=ru.ru_utime,
        cpu_sys_s=ru.ru_stime,
    )
    N.nvmlShutdown()


def _gemm_setup(gpu: int):
    import torch

    torch.cuda.set_device(gpu)
    a = torch.randn(GEMM_N, GEMM_N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(GEMM_N, GEMM_N, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
    return torch, a, b


def _burst(torch, a, b, until_ns: int) -> int:
    n = 0
    while time.monotonic_ns() < until_ns:
        for _ in range(2):
            a @ b
        torch.cuda.synchronize()
        n += 2
    return n


def _sleep_until(t_ns: int):
    while True:
        d = t_ns - time.monotonic_ns()
        if d <= 0:
            return
        time.sleep(min(d / 1e9, 0.0005) if d < 2_000_000 else d / 1e9 - 0.001)


def square_wave_loader(gpu: int, out_path: str, start_ns: int):
    """Run a step response and then square waves, recording the exact on/off edges."""
    torch, a, b = _gemm_setup(gpu)
    edges = []  # (label, on_start_ns, on_end_ns)
    _sleep_until(start_ns)
    t = start_ns + int(STEP_IDLE_S * 1e9)
    _sleep_until(t)
    end = t + int(STEP_ON_S * 1e9)
    on0 = time.monotonic_ns()
    _burst(torch, a, b, end)
    edges.append(("step", on0, time.monotonic_ns()))
    _sleep_until(time.monotonic_ns() + int(STEP_IDLE_S * 1e9))
    for label, on_ms, off_ms, dur_s in SQUARE_WAVES:
        t_end = time.monotonic_ns() + int(dur_s * 1e9)
        next_on = time.monotonic_ns()
        while next_on < t_end:
            _sleep_until(next_on)
            on0 = time.monotonic_ns()
            _burst(torch, a, b, on0 + on_ms * 1_000_000)
            edges.append((label, on0, time.monotonic_ns()))
            next_on = on0 + (on_ms + off_ms) * 1_000_000
        _sleep_until(time.monotonic_ns() + 2_000_000_000)  # 2 s idle between patterns
    np.savez(
        out_path,
        labels=np.asarray([e[0] for e in edges]),
        on_start_ns=np.asarray([e[1] for e in edges], dtype=np.int64),
        on_end_ns=np.asarray([e[2] for e in edges], dtype=np.int64),
    )


def steady_loader(gpu: int, seconds: float, result_q):
    """Run back-to-back GEMMs for ``seconds`` and report achieved TFLOP/s."""
    torch, a, b = _gemm_setup(gpu)
    start = time.monotonic_ns()
    n = _burst(torch, a, b, start + int(seconds * 1e9))
    dt = (time.monotonic_ns() - start) / 1e9
    result_q.put((gpu, n * 2 * GEMM_N**3 / dt / 1e12))


# ---------------------------------------------------------------------------------------------
# Phases 2-6
# ---------------------------------------------------------------------------------------------


def _start_pollers(ctx, gpus, out_dir: Path, tag: str, rate_hz: float, with_clocks: bool):
    stop = ctx.Event()
    procs, readies = [], []
    for g in gpus:
        ready = ctx.Event()
        p = ctx.Process(
            target=poller,
            args=(g, str(out_dir / f"{tag}_gpu{g}.npz"), stop, rate_hz, with_clocks, ready),
        )
        p.start()
        procs.append(p)
        readies.append(ready)
    for r in readies:
        r.wait(60)
    return stop, procs


def phase_idle(ctx, gpus, out_dir: Path):
    stop, procs = _start_pollers(ctx, gpus, out_dir, "idle", -1, True)
    time.sleep(10.0)
    stop.set()
    for p in procs:
        p.join()


def phase_square_wave(ctx, gpus, out_dir: Path):
    stop, polls = _start_pollers(ctx, gpus, out_dir, "square_poll", -1, True)
    start_ns = time.monotonic_ns() + 20_000_000_000  # leave time for CUDA init in loaders
    loaders = [
        ctx.Process(
            target=square_wave_loader, args=(g, str(out_dir / f"square_sched_gpu{g}.npz"), start_ns)
        )
        for g in gpus
    ]
    for p in loaders:
        p.start()
    for p in loaders:
        p.join()
    time.sleep(3.0)
    stop.set()
    for p in polls:
        p.join()


def gpm_worker(gpu: int, out_path: str):
    import nvml_util as U
    import pynvml as N

    torch, a, b = _gemm_setup(gpu)
    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByIndex(gpu)
    reader = U.GpmReader(h)
    rows = []
    if reader.supported:
        # Keep the GPU busy from a side stream while sampling GPM from this thread.
        stream = torch.cuda.Stream()
        for interval_ms in GPM_INTERVALS_MS:
            reader.sample()
            t_end = time.monotonic_ns() + 3_000_000_000
            while time.monotonic_ns() < t_end:
                with torch.cuda.stream(stream):
                    for _ in range(max(1, interval_ms // 2)):
                        a @ b
                time.sleep(interval_ms / 1000)
                try:
                    vals = reader.sample()
                except Exception as e:  # noqa: BLE001
                    vals = None
                    rows.append([interval_ms, time.monotonic_ns(), str(e)])
                    continue
                rows.append([interval_ms, time.monotonic_ns(), vals])
            torch.cuda.synchronize()
    reader.close()
    N.nvmlShutdown()
    Path(out_path).write_text(
        json.dumps(
            {"supported": reader.supported, "metrics": [m[0] for m in U.GPM_METRICS], "rows": rows}
        )
    )


def phase_gpm(ctx, gpus, out_dir: Path):
    procs = [
        ctx.Process(target=gpm_worker, args=(g, str(out_dir / f"gpm_gpu{g}.json"))) for g in gpus
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def phase_overhead(ctx, gpus, out_dir: Path):
    results = {}
    for rate in OVERHEAD_RATES_HZ:
        tag = f"overhead_{'max' if rate < 0 else rate}hz"
        stop, polls = (None, [])
        if rate != 0:
            stop, polls = _start_pollers(ctx, gpus, out_dir, tag, rate, True)
        q = ctx.Queue()
        loaders = [ctx.Process(target=steady_loader, args=(g, OVERHEAD_SECONDS, q)) for g in gpus]
        for p in loaders:
            p.start()
        tflops = dict(q.get() for _ in loaders)
        for p in loaders:
            p.join()
        if stop is not None:
            stop.set()
            for p in polls:
                p.join()
        results[tag] = tflops
        log(f"{tag}: mean {np.mean(list(tflops.values())):.1f} TFLOP/s/GPU")
    (out_dir / "overhead.json").write_text(json.dumps(results, indent=2))


def phase_host(out_dir: Path):
    sources = {
        "proc_stat": ["/proc/stat"],
        "proc_meminfo": ["/proc/meminfo"],
        "proc_net_dev": ["/proc/net/dev"],
        "proc_diskstats": ["/proc/diskstats"],
        "ib_counters": sorted(glob.glob("/sys/class/infiniband/*/ports/*/counters/port_*_data")),
        "rapl": sorted(glob.glob("/sys/class/powercap/*/energy_uj")),
    }
    out = {}
    for name, paths in sources.items():
        if not paths:
            out[name] = {"n_files": 0}
            continue
        n_ok, t0 = 0, time.perf_counter()
        for _ in range(200):
            for p in paths:
                try:
                    with open(p, "rb") as f:
                        f.read()
                    n_ok += 1
                except OSError:
                    pass
        dt = time.perf_counter() - t0
        out[name] = {"n_files": len(paths), "ok_reads": n_ok, "us_per_sweep": dt / 200 * 1e6}
    (out_dir / "host_read_cost.json").write_text(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--phases", default="inventory,idle,square_wave,gpm,overhead,host", help="Comma-separated."
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pynvml as N

    N.nvmlInit()
    gpus = list(range(N.nvmlDeviceGetCount()))
    N.nvmlShutdown()
    log(f"{len(gpus)} GPUs, writing to {out_dir}")

    ctx = mp.get_context("spawn")
    status = {}
    for phase in args.phases.split(","):
        t0 = time.time()
        log(f"phase {phase} ...")
        try:
            if phase == "inventory":
                inventory(out_dir)
            elif phase == "idle":
                phase_idle(ctx, gpus, out_dir)
            elif phase == "square_wave":
                phase_square_wave(ctx, gpus, out_dir)
            elif phase == "gpm":
                phase_gpm(ctx, gpus, out_dir)
            elif phase == "overhead":
                phase_overhead(ctx, gpus, out_dir)
            elif phase == "host":
                phase_host(out_dir)
            else:
                raise ValueError(f"unknown phase {phase}")
            status[phase] = {"ok": True, "seconds": time.time() - t0}
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            log(f"phase {phase} FAILED:\n{tb}")
            status[phase] = {"ok": False, "seconds": time.time() - t0, "error": tb}
        (out_dir / "status.json").write_text(json.dumps(status, indent=2))
    log(f"done: {json.dumps({k: v['ok'] for k, v in status.items()})}")


if __name__ == "__main__":
    main()
