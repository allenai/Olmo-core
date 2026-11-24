"""
Small helpers shared by the telemetry scripts in this directory (``probe_sensors.py`` and
``power_sampler.py``). Requires ``nvidia-ml-py``.
"""

from __future__ import annotations

import pynvml as N

# Power fields are read with the GPU power scope (as opposed to module or memory scope).
POWER_SCOPE_GPU = 0

# (name, field id, scope id). Values are decoded by :func:`decode`.
FIELD_POWER_INSTANT = ("power_instant_mw", N.NVML_FI_DEV_POWER_INSTANT, POWER_SCOPE_GPU)
FIELD_POWER_AVERAGE = ("power_average_mw", N.NVML_FI_DEV_POWER_AVERAGE, POWER_SCOPE_GPU)
FIELD_ENERGY = ("energy_mj", N.NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION, 0)

# Accumulated time (ns) during which clocks were held back by each policy. Differencing these
# gives the exact fraction of an interval spent throttled, with no sampling gaps.
THROTTLE_FIELDS = [
    ("throttle_power_ns", N.NVML_FI_DEV_PERF_POLICY_POWER, 0),
    ("throttle_thermal_ns", N.NVML_FI_DEV_PERF_POLICY_THERMAL, 0),
    ("throttle_sync_boost_ns", N.NVML_FI_DEV_PERF_POLICY_SYNC_BOOST, 0),
    ("throttle_board_limit_ns", N.NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT, 0),
    ("throttle_low_util_ns", N.NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION, 0),
    ("throttle_reliability_ns", N.NVML_FI_DEV_PERF_POLICY_RELIABILITY, 0),
    ("throttle_app_clocks_ns", N.NVML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS, 0),
    ("throttle_base_clocks_ns", N.NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS, 0),
]

SLOW_FIELDS = [
    ("mem_temp_c", N.NVML_FI_DEV_MEMORY_TEMP, 0),
    ("power_smoothing_enabled", getattr(N, "NVML_FI_PWR_SMOOTHING_ENABLED", 251), 0),
    ("power_current_limit_mw", N.NVML_FI_DEV_POWER_CURRENT_LIMIT, 0),
    ("pcie_tx_bytes", N.NVML_FI_DEV_PCIE_COUNT_TX_BYTES, 0),
    ("pcie_rx_bytes", N.NVML_FI_DEV_PCIE_COUNT_RX_BYTES, 0),
    ("nvlink_tx_kib", N.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, 0xFFFFFFFF),
    ("nvlink_rx_kib", N.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, 0xFFFFFFFF),
]

GPM_METRICS = [
    ("gpm_sm_util", N.NVML_GPM_METRIC_SM_UTIL),
    ("gpm_sm_occupancy", N.NVML_GPM_METRIC_SM_OCCUPANCY),
    ("gpm_tensor_util", N.NVML_GPM_METRIC_ANY_TENSOR_UTIL),
    ("gpm_dram_bw_util", N.NVML_GPM_METRIC_DRAM_BW_UTIL),
    ("gpm_pcie_tx_mibps", N.NVML_GPM_METRIC_PCIE_TX_PER_SEC),
    ("gpm_pcie_rx_mibps", N.NVML_GPM_METRIC_PCIE_RX_PER_SEC),
    ("gpm_nvlink_tx_mibps", N.NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC),
    ("gpm_nvlink_rx_mibps", N.NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC),
]


def field_ids(fields):
    return [(fid, scope) for _, fid, scope in fields]


def decode(fv) -> float | None:
    """Decode one ``c_nvmlFieldValue_t``; returns ``None`` if the read failed."""
    if fv.nvmlReturn != N.NVML_SUCCESS:
        return None
    t = fv.valueType
    v = fv.value
    if t == N.NVML_VALUE_TYPE_DOUBLE:
        return float(v.dVal)
    if t == N.NVML_VALUE_TYPE_UNSIGNED_INT:
        return float(v.uiVal)
    if t == N.NVML_VALUE_TYPE_UNSIGNED_LONG:
        return float(v.ulVal)
    if t == N.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        return float(v.ullVal)
    if t == N.NVML_VALUE_TYPE_SIGNED_LONG_LONG:
        return float(v.sllVal)
    if t == N.NVML_VALUE_TYPE_SIGNED_INT:
        return float(v.siVal)
    if t == N.NVML_VALUE_TYPE_UNSIGNED_SHORT:
        return float(v.usVal)
    return None


def read_fields(handle, fields) -> list[float | None]:
    return [decode(fv) for fv in N.nvmlDeviceGetFieldValues(handle, field_ids(fields))]


class GpmReader:
    """
    Computes GPM metrics between consecutive calls to :meth:`sample`. Each value is the average
    over the interval since the previous call.
    """

    def __init__(self, handle, metrics=GPM_METRICS):
        self.handle = handle
        self.metrics = metrics
        self.supported = bool(N.nvmlGpmQueryDeviceSupport(handle).isSupportedDevice)
        self._samples = [N.nvmlGpmSampleAlloc(), N.nvmlGpmSampleAlloc()]
        self._have_prev = False
        self._cur = 0

    def sample(self) -> list[float | None] | None:
        if not self.supported:
            return None
        cur = self._samples[self._cur]
        N.nvmlGpmSampleGet(self.handle, cur)
        out = None
        if self._have_prev:
            prev = self._samples[1 - self._cur]
            mg = N.c_nvmlGpmMetricsGet_t()
            mg.version = N.NVML_GPM_METRICS_GET_VERSION
            mg.numMetrics = len(self.metrics)
            mg.sample1 = prev
            mg.sample2 = cur
            for i, (_, mid) in enumerate(self.metrics):
                mg.metrics[i].metricId = mid
            N.nvmlGpmMetricsGet(mg)
            out = [
                float(mg.metrics[i].value) if mg.metrics[i].nvmlReturn == N.NVML_SUCCESS else None
                for i in range(len(self.metrics))
            ]
        self._have_prev = True
        self._cur = 1 - self._cur
        return out

    def close(self):
        for s in self._samples:
            try:
                N.nvmlGpmSampleFree(s)
            except N.NVMLError:
                pass
