import logging
import os
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch

from olmo_core.config import DType
from olmo_core.distributed.utils import get_world_size

from ..common import ReduceType
from ..train_module import (
    OLMoDDPTrainModule,
    TransformerPipelineTrainModule,
    TransformerTrainModule,
)
from .callback import Callback

log = logging.getLogger(__name__)


def get_device_peak_flops_per_second(
    device_name: str, *, using_half_precision: bool
) -> Optional[int]:
    """
    Return the device's peak dense half-precision FLOP/s for known GPUs, or ``None`` when peak
    throughput is unknown (e.g. full precision, where MFU is not reported).
    """
    if not using_half_precision:
        return None

    dense_correction = 0.5  # dense specs are one-half of the listed sparse specs
    if "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        if "NVL" in device_name:
            return int(1671e12 * dense_correction)
        elif "PCIe" in device_name:
            return int(1513e12 * dense_correction)
        else:  # for SXM and other variants
            return int(1979e12 * dense_correction)
    elif "B200" in device_name or "B300" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/hgx/
        # HGX B200/B300 lists 36 PFLOPS FP16/BF16 for 8 GPUs.
        return int(4.5e15 * dense_correction)
    elif "RTX PRO 6000" in device_name:
        # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf
        return int(1008e12 * dense_correction)
    else:  # for other GPU types, assume A100
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return int(624e12 * dense_correction)


@dataclass
class SpeedMonitorCallback(Callback):
    """
    Monitors throughput.

    .. important::
        This callback gets added automatically if you don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    priority: ClassVar[int] = -2

    num_flops_per_token: Optional[int] = None
    num_params: Optional[int] = None
    device_peak_flops_per_second: Optional[int] = None

    _total_steps: int = 0
    _total_tokens: int = 0
    _total_flops: int = 0
    _start_time: float = 0.0
    _first_step: bool = True
    _step_last_logged: float = 0.0
    _batch_load_start: float = 0.0
    _batch_load_time: float = 0.0
    _step_tokens: int = 0
    _step_seq_len: int = 0
    _step_flops: int = 0
    _step_examples: int = 0
    _total_examples: int = 0
    _parallel_degree: int = 1
    _batch_load_warn_threshold: Optional[float] = None
    _bps_avg: Optional[float] = None
    _tps_avg: Optional[float] = None
    _mfu_avg: Optional[float] = None

    def reset(self):
        self._first_step = True
        self._bps_avg = None

    @property
    def bps_avg(self) -> Optional[float]:
        return self._bps_avg

    @property
    def tps_avg(self) -> Optional[float]:
        return self._tps_avg

    @property
    def mfu_avg(self) -> Optional[float]:
        return self._mfu_avg

    def _get_num_flops_per_token(self, seq_len: int) -> Optional[int]:
        if self.num_flops_per_token is not None:
            return self.num_flops_per_token
        elif isinstance(
            self.trainer.train_module,
            (TransformerTrainModule, TransformerPipelineTrainModule, OLMoDDPTrainModule),
        ):
            return self.trainer.train_module.num_flops_per_token(seq_len)
        else:
            return None

    def pre_train(self):
        self._first_step = True

        if self.trainer.dp_process_group is not None:
            self._parallel_degree = get_world_size() // get_world_size(
                self.trainer.dp_process_group
            )

        if self.num_params is None and isinstance(
            self.trainer.train_module, TransformerTrainModule
        ):
            self.num_params = self.trainer.train_module.model.num_non_embedding_params

        if (
            self.device_peak_flops_per_second is None
            and self.trainer.device.type == "cuda"
            and isinstance(
                self.trainer.train_module,
                (TransformerTrainModule, TransformerPipelineTrainModule, OLMoDDPTrainModule),
            )
        ):
            device_name = torch.cuda.get_device_name(self.trainer.device)

            tm = self.trainer.train_module
            # The device FLOP table is for FP16/BF16 Tensor Core peaks, so treat either as half
            # precision. OLMoDDP (MoE) always runs experts/activations in bfloat16.
            half_precision_dtypes = (torch.bfloat16, torch.float16)
            using_half_precision = (
                isinstance(tm, OLMoDDPTrainModule)
                or tm.autocast_precision in half_precision_dtypes
                or (
                    tm.dp_config is not None
                    and tm.dp_config.param_dtype in (DType.bfloat16, DType.float16)
                )
            )
            self.device_peak_flops_per_second = get_device_peak_flops_per_second(
                device_name, using_half_precision=using_half_precision
            )
            log.info(
                f"Device: {device_name}, Device peak Flops/s: {self.device_peak_flops_per_second}"
            )

    def pre_load_batch(self):
        self._batch_load_start = time.perf_counter()

    def pre_step(self, batch: Dict[str, Any]):
        self._batch_load_time = time.perf_counter() - self._batch_load_start
        if self._batch_load_warn_threshold is None:
            raw_threshold = os.environ.get("OLMO_BATCH_LOAD_WARN_SECONDS")
            if raw_threshold:
                try:
                    self._batch_load_warn_threshold = float(raw_threshold)
                except ValueError:
                    log.warning(
                        "Ignoring invalid OLMO_BATCH_LOAD_WARN_SECONDS=%r; expected a number",
                        raw_threshold,
                    )
                    self._batch_load_warn_threshold = 0.0
            else:
                self._batch_load_warn_threshold = 0.0

        if (
            self._batch_load_warn_threshold > 0
            and self._batch_load_time >= self._batch_load_warn_threshold
        ):
            log.warning(
                "Slow batch load: step=%s rank=%s local_rank=%s load_time=%.3fs threshold=%.3fs",
                self.step,
                os.environ.get("RANK", "?"),
                os.environ.get("LOCAL_RANK", "?"),
                self._batch_load_time,
                self._batch_load_warn_threshold,
            )

        if self._first_step:
            # We don't record the first batch since the first one tends to take
            # unusually long.
            return

        self._total_steps += 1
        if "input_ids" in batch:
            tokens_in_batch = batch["input_ids"].numel()
            self._step_tokens = tokens_in_batch // self._parallel_degree
            self._step_seq_len = batch["input_ids"].shape[1]
            self._total_tokens += self._step_tokens

            # Examples (= images, for image-bearing data) per step — a packing-aware
            # throughput measure (tokens conflate real/pad/image). With sequence packing a
            # row holds several examples (per-token ``example_ids`` 0..K-1, pad = -1), so
            # count ``max(example_ids)+1`` per row; otherwise one example per row.
            if "example_ids" in batch:
                per_row = batch["example_ids"].amax(dim=1) + 1  # all-pad row -> 0
                examples_in_batch = int(per_row.clamp(min=0).sum())
            else:
                examples_in_batch = batch["input_ids"].shape[0]
            self._step_examples = examples_in_batch // self._parallel_degree
            self._total_examples += self._step_examples

            self._step_flops = 0
            if (
                num_flops_per_token := self._get_num_flops_per_token(self._step_seq_len)
            ) is not None:
                self._step_flops = num_flops_per_token * self._step_tokens
                # Train modules with a non-token-based compute component (e.g. a vision
                # encoder running per-image) can report extra FLOPs for this batch that
                # don't scale with token count. Opt-in via `extra_flops_per_batch`.
                extra_fn = getattr(self.trainer.train_module, "extra_flops_per_batch", None)
                if extra_fn is not None and (extra := extra_fn(batch)):
                    self._step_flops += extra // self._parallel_degree
                self._total_flops += self._step_flops

    def post_step(self):
        counter = time.perf_counter()
        self.trainer.record_metric(
            "throughput/device/data loading (s)", self._batch_load_time, reduce_type=ReduceType.max
        )

        if self._first_step:
            # Now we can start recording.
            self._total_steps = 0
            self._total_tokens = 0
            self._total_flops = 0
            self._total_examples = 0
            self._start_time = counter
            self._first_step = False
            self._step_last_logged = counter
            return

        step_time = counter - self._step_last_logged
        total_time = counter - self._start_time
        self._step_last_logged = counter

        if self._step_examples and self._total_examples:
            self.trainer.record_metric(
                "throughput/device/examples per second", self._step_examples / step_time
            )
            self.trainer.record_metric(
                "throughput/device/examples per second (actual avg)",
                self._total_examples / total_time,
            )

        if self._step_tokens and self._total_tokens:
            tps = self._step_tokens / step_time
            tps_avg = self._total_tokens / total_time
            self._tps_avg = tps_avg
            self.trainer.record_metric("throughput/device/TPS", tps)
            self.trainer.record_metric("throughput/device/TPS (actual avg)", tps_avg)

        if self.trainer.global_train_tokens_seen is not None:
            self.trainer.record_metric(
                "throughput/total tokens", self.trainer.global_train_tokens_seen
            )
            if self.num_params is not None:
                self.trainer.record_metric(
                    "throughput/chinchilla multiple",
                    self.trainer.global_train_tokens_seen / (20 * self.num_params),
                )

        flops_ps: Optional[float] = None
        flops_ps_avg: Optional[float] = None
        if self._step_flops and self._total_flops:
            flops_ps = self._step_flops / step_time
            flops_ps_avg = self._total_flops / total_time
            self.trainer.record_metric("throughput/device/flopsPS", flops_ps)
            self.trainer.record_metric("throughput/device/flopsPS (actual avg)", flops_ps_avg)
            self.trainer.record_metric("throughput/device/TFLOPs_per_GPU", flops_ps / 1e12)
            self.trainer.record_metric(
                "throughput/total petaflops", self.trainer.global_train_petaflops
            )

        bps = 1 / step_time
        bps_avg = self._total_steps / total_time
        self._bps_avg = bps_avg
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (actual avg)", bps_avg)

        data_pct = 100 * self._batch_load_time / step_time
        self.trainer.record_metric(
            "throughput/device/data loading (%)", data_pct, reduce_type=ReduceType.max
        )

        if (
            self.device_peak_flops_per_second is not None
            and flops_ps is not None
            and flops_ps_avg is not None
        ):
            # model FLOPS utilization
            # For its definition and calculation, please refer to the PaLM paper:
            # https://arxiv.org/abs/2204.02311
            # MFU is computed from FLOPs/sec. This stays correct even if sequence length changes.
            mfu = 100 * flops_ps / self.device_peak_flops_per_second
            mfu_avg = 100 * flops_ps_avg / self.device_peak_flops_per_second
            self._mfu_avg = mfu_avg
            self.trainer.record_metric("throughput/device/MFU", mfu)
            self.trainer.record_metric("throughput/device/MFU (actual avg)", mfu_avg)
