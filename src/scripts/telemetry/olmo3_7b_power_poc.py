"""
OLMo 3 7B continued pretraining on a single node, instrumented for sub-second power telemetry.

This is the proof-of-concept run for measuring GPU power and system metrics during realistic
pretraining. It starts from a converted OLMo 3 7B stage-1 checkpoint and trains on the Dolma 3.5
14T mix. Telemetry is collected without touching the training processes' hot path:

* ``power_sampler.py`` runs as a separate, pinned, low-priority process on each node. It is
  started from the Beaker ``post_setup`` hook (see :func:`finalize_config`).
* :class:`PowerTimelineCallback` records step-phase timestamps on every rank using host clocks
  and CUDA events, without adding any synchronization during training.

Launch on Beaker::

    python src/scripts/telemetry/olmo3_7b_power_poc.py launch RUN_NAME ai2/holmes [OVERRIDES...]

Set ``POWER_POC_SIDECAR=0`` in the launching environment to run without the sampler sidecar
(used as the throughput control).
"""

import json
import os
import socket
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import torch

from olmo_core.config import DType
from olmo_core.data import (
    DataMixBase,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerName,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank, get_rank
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    build_config,
    main,
)
from olmo_core.launch.beaker import BeakerEnvVar, BeakerWekaBucket
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import ConstantWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 8192
# 128 sequences (~1M tokens): 16 sequences per GPU per step on one 8-GPU node.
GLOBAL_BATCH_SIZE = 128 * SEQUENCE_LENGTH
# 4 sequences per microbatch -> 4 microbatches per step. Comfortably within B300 memory.
RANK_MICROBATCH_SEQUENCES = 4
MAX_STEPS = 220
SAVE_INTERVAL = 60
LR = 1e-4
WARMUP_STEPS = 50

PROJECT_DIR = "/weka/oe-training-default/jacobm/olmo-energy-data"
LOAD_PATH = f"{PROJECT_DIR}/checkpoints/olmo-core/Olmo-3-1025-7B-stage1-step1000000"
DOLMA_3P5_ROOT = "/weka/dolma-3p5/ai2-llm"
BEAKER_IMAGE = "akshitab/olmo-core-tch2130cu130-fa4-rma-2026-09-11"
WORKSPACE = "ai2/olmo3p5-training"

TELEMETRY_DIR_ENV = "POWER_TELEMETRY_DIR"
SIDECAR_ENV = "POWER_POC_SIDECAR"


class Dolma3p5Mix(DataMixBase):
    """
    The Dolma 3.5 14T pretraining mix, materialized under the dolma2 tokenizer. The manifest is a
    copy of ``ladders/mainline/workloads/mixes/Dolma3p5-14t.txt`` from ``allenai/scaling-ladders``.
    """

    dolma3p5_14t = "Dolma3p5-14t"

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        if tokenizer != TokenizerName.dolma2:
            raise ValueError(f"'{self}' is only materialized for the dolma2 tokenizer")
        base_dir = base_dir.rstrip("/") + "/"
        paths, labels = [], []
        with (Path(__file__).parent / f"{self.value}.txt").open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                label, path = line.split(",")
                paths.append(base_dir + path.replace("{TOKENIZER}", tokenizer))
                labels.append(label)
        return paths, labels


@dataclass
class PowerTimelineCallback(Callback):
    """
    Records training-phase markers on every rank so sidecar telemetry can be aligned with steps,
    optimizer updates, and checkpoint saves.

    Each marker stores the host ``CLOCK_MONOTONIC`` time when the hook ran. Step start, optimizer
    start, and step end also record a CUDA event, which is resolved later (without blocking) into
    the time the GPU actually reached that point, on the same monotonic clock. Rows are appended
    to ``<telemetry_dir>/<hostname>/timeline/rank<N>.jsonl``.

    When training ends, local rank 0 writes the sampler's stop file and waits for the sampler to
    finish its cool-down so the container stays up for it.
    """

    # Run before the checkpointer (priority 1) so a step's end marker precedes its save.
    priority: ClassVar[int] = 10

    flush_interval: int = 20
    sampler_wait_s: float = 120.0

    # Runtime state (unannotated so it is not part of the serialized config).
    _rows = None
    _pending = None
    _anchor = None
    _anchor_mono_ns = 0
    _path = None

    def _host_dir(self) -> Optional[Path]:
        base = os.environ.get(TELEMETRY_DIR_ENV)
        return None if not base else Path(base) / socket.gethostname()

    def _mark(self, event: str, cuda_event: bool = False, **extra):
        assert self._rows is not None and self._pending is not None
        row = {
            "step": self.step,
            "event": event,
            "host_mono_ns": time.monotonic_ns(),
            "host_wall_ns": time.time_ns(),
            **extra,
        }
        if cuda_event and self._anchor is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending.append((row, ev))
        self._rows.append(row)

    def _resolve(self, block: bool = False):
        assert self._pending is not None
        still_pending = []
        for row, ev in self._pending:
            if block or ev.query():
                if block:
                    ev.synchronize()
                ms = self._anchor.elapsed_time(ev)
                row["gpu_mono_ns"] = self._anchor_mono_ns + int(ms * 1e6)
            else:
                still_pending.append((row, ev))
        self._pending = still_pending

    def _flush(self):
        assert self._rows is not None and self._pending is not None
        if self._path is None:
            self._rows = []
            return
        # Only write rows whose CUDA events are resolved; keep the rest for the next flush.
        pending_ids = {id(row) for row, _ in self._pending}
        done = [r for r in self._rows if id(r) not in pending_ids]
        self._rows = [r for r in self._rows if id(r) in pending_ids]
        with open(self._path, "a") as f:
            for r in done:
                f.write(json.dumps(r) + "\n")

    def pre_train(self):
        self._rows, self._pending = [], []
        self._anchor = None
        self._path = None
        host_dir = self._host_dir()
        if host_dir is not None:
            (host_dir / "timeline").mkdir(parents=True, exist_ok=True)
            self._path = host_dir / "timeline" / f"rank{get_rank()}.jsonl"
        if torch.cuda.is_available():
            # One synchronization before training starts anchors GPU event times to the host clock.
            torch.cuda.synchronize()
            self._anchor = torch.cuda.Event(enable_timing=True)
            t0 = time.monotonic_ns()
            self._anchor.record()
            torch.cuda.synchronize()
            self._anchor_mono_ns = (t0 + time.monotonic_ns()) // 2
        self._mark(
            "train_start",
            rank=get_rank(),
            local_rank=get_local_rank(),
            global_batch_tokens=GLOBAL_BATCH_SIZE,
            hostname=socket.gethostname(),
        )

    def pre_step(self, batch):
        del batch
        self._resolve()
        self._mark("step_start", cuda_event=True)

    def pre_optim_step(self):
        self._mark("optim_start", cuda_event=True)

    def post_step(self):
        self._mark("step_end", cuda_event=True)
        if self.step % self.flush_interval == 0:
            self._flush()

    def post_checkpoint_saved(self, path):
        self._mark("checkpoint_saved", path=str(path))

    def post_train(self):
        self._mark("train_end")
        self._finish(wait_for_sampler=True)

    def on_error(self, exc: BaseException):
        del exc
        if self._rows is not None:
            self._mark("train_error")
            self._finish(wait_for_sampler=False)

    def _finish(self, wait_for_sampler: bool):
        if self._pending is None or self._rows is None:
            return
        self._resolve(block=True)
        self._flush()
        host_dir = self._host_dir()
        if host_dir is None or get_local_rank() != 0 or os.environ.get(SIDECAR_ENV) == "0":
            return
        (host_dir / "STOP").write_text(str(time.time_ns()))
        if wait_for_sampler:
            deadline = time.monotonic() + self.sampler_wait_s
            while time.monotonic() < deadline and not (host_dir / "DONE").exists():
                time.sleep(1.0)
        self._pending, self._rows = None, None


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo3_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_4,
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SEQUENCES * common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=ConstantWithWarmup(warmup=WARMUP_STEPS),
    )


def build_data_components(common: CommonComponents) -> DataComponents:
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        Dolma3p5Mix.dolma3p5_14t,
        tokenizer=common.tokenizer,
        mix_base_dir=DOLMA_3P5_ROOT,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=common.max_sequence_length,
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
    )
    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10
    run_dir = f"{PROJECT_DIR}/runs/{common.run_name}"
    return (
        TrainerConfig(
            save_folder=f"{run_dir}/checkpoints",
            save_overwrite=True,
            load_path=LOAD_PATH,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL, ephemeral_save_interval=None, save_async=False
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                group="olmo3-7b-power-poc",
                entity="ai2-llm",
                project="olmo-energy-poc",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        # Kernel-level timeline for two steps near the end of the run (rank 0 only). The
        # profiler only records inside this window; those steps are excluded from analysis.
        .with_callback(
            "profiler",
            ProfilerCallback(
                skip_first=MAX_STEPS - 16, wait=1, warmup=1, active=2, with_stack=False
            ),
        )
        .with_callback("power_timeline", PowerTimelineCallback())
    )


def _sampler_post_setup(telemetry_dir: str) -> str:
    host_dir = f"{telemetry_dir}/$(hostname)"
    return (
        "pip install -q -U nvidia-ml-py && "
        f'mkdir -p "{host_dir}" && '
        "(nohup setsid python src/scripts/telemetry/power_sampler.py "
        f'--out-dir "{host_dir}" > "{host_dir}/sampler.log" 2>&1 &) && '
        f'sleep 10 && tail -n 5 "{host_dir}/sampler.log"'
    )


def finalize_config(config: ExperimentConfig):
    telemetry_dir = f"{PROJECT_DIR}/runs/{config.run_name}/telemetry"
    if config.launch is None:
        return
    config.launch.priority = "urgent"
    config.launch.budget = "ai2/oe-other"
    dolma = BeakerWekaBucket("dolma-3p5", "/weka/dolma-3p5")
    if dolma not in config.launch.weka_buckets:
        config.launch.weka_buckets.append(dolma)
    training_bucket = BeakerWekaBucket("oe-training-default", "/weka/oe-training-default")
    if training_bucket not in config.launch.weka_buckets:
        config.launch.weka_buckets.append(training_bucket)
    sidecar = os.environ.get(SIDECAR_ENV, "1") != "0"
    env = {e.name: e for e in config.launch.env_vars}
    env[TELEMETRY_DIR_ENV] = BeakerEnvVar(name=TELEMETRY_DIR_ENV, value=telemetry_dir)
    env[SIDECAR_ENV] = BeakerEnvVar(name=SIDECAR_ENV, value="1" if sidecar else "0")
    config.launch.env_vars = list(env.values())
    if sidecar:
        config.launch.post_setup = _sampler_post_setup(telemetry_dir)


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=finalize_config,
        beaker_image=BEAKER_IMAGE,
        beaker_workspace=WORKSPACE,
        include_default_evals=False,
    )
    main(config_builder=config_builder)
