"""
Configuration helpers for LLM cooking.

Indended to aid those migrating from olmo-cookbook to Olmo-core. Implements the same
defaults as olmo-cookbook where possible, and even includes some QOL improvements.
"""

import logging
from typing import Dict, Optional

import torch

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.common import get_beaker_username
from olmo_core.io import dir_is_empty
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    BeakerCallback,
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GAPMonitorCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    ProfilerCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)


def configure_train_module(
    max_sequence_length: int,
    rank_microbatch_size: int,
    learning_rate: float,
    scheduler: Scheduler,
    float8_enabled: bool = False,
    activation_memory_budget: float = 1.0,  # smaller memory budget means more checkpointing
    dp_shard_degree: Optional[int] = None,
    cp_degree: Optional[int] = None,
) -> TransformerTrainModuleConfig:
    if not (0.0 < activation_memory_budget <= 1.0):
        raise ValueError("activation_memory_budget must be in the range [0.0, 1.0].")
    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            foreach=True,
        ),
        scheduler=scheduler,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=dp_shard_degree,
        ),
        cp_config=(
            TransformerContextParallelConfig.llama3(degree=cp_degree, head_stride=4)
            if cp_degree
            else None
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=activation_memory_budget,
        )
        if activation_memory_budget < 1.0
        else None,
        float8_config=Float8Config(enabled=float8_enabled, ao=AOFloat8LinearConfig.recommended()),
    )


def configure_trainer(
    max_duration: Duration,
    checkpoint_dir: str,
    work_dir: str,
    load_path: Optional[str] = None,
    load_trainer_state: Optional[bool] = None,
    load_optim_state: Optional[bool] = None,
    hard_stop: Optional[Duration] = None,
) -> TrainerConfig:
    load_strategy = LoadStrategy.always if load_path else LoadStrategy.if_available
    if load_path and dir_is_empty(load_path):
        raise FileNotFoundError(f"{load_path=} was provided, but the directory is empty.")
    trainer_config = TrainerConfig(
        max_duration=max_duration,
        load_path=load_path,
        save_folder=checkpoint_dir,
        work_dir=work_dir,
        hard_stop=hard_stop,
        load_strategy=load_strategy,
        load_trainer_state=load_trainer_state,
        load_optim_state=load_optim_state,
        save_overwrite=True,
    )
    return trainer_config


def configure_required_callbacks(run_name: str) -> Dict[str, Callback]:
    callbacks = {
        "config_saver": ConfigSaverCallback(),
        "profiler": ProfilerCallback(enabled=False),
        "garbage_collector": GarbageCollectorCallback(),
        "slack_notifier": SlackNotifierCallback(name=run_name, enabled=False),
        "gap_monitor": GAPMonitorCallback(enabled=False),
    }
    beaker_user = get_beaker_username()
    if beaker_user is not None:
        callbacks["beaker"] = BeakerCallback()
    if torch.cuda.is_available():
        callbacks["gpu_monitor"] = GPUMemoryMonitorCallback()
    return callbacks


def configure_default_callbacks(
    run_name: str,
    wandb_group_name: str,
    wandb_project: str = "olmo-cookbook",
    checkpoint_save_interval: int | None = 1000,
    ephemeral_checkpoint_save_interval: int | None = None,
) -> Dict[str, Callback]:
    callbacks = configure_required_callbacks(run_name)
    callbacks["checkpointer"] = CheckpointerCallback(
        save_interval=checkpoint_save_interval,
        ephemeral_save_interval=ephemeral_checkpoint_save_interval,
        save_async=True,
    )
    callbacks["wandb"] = WandBCallback(
        name=run_name,
        group=wandb_group_name,
        project=wandb_project,
        entity="ai2-llm",
        cancel_check_interval=20,
        enabled=True,
    )
    return callbacks
