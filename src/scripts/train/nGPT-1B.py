"""
Train a 1B nGPT model. Run this script without any arguments to see usage info.
"""

from functools import partial

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, build_config, main
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamConfig, CosWithWarmup, SchedulerUnits
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 1024 * 4096


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.ngpt_1B(
        vocab_size=common.tokenizer.padded_vocab_size(),
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=4 * 4096,  # TODO: can we increase this?
        max_sequence_length=common.max_sequence_length,
        optim=AdamConfig(
            lr=4e-4,
            betas=(0.9, 0.95),
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(units=SchedulerUnits.steps, warmup=0),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=1000,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMo-core-1B",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-1B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
    )
    main(config_builder=config_builder)
