"""
Train a 7B OLMo model on long contexts. Run this script without any arguments to see usage info.
"""

import logging
from functools import partial

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, build_config, main
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
)
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)
CONTEXT_LENGTH = 4 * 16_384
GLOBAL_BATCH_SIZE = 64 * CONTEXT_LENGTH
INTRA_DOCUMENT_MASKING = True
# 64K length, 32 GPUs, FP8, no intra-doc masking -> 2,750 TPS
# 64K length, 32 GPUs, no FP8, intra-doc masking -> 3,250 TPS
# 64K length, 32 GPUs, FP8, intra-doc masking    -> 3,500 TPS


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo2_7B(
        vocab_size=common.tokenizer.padded_vocab_size(), attn_backend=AttentionBackendName.flash_2
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=AdamWConfig(
            lr=1e-5,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        z_loss_multiplier=1e-5,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.fine_grained,
        ),
        cp_config=TransformerContextParallelConfig.llama3(degree=8)
        if INTRA_DOCUMENT_MASKING
        else TransformerContextParallelConfig.zig_zag(degree=8),
        float8_config=Float8Config(enabled=False),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(units=SchedulerUnits.steps, warmup=2000),
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
                ephemeral_save_interval=250,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMo-core-7B",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-7B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=CONTEXT_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        intra_document_masking=INTRA_DOCUMENT_MASKING,
    )
    main(config_builder=config_builder)
