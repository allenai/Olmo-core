"""
Train a large OLMoE model. Run this script without any arguments to see usage info.
"""

import logging
from functools import partial

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.experiment import CommonComponents, build_config, main
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerExpertParallelConfig,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 2048 * 4096

log = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    d_model = 4096
    return TransformerConfig.llama_like_moe(
        vocab_size=common.tokenizer.padded_vocab_size(),
        d_model=d_model,
        n_layers=32,
        n_heads=32,
        num_experts=64,
        top_k=2,
        expert_hidden_size=int(0.5 * d_model),
        shared_expert_hidden_size=d_model * 2,
        capacity_factor=1.05,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=2 * 4096,
        max_sequence_length=common.max_sequence_length,
        optim=AdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.bfloat16,
            num_replicas=1,  # to enable full-way expert parallel
            prefetch_factor=1,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ep_config=TransformerExpertParallelConfig(degree=-1),
        float8_config=Float8Config(ao=AOFloat8LinearConfig.recommended()),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
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
                project="OLMoE",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMoE",
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
        include_default_evals=False,
        # Preserve the original training environment. Run this historical script from its
        # corresponding Olmo-core revision, whose dependencies match this image.
        beaker_image=OLMoCoreBeakerImage.tch271_cu126,
        num_nodes=8,
    )
    main(config_builder=config_builder)
