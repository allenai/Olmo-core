"""FSDP-based (HSDP) side of the configurable eight-layer MoE benchmark.

This follows the OLMo3-7B data-parallel recipe: parameters are sharded within
an 8-rank HSDP group and replicas scale across groups. Expert parallelism is
intentionally disabled, so executable expert weights are materialized through
FSDP rather than kept on fixed EP owners.
"""

import math
from functools import partial

from moe_8l_common import (
    BETAS,
    CAPACITY_FACTOR,
    D_MODEL,
    DENSE_LAYER_MLP,
    GLOBAL_BATCH_SIZE,
    HEAD_DIM,
    LEARNING_RATE,
    MOE_HIDDEN_SIZE,
    NUM_EXPERTS,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    PARALLEL_DEGREE,
    RANK_MICROBATCH_SEQUENCES,
    SEED,
    SEQUENCE_LENGTH,
    SHARED_MLP_HIDDEN_SIZE,
    TOP_K,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    build_data_components,
    build_trainer_config,
    finalize_config,
)

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import get_work_dir
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    SubCmd,
    build_config,
    train,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.moe import (
    MoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterConfig,
    MoERouterGatingFunction,
    MoEType,
)
from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)


def _make_first_block_dense(block):
    block.name = TransformerBlockType.reordered_norm
    block.feed_forward_moe = None
    block.feed_forward = FeedForwardConfig(
        hidden_size=DENSE_LAYER_MLP,
        bias=False,
        dtype=DType.float32,
    )
    return block


def build_model_config(common: CommonComponents) -> TransformerConfig:
    moe = MoEConfig(
        # The hybrid block's combined forward calls the dropless expert API
        # (global_permute_mlp_unpermute[_no_ep]), so it must build
        # ParallelDroplessMLP rather than the capacity-dropping ParallelMLP.
        name=MoEType.dropless,
        num_experts=NUM_EXPERTS,
        hidden_size=MOE_HIDDEN_SIZE,
        capacity_factor=CAPACITY_FACTOR,
        router=MoERouterConfig(
            top_k=TOP_K,
            gating_function=MoERouterGatingFunction.softmax,
            normalize_expert_weights=1.0,
            uniform_expert_assignment=False,
            random_expert_assignment=True,
            dtype=DType.float32,
        ),
        lb_loss_weight=0.015,
        z_loss_weight=1e-4,
        lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
        scale_loss_by_num_layers=False,
        dtype=DType.float32,
    )
    config = TransformerConfig.llama_like(
        init_seed=SEED,
        d_model=D_MODEL,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        name=TransformerType.moe,
        block_name=TransformerBlockType.moe_hybrid_reordered_norm,
        block_mods={0: _make_first_block_dense},
        qk_norm=True,
        use_head_qk_norm=True,
        rope_theta=500_000,
        rope_full_precision=True,
        layer_norm_eps=1e-6,
        attn_backend=AttentionBackendName.flash_4,
        feed_forward_moe=moe,
        feed_forward=FeedForwardConfig(
            hidden_size=SHARED_MLP_HIDDEN_SIZE,
            bias=False,
            dtype=DType.float32,
        ),
        embed_scale=math.sqrt(D_MODEL),
        init_std=0.01,
        dtype=DType.float32,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=BETAS,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            # OLMo3-7B uses HSDP: FSDP within each shard group, replicated
            # across groups. With one 8-GPU node this degenerates to FSDP.
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=PARALLEL_DEGREE,
            prefetch_factor=0,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        ep_config=None,
        pp_config=None,
        ac_config=None,
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=WARMUP_STEPS),
    )


WORK_DIR = "/workspace"


def _build_common_components(
    cli_context: CliContext,
    *,
    tokenizer: TokenizerConfig,
    global_batch_size: int,
    max_sequence_length: int,
    **_,
) -> CommonComponents:
    # Resolve storage locally under WORK_DIR instead of the cluster-based Beaker lookup, so the
    # script needs only a run name.
    return CommonComponents(
        run_name=cli_context.run_name,
        root_dir=WORK_DIR,
        work_dir=get_work_dir(WORK_DIR),
        save_folder=f"{WORK_DIR}/checkpoints/{cli_context.run_name}",
        launch=None,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        global_batch_size=global_batch_size,
    )


if __name__ == "__main__":
    # Generic-launcher style: `torchrun ... SCRIPT RUN_NAME [OVERRIDES...]`.
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: torchrun ... {sys.argv[0]} RUN_NAME [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        cli_context = CliContext(
            script=sys.argv[0],
            cmd=SubCmd.train,
            run_name=run_name,
            cluster="",
            overrides=list(overrides),
        )
        config = build_config(
            cli_context,
            common_config_builder=_build_common_components,
            global_batch_size=GLOBAL_BATCH_SIZE,
            max_sequence_length=SEQUENCE_LENGTH,
            data_config_builder=build_data_components,
            model_config_builder=build_model_config,
            train_module_config_builder=build_train_module_config,
            trainer_config_builder=partial(build_trainer_config, variant="fsdp-hsdp"),
            finalize_config=partial(finalize_config, variant="fsdp-hsdp"),
            include_instance_filter=False,
            include_default_evals=False,
            flight_recorder=True,
        )
        train(config)
    finally:
        teardown_training_environment()
