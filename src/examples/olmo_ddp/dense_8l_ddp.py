"""OLMo-owned DDP dense baseline for the tech-report benchmark."""

import math
from functools import partial

from moe_8l_common import (
    BETAS,
    D_ATTN,
    D_MODEL,
    DENSE_LAYER_MLP,
    GLOBAL_BATCH_SIZE,
    HEAD_DIM,
    LEARNING_RATE,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    RANK_MICROBATCH_SEQUENCES,
    SEED,
    SEQUENCE_LENGTH,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    build_data_components,
    build_trainer_config,
    finalize_config,
)

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_work_dir
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    SubCmd,
    build_config,
    train,
)
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
)


def _layer_norm() -> LayerNormConfig:
    return LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )


def _attention(layer_norm: LayerNormConfig) -> AttentionConfig:
    return AttentionConfig(
        name=AttentionType.default,
        n_heads=NUM_HEADS,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        bias=False,
        rope=RoPEConfig(
            name=RoPEType.default,
            theta=500_000,
            full_precision=True,
        ),
        qk_norm=layer_norm,
        use_head_qk_norm=True,
        backend=AttentionBackendName.flash_4,
        dtype=DType.float32,
    )


def benchmark_num_flops_per_token(vocab_size: int, seq_len: int = SEQUENCE_LENGTH) -> int:
    """Idealized dense FLOPs/token under the shared benchmark convention."""
    projection_flops = 6 * (
        D_MODEL * D_ATTN + 2 * D_MODEL * NUM_KV_HEADS * HEAD_DIM + D_ATTN * D_MODEL
    )
    attention_positions = seq_len * (seq_len + 1) // 2
    attention_flops = projection_flops + (
        12 * NUM_HEADS * HEAD_DIM * attention_positions // seq_len
    )
    mlp_flops = 18 * D_MODEL * DENSE_LAYER_MLP
    return NUM_LAYERS * (attention_flops + mlp_flops) + 6 * D_MODEL * vocab_size


def build_model_config(common: CommonComponents) -> OLMoDDPModelConfig:
    layer_norm = _layer_norm()
    block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        use_peri_norm=False,
        sequence_mixer=_attention(layer_norm),
        layer_norm=layer_norm,
        shared_experts=SharedExpertsConfig(
            d_model=D_MODEL,
            hidden_size=DENSE_LAYER_MLP,
            num_experts=1,
            bias=False,
            dtype=DType.float32,
        ),
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
    )
    config = OLMoDDPModelConfig(
        init_seed=SEED,
        d_model=D_MODEL,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(D_MODEL),
        embedding_norm=layer_norm,
        block=block,
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=DType.float32),
        name=TransformerType.moe_fused_v2,
        init_std=0.01,
        dtype=DType.float32,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    return config


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH,
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=BETAS,
            group_overrides=[
                OptimGroupOverride(
                    params=["*embeddings.weight"],
                    opts=dict(weight_decay=0.0, use_muon=False),
                )
            ],
            compile=True,
            dtype=DType.float32,
            sigma_factor=12,
            use_distributed=True,
        ),
        compile_model=True,
        ac_config=None,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=None,
        pp_config=None,
        float8_config=None,
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
            trainer_config_builder=partial(
                build_trainer_config,
                variant="dense-ddp",
                flops_per_token_builder=benchmark_num_flops_per_token,
            ),
            finalize_config=partial(finalize_config, variant="dense-ddp"),
            include_instance_filter=False,
            include_default_evals=False,
            flight_recorder=True,
        )
        train(config)
    finally:
        teardown_training_environment()
