"""DDP + EP side of the configurable eight-layer MoE benchmark."""

import math
from copy import deepcopy
from functools import partial
from typing import Any

from moe_8l_common import (
    BETAS,
    CAPACITY_FACTOR,
    D_MODEL,
    DENSE_LAYER_MLP,
    EP_PATH_NAME,
    FORCE_FUSED_ATTENTION,
    GLOBAL_BATCH_SIZE,
    HEAD_DIM,
    LEARNING_RATE,
    MOE_HIDDEN_SIZE,
    MXFP8_ATTN_OUT,
    MXFP8_ATTN_QKV,
    MXFP8_ATTN_SAVE_QKV,
    MXFP8_MLP,
    NUM_EXPERTS,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_SHARED_EXPERTS,
    PARALLEL_DEGREE,
    RANK_MICROBATCH_SEQUENCES,
    RECOMPUTE_EACH_BLOCK,
    ROWWISE_GET_NBLOCKS,
    ROWWISE_PUT_NBLOCKS,
    ROWWISE_WEIGHTED_PUT_NBLOCKS,
    SEED,
    SEQUENCE_LENGTH,
    SHARED_MLP_HIDDEN_SIZE,
    TOP_K,
    TWO_BATCH_OVERLAP,
    UNIFORM_ROUTING,
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
from olmo_core.nn.moe import MoELoadBalancingLossGranularity, MoERouterGatingFunction
from olmo_core.nn.moe.v2.ep_config import (
    ExpertParallelConfig,
    ExpertParallelPath,
    ExpertParallelSchedule,
)
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride, SchedulerUnits
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)


def _layer_norm() -> LayerNormConfig:
    return LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )


def _attention(layer_norm: LayerNormConfig) -> AttentionConfig:
    # Use the ordinary attention module with the same FlashAttention backend in
    # both configs. The benchmark is about sparse-model distribution, not an
    # attention-kernel comparison.
    use_mxfp8 = MXFP8_ATTN_QKV or MXFP8_ATTN_OUT or MXFP8_ATTN_SAVE_QKV
    mxfp8_options: dict[str, Any] = {}
    if use_mxfp8:
        # MXFP8 projection support is implemented by the fused-v2 attention
        # module. Do not pass these module-specific options to default
        # attention, even when their values are false.
        mxfp8_options = {
            "mxfp8_qkv_projection": MXFP8_ATTN_QKV,
            "mxfp8_out_projection": MXFP8_ATTN_OUT,
            "mxfp8_save_qkv_for_backward": MXFP8_ATTN_SAVE_QKV,
        }
    return AttentionConfig(
        name=(
            AttentionType.fused_v2 if FORCE_FUSED_ATTENTION or use_mxfp8 else AttentionType.default
        ),
        n_heads=NUM_HEADS,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        # d_attn=D_ATTN,
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
        **mxfp8_options,
    )


def _expert_parallel_path() -> ExpertParallelPath:
    if PARALLEL_DEGREE == 1:
        if EP_PATH_NAME not in {"auto", ExpertParallelPath.sync_1d.value}:
            raise ValueError(
                "TECH_REPORT_EP_PATH must be 'auto' or 'sync_1d' when "
                "TECH_REPORT_PARALLEL_DEGREE=1"
            )
        return ExpertParallelPath.sync_1d
    if EP_PATH_NAME == "auto":
        return ExpertParallelPath.rowwise_nvshmem
    return ExpertParallelPath(EP_PATH_NAME)


def build_model_config(common: CommonComponents) -> OLMoDDPModelConfig:
    layer_norm = _layer_norm()
    ep_path = _expert_parallel_path()
    if TWO_BATCH_OVERLAP:
        # The EP 'tbo' schedule selected below is not yet wired into the core block/train dispatch
        # (ExpertParallelConfig.validate rejects it), so enabling two-batch overlap would fail
        # during model construction. Reject it up front with a clear message until that path lands.
        raise ValueError(
            "TECH_REPORT_TWO_BATCH_OVERLAP=1 is not supported yet: the EP 'tbo' schedule is not "
            "wired into the core dispatch. Run without two-batch overlap."
        )
    if MXFP8_MLP and ep_path != ExpertParallelPath.rowwise_nvshmem:
        raise ValueError(
            "TECH_REPORT_MXFP8_MLP currently requires "
            "TECH_REPORT_EP_PATH=rowwise_nvshmem (or auto with EP > 1)"
        )
    rowwise_fp8 = (
        MoERowwiseFP8Config(
            enabled=True,
            fused_autograd_recompute_swiglu=False,
        )
        if MXFP8_MLP
        else None
    )
    block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        # False gives reordered/post-branch normalization, matching the generic
        # FSDP model more closely than the production peri-norm recipe.
        use_peri_norm=False,
        ep=ExpertParallelConfig(
            path=ep_path,
            schedule=(
                ExpertParallelSchedule.tbo if TWO_BATCH_OVERLAP else ExpertParallelSchedule.normal
            ),
            shared_slots=2 if TWO_BATCH_OVERLAP else 1,
            rowwise_get_nblocks=ROWWISE_GET_NBLOCKS,
            rowwise_put_nblocks=ROWWISE_PUT_NBLOCKS,
            rowwise_weighted_put_nblocks=ROWWISE_WEIGHTED_PUT_NBLOCKS,
            share_dispatch_out=RECOMPUTE_EACH_BLOCK,
            share_combine_out=RECOMPUTE_EACH_BLOCK,
            capacity_factor=CAPACITY_FACTOR,
            checkpoint_tbo=False,
        ),
        rowwise_fp8=rowwise_fp8,
        sequence_mixer=_attention(layer_norm),
        layer_norm=layer_norm,
        routed_experts=RoutedExpertsConfig(
            d_model=D_MODEL,
            hidden_size=MOE_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            bias=False,
            dtype=DType.float32,
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=D_MODEL,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=UNIFORM_ROUTING,
            random_expert_assignment=not UNIFORM_ROUTING,
            lb_loss_weight=0.015,
            z_loss_weight=1e-4,
            lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
            dtype=DType.float32,
            normalize_expert_weights=1.0,
            restore_weight_scale=True,
            use_recompute_fp32_cast=False,
        ),
        shared_experts=SharedExpertsConfig(
            d_model=D_MODEL,
            hidden_size=SHARED_MLP_HIDDEN_SIZE,
            num_experts=NUM_SHARED_EXPERTS,
            bias=False,
            dtype=DType.float32,
        ),
        # feed_forward_norm=layer_norm,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
    )
    dense_first_block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        use_peri_norm=False,
        rowwise_fp8=rowwise_fp8,
        sequence_mixer=_attention(layer_norm),
        layer_norm=layer_norm,
        shared_experts=SharedExpertsConfig(
            d_model=D_MODEL,
            hidden_size=DENSE_LAYER_MLP,
            num_experts=1,
            bias=False,
            dtype=DType.float32,
        ),
        # feed_forward_norm=layer_norm,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
    )

    config = OLMoDDPModelConfig(
        init_seed=SEED,
        d_model=D_MODEL,
        two_batch_overlap=TWO_BATCH_OVERLAP,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(D_MODEL),
        embedding_norm=layer_norm,
        block=block,
        block_overrides={0: deepcopy(dense_first_block)},
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=DType.float32),
        name=TransformerType.moe_fused_v2,
        recompute_each_block=RECOMPUTE_EACH_BLOCK,
        recompute_all_blocks_by_chunk=False,
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
        ep_config=TransformerExpertParallelConfig(degree=PARALLEL_DEGREE)
        if PARALLEL_DEGREE > 1
        else None,
        pp_config=None,
        float8_config=None,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(units=SchedulerUnits.steps, warmup=WARMUP_STEPS),
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
    # script needs only a run name. build_trainer_config sets the real save_folder (TECH_REPORT_SAVE_ROOT).
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
    # Generic-launcher style: `python -m olmo_core.launch.beaker ... -- moe_8l_ddp.py RUN_NAME`.
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
            trainer_config_builder=partial(build_trainer_config, variant="ddp-ep"),
            finalize_config=partial(finalize_config, variant="ddp-ep"),
            include_instance_filter=False,
            include_default_evals=False,
            flight_recorder=True,
        )
        train(config)
    finally:
        teardown_training_environment()
