"""Standalone eight-layer, 128-expert rowwise-NVSHMEM EP8 profile without shared experts."""

from __future__ import annotations

import math
import os
import socket
from copy import deepcopy
from pathlib import Path

import torch

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_work_dir
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
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
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    NvidiaProfilerCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)


def _set_default_triton_cache_dir() -> None:
    """Give every local rank an isolated compilation cache."""
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    local_rank = os.environ.get("LOCAL_RANK", "0")
    job_id = os.environ.get("JOB_ID", "moe-8l-ep8-rowwise-no-shared-profile")
    host = socket.gethostname().split(".")[0] or "host"
    cache_root = Path(os.environ.get("OLMO_TRITON_CACHE_BASE", "/tmp/olmo-triton-cache"))
    cache_dir = cache_root / job_id / host / f"local_rank_{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


_set_default_triton_cache_dir()


# Standalone, matched experiment constants. Only EP_PATH differs from the
# synchronized entry point.
SEED = 2026
SEQUENCE_LENGTH = 8192
RANK_MICROBATCH_SEQUENCES = int(os.environ.get("TECH_REPORT_RANK_MICROBATCH_SEQUENCES", "4"))
if RANK_MICROBATCH_SEQUENCES not in (2, 4):
    raise ValueError(
        f"TECH_REPORT_RANK_MICROBATCH_SEQUENCES must be 2 or 4; got {RANK_MICROBATCH_SEQUENCES}"
    )
# 524,288 tokens at rank microbatch 2. Scale proportionally so both choices
# retain four distributed microbatches per optimizer step on eight GPUs.
GLOBAL_BATCH_SIZE = 524_288 * RANK_MICROBATCH_SEQUENCES // 2
MAX_STEPS = 20
PROFILE_START = 15
PROFILE_END = 18

D_MODEL = 2560
D_ATTN = 2560
HEAD_DIM = 128
NUM_HEADS = D_ATTN // HEAD_DIM
NUM_KV_HEADS = NUM_HEADS // 4
NUM_LAYERS = 8

NUM_EXPERTS = 128
TOP_K = 8
MOE_HIDDEN_SIZE = 1280
NUM_SHARED_EXPERTS = 0
SHARED_MLP_HIDDEN_SIZE = 0
DENSE_LAYER_MLP = TOP_K * MOE_HIDDEN_SIZE
CAPACITY_FACTOR = 1.25
EP_DEGREE = 8
EP_PATH = ExpertParallelPath.rowwise_nvshmem

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_STEPS = 10

torch.set_float32_matmul_precision("high")


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


def build_model_config(common: CommonComponents) -> OLMoDDPModelConfig:
    layer_norm = _layer_norm()
    moe_block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        use_peri_norm=False,
        ep=ExpertParallelConfig(
            path=EP_PATH,
            schedule=ExpertParallelSchedule.normal,
            shared_slots=1,
            rowwise_get_nblocks=256,
            rowwise_put_nblocks=256,
            rowwise_weighted_put_nblocks=128,
            share_dispatch_out=False,
            share_combine_out=False,
            capacity_factor=CAPACITY_FACTOR,
            checkpoint_tbo=False,
        ),
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
            uniform_expert_assignment=False,
            random_expert_assignment=True,
            lb_loss_weight=0.015,
            z_loss_weight=1e-4,
            lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
            dtype=DType.float32,
            normalize_expert_weights=1.0,
            restore_weight_scale=True,
            use_recompute_fp32_cast=False,
        ),
        shared_experts=None,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
    )
    dense_first_block = OLMoDDPTransformerBlockConfig(
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
        two_batch_overlap=False,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(D_MODEL),
        embedding_norm=layer_norm,
        block=moe_block,
        block_overrides={0: deepcopy(dense_first_block)},
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=DType.float32),
        name=TransformerType.moe_fused_v2,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
        init_std=0.01,
        dtype=DType.float32,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    return config


def build_data_components(
    common: CommonComponents,
    include_instance_filter: bool = False,
) -> DataComponents:
    del include_instance_filter
    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        tokenizer=common.tokenizer,
        mix_base_dir="s3://ai2-llm",
        work_dir=common.work_dir,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=SEQUENCE_LENGTH,
        generate_doc_lengths=False,
        instance_filter_config=None,
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=8,
        prefetch_factor=8,
    )
    return DataComponents(dataset=dataset, data_loader=data_loader)


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
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
        ep_config=TransformerExpertParallelConfig(degree=EP_DEGREE),
        pp_config=None,
        float8_config=None,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=WARMUP_STEPS),
    )


def benchmark_num_flops_per_token(vocab_size: int) -> int:
    projection_flops = 6 * (
        D_MODEL * D_ATTN + 2 * D_MODEL * NUM_KV_HEADS * HEAD_DIM + D_ATTN * D_MODEL
    )
    attention_positions = SEQUENCE_LENGTH * (SEQUENCE_LENGTH + 1) // 2
    attention_flops = projection_flops + (
        12 * NUM_HEADS * HEAD_DIM * attention_positions // SEQUENCE_LENGTH
    )
    flops = attention_flops + 18 * D_MODEL * DENSE_LAYER_MLP
    for _ in range(1, NUM_LAYERS):
        flops += attention_flops
        flops += 6 * D_MODEL * NUM_EXPERTS
        flops += 18 * D_MODEL * MOE_HIDDEN_SIZE * TOP_K
        flops += 18 * D_MODEL * SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS
    flops += 6 * D_MODEL * vocab_size
    return flops


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=str(
                Path("/workspace/checkpoint/tech_report")
                / f"{common.run_name}-ep8-rowwise-no-shared-profile"
            ),
            save_overwrite=True,
            no_checkpoints=True,
            no_evals=True,
            metrics_collect_interval=5,
            cancel_check_interval=10,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                group=os.environ.get("TECH_REPORT_WANDB_GROUP", "moe-8l-ep8-profiles"),
                entity="ai2-llm",
                project=os.environ.get("TECH_REPORT_WANDB_PROJECT", "olmoe-tech-report"),
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "profiler",
            NvidiaProfilerCallback(
                enabled=True,
                profile_ranks=[0],
                start=PROFILE_START,
                end=PROFILE_END,
            ),
        )
        .with_callback(
            "speed_monitor",
            SpeedMonitorCallback(
                num_flops_per_token=benchmark_num_flops_per_token(
                    common.tokenizer.padded_vocab_size()
                )
            ),
        )
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
            trainer_config_builder=build_trainer_config,
            include_instance_filter=False,
            include_default_evals=False,
            flight_recorder=True,
        )
        train(config)
    finally:
        teardown_training_environment()
