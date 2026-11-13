# Two-node launch (8 GPUs/node, using hostfile2):
# bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile2 /workspace/beaker-toolbox/node_cmd-tech-report-lbl.sh "moe_10l_ddp_lbl_0p50 token-gerrymandering-10l-lbl-0p50"

"""Self-contained ten-layer MoE run for the LBL-weight experiment."""

from __future__ import annotations

import logging
import math
import os
import socket
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, cast


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _set_default_triton_cache_dir() -> None:
    if os.environ.get("TRITON_CACHE_DIR") or os.environ.get("OLMO_DISABLE_PER_RANK_TRITON_CACHE"):
        return
    local_rank = (
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or "0"
    )
    job_id = (
        os.environ.get("BEAKER_EXPERIMENT_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
        or "tech-report-lbl"
    )
    host = socket.gethostname().split(".")[0] or "host"
    cache_root = Path(os.environ.get("OLMO_TRITON_CACHE_BASE", "/tmp/olmo-triton-cache"))
    cache_dir = cache_root / str(job_id) / host / f"local_rank_{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


PROFILE = _env_flag("TECH_REPORT_PROFILE", False)
PROFILE_START = int(os.environ.get("TECH_REPORT_PROFILE_START", 20))
PROFILE_END = int(os.environ.get("TECH_REPORT_PROFILE_END", 25))
if PROFILE_END <= PROFILE_START:
    raise ValueError(
        "TECH_REPORT_PROFILE_END must be greater than TECH_REPORT_PROFILE_START; "
        f"got {PROFILE_START=} and {PROFILE_END=}"
    )
if not PROFILE:
    # NVTX reads this at import time.
    os.environ.setdefault("NVTX_DISABLE", "1")
_set_default_triton_cache_dir()

import torch  # noqa: E402

from olmo_core.config import DType  # noqa: E402
from olmo_core.data import TokenizerConfig  # noqa: E402
from olmo_core.data import (  # noqa: E402
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType  # noqa: E402
from olmo_core.internal.common import get_work_dir  # noqa: E402
from olmo_core.internal.experiment import (  # noqa: E402
    CliContext,
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    SubCmd,
    build_config,
    train,
)
from olmo_core.nn.attention import AttentionConfig, AttentionType  # noqa: E402
from olmo_core.nn.attention.backend import AttentionBackendName  # noqa: E402
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig  # noqa: E402
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType  # noqa: E402
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation  # noqa: E402
from olmo_core.nn.moe import (  # noqa: E402
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.ep_config import (  # noqa: E402
    ExpertParallelConfig,
    ExpertParallelPath,
    ExpertParallelSchedule,
)
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig  # noqa: E402
from olmo_core.nn.moe.v2.router import MoERouterConfigV2  # noqa: E402
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig  # noqa: E402
from olmo_core.nn.rope import RoPEConfig, RoPEType  # noqa: E402
from olmo_core.nn.transformer import (  # noqa: E402
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride  # noqa: E402
from olmo_core.optim.scheduler import CosWithWarmup  # noqa: E402
from olmo_core.train import (  # noqa: E402
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (  # noqa: E402
    CheckpointerCallback,
    NvidiaProfilerCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (  # noqa: E402
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)

log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

# Experiment controls. Architecture, batching, routing, and LBL weight are
# intentionally constants so matched scripts differ only in LB_LOSS_WEIGHT and
# run-identifying strings.
SEED = 2026
SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # 4,194,304 tokens.
RANK_MICROBATCH_SEQUENCES = 4
MAX_TOKENS = 100_000_000_000
SAVE_INTERVAL = 2_000

D_MODEL = 2048
D_ATTN = 2048
HEAD_DIM = 128
NUM_HEADS = D_ATTN // HEAD_DIM
NUM_KV_HEADS = NUM_HEADS // 4
NUM_LAYERS = 10

NUM_EXPERTS = 64
TOP_K = 4
MOE_HIDDEN_SIZE = 2048
NUM_SHARED_EXPERTS = 0
SHARED_MLP_HIDDEN_SIZE = 0
DENSE_LAYER_MLP = TOP_K * MOE_HIDDEN_SIZE
CAPACITY_FACTOR = 1.25
EXPERT_PARALLEL_DEGREE = 1

LEARNING_RATE = 4e-4 * math.sqrt(GLOBAL_BATCH_SIZE / (2 * 1024 * 1024))
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_STEPS = 1_000
LB_LOSS_WEIGHT = 0.5
RUN_VARIANT = "lbl-0p50"


def benchmark_num_flops_per_token(vocab_size: int, seq_len: int = SEQUENCE_LENGTH) -> int:
    """Idealized FLOPs/token estimate used by the speed monitor."""
    projection_flops = 6 * (
        D_MODEL * D_ATTN + 2 * D_MODEL * NUM_KV_HEADS * HEAD_DIM + D_ATTN * D_MODEL
    )
    attention_positions = seq_len * (seq_len + 1) // 2
    attention_flops = projection_flops + (
        12 * NUM_HEADS * HEAD_DIM * attention_positions // seq_len
    )

    # Layer 0 is dense; layers 1--9 are MoE.
    flops = attention_flops + 18 * D_MODEL * DENSE_LAYER_MLP
    for _ in range(1, NUM_LAYERS):
        flops += attention_flops
        flops += 6 * D_MODEL * NUM_EXPERTS
        flops += 18 * D_MODEL * MOE_HIDDEN_SIZE * TOP_K
        flops += 18 * D_MODEL * SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS
    flops += 6 * D_MODEL * vocab_size
    return flops


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    del include_instance_filter
    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        tokenizer=common.tokenizer,
        mix_base_dir=os.environ.get("TECH_REPORT_DATA_ROOT", "s3://ai2-llm"),
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(common.max_sequence_length, SEQUENCE_LENGTH),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None,
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=8,
        prefetch_factor=8,
    )
    return DataComponents(dataset=dataset, data_loader=data_loader)


def build_trainer_config(
    common: CommonComponents,
    *,
    variant: str,
    flops_per_token_builder: Callable[[int, int], int] = benchmark_num_flops_per_token,
) -> TrainerConfig:
    save_root = Path(os.environ.get("TECH_REPORT_SAVE_ROOT", "/workspace/checkpoint/tech_report"))
    group = os.environ.get("TECH_REPORT_WANDB_GROUP", "token-gerrymandering-lbl-weight")

    return (
        TrainerConfig(
            save_folder=str(save_root / f"{common.run_name}-{variant}"),
            save_overwrite=True,
            no_checkpoints=False,
            no_evals=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                pre_train_checkpoint=False,
                save_async=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=f"{common.run_name}-{variant}",
                group=group,
                entity="ai2-llm",
                project=os.environ.get("TECH_REPORT_WANDB_PROJECT", "olmoe-tech-report"),
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "profiler",
            NvidiaProfilerCallback(
                enabled=PROFILE,
                profile_ranks=[0],
                start=PROFILE_START,
                end=PROFILE_END,
            ),
        )
        .with_callback(
            "speed_monitor",
            SpeedMonitorCallback(
                num_flops_per_token=flops_per_token_builder(
                    common.tokenizer.padded_vocab_size(),
                    common.max_sequence_length,
                )
            ),
        )
    )


def finalize_config(config: ExperimentConfig, *, variant: str) -> None:
    active_b = config.model.num_active_params / 1e9
    total_b = config.model.num_params / 1e9
    log.info(
        "%s model: %.6fB active / %.6fB total parameters",
        variant,
        active_b,
        total_b,
    )
    wandb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    wandb.name = f"{wandb.name}_{active_b:.3f}A-{total_b:.3f}T"


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
    block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        use_peri_norm=False,
        ep=ExpertParallelConfig(
            path=ExpertParallelPath.sync_1d,
            schedule=ExpertParallelSchedule.normal,
            shared_slots=1,
            rowwise_get_nblocks=128,
            rowwise_put_nblocks=128,
            rowwise_weighted_put_nblocks=128,
            share_dispatch_out=False,
            share_combine_out=False,
            capacity_factor=CAPACITY_FACTOR,
            checkpoint_tbo=False,
        ),
        rowwise_fp8=None,
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
            # Both benchmarking overrides must be false so the router learns.
            uniform_expert_assignment=False,
            random_expert_assignment=False,
            lb_loss_weight=LB_LOSS_WEIGHT,
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
        rowwise_fp8=None,
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
        block=block,
        block_overrides={0: deepcopy(dense_first_block)},
        lm_head=LMHeadConfig(
            layer_norm=layer_norm,
            bias=False,
            dtype=DType.float32,
        ),
        name=TransformerType.moe_fused_v2,
        recompute_each_block=False,
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
        ep_config=(
            TransformerExpertParallelConfig(degree=EXPERT_PARALLEL_DEGREE)
            if EXPERT_PARALLEL_DEGREE > 1
            else None
        ),
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
                variant=RUN_VARIANT,
            ),
            finalize_config=partial(finalize_config, variant=RUN_VARIANT),
            include_instance_filter=False,
            include_default_evals=False,
            flight_recorder=True,
        )
        train(config)
    finally:
        teardown_training_environment()
