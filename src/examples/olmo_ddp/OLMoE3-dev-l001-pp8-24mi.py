# Tech-report operating point: Large-128E, PP8, 24 Mi GBS, random routing (alternate).
# Identical to OLMoE3-dev-l001.py except: PP_DIM = 4 -> PP_DIM = 8; GLOBAL_BATCH_SIZE_SEQ = (8 * 8) * 2 * 32 -> GLOBAL_BATCH_SIZE_SEQ = (8 * 8) * 2 * 24.
import logging
import math
import os
import socket
from pathlib import Path


def _default_triton_cache_dir() -> None:
    if os.environ.get("TRITON_CACHE_DIR") or os.environ.get("OLMO_DISABLE_PER_RANK_TRITON_CACHE"):
        return
    local_rank = (
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
        or "0"
    )
    job_id = (
        os.environ.get("BEAKER_EXPERIMENT_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
        or "default"
    )
    host = socket.gethostname().split(".")[0] or "host"
    base = Path(os.environ.get("OLMO_TRITON_CACHE_BASE", "/tmp/olmo-triton-cache"))
    cache_dir = base / str(job_id) / host / f"local_rank_{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


_default_triton_cache_dir()

# Keep this before any olmo_core imports: several modules import nvtx at import
# time, and NVTX_DISABLE only works if it is set before nvtx is imported.
USE_NV_PROFILE = False
if not USE_NV_PROFILE:
    os.environ["NVTX_DISABLE"] = "1"


from typing import cast  # noqa: E402

import torch  # noqa: E402

from olmo_core.config import DType  # noqa: E402
from olmo_core.data import (  # noqa: E402
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType  # noqa: E402
from olmo_core.distributed.parallel.pipeline_parallel import (  # noqa: E402
    PipelineP2PBackend,
    PipelineScheduleType,
)
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
from olmo_core.nn.lm_head import LMLossImplementation  # noqa: E402
from olmo_core.nn.moe import (  # noqa: E402
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig  # noqa: E402
from olmo_core.nn.moe.v2.router import MoERouterConfigV2  # noqa: E402
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig  # noqa: E402
from olmo_core.nn.transformer import (  # noqa: E402
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import (  # noqa: E402
    OLMoDDPOptimizerConfig,
    OptimGroupOverride,
    SchedulerUnits,
)
from olmo_core.optim.scheduler import (  # noqa: E402
    ComposableScheduler,
    ComposableSchedulerStage,
    ComposableSchedulerStageType,
    OverrideDecay,
)
from olmo_core.train import (  # noqa: E402
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import WandBCallback  # noqa: E402
from olmo_core.train.callbacks import (  # noqa: E402
    CheckpointerCallback,
    NvidiaProfilerCallback,
    TorchMemoryHistoryCallback,
)
from olmo_core.train.train_module import (  # noqa: E402
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer import (  # noqa: E402
    TransformerPipelineParallelConfig,
)

# import transformer_engine  # unused on moe-v2-core; kept for reference


log = logging.getLogger(__name__)


def _get_split_points(original_num_layers: int, num_stages: int, minus_last_stage: int):
    assert (
        original_num_layers % num_stages == 0
    ), "Original number of layers must be divisible by number of stages"
    layers_per_stage = original_num_layers // num_stages

    new_num_layers = original_num_layers - minus_last_stage

    split_points = []
    for i in range(1, num_stages):
        split_points.append(i * layers_per_stage)
    # if minus_last_stage > 0:
    #     split_points.append(original_num_layers - minus_last_stage)
    return new_num_layers, split_points


SEQUENCE_LENGTH = 8192

torch.set_float32_matmul_precision("high")

import sys  # noqa: E402

EVAL_INTERVAL = 2000
SAVE_INTERVAL = 2000

NUM_EXPERTS = 128
TOP_K = 4
ORIGINAL_TOP_K = None
D_MODEL = 4 * 1024
D_ATTN = 5 * 1024

HEAD_DIM = 128
NUM_HEAD = D_ATTN // HEAD_DIM
NUM_KV_HEAD = NUM_HEAD // 4
MOE_HIDDEN_SIZE = 4 * 1024
NUM_SHARED_EXPERTS = 1  # Number of shared experts in the shared MLP
SHARED_MLP_HIDDEN_SIZE = (
    4 * 1024
)  # Hidden size for shared MLP (or dense branch MLP in arctic) in MoE blocks

EFFECTIVE_MLP = MOE_HIDDEN_SIZE * TOP_K + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS
MLP_RATIO = EFFECTIVE_MLP / D_MODEL

# the first dense layer MLP
DENSE_LAYER_MLP = TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS

# DP_DIM=2
EP_DIM = 8
PP_DIM = 8

# ref
REF_NUM_NODES = 64
TAG = "rep"

LR_ALPHA = 0.53

# stage 1 - xM -
MAX_DURATION = int(200e9)
MICRO_BSZ = 1
GLOBAL_BATCH_SIZE_SEQ = (8 * 8) * 2 * 24
LR = 1.2e-4  # the LR is set for stable stage
LR_REF_BSZ_IN_M = 8
USE_FP8 = False


GLOBAL_BATCH_SIZE = (GLOBAL_BATCH_SIZE_SEQ) * SEQUENCE_LENGTH
NUM_MICRO_BATCHES = GLOBAL_BATCH_SIZE_SEQ // (REF_NUM_NODES * 8) // MICRO_BSZ * PP_DIM

GLOBAL_BATCH_TOKENS_IN_M = GLOBAL_BATCH_SIZE // 1024 // 1024


SCHED_WARMUP_TOKENS = int((30e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
SCHED_FAST_DECAY_TOKENS = int((0e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
SCHED_LONG_DECAY_TOKENS = int((19990e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
SCHED_MID_FRACTION = 1.0
SCHED_FINAL_FRACTION = 1.0  # WSD


LR = LR / SCHED_MID_FRACTION  # transform LR to peak at fast warmup

LR = (
    LR * (GLOBAL_BATCH_SIZE / (LR_REF_BSZ_IN_M * 1024 * 1024)) ** LR_ALPHA
)  # lr is for X Million token

EXPERT_LR = LR
# EXPERT_LR = LR * math.sqrt(TOP_K / NUM_EXPERTS)  # scale lr for expert params, # 1/4.8989 = 0.204
# EXPERT_LR = LR * 0.5  # scale lr for expert params, empirical choice

ORIGINAL_NUM_LAYERS = 48
MINUS_LAST_STAGE = 1
NUM_LAYERS = ORIGINAL_NUM_LAYERS - MINUS_LAST_STAGE

if PP_DIM > 1:
    _, SPLIT_POINTS = _get_split_points(
        ORIGINAL_NUM_LAYERS, PP_DIM * 2, minus_last_stage=MINUS_LAST_STAGE
    )
else:
    SPLIT_POINTS = None


############


# SPLIT_POINTS = None
USE_COMPILE = True
USE_NO_SYNC_EP = True
# USE_AC=False
PER_LAYER_RECOMPUTE = False
USE_TBO = False
GRAD_ACC_IN_FP32 = True
GRAD_REDUCE_IN_FP32 = True
UNIFORM_ASSIGN = False
RANDOM_ASSIGN = True
USE_ROWWISE_A2A = True

USE_FP8_ATTN_QKV = USE_FP8
USE_FP8_ATTN_OUT = USE_FP8
USE_FP8_ATTN_SAVE_QKV = False

# USE_FP8_ATTN_QKV=False
# USE_FP8_ATTN_OUT=False
# USE_FP8_ATTN_SAVE_QKV=False

FP8_FUSED_AUTOGRAD_RECOMPUTE_SWIGLU = False
ROWWISE_A2A_NBLOCKS = (
    256 if EP_DIM <= 8 else 64
)  # for intra-node, can use more blocks to increase overlap; for inter-node, the bottleneck is the network, so fewer blocks can reduce overhead.
SEED = 2026
USE_MUON = False
USE_PERI_NORM = True
PRODUCTION_RUN = True
EP_NO_SYNC_CAPACITY_FACTOR = 1.1875
# save a little bit of memory
# import torch._functorch.config  # Force initialization by accessing dynamo first
# torch._functorch.config.activation_memory_budget = 0.1

if RANDOM_ASSIGN:
    TAG += "-R"

from olmo_core.nn.attention import AttentionConfig, AttentionType  # noqa: E402
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType  # noqa: E402
from olmo_core.nn.lm_head import LMHeadConfig  # noqa: E402
from olmo_core.nn.rope import RoPEConfig, RoPEType  # noqa: E402


def build_model_config(common: CommonComponents) -> OLMoDDPModelConfig:
    from olmo_core.nn.attention.backend import AttentionBackendName
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
    from olmo_core.nn.moe.v2.ep_config import (
        ExpertParallelConfig,
        ExpertParallelPath,
        ExpertParallelSchedule,
    )
    from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config

    d_model = D_MODEL
    dtype = DType.float32

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=dtype,
    )
    use_block_no_sync_ep = USE_NO_SYNC_EP and EP_DIM > 1
    block_ep_path = (
        ExpertParallelPath.rowwise_nvshmem
        # ExpertParallelPath.deepep_v2
        if use_block_no_sync_ep and USE_ROWWISE_A2A
        else ExpertParallelPath.no_sync_1d
        if use_block_no_sync_ep
        else ExpertParallelPath.sync_1d
    )
    block_ep_schedule = (
        ExpertParallelSchedule.tbo
        if USE_TBO and block_ep_path == ExpertParallelPath.rowwise_nvshmem
        else ExpertParallelSchedule.normal
    )
    config = OLMoDDPModelConfig(
        init_seed=SEED,
        d_model=d_model,
        two_batch_overlap=USE_TBO,
        recompute_each_block=PER_LAYER_RECOMPUTE,
        recompute_all_blocks_by_chunk=False,
        # recompute_block_keys=["0", "1", "2"], # recompute dense layer
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(d_model),
        embedding_norm=layer_norm,
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            use_peri_norm=USE_PERI_NORM,
            ep=ExpertParallelConfig(
                path=block_ep_path,
                schedule=block_ep_schedule,
                share_combine_out=PER_LAYER_RECOMPUTE,  # if layer-recompute, want to make combine_out shared (not per-layer persistent) to save memory; extra copy overhead applies.
                share_dispatch_out=PER_LAYER_RECOMPUTE,  # if layer-recompute, want to make dispatch_out shared (not per-layer persistent) to save memory; extra copy overhead applies.
                shared_slots=2 if block_ep_schedule == ExpertParallelSchedule.tbo else 1,
                rowwise_get_nblocks=ROWWISE_A2A_NBLOCKS,
                rowwise_put_nblocks=ROWWISE_A2A_NBLOCKS,
                rowwise_weighted_put_nblocks=min(ROWWISE_A2A_NBLOCKS, 128),
                capacity_factor=EP_NO_SYNC_CAPACITY_FACTOR,
            ),
            checkpoint_permute_moe_unpermute=False,
            checkpoint_attn=False,
            checkpoint_second_unpermute=False,
            rowwise_fp8=MoERowwiseFP8Config(
                enabled=USE_FP8, fused_autograd_recompute_swiglu=FP8_FUSED_AUTOGRAD_RECOMPUTE_SWIGLU
            )
            if USE_ROWWISE_A2A
            else None,
            sequence_mixer=AttentionConfig(
                name=AttentionType.fused_v2,
                # name=AttentionType.default,
                n_heads=NUM_HEAD,
                head_dim=HEAD_DIM,  # attn dim = n_heads*head_dim (was d_attn; d_attn field is gone in moe-v2-core)
                n_kv_heads=NUM_KV_HEAD,
                bias=False,
                rope=RoPEConfig(
                    name=RoPEType.default, theta=500_000, scaling=None, full_precision=True
                ),
                qk_norm=layer_norm,
                backend=AttentionBackendName.flash_4,
                use_head_qk_norm=True,
                dtype=dtype,
                # d_attn=D_ATTN,
                mxfp8_qkv_projection=USE_FP8_ATTN_QKV,
                mxfp8_out_projection=USE_FP8_ATTN_OUT,
                mxfp8_save_qkv_for_backward=USE_FP8_ATTN_SAVE_QKV,
                use_recompute_qkv_prep=False,
                # use_recompute_qkv_prep=not PER_LAYER_RECOMPUTE, # only enable when not doing per-layer recompute
            ),
            layer_norm=layer_norm,
            routed_experts=RoutedExpertsConfig(
                d_model=d_model,
                hidden_size=MOE_HIDDEN_SIZE,
                num_experts=NUM_EXPERTS,
                bias=False,
                dtype=dtype,
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                gating_function=MoERouterGatingFunction.softmax,
                uniform_expert_assignment=UNIFORM_ASSIGN,
                random_expert_assignment=RANDOM_ASSIGN,
                lb_loss_weight=0.005,
                z_loss_weight=2e-4,
                lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
                dtype=dtype,
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
                # use_recompute_fp32_cast=not PER_LAYER_RECOMPUTE, # only enable when not doing per-layer recompute
                use_recompute_fp32_cast=False,
                original_top_k=ORIGINAL_TOP_K,
            ),
            shared_experts=SharedExpertsConfig(
                d_model=d_model,
                hidden_size=SHARED_MLP_HIDDEN_SIZE,
                num_experts=NUM_SHARED_EXPERTS,
                bias=False,
                dtype=dtype,
            )
            if NUM_SHARED_EXPERTS > 0
            else None,
            shared_experts_router=MoERouterConfigV2(
                d_model=d_model,
                num_experts=NUM_SHARED_EXPERTS,
                top_k=NUM_SHARED_EXPERTS,  # all experts are used
                gating_function=MoERouterGatingFunction.sigmoid,
                uniform_expert_assignment=False,
                lb_loss_weight=None,
                z_loss_weight=None,
                lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
                dtype=dtype,
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
                # use_recompute_fp32_cast=not PER_LAYER_RECOMPUTE, # only enable when not doing per-layer recompute
                use_recompute_fp32_cast=False,
            )
            if NUM_SHARED_EXPERTS > 1
            else None,  # only need router if > 1 expert
            # feed_forward_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,
        init_std=0.01,
        dtype=dtype,
    )

    # config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    config.lm_head.loss_implementation = LMLossImplementation.default
    # WINDOW_SIZE=2048
    # config.block.sequence_mixer.sliding_window = SlidingWindowAttentionConfig(
    #     force_full_attention_on_first_layer=False,
    #     force_full_attention_on_last_layer=True,
    #     pattern=[WINDOW_SIZE, -1]
    # )

    dense_block_config = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        use_peri_norm=USE_PERI_NORM,
        rowwise_fp8=MoERowwiseFP8Config(
            enabled=USE_FP8, fused_autograd_recompute_swiglu=FP8_FUSED_AUTOGRAD_RECOMPUTE_SWIGLU
        )
        if USE_ROWWISE_A2A
        else None,
        sequence_mixer=AttentionConfig(
            name=AttentionType.fused_v2,
            # name=AttentionType.default,
            n_heads=NUM_HEAD,
            head_dim=HEAD_DIM,  # attn dim = n_heads*head_dim (was d_attn; d_attn field is gone in moe-v2-core)
            n_kv_heads=NUM_KV_HEAD,
            bias=False,
            rope=RoPEConfig(
                name=RoPEType.default, theta=500_000, scaling=None, full_precision=True
            ),
            qk_norm=layer_norm,
            backend=AttentionBackendName.flash_4,
            use_head_qk_norm=True,
            dtype=dtype,
            # d_attn=D_ATTN,
            mxfp8_qkv_projection=USE_FP8_ATTN_QKV,
            mxfp8_out_projection=USE_FP8_ATTN_OUT,
            mxfp8_save_qkv_for_backward=USE_FP8_ATTN_SAVE_QKV,
            use_recompute_qkv_prep=False,
            # use_recompute_qkv_prep=not PER_LAYER_RECOMPUTE, # only enable when not doing per-layer recompute
        ),
        routed_experts=None,
        routed_experts_router=None,
        shared_experts=SharedExpertsConfig(
            d_model=d_model,
            hidden_size=DENSE_LAYER_MLP,
            num_experts=1,
            bias=False,
            dtype=dtype,
        ),
        shared_experts_router=None,
        layer_norm=layer_norm,
        # feed_forward_norm=layer_norm,
    )
    from copy import deepcopy

    # First block will be a regular transformer block (no MoE component).
    config.block_overrides = {
        0: deepcopy(dense_block_config),
        1: deepcopy(dense_block_config),
        # also make last layer dense
        # NUM_LAYERS-1: deepcopy(dense_block_config),
    }

    return config


# patch
MONKEY_PATCH_DECAY_START_TOKENS: int | None = None  # None for disabled, or int((1000e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE) for starting at 1T tokens
MONKEY_PATCH_DECAY_DURATION_TOKENS = int((200e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
MONKEY_PATCH_DECAY_END_FRACTION = SCHED_FINAL_FRACTION
MONKEY_PATCH_DECAY_SHAPE = ComposableSchedulerStageType.cosine


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=MICRO_BSZ * SEQUENCE_LENGTH,
        max_sequence_length=common.max_sequence_length,
        # reset_optimizer_states_on_load=True, # TODO: only on first load step0,
        optim=OLMoDDPOptimizerConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            # muon_adjust_lr_fn="match_rms_adamw" if USE_MUON else None,
            group_overrides=[
                OptimGroupOverride(
                    params=[
                        "*embeddings.weight",
                        "*embedding_norm.weight",
                        "*q_norm.weight",
                        "*k_norm.weight",
                        "*input_norm.weight",
                        # "*norm.weight",
                        # "*lm_head.w_out.weight",
                        "*lm_head.norm.weight",
                        "*attention_norm.weight",
                        "*feed_forward_norm.weight",
                    ],
                    opts=dict(weight_decay=0.0, use_muon=False),
                ),
                # OptimGroupOverride(
                #     params=[
                #         "*.w_q.weight",
                #         "*.w_k.weight",
                #         "*.w_v.weight",
                #         "*attention.w_out.weight", # to exclude "lm_head.w_out.weight"
                #         "*.w1.weight", "*.w2.weight", "*.w3.weight"
                #         ], # attention + dense mlp
                #     opts=dict(use_muon=USE_MUON),
                # ),
                # OptimGroupOverride(
                #     params=["*routed_experts.w_up_gate", "*routed_experts.w_down"],
                #     opts=dict(lr=EXPERT_LR, use_muon=USE_MUON),
                # ),
            ],
            compile=USE_COMPILE,
            dtype=DType.float32,
            sigma_factor=12,
            use_distributed=True,
            check_nan_inf_grad=False,
        ),
        compile_model=USE_COMPILE,
        ac_config=None,  # AC handled elsewhere for MoE
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=GRAD_REDUCE_IN_FP32,
            accumulate_grads_in_fp32=GRAD_ACC_IN_FP32,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_DIM)
        if EP_DIM != 1
        else None,  # EP=1 means no expert parallel
        pp_config=TransformerPipelineParallelConfig(
            degree=PP_DIM,
            # schedule=PipelineScheduleType.custom_1F1B,
            # schedule=PipelineScheduleType.custom_1F1B_V,  # V placement for comparison against interleaved 1F1B
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            use_custom_stage_implementation=True,  # use custom stage implementation that re-uses receive buffers across micro-batches
            p2p_use_separate_group=True,
            p2p_backend=PipelineP2PBackend.nccl,
            # p2p_backend="nccl_rma_ack",
            # p2p_nccl_min_ctas=1,
            # p2p_nccl_max_ctas=2,
            # forward_pull_ahead_extra_activations=[1, 0, 1, 1],
            split_points=SPLIT_POINTS,
        )
        if PP_DIM > 1
        else None,
        float8_config=None,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        scheduler=ComposableScheduler(
            units=SchedulerUnits.tokens,
            stages=[
                ComposableSchedulerStage(
                    duration=SCHED_WARMUP_TOKENS,
                    shape=ComposableSchedulerStageType.linear,
                    start_lr_fraction=0.0,
                    end_lr_fraction=1.0,
                ),
                # ComposableSchedulerStage(
                #     duration=SCHED_FAST_DECAY_TOKENS,
                #     shape=ComposableSchedulerStageType.cosine,
                #     end_lr_fraction=SCHED_MID_FRACTION,
                # ),
                ComposableSchedulerStage(
                    duration=SCHED_LONG_DECAY_TOKENS,
                    shape=ComposableSchedulerStageType.cosine,
                    end_lr_fraction=SCHED_FINAL_FRACTION,
                ),
            ],
            override_decay=(
                OverrideDecay(
                    start=MONKEY_PATCH_DECAY_START_TOKENS,
                    duration=MONKEY_PATCH_DECAY_DURATION_TOKENS,
                    shape=MONKEY_PATCH_DECAY_SHAPE,
                    end_lr_fraction=MONKEY_PATCH_DECAY_END_FRACTION,
                )
                if MONKEY_PATCH_DECAY_START_TOKENS is not None
                else None
            ),
        ),
    )


WORK_DIR = "/workspace"


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    from olmo_core.train.checkpoint import CheckpointerConfig

    config = (
        TrainerConfig(
            save_folder=f"{WORK_DIR}/checkpoint/{common.run_name}_{D_MODEL}d{D_ATTN}a_{NUM_LAYERS}L{MOE_HIDDEN_SIZE}M{SHARED_MLP_HIDDEN_SIZE}S_{NUM_EXPERTS}E{TOP_K}K{NUM_SHARED_EXPERTS}S_{TAG}",
            # load_path="/workspace/checkpoint/OLMoE3-dev-260429-t001_2048d2560a_16L2048M1536S_40E4K1S_p1/step83448",
            save_overwrite=True,
            checkpointer=CheckpointerConfig(
                save_thread_count=3, load_thread_count=2, throttle_uploads=True
            ),
            metrics_collect_interval=20,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
            # steps_to_skip=[StepSkipRange(start=41312, stop=41329)]
            checkpoints_to_eval=[
                # '/workspace/checkpoint/OLMoE3-dev-260614-s002_2048d4096a_31L2560M2560S_64E4K1S_p1/step15*00',
            ],
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                save_async=False,
                # pre_train_checkpoint=PRODUCTION_RUN,
                pre_train_checkpoint=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="olmoe-dev-v2",
                enabled=PRODUCTION_RUN,
                # enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "profiler",
            NvidiaProfilerCallback(
                enabled=USE_NV_PROFILE,
                profile_ranks=list(range(0, 8 * 32, 8)),
                start=21 if RANDOM_ASSIGN else 151,
                end=25 if RANDOM_ASSIGN else 155,
            ),
        )
        .with_callback(
            "torch_mem_history",
            TorchMemoryHistoryCallback(
                enabled=False,  # NOTE: change this
                profile_ranks=list(range(0, 8 * 128, 8)),
                start=59161,
                end=59164,
                output_dir="/workspace/tmp",
            ),
        )
    )

    return config


def finalize_config(config: ExperimentConfig):
    # add active & total params to the wandb name
    total_params_in_B = config.model.num_params / 1000 / 1000 / 1000
    active_params_in_B = config.model.num_active_params / 1000 / 1000 / 1000
    log.info(f"Total params: {total_params_in_B:.2f}B, Active params: {active_params_in_B:.2f}B")

    wandb_cb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    wandb_original_name = wandb_cb.name
    assert isinstance(wandb_cb.name, str), "WandB callback name must be initialized"
    wandb_cb.name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"
    wandb_cb.name += f"_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"
    # _{EP_DIM}EP{PP_DIM}PP
    wandb_cb.group = f"{wandb_original_name}_{active_params_in_B:.2f}@{total_params_in_B:.2f}B_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = True,
) -> DataComponents:
    # DATA_ROOT = "/weka/oe-training-default/ai2-llm"
    # DATA_WORK_DIR = "/tmp/dataset-cache"

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        # DataMix.OLMo_mix_0625,
        # DataMix.OLMoE_mix_0824_dev,
        tokenizer=common.tokenizer,
        # mix_base_dir=common.root_dir,
        mix_base_dir="s3://ai2-llm",
        # mix_base_dir="/workspace/data/ai2-llm",
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=8,
        prefetch_factor=8,
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def _build_common_components(
    cli_context: CliContext,
    *,
    tokenizer: TokenizerConfig,
    global_batch_size: int,
    max_sequence_length: int,
    **_,
) -> CommonComponents:
    # Resolve storage locally under WORK_DIR instead of the default cluster-based Beaker lookup,
    # so the script only needs a run name (no cluster arg) and no Beaker access at config-build
    # time. build_trainer_config sets the actual save_folder under WORK_DIR anyway.
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
    # Generic-launcher style: train directly under torchrun (or the generic Beaker launcher,
    # `python -m olmo_core.launch.beaker ... -- <script>.py RUN_NAME`). No bespoke launch/
    # subcommand dispatch — the launcher handles Beaker submission and torchrun wrapping.
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
            flight_recorder=True,
            include_instance_filter=True,
            include_default_evals=False,
            finalize_config=finalize_config,
        )
        train(config)
    finally:
        teardown_training_environment()
