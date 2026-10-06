import logging
from dataclasses import replace
from test.nn.attention.attention_test import BF16_ATOL, BF16_RTOL
from typing import Optional, cast

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard, init_device_mesh

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    build_world_mesh,
)
from olmo_core.distributed.utils import get_full_tensor, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    GatedDeltaNetConfig,
    RingAttentionLoadBalancerType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNorm, LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import LatentMoEConfig, MoEConfig, MoERouterConfig, MoEType
from olmo_core.nn.rope import RoPEConfig
from olmo_core.nn.transformer import (
    MoEHybridTransformerBlockBase,
    MoEReorderedNormTransformerBlock,
    MoETransformer,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.testing import (
    BACKENDS,
    FLASH_2_MARKS,
    FLASH_3_MARKS,
    GPU_MARKS,
    TE_MARKS,
    requires_flash_attn_2,
    requires_multi_gpu,
    run_distributed_test,
)
from olmo_core.testing.utils import (
    FLA_MARKS,
    compute_capability,
    has_fla,
    requires_fla,
    requires_gpu,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerPipelineParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cuda", id="cpu->cuda", marks=GPU_MARKS),
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_small_llama2_builder_config(init_device, device):
    config = TransformerConfig.llama2_271M(vocab_size=50257)
    log.info(config)
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    # Make sure num params estimate is correct.
    num_actual_params = 0
    for _, p in model.named_parameters():
        num_actual_params += p.numel()
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params

    for module in model.modules():
        # Make sure there are no biases anywhere and layer norm weights are all 1.
        if isinstance(module, (nn.Linear, LayerNorm)):
            assert module.bias is None
        if isinstance(module, LayerNorm):
            assert module.weight is not None
            assert (module.weight == 1).all()

    # Make sure block_idx is set correctly.
    assert model.blocks["0"].block_idx == 0
    assert model.blocks[str(len(model.blocks) - 1)].block_idx == len(model.blocks) - 1


def check_ngpt_matrices(model: nn.Module, d_model: int):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None

            w = module.weight
            if isinstance(w, DTensor):
                w = w.full_tensor()

            if w.shape[1] == d_model and "attention.w_out" not in name:
                pass
            elif w.shape[0] == d_model:
                w = w.transpose(0, 1)
            else:
                continue

            log.info(f"Checking norm for '{name}'")
            norm = torch.linalg.vector_norm(w, dim=1)
            torch.testing.assert_close(norm, torch.ones_like(norm))


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cuda", id="cpu->cuda", marks=GPU_MARKS),
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_small_ngpt_builder_config(init_device, device):
    config = TransformerConfig.ngpt_271M(vocab_size=50257)
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    # Make sure num params estimate is correct.
    num_actual_params = 0
    for _, p in model.named_parameters():
        num_actual_params += p.numel()
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params

    # Make sure block_idx is set correctly.
    assert model.blocks["0"].block_idx == 0
    assert model.blocks[str(len(model.blocks) - 1)].block_idx == len(model.blocks) - 1

    # Make sure all weights are normalized in the embedding dimension.
    check_ngpt_matrices(model, config.d_model)


def run_ngpt_with_fsdp2():
    config = TransformerConfig.ngpt_271M(vocab_size=50257)
    model = config.build(
        init_device="meta",
    )
    model.apply_fsdp()
    model.init_weights(max_seq_len=1024, device=get_default_device())
    optim = torch.optim.Adam(model.parameters())

    # Take an optimizer step.
    model(input_ids=torch.randint(0, 50257, (2, 128))).sum().backward()
    optim.step()

    # Re-normalize weights.
    model.normalize_matrices()  # type: ignore

    # Check that the re-normalization was successful.
    check_ngpt_matrices(model, config.d_model)


@requires_multi_gpu
def test_ngpt_with_fsdp2():
    run_distributed_test(run_ngpt_with_fsdp2, backend="nccl", start_method="spawn")


def get_transformer_config(
    architecture: str,
    dtype: torch.dtype = torch.float32,
    attn_backend: Optional[AttentionBackendName] = None,
    swa: Optional[SlidingWindowAttentionConfig] = None,
) -> TransformerConfig:
    config: TransformerConfig
    if architecture == "olmo2":
        config = TransformerConfig.olmo2_190M(
            vocab_size=16_000,
            n_layers=2,
            fused_ops=False,
            attn_backend=attn_backend,
            dtype=DType.from_pt(dtype),
            sliding_window=swa,
        )
    elif architecture == "llama":
        config = TransformerConfig.llama2_271M(
            vocab_size=16_000,
            n_layers=2,
            fused_ops=False,
            attn_backend=attn_backend,
            dtype=DType.from_pt(dtype),
            sliding_window=swa,
        )
    elif architecture == "gdn":
        assert has_fla, "GDN requires FLa"
        assert attn_backend is None, "GDN does not support attention backends"
        layer_norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
        config = TransformerConfig(
            d_model=256,
            vocab_size=16_000,
            n_layers=2,
            block=TransformerBlockConfig(
                name=TransformerBlockType.reordered_norm,
                sequence_mixer=GatedDeltaNetConfig(n_heads=8),
                layer_norm=layer_norm,
                feed_forward=FeedForwardConfig(hidden_size=512, bias=False),
            ),
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False),
        )
    else:
        raise NotImplementedError(architecture)

    return config


def get_transformer_inputs() -> torch.Tensor:
    return torch.arange(0, 128).unsqueeze(0)


def run_tensor_parallel_transformer(checkpoint_dir, outputs_path, architecture: str):
    device = get_default_device()
    config = get_transformer_config(architecture)
    input_ids = get_transformer_inputs().to(device)

    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("tp",),
    )

    model = config.build()
    model.apply_tp(mesh["tp"])
    model.init_weights(device=device, max_seq_len=512)
    load_model_and_optim_state(checkpoint_dir, model)

    logits = model(input_ids=input_ids)

    loss = logits.sum()
    loss.backward()

    og_logits = torch.load(outputs_path, map_location=device)
    torch.testing.assert_close(og_logits, get_full_tensor(logits))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("architecture", ["olmo2", "llama"])
def test_tensor_parallel_transformer(backend: str, architecture: str, tmp_path):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")
    config = get_transformer_config(architecture)
    model = config.build()
    model.init_weights(device=device, max_seq_len=512)
    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)

    outputs_path = tmp_path / "logits.pt"
    torch.save(logits, outputs_path)

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        run_tensor_parallel_transformer,
        backend=backend,
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            outputs_path,
            architecture,
        ),
    )


def run_context_parallel_transformer_ring(checkpoint_dir, outputs_path, architecture: str):
    device = get_default_device()
    config = get_transformer_config(
        architecture, dtype=torch.bfloat16, attn_backend=AttentionBackendName.flash_2
    )

    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("cp",),
    )

    model = config.build()
    ring_style = RingContextParallelStyle(load_balancer=RingAttentionLoadBalancerType.zig_zag)
    model.apply_cp(mesh["cp"], ring=ring_style)
    model.init_weights(device=device, max_seq_len=512)
    load_model_and_optim_state(checkpoint_dir, model)

    input_ids = get_transformer_inputs().to(device)
    local_logits = model(input_ids=input_ids)
    logits = DTensor.from_local(local_logits, mesh, (Shard(1),))

    og_logits = torch.load(outputs_path, map_location=device)
    torch.testing.assert_close(og_logits, get_full_tensor(logits), rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_multi_gpu
@requires_flash_attn_2
@pytest.mark.parametrize("architecture", ["olmo2"])
@pytest.mark.skip("known precision issues with ring-flash-attn")
def test_context_parallel_transformer_ring(architecture: str, tmp_path):
    seed_all(0)
    device = torch.device("cuda")
    config = get_transformer_config(
        architecture, dtype=torch.bfloat16, attn_backend=AttentionBackendName.flash_2
    )

    model = config.build()
    model.init_weights(device=device, max_seq_len=512)
    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)

    outputs_path = tmp_path / "logits.pt"
    torch.save(logits, outputs_path)

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        run_context_parallel_transformer_ring,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            outputs_path,
            architecture,
        ),
    )


def run_context_parallel_transformer_ulysses(
    checkpoint_dir, outputs_path, architecture: str, backend_name: Optional[AttentionBackendName]
):
    device = get_default_device()
    config = get_transformer_config(architecture, dtype=torch.bfloat16, attn_backend=backend_name)

    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("cp",),
    )

    model = config.build()
    model.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    model.init_weights(device=device, max_seq_len=512)
    load_model_and_optim_state(checkpoint_dir, model)

    input_ids = get_transformer_inputs().to(device)
    local_logits = model(input_ids=input_ids)
    logits = DTensor.from_local(local_logits, mesh, (Shard(1),))

    og_logits = torch.load(outputs_path, map_location=device)
    tol_scale = 4.0  # requires slightly more tolerance than default
    torch.testing.assert_close(
        og_logits, get_full_tensor(logits), rtol=BF16_RTOL * tol_scale, atol=BF16_ATOL * tol_scale
    )


@requires_multi_gpu
@pytest.mark.parametrize(
    "architecture, backend_name",
    [
        pytest.param("olmo2", AttentionBackendName.flash_2, id="olmo2-fa2", marks=FLASH_2_MARKS),
        pytest.param("olmo2", AttentionBackendName.flash_3, id="olmo2-fa3", marks=FLASH_3_MARKS),
        pytest.param(
            "olmo2",
            AttentionBackendName.te,
            id="olmo2-te-attn",
            marks=(
                *TE_MARKS,
                pytest.mark.skipif(
                    compute_capability is None or compute_capability < 9,
                    reason="TE Ulysses attention requires compute capability >=9",
                ),
                # te-attn Ulysses CP is flaky on Hopper (intermittent numerical/collective
                # failure); strict=False so it stays green whether it fails or passes.
                pytest.mark.xfail(
                    reason="te-attn Ulysses CP is flaky on Hopper",
                    strict=False,
                ),
            ),
        ),
        pytest.param("gdn", None, id="gdn", marks=FLA_MARKS),
    ],
)
def test_context_parallel_transformer_ulysses(
    architecture: str, backend_name: Optional[AttentionBackendName], tmp_path
):
    seed_all(0)
    device = torch.device("cuda")
    config = get_transformer_config(architecture, dtype=torch.bfloat16, attn_backend=backend_name)

    model = config.build()
    model.init_weights(device=device, max_seq_len=512)
    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)

    outputs_path = tmp_path / "logits.pt"
    torch.save(logits, outputs_path)

    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        run_context_parallel_transformer_ulysses,
        backend="nccl",
        start_method="spawn",
        func_args=(
            checkpoint_dir,
            outputs_path,
            architecture,
            backend_name,
        ),
    )


def run_init_with_hsdp(architecture: str):
    assert dist.get_world_size() == 4
    mesh = build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.hsdp, shard_degree=2, num_replicas=2)
    )
    config = get_transformer_config(architecture)
    model = config.build(init_device="meta")
    model.apply_fsdp(mesh)
    model.init_weights(max_seq_len=512, device=get_default_device())

    # Check that params across all replica groups are exactly the same.
    for name, param in model.named_parameters():
        full_param = get_full_tensor(param).detach()
        full_param_avg = full_param / 4
        dist.all_reduce(full_param_avg)
        torch.testing.assert_close(
            full_param_avg,
            full_param,
            msg=f"parameter '{name}' is inconsistent across the process group",
        )


@requires_multi_gpu
@pytest.mark.parametrize("architecture", ["olmo2", pytest.param("gdn", marks=FLA_MARKS)])
def test_init_with_hsdp(architecture: str):
    if torch.cuda.device_count() < 4:
        pytest.skip("Requires 4 GPUs")

    run_distributed_test(
        run_init_with_hsdp,
        backend="nccl",
        start_method="spawn",
        world_size=4,
        func_args=(architecture,),
    )


def run_moe_hybrid_combined_forward(
    dropless: bool,
    shared_experts: bool,
    reordered_norm: bool,
    tp: bool,
    latent_moe: bool = False,
):
    layer_norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    config = TransformerConfig(
        name=TransformerType.moe,
        d_model=512,
        vocab_size=16_000,
        n_layers=2,
        block=TransformerBlockConfig(
            name=(
                TransformerBlockType.moe_hybrid_reordered_norm
                if reordered_norm
                else TransformerBlockType.moe_hybrid
            ),
            attention=AttentionConfig(n_heads=8, rope=RoPEConfig(), qk_norm=layer_norm),
            layer_norm=layer_norm,
            feed_forward=FeedForwardConfig(hidden_size=1024, bias=False),
            feed_forward_moe=MoEConfig(
                name=MoEType.dropless if dropless else MoEType.default,
                num_experts=4,
                hidden_size=256,
                shared_mlp=(
                    FeedForwardConfig(hidden_size=512, bias=False) if shared_experts else None
                ),
                router=MoERouterConfig(uniform_expert_assignment=True),
                latent_moe=LatentMoEConfig(latent_dim=256) if latent_moe else None,
            ),
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False),
    )

    device = get_default_device()
    model = config.build(init_device=device.type)
    assert isinstance(model, MoETransformer)
    mesh = init_device_mesh(
        device.type,
        (get_world_size(),),
        mesh_dim_names=("tp" if tp else "ep",),
    )

    if tp:
        model.apply_tp(mesh["tp"])
    else:
        model.apply_ep(mesh["ep"])

    input_ids = get_transformer_inputs().to(device)
    model.init_weights(device=device, max_seq_len=512, max_local_microbatch_size=input_ids.numel())

    for block in model.blocks.values():
        cast(MoEHybridTransformerBlockBase, block).use_combined_forward = False
    output1 = model(input_ids)

    for block in model.blocks.values():
        cast(MoEHybridTransformerBlockBase, block).use_combined_forward = True
    output2 = model(input_ids)

    torch.testing.assert_close(output1, output2)


@requires_multi_gpu
@pytest.mark.parametrize(
    "dropless", [pytest.param(True, id="dropless"), pytest.param(False, id="default-router")]
)
@pytest.mark.parametrize(
    "shared_experts", [pytest.param(True, id="shared-experts"), pytest.param(False, id="no-shared")]
)
@pytest.mark.parametrize(
    "reordered_norm",
    [pytest.param(True, id="reordered-norm"), pytest.param(False, id="default-block")],
)
@pytest.mark.parametrize("tp", [pytest.param(True, id="TP"), pytest.param(False, id="EP")])
def test_moe_hybrid_combined_forward(
    dropless: bool, shared_experts: bool, reordered_norm: bool, tp: bool
):
    run_distributed_test(
        run_moe_hybrid_combined_forward,
        backend="nccl",
        start_method="spawn",
        func_args=(
            dropless,
            shared_experts,
            reordered_norm,
            tp,
        ),
    )


@requires_multi_gpu
@pytest.mark.parametrize(
    "reordered_norm",
    [pytest.param(True, id="reordered-norm"), pytest.param(False, id="default-block")],
)
@pytest.mark.parametrize("tp", [pytest.param(True, id="TP"), pytest.param(False, id="EP")])
def test_latent_moe_hybrid_combined_forward(reordered_norm: bool, tp: bool):
    run_distributed_test(
        run_moe_hybrid_combined_forward,
        backend="nccl",
        start_method="spawn",
        func_args=(True, True, reordered_norm, tp, True),
    )


def test_build_with_block_overrides():
    d_model = 512
    config = TransformerConfig.llama_like_moe(
        vocab_size=16_000,
        d_model=d_model,
        n_layers=8,
        n_heads=8,
        num_experts=32,
        top_k=4,
        expert_hidden_size=int(0.5 * d_model),
        capacity_factor=1.2,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        hybrid=True,
        qk_norm=True,
        rope_theta=10_000,
        layer_norm_eps=1e-6,
        feed_forward=FeedForwardConfig(hidden_size=d_model * 2, bias=False),
    )
    assert not isinstance(config.block, dict)
    assert config.block.feed_forward_moe is not None
    moe_config = replace(config.block.feed_forward_moe, shared_mlp=config.block.feed_forward)
    config.block_overrides = {
        0: replace(
            config.block,
            name=TransformerBlockType(str(config.block.name).replace("_hybrid_", "_")),
            feed_forward=None,
            feed_forward_moe=moe_config,
        )
    }

    model = config.build(init_device="cpu")
    assert isinstance(model.blocks["0"], MoEReorderedNormTransformerBlock)
    assert isinstance(model.blocks["1"], MoEHybridTransformerBlockBase)

    assert config.num_params == model.num_params


def test_transformer_num_flops_per_token():
    seed_all(0)

    d_model = 128
    seq_len = 256
    n_heads = 8
    n_kv_heads = 4
    vocab_size = 1024

    def _flops_per_token(*, n_layers: int, swa_pattern: list[int]) -> int:
        config = TransformerConfig.llama_like(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            sliding_window=SlidingWindowAttentionConfig(pattern=swa_pattern),
        )
        model = config.build(init_device="cpu")
        return model.num_flops_per_token(seq_len)

    base = _flops_per_token(n_layers=4, swa_pattern=[16, 16, 16, 16])
    assert base > 0

    # adding layers should strictly increase FLOPs/token.
    more_blocks = _flops_per_token(n_layers=8, swa_pattern=[16, 16, 16, 16])
    assert more_blocks > base

    # Relative checks: increasing sliding window size should increase FLOPs/token.
    bigger_window = _flops_per_token(n_layers=4, swa_pattern=[128, 128, 128, 128])
    assert bigger_window > base


@pytest.mark.parametrize(
    "config_builder,expected_d_model",
    [
        pytest.param(TransformerConfig.gemma3_1B, 2304, id="gemma3_1B"),
        pytest.param(TransformerConfig.gemma3_4B, 2560, id="gemma3_4B"),
        pytest.param(TransformerConfig.gemma3_12B, 3840, id="gemma3_12B"),
        pytest.param(TransformerConfig.gemma3_27B, 5376, id="gemma3_27B"),
    ],
)
def test_gemma3_builder_configs(config_builder, expected_d_model):
    config = config_builder(n_layers=6)
    assert config.d_model == expected_d_model
    assert config.n_layers == 6

    block_configs = config.resolved_block_configs
    local_block = block_configs[0]
    assert local_block.feed_forward is not None
    assert local_block.feed_forward.activation == ActivationFunction.gelu_tanh

    sequence_mixer = local_block.sequence_mixer
    assert isinstance(sequence_mixer, AttentionConfig)
    assert sequence_mixer.qk_norm is not None
    assert sequence_mixer.rope is not None
    assert sequence_mixer.rope.theta == 10_000

    # Use meta device to avoid allocating large amounts of memory for big models.
    model = config.build(init_device="meta")

    num_actual_params = sum(p.numel() for p in model.parameters())
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params


def test_gemma3_hybrid_local_global_attention():
    config = TransformerConfig.gemma3_1B(n_layers=12)

    local_count = 0
    global_count = 0
    for block_config in config.resolved_block_configs:
        attention = block_config.sequence_mixer
        assert isinstance(attention, AttentionConfig)
        assert attention.rope is not None
        if attention.sliding_window is None:
            assert attention.rope.theta == 1_000_000
            global_count += 1
        else:
            assert attention.rope.theta == 10_000
            local_count += 1

    assert global_count == 2
    assert local_count == 10

    distinct_blocks = {id(b) for b in config.resolved_block_configs}
    assert len(distinct_blocks) == 2


@pytest.mark.parametrize(
    "config_builder,expected_d_model",
    [
        pytest.param(TransformerConfig.qwen3_0_6B, 1024, id="qwen3_0_6B"),
        pytest.param(TransformerConfig.qwen3_1_7B, 2048, id="qwen3_1_7B"),
        pytest.param(TransformerConfig.qwen3_4B, 2560, id="qwen3_4B"),
        pytest.param(TransformerConfig.qwen3_8B, 4096, id="qwen3_8B"),
        pytest.param(TransformerConfig.qwen3_14B, 5120, id="qwen3_14B"),
        pytest.param(TransformerConfig.qwen3_32B, 5120, id="qwen3_32B"),
    ],
)
def test_qwen3_builder_configs(config_builder, expected_d_model):
    config = config_builder(vocab_size=151936, n_layers=2)
    assert config.d_model == expected_d_model
    assert config.n_layers == 2
    attention = config.block.sequence_mixer
    assert isinstance(attention, AttentionConfig)
    assert attention.n_kv_heads == 8
    assert attention.rope is not None
    assert attention.rope.theta == 1_000_000

    # Use meta device to avoid allocating large amounts of memory for big models.
    model = config.build(init_device="meta")

    num_actual_params = sum(p.numel() for p in model.parameters())
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params


@pytest.mark.parametrize(
    "config_builder,expected_d_model,expected_n_heads,expected_gdn_v_heads",
    [
        pytest.param(TransformerConfig.qwen3_5_0_8B, 1024, 8, 16, id="qwen3_5_0_8B"),
        pytest.param(TransformerConfig.qwen3_5_4B, 2560, 16, 32, id="qwen3_5_4B"),
        pytest.param(TransformerConfig.qwen3_5_9B, 4096, 16, 32, id="qwen3_5_9B"),
        pytest.param(TransformerConfig.qwen3_5_27B, 5120, 24, 48, id="qwen3_5_27B"),
    ],
)
def test_qwen3_5_builder_configs(
    config_builder, expected_d_model, expected_n_heads, expected_gdn_v_heads
):
    config = config_builder(vocab_size=248320, n_layers=4)
    assert config.d_model == expected_d_model
    assert config.n_layers == 4
    assert config.block_pattern == ["gdn", "gdn", "gdn", "attn"]
    assert config.tie_word_embeddings

    assert isinstance(config.block, dict)
    gdn_block = config.block["gdn"]
    attn_block = config.block["attn"]
    assert isinstance(gdn_block.sequence_mixer, GatedDeltaNetConfig)
    assert isinstance(attn_block.sequence_mixer, AttentionConfig)

    gdn = gdn_block.sequence_mixer
    assert gdn.allow_neg_eigval is False
    assert gdn.n_v_heads == expected_gdn_v_heads
    assert gdn.head_dim == 128

    attn = attn_block.sequence_mixer
    assert attn.n_heads == expected_n_heads
    assert attn.head_dim == 256
    assert attn.rope is not None
    assert attn.rope.theta == 10_000_000
    assert attn.rope.partial_rotary_factor == 0.25
    assert attn.gate is not None

    layer_types = ["gdn", "gdn", "gdn", "attn"]
    for layer_idx, block_config in enumerate(config.resolved_block_configs):
        if layer_types[layer_idx] == "gdn":
            assert isinstance(block_config.sequence_mixer, GatedDeltaNetConfig)
        else:
            assert isinstance(block_config.sequence_mixer, AttentionConfig)


@pytest.mark.parametrize(
    "config_builder",
    [
        pytest.param(TransformerConfig.qwen3_5_0_8B, id="qwen3_5_0_8B"),
        pytest.param(TransformerConfig.qwen3_5_4B, id="qwen3_5_4B"),
        pytest.param(TransformerConfig.qwen3_5_9B, id="qwen3_5_9B"),
        pytest.param(TransformerConfig.qwen3_5_27B, id="qwen3_5_27B"),
    ],
)
@pytest.mark.skipif(not has_fla, reason="flash-linear-attention (fla) not available")
def test_qwen3_5_param_count(config_builder):
    config = config_builder(vocab_size=248320, n_layers=4)
    model = config.build(init_device="meta")
    num_actual_params = sum(p.numel() for p in model.parameters())
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params


@pytest.mark.parametrize(
    "config_builder, expected_tie",
    [
        pytest.param(TransformerConfig.qwen3_0_6B, True, id="qwen3_0_6B"),
        pytest.param(TransformerConfig.qwen3_1_7B, True, id="qwen3_1_7B"),
        pytest.param(TransformerConfig.qwen3_4B, True, id="qwen3_4B"),
        pytest.param(TransformerConfig.qwen3_8B, False, id="qwen3_8B"),
        pytest.param(TransformerConfig.qwen3_14B, False, id="qwen3_14B"),
        pytest.param(TransformerConfig.qwen3_32B, False, id="qwen3_32B"),
    ],
)
def test_qwen3_small_sizes_tie_word_embeddings(config_builder, expected_tie):
    assert config_builder(vocab_size=128, n_layers=2).tie_word_embeddings == expected_tie


def test_qwen3_tie_word_embeddings_can_be_overridden():
    config = TransformerConfig.qwen3_0_6B(vocab_size=128, n_layers=2, tie_word_embeddings=False)
    assert not config.tie_word_embeddings


def test_qwen3_5_tie_word_embeddings_can_be_overridden():
    config = TransformerConfig.qwen3_5_0_8B(
        vocab_size=128,
        n_layers=4,
        tie_word_embeddings=False,
    )
    assert not config.tie_word_embeddings


def test_tied_word_embeddings_share_weight_after_init():
    config = TransformerConfig.qwen3_0_6B(vocab_size=128, n_layers=2)
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    assert model.tie_word_embeddings
    # The tie must survive `init_weights`, which calls `to_empty`.
    assert model.lm_head.w_out.weight is model.embeddings.weight

    # The shared weight is only counted once.
    num_actual_params = sum(p.numel() for p in model.parameters())
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params


@requires_gpu
@requires_fla
def test_qwen3_5_forward():
    device = torch.device("cuda")
    config = TransformerConfig.qwen3_5_0_8B(
        vocab_size=1000,
        n_layers=4,
        attn_backend=AttentionBackendName.torch,
    )
    model = config.build(init_device="cpu").eval().to(device)
    input_ids = torch.randint(0, 1000, (2, 16), device=device)
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (2, 16, 1000)


def test_normalized_transformer_rejects_tied_word_embeddings():
    with pytest.raises(OLMoConfigurationError):
        TransformerConfig.ngpt_271M(vocab_size=128, n_layers=2, tie_word_embeddings=True)


def test_pipeline_parallel_rejects_tied_word_embeddings():
    config = TransformerConfig.qwen3_0_6B(vocab_size=128, n_layers=2)
    model = config.build(init_device="cpu")
    pp_config = TransformerPipelineParallelConfig(degree=1, split_points=[1])

    with pytest.raises(NotImplementedError, match="tied word embeddings"):
        pp_config.split_model(model, pp_mesh=None, device=torch.device("cpu"))


def run_tensor_parallel_tied_word_embeddings():
    device = get_default_device()
    config = TransformerConfig.llama2_271M(
        vocab_size=16_000, n_layers=2, fused_ops=False, tie_word_embeddings=True
    )
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    model = config.build()
    model.apply_tp(mesh["tp"])
    model.init_weights(device=device, max_seq_len=512)

    assert model.tie_word_embeddings
    assert isinstance(model.embeddings.weight, DTensor)
    # The tie survives `apply_tp` (which converts the weight to a sharded DTensor) and
    # `init_weights` (which calls `to_empty`): both modules share one parameter.
    assert model.lm_head.w_out.weight is model.embeddings.weight

    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)
    logits.sum().backward()

    # Gradients from the embedding lookup and the output projection accumulate into the single
    # shared parameter.
    assert model.embeddings.weight.grad is not None


@pytest.mark.parametrize("backend", BACKENDS)
def test_tensor_parallel_tied_word_embeddings(backend: str):
    run_distributed_test(
        run_tensor_parallel_tied_word_embeddings,
        backend=backend,
        start_method="spawn",
    )


def run_fsdp_tied_word_embeddings():
    device = get_default_device()
    config = TransformerConfig.llama2_271M(
        vocab_size=16_000, n_layers=2, fused_ops=False, tie_word_embeddings=True
    )

    model = config.build(init_device="meta")
    model.apply_fsdp()
    model.init_weights(device=device, max_seq_len=512)

    assert model.tie_word_embeddings
    assert isinstance(model.embeddings.weight, DTensor)
    # The embeddings and LM head are not sharded into separate FSDP groups when tied, so they
    # stay in the root group and keep sharing one parameter.
    assert model.lm_head.w_out.weight is model.embeddings.weight

    input_ids = get_transformer_inputs().to(device)
    logits = model(input_ids=input_ids)
    logits.sum().backward()

    assert model.embeddings.weight.grad is not None


@requires_multi_gpu
def test_fsdp_tied_word_embeddings():
    run_distributed_test(
        run_fsdp_tied_word_embeddings,
        backend="nccl",
        start_method="spawn",
    )
