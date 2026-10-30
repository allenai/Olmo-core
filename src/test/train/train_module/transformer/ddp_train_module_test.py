"""Tests for :class:`OLMoDDPTrainModule` config and construction."""

from typing import Optional

import pytest
import torch
import torch.distributed.checkpoint as dcp

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.train.train_module import OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerPipelineParallelConfig,
)


def test_moe_v2_train_module_config_roundtrips():
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
    )
    restored = OLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.optim.lr == 1e-3


def test_moe_v2_train_module_config_roundtrips_with_parallelism():
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, reduce_grads_in_fp32=False
        ),
        pp_config=TransformerPipelineParallelConfig(degree=2),
    )
    restored = OLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.dp_config is not None and restored.dp_config.reduce_grads_in_fp32 is False
    assert restored.pp_config is not None and restored.pp_config.degree == 2


def _tiny_model_config(
    *,
    d_model: int = 64,
    n_layers: int = 2,
    dtype: DType = DType.float32,
    router_bias_gamma: Optional[float] = None,
    per_head_qk: Optional[bool] = None,
) -> OLMoDDPModelConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    return OLMoDDPModelConfig(
        init_seed=0,
        d_model=d_model,
        recompute_each_block=False,
        vocab_size=128,
        n_layers=n_layers,
        name=TransformerType.moe_fused_v2,
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=4,
                bias=False,
                backend=AttentionBackendName.torch,
                dtype=dtype,
                n_kv_heads=2 if per_head_qk is not None else None,
                qk_norm=layer_norm if per_head_qk is not None else None,
                use_head_qk_norm=per_head_qk is not None,
                qk_norm_per_head_gains=bool(per_head_qk),
            ),
            routed_experts=RoutedExpertsConfig(
                d_model=d_model, hidden_size=128, num_experts=4, bias=False, dtype=dtype
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model, num_experts=4, top_k=2, dtype=dtype, bias_gamma=router_bias_gamma
            ),
            shared_experts=None,
            layer_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
    )


def _run_construct_no_ep():
    model = _tiny_model_config().build(init_device="cpu")
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
    )
    legacy_config = config.as_config_dict()
    legacy_config["max_grad_norm"] = 0.25
    config = OLMoDDPTrainModuleConfig.from_dict(legacy_config)
    # eval_only=True skips the optimizer build (its fp32-master-param setup is exercised on GPU);
    # this covers the world-mesh build + data-parallel wrapping with no expert parallelism.
    train_module = config.build(model, device=torch.device("cpu"), eval_only=True)

    assert len(train_module.model_parts) == 1  # no pipeline parallelism
    assert train_module.dp_world_size == 2
    assert train_module.world_mesh["dense"] is not None
    assert train_module.moe_mesh is None  # no expert parallelism
    assert not hasattr(train_module, "max_grad_norm")


def test_moe_v2_train_module_construction_no_ep():
    run_distributed_test(
        _run_construct_no_ep,
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )


def _run_load_shared_qk_checkpoint(path, key_format, eval_only=True, constant_memory=False):
    device = "cpu" if eval_only else "cuda"
    dtype = DType.float32 if eval_only else DType.bfloat16
    source = _tiny_model_config(dtype=dtype, per_head_qk=False).build(init_device=device)
    source.init_weights(max_seq_len=32)
    with torch.no_grad():
        for name, param in source.named_parameters():
            if name.endswith((".q_norm.weight", ".k_norm.weight")):
                param.copy_(torch.arange(param.numel(), device=device).reshape(param.shape) + 1)
    saved = {
        (f"model.{name}" if key_format == "model" else f"{name}.main"): (
            param.detach() if key_format == "model" else param.detach().flatten()
        )
        for name, param in source.named_parameters()
    }
    dcp.save(saved, checkpoint_id=path)
    target = _tiny_model_config(dtype=dtype, per_head_qk=True).build(init_device=device)
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        expand_shared_qk_norm_on_load=True,
    )
    tm = config.build(target, device=torch.device(device), eval_only=eval_only)
    # Without the opt-in, a shared/per-head shape mismatch must remain an error.
    tm.expand_shared_qk_norm_on_load = False
    with pytest.raises(RuntimeError, match="shape mismatch"):
        tm.load_state_dict_direct(
            path, load_optim_state=False, constant_memory_planning=constant_memory
        )
    tm.expand_shared_qk_norm_on_load = True
    tm.load_state_dict_direct(
        path, load_optim_state=False, constant_memory_planning=constant_memory
    )
    expected = dict(source.named_parameters())
    for part in tm.model_parts:
        for name, param in part.named_parameters():
            original = expected[tm._strip_wrapper_prefixes(name)]
            if original.shape != param.shape:
                original = original.expand_as(param)
            torch.testing.assert_close(param, original, rtol=0, atol=0)
    if not eval_only:
        assert tm.optim is not None
        state = tm.optim.state_dict()
        for group in tm.optim.param_groups:
            for name, param in group["named_params"].items():
                main = state[f"{name}.main"]
                torch.testing.assert_close(
                    main.full_tensor(), param.float().flatten(), rtol=0, atol=0
                )


@pytest.mark.parametrize("key_format", ["model", "main"])
@pytest.mark.parametrize("constant_memory", [False, True])
def test_eval_load_expands_shared_qk_gains(tmp_path, key_format, constant_memory):
    run_distributed_test(
        _run_load_shared_qk_checkpoint,
        func_args=(str(tmp_path / "checkpoint"), key_format, True, constant_memory),
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )


@requires_multi_gpu
def test_model_only_training_load_expands_shared_qk_gains(tmp_path):
    run_distributed_test(
        _run_load_shared_qk_checkpoint,
        func_args=(str(tmp_path / "checkpoint"), "model", False),
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )


def _run_construct_ep():
    # bf16 params → the fused optimizer maintains fp32 master params (its realistic config); a pure
    # fp32 model instead takes the optimizer's "expect fp32 param" branch.
    model = _tiny_model_config(dtype=DType.bfloat16).build(init_device="cuda")
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        ep_config=TransformerExpertParallelConfig(degree=2),
    )
    # Full build (eval_only=False): wires expert parallelism through the train module (moe mesh +
    # apply_ep sharding the experts across the two ranks + DP wrapping) and builds the optimizer.
    train_module = config.build(model, device=torch.device("cuda"), eval_only=False)

    assert len(train_module.model_parts) == 1  # no pipeline parallelism
    assert train_module.moe_mesh is not None
    assert train_module.ep_mp_group is not None
    assert train_module.optim is not None
    assert train_module.num_flops_per_token(seq_len=512) > 0


@requires_multi_gpu
def test_moe_v2_train_module_construction_ep():
    run_distributed_test(
        _run_construct_ep,
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )


def test_moe_v2_train_module_config_reset_optimizer_states_roundtrips():
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        reset_optimizer_states_on_resume=True,
    )
    restored = OLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.reset_optimizer_states_on_resume is True
    # The resume flag is distinct from the generic on-load flag, which stays at its default.
    assert restored.reset_optimizer_states_on_load is False


def _run_rejects_per_microbatch_allreduce():
    model = _tiny_model_config().build(init_device="cpu")
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp, only_allreduce_last_microbatch=False
        ),
    )
    # MultiGroupDistributedDataParallel reduces each bucket once per accumulation window, so
    # per-micro-batch all-reduce is unsupported and must be rejected up front.
    with pytest.raises(OLMoConfigurationError, match="only_allreduce_last_microbatch"):
        config.build(model, device=torch.device("cpu"), eval_only=True)


def test_moe_v2_train_module_rejects_per_microbatch_allreduce():
    run_distributed_test(
        _run_rejects_per_microbatch_allreduce,
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )


_MOMENT_SUFFIXES = (".exp_avg", ".exp_avg_sq")


def _build_ddp_train_module_for_checkpoint(*, router_bias_gamma: Optional[float] = None):
    model = _tiny_model_config(dtype=DType.bfloat16, router_bias_gamma=router_bias_gamma).build(
        init_device="cuda"
    )
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
    )
    return config.build(model, device=torch.device("cuda"), eval_only=False)


def _run_resume_resets_optimizer_moments(save_dir):
    # Save a checkpoint carrying non-zero optimizer moments, then verify that the resume flag
    # (threaded through as reset_optimizer_states_on_load) actually controls whether those moments
    # are restored or discarded on load.
    tm = _build_ddp_train_module_for_checkpoint()
    assert tm.optim is not None
    for key, state in tm.optim.states.items():
        if key.endswith(_MOMENT_SUFFIXES):
            state.to_local().fill_(0.5)
    tm.save_state_dict_direct(save_dir)

    # Reset on load: only the main params are restored, so freshly zero-initialized moments stay zero.
    tm_reset = _build_ddp_train_module_for_checkpoint()
    assert tm_reset.optim is not None
    tm_reset.load_state_dict_direct(save_dir, reset_optimizer_states_on_load=True)
    for key, state in tm_reset.optim.states.items():
        if key.endswith(_MOMENT_SUFFIXES):
            assert torch.count_nonzero(state.to_local()) == 0, key

    # No reset: the saved (non-zero) moments are restored.
    tm_restore = _build_ddp_train_module_for_checkpoint()
    assert tm_restore.optim is not None
    tm_restore.load_state_dict_direct(save_dir, reset_optimizer_states_on_load=False)
    restored_any_moment = any(
        key.endswith(_MOMENT_SUFFIXES) and torch.count_nonzero(state.to_local()) > 0
        for key, state in tm_restore.optim.states.items()
    )
    assert restored_any_moment


@requires_multi_gpu
def test_moe_v2_train_module_resume_resets_optimizer_moments(tmp_path):
    run_distributed_test(
        _run_resume_resets_optimizer_moments,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(str(tmp_path / "checkpoint"),),
    )


def _score_bias_buffers(train_module):
    return {
        name: buf
        for model_part in train_module.model_parts
        for name, buf in model_part.named_buffers()
        if name.endswith("score_bias") and buf is not None
    }


def _run_direct_checkpoint_restores_buffers(save_dir):
    # Persistent buffers (the router's aux-loss-free score_bias) are model state updated outside
    # the optimizer; the direct checkpoint must round-trip them.
    tm = _build_ddp_train_module_for_checkpoint(router_bias_gamma=1e-3)
    mutated = _score_bias_buffers(tm)
    assert mutated, "expected at least one score_bias buffer with bias_gamma set"
    saved = {}
    for name, buf in mutated.items():
        buf.copy_(torch.arange(buf.numel(), device=buf.device, dtype=buf.dtype) + 1.0)
        saved[name] = buf.detach().clone()
    tm.save_state_dict_direct(save_dir)

    tm_restored = _build_ddp_train_module_for_checkpoint(router_bias_gamma=1e-3)
    restored = _score_bias_buffers(tm_restored)
    # Freshly built buffers are zero-initialized, so they must differ before the load.
    for name, buf in restored.items():
        assert torch.count_nonzero(buf) == 0, name
    tm_restored.load_state_dict_direct(save_dir, reset_optimizer_states_on_load=False)
    for name, expected in saved.items():
        torch.testing.assert_close(restored[name], expected)


@requires_multi_gpu
def test_moe_v2_train_module_direct_checkpoint_restores_buffers(tmp_path):
    run_distributed_test(
        _run_direct_checkpoint_restores_buffers,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(str(tmp_path / "checkpoint"),),
    )


@pytest.mark.parametrize("module_norm", [None, 1.0, 0.25, 0.0])
@pytest.mark.parametrize("optimizer_norm", [None, 1.0, 2.0])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_max_grad_norm_config_precedence(module_norm, optimizer_norm, with_metadata):
    """Loaded configs forward module clipping overrides without mutating the optimizer config."""
    import json
    from unittest.mock import Mock, patch

    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
    )
    serialized = config.as_config_dict() if with_metadata else config.as_dict()
    if module_norm is not None:
        serialized["max_grad_norm"] = module_norm
    if optimizer_norm is None:
        serialized["optim"].pop("max_grad_norm")
    else:
        serialized["optim"]["max_grad_norm"] = optimizer_norm
    restored = OLMoDDPTrainModuleConfig.from_dict(json.loads(json.dumps(serialized)))
    original_optimizer_norm = 1.0 if optimizer_norm is None else optimizer_norm
    assert restored.max_grad_norm == module_norm
    assert restored.optim.max_grad_norm == original_optimizer_norm
    assert OLMoDDPTrainModuleConfig.from_dict(restored.as_config_dict()) == restored

    with patch(
        "olmo_core.train.train_module.transformer.ddp_train_module.OLMoDDPTrainModule"
    ) as train_module_cls:
        restored.build(Mock())
    kwargs = train_module_cls.call_args.kwargs
    assert "max_grad_norm" not in kwargs
    assert kwargs["optim"].max_grad_norm == (
        original_optimizer_norm if module_norm is None else module_norm
    )
    assert restored.optim.max_grad_norm == original_optimizer_norm
    if module_norm is not None:
        assert kwargs["optim"] is not restored.optim
