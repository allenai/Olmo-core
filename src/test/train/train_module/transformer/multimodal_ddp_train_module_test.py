"""Multimodal OLMoDDP training, metrics, and native checkpoint contracts."""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import RemoteFileSystemReader
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.loss import MoELoadBalancingLossGranularity
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.nn.vision import (
    MultimodalLMConfig,
    MultimodalOLMoDDPModel,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.optim.multimodal_optimizer import (
    MultimodalOLMoDDPOptimizer,
    MultimodalOLMoDDPOptimizerConfig,
)
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.train import ReduceType
from olmo_core.train.train_module import OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModule,
    MultimodalOLMoDDPTrainModuleConfig,
)


class _MetricTrainerStub:
    def __init__(self):
        self.global_step = 1
        self.metrics = {}

    def record_metric(self, name, value, *, namespace=None, **kwargs):
        del kwargs
        self.metrics[f"{namespace}/{name}" if namespace else name] = value


@pytest.mark.parametrize("frozen", [False, True])
def test_vision_model_load_synchronizes_only_trainable_optimizer_masters(frozen):
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    vision = torch.nn.Linear(2, 2)
    vision.requires_grad_(not frozen)
    module.model_parts = [SimpleNamespace(vision=vision)]
    calls = []
    module.optim = object.__new__(MultimodalOLMoDDPOptimizer)
    module.optim.param_groups = [
        {"named_params": {f"vision.{name}": param for name, param in vision.named_parameters()}}
    ]
    module.optim._copy_model_params_to_main_params = lambda names: calls.append(("copy", names))
    module.optim._check_model_param_main_param_the_same = lambda names: calls.append(
        ("check", names)
    )
    state = {name: torch.full_like(param, 0.5) for name, param in vision.named_parameters()}

    module.load_vision_state_dict(state)

    for name, param in vision.named_parameters():
        torch.testing.assert_close(param, state[name], rtol=0, atol=0)
    names = {"vision.weight", "vision.bias"}
    assert calls == ([] if frozen else [("copy", names), ("check", names)])
    module.assert_vision_optimizer_state_synced()
    assert calls[-1] == ("check", set() if frozen else names)


def _tiny_model_config(
    *,
    d_model: int = 64,
    n_layers: int = 2,
    dtype: DType = DType.float32,
    router_bias_gamma: Optional[float] = None,
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
                use_flash=False,
                dtype=dtype,
            ),
            routed_experts=RoutedExpertsConfig(
                d_model=d_model, hidden_size=128, num_experts=4, bias=False, dtype=dtype
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model,
                num_experts=4,
                top_k=2,
                dtype=dtype,
                bias_gamma=router_bias_gamma,
            ),
            shared_experts=None,
            layer_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
    )


def _tiny_multimodal_model_config(*, dtype: DType = DType.float32) -> MultimodalLMConfig:
    lm = _tiny_model_config(dtype=dtype)
    vision = VisionEncoderConfig(
        name=VisionEncoderType.siglip,
        use_cls_token=False,
        patch_embedding_bias=True,
        use_pre_ln=False,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=16,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=1,
        image_head_dim=8,
        image_mlp_dim=32,
        image_num_pos=4,
        dtype=dtype,
    )
    return MultimodalLMConfig(
        lm=lm,
        vision=vision,
        connector=VisionConnectorConfig.from_vision_encoder(
            vision, output_dim=lm.d_model, mlp_hidden_size=32
        ),
        image_patch_token_id=120,
    )


def test_multimodal_olmo_ddp_model_materializes_all_components():
    model = _tiny_multimodal_model_config().build(init_device="meta")
    assert isinstance(model, MultimodalOLMoDDPModel)

    model.init_weights(
        max_seq_len=16,
        max_local_microbatch_size=16,
        device=torch.device("cpu"),
        world_mesh={},
    )
    assert all(not param.is_meta for param in model.parameters())

    for param in model.vision.parameters():
        param.requires_grad_(False)
    model.train()
    assert model.lm.training
    assert model.connector.training
    assert not model.vision.training


def test_multimodal_olmo_ddp_config_and_native_checkpoint_aliases():
    config = MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=16,
        max_sequence_length=16,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        freeze_params=["vision.*"],
        vision_activation_checkpointing=True,
        connector_activation_checkpointing=True,
        response_logits_only=True,
        train_embedding_rows=[120, 121],
    )
    restored = MultimodalOLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.vision_activation_checkpointing
    assert restored.connector_activation_checkpointing
    assert restored.response_logits_only
    assert restored.train_embedding_rows == [120, 121]

    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    checkpoint_keys = {
        "module.embeddings.weight.main",
        "module.blocks.0.attention.w_qkv.weight.main",
    }
    assert (
        train_module._resolve_optimizer_checkpoint_key(
            "module.lm.embeddings.weight.main", checkpoint_keys
        )
        == "module.embeddings.weight.main"
    )
    assert (
        train_module._resolve_optimizer_checkpoint_key(
            "module.lm.blocks.0.attention.w_qkv.weight.main", checkpoint_keys
        )
        == "module.blocks.0.attention.w_qkv.weight.main"
    )
    assert train_module._allow_missing_optimizer_checkpoint_key(
        "module.connector.projector.w1.weight.main"
    )
    assert train_module._allow_missing_optimizer_checkpoint_key(
        "module.vision.patch_embedding.weight.main"
    )
    assert not train_module._allow_missing_optimizer_checkpoint_key(
        "module.lm.embeddings.weight.main"
    )


def test_multimodal_checkpoint_frozen_params_can_become_trainable():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    model = torch.nn.Module()
    model.vision = torch.nn.Linear(2, 2, bias=False)
    object.__setattr__(train_module, "model_parts", [model])

    checkpoint_keys = {"lm.weight.main", "frozen_model.vision.weight"}
    frozen_params = train_module._frozen_checkpoint_param_state_dict_for_load(checkpoint_keys)
    assert model.vision.weight.requires_grad
    assert set(frozen_params) == {"frozen_model.vision.weight"}
    assert frozen_params["frozen_model.vision.weight"] is model.vision.weight

    model.vision.weight.requires_grad_(False)
    native_checkpoint_keys = {"lm.weight.main", "module.vision.weight.main"}
    native_frozen_params = train_module._frozen_checkpoint_param_state_dict_for_load(
        native_checkpoint_keys
    )
    assert set(native_frozen_params) == {"module.vision.weight.main"}
    assert native_frozen_params["module.vision.weight.main"].shape == (4,)
    assert (
        native_frozen_params["module.vision.weight.main"].data_ptr()
        == model.vision.weight.data_ptr()
    )
    model.vision.weight.requires_grad_(True)

    # A trainable parameter that the checkpoint stored as frozen keeps its fresh master until
    # the in-place load has restored the weights, then the master is resynchronized from them.
    train_module._persistent_model_buffer_state_dict = lambda: {}
    train_module._load_key_map = {}
    train_module._load_deferred = {}
    train_module._load_in_place_keys = set()
    train_module._load_resync_names = set()
    train_module._load_expansions = {}
    train_module._load_metadata = None
    train_module.expand_shared_qk_norm_on_load = False
    fresh_master = torch.zeros(4)
    checkpoint_state = train_module._optimizer_state_dict_for_load(
        {"lm.weight.main": torch.zeros(1), "vision.weight.main": fresh_master},
        checkpoint_keys,
    )
    assert set(checkpoint_state) == {"lm.weight.main", "frozen_model.vision.weight"}
    assert checkpoint_state["frozen_model.vision.weight"] is model.vision.weight
    assert train_module._load_deferred == {"vision.weight.main": fresh_master}
    assert train_module._load_resync_names == {"vision.weight"}


def test_multimodal_loss_divisor_uses_float_weights():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.label_ignore_index = -100
    train_module.device = torch.device("cpu")
    loss_masks = torch.tensor([[0.25, 1.0, 0.0, 0.5]])
    kwargs = train_module._batch_auxiliary_loss_kwargs(
        {"input_ids": torch.zeros((1, 4), dtype=torch.long), "loss_masks": loss_masks}
    )
    torch.testing.assert_close(kwargs["loss_weight_div_factor"], torch.tensor(1.75))
    # Every token counts for the router divisor when no token mask is given.
    torch.testing.assert_close(kwargs["router_loss_div_factor"], torch.tensor(4))


def test_multimodal_source_metrics_report_realized_loss_mass():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.source_loss_mass_targets = {"caption": 0.75, "native": 0.25}
    recorded = {}

    def record_metric(name, value, reduce_type=None, namespace=None, **kwargs):
        del kwargs
        recorded[f"{namespace}/{name}"] = (value, reduce_type)

    train_module.record_metric = record_metric
    train_module._record_data_metrics(
        {
            "router_token_mask": torch.tensor([[True, True, True, True, True, False]]),
            "loss_masks": torch.tensor([[1.0, 1.0, 1.0, 0.5, 0.5, 0.0]]),
            "labels": torch.tensor([[10, 11, 12, 13, -100, -100]]),
            "token_type_ids": torch.zeros((1, 6), dtype=torch.long),
            "example_ids": torch.tensor([[0, 0, 0, 1, 1, -1]]),
            "pack_source_names": [["caption", "native"]],
        }
    )

    torch.testing.assert_close(recorded["data/source/caption/examples"][0], torch.tensor(1.0))
    torch.testing.assert_close(recorded["data/source/caption/tokens"][0], torch.tensor(3.0))
    torch.testing.assert_close(recorded["data/source/caption/loss_weight"][0], torch.tensor(3.0))
    torch.testing.assert_close(
        recorded["data/source/native/active_loss_weight"][0], torch.tensor(0.5)
    )
    torch.testing.assert_close(recorded["data/source/native/positive_tokens"][0], torch.tensor(1.0))
    torch.testing.assert_close(
        recorded["data/source/caption/loss_mass_share"][0], torch.tensor(0.75)
    )
    torch.testing.assert_close(
        recorded["data/source/native/loss_mass_target_abs_error"][0], torch.tensor(0.0)
    )
    assert recorded["data/source/caption/examples"][1] == ReduceType.mean
    assert recorded["data/source/caption/loss_mass_share"][1] == ReduceType.mean


def test_multimodal_router_loss_divisor_counts_every_token_without_a_mask():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.device = torch.device("cpu")
    kwargs = train_module._batch_auxiliary_loss_kwargs(
        {"input_ids": torch.zeros((2, 4), dtype=torch.long)}
    )
    torch.testing.assert_close(kwargs["router_loss_div_factor"], torch.tensor(8))
    assert "loss_weight_div_factor" not in kwargs
    with pytest.raises(OLMoConfigurationError, match="must match input_ids"):
        train_module._batch_auxiliary_loss_kwargs(
            {
                "input_ids": torch.zeros((2, 4), dtype=torch.long),
                "router_token_mask": torch.ones((2, 3), dtype=torch.bool),
            }
        )


def _run_multimodal_router_loss_divisor_distributed():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.device = torch.device("cpu")
    train_module.dp_group = dist.group.WORLD
    rank = dist.get_rank()
    token_mask = torch.zeros((1, 4), dtype=torch.bool)
    token_mask[:, : 1 + 2 * rank] = True

    kwargs = train_module._batch_auxiliary_loss_kwargs(
        {
            "input_ids": torch.zeros_like(token_mask, dtype=torch.long),
            "router_token_mask": token_mask,
        }
    )

    # Rank-local valid counts are 1 and 3; OLMo DDP uses their global average.
    torch.testing.assert_close(kwargs["router_loss_div_factor"], torch.tensor(2.0))


def _run_multimodal_ep_step_impl(
    *, freeze_vision: bool, padded_router_compile: bool = False, fp32_accum: bool = False
):
    # The production Stage 1 model uses BF16 parameters backed by FP32 optimizer masters. Keep
    # the unfrozen test on that path so a post-optimizer vision load cannot silently regress.
    model_config = _tiny_multimodal_model_config(
        dtype=DType.float32 if freeze_vision else DType.bfloat16
    )
    if padded_router_compile:
        model_config.lm.recompute_each_block = True
        model_config.lm.block.ep = ExpertParallelConfig(
            path=ExpertParallelPath.rowwise_nvshmem,
            capacity_factor=8.0,
            major_align=1,
        )
        router = model_config.lm.block.routed_experts_router
        assert router is not None
        router.lb_loss_weight = 0.015
        router.lb_loss_granularity = MoELoadBalancingLossGranularity.instance
        router.z_loss_weight = 0.0001
    model = model_config.build(init_device="meta")
    config = MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=8,
        max_sequence_length=8,
        optim=MultimodalOLMoDDPOptimizerConfig(
            lr=1e-3,
            weight_decay=0.0,
            group_overrides=[
                OptimGroupOverride(
                    params=["*connector.*", "*lm.embeddings.weight"],
                    opts={"scheduler_name": "connector"},
                ),
                OptimGroupOverride(params=["*vision.*"], opts={"scheduler_name": "vision"}),
            ],
            foreach_chunk_size=32,
            max_grad_norm=1.0,
            clip_grad_norm_by_scheduler_group=True,
            check_nan_inf_grad=True,
        ),
        freeze_params=(
            ["vision.*", "lm.lm_head.w_out.weight"]
            if freeze_vision
            else ["lm.lm_head.w_out.weight"]
        ),
        vision_activation_checkpointing=not freeze_vision,
        connector_activation_checkpointing=not freeze_vision,
        response_logits_only=True,
        diagnostics_interval=1,
        train_embedding_rows=[120, 121],
        compile_model=padded_router_compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            only_allreduce_last_microbatch=True,
            accumulate_grads_in_fp32=fp32_accum,
            reduce_grads_in_fp32=fp32_accum,
        ),
        ep_config=TransformerExpertParallelConfig(degree=2),
    )
    train_module = config.build(model, device=torch.device("cuda"))
    multimodal = train_module.multimodal_model

    train_module.reset_image_token_rows([120, 121], seed=19, reset_output_rows=False)
    optim = train_module._require_optimizer()
    assert optim.foreach_chunk_size == 32
    optim._check_model_param_main_param_the_same()
    lm_head_norm_name = next(
        name
        for group in optim.param_groups
        for name, param in group["named_params"].items()
        if param is multimodal.lm.lm_head.norm.weight
    )
    lm_head_norm_main_before = optim.states[f"{lm_head_norm_name}.main"].to_local().clone()

    if not freeze_vision:
        external_vision_state = {
            name: (
                tensor.detach().clone() + 0.125
                if tensor.is_floating_point()
                else tensor.detach().clone()
            )
            for name, tensor in multimodal.vision.state_dict().items()
        }
        train_module.load_vision_state_dict(external_vision_state)
        train_module.assert_vision_optimizer_state_synced()

        # An optimizer step starts by copying the masters into the model. Exercise that operation
        # directly and prove it preserves the externally loaded tower exactly.
        optim._copy_main_params_to_model_params()
        for name, tensor in multimodal.vision.state_dict().items():
            torch.testing.assert_close(tensor, external_vision_state[name], rtol=0, atol=0)
    vision_params = list(multimodal.vision.parameters())
    connector_params = list(multimodal.connector.parameters())
    routed_params = [
        param for name, param in multimodal.lm.named_parameters() if "routed_experts.w_" in name
    ]
    connector_before = [param.detach().clone() for param in connector_params]
    vision_before = [param.detach().clone() for param in vision_params]
    routed_before = [param.detach().clone() for param in routed_params]
    embedding_before = multimodal.lm.embeddings.weight.detach().clone()
    lm_head_before = multimodal.lm.lm_head.w_out.weight.detach().clone()

    rank = dist.get_rank()
    input_ids = torch.tensor([[120, 2 + rank, 4, 5, 6, 7, 8, 9]], device="cuda", dtype=torch.long)
    labels = torch.tensor([[2 + rank, 4, 5, 6, 7, 8, 9, 10]], device="cuda", dtype=torch.long)
    router_token_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_masks = torch.tensor([[0.0, 0.25, 1.0, 0.5, 1.0, 0.75, 1.0, 0.5]], device="cuda")
    if padded_router_compile:
        router_token_mask[:, -3:] = False
        input_ids[:, -3:] = 0
        labels[:, -3:] = -100
        loss_masks[:, -3:] = 0
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "router_token_mask": router_token_mask,
        "loss_masks": loss_masks,
        "token_type_ids": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], device="cuda", dtype=torch.long),
        "images": torch.randn(1, 1, 4, 14 * 14 * 3, device="cuda"),
        "pooled_patches_idx": torch.tensor([[[0, 1, 2, 3]]], device="cuda", dtype=torch.long),
    }

    train_module.zero_grads()
    second_batch = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    if not freeze_vision:
        multimodal.set_input_diagnostics(True)
    train_module.train_batch(batch, dry_run=True)
    if fp32_accum:
        # Exercise the production FP32 accumulation buffer across two backwards before clipping
        # and stepping. The embedding-row hook must mask every contributing microbatch.
        train_module.train_batch(second_batch, dry_run=True)

    def has_nonzero_grad(param):
        grad = getattr(param, "_main_grad_fp32", None) if fp32_accum else param.grad
        return grad is not None and torch.count_nonzero(grad) > 0

    if freeze_vision:
        assert all(param.grad is None for param in vision_params)
        assert not multimodal.vision.training
    else:
        assert any(has_nonzero_grad(param) for param in vision_params)
        assert multimodal.vision.training
        diagnostics = multimodal.pop_input_diagnostics(
            reduce_across_process_group=True,
            process_group=train_module.dp_process_group,
        )
        assert set(diagnostics) == {
            "text embedding RMS",
            "connector output RMS",
            "spliced image embedding RMS",
        }
        assert all(torch.isfinite(value) and value > 0 for value in diagnostics.values())
    assert any(has_nonzero_grad(param) for param in connector_params)
    assert any(has_nonzero_grad(param) for param in routed_params)
    trainer = _MetricTrainerStub()
    train_module._trainer = trainer  # type: ignore[assignment]
    optim.latest_loss = torch.zeros((), device="cuda")
    train_module.optim_step()
    expected_clip_groups = {optim.DEFAULT_CLIP_GROUP_NAME, "connector"}
    if not freeze_vision:
        expected_clip_groups.add("vision")
    assert set(optim.latest_clip_group_grad_norms) == expected_clip_groups
    assert set(optim.latest_clip_group_coefficients) == expected_clip_groups
    assert all(torch.isfinite(value) for value in optim.latest_clip_group_coefficients.values())
    expected_components = {
        "connector",
        "input embeddings",
        "LM attention",
        "LM routed experts",
        "LM routers",
        "LM normalization",
    }
    if not freeze_vision:
        expected_components.add("vision")
    assert set(optim.latest_component_grad_norms) == expected_components
    assert "optim/LM output head grad norm" not in trainer.metrics
    assert all(
        f"optim/{component} grad norm" in trainer.metrics for component in expected_components
    )
    assert all(
        torch.isfinite(norm) and norm > 0 for norm in optim.latest_component_grad_norms.values()
    )
    assert optim._component_grad_norm_patterns is None

    assert any(
        not torch.equal(param, before) for param, before in zip(connector_params, connector_before)
    )
    assert any(
        not torch.equal(param, before) for param, before in zip(routed_params, routed_before)
    )
    ordinary_embedding_rows = torch.ones(multimodal.lm.vocab_size, dtype=torch.bool, device="cuda")
    ordinary_embedding_rows[[120, 121]] = False
    torch.testing.assert_close(
        multimodal.lm.embeddings.weight[ordinary_embedding_rows],
        embedding_before[ordinary_embedding_rows],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(multimodal.lm.embeddings.weight[120], embedding_before[120])
    torch.testing.assert_close(multimodal.lm.lm_head.w_out.weight, lm_head_before, rtol=0, atol=0)
    assert not torch.equal(
        optim.states[f"{lm_head_norm_name}.main"].to_local(), lm_head_norm_main_before
    )
    if freeze_vision:
        for param, before in zip(vision_params, vision_before):
            torch.testing.assert_close(param, before, rtol=0, atol=0)
    else:
        assert any(
            not torch.equal(param, before) for param, before in zip(vision_params, vision_before)
        )


def _run_multimodal_ep_step():
    _run_multimodal_ep_step_impl(freeze_vision=True)


def _run_multimodal_ep_unfrozen_vision_step():
    _run_multimodal_ep_step_impl(freeze_vision=False, fp32_accum=True)


def _run_multimodal_ep_padded_compile_step():
    _run_multimodal_ep_step_impl(freeze_vision=True, padded_router_compile=True)


@requires_multi_gpu
def test_multimodal_olmo_ddp_ep_padding_compile_and_checkpoint_step():
    run_distributed_test(
        _run_multimodal_ep_padded_compile_step,
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )


def _build_multimodal_ddp_train_module_for_checkpoint(
    *,
    freeze_vision: bool = True,
    freeze_lm_head: bool = False,
    freeze_lm_blocks: bool = False,
):
    model = _tiny_multimodal_model_config(dtype=DType.bfloat16).build(init_device="meta")
    freeze_params = []
    if freeze_vision:
        freeze_params.append("vision.*")
    if freeze_lm_head:
        freeze_params.append("lm.lm_head.w_out.weight")
    if freeze_lm_blocks:
        freeze_params.append("lm.blocks.*")
    config = MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=8,
        max_sequence_length=8,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        freeze_params=freeze_params or None,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        ep_config=TransformerExpertParallelConfig(degree=2),
    )
    return config.build(model, device=torch.device("cuda"))


def _run_native_checkpoint_into_multimodal(native_dir, hybrid_dir):
    native = _build_ddp_train_module_for_checkpoint(ep_degree=2)
    native_model = getattr(native.model_parts[0], "module", native.model_parts[0])
    native_optim = native._require_optimizer()
    assert native.moe_mesh is not None
    ep_mp_rank = native.moe_mesh["ep_mp"].get_local_rank()
    mutated_expert_names = set()
    expert_idx = 0
    with torch.no_grad():
        for group in native_optim.param_groups:
            for name, param in group["named_params"].items():
                if ".routed_experts.w_" not in name:
                    continue
                param.fill_(100 * (ep_mp_rank + 1) + expert_idx)
                expert_idx += 1
                mutated_expert_names.add(name)
    assert mutated_expert_names
    native_optim._copy_model_params_to_main_params(mutated_expert_names)
    expected_lm = {name: param.detach().clone() for name, param in native_model.named_parameters()}
    native.save_state_dict_direct(native_dir)

    unfrozen_hybrid = _build_multimodal_ddp_train_module_for_checkpoint(freeze_vision=False)
    unfrozen_model = unfrozen_hybrid.multimodal_model
    unfrozen_vision_before = {
        name: param.detach().clone() for name, param in unfrozen_model.vision.named_parameters()
    }
    unfrozen_connector_before = {
        name: param.detach().clone() for name, param in unfrozen_model.connector.named_parameters()
    }
    unfrozen_hybrid.load_state_dict_direct(native_dir, load_optim_state=False)
    for name, param in unfrozen_model.lm.named_parameters():
        torch.testing.assert_close(param, expected_lm[name], rtol=0, atol=0)
    for name, param in unfrozen_model.vision.named_parameters():
        torch.testing.assert_close(param, unfrozen_vision_before[name], rtol=0, atol=0)
    for name, param in unfrozen_model.connector.named_parameters():
        torch.testing.assert_close(param, unfrozen_connector_before[name], rtol=0, atol=0)
    unfrozen_hybrid._require_optimizer()._check_model_param_main_param_the_same()

    # Frozen expert weights under expert parallelism are a documented limitation of this
    # layer: they cannot be checkpointed, so a model with frozen LM blocks refuses to save.
    frozen_blocks = _build_multimodal_ddp_train_module_for_checkpoint(
        freeze_lm_head=True,
        freeze_lm_blocks=True,
    )
    assert all(
        not param.requires_grad for param in frozen_blocks.multimodal_model.lm.blocks.parameters()
    )
    with pytest.raises(NotImplementedError, match="frozen expert-parallel"):
        frozen_blocks._frozen_model_param_state_dict()
    with pytest.raises(NotImplementedError, match="frozen expert-parallel"):
        frozen_blocks.save_state_dict_direct(hybrid_dir)
    del frozen_blocks

    hybrid = _build_multimodal_ddp_train_module_for_checkpoint(freeze_lm_head=True)
    multimodal = hybrid.multimodal_model
    assert not multimodal.lm.lm_head.w_out.weight.requires_grad
    connector_before = {
        name: param.detach().clone() for name, param in multimodal.connector.named_parameters()
    }
    with torch.no_grad():
        for param in multimodal.lm.parameters():
            param.zero_()

    hybrid.load_state_dict_direct(native_dir, load_optim_state=False)
    for name, param in multimodal.lm.named_parameters():
        torch.testing.assert_close(param, expected_lm[name], rtol=0, atol=0)
    for name, param in multimodal.connector.named_parameters():
        torch.testing.assert_close(param, connector_before[name], rtol=0, atol=0)

    frozen_state = hybrid._frozen_model_param_state_dict()
    assert set(frozen_state) == {
        f"frozen_model.{name}"
        for name, param in multimodal.named_parameters()
        if not param.requires_grad
    }
    assert "frozen_model.lm.lm_head.w_out.weight" in frozen_state
    assert not any(".routed_experts." in key for key in frozen_state)

    optim = hybrid._require_optimizer()
    embedding_name = next(
        name
        for group in optim.param_groups
        for name, param in group["named_params"].items()
        if param is multimodal.lm.embeddings.weight
    )
    embedding_main = optim.states[f"{embedding_name}.main"]
    # Model-only loading can retain precision in the FP32 optimizer master that is not
    # representable in the BF16 model. Resetting image rows must preserve that precision for
    # every ordinary token row.
    embedding_main.to_local().add_(1e-4)
    optim._copy_main_params_to_model_params()
    main_before_reset = (
        embedding_main.full_tensor().reshape_as(multimodal.lm.embeddings.weight).clone()
    )
    lm_head_before_reset = multimodal.lm.lm_head.w_out.weight.detach().clone()

    hybrid.reset_image_token_rows([120, 121], seed=19, reset_output_rows=False)
    main_after_reset = embedding_main.full_tensor().reshape_as(multimodal.lm.embeddings.weight)
    ordinary_rows = torch.ones(128, dtype=torch.bool, device="cuda")
    ordinary_rows[[120, 121]] = False
    torch.testing.assert_close(
        main_after_reset[ordinary_rows],
        main_before_reset[ordinary_rows],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        main_after_reset[[120, 121]],
        multimodal.lm.embeddings.weight[[120, 121]].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        multimodal.lm.lm_head.w_out.weight,
        lm_head_before_reset,
        rtol=0,
        atol=0,
    )

    # Restore exact BF16/FP32 equality for the generic optimizer invariant used below.
    optim._copy_model_params_to_main_params({embedding_name})
    optim._check_model_param_main_param_the_same()

    with torch.no_grad():
        for param in multimodal.vision.parameters():
            param.fill_(0.125)
    saved_vision = {
        name: param.detach().clone() for name, param in multimodal.vision.named_parameters()
    }
    saved_connector = {
        name: param.detach().clone() for name, param in multimodal.connector.named_parameters()
    }
    saved_lm = {name: param.detach().clone() for name, param in multimodal.lm.named_parameters()}
    hybrid.save_state_dict_direct(hybrid_dir)

    metadata = RemoteFileSystemReader(hybrid_dir).read_metadata()
    for key, tensor in frozen_state.items():
        assert metadata.state_dict_metadata[key].size.numel() == tensor.numel()

    with torch.no_grad():
        for param in multimodal.parameters():
            param.fill_(-0.75)
    hybrid.load_state_dict_direct(hybrid_dir)

    for name, param in multimodal.vision.named_parameters():
        torch.testing.assert_close(param, saved_vision[name], rtol=0, atol=0)
    for name, param in multimodal.connector.named_parameters():
        torch.testing.assert_close(param, saved_connector[name], rtol=0, atol=0)
    for name, param in multimodal.lm.named_parameters():
        torch.testing.assert_close(param, saved_lm[name], rtol=0, atol=0)
    optim._check_model_param_main_param_the_same()

    # A Stage 1 checkpoint stores its frozen vision tower outside the optimizer. Stage 2
    # unfreezes that tower, so a model-only load must restore those weights and seed their new
    # FP32 optimizer masters while retaining the trainable LM and connector weights.
    stage2 = _build_multimodal_ddp_train_module_for_checkpoint(freeze_vision=False)
    stage2_model = stage2.multimodal_model
    with torch.no_grad():
        for param in stage2_model.parameters():
            param.fill_(-0.5)
    stage2.load_state_dict_direct(hybrid_dir, load_optim_state=False)

    for name, param in stage2_model.vision.named_parameters():
        torch.testing.assert_close(param, saved_vision[name], rtol=0, atol=0)
    for name, param in stage2_model.connector.named_parameters():
        torch.testing.assert_close(param, saved_connector[name], rtol=0, atol=0)
    for name, param in stage2_model.lm.named_parameters():
        torch.testing.assert_close(param, saved_lm[name], rtol=0, atol=0)
    stage2._require_optimizer()._check_model_param_main_param_the_same()


@requires_multi_gpu
def test_native_checkpoint_loads_into_multimodal_and_roundtrips(tmp_path):
    run_distributed_test(
        _run_native_checkpoint_into_multimodal,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(str(tmp_path / "native"), str(tmp_path / "hybrid")),
    )


def _build_ddp_train_module_for_checkpoint(
    *, router_bias_gamma: Optional[float] = None, ep_degree: Optional[int] = None
):
    model = _tiny_model_config(dtype=DType.bfloat16, router_bias_gamma=router_bias_gamma).build(
        init_device="cuda"
    )
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        ep_config=(
            TransformerExpertParallelConfig(degree=ep_degree) if ep_degree is not None else None
        ),
    )
    return config.build(model, device=torch.device("cuda"), eval_only=False)
