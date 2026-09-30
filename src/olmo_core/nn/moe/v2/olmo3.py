"""Olmo3Moe builders and correctness-first OLMoDDP/HF weight interchange."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig

from olmo_core.config import DType
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GateGranularity,
    KimiDeltaAttentionConfig,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.hf.convert import (
    convert_state_from_hf,
    convert_state_to_hf,
    iter_olmo3moe_state_to_hf,
)
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import (
    EmoRouterConfig,
    LatentMoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerBlockType, TransformerType
from olmo_core.nn.transformer.config import OLMoDDPModelConfig, TransformerBlockConfig

OLMO3_FULL_ATTENTION = "full_attention"
OLMO3_SLIDING_ATTENTION = "sliding_attention"
OLMO3_LINEAR_ATTENTION = "linear_attention"


def _as_mapping(config: PretrainedConfig | Mapping[str, Any]) -> Mapping[str, Any]:
    return config.to_dict() if isinstance(config, PretrainedConfig) else config


def _emo_config_from_hf(config: Mapping[str, Any]) -> EmoRouterConfig | None:
    fields = (
        "emo_min_document_expert_pool",
        "emo_max_document_expert_pool",
        "emo_eval_document_expert_pool",
        "emo_eos_token_id",
    )
    mode = config.get("emo_routing_mode")
    if mode not in (None, "full_pool"):
        raise NotImplementedError(f"Unsupported EMO routing mode: {mode!r}")
    if all(config.get(key) is None for key in fields):
        return None
    if mode != "full_pool":
        raise NotImplementedError(
            "EMO HF training requires explicit emo_routing_mode='full_pool'; "
            "native document-pool training is not a causal RL policy."
        )
    for key in fields:
        value = config.get(key)
        if type(value) is not int or value < (0 if key == "emo_eos_token_id" else 1):
            raise ValueError(f"EMO requires an explicit valid integer {key}")
    emo = EmoRouterConfig(
        eos_token_id=config["emo_eos_token_id"],
        min_document_expert_pool=config["emo_min_document_expert_pool"],
        max_document_expert_pool=config["emo_max_document_expert_pool"],
        eval_document_expert_pool=config["emo_eval_document_expert_pool"],
        full_pool=True,
        source_config=deepcopy(config.get("emo_source_config")),
    )
    emo.validate_for_router(
        num_experts=config["n_routed_experts"], top_k=config["num_experts_per_tok"]
    )
    return emo


def build_olmo3_moe_config_from_hf_config(
    hf_config: PretrainedConfig | Mapping[str, Any],
    *,
    dtype: DType = DType.bfloat16,
    attention_backend: AttentionBackendName = AttentionBackendName.flash_4,
    attention_type: AttentionType = AttentionType.default,
    ep_path: ExpertParallelPath | str = ExpertParallelPath.sync_1d,
    ep_capacity_factor: float = 1.25,
    router_aux_loss_weight: float | None = None,
    router_z_loss_weight: float | None = None,
    init_seed: int = 2026,
) -> OLMoDDPModelConfig:
    """Build an OLMoDDP model from a supported Olmo3Moe checkpoint config."""
    config = _as_mapping(hf_config)
    if config.get("model_type") != "olmo3moe":
        raise ValueError(f"Expected model_type='olmo3moe', got {config.get('model_type')!r}.")
    emo = _emo_config_from_hf(config)
    rope_parameters = config.get("rope_parameters") or config.get("rope_scaling") or {}
    if rope_parameters and rope_parameters.get("rope_type", "default") != "default":
        raise NotImplementedError("The OLMo3 MoE builder does not support scaled RoPE.")
    if config.get("attention_bias", False):
        raise NotImplementedError("Biased Olmo3Moe attention is not supported.")
    if config.get("hidden_act", "silu") != "silu":
        raise NotImplementedError("Only SwiGLU Olmo3Moe experts are supported.")
    if not config.get("use_head_qk_norm", False):
        raise NotImplementedError("Olmo3Moe conversion requires head-wise QK norm.")
    if attention_type not in (AttentionType.default, AttentionType.fused_v2):
        raise NotImplementedError(
            f"Olmo3Moe conversion does not support attention type {attention_type!r}."
        )

    n_layers = int(config["num_hidden_layers"])
    dense_layers = {int(idx) for idx in config.get("dense_layers_indices") or ()}
    invalid_dense_layers = sorted(idx for idx in dense_layers if idx < 0 or idx >= n_layers)
    if invalid_dense_layers:
        raise ValueError(
            f"dense_layers_indices must be in [0, {n_layers}), got {invalid_dense_layers}."
        )
    dense_hidden = config.get("dense_mlp_intermediate_size")
    if dense_layers and dense_hidden is None:
        raise ValueError(
            "dense_mlp_intermediate_size must be set when dense_layers_indices is non-empty."
        )
    layer_types = tuple(config.get("layer_types") or (OLMO3_FULL_ATTENTION,) * n_layers)
    if len(layer_types) != n_layers:
        raise ValueError(f"Expected {n_layers} layer_types, got {len(layer_types)}.")
    unsupported = set(layer_types) - {
        OLMO3_FULL_ATTENTION,
        OLMO3_SLIDING_ATTENTION,
        OLMO3_LINEAR_ATTENTION,
    }
    if unsupported:
        raise ValueError(f"Unsupported Olmo3Moe layer types: {sorted(unsupported)}")

    d_model = int(config["hidden_size"])
    num_experts = int(config["n_routed_experts"])
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=float(config["rms_norm_eps"]),
        bias=False,
        dtype=dtype,
    )
    latent_dim = config.get("latent_moe_dim")
    latent_moe = (
        LatentMoEConfig(
            latent_dim=int(latent_dim),
            bias=bool(config.get("latent_moe_bias", False)),
            up_proj_input_norm_enabled=bool(config.get("latent_moe_up_proj_input_norm", False)),
        )
        if latent_dim is not None
        else None
    )
    routed_experts = RoutedExpertsConfig(
        d_model=d_model if latent_dim is None else int(latent_dim),
        hidden_size=int(config["moe_intermediate_size"]),
        num_experts=num_experts,
        bias=False,
        dtype=dtype,
    )
    routed_router = MoERouterConfigV2(
        emo=emo,
        d_model=d_model,
        num_experts=num_experts,
        top_k=int(config["num_experts_per_tok"]),
        gating_function=MoERouterGatingFunction(config.get("gating_function", "softmax")),
        normalize_expert_weights=config.get("normalize_expert_weights"),
        restore_weight_scale=bool(config.get("restore_weight_scale", False)),
        original_top_k=config.get("original_num_experts_per_tok"),
        lb_loss_weight=router_aux_loss_weight,
        z_loss_weight=router_z_loss_weight,
        global_load_balancing=bool(config.get("global_load_balancing", False)),
        lb_loss_granularity=(
            MoELoadBalancingLossGranularity.local_batch
            if config.get("global_load_balancing", False)
            else MoELoadBalancingLossGranularity.instance
        ),
        dtype=dtype,
    )
    shared_hidden = config.get("shared_expert_intermediate_size")
    shared_experts = (
        None
        if shared_hidden is None
        else SharedExpertsConfig(
            d_model=d_model,
            hidden_size=int(shared_hidden),
            num_experts=1,
            bias=False,
            dtype=dtype,
        )
    )
    ep = ExpertParallelConfig(
        path=ExpertParallelPath(ep_path),
        capacity_factor=ep_capacity_factor,
    )
    ep.validate()

    def make_block(layer_type: str, *, dense: bool) -> OLMoDDPTransformerBlockConfig:
        window = int(config["sliding_window"])
        block_shared_experts: SharedExpertsConfig | None
        if dense:
            assert dense_hidden is not None
            block_shared_experts = SharedExpertsConfig(
                d_model=d_model,
                hidden_size=int(dense_hidden),
                num_experts=1,
                bias=False,
                dtype=dtype,
            )
        else:
            block_shared_experts = shared_experts
        mixer: AttentionConfig | KimiDeltaAttentionConfig = AttentionConfig(
            name=attention_type,
            n_heads=int(config["num_attention_heads"]),
            n_kv_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            bias=False,
            dropout=float(config.get("attention_dropout", 0.0)),
            rope=(
                RoPEConfig(
                    name=RoPEType.default,
                    theta=config.get("rope_theta", 10_000),
                    full_precision=True,
                )
                if config.get("use_rope", True)
                else None
            ),
            gate=(
                GateConfig(
                    granularity=GateGranularity(config["attention_gate_type"]),
                    full_precision=bool(config.get("attention_gate_full_precision", True)),
                )
                if config.get("attention_gate_type")
                else None
            ),
            qk_norm=layer_norm,
            use_head_qk_norm=True,
            qk_norm_per_head_gains=bool(config.get("qk_norm_per_head_gains", False)),
            scalable_softmax=bool(config.get("scalable_softmax", False)),
            backend=attention_backend,
            dtype=dtype,
            sliding_window=(
                SlidingWindowAttentionConfig(
                    pattern=[window],
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=False,
                )
                if layer_type == OLMO3_SLIDING_ATTENTION
                else None
            ),
        )
        if layer_type == OLMO3_LINEAR_ATTENTION:
            key_dim = int(config["linear_key_head_dim"])
            value_dim = int(config["linear_value_head_dim"])
            mixer = KimiDeltaAttentionConfig(
                n_heads=int(config["linear_num_key_heads"]),
                n_v_heads=int(config["linear_num_value_heads"]),
                head_dim=key_dim,
                expand_v=value_dim / key_dim,
                conv_size=int(config.get("linear_conv_kernel_dim", 4)),
                allow_neg_eigval=bool(config.get("linear_allow_neg_eigval", False)),
                norm_eps=float(config.get("linear_norm_eps", 1e-5)),
                dtype=dtype,
            )
        return OLMoDDPTransformerBlockConfig(
            sequence_mixer=mixer,
            latent_moe=None if dense else latent_moe,
            ep=None if dense else ep,
            routed_experts=None if dense else routed_experts,
            routed_experts_router=None if dense else routed_router,
            shared_experts=block_shared_experts,
            name=TransformerBlockType.moe_fused_v2,
            use_pre_norm=False,
            use_peri_norm=bool(config.get("use_peri_ln", False)),
            layer_norm=deepcopy(layer_norm),
            shared_experts_router=None,
        )

    def block_name(layer_type: str, *, dense: bool) -> str:
        return f"{layer_type}_dense" if dense else layer_type

    block_pattern = [
        block_name(layer_type, dense=layer_idx in dense_layers)
        for layer_idx, layer_type in enumerate(layer_types)
    ]
    blocks = {}
    for layer_idx, layer_type in enumerate(layer_types):
        dense = layer_idx in dense_layers
        name = block_name(layer_type, dense=dense)
        if name not in blocks:
            blocks[name] = make_block(layer_type, dense=dense)

    block: TransformerBlockConfig | dict[str, TransformerBlockConfig]
    resolved_block_pattern: list[str] | None
    if len(blocks) == 1:
        block = next(iter(blocks.values()))
        resolved_block_pattern = None
    else:
        block = dict(blocks)
        resolved_block_pattern = block_pattern

    model_config = OLMoDDPModelConfig(
        init_seed=init_seed,
        init_std=float(config.get("initializer_range", 0.02)),
        d_model=d_model,
        vocab_size=int(config["vocab_size"]),
        n_layers=n_layers,
        block=block,
        block_pattern=resolved_block_pattern,
        embedding_norm=layer_norm if config.get("embed_norm", False) else None,
        embed_scale=float(config.get("embed_scale", 1.0)),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,
        dtype=dtype,
        two_batch_overlap=False,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
    )
    model_config.lm_head.loss_implementation = LMLossImplementation.default
    return model_config


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    wrapped = getattr(model, "module", None)
    return wrapped if isinstance(wrapped, torch.nn.Module) else model


def _config_for_native_dense_layout(
    model: torch.nn.Module, config: PretrainedConfig
) -> PretrainedConfig:
    # This HF field describes the source Core tensor layout. Both layouts export
    # exactly the same HF MLP. The adapter always builds shared-expert dense blocks,
    # but can import HF files originally exported from ordinary dense blocks.
    layouts = {
        hasattr(model.get_submodule(f"blocks.{index}"), "shared_experts")
        and model.get_submodule(f"blocks.{index}").shared_experts is not None
        for index in (config.dense_layers_indices or [])
    }
    if len(layouts) > 1:
        raise NotImplementedError("Mixed native dense tensor layouts are unsupported.")
    if layouts and next(iter(layouts)) != config.dense_layers_use_shared_expert:
        config = deepcopy(config)
        config.dense_layers_use_shared_expert = next(iter(layouts))
    return config


def load_olmo3_moe_hf_state(
    model: torch.nn.Module, hf_config: PretrainedConfig, hf_state: Mapping[str, torch.Tensor]
) -> None:
    """Load a full HF state into an unsharded or EP-sharded OLMoDDP model."""
    model = _unwrap_model(model)
    hf_config = _config_for_native_dense_layout(model, hf_config)
    native_state = convert_state_from_hf(hf_config, dict(hf_state), model_type="olmo3moe")
    parameters = dict(model.named_parameters())
    for layer_idx in range(hf_config.num_hidden_layers):
        prefix = f"blocks.{layer_idx}.attention."
        fused_key = f"{prefix}w_qkv.weight"
        if fused_key in parameters:
            native_state[fused_key] = torch.cat(
                [
                    native_state.pop(f"{prefix}w_q.weight"),
                    native_state.pop(f"{prefix}w_k.weight"),
                    native_state.pop(f"{prefix}w_v.weight"),
                ],
                dim=0,
            ).contiguous()
    missing = set(parameters) - set(native_state)
    if missing:
        raise RuntimeError(f"Converted Olmo3Moe state is missing parameters: {sorted(missing)}")
    unexpected = set(native_state) - set(parameters)
    if unexpected:
        raise RuntimeError(
            f"Converted Olmo3Moe state has unexpected parameters: {sorted(unexpected)}"
        )

    with torch.no_grad():
        for name, target in parameters.items():
            source = native_state[name]
            owner_name = name.rsplit(".", 1)[0]
            owner = model.get_submodule(owner_name)
            if getattr(owner, "_ep_sharded", False):
                local_experts = int(owner.num_local_experts)
                start = int(owner.ep_rank) * local_experts
                source = source[start : start + local_experts]
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"Shape mismatch for {name}: converted={tuple(source.shape)}, "
                    f"model={tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def _gather_native_tensor(model: torch.nn.Module, name: str, value: torch.Tensor) -> torch.Tensor:
    """Gather an expert-sharded tensor in EP rank order; leave other tensors local."""
    local = get_local_tensor(value) if isinstance(value, DTensor) else value
    owner = model.get_submodule(name.rsplit(".", 1)[0])
    if getattr(owner, "_ep_sharded", False):
        group = owner.ep_mesh["ep_mp"].get_group()
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered, local.contiguous(), group=group)
        local = torch.cat(gathered, dim=0)
    return local


def _split_fused_qkv(weight: torch.Tensor, config: PretrainedConfig) -> tuple[torch.Tensor, ...]:
    """Return separate Q/K/V views using the checkpoint's query and KV head counts."""
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    return weight.split((q_dim, kv_dim, kv_dim), dim=0)


def gather_olmo3_moe_hf_state(
    model: torch.nn.Module, hf_config: PretrainedConfig, *, cpu: bool = False
) -> dict[str, torch.Tensor]:
    """Collect an EP-sharded OLMoDDP model and return a full HF state on every rank."""
    model = _unwrap_model(model)
    native_state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        local = _gather_native_tensor(model, name, value)
        native_state[name] = local.cpu() if cpu else local

    for layer_idx in range(hf_config.num_hidden_layers):
        prefix = f"blocks.{layer_idx}.attention."
        fused_key = f"{prefix}w_qkv.weight"
        if fused_key in native_state:
            q, k, v = _split_fused_qkv(native_state.pop(fused_key), hf_config)
            native_state[f"{prefix}w_q.weight"] = q
            native_state[f"{prefix}w_k.weight"] = k
            native_state[f"{prefix}w_v.weight"] = v

    return convert_state_to_hf(_config_for_native_dense_layout(model, hf_config), native_state)


class _GatheredMoEState(Mapping[str, torch.Tensor]):
    """Read-only state inventory that gathers a parameter only when consumed."""

    def __init__(self, model: torch.nn.Module, config: PretrainedConfig):
        self.model = _unwrap_model(model)
        self.state = self.model.state_dict()
        self.aliases: dict[str, tuple[str, int]] = {}
        self.config = config
        self.inventory = list(self.state)
        for key in list(self.inventory):
            if key.endswith(".attention.w_qkv.weight"):
                self.inventory.remove(key)
                for index, name in enumerate(("w_q", "w_k", "w_v")):
                    alias = key.replace("w_qkv", name)
                    self.aliases[alias] = (key, index)
                    self.inventory.append(alias)

    def __iter__(self):
        return iter(self.inventory)

    def __len__(self):
        return len(self.inventory)

    def __getitem__(self, name):
        key, part = self.aliases.get(name, (name, None))
        local = _gather_native_tensor(self.model, key, self.state[key])
        if part is not None:
            local = _split_fused_qkv(local, self.config)[part]
        return local


def iter_olmo3_moe_hf_state(
    model: torch.nn.Module, hf_config: PretrainedConfig, *, fused_experts: bool = False
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream HF tensors on the model device using the canonical converter.

    All EP ranks must consume the iterator in the same order. No complete model
    CPU replica is staged. The current expert slabs and consumer's output bucket
    still need device memory; this is not an arbitrarily small-memory exporter.
    ``fused_experts`` yields each layer's routed experts as two stacked tensors
    in the serving engine's layout instead of per-expert slices.
    """
    hf_config = _config_for_native_dense_layout(_unwrap_model(model), hf_config)
    yield from iter_olmo3moe_state_to_hf(
        hf_config, _GatheredMoEState(model, hf_config), fused_experts=fused_experts
    )
