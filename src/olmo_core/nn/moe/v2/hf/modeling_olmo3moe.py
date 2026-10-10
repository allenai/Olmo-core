from collections.abc import Callable
from inspect import signature
from typing import Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.olmo3.modeling_olmo3 import eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils.auto_docstring import auto_docstring
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import TransformersKwargs, can_return_tuple

from .configuration_olmo3moe import Olmo3MoeConfig
from .moe_numerics import core_combine, core_swiglu


def _create_mask_compat(mask_fn: Callable, **kwargs):
    """Call Transformers mask helpers across the `input_embeds` rename."""
    params = set(signature(mask_fn).parameters)
    if "input_embeds" in params and "inputs_embeds" in kwargs:
        kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
    return mask_fn(**{k: v for k, v in kwargs.items() if k in params})


def _uses_layer_type_rope_parameters(config: Olmo3MoeConfig) -> bool:
    rope_parameters = getattr(config, "rope_parameters", None)
    layer_types = set(getattr(config, "layer_types", []) or [])
    return (
        isinstance(rope_parameters, dict)
        and bool(rope_parameters)
        and bool(layer_types)
        and set(rope_parameters).issubset(layer_types)
    )


def _get_rope_parameters(config: Olmo3MoeConfig, layer_type: Optional[str] = None) -> dict:
    rope_parameters = config.rope_parameters
    if layer_type is not None and _uses_layer_type_rope_parameters(config):
        return rope_parameters[layer_type]
    return rope_parameters


class Olmo3MoeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Olmo3MoeConfig, device=None, layer_type: Optional[str] = None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.layer_type = layer_type

        rope_parameters = _get_rope_parameters(config, layer_type)
        rope_type = rope_parameters["rope_type"]
        self.rope_type = {layer_type: rope_type} if layer_type is not None else rope_type
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)

        prefix = f"{layer_type}_" if layer_type is not None else ""
        self.register_buffer(f"{prefix}inv_freq", inv_freq, persistent=False)
        self.register_buffer(f"{prefix}original_inv_freq", inv_freq.clone(), persistent=False)
        setattr(self, f"{prefix}attention_scaling", attention_scaling)
        self.attention_scaling = attention_scaling

    @staticmethod
    def compute_default_rope_parameters(
        config: Olmo3MoeConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: Optional[str] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        assert config is not None
        rope_parameters = _get_rope_parameters(config, layer_type)
        base = rope_parameters["rope_theta"]
        dim = (
            getattr(config, "head_dim", None)
            or config.attention_hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                / dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type: Optional[str] = None):
        layer_type = layer_type or self.layer_type
        if layer_type is None:
            inv_freq = self.inv_freq
            attention_scaling = self.attention_scaling
        else:
            inv_freq = getattr(self, f"{layer_type}_inv_freq")
            attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = (
            inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
            return cos, sin


def _core_shared_expert_forward(expert, x):
    # Match SharedExperts' packed GEMM and operand layout. Separate up/gate
    # projections can select a different accumulation order on Ampere.
    weight = torch.cat((expert.up_proj.weight.t(), expert.gate_proj.weight.t()), dim=1)
    up, gate = (x.reshape(-1, x.shape[-1]) @ weight).chunk(2, dim=-1)
    hidden = expert.act_fn(gate) * up
    down_weight = expert.down_proj.weight.t().contiguous().unsqueeze(0)
    return torch.bmm(hidden.unsqueeze(0), down_weight).reshape(x.shape)


class Olmo3MoeDenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.dense_mlp_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.moe_use_core_numerics:
            return _core_shared_expert_forward(self, x)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Olmo3MoeExpert(nn.Module):
    def __init__(self, hidden_size, moe_intermediate_size, hidden_act):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.moe_intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Olmo3MoeCoreSharedExpert(Olmo3MoeExpert):
    """Shared expert preserving core's packed projection and GEMM layouts."""

    def forward(self, x):
        return _core_shared_expert_forward(self, x)


class Olmo3MoeExperts(nn.ModuleList):
    """Container for routed experts.

    vLLM detects a child module named ``experts`` that is a ``ModuleList`` and
    replaces it with a fused implementation at load time (TransformersFusedMoE).
    This class provides an eager reference implementation, and a compile-safe
    fallback (very slow) to avoid TorchDynamo graph breaks if it is ever traced.
    """

    @staticmethod
    def _torch_grouped_mm_available() -> bool:
        if not hasattr(F, "grouped_mm"):
            return False

        # Keep a conservative version guard for environments where the symbol
        # may exist before the `offs=` API used below is supported.
        torch_version = torch.__version__.split("+")[0]
        try:
            major_str, minor_str, *_ = torch_version.split(".")
            major, minor = int(major_str), int(minor_str)
        except (ValueError, TypeError):
            return True

        return major > 2 or (major == 2 and minor >= 10)

    def _can_use_grouped_mm(self, hidden_states: torch.Tensor) -> bool:
        if not self._torch_grouped_mm_available() or len(self) == 0:
            return False
        if hidden_states.device.type not in {"cpu", "cuda"}:
            return False
        if hidden_states.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            return False

        first_expert = cast(Olmo3MoeExpert, self[0])
        hidden_size = first_expert.hidden_size
        intermediate_size = first_expert.moe_intermediate_size

        # grouped_mm requires row strides to be aligned to 16 bytes.
        element_size = hidden_states.element_size()
        return (hidden_size * element_size) % 16 == 0 and (
            intermediate_size * element_size
        ) % 16 == 0

    def _forward_compile_fallback(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        out = hidden_states.new_zeros(hidden_states.shape, dtype=torch.float32)
        for expert_id, expert in enumerate(self):
            # Aggregate the routing weights for this expert across the K slots.
            w = (topk_weights * (topk_ids == expert_id).to(topk_weights.dtype)).sum(
                dim=1, keepdim=True
            )  # (N, 1)
            out = out + expert(hidden_states).float() * w.float()
        return out.to(hidden_states.dtype)

    def _forward_loop(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        N, H = hidden_states.shape
        out = hidden_states.new_zeros((N, H), dtype=torch.float32)
        for expert_id, expert in enumerate(self):
            mask = topk_ids == expert_id  # (N, K) bool
            if not mask.any():
                continue
            token_ids, k_ids = mask.nonzero(as_tuple=True)  # both (M,)
            x_sel = hidden_states.index_select(0, token_ids)  # (M, H)
            y_sel = expert(x_sel)  # (M, H)
            w_sel = topk_weights[token_ids, k_ids].unsqueeze(-1).float()
            out.index_add_(0, token_ids, y_sel.float() * w_sel)
        return out.to(hidden_states.dtype)

    def _forward_grouped_mm(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run routed experts by grouping tokens per expert and using grouped_mm."""
        N, H = hidden_states.shape
        K = topk_ids.shape[-1]
        num_experts = len(self)

        route_token_ids = torch.arange(N, device=hidden_states.device).repeat_interleave(K)
        route_expert_ids = topk_ids.reshape(-1)
        route_weights = topk_weights.reshape(-1).float()

        sorted_route_ids = torch.argsort(route_expert_ids)
        sorted_expert_ids = route_expert_ids.index_select(0, sorted_route_ids)
        sorted_token_ids = route_token_ids.index_select(0, sorted_route_ids)
        sorted_weights = route_weights.index_select(0, sorted_route_ids)

        batch_size_per_expert = torch.bincount(sorted_expert_ids, minlength=num_experts).to(
            dtype=torch.int32
        )
        offs = torch.cumsum(batch_size_per_expert, dim=0, dtype=torch.int32)
        x_grouped = hidden_states.index_select(0, sorted_token_ids)

        w_gate_up = torch.stack(
            [
                torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0).transpose(0, 1)
                for expert in self
            ]
        )
        w_down = torch.stack([expert.down_proj.weight.transpose(0, 1) for expert in self])

        gate_up = F.grouped_mm(x_grouped, w_gate_up, offs=offs)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = cast(Olmo3MoeExpert, self[0]).act_fn(gate) * up
        y_grouped = F.grouped_mm(hidden, w_down, offs=offs)

        # Reduce in the same expert-order as the reference loop. This avoids
        # duplicate-index CUDA atomics and keeps close greedy decisions stable.
        weighted_y_grouped = y_grouped.float() * sorted_weights.unsqueeze(-1)
        token_expert_order = torch.argsort(sorted_token_ids * num_experts + sorted_expert_ids)
        weighted_y = weighted_y_grouped.index_select(0, token_expert_order)
        weighted_y = weighted_y.reshape(N, K, H)
        out = hidden_states.new_zeros((N, H), dtype=torch.float32)
        for slot in range(K):
            out = out + weighted_y[:, slot]
        return out.to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (N, H)
        topk_ids: torch.Tensor,  # (N, K)
        topk_weights: torch.Tensor,  # (N, K)
    ) -> torch.Tensor:
        # Use a compile-safe fallback if TorchDynamo is tracing this module.
        # NOTE: This is extremely slow because it runs every expert on every token.
        try:
            is_compiling = torch._dynamo.is_compiling()
        except Exception:
            is_compiling = False

        if is_compiling:
            return self._forward_compile_fallback(hidden_states, topk_ids, topk_weights)

        if self._can_use_grouped_mm(hidden_states):
            # `_can_use_grouped_mm` is a cheap gate, but whether `grouped_mm` actually accepts a
            # given device/dtype combination varies across torch builds. Fall back to the reference
            # loop if the op rejects the operands rather than failing the forward.
            try:
                return self._forward_grouped_mm(hidden_states, topk_ids, topk_weights)
            except (NotImplementedError, RuntimeError):
                pass

        # Eager reference routing, used when torch grouped_mm is unavailable or unsupported.
        return self._forward_loop(hidden_states, topk_ids, topk_weights)


class Olmo3MoeCoreExperts(Olmo3MoeExperts):
    """Independent HF expert computation with OLMoDDP inference rounding.

    This is part of the exported model, not a conversion-only verifier. Optimized
    inference engines can replace this ModuleList, with their own numerical
    qualification, just as they replace the original HF expert implementation.
    """

    @staticmethod
    def _expert_forward(expert, x):
        weight = torch.cat((expert.up_proj.weight, expert.gate_proj.weight), dim=0)
        hidden = core_swiglu(F.linear(x, weight))
        return expert.down_proj(hidden)

    def _forward_compile_fallback(self, hidden_states, topk_ids, topk_weights):
        N, H = hidden_states.shape
        K = topk_ids.shape[1]
        values = hidden_states.new_zeros((N, K, H))
        for expert_id, expert in enumerate(self):
            output = self._expert_forward(expert, hidden_states)
            values = torch.where((topk_ids == expert_id)[..., None], output[:, None], values)
        return core_combine(values, topk_weights)

    def _forward_loop(self, hidden_states, topk_ids, topk_weights):
        N, H = hidden_states.shape
        K = topk_ids.shape[1]
        values = hidden_states.new_zeros((N, K, H))
        for expert_id, expert in enumerate(self):
            token_ids, slots = (topk_ids == expert_id).nonzero(as_tuple=True)
            if token_ids.numel():
                values[token_ids, slots] = self._expert_forward(expert, hidden_states[token_ids])
        return core_combine(values, topk_weights)

    def _forward_grouped_mm(self, hidden_states, topk_ids, topk_weights):
        N, H = hidden_states.shape
        K = topk_ids.shape[1]
        expert_ids = topk_ids.reshape(-1)
        order = torch.argsort(expert_ids, stable=True)
        token_ids = torch.arange(N, device=hidden_states.device).repeat_interleave(K)[order]
        counts = torch.bincount(expert_ids, minlength=len(self)).to(torch.int32)
        offsets = counts.cumsum(0, dtype=torch.int32)
        # Preserve core's packed storage layout and [up, gate] ordering so the
        # grouped GEMM chooses the same operand layout as the source model.
        w_up_gate = torch.stack(
            [torch.cat((expert.up_proj.weight, expert.gate_proj.weight), dim=0) for expert in self]
        )
        w_down = torch.stack([expert.down_proj.weight.t() for expert in self])
        up_gate = F.grouped_mm(hidden_states[token_ids], w_up_gate.transpose(1, 2), offs=offsets)
        output = F.grouped_mm(core_swiglu(up_gate), w_down, offs=offsets)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        values = output[inverse].reshape(N, K, H)
        return core_combine(values, topk_weights)


class Olmo3MoeSparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = Olmo3MoeRouter(config)
        self.routed_hidden_size = (
            config.latent_moe_dim if config.latent_moe_dim is not None else config.hidden_size
        )
        self.latent_down_proj: Optional[nn.Linear]
        self.latent_up_proj_input_norm: Optional[Olmo3MoeRMSNorm]
        self.latent_up_proj: Optional[nn.Linear]
        if config.latent_moe_dim is None:
            self.latent_down_proj = None
            self.latent_up_proj_input_norm = None
            self.latent_up_proj = None
        else:
            self.latent_down_proj = nn.Linear(
                config.hidden_size,
                config.latent_moe_dim,
                bias=config.latent_moe_bias,
            )
            self.latent_up_proj_input_norm = (
                Olmo3MoeRMSNorm(config.latent_moe_dim, eps=config.rms_norm_eps)
                if config.latent_moe_up_proj_input_norm
                else None
            )
            self.latent_up_proj = nn.Linear(
                config.latent_moe_dim,
                config.hidden_size,
                bias=config.latent_moe_bias,
            )
        self.experts = Olmo3MoeCoreExperts() if config.moe_use_core_numerics else Olmo3MoeExperts()
        for _ in range(config.n_routed_experts):
            expert = Olmo3MoeExpert(
                hidden_size=self.routed_hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
            )
            self.experts.append(expert)
        self.shared_expert: Optional[Olmo3MoeExpert]
        if config.shared_expert_intermediate_size is not None:
            shared_expert_cls = (
                Olmo3MoeCoreSharedExpert if config.moe_use_core_numerics else Olmo3MoeExpert
            )
            self.shared_expert = shared_expert_cls(
                hidden_size=config.hidden_size,
                moe_intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
            )
        else:
            self.shared_expert = None

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        B, S, H = x.shape

        # Compute gating weights and expert indices
        # expert_weights: (batch_size, seq_len, top_k)
        # expert_indices: (batch_size, seq_len, top_k)
        expert_weights, expert_indices = self.router(x)
        K = expert_indices.size(-1)
        # Flatten tokens: N = B*S. vLLM's fused experts expects (N, H), (N, K), (N, K).
        routed_x = self.latent_down_proj(x) if self.latent_down_proj is not None else x
        routed_h = routed_x.shape[-1]
        x_flat = routed_x.reshape(B * S, routed_h)
        idx_flat = expert_indices.reshape(B * S, K)  # (N, K)
        # Preserve FP32 router probabilities until the selected expert implementation
        # applies its combination policy (TE index-map or the original FP32 HF sum).
        w_flat = expert_weights.reshape(B * S, K).float()  # (N, K)

        out_flat = self.experts(x_flat, topk_ids=idx_flat, topk_weights=w_flat)
        routed_expert_out = out_flat.view(B, S, routed_h)
        if self.latent_up_proj is not None:
            if self.latent_up_proj_input_norm is not None:
                routed_expert_out = self.latent_up_proj_input_norm(routed_expert_out)
            routed_expert_out = self.latent_up_proj(routed_expert_out)

        # shared expert
        if self.shared_expert is None:
            out = routed_expert_out
        else:
            shared_expert_out = self.shared_expert(x)
            out = routed_expert_out + shared_expert_out

        return out


class Olmo3MoeRouter(nn.Module):
    def __init__(self, config: Olmo3MoeConfig):
        super().__init__()
        self.config = config
        self.gating_function = config.gating_function
        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.original_num_experts_per_tok = config.original_num_experts_per_tok
        self.gate = nn.Linear(self.hidden_size, config.n_routed_experts, bias=False)
        self.normalize_expert_weights = config.normalize_expert_weights
        self.restore_weight_scale = config.restore_weight_scale

    def forward(self, x):
        # Match core routing even when model.to(bfloat16) casts the router weight.
        # Rounding logits/probabilities can change both top-k membership and mixing.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.gate.weight.float())

        if self.gating_function == "softmax":
            scores = logits.softmax(dim=-1)
        elif self.gating_function == "sigmoid":
            scores = torch.sigmoid(logits)
            # to avoid NaNs in the load balancing loss
            # if all logits of a token are very negative for all experts, sigmoid gives 0 for all experts, causing NaNs when we div by the sum.
            scores = scores + 1e-7
        else:
            raise NotImplementedError(self.gating_function)

        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        if self.restore_weight_scale:
            expert_weights = expert_weights * self.num_experts_per_tok

        if (
            self.original_num_experts_per_tok is not None
            and self.num_experts_per_tok != self.original_num_experts_per_tok
        ):
            expert_weights = (
                expert_weights
                * (self.original_num_experts_per_tok / self.num_experts_per_tok) ** 0.5
            )

        return expert_weights, expert_indices


class Olmo3MoeCausalConv1d(nn.Conv1d):
    """Depthwise causal convolution with the same FLA path as Olmo-core KDA."""

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=False,
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from fla.modules.convolution import causal_conv1d
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError(
                "KDA inference requires flash-linear-attention with "
                "fla.modules.convolution.causal_conv1d"
            ) from exc
        output, _ = causal_conv1d(
            x=x,
            weight=self.weight.squeeze(1),
            bias=None,
            activation="silu",
            backend="triton",
            cu_seqlens=None,
        )
        return output


class Olmo3MoeKimiDeltaAttention(nn.Module):
    """HF-side KDA matching :class:`olmo_core.nn.attention.KimiDeltaAttention`."""

    def __init__(self, config: Olmo3MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if config.linear_num_key_heads is None or config.linear_key_head_dim is None:
            raise ValueError("KDA layers require linear_num_key_heads and linear_key_head_dim")
        if config.linear_value_head_dim is None:
            raise ValueError("KDA layers require linear_value_head_dim")

        try:
            from fla.modules import FusedRMSNormGated
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError(
                "KDA inference requires flash-linear-attention with fla.modules.FusedRMSNormGated"
            ) from exc

        self.n_heads = config.linear_num_key_heads
        self.n_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.n_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim
        self.gate_dim = self.n_heads * self.head_k_dim
        self.allow_neg_eigval = config.linear_allow_neg_eigval

        self.q_proj = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.f_proj_1 = nn.Linear(config.hidden_size, self.head_v_dim, bias=False)
        self.f_proj_2 = nn.Linear(self.head_v_dim, self.gate_dim, bias=False)
        self.beta_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.A_log = nn.Parameter(
            torch.empty(self.n_heads, dtype=torch.float32).uniform_(1, 16).log_()
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.gate_dim, dtype=torch.float32))
        self.q_conv1d = Olmo3MoeCausalConv1d(self.key_dim, config.linear_conv_kernel_dim)
        self.k_conv1d = Olmo3MoeCausalConv1d(self.key_dim, config.linear_conv_kernel_dim)
        self.v_conv1d = Olmo3MoeCausalConv1d(self.value_dim, config.linear_conv_kernel_dim)
        self.g_proj_1 = nn.Linear(config.hidden_size, self.head_v_dim, bias=False)
        self.g_proj_2 = nn.Linear(self.head_v_dim, self.value_dim, bias=True)
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=config.linear_norm_eps,
            activation="sigmoid",
        )
        self.o_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del kwargs
        if past_key_values is not None:
            raise NotImplementedError(
                "The exported KDA model does not yet implement recurrent-state caching; "
                "run with use_cache=False."
            )
        try:
            from fla.ops.kda import chunk_kda
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError(
                "KDA inference requires flash-linear-attention with fla.ops.kda.chunk_kda"
            ) from exc

        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_conv1d(self.q_proj(hidden_states))
        k = self.k_conv1d(self.k_proj(hidden_states))
        v = self.v_conv1d(self.v_proj(hidden_states))
        raw_decay = self.f_proj_2(self.f_proj_1(hidden_states))
        beta = self.beta_proj(hidden_states).float().sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        q = q.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.n_v_heads, self.head_v_dim)
        raw_decay = raw_decay.view(batch_size, seq_len, self.n_v_heads, self.head_k_dim)
        output, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=raw_decay,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
        )
        output_gate = self.g_proj_2(self.g_proj_1(hidden_states)).view(
            batch_size, seq_len, self.n_v_heads, self.head_v_dim
        )
        output = self.o_norm(output, output_gate).view(batch_size, seq_len, -1)
        return self.o_proj(output), None


class Olmo3MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Olmo3MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.layer_types[layer_idx] == "linear_attention":
            self.self_attn = Olmo3MoeKimiDeltaAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = Olmo3MoeAttention(config=config, layer_idx=layer_idx)

        if layer_idx in config.dense_layers_indices:
            self.mlp = Olmo3MoeDenseMLP(config)
        else:
            self.mlp = Olmo3MoeSparseMLP(config)

        self.post_attention_layernorm = Olmo3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.pre_attention_layernorm: Optional[Olmo3MoeRMSNorm]
        self.pre_feedforward_layernorm: Optional[Olmo3MoeRMSNorm]
        if config.use_peri_ln:
            self.pre_attention_layernorm = Olmo3MoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm = Olmo3MoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.pre_attention_layernorm = None
            self.pre_feedforward_layernorm = None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        if self.pre_attention_layernorm is not None:
            hidden_states = self.pre_attention_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Olmo3MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Olmo3MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.attention_hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_head_qk_norm = config.use_head_qk_norm
        self.gate_type = config.attention_gate_type
        self.gate_full_precision = config.attention_gate_full_precision

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.g_proj: Optional[nn.Linear]
        if self.gate_type == "elementwise":
            self.g_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=config.attention_bias,
            )
        elif self.gate_type == "headwise":
            self.g_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads,
                bias=config.attention_bias,
            )
        elif self.gate_type is None:
            self.g_proj = None
        else:
            raise ValueError(f"Unsupported attention_gate_type={self.gate_type!r}")
        if config.use_head_qk_norm:
            q_shape = (
                (config.num_attention_heads, self.head_dim)
                if config.qk_norm_per_head_gains
                else self.head_dim
            )
            k_shape = (
                (config.num_key_value_heads, self.head_dim)
                if config.qk_norm_per_head_gains
                else self.head_dim
            )
            self.q_norm = Olmo3MoeRMSNorm(q_shape, config.rms_norm_eps)
            self.k_norm = Olmo3MoeRMSNorm(k_shape, config.rms_norm_eps)
        else:
            self.q_norm = Olmo3MoeRMSNorm(
                config.num_attention_heads * self.head_dim, config.rms_norm_eps
            )
            self.k_norm = Olmo3MoeRMSNorm(
                config.num_key_value_heads * self.head_dim, config.rms_norm_eps
            )
        self.ssmax_scale = (
            nn.Parameter(torch.ones(config.num_attention_heads))
            if config.scalable_softmax
            else None
        )
        assert config.layer_types is not None
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = (
            config.sliding_window if self.attention_type == "sliding_attention" else None
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # QK norm behavior:
        # - use_head_qk_norm=False: normalize over the flattened projection dim (matches File 1 "full-dim" norm path)
        # - use_head_qk_norm=True: reshape into heads first, then normalize per head over head_dim (matches File 1 head-wise path)
        if not self.use_head_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Apply [H,D] gains in [B,T,H,D] layout, so H never broadcasts against T.
        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape).transpose(
            1, 2
        )  # (B, n_kv_heads, T, head_dim)

        if self.use_head_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        cos: Optional[torch.Tensor] = None
        sin: Optional[torch.Tensor] = None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Match core's rounding order: Q/K norm, RoPE, then scalable-softmax query scaling.
        if self.ssmax_scale is not None:
            positions = position_ids if position_ids is not None else cache_position
            if positions is None:
                positions = torch.arange(input_shape[1], device=hidden_states.device)
            lengths = positions.to(hidden_states.device).reshape(-1, input_shape[1]) + 1
            scale = lengths.log().to(query_states.dtype).unsqueeze(1)
            scale = scale * self.ssmax_scale.to(query_states.dtype).view(1, -1, 1)
            query_states = query_states * scale.unsqueeze(-1)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            if sin is not None and cos is not None:
                cache_kwargs.update({"sin": sin, "cos": cos})
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.g_proj is not None:
            gate = self.g_proj(hidden_states)
            if self.gate_full_precision:
                gate = gate.float()
            gate = torch.sigmoid(gate).to(attn_output.dtype)
            if self.gate_type == "headwise":
                attn_output = attn_output.view(
                    *input_shape, self.config.num_attention_heads, self.head_dim
                )
                attn_output = attn_output * gate.unsqueeze(-1)
                attn_output = attn_output.reshape(*input_shape, -1)
            else:
                attn_output = attn_output * gate
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@auto_docstring
class Olmo3MoePreTrainedModel(PreTrainedModel):
    config: Olmo3MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Olmo3MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Olmo3MoeDecoderLayer,
        "attentions": Olmo3MoeAttention,
    }


class Olmo3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def _validate_linear_attention_mask(
    attention_mask: Optional[torch.Tensor | dict[str, Optional[torch.Tensor]]],
) -> None:
    if attention_mask is None:
        return
    linear_attention_mask = (
        attention_mask.get("linear_attention")
        if isinstance(attention_mask, dict)
        else attention_mask
    )
    if linear_attention_mask is not None and (
        linear_attention_mask.ndim != 2 or not bool(torch.all(linear_attention_mask != 0))
    ):
        raise NotImplementedError(
            "KDA attention-mask support is not implemented; only unpadded inputs "
            "(or an all-ones 2D attention mask) are supported."
        )


@auto_docstring
class Olmo3MoeModel(Olmo3MoePreTrainedModel):
    def __init__(self, config: Olmo3MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.embed_scale = config.embed_scale
        self.embed_norm = (
            Olmo3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.embed_norm
            else None
        )

        self.layers = nn.ModuleList(
            [
                Olmo3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Olmo3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        if not config.use_rope:
            self.rotary_embs = None
        elif _uses_layer_type_rope_parameters(config):
            # LC exports can use YaRN for full attention and default RoPE for
            # sliding attention, so cache one rotary module per layer type.
            self.rotary_embs = nn.ModuleDict(
                {
                    layer_type: Olmo3MoeRotaryEmbedding(config=config, layer_type=layer_type)
                    for layer_type in sorted(set(config.layer_types))
                }
            )
        else:
            self.rotary_embs = Olmo3MoeRotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        cache_position (`torch.Tensor`, *optional*):
            Indices describing the positions of input tokens in the sequence. This is used to
            update a static cache in the correct position and to infer `position_ids` when those
            are not provided. Scalable softmax instead infers visible-token positions from a
            supplied 2D padding mask. Prepared masks require explicit `position_ids`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.config.scalable_softmax and (use_cache or past_key_values is not None):
            raise NotImplementedError(
                "Scalable-softmax KV caching is not supported in the exported HF model; "
                "run with use_cache=False and no past_key_values."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            assert inputs_embeds is not None
            inputs_embeds = inputs_embeds * self.embed_scale
            if self.embed_norm is not None:
                inputs_embeds = self.embed_norm(inputs_embeds)

        has_linear_attention = "linear_attention" in self.config.layer_types
        if has_linear_attention and any(
            kwargs.get(key) is not None for key in ("cu_doc_lens", "cu_seqlens")
        ):
            raise NotImplementedError(
                "Packed documents are not supported by the exported KDA model."
            )
        if has_linear_attention and position_ids is not None and position_ids.shape[-1] > 1:
            if not bool(torch.all(position_ids[..., 1:] == position_ids[..., :-1] + 1)):
                raise NotImplementedError(
                    "Packed/reset position IDs are not supported by the exported KDA model."
                )
        if use_cache and has_linear_attention:
            raise NotImplementedError(
                "KDA recurrent-state caching is not implemented in the exported HF model; "
                "run with use_cache=False."
            )
        if has_linear_attention:
            _validate_linear_attention_mask(attention_mask)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            assert inputs_embeds is not None
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            if self.config.scalable_softmax and attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
                    raise ValueError(
                        "Scalable softmax requires explicit position_ids with prepared attention "
                        "masks; supply a 2D padding mask to infer visible token positions."
                    )
                # Padding must not increase log(context length). Use the same visible-token
                # positions for query scaling and RoPE; cache slots still use cache_position.
                padding_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.long)
                position_ids = (padding_mask.cumsum(-1) - 1).masked_fill(padding_mask == 0, 0)
                position_ids = position_ids[:, -inputs_embeds.shape[1] :]
            else:
                position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments. ``_create_mask_compat`` filters kwargs to the mask helper's
            # signature, so it tolerates transformers versions that added/removed ``cache_position``
            # or renamed ``inputs_embeds`` -> ``input_embeds``.
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": _create_mask_compat(create_causal_mask, **mask_kwargs),
            }
            if "sliding_attention" in self.config.layer_types:
                causal_mask_mapping["sliding_attention"] = _create_mask_compat(
                    create_sliding_window_causal_mask, **mask_kwargs
                )

        hidden_states = inputs_embeds
        if self.rotary_embs is None:
            position_embeddings = None
        elif isinstance(self.rotary_embs, nn.ModuleDict):
            position_embeddings = {
                layer_type: rotary_emb(hidden_states, position_ids, layer_type=layer_type)
                for layer_type, rotary_emb in self.rotary_embs.items()
            }
        else:
            position_embeddings = self.rotary_embs(hidden_states, position_ids)

        for decoder_layer in self.layers:
            # if used in vllm with PP, a few layers will be replaced by PPMissingLayer(), which just passes the inputs through, so we need to skip the attention mask and position embeddings in that case
            if not isinstance(decoder_layer, Olmo3MoeDecoderLayer):
                hidden_states = decoder_layer(hidden_states)
                continue

            decoder_layer = cast(Olmo3MoeDecoderLayer, decoder_layer)
            attention_type = self.config.layer_types[decoder_layer.self_attn.layer_idx]
            attention_mask = causal_mask_mapping.get(attention_type)
            layer_position_embeddings = (
                position_embeddings[attention_type]
                if isinstance(position_embeddings, dict)
                else position_embeddings
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=layer_position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class Olmo3MoeForCausalLM(Olmo3MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Olmo3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        cache_position (`torch.LongTensor`, *optional*):
            Indices describing the positions of input tokens in the sequence. This is forwarded
            to the base model for cache placement and position inference.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Olmo3MoeConfig",
    "Olmo3MoeForCausalLM",
    "Olmo3MoeModel",
    "Olmo3MoePreTrainedModel",
]
