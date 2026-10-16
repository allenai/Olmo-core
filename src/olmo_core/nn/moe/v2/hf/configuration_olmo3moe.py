from typing import List, Optional

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation


class Olmo3MoeConfig(PretrainedConfig):
    model_type = "olmo3moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"truncate"}
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        attention_hidden_size=4096,
        head_dim=None,
        dense_mlp_intermediate_size=11008,
        moe_intermediate_size=2048,
        shared_expert_intermediate_size=2048,
        n_routed_experts=64,
        num_experts_per_tok=4,
        original_num_experts_per_tok=None,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        gating_function="softmax",
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        sliding_window=4096,
        use_head_qk_norm=False,
        qk_norm_per_head_gains=False,
        scalable_softmax=False,
        use_rope=True,
        attention_gate_type=None,
        attention_gate_full_precision=True,
        linear_num_key_heads=None,
        linear_num_value_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        linear_conv_kernel_dim=4,
        linear_allow_neg_eigval=False,
        linear_norm_eps=1e-5,
        latent_moe_dim=None,
        latent_moe_bias=False,
        latent_moe_up_proj_input_norm=False,
        emo_min_document_expert_pool=None,
        emo_max_document_expert_pool=None,
        emo_eval_document_expert_pool=None,
        emo_eos_token_id=None,
        global_load_balancing=False,
        moe_use_core_numerics=False,
        layer_types: Optional[List[str]] = None,
        dense_layers_indices: Optional[List[int]] = None,
        dense_layers_use_shared_expert=False,
        embed_scale=1.0,
        embed_norm=False,
        use_peri_ln=False,
        **kwargs,
    ):
        # Newer transformers pass RoPE settings as ``rope_parameters``; fall back to ``rope_scaling``.
        rope_parameters = kwargs.pop("rope_parameters", rope_scaling)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.attention_hidden_size = (
            attention_hidden_size if attention_hidden_size is not None else hidden_size
        )
        self.head_dim = (
            head_dim if head_dim is not None else self.attention_hidden_size // num_attention_heads
        )

        # for dense MLP layers
        self.dense_mlp_intermediate_size = dense_mlp_intermediate_size

        # for sparse MLP layers
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = (
            shared_expert_intermediate_size  # if None, no shared experts
        )
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.original_num_experts_per_tok = original_num_experts_per_tok
        assert gating_function in [
            "softmax",
            "sigmoid",
        ], "supported gating function: 'softmax' or 'sigmoid'"
        self.gating_function = gating_function
        self.normalize_expert_weights = normalize_expert_weights
        self.restore_weight_scale = restore_weight_scale
        # New core exports preserve no-EP SwiGLU and TE index-map reduction
        # rounding. Existing HF configs retain their original FP32 combination.
        self.moe_use_core_numerics = moe_use_core_numerics
        if moe_use_core_numerics and hidden_act != "silu":
            raise ValueError("moe_use_core_numerics requires hidden_act='silu'")

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.sliding_window = sliding_window
        self.layer_types: List[str]
        if layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 2 != 0 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        layer_type_validation(self.layer_types)

        # Newer transformers validates nested per-layer RoPE configs against ``self.layer_types``,
        # so layer metadata must be set before assigning/validating RoPE.
        self.rope_theta = rope_theta
        self.rope_scaling = rope_parameters
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.rms_norm_eps = rms_norm_eps
        self.use_head_qk_norm = use_head_qk_norm
        if qk_norm_per_head_gains and not use_head_qk_norm:
            raise ValueError("qk_norm_per_head_gains requires use_head_qk_norm")
        self.qk_norm_per_head_gains = qk_norm_per_head_gains
        self.scalable_softmax = scalable_softmax
        self.use_rope = use_rope
        self.attention_gate_type = attention_gate_type
        self.attention_gate_full_precision = attention_gate_full_precision

        # Kimi Delta Attention (KDA) fields. They are model-wide because the
        # Olmo-core ladder uses one KDA shape for every linear-attention layer.
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = (
            linear_num_value_heads if linear_num_value_heads is not None else linear_num_key_heads
        )
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_allow_neg_eigval = linear_allow_neg_eigval
        self.linear_norm_eps = linear_norm_eps

        # LatentMoE keeps the router and shared expert at ``hidden_size`` while
        # running only the routed expert payload through this narrower width.
        self.latent_moe_dim = latent_moe_dim
        self.latent_moe_bias = latent_moe_bias
        self.latent_moe_up_proj_input_norm = latent_moe_up_proj_input_norm

        # EMo document-pool metadata. The HF implementation is inference-only;
        # export currently requires the deterministic evaluation pool to span
        # every expert, which is equivalent to ordinary global top-k routing.
        self.emo_min_document_expert_pool = emo_min_document_expert_pool
        self.emo_max_document_expert_pool = emo_max_document_expert_pool
        self.emo_eval_document_expert_pool = emo_eval_document_expert_pool
        self.emo_eos_token_id = emo_eos_token_id
        self.global_load_balancing = global_load_balancing

        self.embed_scale = embed_scale
        self.embed_norm = embed_norm
        self.use_peri_ln = use_peri_ln

        self.dense_layers_indices = (
            dense_layers_indices if dense_layers_indices is not None else [0]
        )
        self.dense_layers_use_shared_expert = dense_layers_use_shared_expert

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        rope_config_validation(self)


__all__ = ["Olmo3MoeConfig"]
