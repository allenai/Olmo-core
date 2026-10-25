"""The optional wrapper hooks on :class:`~olmo_core.nn.ddp.model.OLMoDDPModel` are inert."""

import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)


def _shared_only_block_config() -> OLMoDDPTransformerBlockConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=DType.float32)
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=2,
            bias=False,
            backend=AttentionBackendName.torch,
            dtype=DType.float32,
        ),
        layer_norm=layer_norm,
        routed_experts=None,
        routed_experts_router=None,
        shared_experts=SharedExpertsConfig(
            d_model=16, hidden_size=32, num_experts=1, bias=False, dtype=DType.float32
        ),
        shared_experts_router=None,
    )


def _tiny_model():
    torch.manual_seed(0)
    config = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=16,
        vocab_size=64,
        n_layers=2,
        lm_head=LMHeadConfig(bias=False, dtype=DType.float32),
        block=_shared_only_block_config(),
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
    )
    model = config.build(init_device="cpu")
    model.init_weights(max_seq_len=8, device=torch.device("cpu"))
    return model


def test_input_embeddings_hook_matches_token_path():
    model = _tiny_model()
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        expected = model(input_ids)
        actual = model(input_ids, input_embeddings=model.forward_embed(input_ids))
        defaults = model(input_ids, input_embeddings=None, router_loss_div_factor=None)
    assert torch.equal(actual, expected)
    assert torch.equal(defaults, expected)


def test_router_loss_div_factor_only_changes_the_block_divisor(monkeypatch):
    model = _tiny_model()
    seen = {}
    original = model._forward_blocks

    def spy(h, all_block_kwargs, per_block_kwargs):
        seen["block"] = all_block_kwargs.get("loss_div_factor")
        return original(h, all_block_kwargs, per_block_kwargs)

    monkeypatch.setattr(model, "_forward_blocks", spy)
    input_ids = torch.randint(0, 64, (2, 8))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels, loss_reduction="sum", loss_div_factor=16.0)
    assert float(seen["block"]) == 16.0
    out_router = model(
        input_ids,
        labels=labels,
        loss_reduction="sum",
        loss_div_factor=16.0,
        router_loss_div_factor=4.0,
    )
    assert float(seen["block"]) == 4.0
    # The LM head divisor is unchanged, so the CE loss is identical.
    assert torch.equal(out.ce_loss, out_router.ce_loss)
